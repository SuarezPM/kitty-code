#!/usr/bin/env bash
#
# Kitty Code installer — W1 (plan SDD-ESTADO-ARTE §3 W-INSTALAR).
#
#   curl -fsSL https://raw.githubusercontent.com/SuarezPM/kitty-code/main/install.sh | bash
#
# What it does:
#   1. Tries the signed release binary for this platform (curl -f, sha256
#      verified fail-closed).
#   2. Falls back to a source build (`cargo install --locked ...`).
#   3. Installs `kittycode` into $KITTY_INSTALL_DIR (default ~/.local/bin) plus
#      compat `kitty` / `kitty-cli` symlinks, then prints the 3-command quickstart.
#
# Source modes (--from-source):
#   a. Default: `cargo install --git https://github.com/${REPO}` — requiere
#      red + repo PÚBLICO (hoy el repo es privado ⇒ falla sin credenciales;
#      ver NOTA-REPO-PRIVADO abajo).
#   b. Local:  `--source-dir DIR` o `KITTY_INSTALL_SOURCE=DIR` — usa
#      `cargo install --path DIR` sin red contra GitHub. Es el path para
#      contenedor limpio con mount local:
#        docker run --rm -v "$PWD:/src" -w /src rust:stable \
#          bash install.sh --from-source --source-dir /src --dir /tmp/bin
#      (o KITTY_INSTALL_SOURCE=/src bash install.sh --from-source).
#
# NOTA-REPO-PRIVADO (Pass 3 Lane C, 2026-10-24):
#   Repo privado + sin tag pusheado ⇒ `curl|sh` público imposible hoy:
#   el paso 1 (resolve_tag + release binary) devuelve 404 y el paso 2a
#   (`cargo install --git`) pide credenciales. Secuencia de desbloqueo:
#   1) hacer público el repo, 2) pushear tag v*, 3) CI release genera
#   assets + sha256 + SBOM + attest/cosign, 4) `curl|sh` funciona.
#   Hasta entonces, único path soportado sin credenciales: modo (b) local.
#
# Phase scope: Linux x86_64 only (mac/win explicitly out until CI ships them).
# Every step checks its exit code and aborts with an actionable message.
set -euo pipefail

REPO="${KITTY_REPO:-SuarezPM/kitty-code}"
BIN="kittycode"
VERSION="${KITTY_VERSION:-latest}"
INSTALL_DIR="${KITTY_INSTALL_DIR:-$HOME/.local/bin}"
SOURCE_DIR="${KITTY_INSTALL_SOURCE:-}"
FROM_SOURCE=0

usage() {
    cat <<EOF
Usage: install.sh [--dir DIR] [--version TAG] [--from-source] [--source-dir DIR] [--help]

  --dir DIR        install directory (default: \$HOME/.local/bin)
  --version TAG    release tag (default: latest)
  --from-source    skip the release binary, build from source with cargo
  --source-dir DIR build from local checkout DIR (cargo install --path)
                   instead of --git (no GitHub network needed; implies
                   --from-source)
  --help           this text

Env: KITTY_REPO, KITTY_VERSION, KITTY_INSTALL_DIR, KITTY_INSTALL_SOURCE
     (== --source-dir) override the flags.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) INSTALL_DIR="${2:?--dir needs a value}"; shift 2 ;;
        --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
        --source-dir) SOURCE_DIR="${2:?--source-dir needs a value}"; FROM_SOURCE=1; shift 2 ;;
        --from-source) FROM_SOURCE=1; shift ;;
        --help | -h) usage; exit 0 ;;
        *) echo "error: unknown flag: $1 (see --help)" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        die "missing required tool: $1 ($2)"
    fi
}

# cargo install --path needs a PACKAGE manifest, but the repo root is a
# virtual workspace ([workspace] without [package]). When --source-dir points
# at the checkout root, auto-descend into the CLI package so both
# `--source-dir $PWD` and `--source-dir $PWD/crates/kitty-cli` work.
resolve_package_dir() {
    local d="$1"
    if [ -f "$d/crates/kitty-cli/Cargo.toml" ] \
        && ! grep -q '^\[package\]' "$d/Cargo.toml" 2>/dev/null; then
        printf '%s' "$d/crates/kitty-cli"
    else
        printf '%s' "$d"
    fi
}

# ── 0. Platform gate (phase scope: linux/x86_64) ──────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    die "unsupported platform: $OS/$ARCH — this phase ships linux/x86_64 only (mac/win pending CI). Build from source instead: cargo build --release -p kitty-cli"
fi
need curl "install curl and re-run"
need sha256sum "install coreutils and re-run"

resolve_tag() {
    # `latest` follows the GitHub redirect to the newest tag (curl -f: 404 fails here, not later).
    if [ "$VERSION" = "latest" ]; then
        curl -fsSL -o /dev/null -w "%{url_effective}" "https://github.com/${REPO}/releases/latest" \
            | sed 's#.*/tag/##'
    else
        printf '%s' "$VERSION"
    fi
}

install_release_binary() {
    local tag="$1"
    local base="https://github.com/${REPO}/releases/download/${tag}"
    local tmp
    tmp="$(mktemp -d)" || die "mktemp failed"
    trap 'rm -rf "$tmp"' RETURN
    echo "downloading $BIN $tag ..."
    if ! curl -fSL --retry 3 -o "$tmp/$BIN" "$base/$BIN"; then
        echo "warn: no release binary at $base/$BIN (release may not ship it yet)" >&2
        return 1
    fi
    echo "downloading checksum ..."
    if ! curl -fSL -o "$tmp/$BIN.sha256" "$base/$BIN.sha256"; then
        die "release binary downloaded but checksum file missing at $base/$BIN.sha256 — refusing to install unverified bytes (retry with --from-source)"
    fi
    ( cd "$tmp" && sha256sum -c "$BIN.sha256" ) || die "checksum mismatch for $BIN $tag"
    install -m 0755 "$tmp/$BIN" "$INSTALL_DIR/$BIN" || die "install to $INSTALL_DIR failed"
    return 0
}

install_from_source() {
    # Local-source mode: no GitHub network. Validates the checkout FIRST
    # (before the cargo check) so a bad --source-dir / KITTY_INSTALL_SOURCE
    # fails fast with a clear msg even on machines without cargo yet.
    if [ -n "$SOURCE_DIR" ]; then
        [ -d "$SOURCE_DIR" ] || die "source dir not found: $SOURCE_DIR (flag --source-dir or env KITTY_INSTALL_SOURCE)"
        [ -f "$SOURCE_DIR/Cargo.toml" ] || die "no Cargo.toml in $SOURCE_DIR — point --source-dir at the repo checkout root"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        die "cargo not found — install Rust via https://rustup.rs then re-run (or use a release binary)"
    fi
    if [ -n "$SOURCE_DIR" ]; then
        echo "building $BIN from LOCAL source $SOURCE_DIR (this takes a few minutes) ..."
        pkg_dir="$(resolve_package_dir "$SOURCE_DIR")"
        [ -f "$pkg_dir/Cargo.toml" ] || die "no Cargo.toml in $pkg_dir"
        grep -q '^\[package\]' "$pkg_dir/Cargo.toml" \
            || die "no [package] in $pkg_dir/Cargo.toml — point --source-dir at the repo root or at crates/kitty-cli"
        if [ "$pkg_dir" != "$SOURCE_DIR" ]; then
            echo "note: workspace root detected, using package dir $pkg_dir" >&2
        fi
        local tmp
        tmp="$(mktemp -d)" || die "mktemp failed"
        # Build into a temp root, then place the binary in $INSTALL_DIR so
        # --dir is honored verbatim (cargo always appends /bin to --root).
        if ! cargo install --locked --features tui --path "$pkg_dir" --bin "$BIN" --root "$tmp/root" --force; then
            rm -rf "$tmp"
            die "cargo install --path failed"
        fi
        if [ ! -x "$tmp/root/bin/$BIN" ]; then
            rm -rf "$tmp"
            die "cargo install succeeded but $tmp/root/bin/$BIN is missing"
        fi
        install -m 0755 "$tmp/root/bin/$BIN" "$INSTALL_DIR/$BIN" || { rm -rf "$tmp"; die "install to $INSTALL_DIR failed"; }
        rm -rf "$tmp"
        return 0
    fi
    echo "building $BIN from source (this takes a few minutes) ..."
    echo "note: git mode requires a PUBLIC repo; for a private checkout use --source-dir DIR or KITTY_INSTALL_SOURCE=DIR" >&2
    local tmp
    tmp="$(mktemp -d)" || die "mktemp failed"
    # Build into a temp root, then place the binary in $INSTALL_DIR so
    # --dir is honored verbatim (cargo always appends /bin to --root).
    if ! cargo install --locked --features tui --git "https://github.com/${REPO}" --bin "$BIN" --root "$tmp/root" --force; then
        rm -rf "$tmp"
        die "cargo install failed"
    fi
    if [ ! -x "$tmp/root/bin/$BIN" ]; then
        rm -rf "$tmp"
        die "cargo install succeeded but $tmp/root/bin/$BIN is missing"
    fi
    install -m 0755 "$tmp/root/bin/$BIN" "$INSTALL_DIR/$BIN" || { rm -rf "$tmp"; die "install to $INSTALL_DIR failed"; }
    rm -rf "$tmp"
}

# ── 1. Install ─────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR"

if [ "$FROM_SOURCE" -eq 1 ]; then
    install_from_source
else
    TAG="$(resolve_tag)" || die "could not resolve release tag (network or no releases yet)"
    echo "release tag: $TAG"
    if ! install_release_binary "$TAG"; then
        echo "falling back to source build ..."
        install_from_source
    fi
fi

# ── 2. Compat symlinks (bench/harbor scripts still call `kitty`/`kitty-cli`) ────
for alias in kitty kitty-cli; do
    if [ ! -e "$INSTALL_DIR/$alias" ]; then
        ln -s "$BIN" "$INSTALL_DIR/$alias" || die "symlink $alias failed"
    fi
done

# ── 3. Verify ──────────────────────────────────────────────────────────
if ! "$INSTALL_DIR/$BIN" --version >/dev/null 2>&1; then
    die "'$INSTALL_DIR/$BIN --version' failed — installation is broken"
fi
echo "installed: $("$INSTALL_DIR/$BIN" --version)"

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo "warn: $INSTALL_DIR is not on PATH — add: export PATH=\"\$PATH:$INSTALL_DIR\"" >&2 ;;
esac

# ── 4. Next steps (same 3 commands as README quickstart) ────────────────
cat <<EOF
quickstart:
  1. $BIN doctor
  2. ollama pull ornith:9b   # exact id proposed by 'doctor' when missing
  3. $BIN run "hola mundo"
EOF
