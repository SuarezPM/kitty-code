#!/usr/bin/env bash
#
# Kitty Code installer — W1 (plan SDD-ESTADO-ARTE §3 W-INSTALAR).
#
#   curl -fsSL https://raw.githubusercontent.com/SuarezPM/kitty-code/main/install.sh | bash
#
# What it does:
#   1. Tries the signed release binary for this platform (curl -f, sha256
#      verified fail-closed).
#   2. Falls back to a source build (`cargo install --locked --git ...`).
#   3. Installs `kitty` into $KITTY_INSTALL_DIR (default ~/.local/bin) plus a
#      transitional `kitty-cli` symlink, then prints the 3-command quickstart.
#
# Phase scope: Linux x86_64 only (mac/win explicitly out until CI ships them).
# Every step checks its exit code and aborts with an actionable message.
set -euo pipefail

REPO="${KITTY_REPO:-SuarezPM/kitty-code}"
BIN="kitty"
VERSION="${KITTY_VERSION:-latest}"
INSTALL_DIR="${KITTY_INSTALL_DIR:-$HOME/.local/bin}"
FROM_SOURCE=0

usage() {
    cat <<EOF
Usage: install.sh [--dir DIR] [--version TAG] [--from-source] [--help]

  --dir DIR      install directory (default: \$HOME/.local/bin)
  --version TAG  release tag (default: latest)
  --from-source  skip the release binary, build from source with cargo
  --help         this text

Env: KITTY_REPO, KITTY_VERSION, KITTY_INSTALL_DIR override the flags.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) INSTALL_DIR="${2:?--dir needs a value}"; shift 2 ;;
        --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
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
    if ! command -v cargo >/dev/null 2>&1; then
        die "cargo not found — install Rust via https://rustup.rs then re-run (or use a release binary)"
    fi
    echo "building $BIN from source (this takes a few minutes) ..."
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

# ── 2. Compat symlink (bench/harbor scripts still call `kitty-cli`) ────
if [ ! -e "$INSTALL_DIR/kitty-cli" ]; then
    ln -s "$BIN" "$INSTALL_DIR/kitty-cli" || die "symlink failed"
fi

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
