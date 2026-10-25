#!/usr/bin/env bash
# Build a glibc-2.35-compatible kitty-cli binary for benchmark containers.
#
# Why: harbor task images (e.g. aider-polyglot) are based on Ubuntu 22.04
# (buildpack-deps:jammy, glibc 2.35). A kitty-cli built on the host (CachyOS,
# glibc >= 2.39) fails there with "GLIBC_2.39 not found". This script builds
# once inside an ubuntu:22.04 container and drops the artifact at
# bench/harbor/adapters/kitty/bin/kitty-cli, which the KittyAgent uploads into
# every trial container.
#
# Usage:
#   sudo -u "$USER" -g docker bash bench/harbor/adapters/kitty/build-binary.sh
#   (or run directly if your user is in the docker group)
#
# The cargo registry is cached in the docker volume `kitty-bench-cargo-registry`
# so repeated builds are fast.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/bin"
IMAGE="ubuntu:22.04"

mkdir -p "${OUT_DIR}"

docker run --rm \
  -v "${REPO_ROOT}:/src:ro" \
  -v "${OUT_DIR}:/out" \
  -v kitty-bench-cargo-registry:/usr/local/cargo/registry \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "${IMAGE}" bash -exc '
    apt-get update -qq
    apt-get install -y --no-install-recommends curl build-essential git ca-certificates >/dev/null
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    . "$HOME/.cargo/env"
    mkdir /build
    cp -r /src/Cargo.toml /src/Cargo.lock /src/rust-toolchain.toml /src/crates /src/engine /build/
    # kitty-cli embebe bench/demo-12/demo-12.json vía include_str! con ruta
    # relativa a la raíz del repo; copiarlo para que el build lo encuentre.
    mkdir -p /build/bench/demo-12
    cp /src/bench/demo-12/demo-12.json /build/bench/demo-12/demo-12.json
    cd /build
    cargo build --release -p kitty-cli
    cp target/release/kitty-cli /out/kitty-cli
    chown "$HOST_UID:$HOST_GID" /out/kitty-cli
  '

echo "Built ${OUT_DIR}/kitty-cli"
file "${OUT_DIR}/kitty-cli"
