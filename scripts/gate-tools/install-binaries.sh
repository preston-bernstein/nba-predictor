#!/usr/bin/env bash
# Installs the pinned trivy and vale release binaries that PyPI does not carry.
#
#   install-binaries.sh DEST_DIR
#
# Each archive is checked against a sha256 recorded here before it is unpacked.
# To upgrade, change the version and the sha256 together, and read the release notes.
set -euo pipefail

TRIVY_VERSION="0.76.2"
TRIVY_SHA256="bc0fc726e017b8927a62b2d3531d7a84ee01769daf5c99ed866dced54089fdbf"
VALE_VERSION="3.27.0"
VALE_SHA256="c80d62f6bce87b81843ac5b512a26e63213927b957adba0a64fc594570da2000"

# Global so the EXIT trap can still read it after main returns.
work=""

log() { echo "install-binaries: $*" >&2; }
die() {
  log "$*"
  exit "${2:-1}"
}

need() { command -v "$1" >/dev/null 2>&1 || die "missing tool: $1" 127; }

fetch_verified() {
  local url="$1" sha="$2" out="$3"
  curl --fail --silent --show-error --location --output "$out" "$url"
  echo "$sha  $out" | sha256sum --check --status || die "checksum mismatch for $url"
}

main() {
  local dest="${1:-}"
  [ -n "$dest" ] || die "usage: install-binaries.sh DEST_DIR" 2
  need curl
  need sha256sum
  need tar
  mkdir -p "$dest"
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT

  local base="https://github.com"
  fetch_verified "$base/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz" \
    "$TRIVY_SHA256" "$work/trivy.tar.gz"
  tar -xzf "$work/trivy.tar.gz" -C "$dest" trivy
  fetch_verified "$base/vale-cli/vale/releases/download/v${VALE_VERSION}/vale_${VALE_VERSION}_Linux_64-bit.tar.gz" \
    "$VALE_SHA256" "$work/vale.tar.gz"
  tar -xzf "$work/vale.tar.gz" -C "$dest" vale
  log "installed trivy $TRIVY_VERSION and vale $VALE_VERSION in $dest"
}

main "$@"
