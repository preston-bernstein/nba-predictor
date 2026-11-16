#!/usr/bin/env bats

bats_require_minimum_version 1.5.0

setup() {
  SCRIPT="$BATS_TEST_DIRNAME/../../scripts/gate-tools/install-binaries.sh"
}

@test "exits 2 and prints usage when no destination is given" {
  run -2 bash "$SCRIPT"
  [[ "$output" == *"usage: install-binaries.sh DEST_DIR"* ]]
}

@test "exits 127 when curl is missing" {
  run -127 env PATH=/nonexistent /bin/bash "$SCRIPT" "$BATS_TEST_TMPDIR/bin"
  [[ "$output" == *"missing tool: curl"* ]]
}
