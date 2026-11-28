#!/usr/bin/env bash
# Validation gate for agent-written Python. Pack: python-uv, rule py-one-gate-command.
#
#   py-validate.sh [ROOT]
#
# Order: ruff check -> ruff format --check -> vulture (dead code) -> type check -> pytest.
# With a uv.lock, every tool runs as `uv run --locked <tool>`, so it uses the locked environment.
# Without one, the tools must already be on PATH and the project installed there (pip install -e .).
# Every tool is checked before any step runs. A missing tool fails the gate with exit 127. Nothing is skipped.
# Env:
#   PY_TYPECHECKER   mypy or ty (default: mypy when pyproject.toml has [tool.mypy], ty when it has [tool.ty...])
#   PY_TYPECHECK_PATHS   what the type checker reads (default: src when it exists, else the checker reads its own config)
set -euo pipefail
# A git hook exports GIT_DIR, GIT_INDEX_FILE and the like. A test that runs git in a temp folder would inherit them
# and write into this repo instead, so drop git's repo-local variables before any tool runs.
while IFS= read -r v; do unset "$v"; done < <(git rev-parse --local-env-vars 2>/dev/null || true)

ROOT="${1:-.}"
cd "$ROOT"

fail() {
  echo "py-validate: $*" >&2
  exit "${2:-1}"
}
need() { command -v "$1" >/dev/null 2>&1 || fail "missing tool: $1 (the gate never skips a step)" 127; }

if [ ! -f pyproject.toml ]; then
  echo "py-validate: no pyproject.toml, nothing to check"
  exit 0
fi

checker="${PY_TYPECHECKER:-}"
if [ -z "$checker" ]; then
  if grep -q '^\[tool\.mypy\]' pyproject.toml; then
    checker=mypy
  elif grep -q '^\[tool\.ty[].]' pyproject.toml; then
    checker=ty
  else
    fail "no type checker configured: add [tool.mypy] (strict) or [tool.ty] to pyproject.toml (see templates/pyproject.tooling.toml)" 2
  fi
fi
[ "$checker" = mypy ] || [ "$checker" = ty ] || fail "PY_TYPECHECKER must be mypy or ty, got '$checker'" 2
[ -d tests ] || fail "no tests/ directory: a feature ships with tests" 2
grep -q '^\[tool\.vulture\]' pyproject.toml \
  || fail "no dead-code check configured: add [tool.vulture] to pyproject.toml (see templates/pyproject.tooling.toml)" 2

RUN=()
if [ -f uv.lock ]; then
  need uv
  RUN=(uv run --locked)
else
  need ruff
  need vulture
  need "$checker"
  need pytest
fi

if [ -n "${PY_TYPECHECK_PATHS+x}" ]; then
  paths="$PY_TYPECHECK_PATHS"
elif [ -d src ]; then
  paths=src
else
  paths="" # no src/: the checker reads its own config ([tool.mypy] files, or ty's project discovery)
fi
run() { "${RUN[@]+"${RUN[@]}"}" "$@"; }

run ruff check .
run ruff format --check .
run vulture
if [ "$checker" = ty ]; then
  # shellcheck disable=SC2086  # PY_TYPECHECK_PATHS is a space-separated list of paths
  run ty check $paths
else
  # shellcheck disable=SC2086  # PY_TYPECHECK_PATHS is a space-separated list of paths
  run mypy $paths
fi
run pytest -q
echo "py-validate: all checks passed"
