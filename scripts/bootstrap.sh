#!/usr/bin/env bash
# sh-allow: unpinned-install installs from the package index into a venv; no hash-locked requirements file yet
set -euo pipefail

python -m pip install -U pip pre-commit

use_constraints=""
if [ -f constraints.txt ]; then
  use_constraints="-c constraints.txt"
fi

# shellcheck disable=SC2086 # $use_constraints is "-c file" and must split into two words
{ [ -f requirements.txt ] && pip install --upgrade -r requirements.txt $use_constraints; } || true
# shellcheck disable=SC2086 # same: $use_constraints must split
{ [ -f requirements-dev.txt ] && pip install --upgrade -r requirements-dev.txt $use_constraints; } || true

if [ -f pyproject.toml ]; then
  # shellcheck disable=SC2086 # same: $use_constraints must split
  pip install --upgrade --upgrade-strategy eager -e . $use_constraints
fi

pip check || true

git config --global --unset core.hooksPath || true
pre-commit install --install-hooks
pre-commit run --all-files || true

git config alias.cfix '!f(){ pre-commit run --all-files && git add -A && git commit -m "$*"; }; f'
git config alias.pc '!pre-commit run --all-files'
