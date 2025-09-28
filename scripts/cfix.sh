set -euo pipefail
pre-commit run --all-files
git add -A
git commit -m "${1:-apply pre-commit autofixes}"
