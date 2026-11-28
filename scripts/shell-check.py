#!/usr/bin/env python3
"""Audit shell scripts for the mistakes agents make most. Pack: shell.

    shell-check.py [ROOT]

Reads every *.sh and *.bash under ROOT as text (standard library only; dot-dirs and vendored dirs are skipped).
Exit 0 when clean, 1 with one line per problem. It complements shellcheck (quoting, pitfalls) and shfmt (format);
it checks what they do not: the shebang, strict mode, bash 3.2 portability (macOS /bin/bash is 3.2), mktemp
cleanup, `curl | sh`, eval and pip installs that skip --require-hashes.

A finding is waived for one file with a comment that names it and gives a reason:
    # sh-allow: nostrict <reason>    # sh-allow: bash4 <reason>    # sh-allow: eval <reason>    # sh-allow: posix <reason>
    # sh-allow: unpinned-install <reason>
A sourced library (no shebang, named *.bash or marked `# shellcheck shell=bash`) needs no shebang or strict mode.
"""
from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

SKIP_DIRS = {"node_modules", "vendor", "__pycache__", "dist", "build", "venv"}
BASH4 = [
    (r"\b(mapfile|readarray)\b", "mapfile/readarray"),
    (r"\b(declare|local|typeset)\s+-[a-zA-Z]*A", "associative arrays (declare -A)"),
    (r"\b(declare|local|typeset)\s+-[a-zA-Z]*n\b", "namerefs (declare -n)"),
    (r"\$\{[A-Za-z_]\w*(,,|,|\^\^|\^)[^}]*\}", "case-changing expansion (${x,,} ${x^^})"),
    (r"\$\{[A-Za-z_]\w*\[-\d+\]\}", "negative array index"),
    (r"\$\{\w+@[QEPAa]\}", "${x@Q}-style transforms"),
    (r"\|&", "|& pipe"),
    (r"&>>", "&>> redirect"),
    (r"\bcoproc\b", "coproc"),
    (r"\bwait\s+-n\b", "wait -n"),
    (r"\[\[\s+-v\s", "[[ -v var ]]"),
]


def load_skips(root: Path):
    """.shell-allow: one `<glob> <reason>` per line. Returns (globs, problems). The reason is required."""
    globs, problems = [], []
    f = root / ".shell-allow"
    if f.is_file():
        for n, line in enumerate(f.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                problems.append(f".shell-allow:{n}: needs `<glob> <reason>`, the reason is required")
            else:
                globs.append(parts[0])
    return globs, problems


def submodule_paths(root: Path) -> list[str]:
    try:
        text = (root / ".gitmodules").read_text()
    except OSError:
        return []
    return [m.group(1).strip().strip("/") for m in re.finditer(r"^\s*path\s*=\s*(.+)$", text, re.M)]


def walk(root: Path, skips=()):
    for p in sorted(root.rglob("*")):
        if any(fnmatch.fnmatchcase(p.relative_to(root).as_posix(), g) for g in skips):
            continue
        rel = p.relative_to(root).parts
        if p.suffix in {".sh", ".bash"} and p.is_file() and not any(
                part.startswith(".") or part in SKIP_DIRS for part in rel[:-1]):
            yield p


def scrub(line: str) -> str:
    """Drop comments and quoted text, but keep ${...} and $(...) inside double quotes (they are real code)."""
    line = re.sub(r"'[^']*'", "''", line)

    def dq(m: re.Match) -> str:
        return '"' + " ".join(re.findall(r"\$\{[^}]*\}|\$\([^)]*\)", m.group(0))) + '"'

    line = re.sub(r'"(?:[^"\\]|\\.)*"', dq, line)
    return re.sub(r"(^|\s)#.*$", "", line)


def _msg(rel: Path, n: int, msg: str) -> str:
    return f"{rel}:{n}: {msg}" if n else f"{rel}: {msg}"


def _strict_mode_ok(code: list[str]) -> bool:
    flags, pipefail = set(), False
    for c in code[:60]:
        m = re.match(r"\s*set\s+(.*)", c)
        if not m:
            continue
        toks = m.group(1).split()
        for i, t in enumerate(toks):
            if re.fullmatch(r"-[a-zA-Z]+", t):
                flags |= set(t[1:])
                if "o" in t[1:] and i + 1 < len(toks) and toks[i + 1] == "pipefail":
                    pipefail = True
    return {"e", "u"} <= flags and pipefail


def _unpinned_pip(shown: str, allow: set[str]) -> bool:
    if "unpinned-install" in allow or shown.lstrip().startswith("#"):
        return False
    if not re.search(r"\bpip3?[\"']?\s+install\b", shown):
        return False
    return not re.search(r"--require-hashes|install\s+(-\S+\s+)*pip(\s|$)", shown)


def _line_problems(c: str, shown: str, allow: set[str]) -> list[str]:
    """The messages for one scrubbed line `c` (`shown` is its raw text), in the order they are reported."""
    found: list[str] = []
    if re.search(r"(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?(ba|z)?sh\b", c):
        found.append("pipes a download into a shell; download, verify a checksum, then run")
    if _unpinned_pip(shown, allow):
        found.append("pip install without --require-hashes; install from a hash-locked file, or `# sh-allow: unpinned-install <reason>`")
    if "eval" not in allow and re.search(r"(^|[;&|(]\s*|\s)eval(\s|$)", c):
        found.append("eval runs a string as code; avoid it, or `# sh-allow: eval <reason>`")
    if "bash4" not in allow:
        for pattern, label in BASH4:
            if re.search(pattern, c):
                found.append(f"bash 3.2: {label} needs bash 4+ (macOS /bin/bash is 3.2); "
                             "use a while-read loop or a plain array, or `# sh-allow: bash4 <reason>`")
    return found


def _head_problems(rel: Path, lines: list[str], library: bool, allow: set[str]) -> list[str]:
    first = lines[0] if lines else ""
    if not library and "posix" not in allow and first.strip() != "#!/usr/bin/env bash":
        return [_msg(rel, 1, f"shebang is '{first.strip() or '(none)'}'; use #!/usr/bin/env bash (a sourced library: name it *.bash)")]
    return []


def _script_problems(rel: Path, code: list[str], library: bool, allow: set[str]) -> list[str]:
    out: list[str] = []
    if not library and "nostrict" not in allow and not _strict_mode_ok(code):
        out.append(_msg(rel, 0, "strict mode: set -euo pipefail near the top (or `# sh-allow: nostrict <reason>`)"))
    if re.search(r"\bmktemp\b", "\n".join(code)) and not re.search(r"(^|[;\s])trap\s", "\n".join(code)):
        out.append(_msg(rel, 0, "mktemp without a trap cleanup (trap 'rm -rf \"$tmp\"' EXIT)"))
    return out


def check(path: Path, root: Path) -> list[str]:
    rel = path.relative_to(root)
    raw = path.read_text(errors="replace")
    lines = raw.splitlines()
    allow = set(re.findall(r"sh-allow:\s*([\w-]+)", raw))

    first = lines[0] if lines else ""
    library = not first.startswith("#!") and (path.suffix == ".bash" or any("shellcheck shell=" in l for l in lines[:5]))
    out = _head_problems(rel, lines, library, allow)
    code = [scrub(l) for l in lines]
    out += _script_problems(rel, code, library, allow)

    for n, c in enumerate(code, 1):
        shown = lines[n - 1] if n <= len(lines) else c  # the raw line: scrub() blanks a quoted "$VENV/bin/pip"
        out += [_msg(rel, n, m) for m in _line_problems(c, shown, allow)]
    return out


def _skips_and_problems(root: Path):
    skips, problems = load_skips(root)
    skips = list(skips) + [g for sub in submodule_paths(root) for g in (sub, sub + "/*")]      # a git submodule is third-party code
    return skips, problems


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    skips, problems = _skips_and_problems(root)
    files = list(walk(root, skips))
    if not files and not problems:
        print("shell-check: no shell scripts, nothing to check")
        return 0
    problems += [p for f in files for p in check(f, root)]
    for p in problems:
        print(p)
    if not problems:
        print("shell-check: clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
