from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


@dataclass
class GapInfo:
    lines: list[int]
    branches: dict[int, tuple[int, int]]  # lineno -> (covered, total)


def function_spans(path: Path) -> list[tuple[int, int, str]]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    mod = ast.parse(src, filename=str(path))
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is not None and end is not None:
                spans.append((int(start), int(end), node.name))
    spans.sort()
    return spans


def assign_func(spans: list[tuple[int, int, str]], lineno: int) -> str:
    for s, e, n in spans:
        if s <= lineno <= e:
            return n
    return "<module>"


def compress_ranges(nums: list[int]) -> str:
    out: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif prev is not None and n == prev + 1:
            prev = n
        else:
            out.append((int(start), int(prev if prev is not None else start)))
            start = prev = n
    if start is not None:
        out.append((int(start), int(prev if prev is not None else start)))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def parse_condition(cond: str) -> tuple[int, int] | None:
    """Parse coverage's "NN% (x/y)" into (covered, total); None when malformed."""
    try:
        covered_s, total_s = cond.split("(")[1].split(")")[0].split("/")
        return int(covered_s), int(total_s)
    except (IndexError, ValueError):
        return None


def resolve_source(filename: str | None) -> Path | None:
    """The .py file a coverage <class> names, or None when it is not Python or not on disk."""
    if not filename or not filename.endswith(".py"):
        return None
    for path in (Path(filename), Path.cwd() / filename):
        if path.exists():
            return path
    return None


def scan_line(line: etree._Element, mode: str) -> tuple[int, bool, tuple[int, int] | None] | None:
    """(line number, missed, partial branch counts) for one <line>, or None when unusable."""
    try:
        ln = int(line.attrib["number"])
        hits = int(line.attrib.get("hits", "0"))
    except (KeyError, ValueError):
        return None
    missed = mode in ("lines", "both") and hits == 0
    partial = None
    cond = line.attrib.get("condition-coverage")  # e.g., "50% (1/2)"
    if mode in ("branches", "both") and line.attrib.get("branch", "false") == "true" and cond:
        counts = parse_condition(cond)
        if counts is not None and counts[0] < counts[1]:
            partial = counts
    return ln, missed, partial


def scan_class(cls: etree._Element, mode: str) -> dict[str, GapInfo]:
    """Function name -> gaps for one <class> element."""
    path = resolve_source(cls.attrib.get("filename"))
    if path is None:
        return {}
    spans = function_spans(path)
    found: dict[str, GapInfo] = {}
    for line in cls.findall("./lines/line"):
        scanned = scan_line(line, mode)
        if scanned is None:
            continue
        ln, missed, partial = scanned
        info = found.setdefault(assign_func(spans, ln), GapInfo(lines=[], branches={}))
        if missed:
            info.lines.append(ln)
        if partial is not None:
            info.branches[ln] = partial
    return found


def collect_gaps(root: etree._Element, mode: str) -> dict[str, dict[str, GapInfo]]:
    """File -> function -> gaps for every class in the coverage XML."""
    gaps: dict[str, dict[str, GapInfo]] = {}
    for cls in root.findall(".//class"):
        path = resolve_source(cls.attrib.get("filename"))
        if path is not None:
            file_map = gaps.setdefault(str(path), {})
            for fn, info in scan_class(cls, mode).items():
                merged = file_map.setdefault(fn, GapInfo(lines=[], branches={}))
                merged.lines.extend(info.lines)
                merged.branches.update(info.branches)
    return gaps


def report_function(fn: str, info: GapInfo) -> list[str]:
    out: list[str] = []
    if info.lines:
        out.append(f"  - {fn}: missed lines -> {compress_ranges(info.lines)}")
    if info.branches:
        items = ", ".join(f"{ln} ({c}/{t})" for ln, (c, t) in sorted(info.branches.items()))
        out.append(f"  - {fn}: partial branches -> {items}")
    return out


def report_lines(gaps: dict[str, dict[str, GapInfo]]) -> list[str]:
    """The report text, one entry per output line."""
    out: list[str] = []
    for file, fdata in sorted(gaps.items()):
        body = [line for fn, info in sorted(fdata.items()) for line in report_function(fn, info)]
        if body:
            out += [f"\n{file}", *body]
    return out or ["No uncovered lines/branches detected (100% line/branch coverage)."]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report uncovered lines and branches grouped by function."
    )
    ap.add_argument("--xml", default="coverage.xml", help="Path to coverage XML")
    ap.add_argument("--mode", choices=["lines", "branches", "both"], default="both")
    args = ap.parse_args(argv)

    xml_path = Path(args.xml)
    if not xml_path.exists():
        sys.stderr.write("Run pytest with --cov-report=xml:coverage.xml first.\n")
        return 2

    # coverage.xml is a local file pytest-cov wrote; still never resolve entities or use the net.
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.parse(str(xml_path), parser).getroot()
    sys.stdout.write("\n".join(report_lines(collect_gaps(root, args.mode))) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
