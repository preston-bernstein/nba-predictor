from __future__ import annotations

import argparse
import ast
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report uncovered lines and branches grouped by function."
    )
    ap.add_argument("--xml", default="coverage.xml", help="Path to coverage XML")
    ap.add_argument("--mode", choices=["lines", "branches", "both"], default="both")
    args = ap.parse_args(argv)

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print("Run pytest with --cov-report=xml:coverage.xml first.", file=sys.stderr)
        return 2

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # file -> fn -> GapInfo
    gaps: dict[str, dict[str, GapInfo]] = {}

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename")
        if not filename or not filename.endswith(".py"):
            continue
        path = Path(filename)
        if not path.exists():
            path = Path.cwd() / filename
            if not path.exists():
                continue

        spans = function_spans(path)
        path_key = str(path)

        for line in cls.findall("./lines/line"):
            num_s = line.attrib.get("number")
            hits_s = line.attrib.get("hits", "0")
            if num_s is None or hits_s is None:
                continue

            try:
                ln = int(num_s)
                hits = int(hits_s)
            except ValueError:
                continue

            branch_flag = line.attrib.get("branch", "false") == "true"
            cond = line.attrib.get("condition-coverage")  # e.g., "50% (1/2)"

            fn = assign_func(spans, ln)
            file_map = gaps.setdefault(path_key, {})
            info = file_map.setdefault(fn, GapInfo(lines=[], branches={}))

            if args.mode in ("lines", "both") and hits == 0:
                info.lines.append(ln)

            if args.mode in ("branches", "both") and branch_flag and cond:
                # Parse "NN% (x/y)"
                try:
                    frac = cond.split("(")[1].split(")")[0]  # "x/y"
                    covered_s, total_s = frac.split("/")
                    covered, total = int(covered_s), int(total_s)
                    if covered < total:
                        info.branches[ln] = (covered, total)
                except Exception:
                    # Ignore malformed entries
                    pass

    printed_any = False
    for file, fdata in sorted(gaps.items()):
        header_printed = False
        for fn, info in sorted(fdata.items()):
            missed = info.lines
            bmiss = info.branches
            if not missed and not bmiss:
                continue
            if not header_printed:
                print(f"\n{file}")
                header_printed = True
            printed_any = True
            if missed:
                print(f"  - {fn}: missed lines -> {compress_ranges(missed)}")
            if bmiss:
                items = ", ".join(f"{ln} ({c}/{t})" for ln, (c, t) in sorted(bmiss.items()))
                print(f"  - {fn}: partial branches -> {items}")

    if not printed_any:
        print("No uncovered lines/branches detected (100% line/branch coverage).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
