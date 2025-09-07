from __future__ import annotations
import ast, sys, argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

def function_spans(path: Path):
    src = path.read_text(encoding="utf-8", errors="ignore")
    mod = ast.parse(src, filename=str(path))
    spans: list[tuple[int,int,str]] = []
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is not None and end is not None:
                spans.append((start, end, node.name))
    spans.sort()
    return spans

def assign_func(spans, lineno):
    for s, e, n in spans:
        if s <= lineno <= e:
            return n
    return "<module>"

def compress_ranges(nums: list[int]) -> str:
    out = []
    start = prev = None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append((start, prev))
            start = prev = n
    if start is not None:
        out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)

def main():
    ap = argparse.ArgumentParser(description="Report uncovered lines and branches grouped by function.")
    ap.add_argument("--xml", default="coverage.xml", help="Path to coverage XML")
    ap.add_argument("--mode", choices=["lines", "branches", "both"], default="both")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        sys.exit("Run pytest with --cov-report=xml:coverage.xml first (e.g., make test-cov-gaps)")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # file -> fn -> { lines: [..], branches: {lineno: (covered,total)} }
    gaps = defaultdict(lambda: defaultdict(lambda: {"lines": [], "branches": {}}))

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

        for line in cls.findall("./lines/line"):
            ln = int(line.attrib["number"])
            hits = int(line.attrib.get("hits", "0"))
            branch_flag = line.attrib.get("branch", "false") == "true"
            cond = line.attrib.get("condition-coverage")  # e.g., "50% (1/2)"

            fn = assign_func(spans, ln)

            if args.mode in ("lines", "both") and hits == 0:
                gaps[str(path)][fn]["lines"].append(ln)

            if args.mode in ("branches", "both") and branch_flag and cond:
                # Parse "NN% (x/y)"
                try:
                    frac = cond.split("(")[1].split(")")[0]  # "1/2"
                    covered, total = map(int, frac.split("/"))
                    if covered < total:
                        gaps[str(path)][fn]["branches"][ln] = (covered, total)
                except Exception:
                    pass

    printed_any = False
    for file, fdata in sorted(gaps.items()):
        header_printed = False
        for fn, info in sorted(fdata.items()):
            missed = info["lines"]
            bmiss = info["branches"]
            if not missed and not bmiss:
                continue
            if not header_printed:
                print(f"\n{file}")
                header_printed = True
            printed_any = True
            if missed:
                print(f"  - {fn}: missed lines -> {compress_ranges(missed)}")
            if bmiss:
                items = ", ".join(f"{ln} ({c}/{t})" for ln,(c,t) in sorted(bmiss.items()))
                print(f"  - {fn}: partial branches -> {items}")

    if not printed_any:
        print("No uncovered lines/branches detected (line coverage perfect; branch coverage may be 100%).")

if __name__ == "__main__":
    main()
