#!/usr/bin/env python3
"""Compare two Android ELF shared libraries with the configured NDK tools."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def run(tool: Path, *args: str) -> str:
    return subprocess.run(
        [str(tool), *args], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout


def sections(readelf: Path, path: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    # llvm-readelf -W -S uses: [Nr] Name Type Address Off Size ES Flg Lk Inf Al
    pattern = re.compile(
        r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+"
        r"[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+"
        r"[0-9a-fA-F]+\s+(\S+)\s+\S+\s+\S+\s+([0-9]+)\s*$"
    )
    for line in run(readelf, "--wide", "--sections", str(path)).splitlines():
        match = pattern.match(line)
        if match:
            name, kind, size_hex, flags, alignment = match.groups()
            result[name] = {
                "type": kind,
                "size": int(size_hex, 16),
                "flags": flags,
                "alignment": int(alignment),
            }
    return result


def needed(readelf: Path, path: Path) -> list[str]:
    values = []
    for line in run(readelf, "--wide", "--dynamic", str(path)).splitlines():
        match = re.search(r"\(NEEDED\).*\[(.+?)\]", line)
        if match:
            values.append(match.group(1))
    return sorted(values)


def symbols(nm: Path, path: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    output = run(nm, "--print-size", "--size-sort", "--demangle", str(path))
    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        size, kind, name = parts[1], parts[2], parts[3]
        if kind in {"T", "t", "W", "w", "I", "i"} and re.fullmatch(r"[0-9a-fA-F]+", size):
            result[name] = {"size": int(size, 16), "kind": kind}
    return result


def imports(readelf: Path, path: Path) -> list[str]:
    result = []
    for line in run(readelf, "--wide", "--dyn-syms", str(path)).splitlines():
        # Undefined dynamic symbols have UND in the Ndx column.
        fields = line.split()
        if len(fields) >= 8 and fields[6] == "UND":
            result.append(fields[7].split("@", 1)[0])
    return sorted(set(result))


def relocations(readelf: Path, path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in run(readelf, "--wide", "--relocations", str(path)).splitlines():
        fields = line.split()
        if len(fields) >= 3 and re.fullmatch(r"[0-9a-fA-F]+", fields[0]):
            counts[fields[2]] += 1
    return dict(sorted(counts.items()))


def delta_map(before: dict, after: dict) -> dict:
    names = sorted(set(before) | set(after))
    return {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in names if before.get(name) != after.get(name)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--tool-bin", required=True, type=Path,
                        help="NDK llvm tool directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    readelf = args.tool_bin / "llvm-readelf.exe"
    nm = args.tool_bin / "llvm-nm.exe"
    for path in (args.before, args.after, readelf, nm):
        if not path.is_file():
            parser.error(f"missing input/tool: {path}")

    before_sections, after_sections = sections(readelf, args.before), sections(readelf, args.after)
    before_symbols, after_symbols = symbols(nm, args.before), symbols(nm, args.after)
    before_relocations, after_relocations = relocations(readelf, args.before), relocations(readelf, args.after)
    report = {
        "before": args.before.name,
        "after": args.after.name,
        "elf_sections": {
            "before_total_bytes": sum(int(item["size"]) for item in before_sections.values()),
            "after_total_bytes": sum(int(item["size"]) for item in after_sections.values()),
            "before_text_bytes": int(before_sections.get(".text", {}).get("size", 0)),
            "after_text_bytes": int(after_sections.get(".text", {}).get("size", 0)),
            "changed": delta_map(before_sections, after_sections),
        },
        "dynamic_dependencies": {
            "before": needed(readelf, args.before),
            "after": needed(readelf, args.after),
        },
        "undefined_dynamic_symbols": {
            "before_count": len(imports(readelf, args.before)),
            "after_count": len(imports(readelf, args.after)),
            "added": sorted(set(imports(readelf, args.after)) - set(imports(readelf, args.before))),
            "removed": sorted(set(imports(readelf, args.before)) - set(imports(readelf, args.after))),
        },
        "relocation_counts_by_type": {
            "before": before_relocations,
            "after": after_relocations,
            "changed": delta_map(before_relocations, after_relocations),
        },
        "symbols": {
            "before_count": len(before_symbols),
            "after_count": len(after_symbols),
            "changed": delta_map(before_symbols, after_symbols),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"native_artifact_report={args.output}")
    print(f"section_delta_count={len(report['elf_sections']['changed'])}")
    print(f"symbol_delta_count={len(report['symbols']['changed'])}")
    print(f"dependencies_equal={report['dynamic_dependencies']['before'] == report['dynamic_dependencies']['after']}")
    print(f"imports_equal={not report['undefined_dynamic_symbols']['added'] and not report['undefined_dynamic_symbols']['removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
