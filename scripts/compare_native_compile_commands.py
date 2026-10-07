#!/usr/bin/env python3
"""Compare one translation unit's compile_commands.json entries."""
from __future__ import annotations

import argparse
import collections
import difflib
import json
from pathlib import Path


def read_command(path: Path, source_name: str) -> dict[str, str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in entries if Path(item.get("file", "")).name == source_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {source_name} command in {path.name}; found {len(matches)}")
    entry = matches[0]
    command = entry.get("command")
    if not command:
        command = " ".join(entry.get("arguments", []))
    if not command:
        raise ValueError(f"compile command is empty in {path}")
    return {"directory": entry.get("directory", ""), "file": entry.get("file", ""), "command": command}


def normalize(item: dict[str, str], database: Path) -> str:
    command = item["command"]
    source = Path(item["file"]).resolve()
    build = Path(item["directory"]).resolve()
    source_tokens = {str(source), source.as_posix(), str(source).replace("/", "\\"), source.as_posix().replace("/", "\\")}
    for token in sorted(source_tokens, key=len, reverse=True):
        command = command.replace(token, "<SOURCE>/" + source.name)
    build_tokens = {str(build), build.as_posix(), str(build).replace("/", "\\"), build.as_posix().replace("/", "\\")}
    for token in sorted(build_tokens, key=len, reverse=True):
        command = command.replace(token, "<BUILD>")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--source", default="tiny_language_model_cpu.cpp")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before = read_command(args.before, args.source)
    after = read_command(args.after, args.source)
    before_norm = normalize(before, args.before)
    after_norm = normalize(after, args.after)
    before_tokens = before_norm.split()
    after_tokens = after_norm.split()
    before_counts, after_counts = collections.Counter(before_tokens), collections.Counter(after_tokens)
    added = list((after_counts - before_counts).elements())
    removed = list((before_counts - after_counts).elements())
    expected = after_counts == (before_counts + collections.Counter(["-O2"]))
    output = {
        "schema_version": 1,
        "source": args.source,
        "before_compile_command": before_norm,
        "after_compile_command": after_norm,
        "added_tokens": added,
        "removed_tokens": removed,
        "only_difference_is_added_source_O2": expected and added == ["-O2"] and not removed,
        "unified_diff": "\n".join(difflib.unified_diff(before_norm.split(), after_norm.split(), fromfile="origin-main", tofile="candidate", lineterm="")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("compile_command_diff=ONLY_ADDED_-O2" if output["only_difference_is_added_source_O2"] else f"compile_command_diff=UNEXPECTED added={added} removed={removed}")
    return 0 if output["only_difference_is_added_source_O2"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
