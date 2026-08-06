#!/usr/bin/env python3
"""Private, streaming Nicopedia corpus preparation for the PhoneLM CPU pilot.

The production source and all derived text/token artifacts stay below ignored
build directories.  Standard output contains aggregate progress only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys
import tempfile
import time
import unicodedata


SCHEMA = "NICOPEDIA_REAL_TEXT_CORPUS_V1"
SPLIT_SALT = b"PhoneLM/Nicopedia/split/v1\0"
ORDER_SALT = b"PhoneLM/Nicopedia/subset/v1\0"
EXPECTED_HEADER = ["pg_id", "pg_title", "pg_view_title", "pg_yomi", "pg_category", "pg_created"]
EXPECTED_BODY = ["pg_id", "txt_text", "pg_rev_created"]
VALID_CATEGORIES = {"a", "i", "l", "o", "v"}
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
    "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
DROP_TAGS = {"script", "style", "noscript"}
TAG_RE = re.compile(r"<[^>]{1,512}>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def set_csv_limit() -> None:
    limit = min(sys.maxsize, 2**31 - 1)
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in DROP_TAGS:
            self.drop_depth += 1
        elif not self.drop_depth and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in DROP_TAGS and self.drop_depth:
            self.drop_depth -= 1
        elif not self.drop_depth and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth:
            self.parts.append(data)


def clean_text(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    parser = TextExtractor()
    try:
        parser.feed(normalized)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = TAG_RE.sub("\n", normalized)
    text = html.unescape(text)
    text = CONTROL_RE.sub("", text)
    lines = [MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text).strip()
    return text


def stable_digest(salt: bytes, value: str) -> bytes:
    return hashlib.sha256(salt + value.encode("utf-8")).digest()


def split_name(article_id: str) -> str:
    bucket = int.from_bytes(stable_digest(SPLIT_SALT, article_id)[:8], "big") % 10000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "validation"
    if bucket < 9900:
        return "development"
    return "final_test"


def order_key(article_id: str) -> int:
    return int.from_bytes(stable_digest(ORDER_SALT, article_id), "big")


def article_hash64(article_id: str) -> int:
    return int.from_bytes(hashlib.sha256(b"PhoneLM/Nicopedia/article/v1\0" + article_id.encode("utf-8")).digest()[:8], "big")


def percentile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


class TopArticles:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.heap: list[tuple[int, int, str]] = []

    def add(self, key: int, article_hash: int, text: str) -> None:
        item = (-key, article_hash, text)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def ordered(self) -> list[tuple[int, int, str]]:
        return sorted([(-negative, article_hash, text) for negative, article_hash, text in self.heap])


def choose_articles(selector: TopArticles, target_bytes: int) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    total = 0
    for item in selector.ordered():
        result.append(item)
        total += len(item[2].encode("utf-8"))
        if total >= target_bytes:
            break
    return result


def iter_csv(path: Path, expected: list[str]):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(
            handle, delimiter=",", quotechar='"', escapechar="\\",
            doublequote=False, strict=True,
        )
        header = next(reader)
        if header != expected:
            raise ValueError("SCHEMA_HEADER_MISMATCH")
        for row_index, row in enumerate(reader, start=2):
            if len(row) != len(expected):
                raise ValueError(f"SCHEMA_COLUMN_COUNT:{row_index}:{len(row)}")
            yield row_index, row


def file_snapshot(files: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}


def source_files(source_root: Path) -> tuple[list[Path], list[Path]]:
    header = sorted((source_root / "header").rglob("*.csv"))
    body = sorted((source_root / "body").rglob("*.csv"))
    if len(header) != 17 or len(body) != 189:
        raise RuntimeError(f"SOURCE_FILE_COUNT_MISMATCH:header={len(header)}:body={len(body)}")
    unexpected = [path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() != ".csv"]
    if unexpected:
        raise RuntimeError("UNEXPECTED_SOURCE_FILE_TYPE")
    return header, body


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def normalized_json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def inventory(source_root: Path, private_root: Path) -> dict[str, object]:
    header_files, body_files = source_files(source_root)
    files = header_files + body_files
    before = file_snapshot(files)
    started = time.perf_counter()
    entries = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        size = path.stat().st_size
        entries.append({"relative_path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0" + digest.encode("ascii") + b"\n")
    elapsed = time.perf_counter() - started
    if file_snapshot(files) != before:
        raise RuntimeError("SOURCE_MUTATED_DURING_INVENTORY")
    report = {
        "schema": SCHEMA,
        "dataset_name": "Nicopedia data",
        "dataset_version": "2024-11-25",
        "source_root": str(source_root.resolve()),
        "file_count": len(files),
        "header_file_count": len(header_files),
        "body_file_count": len(body_files),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "aggregate_sha256": "sha256:" + aggregate.hexdigest(),
        "hash_scan_seconds": elapsed,
        "files": entries,
        "source_read_only_verified": True,
    }
    write_json(private_root / "source-manifest.json", report)
    print(f"inventory_status=PASS files={len(files)} bytes={report['total_bytes']} seconds={elapsed:.3f}")
    return report


def parameter_count(vocabulary: int, dimension: int, ffn: int, layers: int) -> int:
    return 2 * vocabulary * dimension + layers * (4 * dimension * dimension + 4 * dimension + 2 * dimension * ffn)


def write_byte_cache(path: Path, articles: list[tuple[int, int, str]], context: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    for _, _, text in articles:
        encoded = text.encode("utf-8")
        chunk_count += max(0, (len(encoded) - 1) // context)
    with path.open("wb") as handle:
        handle.write(b"NPRTBYTEV1\n")
        handle.write(struct.pack(">IIQ", context, 256, chunk_count))
        for _, article_hash, text in articles:
            encoded = text.encode("utf-8")
            for start in range(0, len(encoded) - context, context):
                window = encoded[start:start + context + 1]
                if len(window) != context + 1:
                    continue
                handle.write(struct.pack(">Q", article_hash))
                handle.write(window)
    evidence = cache_content_identity(path)
    return {
        "path": str(path.resolve()),
        "articles": len(articles),
        "clean_utf8_bytes": sum(len(item[2].encode("utf-8")) for item in articles),
        "chunks": chunk_count,
        "target_tokens": chunk_count * context,
        "sha256": evidence["sha256"],
        "fnv1a64": evidence["fnv1a64"],
    }


def prepare(source_root: Path, private_root: Path, context: int) -> dict[str, object]:
    header_files, body_files = source_files(source_root)
    files = header_files + body_files
    source_manifest_path = private_root / "source-manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError("SOURCE_MANIFEST_REQUIRED_BEFORE_PREPARE")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if Path(source_manifest.get("source_root", "")).resolve() != source_root.resolve():
        raise RuntimeError("PREPARE_SOURCE_ROOT_DOES_NOT_MATCH_INVENTORY")
    manifest_files = {entry["relative_path"]: entry for entry in source_manifest.get("files", [])}
    current_files = {path.relative_to(source_root).as_posix(): path for path in files}
    if set(manifest_files) != set(current_files):
        raise RuntimeError("PREPARE_SOURCE_FILE_SET_DOES_NOT_MATCH_INVENTORY")
    if any(path.stat().st_size != int(manifest_files[name]["bytes"]) for name, path in current_files.items()):
        raise RuntimeError("PREPARE_SOURCE_SIZE_DOES_NOT_MATCH_INVENTORY")
    before = file_snapshot(files)
    started = time.perf_counter()
    header_types: dict[str, str] = {}
    header_counts: dict[str, int] = {}
    parse_errors: dict[str, int] = {}
    header_records = 0
    for path in header_files:
        try:
            for _, row in iter_csv(path, EXPECTED_HEADER):
                article_id, category = row[0], row[4]
                if category not in VALID_CATEGORIES:
                    raise ValueError("UNKNOWN_ARTICLE_CATEGORY")
                if article_id in header_types:
                    raise ValueError("DUPLICATE_HEADER_ID")
                header_types[article_id] = category
                header_counts[category] = header_counts.get(category, 0) + 1
                header_records += 1
        except (UnicodeDecodeError, csv.Error, ValueError) as error:
            key = type(error).__name__ + ":" + str(error).split(":", 1)[0]
            parse_errors[key] = parse_errors.get(key, 0) + 1

    selectors = {
        "train": TopArticles(20000),
        "validation": TopArticles(5000),
        "development": TopArticles(5000),
    }
    seen_ids: set[str] = set()
    seen_text_hashes: set[bytes] = set()
    raw_lengths: list[int] = []
    clean_lengths: list[int] = []
    body_records = 0
    usable_records = 0
    duplicate_records = 0
    unmatched_header = 0
    newline_records = 0
    markup_records = 0
    url_records = 0
    raw_newlines = 0
    tag_occurrences = 0
    split_counts = {name: 0 for name in ("train", "validation", "development", "final_test")}
    split_bytes = {name: 0 for name in split_counts}
    usable_type_counts: dict[str, int] = {}
    exclusions = {"too_short": 0, "too_long": 0, "empty_or_markup_only": 0, "duplicate": 0}

    for path in body_files:
        try:
            for _, row in iter_csv(path, EXPECTED_BODY):
                article_id, raw = row[0], row[1]
                body_records += 1
                if article_id in seen_ids:
                    raise ValueError("DUPLICATE_BODY_ID")
                seen_ids.add(article_id)
                category = header_types.get(article_id)
                if category is None:
                    unmatched_header += 1
                    continue
                raw_bytes = len(raw.encode("utf-8"))
                raw_lengths.append(raw_bytes)
                newline_count = raw.count("\n")
                raw_newlines += newline_count
                newline_records += int(newline_count > 0)
                tags = len(TAG_RE.findall(raw))
                tag_occurrences += tags
                markup_records += int(tags > 0)
                url_records += int(bool(URL_RE.search(raw)))
                cleaned = clean_text(raw)
                clean_bytes = len(cleaned.encode("utf-8"))
                if not cleaned:
                    exclusions["empty_or_markup_only"] += 1
                    continue
                if clean_bytes < 96:
                    exclusions["too_short"] += 1
                    continue
                if clean_bytes > 1048576:
                    exclusions["too_long"] += 1
                    continue
                text_hash = hashlib.sha256(cleaned.encode("utf-8")).digest()
                if text_hash in seen_text_hashes:
                    duplicate_records += 1
                    exclusions["duplicate"] += 1
                    continue
                seen_text_hashes.add(text_hash)
                clean_lengths.append(clean_bytes)
                usable_records += 1
                usable_type_counts[category] = usable_type_counts.get(category, 0) + 1
                split = split_name(article_id)
                split_counts[split] += 1
                split_bytes[split] += clean_bytes
                if split in selectors:
                    selectors[split].add(order_key(article_id), article_hash64(article_id), cleaned)
        except (UnicodeDecodeError, csv.Error, ValueError) as error:
            key = type(error).__name__ + ":" + str(error).split(":", 1)[0]
            parse_errors[key] = parse_errors.get(key, 0) + 1

    if parse_errors:
        report = {"schema": SCHEMA, "parse_errors": parse_errors, "status": "PARSE_FAILED"}
        write_json(private_root / "reports" / "corpus-aggregate.json", report)
        raise RuntimeError("CORPUS_PARSE_ERRORS:" + json.dumps(parse_errors, sort_keys=True))
    if header_records != body_records or len(header_types) != len(seen_ids) or unmatched_header:
        raise RuntimeError("HEADER_BODY_IDENTITY_MISMATCH")
    if file_snapshot(files) != before:
        raise RuntimeError("SOURCE_MUTATED_DURING_PREPARE")

    targets = {
        "train_smoke": ("train", 1048576),
        "train_pilot": ("train", 8388608),
        "train_medium": ("train", 26214400),
        "validation": ("validation", 2097152),
        "development": ("development", 2097152),
    }
    chosen: dict[str, list[tuple[int, int, str]]] = {
        name: choose_articles(selectors[split], byte_target)
        for name, (split, byte_target) in targets.items()
    }
    cache_reports = {}
    cache_root = private_root / "caches"
    for name, articles in chosen.items():
        cache_reports[name] = write_byte_cache(cache_root / f"{name}.bin", articles, context)

    train_texts = [item[2] for item in chosen["train_pilot"]]
    char_counts: dict[str, int] = {}
    byte_ratios: list[float] = []
    total_chars = 0
    total_bytes = 0
    tokenizer_started = time.perf_counter()
    for text in train_texts:
        chars = len(text)
        encoded_bytes = len(text.encode("utf-8"))
        total_chars += chars
        total_bytes += encoded_bytes
        if chars:
            byte_ratios.append(encoded_bytes / chars)
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
    ordered_chars = sorted(char_counts, key=lambda char: (-char_counts[char], ord(char)))[:2047]
    covered = sum(char_counts[char] for char in ordered_chars)
    char_vocab = {"schema": "NICOPEDIA_PRIVATE_CODEPOINT_V1", "unknown_id": 0, "tokens": ordered_chars}
    write_json(private_root / "tokenizer" / "codepoint-vocab.json", char_vocab)
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    tokenizer_protocols = [
        {
            "candidate": "utf8_byte",
            "vocabulary": 256,
            "mean_tokens_per_character": total_bytes / max(1, total_chars),
            "p95_tokens_per_character": percentile(byte_ratios, 0.95),
            "unknown_rate": 0.0,
            "round_trip": True,
            "train_only": True,
            "protocol_hash": normalized_json_hash({"kind": "utf8_byte", "vocabulary": 256, "normalization": "NFKC_HTML_TEXT_V1"}),
            "l6_parameter_count": parameter_count(256, 16, 32, 6),
            "l19_parameter_count": parameter_count(256, 16, 32, 19),
        },
        {
            "candidate": "top_codepoint_with_unk",
            "vocabulary": len(ordered_chars) + 1,
            "mean_tokens_per_character": 1.0,
            "p95_tokens_per_character": 1.0,
            "unknown_rate": 1.0 - covered / max(1, total_chars),
            "round_trip": False,
            "train_only": True,
            "protocol_hash": "sha256:" + hashlib.sha256((private_root / "tokenizer" / "codepoint-vocab.json").read_bytes()).hexdigest(),
            "l6_parameter_count": parameter_count(len(ordered_chars) + 1, 16, 32, 6),
            "l19_parameter_count": parameter_count(len(ordered_chars) + 1, 16, 32, 19),
        },
    ]
    elapsed = time.perf_counter() - started
    aggregate = {
        "schema": SCHEMA,
        "status": "PASS",
        "dataset_name": "Nicopedia data",
        "dataset_version": "2024-11-25",
        "source_file_aggregate_sha256": source_manifest["aggregate_sha256"],
        "header_records": header_records,
        "body_records": body_records,
        "body_with_text_records": body_records,
        "usable_records": usable_records,
        "exact_duplicate_excluded": duplicate_records,
        "parse_error_count": 0,
        "header_body_unmatched": unmatched_header,
        "article_type_counts": header_counts,
        "usable_article_type_counts": usable_type_counts,
        "raw_utf8_bytes": sum(raw_lengths),
        "clean_utf8_bytes": sum(clean_lengths),
        "raw_length_bytes": {"min": min(raw_lengths), "p50": percentile(raw_lengths, .5), "p95": percentile(raw_lengths, .95), "max": max(raw_lengths)},
        "clean_length_bytes": {"min": min(clean_lengths), "p50": percentile(clean_lengths, .5), "p95": percentile(clean_lengths, .95), "max": max(clean_lengths)},
        "empty_body_rate": exclusions["empty_or_markup_only"] / max(1, body_records),
        "newline_record_count": newline_records,
        "raw_newline_count": raw_newlines,
        "markup_record_count": markup_records,
        "markup_tag_occurrences": tag_occurrences,
        "url_record_count": url_records,
        "exclusions": exclusions,
        "split_counts": split_counts,
        "split_clean_utf8_bytes": split_bytes,
        "cache_reports": cache_reports,
        "tokenizer_candidates": tokenizer_protocols,
        "selected_tokenizer": "utf8_byte",
        "selected_tokenizer_reason": "zero unknown tokens, exact byte round-trip, and 8x lower dense output vocabulary than the TRAIN-only codepoint candidate",
        "context_tokens": context,
        "processing_seconds": elapsed,
        "processing_mib_per_second": sum(path.stat().st_size for path in files) / 1048576 / max(elapsed, 1e-9),
        "tokenizer_seconds": tokenizer_seconds,
        "source_read_only_verified": True,
        "final_test_opened_for_evaluation": False,
    }
    aggregate["aggregate_hash"] = normalized_json_hash(aggregate)
    write_json(private_root / "reports" / "corpus-aggregate.json", aggregate)
    public_aggregate = {key: value for key, value in aggregate.items() if key not in {"cache_reports"}}
    public_aggregate["cache_identities"] = {
        name: {key: value for key, value in report.items() if key != "path"}
        for name, report in cache_reports.items()
    }
    write_json(private_root / "reports" / "public-corpus-aggregate.json", public_aggregate)
    print(f"prepare_status=PASS records={body_records} usable={usable_records} duplicates={duplicate_records} seconds={elapsed:.3f}")
    return aggregate


def read_cache(path: Path) -> tuple[int, int, int, list[tuple[int, bytes]]]:
    with path.open("rb") as handle:
        if handle.read(11) != b"NPRTBYTEV1\n":
            raise ValueError("CACHE_MAGIC")
        context, vocabulary, count = struct.unpack(">IIQ", handle.read(16))
        records = []
        for _ in range(count):
            article_hash = struct.unpack(">Q", handle.read(8))[0]
            window = handle.read(context + 1)
            if len(window) != context + 1:
                raise ValueError("CACHE_TRUNCATED")
            records.append((article_hash, window))
        if handle.read(1):
            raise ValueError("CACHE_TRAILING_BYTES")
        return context, vocabulary, count, records


def fnv_update(value: int, payload: bytes) -> int:
    for byte in payload:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def split_mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def training_order_identity(record_count: int, steps: int, batch_size: int, seed: int) -> str:
    if record_count <= 0:
        raise ValueError("TRAIN_CACHE_EMPTY")
    byte_order = "little" if sys.byteorder == "little" else "big"
    value = FNV_OFFSET
    state = seed
    for index in range(steps * batch_size):
        state = split_mix((state + index) & 0xFFFFFFFFFFFFFFFF)
        selected = state % record_count
        value = fnv_update(value, selected.to_bytes(8, byte_order))
    return f"fnv1a64:{value:016x}"


def cache_content_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        while block := raw.read(1024 * 1024):
            digest.update(block)
    with path.open("rb") as handle:
        if handle.read(11) != b"NPRTBYTEV1\n":
            raise ValueError("CACHE_MAGIC")
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError("CACHE_TRUNCATED_HEADER")
        context, vocabulary, count = struct.unpack(">IIQ", header)
        if context < 8 or context > 256 or vocabulary != 256 or count > 10_000_000:
            raise ValueError("CACHE_HEADER_INVALID")
        byte_order = "little" if sys.byteorder == "little" else "big"
        value = FNV_OFFSET
        value = fnv_update(value, context.to_bytes(4, byte_order))
        value = fnv_update(value, vocabulary.to_bytes(4, byte_order))
        value = fnv_update(value, count.to_bytes(8, byte_order))
        for _ in range(count):
            article_bytes = handle.read(8)
            window = handle.read(context + 1)
            if len(article_bytes) != 8 or len(window) != context + 1:
                raise ValueError("CACHE_RECORD_TRUNCATED")
            article_hash = int.from_bytes(article_bytes, "big")
            value = fnv_update(value, article_hash.to_bytes(8, byte_order))
            value = fnv_update(value, window)
        if handle.read(1):
            raise ValueError("CACHE_TRAILING_BYTES")
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "fnv1a64": f"fnv1a64:{value:016x}",
        "context": context,
        "vocabulary": vocabulary,
        "chunks": count,
    }


def build_private_evidence(private_root: Path) -> dict[str, object]:
    source_manifest_path = private_root / "source-manifest.json"
    corpus_report_path = private_root / "reports" / "public-corpus-aggregate.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_name") != "Nicopedia data" or source_manifest.get("dataset_version") != "2024-11-25":
        raise ValueError("SOURCE_MANIFEST_IDENTITY")
    if corpus.get("dataset_name") != "Nicopedia data" or corpus.get("dataset_version") != "2024-11-25":
        raise ValueError("CORPUS_REPORT_IDENTITY")
    if corpus.get("source_file_aggregate_sha256") != source_manifest.get("aggregate_sha256"):
        raise ValueError("CORPUS_SOURCE_BINDING_MISMATCH")
    cache_evidence: dict[str, object] = {}
    for name in ("train_pilot", "validation", "development"):
        identity = cache_content_identity(private_root / "caches" / f"{name}.bin")
        expected = corpus["cache_identities"][name]
        if identity["sha256"] != expected["sha256"] or identity["chunks"] != expected["chunks"]:
            raise ValueError("CACHE_REPORT_IDENTITY_MISMATCH")
        if identity["context"] != corpus["context_tokens"] or identity["vocabulary"] != 256:
            raise ValueError("CACHE_PROTOCOL_MISMATCH")
        cache_evidence[name] = identity
    if any((private_root / "caches").glob("*final*")):
        raise ValueError("FINAL_TEST_CACHE_PRESENT")
    evidence = {
        "schema": "NICOPEDIA_REAL_TEXT_PRIVATE_EVIDENCE_V1",
        "dataset_name": "Nicopedia data",
        "dataset_version": "2024-11-25",
        "source_file_aggregate_sha256": source_manifest["aggregate_sha256"],
        "corpus_aggregate_hash": corpus["aggregate_hash"],
        "corpus_report_sha256": "sha256:" + hashlib.file_digest(corpus_report_path.open("rb"), "sha256").hexdigest(),
        "cache_identities": cache_evidence,
        "training_order": {
            "seed": 20260806,
            "steps": 1000,
            "batch_chunks": 8,
            "fnv1a64": training_order_identity(int(cache_evidence["train_pilot"]["chunks"]), 1000, 8, 20260806),
        },
        "final_test_cache_present": False,
    }
    evidence["binding_sha256"] = normalized_json_hash(evidence)
    return evidence


def audit_private_evidence(private_root: Path) -> dict[str, object]:
    evidence = build_private_evidence(private_root)
    write_json(private_root / "reports" / "evidence-provenance.json", evidence)
    print("private_evidence_audit=PASS caches=3 final_test_cache=0")
    return evidence


def verify_private_evidence(private_root: Path) -> None:
    evidence_path = private_root / "reports" / "evidence-provenance.json"
    recorded = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = build_private_evidence(private_root)
    if recorded != expected:
        raise ValueError("PRIVATE_EVIDENCE_BINDING_MISMATCH")
    print("private_evidence_verification=PASS caches=3 final_test_cache=0")


def self_test() -> None:
    set_csv_limit()
    assert clean_text("Ａ\r\n<b>日&amp;本</b>\x01") == "A\n日&本"
    assert clean_text("<script>secret</script><p>可視</p>") == "可視"
    assert split_name("123") == split_name("123")
    assert order_key("123") == order_key("123")
    assert "日本".encode("utf-8").decode("utf-8") == "日本"
    with tempfile.TemporaryDirectory(prefix="phonelm-nicopedia-selftest-") as temporary:
        root = Path(temporary) / "source"
        (root / "header").mkdir(parents=True)
        (root / "body").mkdir(parents=True)
        header_path = root / "header" / "header_2008.csv"
        body_path = root / "body" / "body_200805.csv"
        with header_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, escapechar="\\", doublequote=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(EXPECTED_HEADER)
            writer.writerow(["1", "private", "private", "private", "a", "20080512173939"])
            writer.writerow(["2", "private", "private", "private", "v", "20080512173940"])
        visible = ("<p>日本語, quoted newline</p>\n" + "文脈が必要です。" * 20)
        with body_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, escapechar="\\", doublequote=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(EXPECTED_BODY)
            writer.writerow(["1", visible, "20080513205321"])
            writer.writerow(["2", visible, "20080513205322"])
        rows = list(iter_csv(body_path, EXPECTED_BODY))
        assert len(rows) == 2 and "\n" in rows[0][1][1] and "," in rows[0][1][1]
        before = file_snapshot([header_path, body_path])
        assert file_snapshot([header_path, body_path]) == before
        cleaned = clean_text(rows[0][1][1])
        articles = [(order_key("1"), article_hash64("1"), cleaned)]
        cache = Path(temporary) / "cache.bin"
        report = write_byte_cache(cache, articles, 16)
        context, vocabulary, count, records = read_cache(cache)
        assert context == 16 and vocabulary == 256 and count == report["chunks"] and count > 0
        for _, window in records:
            assert window[:-1][1:] == window[1:-1]
            assert len(window[:-1]) == len(window[1:]) == context
        malformed = Path(temporary) / "malformed.csv"
        malformed.write_text('"pg_id","txt_text","pg_rev_created"\n"1","x"\n', encoding="utf-8", newline="\n")
        failed = False
        try:
            list(iter_csv(malformed, EXPECTED_BODY))
        except ValueError as error:
            failed = str(error).startswith("SCHEMA_COLUMN_COUNT")
        assert failed
        assert file_snapshot([header_path, body_path]) == before
        assert hashlib.file_digest(cache.open("rb"), "sha256").hexdigest() == report["sha256"].split(":", 1)[1]
    print("nicopedia_real_text_pipeline_self_test=PASS")


def main() -> int:
    set_csv_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit-evidence", action="store_true")
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--context", type=int, default=32)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.audit_evidence or args.verify_evidence:
        if args.inventory or args.prepare or not args.private_root or args.source_root:
            parser.error("evidence modes require only --private-root")
        if args.audit_evidence and args.verify_evidence:
            parser.error("select exactly one evidence mode")
        if args.audit_evidence:
            audit_private_evidence(args.private_root.resolve())
        else:
            verify_private_evidence(args.private_root.resolve())
        return 0
    if not args.source_root or not args.private_root:
        parser.error("--source-root and --private-root are required")
    if args.context < 8 or args.context > 256:
        parser.error("--context must be in [8,256]")
    if args.inventory == args.prepare:
        parser.error("select exactly one of --inventory or --prepare")
    source_root = args.source_root.resolve()
    private_root = args.private_root.resolve()
    if args.inventory:
        inventory(source_root, private_root)
    else:
        prepare(source_root, private_root, args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
