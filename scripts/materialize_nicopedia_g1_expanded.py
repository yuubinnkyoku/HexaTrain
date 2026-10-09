#!/usr/bin/env python3
"""Build the private deterministic G1 long-run Nicopedia cache.

The source and every derived artifact remain below ignored build/private-data.
The canonical cleaner, split, exact-text dedupe, ordering salt, and byte-BPE
model are imported from the existing corpus pipeline; this tool never trains
or changes the tokenizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata

import nicopedia_real_text_pipeline as protocol
from nicopedia_byte_bpe import CACHE_MAGIC, VOCABULARY, load_model, read_bpe_cache


EXPECTED_TOKENIZER_SHA256 = "9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798"
ORDER_SEED = 20260806
HARD_CEILING_STEPS = 100_000
BATCH_SIZE = 8
MIN_EXPANDED_RECORDS = 800_000
MAX_DEVICE_RECORDS = 15_000_000
SUBSET_MODULUS = 10_000
SUBSET_THRESHOLD = 2_500
CONTEXT = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def read_exact(handle, count: int, label: str) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        raise RuntimeError(f"{label}_TRUNCATED")
    return payload


def cache_header(context: int, tokenizer_hash: str, count: int = 0) -> bytes:
    return (CACHE_MAGIC + struct.pack(">II", context, VOCABULARY) +
            bytes.fromhex(tokenizer_hash) + struct.pack(">Q", count))


def order_hash(record_count: int) -> str:
    # Same SplitMix64/modulo sequence and little-endian FNV bytes as the
    # Android implementation. The identity always covers the full hard cap.
    if sys.byteorder != "little":
        raise RuntimeError("TRAINING_ORDER_HOST_ENDIAN_UNSUPPORTED")
    state = ORDER_SEED
    value = protocol.FNV_OFFSET
    mask = 0xFFFFFFFFFFFFFFFF
    for index in range(HARD_CEILING_STEPS * BATCH_SIZE):
        state = protocol.split_mix((state + index) & mask)
        selected = state % record_count
        value = protocol.fnv_update(value, selected.to_bytes(8, "little"))
    return f"fnv1a64:{value:016x}"


def cache_self_test(path: Path, model, expected_count: int) -> dict[str, object]:
    header_size = len(CACHE_MAGIC) + 48
    record_size = 8 + 2 * (CONTEXT + 1)
    size = path.stat().st_size
    with path.open("rb") as handle:
        if read_exact(handle, len(CACHE_MAGIC), "CACHE_MAGIC") != CACHE_MAGIC:
            raise RuntimeError("CACHE_MAGIC_MISMATCH")
        context, vocabulary = struct.unpack(">II", read_exact(handle, 8, "CACHE_SHAPE"))
        tokenizer_hash = read_exact(handle, 32, "CACHE_TOKENIZER").hex()
        record_count = struct.unpack(">Q", read_exact(handle, 8, "CACHE_COUNT"))[0]
        if (context != CONTEXT or vocabulary != VOCABULARY or
                tokenizer_hash != model.sha256 or record_count != expected_count):
            raise RuntimeError("CACHE_HEADER_IDENTITY_MISMATCH")
        if size != header_size + record_count * record_size:
            raise RuntimeError("CACHE_FILE_SIZE_MISMATCH")
        if record_count <= 0:
            raise RuntimeError("CACHE_EMPTY")
        samples = {}
        for index in sorted({0, record_count - 1}):
            handle.seek(header_size + index * record_size)
            article_hash = struct.unpack(">Q", read_exact(handle, 8, "CACHE_ARTICLE"))[0]
            window = struct.unpack(">33H", read_exact(handle, 66, "CACHE_WINDOW"))
            if any(token >= vocabulary for token in window):
                raise RuntimeError("CACHE_TOKEN_RANGE")
            samples[index] = {"article_hash64": f"{article_hash:016x}",
                              "window_sha256": hashlib.sha256(struct.pack(">33H", *window)).hexdigest()}
        # Record index is an order-independent cache cursor; verify modulo wrap
        # at and beyond the end without loading the entire cache into memory.
        for logical in (0, record_count - 1, record_count, record_count + record_count - 1):
            physical = logical % record_count
            handle.seek(header_size + physical * record_size)
            wrapped = read_exact(handle, record_size, "CACHE_WRAP")
            expected = samples.get(physical)
            if expected is not None and hashlib.sha256(wrapped[8:]).hexdigest() != expected["window_sha256"]:
                raise RuntimeError("CACHE_INDEX_WRAP_MISMATCH")
        handle.seek(0, 2)
        if handle.tell() != size:
            raise RuntimeError("CACHE_TRAILING_BYTES")
    return {"header_schema": "NPRTBPEV1", "record_count": record_count,
            "record_size_bytes": record_size, "file_size_bytes": size,
            "tokenizer_identity": model.identity, "first_last_records": samples,
            "record_index_wrap": "PASS"}


def materialize(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    private_root = args.private_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError("SOURCE_ROOT_MISSING")
    protocol.set_csv_limit()
    existing = list(private_root.iterdir()) if private_root.exists() else []
    allowed_inputs = {"source-manifest.json", "tokenizer", "caches"}
    if any(path.name not in allowed_inputs for path in existing):
        raise RuntimeError("PRIVATE_OUTPUT_ROOT_NOT_EMPTY")
    if (private_root / "materialized").exists() or (private_root / "caches" / "train-expanded.bin").exists():
        raise RuntimeError("MATERIALIZATION_OUTPUT_ALREADY_EXISTS")
    private_root.mkdir(parents=True, exist_ok=True)
    data_root = private_root / "materialized"
    data_root.mkdir()

    source_manifest_path = private_root / "source-manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError("SOURCE_INVENTORY_REQUIRED")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if Path(source_manifest.get("source_root", "")).resolve() != source_root:
        raise RuntimeError("SOURCE_ROOT_IDENTITY_MISMATCH")
    header_files, body_files = protocol.source_files(source_root)
    files = header_files + body_files
    manifest_files = {item["relative_path"]: item for item in source_manifest["files"]}
    current_files = {path.relative_to(source_root).as_posix(): path for path in files}
    if set(manifest_files) != set(current_files):
        raise RuntimeError("SOURCE_FILE_SET_MISMATCH")
    if any(path.stat().st_size != int(manifest_files[name]["bytes"])
           for name, path in current_files.items()):
        raise RuntimeError("SOURCE_FILE_SIZE_DIFFERS_FROM_INVENTORY")
    for name, path in current_files.items():
        if sha256_file(path) != manifest_files[name]["sha256"]:
            raise RuntimeError("SOURCE_FILE_SHA256_DIFFERS_FROM_INVENTORY:" + name)
    source_snapshot = protocol.file_snapshot(files)

    model = load_model(args.tokenizer_model)
    tokenizer_sha = sha256_file(args.tokenizer_model)
    if model.sha256 != EXPECTED_TOKENIZER_SHA256 or tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("CANONICAL_TOKENIZER_IDENTITY_MISMATCH")
    if not args.encoder.is_file():
        raise RuntimeError("BPE_ENCODER_MISSING")
    heldout_caches = {}
    for split in ("validation", "development"):
        cache_path = private_root / "caches" / f"{split}.bin"
        if not cache_path.is_file():
            raise RuntimeError(f"HELDOUT_CACHE_MISSING:{split}")
        context, vocabulary, identity, records = read_bpe_cache(cache_path, model)
        if context != CONTEXT or vocabulary != VOCABULARY or identity != model.identity:
            raise RuntimeError(f"HELDOUT_CACHE_IDENTITY_MISMATCH:{split}")
        target_utf8_bytes = sum(
            model.token_byte_length(token)
            for _, window in records
            for token in window[1:]
        )
        heldout_caches[split] = {"records": len(records), "target_bpe_tokens": len(records) * CONTEXT,
                                 "target_original_utf8_bytes": target_utf8_bytes,
                                 "sha256": "sha256:" + sha256_file(cache_path),
                                 "tokenizer_identity": identity}

    header_types: dict[str, str] = {}
    all_split_ids = {name: set() for name in ("train", "validation", "development", "final_test")}
    for path in header_files:
        for _, row in protocol.iter_csv(path, protocol.EXPECTED_HEADER):
            article_id, category = row[0], row[4]
            if category not in protocol.VALID_CATEGORIES or article_id in header_types:
                raise RuntimeError("HEADER_ID_OR_CATEGORY_INVALID")
            header_types[article_id] = category
            all_split_ids[protocol.split_name(article_id)].add(
                hashlib.sha256(article_id.encode("utf-8")).hexdigest())

    split_counts = {name: 0 for name in all_split_ids}
    split_bytes = {name: 0 for name in all_split_ids}
    exclusions = {"empty_or_markup_only": 0, "too_short": 0, "too_long": 0, "duplicate": 0}
    seen_text_hashes: set[bytes] = set()
    seen_body_ids: set[str] = set()
    train_articles: list[tuple[int, str, int, int, int, bool]] = []
    train_eligible_id_hashes: set[str] = set()
    selected_id_hashes: set[str] = set()
    article_hash_owners: dict[int, str] = {}
    body_records = 0
    unmatched = 0
    spool_path = data_root / "eligible-train-text.spool"
    ledger_path = data_root / "article-split-id-hashes.tsv"
    started = time.perf_counter()
    with spool_path.open("wb") as spool, ledger_path.open("w", encoding="ascii", newline="\n") as ledger:
        ledger.write("split\tarticle_id_sha256\teligible\tselected_for_subset\n")
        print(f"phase=scan_train_protocol body_files={len(body_files)}", flush=True)
        for file_index, path in enumerate(body_files, start=1):
            for _, row in protocol.iter_csv(path, protocol.EXPECTED_BODY):
                article_id, raw = row[0], row[1]
                body_records += 1
                if article_id in seen_body_ids:
                    raise RuntimeError("DUPLICATE_BODY_ID")
                seen_body_ids.add(article_id)
                category = header_types.get(article_id)
                if category is None:
                    unmatched += 1
                    continue
                split = protocol.split_name(article_id)
                id_hash = hashlib.sha256(article_id.encode("utf-8")).hexdigest()
                cleaned = protocol.clean_text(raw)
                clean_bytes = cleaned.encode("utf-8")
                eligible = bool(clean_bytes) and 96 <= len(clean_bytes) <= 1_048_576
                if not clean_bytes:
                    exclusions["empty_or_markup_only"] += 1
                elif len(clean_bytes) < 96:
                    exclusions["too_short"] += 1
                elif len(clean_bytes) > 1_048_576:
                    exclusions["too_long"] += 1
                elif hashlib.sha256(clean_bytes).digest() in seen_text_hashes:
                    exclusions["duplicate"] += 1
                    eligible = False
                if eligible:
                    text_hash = hashlib.sha256(clean_bytes).digest()
                    if text_hash in seen_text_hashes:
                        raise RuntimeError("DEDUPLICATION_STATE_MISMATCH")
                    seen_text_hashes.add(text_hash)
                    split_counts[split] += 1
                    split_bytes[split] += len(clean_bytes)
                take = (eligible and split == "train" and
                        protocol.order_key(article_id) % SUBSET_MODULUS < SUBSET_THRESHOLD)
                ledger.write(f"{split}\t{id_hash}\t{int(eligible)}\t{int(take)}\n")
                if eligible and split == "train":
                    article_hash = protocol.article_hash64(article_id)
                    owner = article_hash_owners.setdefault(article_hash, id_hash)
                    if owner != id_hash:
                        raise RuntimeError("ARTICLE_HASH64_COLLISION")
                    if id_hash in train_eligible_id_hashes:
                        raise RuntimeError("TRAIN_ARTICLE_ID_DUPLICATE")
                    train_eligible_id_hashes.add(id_hash)
                    offset = spool.tell()
                    spool.write(struct.pack(">Q", len(clean_bytes)))
                    spool.write(clean_bytes)
                    train_articles.append((protocol.order_key(article_id), id_hash,
                                           article_hash, offset, len(clean_bytes), take))
                    if take:
                        selected_id_hashes.add(id_hash)
            print(f"scan_progress files={file_index}/{len(body_files)} "
                  f"body_records={body_records} eligible_train={len(train_articles)} "
                  f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)

    if protocol.file_snapshot(files) != source_snapshot:
        raise RuntimeError("SOURCE_MUTATED_DURING_MATERIALIZATION")
    if unmatched or body_records != len(header_types) or len(seen_body_ids) != len(header_types):
        raise RuntimeError("HEADER_BODY_IDENTITY_MISMATCH")
    if (split_counts["train"] != args.expected_train_articles or
            split_bytes["train"] != args.expected_train_clean_bytes):
        raise RuntimeError("CANONICAL_TRAIN_CLEANING_COUNT_MISMATCH")
    if not train_articles:
        raise RuntimeError("DETERMINISTIC_SUBSET_EMPTY")
    if selected_id_hashes & all_split_ids["validation"]:
        raise RuntimeError("SPLIT_LEAKAGE_VALIDATION")
    if selected_id_hashes & all_split_ids["development"]:
        raise RuntimeError("SPLIT_LEAKAGE_DEVELOPMENT")
    if selected_id_hashes & all_split_ids["final_test"]:
        raise RuntimeError("SPLIT_LEAKAGE_FINAL_TEST")
    if train_eligible_id_hashes & all_split_ids["validation"]:
        raise RuntimeError("FULL_TRAIN_SPLIT_LEAKAGE_VALIDATION")
    if train_eligible_id_hashes & all_split_ids["development"]:
        raise RuntimeError("FULL_TRAIN_SPLIT_LEAKAGE_DEVELOPMENT")
    if train_eligible_id_hashes & all_split_ids["final_test"]:
        raise RuntimeError("FULL_TRAIN_SPLIT_LEAKAGE_FINAL_TEST")
    train_articles.sort(key=lambda item: (item[0], item[1]))
    if len(train_articles) > 1_000_000:
        raise RuntimeError("ENCODER_ARTICLE_LIMIT")

    encode_input = data_root / "eligible-train.nprtbpe-input"
    encode_output = data_root / "eligible-train.nprtbpe-output"
    # Encode the exact eligible TRAIN split once. This obtains the full-cache
    # record count before choosing full or subset; the choice depends only on
    # the native 10M-record cap, never on model quality.
    with encode_input.open("wb") as encoded_in, spool_path.open("rb") as spool:
        encoded_in.write(b"NPRTBPEEN1\n")
        encoded_in.write(struct.pack(">I", len(train_articles)))
        for sequence, (_, id_hash, article_hash, offset, byte_count, _) in enumerate(train_articles):
            spool.seek(offset)
            stored_length = struct.unpack(">Q", read_exact(spool, 8, "SPOOL_LENGTH"))[0]
            if stored_length != byte_count:
                raise RuntimeError("SPOOL_LENGTH_MISMATCH")
            payload = read_exact(spool, byte_count, "SPOOL_TEXT")
            encoded_in.write(struct.pack(">QQQ", sequence, article_hash, byte_count))
            encoded_in.write(payload)
            if sequence and sequence % 50_000 == 0:
                print(f"encoder_input_progress articles={sequence}/{len(train_articles)} "
                      f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    print(f"phase=encode_full_train eligible_articles={len(train_articles)} "
          f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    completed = subprocess.run([str(args.encoder), str(encode_input), str(args.tokenizer_model),
                                str(encode_output)], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or "bpe_encode_status=PASS" not in completed.stdout:
        raise RuntimeError("BPE_ENCODER_FAILED:" + completed.stderr[-1000:])
    print(completed.stdout.strip(), flush=True)

    caches_root = private_root / "caches"
    caches_root.mkdir(exist_ok=True)
    full_candidate = data_root / "train-full-candidate.bin"
    subset_candidate = data_root / "train-subset-candidate.bin"
    full_records = full_tokens = full_target_bytes = full_clean_bytes = 0
    subset_records = subset_tokens = subset_target_bytes = subset_clean_bytes = 0
    subset_articles = 0
    full_writable = True
    with encode_output.open("rb") as encoded_out, \
            full_candidate.open("wb") as full_cache, \
            subset_candidate.open("wb") as subset_cache:
        if read_exact(encoded_out, 11, "ENCODER_MAGIC") != b"NPRTBPEEO1\n":
            raise RuntimeError("ENCODER_OUTPUT_MAGIC")
        count = struct.unpack(">I", read_exact(encoded_out, 4, "ENCODER_COUNT"))[0]
        if count != len(train_articles):
            raise RuntimeError("ENCODER_OUTPUT_COUNT")
        full_cache.write(cache_header(CONTEXT, model.sha256, 0))
        subset_cache.write(cache_header(CONTEXT, model.sha256, 0))
        for expected_sequence, (_, id_hash, expected_article_hash, _, byte_count, take) in enumerate(train_articles):
            sequence, article_hash, actual_bytes, token_count = struct.unpack(
                ">QQQQ", read_exact(encoded_out, 32, "ENCODER_ARTICLE_HEADER"))
            if (sequence != expected_sequence or article_hash != expected_article_hash or
                    actual_bytes != byte_count or token_count > byte_count):
                raise RuntimeError("ENCODER_ARTICLE_IDENTITY")
            token_payload = read_exact(encoded_out, token_count * 2, "ENCODER_TOKENS")
            tokens = struct.unpack(">" + "H" * token_count, token_payload)
            if any(token >= VOCABULARY for token in tokens):
                raise RuntimeError("ENCODER_TOKEN_RANGE")
            full_clean_bytes += byte_count
            full_tokens += token_count
            chunks = max(0, (token_count - 1) // CONTEXT)
            for start in range(0, token_count - CONTEXT, CONTEXT):
                window = tokens[start:start + CONTEXT + 1]
                if len(window) != CONTEXT + 1:
                    raise RuntimeError("CHUNK_WINDOW_LENGTH")
                record = struct.pack(">Q", article_hash) + struct.pack(">33H", *window)
                target_bytes = sum(model.token_byte_length(token) for token in window[1:])
                if full_writable:
                    if full_records < MAX_DEVICE_RECORDS:
                        full_cache.write(record)
                    else:
                        full_writable = False
                    full_records += 1
                    full_target_bytes += target_bytes
                else:
                    full_records += 1
                    full_target_bytes += target_bytes
                if take:
                    subset_cache.write(record)
                    subset_records += 1
                    subset_target_bytes += target_bytes
            if take:
                subset_articles += 1
                subset_clean_bytes += byte_count
                subset_tokens += token_count
            if expected_sequence and expected_sequence % 50_000 == 0:
                print(f"cache_progress articles={expected_sequence}/{len(train_articles)} "
                      f"full_records={full_records} subset_records={subset_records} "
                      f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
            if chunks != max(0, (token_count - 1) // CONTEXT):
                raise RuntimeError("CHUNK_COUNT_MISMATCH")
        if encoded_out.read(1):
            raise RuntimeError("ENCODER_OUTPUT_TRAILING")
        full_cache.seek(len(CACHE_MAGIC) + 8 + 32)
        full_cache.write(struct.pack(">Q", min(full_records, MAX_DEVICE_RECORDS)))
        full_cache.flush()
        subset_cache.seek(len(CACHE_MAGIC) + 8 + 32)
        subset_cache.write(struct.pack(">Q", subset_records))
        subset_cache.flush()

    use_full = full_records <= MAX_DEVICE_RECORDS
    chosen_records = full_records if use_full else subset_records
    chosen_articles = split_counts["train"] if use_full else subset_articles
    chosen_clean_bytes = full_clean_bytes if use_full else subset_clean_bytes
    chosen_tokens = full_tokens if use_full else subset_tokens
    chosen_target_bytes = full_target_bytes if use_full else subset_target_bytes
    if chosen_records < MIN_EXPANDED_RECORDS:
        raise RuntimeError("EXPANDED_TRAIN_BELOW_800K_RECORDS")
    if chosen_records > MAX_DEVICE_RECORDS:
        raise RuntimeError("CHOSEN_CACHE_EXCEEDS_DEVICE_RECORD_LIMIT")
    if (chosen_articles != (len(train_eligible_id_hashes) if use_full else len(selected_id_hashes)) or
            (train_eligible_id_hashes if use_full else selected_id_hashes) & all_split_ids["final_test"]):
        raise RuntimeError("SELECTED_TRAIN_IDENTITY_MISMATCH")
    chosen_candidate = full_candidate if use_full else subset_candidate
    train_cache = caches_root / "train-expanded.bin"
    chosen_candidate.replace(train_cache)
    full_candidate.unlink(missing_ok=True)
    subset_candidate.unlink(missing_ok=True)
    cache_identity = cache_self_test(train_cache, model, chosen_records)
    cache_hash = sha256_file(train_cache)
    ledger_hash = sha256_file(ledger_path)
    identity = {
        "schema": "NICOPEDIA_G1_EXPANDED_TRAIN_V1",
        "dataset": "Nicopedia data 2024-11-25",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_file_aggregate_sha256": source_manifest["aggregate_sha256"],
        "cleaning_protocol": "NFKC_HTML_TEXT_V1; length 96..1048576 UTF-8 bytes",
        "cleaning_runtime": {"python_version": platform.python_version(),
                             "unicode_database_version": unicodedata.unidata_version,
                             "pipeline_source_sha256": "sha256:" + sha256_file(Path(protocol.__file__))},
        "dedupe_protocol": "exact cleaned UTF-8 SHA-256, source traversal order, corpus-wide",
        "split_protocol": "PhoneLM/Nicopedia/split/v1, 90/5/4/1 article-id hash buckets",
        "eligible_train_articles": split_counts["train"],
        "eligible_train_clean_utf8_bytes": split_bytes["train"],
        "full_split_eligible_articles": split_counts,
        "full_split_eligible_clean_utf8_bytes": split_bytes,
        "full_train_measurement": {"articles": split_counts["train"],
                                   "clean_utf8_bytes": split_bytes["train"],
                                   "exact_t32_records": full_records,
                                   "target_bpe_tokens": full_records * CONTEXT,
                                   "encoded_article_bpe_tokens": full_tokens,
                                   "represented_original_utf8_bytes": full_target_bytes,
                                   "device_record_limit": MAX_DEVICE_RECORDS,
                                   "cache_selected": use_full},
        "selected_subset": {
            "algorithm": "keep eligible TRAIN articles where order_key(article_id) mod 10000 < 2500; then sort by (order_key, article_id_sha256)",
            "order_key": "sha256(PhoneLM/Nicopedia/subset/v1\\0 || UTF8(article_id))",
            "modulus": SUBSET_MODULUS,
            "threshold": SUBSET_THRESHOLD,
            "quality_independent": True,
            "articles": subset_articles,
            "clean_utf8_bytes": subset_clean_bytes,
            "exact_t32_records": subset_records,
            "target_bpe_tokens": subset_records * CONTEXT,
            "article_id_sha256_set_sha256": hashlib.sha256("".join(sorted(selected_id_hashes)).encode("ascii")).hexdigest(),
        },
        "chosen_training_data": {"method": "full_train" if use_full else "deterministic_expanded_subset",
                                 "articles": chosen_articles,
                                 "clean_utf8_bytes": chosen_clean_bytes,
                                 "exact_t32_records": chosen_records,
                                 "target_bpe_tokens": chosen_records * CONTEXT,
                                 "encoded_article_bpe_tokens": chosen_tokens,
                                 "represented_original_utf8_bytes": chosen_target_bytes},
        "tokenizer": {"kind": "byte_bpe", "vocabulary": VOCABULARY,
                      "sha256": "sha256:" + model.sha256, "source": "existing canonical V1024 model"},
        "heldout_cache_identity": heldout_caches,
        "cache": {"path": str(train_cache), "format": "NPRTBPEV1", "context": CONTEXT,
                  "records": chosen_records, "target_bpe_tokens": chosen_records * CONTEXT,
                  "encoded_article_bpe_tokens": chosen_tokens,
                  "represented_original_utf8_bytes": chosen_target_bytes,
                  "sha256": "sha256:" + cache_hash, "self_test": cache_identity},
        "training_order": {"algorithm": "SplitMix64(global_selection_index + state) modulo record_count; with replacement",
                            "seed": ORDER_SEED, "batch_size": BATCH_SIZE,
                            "hard_ceiling_steps": HARD_CEILING_STEPS,
                            "selection_count": HARD_CEILING_STEPS * BATCH_SIZE,
                            "hash": order_hash(chosen_records)},
        "split_leakage": {"selected_train_intersects_validation": 0,
                          "selected_train_intersects_development": 0,
                          "selected_train_intersects_final_test": 0,
                          "validation_article_ids": len(all_split_ids["validation"]),
                          "development_article_ids": len(all_split_ids["development"]),
                          "final_test_article_ids": len(all_split_ids["final_test"]),
                          "id_hash_ledger_sha256": "sha256:" + ledger_hash},
        "exclusions": exclusions,
        "body_records": body_records,
        "materialization_seconds": time.perf_counter() - started,
        "final_test_opened": False,
        "final_test_body_scan_scope": "cleaning_and_exact_text_deduplication_only",
        "final_test_dedupe_only_scan": {
            "performed": True,
            "purpose": "preserve corpus-wide exact-text deduplication before article split assignment and verify split integrity",
            "cleaned_text_identity_only": True,
            "tokenized_for_training_or_evaluation": False,
            "cached_for_training_or_evaluation": False,
            "sampled_for_quality_evaluation": False,
            "model_facing_or_quality_facing_access": False,
        },
        "final_test_tokenized_for_training_or_evaluation": False,
        "final_test_used_for_training_or_quality": False,
        "final_test_evaluated": False,
    }
    identity["manifest_sha256"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = private_root / "expanded-train-manifest.json"
    write_json(manifest_path, identity)
    spool_path.unlink()
    encode_input.unlink()
    encode_output.unlink()
    for candidate in (full_candidate, subset_candidate):
        candidate.unlink(missing_ok=True)
    print("materialization_status=PASS")
    print(f"eligible_train_articles={split_counts['train']}")
    print(f"eligible_train_clean_utf8_bytes={split_bytes['train']}")
    print(f"chosen_data_method={'full_train' if use_full else 'deterministic_expanded_subset'}")
    print(f"full_train_t32_records={full_records}")
    print(f"subset_articles={subset_articles}")
    print(f"subset_records={subset_records}")
    print(f"chosen_articles={chosen_articles}")
    print(f"chosen_records={chosen_records}")
    print(f"chosen_target_bpe_tokens={chosen_records * CONTEXT}")
    print(f"cache_sha256=sha256:{cache_hash}")
    print(f"training_order_hash={identity['training_order']['hash']}")
    print(f"manifest={manifest_path}")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    # Current canonical corpus remeasurement (source aggregate
    # sha256:b3185ea689ffa64c71f5ea8fe25f3779dbf1f0cd0d103b6e2fe05792e635bf86)
    # yields three fewer eligible TRAIN articles than the older repository
    # metadata. Keep this check fail-closed against the measured source state.
    parser.add_argument("--expected-train-articles", type=int, default=259_417)
    parser.add_argument("--expected-train-clean-bytes", type=int, default=1_084_316_091)
    args = parser.parse_args()
    try:
        materialize(args)
        return 0
    except Exception as error:
        print(f"materialization_status=FAIL error={type(error).__name__}:{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
