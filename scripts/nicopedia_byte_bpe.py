#!/usr/bin/env python3
"""Deterministic raw-byte BPE for private Nicopedia artifacts.

The canonical model is deliberately small and dependency-free:

  NPRTBPEM1\n
  vocabulary:u32be (=1024), merge_count:u32be (=768)
  merge[0..767]: left:u16be, right:u16be

Token 256+i is the concatenation of merge[i].  The SHA-256 of those exact
bytes is the tokenizer identity.  Literal byte IDs 0..255 are immutable.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
from pathlib import Path
import struct
import tempfile
from typing import Iterable, Sequence


MODEL_MAGIC = b"NPRTBPEM1\n"
CACHE_MAGIC = b"NPRTBPEV1\n"
VOCABULARY = 1024
MERGE_COUNT = 768
HASH_BYTES = 32


def _pair(left: int, right: int) -> int:
    return (left << 16) | right


def _unpair(value: int) -> tuple[int, int]:
    return value >> 16, value & 0xFFFF


@dataclass(frozen=True)
class ByteBpeModel:
    merges: tuple[tuple[int, int], ...]
    canonical: bytes
    sha256: str
    expansions: tuple[bytes, ...]

    @property
    def identity(self) -> str:
        return "sha256:" + self.sha256

    def encode(self, payload: bytes) -> list[int]:
        if not payload:
            return []
        tokens = list(payload)
        previous = [index - 1 for index in range(len(tokens))]
        following = [index + 1 for index in range(len(tokens))]
        following[-1] = -1
        rank = {_pair(left, right): index for index, (left, right) in enumerate(self.merges)}
        queue: list[tuple[int, int]] = []

        def enqueue(position: int) -> None:
            if position < 0 or following[position] < 0:
                return
            merge_rank = rank.get(_pair(tokens[position], tokens[following[position]]))
            if merge_rank is not None:
                heapq.heappush(queue, (merge_rank, position))

        for position in range(len(tokens) - 1):
            enqueue(position)
        while queue:
            merge_rank, left_pos = heapq.heappop(queue)
            right_pos = following[left_pos]
            if right_pos < 0:
                continue
            left, right = self.merges[merge_rank]
            if tokens[left_pos] != left or tokens[right_pos] != right:
                continue
            tokens[left_pos] = 256 + merge_rank
            after = following[right_pos]
            following[left_pos] = after
            if after >= 0:
                previous[after] = left_pos
            following[right_pos] = -2
            enqueue(previous[left_pos])
            enqueue(left_pos)
        result: list[int] = []
        position = 0
        while position >= 0:
            result.append(tokens[position])
            position = following[position]
        return result

    def decode(self, tokens: Iterable[int]) -> bytes:
        output = bytearray()
        for token in tokens:
            if token < 0 or token >= VOCABULARY:
                raise ValueError("TOKEN_ID_RANGE")
            output.extend(self.expansions[token])
        return bytes(output)

    def token_byte_length(self, token: int) -> int:
        if token < 0 or token >= VOCABULARY:
            raise ValueError("TOKEN_ID_RANGE")
        return len(self.expansions[token])


def canonical_model_bytes(merges: Sequence[tuple[int, int]]) -> bytes:
    if len(merges) != MERGE_COUNT:
        raise ValueError("BPE_MERGE_COUNT")
    output = bytearray(MODEL_MAGIC)
    output.extend(struct.pack(">II", VOCABULARY, MERGE_COUNT))
    for index, (left, right) in enumerate(merges):
        token_id = 256 + index
        if not (0 <= left < token_id and 0 <= right < token_id):
            raise ValueError("BPE_MERGE_REFERENCE")
        output.extend(struct.pack(">HH", left, right))
    return bytes(output)


def parse_model(payload: bytes) -> ByteBpeModel:
    expected_size = len(MODEL_MAGIC) + 8 + MERGE_COUNT * 4
    if len(payload) != expected_size:
        raise ValueError("BPE_MODEL_SIZE")
    if payload[:len(MODEL_MAGIC)] != MODEL_MAGIC:
        raise ValueError("BPE_MODEL_MAGIC")
    vocabulary, merge_count = struct.unpack_from(">II", payload, len(MODEL_MAGIC))
    if vocabulary != VOCABULARY or merge_count != MERGE_COUNT:
        raise ValueError("BPE_MODEL_HEADER")
    offset = len(MODEL_MAGIC) + 8
    merges = []
    expansions: list[bytes] = [bytes([value]) for value in range(256)]
    for index in range(MERGE_COUNT):
        left, right = struct.unpack_from(">HH", payload, offset)
        offset += 4
        token_id = 256 + index
        if left >= token_id or right >= token_id:
            raise ValueError("BPE_MODEL_FORWARD_REFERENCE")
        expansion = expansions[left] + expansions[right]
        if not expansion:
            raise ValueError("BPE_MODEL_EMPTY_TOKEN")
        merges.append((left, right))
        expansions.append(expansion)
    digest = hashlib.sha256(payload).hexdigest()
    return ByteBpeModel(tuple(merges), payload, digest, tuple(expansions))


def load_model(path: Path) -> ByteBpeModel:
    return parse_model(path.read_bytes())


def train_model(article_payloads: Sequence[bytes], merge_count: int = MERGE_COUNT) -> ByteBpeModel:
    """Train without ever linking the tail of one article to the next.

    Counts are maintained incrementally.  Pair occurrence lists are append-only;
    stale positions are validated when their pair is selected.  Selection is
    max frequency followed by ascending (left token ID, right token ID).
    """
    if merge_count != MERGE_COUNT:
        raise ValueError("BPE_REQUIRES_768_MERGES")
    token = array("H")
    previous = array("i")
    following = array("i")
    counts: dict[int, int] = defaultdict(int)
    occurrences: dict[int, array] = {}

    def add_occurrence(pair_value: int, position: int) -> None:
        bucket = occurrences.get(pair_value)
        if bucket is None:
            bucket = array("I")
            occurrences[pair_value] = bucket
        bucket.append(position)

    for payload in article_payloads:
        if not payload:
            continue
        base = len(token)
        for index, value in enumerate(payload):
            token.append(value)
            previous.append(base + index - 1 if index else -1)
            following.append(base + index + 1 if index + 1 < len(payload) else -1)
        for index in range(len(payload) - 1):
            position = base + index
            pair_value = _pair(payload[index], payload[index + 1])
            counts[pair_value] += 1
            add_occurrence(pair_value, position)
    if not token:
        raise ValueError("BPE_EMPTY_TRAIN_CORPUS")

    priority = [(-count, *_unpair(pair_value), pair_value) for pair_value, count in counts.items()]
    heapq.heapify(priority)
    merges: list[tuple[int, int]] = []
    for merge_index in range(MERGE_COUNT):
        selected = None
        while priority:
            negative, left, right, pair_value = heapq.heappop(priority)
            if counts.get(pair_value, 0) == -negative and -negative > 0:
                selected = pair_value, left, right
                break
        if selected is None:
            raise ValueError(f"BPE_INSUFFICIENT_PAIRS_AT_{merge_index}")
        pair_value, left, right = selected
        new_token = 256 + merge_index
        changed: set[int] = {pair_value}
        for left_pos in occurrences.get(pair_value, ()):
            right_pos = following[left_pos]
            if right_pos < 0 or token[left_pos] != left or token[right_pos] != right:
                continue
            before = previous[left_pos]
            after = following[right_pos]
            counts[pair_value] -= 1
            if before >= 0:
                old = _pair(token[before], token[left_pos])
                counts[old] -= 1
                changed.add(old)
            if after >= 0:
                old = _pair(token[right_pos], token[after])
                counts[old] -= 1
                changed.add(old)
            token[left_pos] = new_token
            following[left_pos] = after
            if after >= 0:
                previous[after] = left_pos
            following[right_pos] = -2
            previous[right_pos] = -2
            if before >= 0:
                new_pair = _pair(token[before], new_token)
                counts[new_pair] += 1
                add_occurrence(new_pair, before)
                changed.add(new_pair)
            if after >= 0:
                new_pair = _pair(new_token, token[after])
                counts[new_pair] += 1
                add_occurrence(new_pair, left_pos)
                changed.add(new_pair)
        merges.append((left, right))
        for changed_pair in changed:
            count = counts.get(changed_pair, 0)
            if count > 0:
                changed_left, changed_right = _unpair(changed_pair)
                heapq.heappush(priority, (-count, changed_left, changed_right, changed_pair))
    return parse_model(canonical_model_bytes(merges))


def write_bpe_cache(path: Path, articles: Sequence[tuple[int, int, str]], context: int,
                    model: ByteBpeModel,
                    preencoded: Sequence[Sequence[int]] | None = None) -> dict[str, object]:
    if context <= 0:
        raise ValueError("CACHE_CONTEXT")
    encoded_articles: list[tuple[int, list[int]]] = []
    chunks = 0
    clean_bytes = 0
    target_bytes = 0
    if preencoded is not None and len(preencoded) != len(articles):
        raise ValueError("BPE_PREENCODED_COUNT")
    for index, (_, article_hash, text) in enumerate(articles):
        raw = text.encode("utf-8")
        encoded = list(preencoded[index]) if preencoded is not None else model.encode(raw)
        if model.decode(encoded) != raw:
            raise ValueError("BPE_ROUND_TRIP")
        clean_bytes += len(raw)
        encoded_articles.append((article_hash, encoded))
        chunks += max(0, (len(encoded) - 1) // context)
        for start in range(0, len(encoded) - context, context):
            window = encoded[start:start + context + 1]
            if len(window) == context + 1:
                target_bytes += sum(model.token_byte_length(value) for value in window[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(CACHE_MAGIC)
        handle.write(struct.pack(">II", context, VOCABULARY))
        handle.write(bytes.fromhex(model.sha256))
        handle.write(struct.pack(">Q", chunks))
        for article_hash, encoded in encoded_articles:
            for start in range(0, len(encoded) - context, context):
                window = encoded[start:start + context + 1]
                if len(window) != context + 1:
                    continue
                handle.write(struct.pack(">Q", article_hash))
                handle.write(struct.pack(">" + "H" * len(window), *window))
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    return {
        "path": str(path.resolve()), "articles": len(articles), "clean_utf8_bytes": clean_bytes,
        "chunks": chunks, "target_tokens": chunks * context,
        "target_utf8_bytes": target_bytes, "tokenizer_hash": model.identity,
        "sha256": "sha256:" + digest,
    }


def read_bpe_cache(path: Path, model: ByteBpeModel) -> tuple[int, int, str, list[tuple[int, tuple[int, ...]]]]:
    payload = path.read_bytes()
    header_size = len(CACHE_MAGIC) + 8 + HASH_BYTES + 8
    if len(payload) < header_size or payload[:len(CACHE_MAGIC)] != CACHE_MAGIC:
        raise ValueError("BPE_CACHE_MAGIC")
    context, vocabulary = struct.unpack_from(">II", payload, len(CACHE_MAGIC))
    offset = len(CACHE_MAGIC) + 8
    tokenizer_hash = payload[offset:offset + HASH_BYTES].hex()
    offset += HASH_BYTES
    count = struct.unpack_from(">Q", payload, offset)[0]
    offset += 8
    if context <= 0 or vocabulary != VOCABULARY:
        raise ValueError("BPE_CACHE_HEADER")
    if tokenizer_hash != model.sha256:
        raise ValueError("BPE_CACHE_TOKENIZER_HASH")
    record_size = 8 + 2 * (context + 1)
    if len(payload) != offset + count * record_size:
        raise ValueError("BPE_CACHE_SIZE")
    records = []
    for _ in range(count):
        article_hash = struct.unpack_from(">Q", payload, offset)[0]
        offset += 8
        window = struct.unpack_from(">" + "H" * (context + 1), payload, offset)
        offset += 2 * (context + 1)
        if any(token >= vocabulary for token in window):
            raise ValueError("BPE_CACHE_TOKEN_RANGE")
        records.append((article_hash, window))
    return context, vocabulary, "sha256:" + tokenizer_hash, records


def quality_metrics(articles: Sequence[tuple[int, int, str]], model: ByteBpeModel,
                    preencoded: Sequence[Sequence[int]] | None = None) -> dict[str, object]:
    total_bytes = total_chars = total_tokens = 0
    ratios: list[float] = []
    if preencoded is not None and len(preencoded) != len(articles):
        raise ValueError("BPE_PREENCODED_COUNT")
    for index, (_, _, text) in enumerate(articles):
        raw = text.encode("utf-8")
        encoded = list(preencoded[index]) if preencoded is not None else model.encode(raw)
        if model.decode(encoded) != raw:
            raise ValueError("BPE_ROUND_TRIP")
        total_bytes += len(raw)
        total_chars += len(text)
        total_tokens += len(encoded)
        if raw:
            ratios.append(len(encoded) / len(raw))
    ratios.sort()
    def pct(q: float) -> float:
        if not ratios:
            return 0.0
        index = min(len(ratios) - 1, max(0, math.ceil(q * len(ratios)) - 1))
        return ratios[index]
    return {
        "articles": len(articles), "utf8_bytes": total_bytes, "bpe_tokens": total_tokens,
        "mean_tokens_per_unicode_character": total_tokens / max(1, total_chars),
        "mean_utf8_bytes_per_token": total_bytes / max(1, total_tokens),
        "compression_ratio_vs_v256": total_tokens / max(1, total_bytes),
        "p50_tokenized_length_ratio": pct(0.50), "p95_tokenized_length_ratio": pct(0.95),
        "unknown_rate": 0.0, "exact_byte_round_trip": True,
    }


def self_test() -> None:
    state = 0x12345678
    randomish = bytearray()
    for _ in range(131072):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        randomish.append((state >> 16) & 0xFF)
    fixture = bytes(randomish) + bytes(range(256)) * 64 + ("日本語 ASCII\n記号😀mixed".encode("utf-8") * 512)
    articles = [fixture[:len(fixture) // 2], fixture[len(fixture) // 2:]]
    first = train_model(articles)
    second = train_model(articles)
    assert first.canonical == second.canonical and first.sha256 == second.sha256
    assert len(first.merges) == MERGE_COUNT and len(first.expansions) == VOCABULARY
    assert all(first.expansions[value] == bytes([value]) for value in range(256))
    samples = [b"", bytes(range(256)), "日本語\nASCII ! 😀".encode("utf-8"), b"\x00\xff\xc0\x80"]
    for sample in samples:
        assert first.decode(first.encode(sample)) == sample
    with tempfile.TemporaryDirectory(prefix="phonelm-bpe-selftest-") as temporary:
        cache_path = Path(temporary) / "cache.bin"
        cache_articles = [(1, 0x1234, "日本語 ASCII\n😀" * 20)]
        report = write_bpe_cache(cache_path, cache_articles, 8, first)
        context, vocabulary, identity, records = read_bpe_cache(cache_path, first)
        assert context == 8 and vocabulary == VOCABULARY and identity == first.identity
        assert records and report["target_tokens"] == len(records) * context
        truncated = Path(temporary) / "truncated.bin"
        truncated.write_bytes(cache_path.read_bytes()[:-1])
        try:
            read_bpe_cache(truncated, first)
            raise AssertionError("truncated cache accepted")
        except ValueError:
            pass
        wrong_hash = bytearray(cache_path.read_bytes())
        wrong_hash[len(CACHE_MAGIC) + 8] ^= 1
        bad_hash_path = Path(temporary) / "wrong-hash.bin"
        bad_hash_path.write_bytes(wrong_hash)
        try:
            read_bpe_cache(bad_hash_path, first)
            raise AssertionError("wrong tokenizer hash accepted")
        except ValueError:
            pass
    # Separate articles have no synthetic (0x41, 0x42) boundary pair.
    boundary_fixture = [b"A" * 2048, b"B" * 2048, fixture]
    boundary_model = train_model(boundary_fixture)
    assert boundary_model.merges[0] in ((65, 65), (66, 66))
    malformed = bytearray(first.canonical)
    struct.pack_into(">H", malformed, len(MODEL_MAGIC) + 8, 256)
    try:
        parse_model(bytes(malformed))
        raise AssertionError("malformed model accepted")
    except ValueError:
        pass
    print("nicopedia_byte_bpe_self_test=PASS tokenizer_hash=" + first.identity)


if __name__ == "__main__":
    self_test()
