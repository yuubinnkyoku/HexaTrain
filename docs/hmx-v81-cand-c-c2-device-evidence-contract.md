# Candidate C — C2 Device Experiment: Evidence Contract and Audit Record

Status: **DESIGN AND AUDIT RECORD, CPU-ONLY**, 2026-10-30. Companion to
`docs/hmx-v81-cand-c-c2-device-prereg.md` (revision 2). This document
authorises nothing: no DSP source, no DSP build, no QAIRT invocation, no ADB,
no device contact. It exists so that the single registered C2 device attempt,
if the user ever approves it, has an unambiguous observation contract, a
complete digest inventory, and a pre-written disassembly audit plan — and so
the CPU-only state of the research is auditable without asking the simulator
to vouch for itself.

Sections:

1. [Digest inventory (9 claimed, 8 named, 7 distinct values)](#1)
2. [Payload census, independently recounted](#2)
3. [Independent classifier verification](#3)
4. [DSP-side evidence contract (design only)](#4)
5. [Host-side evidence contract and the `sequence<long>` ABI](#5)
6. [Disassembly audit plan (future requirement)](#6)
7. [Final three-design comparison (research)](#7)
8. [What remains unproven](#8)

---

## 1. Digest inventory

The C2 pre-registration (§4) mentions nine digests: three in its own table
plus "the six frozen streaming digests". Counted properly there are **8 named
digests, 7 distinct SHA-256 values**, because the pre-registration's third row
(the optional diagnostic recon) **is** one of the six streaming digests, and
because two streaming digests share one value by design. The pre-registration
has been corrected to say so.

| # | Report key | Target array | Bytes | Serialisation | SHA-256 | First registered | Changed 2026-10-30 | Independent verification |
| --: | --- | --- | ---: | --- | --- | --- | --- | --- |
| 1 | `b3_r2_sha256_input_truth` | G8 fixture input truth, X then W | 16,384 | f64 LE, X then W | `49b0c69c…f236d0` | `969e26e` | no | dump re-hash (`hmx_c2_independent_classifier_test.py`) |
| 2 | `b3_r2_sha256_oracle_double` | FP64 oracle output | 8,192 | f64 LE | `5f61ff94…ada02c0` | `969e26e` | no | C-run assert only (not in the dump) |
| 3 | `b3_r2_sha256_operands_deploy` | deployed operand condition (ahi, alo, whi, wlo, a_plain, w_plain) | 12,288 | u16 LE | `e5193d99…1f0d47a3` | `969e26e` | no | dump re-hash |
| 4 | `b3_r2_sha256_plain_expected_bits` | Plain subcase recon (= prereg §4 row 3, the optional diagnostic recon) | 2,048 | u16 LE, decoded cell order `s*32+o` | `0e58d8f8…8e9336bb` | `969e26e` | no | dump re-hash |
| 5 | `b3_r2_sha256_b3_expected_bits` | B3 subcase recon | 2,048 | u16 LE, decoded cell order | `687509a2…84e030939` | `969e26e` | no | dump re-hash |
| 6 | `b3_r2_sha256_operands_fp32` | FP32-derived operand condition | 12,288 | u16 LE | `e5193d99…1f0d47a3` (**equals #3 by design**: `RN16(FP32)` reproduces the deployed operands bit-for-bit on this fixture — the measured "FP32 rounding is a no-op" result) | `969e26e` | no | C-run assert of the equality |
| 7 | `cand_c_sha_plain_pos_bits` | Plain subcase positive raw bits (= prereg §4 row 1, the C2 expected `P`) | 2,048 | u16 LE, decoded cell order | `82107d6e…0f76ffb8` | `4648a0d` | no | dump re-hash **and** full independent re-derivation from the pinned operands (float32 accumulation + RN16 convert law re-implemented in Python) |
| 8 | `cand_c_sha_plain_neg_bits` | Plain subcase negative raw bits (= prereg §4 row 2, the C2 expected `X` final) | 2,048 | u16 LE, decoded cell order | `3fcddb24…6bce1a3309` | `4648a0d` | no | idem |

Not counted as separate digests: `c2_nominal_sha_p` / `c2_nominal_sha_x` are
recomputations of rows 7/8 by the C2 model, and the `# sha256_*` header lines
inside the dump file are echoes of rows 1–5. Re-registering the same bytes
under a new name is not independent evidence.

All eight were re-verified on 2026-10-30 after the audit changes and are
unchanged. No registered digest was edited.

## 2. Payload census, independently recounted

Recounted from the arrays the independent test re-derived itself (not from the
simulator's own census print):

| Set | Count | Notes |
| --- | ---: | --- |
| positive-side non-zero | 548 | = weak (546) + ambiguous subnormal (2) |
| negative-side non-zero | 475 | **strong C2 evidence carriers** — matches the registered census |
| both sides zero | 1 | the single non-discriminating cell |
| discriminating (`pos != neg`) | 1,023 | = 475 + 546 + 2 |
| subnormal positive payload | 2 | negative side has none (`subnormal_neg=0`) |
| mutual exclusivity (`both_nonzero`) | 0 | sign separation holds |
| exclusive partition (strong + weak + ambiguous + both-zero) | 1,024 | 475 + 546 + 2 + 1 |

The exclusive partition sums to exactly 1,024 cells; every set above is
disjoint as labelled. All non-zero payloads have bit 15 clear.

## 3. Independent classifier verification

`host_tests/hmx_acc_residual_rpc/hmx_c2_independent_classifier_test.py`
re-implements, in Python and from the pre-registration text only:

- the convert law (`RN16(64·acc)`, negative clamp) and the float32
  accumulation of the 32-product dot product, re-deriving both registered
  arrays from the digest-pinned operands — both re-derived digests match
  `82107d6e…` / `3fcddb24…`;
- the §9 decision procedure (as amended 2026-10-30: `0xDEAD` before `X`,
  signatures before special values), applied to fixed raw bit arrays and
  synthetic observations.

Results (67 checks, 0 failures):

- **Fixed battery (16 cases)** covers every terminal label of §9/§10:
  nominal (`C2_CPU_NOMINAL_MATCH`), whole-tile post-Store-#2
  (`C2_ORDERING_VIOLATED`), all-`0xDEAD` and half-copy
  (`C2_COPY_NOT_EXECUTED`), single C1 cell (`C1_STORE_VISIBILITY_SUSPECT`),
  X broken (`C2_STORE2_PATH_BROKEN`), X-broken+`0xDEAD` order pin,
  C1+subnormal inconclusive pin, zero-only, FTZ on one and on both subnormal
  cells, unknown FP16 bit pattern, P/X swap, both-tile decode-order swap,
  P-only decode-order swap, contract failure.
- **Coincidental-match question, measured:** a P-only decode-order swap
  produces **zero** coincidental non-zero-negative matches on this fixture
  (`coincidental_c2_nonzero=0`), so it lands on `C2_CAUSE_INCONCLUSIVE`, not
  on a C2 verdict. Sentinel coincidences are impossible by construction
  (`0xA5A5`/`0xDEAD` have bit 15 set; no payload can).
- **Deterministic-seed mixtures (128 seeds, 5 profiles)** — every
  classification satisfies the fail-closed invariants (a `0xDEAD` residue
  always stops first; an X mismatch never survives to a signature verdict;
  `SUPPORTED` requires the full 1,024-cell payload match; a C2 verdict
  requires non-zero negative evidence and excludes C1/`0xDEAD`). Observed
  class spread: `C2_COPY_NOT_EXECUTED` 26, `C2_STORE2_PATH_BROKEN` 37,
  `C2_CAUSE_INCONCLUSIVE` 33, `C1_STORE_VISIBILITY_SUSPECT` 16,
  `C2_ORDERING_VIOLATED` 16. No mismatch was found, so no minimal repro had
  to be saved.
- **Label contract:** no CPU context renders
  `C2_MINIMAL_SEQUENCE_SUPPORTED`; the CPU nominal outcome is
  `C2_CPU_NOMINAL_MATCH`.

Known limitation, stated rather than hidden: the C model and the Python test
are two implementations of the same registered text. If the text itself were
ambiguous in a way both implementations read the same wrong way, this
cross-check would not see it. The fixed battery pins every branch to the class
the pre-registration names for it, which is the mitigation.

## 4. DSP-side evidence contract (design only)

If the user ever approves the registered attempt, the single DSP build must
produce, and the run must preserve, all of the following. Nothing here is
implemented; this is the checklist the future implementation and its audit are
held to.

1. Exactly **1 MMA packet** (after 1 `mxclracc`).
2. Exactly **2 Convert** packets (`Rs = 0x0001` then `Rs = 0x1000`).
3. Exactly **2 HMX Store** packets, both targeting VTCM `+6144` (identical
   physical address; the identity is the experiment).
4. Exactly **1 HVX copy loop of 16 vectors of 128 B**, VTCM `+6144` → L2
   `+384`, between Store #1 and Store #2, with no other instruction class in
   between (§7 of the pre-registration).
5. **Exact `Rs` history** `0x0001`, `0x1000` echoed in metadata.
6. **Bias set selection** echoed per convert (set 0 then set 1).
7. **Store #1/#2 address match** echoed (both `+6144`).
8. **Full 2,048 B of P** (L2) and **full 2,048 B of X** (VTCM) returned to the
   host as zero-extended u16 in decoded cell order (§5 below).
9. **VTCM guards** (`+6016`, `+8192`) and **L2 guards** (L2 `+256`, `+2432`)
   checked on the DSP against `0x5A5A` before harvest, status + first
   mismatch reported.
10. **Untouched bytes** (the two gaps and the tail, 5,376 B) still `0x00`.
11. **Weight-tail evidence**: FNV-1a 32 of the last 128 B of the packed W
    tile, computed twice on the DSP — once in the prologue right after
    packing (before any guard fill) and once after Store #2 — both values
    echoed; the host asserts equality (the guard fill must not have eaten a
    live operand, the revision-1 failure).
12. **Build ID** of the stamped build echoed (twice, Mode 45 convention).
13. **DSP architecture** echo from the identity block.
14. **Cleanup status**: VTCM/L2/rpcmem release bookkeeping, 0 = clean.
15. The **readback order is fixed** (P from L2 first, then X from VTCM, then
    guards/bias) so a truncated run stays interpretable (prereg §2).

## 5. Host-side evidence contract and the `sequence<long>` ABI

### 5.1 The current contract, investigated

- Interface: `hexatrain_hmx_acc_residual.idl` —
  `long run(rout sequence<long> results, rout sequence<long> metadata, in long q, in long mode)`.
- The generated stub signature is
  `run(handle, results, resultsLen, metadata, metadataLen, q, mode)`: the
  **host** allocates both buffers with
  `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, words * sizeof(int))`
  (`probe_host.c`), passes the word lengths, and copies the contents out
  **before** `rpcmem_free` (Mode 45 does exactly this at `probe_host.c`
  lines 1202–1213).
- Current users: the identity block (4 result slots + 9 metadata words) and
  Mode 45 (11 slots + 32 words — `HMX_MODE45_RESULT_SLOTS`,
  `HMX_MODE45_METADATA_LEN` in `hmx_mode45.h`).
- Payload cells travel as **zero-extended u16 in int32 slots**: the DSP does
  `(int)cell` on a `uint16_t`, which yields 0..65,535 and never sign-extends.
  The host must mask `0xFFFF` before comparing against u16 constants.
- Cells are returned in **store crouton (pair-interleaved) order**; the host
  decodes to cell order `s*32+o` with the shared `store_u16` bijection.

### 5.2 The C2 design (documentation only — no IDL/DSP/host change is made)

`sequence<long>` needs **no signature change**. The rout sequences are
length-parameterised per call; the same interface carries Mode 45's 11+32
words and can carry C2's:

- **`results`: 2,048 slots = 8,192 B.** Slots 0..1023 = `P` (the positive
  tile harvested from L2), slots 1024..2047 = `X` final (the negative tile
  harvested from VTCM). Each slot is one u16 cell, zero-extended, **decoded
  cell order** (the host decodes before hashing, or the DSP pre-decodes —
  this must be fixed in the future implementation, and the registered digest
  arithmetic assumes decoded cell order `s*32+o`, u16 LE).
- **`metadata`: 32 words**, same allocation size as Mode 45:

| Word | Content | Expected |
| ---: | --- | --- |
| 0 | Build ID echo | stamped build under test |
| 1 | DSP architecture echo | identity block value |
| 2 / 3 | mode / q echo | 46 / registered q |
| 4 | `mxclracc` count | 1 |
| 5 | MMA packets | 1 |
| 6 | Convert count | 2 |
| 7 | Store count | 2 |
| 8 | HVX copy loops | 1 |
| 9 | HVX copy vectors | 16 |
| 10 / 11 | `Rs` #1 / #2 | `0x0001` / `0x1000` |
| 12 / 13 | bias set used by Convert #1 / #2 | 0 / 1 |
| 14 / 15 | Store #1 / #2 target offset | 6144 / 6144 (host asserts equality) |
| 16 | VTCM guard status | bit0 `guard_lo` intact, bit1 `guard_hi` intact |
| 17 | L2 guard status | bit0 low intact, bit1 high intact |
| 18 | untouched-region status | 1 (all 5,376 B still `0x00`) |
| 19 / 20 | W-tail FNV prologue / post-Store-#2 | equal |
| 21 | bias sets written | 2 |
| 22 | readbacks completed | 2 (P then X) |
| 23 | cleanup status | 0 |
| 24 | Build ID second echo | Mode 45 convention |
| 25 | first-failure reason code | 0 = none (fail-closed enumeration) |
| 26 / 27 | first mismatching guard word value / region code | 0 / 0 |
| 28–31 | reserved | 0 |

- **Host-side evidence, per run** (what the report of the real attempt must
  contain): P/X as 1,024 zero-extended u16 each; the decoded cell order
  mapping; the measured raw bits and their SHA-256; the two pre-registered
  SHA-256 values; per-region mismatch counts (P, X, VTCM guards, L2 guards,
  untouched bytes, W tail); C1/C2/special-value classification counts; the
  first contract-failure reason verbatim; RPC and close status. All of it
  printed, all of it preserved before `rpcmem_free`, and the raw bytes of
  both tiles preserved on any failure per §9 of the pre-registration.
- **Sizing sanity**: 8,192 B + 128 B of ION shared memory is three orders of
  magnitude below the allocations the APK already makes elsewhere; no FastRPC
  practical limit is approached. The mode number this would use (Mode 46)
  remains `NOT_RUN` and unregistered — registering it is DSP work and is
  **not** designed here.

## 6. Disassembly audit plan (future requirement)

If the registered attempt is ever approved, the stamped DSP binary is audited
statically **before** the run, and the audit demands:

1. exactly one MMA construct and no unexpected MMA packet anywhere in the
   mode's path;
2. exactly two Convert constructs and two HMX Store constructs;
3. Store #1 and Store #2 referencing the **same** X base address (register
   and offset evidence, not two symbols that happen to alias);
4. the HVX copy loop body: plain 128 B vector load + 128 B vector store, 16
   iterations, no widened/predicated/unaligned forms;
5. **no** FNV/CRC/checksum/scalar byte-wise read of X or P between Store #1
   and Store #2;
6. **no** `syncht`, barrier packet attribute, DMA-sync intrinsic, or any
   other fence in the critical path — the `asm volatile("" ::: "memory")`
   compiler barrier is allowed and must emit nothing;
7. **no** helper call in the critical path (a call can implicitly contain a
   barrier; audit the callee bodies or forbid the call);
8. **no** host-side reconstruction, reordering or data correction — the host
   compares bytes, it does not fix them.

Two honesty rules bound the audit itself:

- **Source is not machine code.** The number of HMX instructions written in
  the C source is not evidence about the object; the audit reads the actual
  disassembly of the stamped build.
- **Static structure is not execution order.** A clean disassembly shows the
  binary *contains* the registered sequence and *no forbidden construct*; it
  cannot prove how the hardware completed the operations. The ordering
  evidence remains the runtime observation (`P`'s content) under the §9
  procedure. `B3_R2_SYNC_UNPROVEN` is only ever changed by device evidence,
  never by a static audit.

## 7. Final three-design comparison (research)

Counts only — no cycles/ns/percentage estimate is registered anywhere in this
document (prereg §11; the full-tile pre-registration §4.4).

| Item | Streaming (registered, `B3_R2_FULL_TILE_CPU_READY`) | Candidate B (phase-reuse, preferred) | Candidate C (single shared OUT) |
| --- | --- | --- | --- |
| VTCM used | 10,752 B of 12,288 B (fresh act/wgt/pos/neg/recon + 2 bias) | MMA phase 8,704 B; peak 10,752 B (outputs reuse dead operand regions) | 10,752 B (4 resident operands + 1 shared OUT + 2 bias) |
| DSP L2 | 6,144 B staging (Alo, Wlo, Whi copy) + output evidence | 2,048 B (Plain output save) | peak 10,240 B raw / 6,144 B folded (5 tiles = 80 vectors copied out) |
| HMX MMA count (Plain+B3 run) | 4 | 4 | 4 |
| Convert / Store count | 4 / 4 | 4 / 4 | 4 / 4 |
| HVX copies | recon 16 vectors per subcase + operand restores (48 mid-kernel vectors) | recon 16 vectors per subcase + Plain 1-tile save; **0 mid-kernel operand transfers** | 5 tiles = 80 vectors (pos/neg/recon saves) + in-place recon |
| HMX operand reuse | none — every region fresh | out_neg→Wlo, out_recon→Ahi after their last MMA | none for operands (they are never rewritten); outputs share one buffer in place |
| Cross-unit ordering dependencies | MX store → later HVX read (Mode 45-supported direction) | same, **plus** operand-region reuse after MMA #3 — `reuse_after_mma3_unproven`, and **CB-FIX-1 open** | C1 (store→read, supported), **C2 (read→store over same buffer, the open hardware question)**, C3 (supported direction, new context), C4/C5 (program order) |
| CPU numerical result | registered arrays (rows 1–5 of §1) | identical bits to streaming (all six arrays 0/1024 diff) | identical bits (Plain pos/neg/recon digests equal streaming's) |
| Device evidence | components: Mode 45 self-test (`R2_ACC_REUSE_SUPPORTED`, Mode 45 only) | none beyond Mode 45's direction; no full-tile run | none for C2 (`C2 NOT_RUN`); Mode 45 covers C1's direction |
| Fail-closed diagnostics | contract gate + fault battery (14 faults) | + lifetime/alias checks (3 faults) | + C2 classifier with sentinel-based attribution, written-range ledger, independent Python audit (27 faults) |
| Implementation complexity | lowest (fresh buffers) | medium (aliasing bookkeeping) | highest (one buffer, L2 lifetime discipline, evidence evacuation) |
| Unresolved synchronization | `B3_R2_SYNC_UNPROVEN` (full tile) | reuse-after-MMA-#3 undocumented; CB-FIX-1 | C2 unproven (`NOT_RUN`); C4/C5 device-untested |

Read of the table: the three designs compute the same registered bits and
differ in memory schedule and in *which* unproven cross-unit property they
carry. Candidate B stays `CPU_VALIDATED_PREFERRED_DESIGN`. **Even if the C2
minimal experiment one day returns `C2_MINIMAL_SEQUENCE_SUPPORTED`, that does
not establish Candidate C**: C3 (the supported direction re-exercised in the
shared-buffer context), C4 (in-place per-vector recon ordering), C5 (Plain
evidence evacuation before the B3 re-pack), and a full-tile accuracy run would
all still be open. A single clean C2 result can only fail to disprove C2
(prereg §1).

## 8. What remains unproven

1. **C2** — the registered experiment asks the question; it has not run.
2. **C1 / C3** — Mode 45 direction evidence only; not re-established here.
3. **C4 / C5** — program-order properties, device-untested.
4. **CB-FIX-1** — Candidate B's sentinel/ordinal inconsistency, open,
   untouched by this audit.
5. **Device-side address acceptance** of the revision 2 VTCM map (`+6016`
   guard, the 128 B gaps) — allocator behaviour is a separate audit that has
   not been done.
6. **Mode 46** — unregistered, `NOT_RUN`; the ABI above is a design, not an
   implementation.
7. **`B3_R2_SYNC_UNPROVEN`** — unchanged by everything in this audit, by
   construction.

## 9. Reproduction record, host CPU only (2026-10-30)

This section binds the numbers quoted above to the code that produced them, so
a reviewer can re-run instead of trusting. **None of it is device evidence**: no
DSP binary was built, nothing ran on a phone, and `C2` stays `NOT_RUN`.

| Evidence | Producer | Result | Bound to |
| --- | --- | --- | --- |
| C model + fault battery | `clang -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off` on `hmx_b3_r2_full_tile_sim.c` | `b3_r2_checks=12776 fail=0`, `b3_r2_fault_injection=PASS detected=27/27` | classifier and gates of `bae32cf` |
| Map + written-range ledger | same run | `c2_prereg … vtcm_used=8960 margin=3328 rev2_overlaps=0 rev1_overlaps=1`, `c2_written_ranges count=7 bytes=6912 overlaps=0 mismatch_declared=0` | map in `b22836e`, ledger in `bae32cf` |
| Nominal bits | same run | `c2_nominal p_diff=0 x_diff=0 class=C2_CPU_NOMINAL_MATCH`, `sha_p=82107d6e…`, `sha_x=3fcddb24…` | §1 digest inventory |
| Independent classifier | `python host_tests/hmx_acc_residual_rpc/hmx_c2_independent_classifier_test.py` | `checks=67 fail=0`, `verdict=C2_INDEPENDENT_AUDIT_PASS` | `dfaed70` |
| Class enum order | source review | numeric order now equals the §9 order; no behaviour change | d4cfbc2 |

Toolchain used: clang 22.1.8 (LLVM), target `x86_64-pc-windows-msvc`, with the
MSVC 2022 BuildTools environment supplying the UCRT headers.

Three things a re-runner must know:

1. **The gate needs gcc/g++ on `PATH`.** On the development machine they live in
   the MSYS2 UCRT64 tree (`C:\msys64\ucrt64\bin`; gcc/g++ 15.2.0,
   `x86_64-w64-mingw32`) but are not on the default `PATH`. Prepending that
   directory for the session is sufficient: no install and no script change.
   With it set, `scripts/run_hmx_b3_r2_sim.ps1` returns `b3_r2_sim=PASS`
   (`b3_r2_sim_gcc_exit=0`) and `scripts/verify.ps1 -Profile Fast` returns
   `verify-local-fast PASS` (PASS=10 FAIL=0 SKIP=21, `host-fast-contract`
   PASS). Without it both report an environment failure, which must never be
   read as a code defect. The runner's clang leg still prints
   `SKIP (clang not found on PATH)`; the clang 22.1.8 numbers above come from
   the equivalent manual invocation.
2. On the MSVC target `-lm` resolves to `m.lib`, which does not exist; the
   successful build omits `-lm`.
3. `build/hmx_b3_r2_registered_arrays.txt` is regenerated by every run and is
   gitignored. The Python audit re-hashes the sections it finds there, so its
   `digests … re-hashed OK` line only shows that the dump and the pinned
   constants agree. Regenerate the dump from the current source before trusting
   the audit.
4. Cross-compiler agreement was checked directly: the gcc 15.2.0 build and the
   clang 22.1.8 build of the same source produce byte-identical stdout (`diff`
   clean, including all six SHA-256 lines and `b3_r2_checks=12776 fail=0`), so
   the registered digests are not an artefact of one compiler's code
   generation.
