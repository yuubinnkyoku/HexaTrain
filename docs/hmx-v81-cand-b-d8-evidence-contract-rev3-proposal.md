# Candidate B — D8 Device Experiment: Evidence Contract (Revision 3 Proposal)

Status: **`D8_REV3_PROPOSED_CPU_ONLY`**, 2026-11-10. Companion to
`docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md`. This document is a
**proposal**. Revision 2
(`docs/hmx-v81-cand-b-d8-evidence-contract.md`) remains the registered
evidence contract and is not modified.

This document authorises nothing: no DSP source, no DSP build, no QAIRT
invocation, no ADB, no device contact, no Mode 46, no release, no merge.
`D8 NOT_RUN`, `B3_R2_SYNC_UNPROVEN`, Mode 46 unregistered and approval
`NOT GRANTED` are unchanged.

Revision 3 exists because revision 2 §3 could not be implemented against the
existing `run()`: words 0..8 have generic owners, words 3/7/8 are rewritten
after the handler, `q=0..3` is rejected before dispatch, the `rout` length is
not a returned count, a non-zero return discards the arrays, and the guard had
no channel. Revision 3 fixes the contract, not the device.

## 1. Digest and value inventory (unchanged)

| # | Report key | Array | Bytes | Serialisation | Value |
| --: | --- | --- | ---: | --- | --- |
| 1 | `d8_sha256_operands_deploy` | ahi, alo, whi, wlo, a_plain, w_plain | 12,288 | u16 LE | `e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3` |
| 2–5 | `d8_fnv_*` | packed Ahi / Alo / Whi / Wlo | 2,048 each | FNV-1a 32, LE bytes | `c6231266` / `3b2d22d2` / `8b88609f` / `a66af695` |
| 6–9 | `d8_sha256_*` | safe / wlo / ahi / both reference | 2,048 each | u16 LE, cell order `s*32+o` | `c9abf30e…20203d` / `f0b3b4aa…81f6fb` / `b6328e88…760fcc` / `5285b463…2535d` |

No registered digest or expected value is changed by revision 3.

## 2. DSP-side evidence contract

If the attempt is ever approved, the single stamped build must produce and the
host must preserve all of the following.

1. **4 operand packs** into `+0`, `+2048`, `+4096`, `+6144` (16 × 128 B each)
   and **2 bias-set writes** into `+10240`, `+10496`, before `mxclracc`.
2. **Guard init**: `+10752..+12288` set to `0x00` with 12 × 128 B HVX stores,
   before `mxclracc`; then a **guard pre hash** (FNV-1a 32, LE bytes).
3. **Exactly 1 `mxclracc`, exactly 3 MMA packets**, in the registered order:
   `+0/+4096`, `+0/+6144`, `+2048/+4096`. No other MMA in the path.
4. **Fill order after MMA #3**: out_pos `+8192` first, then Wlo `+6144` (B/C),
   then Ahi `+0` (B/D), each 2,048 B of `0xA5A5`, each **16 × 128 B HVX vector
   stores**. Masks A=`1`, B=`7`, C=`3`, D=`5`.
5. **Exactly 1 Convert** (`Rs = 0x0001`, bias set 0, shape 0, scale 64) and
   **exactly 1 HMX Store** into `+8192`, both after every fill.
6. **Guard post hash** after Store #1, compared with the pre hash; the first
   differing byte offset is echoed. The guard is not written during the run.
7. **No `syncht`, barrier, DMA sync, dummy `cvt` wait, host RPC, checksum loop
   or scalar byte read of `+0`/`+6144`/`+8192`** in the critical interval. The
   compiler-only barrier is allowed and must emit nothing.
8. **No write outside the declared regions**: `+0..+11008` as registered, the
   guard `+10752..+12288` only by the registered init, and nothing above
   `+12288`.
9. **Full 2,048 B of `out_pos`** returned as zero-extended u16 in decoded cell
   order `u16 LE`, with `slots_written = 1024`.
10. **Metadata echo** of words 9..31 per §3, with the build id in word 9.
11. **Build identity**: identity mode 30 (its own `q=6`) echoes the build id in
    `metadata[2]`; the D8 run echoes the same value in word 9.
12. **Cleanup status** in word 30, written by `run()` for mode 46.
13. **Readback order**: results, then metadata, then the failure echo.

## 3. Metadata 32-word layout

### 3.1 Ownership

| Words | Owner | Notes |
| --- | --- | --- |
| 0..8 | `run()` | unchanged meaning on every mode; D8 must not write them |
| 9..31 | D8 handler | except word 30, written by `run()` for mode 46 |

`run()` rewrites words 3 (elapsed), 7 (HMX executed) and 8 (trials) after the
handler, so they can never carry D8 evidence. D8 uses `metadata[7]=1` and
`metadata[8]=1` only as generic preconditions checked by the host; it does not
write them.

### 3.2 Layout

| Word | Field | Encoding | Expected |
| ---: | --- | --- | --- |
| 9 | `build_id` | 32-bit unsigned bit pattern | stamped build id |
| 10 | `mxclracc` | int | `1` |
| 11 | `mma` | int | `3` |
| 12 | `convert` | int | `1` |
| 13 | `store` | int | `1` |
| 14 | `pack` | int | `4` |
| 15 | `bias` | int | `2` |
| 16 | `fill` | bits 0..2 mask; 8..23 fill word; 24..27 form; 3..7, 28..31 zero | mask 1/7/3/5, word `0xA5A5`, form `1` |
| 17 | `address` | 0..6 out/store; 7..13 Wlo; 14..20 Ahi; 21..31 zero; 128 B units; `0x7f` invalid | out 64, Wlo 48 (B/C) or 0, Ahi 0 |
| 18 | `selector` | 0..3 condition; 8..12 q; 4..7, 13..31 zero | condition 0..3, q 6..9 |
| 19 | `fnv_ahi` | 32-bit | `0xc6231266` |
| 20 | `fnv_alo` | 32-bit | `0x3b2d22d2` |
| 21 | `fnv_whi` | 32-bit | `0x8b88609f` |
| 22 | `fnv_wlo` | 32-bit | `0xa66af695` |
| 23 | `guard_pre` | FNV-1a 32 of the guard after init | `0xf5ebd5c5` |
| 24 | `guard_post` | FNV-1a 32 of the guard after Store #1 | equals word 23 |
| 25 | `guard_status` | 0..7 status; 8..19 first bad byte; 20..23 init method; 24..27 check method; 28..31 zero | `0x011fff00` |
| 26 | `failure` | `hmx_d8_failure_t` | `0` |
| 27 | `schema` | `0x0d8a5003` | exactly |
| 28 | `readback` | 0..1 trace; 2 readback; 3..10 HVX lock; 11..21 slots; 22..31 zero | trace 0, readback 1, HVX 0, slots 1024 |
| 29 | reserved | 0 | `0` |
| 30 | `cleanup` | first cleanup error, written by `run()` for mode 46 | `0` |
| 31 | reserved | 0 | `0` |

All 32-bit fields are transported as signed FastRPC `long` but compared as
unsigned bit patterns. Packed fields use only low bits; a Build ID or FNV with
the sign bit set cannot corrupt a packed field. The schema value is chosen
below `0x80000000` so it is also positive as a signed int.

### 3.3 Machine-readable index block

`hmx_d8_rev3_contract_test.py` parses this block and requires it to equal the
`HMX_D8_META_*` enum in `host_tests/hmx_acc_residual_rpc/hmx_d8_contract.h`:
23 entries, indices 9..31, no duplicates, no entry in 0..8.

<!-- d8-meta-layout:begin -->
9 build_id
10 mxclracc
11 mma
12 convert
13 store
14 pack
15 bias
16 fill
17 address
18 selector
19 fnv_ahi
20 fnv_alo
21 fnv_whi
22 fnv_wlo
23 guard_pre
24 guard_post
25 guard_status
26 failure
27 schema
28 readback
29 reserved_29
30 cleanup
31 reserved_31
<!-- d8-meta-layout:end -->

### 3.4 Machine-readable sample vectors

The vectors below use the sample build id `0xd8b01110` and the sample elapsed
value `4321`. Words 0..8 are a `run()`-owned sample; words 9..31 are the D8
echo. The Python audit re-derives these 32 words from the header constants and
requires the document to match; the C self-test builds the same vectors.

<!-- d8-meta-values:begin -->
A 0x484D5831 0x00000051 0x00003000 0x000010E1 0x00000000 0x00000000 0x00000000 0x00000001 0x00000001 0xD8B01110 0x00000001 0x00000003 0x00000001 0x00000001 0x00000004 0x00000002 0x01A5A501 0x00000040 0x00000600 0xC6231266 0x3B2D22D2 0x8B88609F 0xA66AF695 0xF5EBD5C5 0xF5EBD5C5 0x011FFF00 0x00000000 0x0D8A5003 0x00200004 0x00000000 0x00000000 0x00000000
B 0x484D5831 0x00000051 0x00003000 0x000010E1 0x00000000 0x00000000 0x00000000 0x00000001 0x00000001 0xD8B01110 0x00000001 0x00000003 0x00000001 0x00000001 0x00000004 0x00000002 0x01A5A507 0x00001840 0x00000701 0xC6231266 0x3B2D22D2 0x8B88609F 0xA66AF695 0xF5EBD5C5 0xF5EBD5C5 0x011FFF00 0x00000000 0x0D8A5003 0x00200004 0x00000000 0x00000000 0x00000000
C 0x484D5831 0x00000051 0x00003000 0x000010E1 0x00000000 0x00000000 0x00000000 0x00000001 0x00000001 0xD8B01110 0x00000001 0x00000003 0x00000001 0x00000001 0x00000004 0x00000002 0x01A5A503 0x00001840 0x00000802 0xC6231266 0x3B2D22D2 0x8B88609F 0xA66AF695 0xF5EBD5C5 0xF5EBD5C5 0x011FFF00 0x00000000 0x0D8A5003 0x00200004 0x00000000 0x00000000 0x00000000
D 0x484D5831 0x00000051 0x00003000 0x000010E1 0x00000000 0x00000000 0x00000000 0x00000001 0x00000001 0xD8B01110 0x00000001 0x00000003 0x00000001 0x00000001 0x00000004 0x00000002 0x01A5A505 0x00000040 0x00000903 0xC6231266 0x3B2D22D2 0x8B88609F 0xA66AF695 0xF5EBD5C5 0xF5EBD5C5 0x011FFF00 0x00000000 0x0D8A5003 0x00200004 0x00000000 0x00000000 0x00000000
<!-- d8-meta-values:end -->

## 4. Host-side gate

The host applies these checks in order and stops at the first failure. The
machine implementation is `hmx_d8_contract_check()` in
`host_tests/hmx_acc_residual_rpc/hmx_d8_contract.h`.

1. `metadataLen == 32`, `resultsLen == 1024`.
2. Words 0..8: magic, arch 81, VTCM 12288, `metadata[7] == 1`,
   `metadata[8] == 1`. Words 3..6 are ignored by D8 (they are `run()`-owned).
3. Schema word 27.
4. Build id word 9 (unsigned) equals the identity-mode echo and the stamp.
5. Selector word 18: condition `== q-6`, q echo `== q`, reserved bits zero.
6. Counts words 10..15.
7. Fill word 16: mask per condition, word `0xA5A5`, form 1, reserved bits zero.
8. Address word 17 per condition; `0x7f` is never accepted.
9. Operand FNVs words 19..22.
10. Guard words 23..25: both digests `0xf5ebd5c5`, status 0, first bad `0xfff`,
    init `1`, check `1`.
11. Failure word 26 == 0; readback word 28 (trace 0, readback 1, HVX 0,
    slots 1024); cleanup word 30 == 0; reserved words 29/31 == 0.
12. Results: 1,024 slots, each `0..65535`, compared exactly with a reference.

Any failure is `D8_CONTRACT_FAILED` with the field name; no numeric
interpretation is made. The C and Python implementations must agree on the
first failing field for the curated cases in §8.

## 5. Guard verification

| Question | Evidence | Limit |
| --- | --- | --- |
| Was the guard zero at the start of the critical path? | `guard_pre` digest + init method | the DSP computes the digest; a dishonest build can lie, the disassembly is the independent check |
| Did anything write the guard during the run? | `guard_post` diff + first bad byte | same self-report limit |
| Was the init actually 12 × 128 B? | init method field + disassembly | the field is a source-level claim |
| Did the run write outside the declared regions? | post digest + access rules | a write that restores the original bytes would be invisible; the disassembly forbids it |

The guard is initialised with zeros, not assumed to be zero. VTCM entry contents
are not relied on. The init and both hashes are outside the MMA #3 → fill →
Convert interval.

Failure classification:

| Condition | `failure` code | Result |
| --- | ---: | --- |
| init not performed / not 12 × 128 B | 6 | `D8_CONTRACT_FAILED` |
| pre digest != `0xf5ebd5c5` | 7 | `D8_CONTRACT_FAILED` |
| post digest != pre, or first bad byte set | 8 | `D8_CONTRACT_FAILED`, no numeric reading |
| post check not performed | 8 | `D8_CONTRACT_FAILED` |
| guard status inconsistent with the digests | 8 | `D8_CONTRACT_FAILED` |

## 6. Evidence independence

| Evidence | During-run measurement | Self-report | Static disassembly | Device-only |
| --- | --- | --- | --- | --- |
| counts (`mxclracc`/MMA/Convert/Store/pack/bias) | no | yes | yes, per path | no |
| Convert vs Store order | no | yes, issued list | yes | no |
| fill mask/word/address | no | yes | yes | no |
| fill form 16 × 128 B | no | yes (form field) | **yes, authoritative** | no |
| operand FNVs | yes, before MMA #1 | hash routine | position only | no |
| guard digests | yes, from memory | hash routine | init/check position | no |
| build id / schema | no | compiled constants | stamp | no |
| selector echo | no | q transport | branch use of q | no |
| `slots_written` | no | yes | store loop bounds | no |
| 1,024-cell output | **yes** | n/a | n/a | yes (outcome) |
| internal read completion | no | no | no | **the question** |

A counter or echo proves the source path executed; it does not prove the
instruction did what the constant says. The numeric outcome and the
disassembly are the independent evidence.

## 7. Disassembly audit

The stamped object must pass `docs/hmx-v81-d8-disasm-audit-spec.md` before the
run. Summary of the required checks:

1. exactly 1 `mxclracc`, 3 MMA packets, 1 Convert, 1 Store, 4 packs, 2 bias
   loads, 12 guard-init vector stores, 16/32/48 fill vector stores for A/C-D/B;
2. operand addresses `+0/+4096`, `+0/+6144`, `+2048/+4096` in order;
3. Convert → Store order, Store target `+8192`;
4. every fill after the third MMA in object order, out_pos → Wlo → Ahi;
5. no `syncht`, barrier, DMA sync, dummy `cvt`, checksum or RPC in the critical
   interval;
6. no write to the guard after the init, no write above `+12288`;
7. any address the parser cannot resolve is a **failure**, not a skip.

## 8. Reproduction record (CPU only, revision 3 proposal, 2026-11-10)

To be completed by the verification run. The keys are stable:

| Evidence | Producer | Result |
| --- | --- | --- |
| D8 order/guard model | `hmx_d8_min_sim.c` (gcc `-O0`/`-O2`, clang `-O2`) | `d8_checks=79 fail=0`, `d8_verdict=D8_CPU_READY` |
| Revision-3 issued order | `hmx_d8_min_sim --d8-sequence-rev3` | four condition signatures |
| Metadata / q / failure contract | `hmx_d8_rev3_selftest.c` | `d8_rev3_checks=104 fail=0` |
| Independent Python contract | `hmx_d8_rev3_contract_test.py` | `checks=140 fail=0`, 244 replay cases, 0 mismatches |
| Disassembly rules | `test_hmx_d8_disasm_audit.ps1` | clean 4/4, mutants 12/12 |
| Existing regressions | B3/R2 12,828 checks + 30/30 faults; C2 81 checks + 5,460 differential + 2/2 mutants; CB-FIX-1 | unchanged |
| Fast / Host | `scripts/verify.ps1` | Fast PASS (PASS=10 FAIL=0 SKIP=21); Host see the PR report |

## 9. What the contract cannot establish

1. **Internal read timing.** The device reports no MX pipeline trace; word 28
   trace bits are always 0. The numeric outcome is the only observable.
2. **A universal ordering guarantee.** A safe result is one build, one
   instruction distance, one run.
3. **The later writes.** Convert #2 → `Wlo` and the HVX recon → `Ahi` are later
   than the fills and are not measured.
4. **The actual fill lowering.** If the shipped fill is not 16 × 128 B HVX
   stores, the registered experiment does not describe it.
5. **Full-tile accuracy, performance, training integration.**
6. **Honest self-report.** Counts, masks, addresses, form and methods are
   written by the same code they describe. The disassembly and the numeric
   outcome are what make the claim falsifiable.

## 10. Change record versus revision 2

| Revision 2 | Revision 3 proposal | Why |
| --- | --- | --- |
| words 0..24, with 0/2/3/7/8 collisions | words 9..31 only | `run()` ownership; words 3/7/8 overwritten after the handler |
| build id at 0 and 24 | build id at 9; identity-mode echo at `metadata[2]` | word 0 is magic; a second handler echo adds no information |
| `q=0..3` | `q=6..9`, `condition=q-6` | the shared gate rejects `q<6` |
| short/long return is a failure | exact host allocation; DSP requires `== 1024` | `rout` length is the client maximum |
| preserve raw bytes on non-zero return | return 0 after start; word 26 reason; pre-exec stays non-zero | non-zero discards `rout` |
| guard flag with no channel | init + pre/post digests + status + first bad byte | no evidence channel in revision 2 |
| cleanup in word 22 (handler) | word 30 written by `run()` for mode 46 | `release` happens after the handler |
| Convert/Store inferred from one store event | issued-operation record | the fused event cannot show the order |
| Ahi fill offset had no presence bit | mask bit 2 + address field | 0 is both "absent" and "Ahi offset" |
