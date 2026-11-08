# Candidate B — D8 Device Experiment: Evidence Contract and Audit Record

Status: **DESIGN AND AUDIT RECORD, CPU-ONLY**, 2026-11-08. Companion to
`docs/hmx-v81-cand-b-d8-device-prereg.md` (revision 1). This document authorises
nothing: no DSP source, no DSP build, no QAIRT invocation, no ADB, no device
contact, no Mode 46.

It exists so that the single registered D8 attempt, if the user ever approves
it, has an unambiguous observation contract, a complete digest inventory and a
pre-written disassembly audit. It does not replace the pre-registration; where
the two differ, the pre-registration wins.

---

## 1. Digest and value inventory

| # | Report key | Array | Bytes | Serialisation | SHA-256 / value | Verified by |
| --: | --- | --- | ---: | --- | --- | --- |
| 1 | `d8_sha256_operands_deploy` | ahi, alo, whi, wlo, a_plain, w_plain | 12,288 | u16 LE | `e5193d99…1f0d47a3` | C model + Python (dump re-hash) |
| 2 | `d8_fnv_ahi` | packed Ahi tile | 2,048 | FNV-1a 32 over LE bytes | `c6231266` | C model + Python |
| 3 | `d8_fnv_alo` | packed Alo tile | 2,048 | idem | `3b2d22d2` | C model + Python |
| 4 | `d8_fnv_whi` | packed Whi tile | 2,048 | idem | `8b88609f` | C model + Python |
| 5 | `d8_fnv_wlo` | packed Wlo tile | 2,048 | idem | `a66af695` | C model + Python |
| 6 | `d8_sha256_safe` | reference output | 2,048 | u16 LE, cell order `s*32+o` | `c9abf30e…ea20203d` | C model + JSON-free Python re-derivation |
| 7 | `d8_sha256_wlo` | Wlo-corrupt reference | 2,048 | idem | `f0b3b4aa…9881f6fb` | idem |
| 8 | `d8_sha256_ahi` | Ahi-corrupt reference | 2,048 | idem | `b6328e88…e0760fcc` | idem |
| 9 | `d8_sha256_both` | both-corrupt reference | 2,048 | idem | `5285b463…de32535d` | idem |

No existing registered digest or expected value is changed. Rows 6–9 are new
registrations for this experiment; row 1 is an existing registration reused
unchanged.

---

## 2. DSP-side evidence contract (design only)

If the attempt is ever approved, the single stamped build must produce and the
host must preserve all of the following.

1. **Exactly 3 MMA packets**, after 1 `mxclracc`; no other MMA in the path.
2. **Exactly 4 operand packs** into `+0`, `+2048`, `+4096`, `+6144` and **2
   bias-set writes** into `+10240`, `+10496`, before `mxclracc`.
3. **One Convert** (`Rs = 0x0001`, bias set 0, shape 0, scale 64) and **one HMX
   Store** into `+8192`. No Convert #2, no HVX reconstruction.
4. **One out_pos sentinel fill** at `+8192` (2,048 B, `0xA5A5`), and the
   condition's operand fills at `+6144` (B/C) and/or `+0` (B/D), each 2,048 B
   of `0xA5A5` and each **16 × 128 B HVX vector stores**.
5. **Fill order is after MMA #3** and before Convert #1. No fill of `+0` or
   `+6144` may execute before the third MMA packet. No branch may skip it for a
   condition that requires it.
6. **No synchronization construct** anywhere in the critical interval:
   no `syncht`, `barrier`, DMA sync intrinsic, dummy `cvt.hf = acc` wait, host
   RPC, checksum loop, scalar byte read of `+0`/`+6144`/`+8192`, or any other
   access that could serialise the reuse against the MMA read. The
   `asm volatile("" ::: "memory")` compiler barrier is allowed and must emit
   nothing.
7. **No write outside the declared regions** during the run; the 1,536 B tail
   of the 12,288 B window stays `0x00`.
8. **Full 2,048 B of `out_pos`** returned to the host as zero-extended u16 in
   decoded cell order `s*32+o`, u16 LE.
9. **Metadata echo** of the values in §3.
10. **Build ID** of the stamped build echoed twice (Mode 45 convention).
11. **Architecture echo** from the identity block.
12. **Cleanup status** 0 for VTCM/L2/rpcmem.
13. **Readback order**: out array, then metadata, then the failure code echo.

Nothing here is implemented. The contract is the checklist the future
implementation and its audit are held to.

---

## 3. Host-side evidence and the `sequence<long>` ABI

### 3.1 Shape

The existing IDL needs **no signature change**:
`run(rout sequence<long> results, rout sequence<long> metadata, in long q, in long mode)`.

- **`results`: 1,024 slots = 4,096 B.** Slot `s*32+o` is one zero-extended u16
  output cell. The host masks `0xFFFF` before comparing.
- **`metadata`: 32 slots = 128 B.**
- **`q`**: proposed condition selector (0=A, 1=B, 2=C, 3=D). This is a design
  proposal, not an implementation. If the future build cannot carry it, the
  experiment must be re-registered.
- **`mode`**: the future registered trigger mode; Mode 46 remains `NOT_RUN`.

Total 4,224 B. Actual allocation, mapping and transport limits are unverified
until an approved implementation is audited.

### 3.2 Metadata words

| Word | Content | Expected |
| ---: | --- | --- |
| 0 | Build ID echo | stamped build |
| 1 | DSP architecture | identity block value |
| 2 | mode echo | registered mode |
| 3 | condition (`q` echo) | 0/1/2/3 as requested |
| 4 | `mxclracc` count | 1 |
| 5 | MMA packets | 3 |
| 6 | Convert count | 1 |
| 7 | HMX Store count | 1 |
| 8 | operand packs | 4 |
| 9 | bias sets written | 2 |
| 10 | fill mask (bit0 out, bit1 Wlo, bit2 Ahi) | 1 / 3 / 5 / 7 for A/B/C/D |
| 11 | fill word | `0x0000A5A5` |
| 12 | store target offset | 8192 |
| 13 | out_pos fill offset | 8192 |
| 14 | Wlo fill offset (0 if none) | 6144 or 0 |
| 15 | Ahi fill offset (informational) | 0; presence is word 10 bit 2 |
| 16 | trace status | 0 = not provided (device never reports internal read completion) |
| 17 | operand FNV Ahi | `0xc6231266` |
| 18 | operand FNV Alo | `0x3b2d22d2` |
| 19 | operand FNV Whi | `0x8b88609f` |
| 20 | operand FNV Wlo | `0xa66af695` |
| 21 | readbacks completed | 1 |
| 22 | cleanup status | 0 |
| 23 | first-failure reason code | 0 = none |
| 24 | Build ID second echo | same as word 0 |
| 25–31 | reserved | 0 |

Word 15 cannot distinguish "Ahi fill at offset 0" from "no Ahi fill"; word 10's
bit 2 is the presence flag, and the disassembly audit confirms the address.

### 3.3 Length and identity requirements

- The host asserts `resultsLen == 1024` and `metadataLen == 32` **before**
  reading any slot. A short or long return is `D8_CONTRACT_FAILED`.
- The condition echo must equal the requested condition.
- The Fill mask must equal: A→`1`, B→`7`, C→`5`, D→`3`.
- The operand FNVs and Build ID must match the registration.
- A mismatch stops the run; it is never "close enough".

### 3.4 Host-side evidence per run (what the report must contain)

- the 1,024-cell raw output and its SHA-256;
- all 32 metadata words;
- the requested condition and the echoed condition;
- the Build ID and architecture;
- the four operand FNVs;
- the store/fill offsets and fill mask;
- RPC status, close status, cleanup status;
- the disassembly hash of the stamped object;
- the first contract-failure reason verbatim;
- on failure, the raw bytes preserved before `rpcmem_free`.

---

## 4. Classification

The terminal classes and their order are defined in the pre-registration §6.
This contract only fixes the evidence semantics:

| Class | Evidence required | What it is not |
| --- | --- | --- |
| `D8_CONTRACT_FAILED` | any §3.3 / §2 count or identity failure | not a synchronization result |
| `D8_CONTROL_SAFE` | condition A, output exactly equal to `safe` | not D8 evidence |
| `D8_CONTROL_ANOMALY` | condition A, any other output | not a D8 violation |
| `D8_NO_VIOLATION_OBSERVED` | B/C/D, output exactly equal to `safe`, contract clean | not `D8_PROVEN`; one build, one run, one distance |
| `D8_VIOLATION_WLO_SUSPECT` | C (or B), output exactly equal to `wlo`, contract clean | not proof of a mechanism |
| `D8_VIOLATION_AHI_SUSPECT` | D (or B), output exactly equal to `ahi` | idem |
| `D8_VIOLATION_BOTH_SUSPECT` | B, output exactly equal to `both` | idem |
| `D8_INCONCLUSIVE_NUMERIC` | output matches none | partial corruption, input/convert/output cause, or a fixture problem |
| `D8_INCONCLUSIVE_TRACE` | invalid access trace (CPU-side only) | not usable on device |

Exact match means all 1,024 cells, no tolerance. The four reference arrays are
pairwise distinct (543/777/853/726/852/640 differing cells), so a match is
specific to one outcome.

**Device label contract.** The CPU model's classes are printed with a
`(simulated)` suffix and never use the device success/failure framing. The CPU
verdict `D8_CPU_READY` / `D8_CPU_PREREG_READY` is a preparation label. No amount
of CPU testing can produce `D8_NO_VIOLATION_OBSERVED` as device evidence.

---

## 5. What the contract cannot establish

1. **Internal read timing.** The device reports no MX pipeline trace; word 16 is
   always 0. The numeric outcome is the only observable.
2. **A universal ordering guarantee.** A safe result is one build, one
   instruction distance, one run. It does not sweep time, voltage, frequency or
   other interleavings.
3. **The later writes.** `Convert #2 → Wlo` and `HVX recon → Ahi` are later than
   the fills and are not measured by this experiment.
4. **The actual Candidate B fill lowering.** If the shipped fill is not 16 ×
   128 B HVX stores, the registered experiment does not describe it.
5. **Full-tile accuracy, performance, training integration.**

---

## 6. Disassembly audit plan

The stamped DSP binary is audited **statically before** the run:

1. exactly 3 MMA constructs and no other MMA in the condition paths;
2. exactly 1 Convert and 1 HMX Store, both targeting `+8192`;
3. the three MMA packets reference `+0/+4096`, `+0/+6144`, `+2048/+4096` in
   that order;
4. the out fill and the condition fills are 16 × 128 B vector stores each, at
   `+8192`, `+6144`, `+0`; no scalar or widened forms;
5. **every fill is after the third MMA packet in object order**, and no branch
   reaches a fill before it; the four condition paths have the registered masks
   (A: out only; B: out+Wlo+Ahi; C: out+Wlo; D: out+Ahi);
6. no `syncht`, barrier packet attribute, DMA-sync intrinsic, dummy
   `cvt.hf = acc`, checksum, scalar byte read of the operand/out regions, or
   host RPC between the third MMA and the fills, or between the fills and
   Convert #1;
7. no helper call in the critical interval; if one exists, audit the callee;
8. no host-side reconstruction, reordering or correction;
9. no write outside the declared regions (the 1,536 B tail stays `0x00`);
10. the operand pack order and the four packed-tile byte ranges match the
    registration (the FNV echo is a runtime cross-check, not a substitute).

Two honesty rules bound the audit:

- **Source is not machine code.** The audit reads the stamped object, not the C.
- **Static structure is not execution order.** A clean disassembly shows the
  sequence is *present* and no forbidden construct exists; it cannot prove how
  the hardware completes the read. The runtime observation remains the
  experiment.

---

## 7. Build identity and evidence preservation

- One stamped build; Build ID echoed twice; the disassembly hash recorded.
- The four operand FNVs and the Build ID are checked before the numeric
  comparison.
- The raw output, metadata, RPC transcript and disassembly hash are preserved
  before `rpcmem_free`.
- On `D8_VIOLATION_*_SUSPECT`, `D8_CONTROL_ANOMALY` or `D8_INCONCLUSIVE_*`:
  preserve and stop. No retry, no re-run, no clamping.
- Raw device logs and local paths are not committed to the repository.

---

## 8. Reproduction record (host CPU only, 2026-11-08)

| Evidence | Producer | Result |
| --- | --- | --- |
| C model + controls | gcc 15.2.0 `-O0`/`-O2`, clang 22.1.8 `-O2`, `-std=gnu11 -Wall -Wextra -Werror -ffp-contract=off` | `d8_checks=20 fail=0`, `d8_verdict=D8_CPU_READY`, four references and six diffs printed |
| Independent Python | `hmx_d8_independent_test.py` | `checks=20 fail=0`, `D8_INDEPENDENT_AUDIT_PASS` |
| Differential | C `--d8-replay` vs Python | `cases=119`, 0 mismatches, comparator mutation detected |
| Mutation regression | `run_hmx_d8_mutation_check.ps1` | 3/3 mutants detected |
| Existing regressions | B3/R2 + C2 gates | unchanged (see the PR report) |
| Fast / Host | `scripts/verify.ps1 -Profile Fast` / `-Profile Host` | Fast PASS (PASS=10 FAIL=0 SKIP=21); Host PASS (exit 0, `run_host_tests=PASS`) |

The CPU references and the safe → shipped-B3 cross-check are the only numerical
claims here. None of this is device evidence.
