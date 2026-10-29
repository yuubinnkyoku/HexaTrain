# Candidate C — C2 Minimal Device Experiment Pre-Registration

Status: **PRE-REGISTERED, NOT AUTHORISED**, 2026-10-29, revision 1.
Scope: CPU-only pre-registration of one minimal device experiment that isolates
the **C2** dependency of the Candidate C design
(`docs/hmx-v81-cand-c-single-output-buffer.md`).

No DSP source was written, no DSP build was made, no QAIRT invocation, no ADB, no
device contact, no Mode 46 registration, no new research mode. Nothing in this
document authorises execution: §11 is the authorisation gate and it stays closed.

Every label in the project is unchanged by this document:
`B3_R2_FULL_TILE_CPU_READY`, `B3_R2_SYNC_UNPROVEN`,
`CPU_VALIDATED_PREFERRED_DESIGN` (Candidate B), Candidate C CPU-only-consistent
and device-unconfirmed, Mode 46 `NOT_RUN`, E-3C2 `UNTRIGGERED`.

---

## 0. The one question

> If software issues **HMX Store #1 → HVX vector read of the same VTCM buffer →
> HMX Store #2**, does the HVX read observe Store #1's bytes, or can Store #2
> (which targets the same address) overtake it?

C2 in the Candidate C dependency table. Everything else about Candidate C — the
arithmetic, the layout, the operand immutability, the L2 lifetime budget — is
already CPU-established and is **not** what this experiment tests.

The experiment is deliberately *not* a Candidate C trial: it has one MMA, one
operand pair, one output buffer, no operand reuse and no reconstruction on the
critical path. A single clean result here says nothing about C1, C3, C4 or C5.

---

## 1. What a result can and cannot establish

| | This experiment |
| --- | --- |
| Can **falsify** | "this exact instruction sequence is order-safe on this part, in this build" |
| Cannot establish | a general HMX/HVX ordering guarantee |
| Cannot establish | Candidate C device validity, or any full-tile accuracy claim |
| Cannot establish | a bound on the race window (one instant is sampled, not swept) |

Explicitly: **one clean run is not proof.** Only a documented guarantee could
prove the ordering; a device experiment can only fail to disprove it. A PASS is
recorded as `C2_MINIMAL_SEQUENCE_SUPPORTED`, never as `C2_PROVEN`.

---

## 2. Fixed instruction sequence (critical path)

```text
# prologue (not on the critical path)
pack A_plain  -> VTCM act  (+0,     16 vectors x 128 B)
pack W_plain  -> VTCM wgt  (+2048,  16 vectors x 128 B)
build bias set 0 (+8192): scale 0x5400 (+64), shape 0 (x),    bo 0
build bias set 1 (+8448): scale 0x5400 (+64), shape 4 (-x),   bo 0
fill X      (+4096, 2048 B) = 0xA5A5
fill guard_lo (+3968, 128 B) = 0x5A5A ;  fill guard_hi (+6144, 128 B) = 0x5A5A
fill P      (L2 +0,  2048 B) = 0xDEAD
fill P guards (L2 -128 and L2 +2048, 128 B each) = 0x5A5A

# critical path - nothing else may run between these packets
mxclracc
MMA #1 packet:  activation.hf = mxmem(act) ; weight.hf = mxmem(wgt)
Convert #1:     cvt.hf = acc(Rs = 0x0001)          # bias set 0, shape 0, retain ACC
Store #1:       mxmem(X) = cvt                     # HMX store into X (+4096)
HVX copy:       for v in 0..15: P[v] = X[v]         # 128 B vector load + store
Convert #2:     cvt.hf = acc(Rs = 0x1000)          # bias set 1, shape 4, clear+swap
Store #2:       mxmem(X) = cvt                     # HMX store into the SAME X

# readback (order fixed; both buffers are final by now)
harvest P (2048 B, L2)
harvest X (2048 B, VTCM)
harvest guard_lo, guard_hi, P guards, and the bias sets
```

Notes that are part of the registration:

- The HVX copy is **16 iterations of a 128 B vector load and a 128 B vector
  store**. No predication tricks, no widened vectors, no unaligned forms: a
  different access width is a different experiment.
- Store #1 and Store #2 target the **identical** 2,048 B range. That identity is
  the entire experiment; splitting X into two buffers deletes it.
- The readback order is fixed so a truncated run is still interpretable.

---

## 3. Fixed control values

| Item | Value | Note |
| --- | --- | --- |
| Tile | 32 x 32 x 32, single MMA | 1,024 output cells |
| Operands | `A_plain = RN16(X)`, `W_plain = RN16(W)` | the registered G8 high tiles |
| Fixture | G8 seed `0xE3C0117E5EEDF00D`, U(-2, 2), pinned anchors | frozen |
| `mxclracc` | 1 | |
| MMA packets | 1 | |
| Convert #1 | `Rs = 0x0001`, bias set 0, shape 0, `bo = 0` | retains ACC (R2) |
| Store #1 address | VTCM `+4096` | |
| Convert #2 | `Rs = 0x1000`, bias set 1, shape 4, `bo = 0` | clears ACC after |
| Store #2 address | VTCM `+4096` (same range) | |
| HVX copy | 1 loop, 16 vectors of 128 B, VTCM `+4096` -> L2 `+0` | |
| Readbacks | 2 (P then X) | |

Lane coverage comes from the frozen fixture, not from a new one: spatial lanes
`s0` (positive), `s1` (negative), `s2` (mixed / cancellation), `s3` (subnormal
edge) and `s4..s31` (realistic). The fixture's mutual-exclusivity property
(`both_nonzero = 0`) is preserved by the sign-separated convert pair, so every
output cell carries its sign in exactly one of the two buffers.

### VTCM map for this experiment (8,704 B of 12,288 B)

| Region | Offset | Size | Initial content |
| --- | ---: | ---: | --- |
| `act` | `+0` | 2,048 B | packed `A_plain` |
| `wgt` | `+2048` | 2,048 B | packed `W_plain` |
| `guard_lo` | `+3968` | 128 B | `0x5A5A` |
| **`X`** | **`+4096`** | **2,048 B** | `0xA5A5` |
| `guard_hi` | `+6144` | 128 B | `0x5A5A` |
| bias set 0 | `+8192` | 256 B | scale/shape 0/bo 0 |
| bias set 1 | `+8448` | 256 B | scale/shape 4/bo 0 |
| unused | `+8704` | 3,584 B | must stay untouched |

This is deliberately **not** the Candidate C map: one MMA needs no `Alo`/`Wlo`
and no operand reuse, so the experiment cannot be mistaken for a layout test.

---

## 4. Pre-generated expected bits (all 1,024 cells)

The expected arrays are the Plain+R2 subcase outputs already frozen by the CPU
model. They are not regenerated for the device run:

| Artifact | Cells | Canonical serialisation | SHA-256 |
| --- | ---: | --- | --- |
| Expected `P` (positive) | 1,024 | `binary16` LE, decoded cell order `s*32+o` | `82107d6e53e2e6d7d07bf007357d722fdfcb4a17c00a3582c3b514e90f76ffb8` |
| Expected `X` final (negative) | 1,024 | idem | `3fcddb24e66920f21ff2818aea0f3003f344ac006b546047e3151e6bce1a3309` |
| Optional diagnostic recon | 1,024 | idem | `0e58d8f88c8ac00d005a3a93e743fedc90bdc3cb0e163d0a23b18c118e9336bb` |

All three are asserted every run by
`scripts/run_hmx_b3_r2_sim.ps1` (`cand_c_sha256_plain_pos_bits`,
`cand_c_sha256_plain_neg_bits`, plus the frozen registration
`b3_r2_sha256_plain_expected_bits`), so a fixture change cannot pass silently.

Derivable cell anchors (from the registered recon pins plus mutual exclusivity,
no new measurement):

| Cell | Expected `P` | Expected `X` final |
| --- | ---: | ---: |
| `(s0, o0)` | `0x5400` | `0x0000` |
| `(s1, o0)` | `0x0000` | `0x5400` |
| `(s2, o3)` | `0x0000` | `0x0000` (the single non-discriminating cell) |

**Discriminating cells: 1,023 of 1,024** (`cand_c_plain_payload
discriminating_cells=1023 non_discriminating=1`). On the one cell where
`pos == neg == 0x0000`, no read-order observation is meaningful, and the
classifier excludes it by content rather than by index.

---

## 5. Why the three possible contents of `P` are mutually distinguishable

This is the property that makes a mismatch attributable instead of ambiguous, and
it is established from the CPU model's own convert law:

> A Convert/Store stores `cvt` only when the sign bit is clear, otherwise it
> stores `0x0000`. Therefore **every stored payload cell has bit 15 = 0.**

| Possible content of a `P` cell | Bit 15 | Distinguishable from a payload? |
| --- | --- | --- |
| Correct positive payload | 0 | — |
| Negative payload (post-Store-#2 bytes) | 0 | yes, on 1,023 cells (`pos != neg`) |
| `0xA5A5` (X's pre-Store-#1 sentinel) | **1** | **always** |
| `0xDEAD` (P's own sentinel, copy never ran) | **1** | **always** |

So a cell never has to be guessed: bit 15 separates "no store landed here" from
"a store landed here", and the payload identity separates "read before Store #2"
from "read after Store #2".

---

## 6. Keeping C1 and C2 apart

Both a C1 failure (Store #1 not visible to the HVX read) and a C2 failure
(Store #2 overtaking the read) corrupt `P`, and in both cases the final `X`
still holds the correct negative payload. `X` therefore **cannot** disambiguate;
only `P`'s content can, via §5.

| `P` observation | Interpretation |
| --- | --- |
| all 1,024 cells == expected positive payload | C1 and C2 both held in this run |
| ≥1 cell == `0xDEAD` | the copy did not execute → contract failure |
| ≥1 cell == `0xA5A5` | `P` saw X before Store #1 → **C1** signature |
| ≥1 discriminating cell == the negative payload | `P` saw X after Store #2 → **C2** signature |
| C1 and C2 signatures both present | `C2_CAUSE_INCONCLUSIVE` |
| any other content | `C2_CAUSE_INCONCLUSIVE` |

### 6.1 Controls, and what Mode 45 already covers

No second device run is registered (budget is one attempt), so the controls are
CPU-side or already-registered evidence:

| Control | Where it comes from | What it buys |
| --- | --- | --- |
| C1 direction (MX store → later HVX read of the same bytes) | **Mode 45**, `out_pos`/`out_neg` read by the `vsub.hf` loop, S1/S2 matched | C1 is `IMPLEMENTATION_SUPPORTED`; this experiment does not re-establish it, and a C1-looking result is labelled separately rather than reported as a C2 result |
| Store → later scalar byte read of the same bytes | Mode 45 FNV read-back loop | shows the easy direction works, nothing about C2 |
| `pos != neg` on 1,023 cells | CPU model, registered digest | makes a mixed vector detectable |
| All payloads have bit 15 = 0 | convert clamp law, CPU model | makes the sentinel signature unambiguous |
| Guard regions | this document | detects a store or copy that lands outside X / P |
| ACC retention across the two converts | Mode 45 (R2, 11 slots) | Convert #2 sees the same ACC, so a numeric failure is not an R2 failure |

What no control covers: the HVX-read → MX-store ordering itself. That is the
experiment.

---

## 7. Critical-path discipline (do not accidentally fix the race)

The following are **forbidden between Store #1 and Store #2**, because each one
adds an instruction that could serialise the two agents and turn a real hazard
into a passing run:

1. any diagnostic checksum, FNV/CRC loop, or byte-wise read-back of `X` or `P`;
2. `syncht`, `barrier` packet attribute, DMA sync intrinsics, or any fence;
3. a dummy `cvt.hf = acc` used as a "wait" (it consumes ACC, not an operand read);
4. host RPCs, logging calls, or any call that could clobber `"memory"` and
   thereby emit code;
5. a third Convert/Store, or any re-use of X for another purpose.

The `asm volatile("" ::: "memory")` **compiler** barrier is allowed and is
expected: it emits nothing, and the sync audit already established that the
shipped object contains zero synchronisation instructions
(`docs/hmx-v81-b3-r2-sync-static-audit.md` §4.1).

Note the trap this avoids: Mode 45's registered sequence reads `out_pos` with a
scalar FNV loop immediately after the store. That pattern is a diagnostic read on
the critical path and must **not** be copied into this experiment.

---

## 8. Registered facts and contract gate

Collected by the host contract, checked before any interpretation:

| Fact | Expected |
| --- | --- |
| `mxclracc` count | 1 |
| MMA packets | 1 |
| Convert count | 2 |
| Store count | 2 |
| HVX copy loops / vectors | 1 / 16 |
| Readbacks | 2 |
| Bias sets written | 2 |
| `Rs` history | `0x0001`, `0x1000` |
| Store addresses | `+4096`, `+4096` (identical) |
| Guard regions | unchanged (`0x5A5A`) |
| Unused VTCM `+8704..12287` | unchanged |
| Build ID | matches the stamped build under test |
| RPC status / close status | 0 / 0 |

Any deviation fails closed as `C2_PREREG_CONTRACT_FAILED` and **no numeric
interpretation is made** from that run.

---

## 9. Decision procedure (evaluated in this order)

```text
[ device run ]
      |
      +--> RPC / Build-ID / counts / Rs / guards / unused-VTCM mismatch?
      |        yes -> C2_PREREG_CONTRACT_FAILED            (stop, no interpretation)
      |        no
      v
   P all 0xDEAD?  --yes--> C2_COPY_NOT_EXECUTED            (stop)
      | no
      v
   X != expected negative payload? --yes--> C2_STORE2_PATH_BROKEN (not a C1/C2 result)
      | no
      v
   P has a C1 signature (0xA5A5) AND a C2 signature (negative payload)?
      | yes -> C2_CAUSE_INCONCLUSIVE
      | no
      v
   P has a C1 signature only? --yes--> C1_STORE_VISIBILITY_SUSPECT
      | no
      v
   P has a C2 signature only? --yes--> C2_ORDERING_VIOLATED
      | no
      v
   P == expected positive payload on all 1,024 cells?
      | yes -> C2_MINIMAL_SEQUENCE_SUPPORTED
      | no  -> C2_CAUSE_INCONCLUSIVE
```

Rules attached to the procedure:

- On `C2_ORDERING_VIOLATED`, `C2_CAUSE_INCONCLUSIVE`,
  `C1_STORE_VISIBILITY_SUSPECT` or `C2_STORE2_PATH_BROKEN`: preserve the raw
  `P`/`X`/guard bytes, the full cell dump, the digests of what was actually
  observed, the Build ID and the RPC transcript, then **stop**. No retry, no
  "run it again to see", no clamping or resetting of anything.
- A mismatch caused by hardware FTZ/DAZ on the two subnormal-edge cells
  (`s3,o30`, `s3,o31`) is classified as a subnormal observation, not as a C1/C2
  signature, exactly as the full-tile pre-registration already provides
  (`B3_R2_HVX_SUB_FTZ_OBSERVED`).
- No numeric tolerance is registered: the comparison is raw-bit exact.

---

## 10. What each outcome means for Candidate C

| Outcome | Consequence for Candidate C |
| --- | --- |
| `C2_MINIMAL_SEQUENCE_SUPPORTED` | C2 is *not disproved* in this build and this interleaving. Candidate C stays a candidate; C1/C3/C4/C5 remain separate contracts; a full-tile device trial is still unauthorised |
| `C2_ORDERING_VIOLATED` | the single-shared-buffer premise is unsafe as written. Candidate B (`CPU_VALIDATED_PREFERRED_DESIGN`) becomes the only surviving CPU-validated design, and a synchronization mechanism would have to be found before Candidate C is revisited |
| `C1_STORE_VISIBILITY_SUSPECT` | contradicts Mode 45; treat as a build/measurement problem first, not as new physics |
| `C2_CAUSE_INCONCLUSIVE` | nothing is learned about C2; the classifier, the build or the fixture must be fixed before re-registering |

---

## 11. Execution budget and authorisation gate

| Item | Value |
| --- | --- |
| Builds | 1 clean stamped DSP build |
| Device sessions | 1 |
| Attempts | 1 |
| Retries | 0 |
| Timing / performance measurement | none |
| Profile | device-run would be **Tier 2** (`BACKGROUND_CORRECTNESS`) per `docs/agent/device-test-tiers.md` |
| Authorisation | **NOT GRANTED** — requires explicit user approval of this document |

Even under Tier 2's conditions (single resolved online endpoint, pinned QAIRT
root and Build ID, APK audit PASS, `focus_takeover_count=0`, no root / SELinux /
system changes), execution requires the user to approve *this* pre-registration
first. Nothing here may be run on the strength of the Candidate C CPU result.

Out of scope and forbidden regardless: DSP source changes, DSP builds, ADB
contact, device connection, performance measurement, `push`, `main` changes and
PR merges.

---

## 12. Reproducing the CPU-side registrations

```powershell
# CPU only; no SDK, no device, no ADB.
.\scripts\run_hmx_b3_r2_sim.ps1
# cand_c_layout vtcm_used=10752 ... margin=1536 ...
# cand_c_l2_lifetimes peak_raw=10240 peak_fold=6144 ...
# cand_c_bits ... all six arrays 0/1024 diff
# cand_c_sha256_plain_pos_bits=82107d6e...
# cand_c_sha256_plain_neg_bits=3fcddb24...
# cand_c_plain_payload discriminating_cells=1023 non_discriminating=1
# b3_r2_fault_injection=PASS detected=22/22
# b3_r2_checks=12704 fail=0
```

Registered artifacts are `build/`-only and gitignored. The digests above are the
registration; the files are not.

---

## 13. Unresolved items carried forward

1. **C2** — this document registers how to ask the question, not the answer.
2. **C1 / C3** — `IMPLEMENTATION_SUPPORTED` from Mode 45 only; not re-tested here.
3. **C4** — in-place per-vector load-before-store; same-thread register
   dependency, no device evidence. A separate diagnostic (the optional
   reconstruction) if it is ever needed: it must run **after** the C2 readback,
   never on the C2 critical path.
4. **C5** — Plain-to-B3 evidence evacuation; CPU-verified ordering, device
   untested.
5. **CB-FIX-1** — Candidate B's sentinel/ordinal inconsistency is still open and
   still unfixed; it does not affect the 1,024-cell bits.
6. **D8a / D8b** — unchanged diagnostic hypotheses, `DEVICE_ONLY`.
7. **Mode 46** — `NOT_RUN`.
