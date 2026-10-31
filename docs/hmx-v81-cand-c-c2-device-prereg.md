# Candidate C — C2 Minimal Device Experiment Pre-Registration

Status: **PRE-REGISTERED, NOT AUTHORISED**, 2026-10-30, **revision 2**.
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

## Revision history

| Revision | Commit | State | What changed |
| --- | --- | --- | --- |
| 1 | `b22836e` | **superseded, never executed** | first registration; VTCM map was wrong (§3.1) and the decision tree could not separate a zeroing law from a C2 violation (§6) |
| 2 | this document | current | guard placement corrected, L2 arena given a checkable allocation, `0xDEAD` defect fixed, special-value classification ordered before any signature, attribution rule tightened, prohibition scope split |

Revision 1 is preserved in git history at `b22836e`; it must not be executed and
it is not amended away, so the defect and its fix stay auditable.

**Revision 2 amendments of 2026-10-30 (classification audit, still
pre-execution).** The audit pinned three things that revision 2 had left
implicit, none of which changes the map, the fixture, the budget or any
expected value:

1. the decision order of the two terminal contract classes is `0xDEAD`
   residue **before** the `X` interpretation (§9), pinned by the
   `c2_control_x_broken_plus_dead` control that constructs both at once;
2. the CPU model and the device run use different success labels
   (§5.1): the CPU model's nominal outcome is `C2_CPU_NOMINAL_MATCH`, and
   `C2_MINIMAL_SEQUENCE_SUPPORTED` is reserved for a device run — no amount of
   CPU-only testing can promote anything to it;
3. the L2 arena's base alignment and the untouched-byte arithmetic of §8 are
   now stated exactly (128 B base alignment; 5,376 untouched bytes, not the
   2,560 B revision 2 first wrote, which mis-summed the two in-extent gaps and
   the tail).

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
pack A_plain  -> VTCM act     (+0,    16 vectors x 128 B)
pack W_plain  -> VTCM wgt     (+2048, 16 vectors x 128 B)
build bias set 0 (+8448): scale 0x5400 (+64), shape 0 (x),  bo 0
build bias set 1 (+8704): scale 0x5400 (+64), shape 4 (-x), bo 0
fill X        (+6144, 2048 B) = 0xA5A5
fill guard_lo (+6016,  128 B) = 0x5A5A ;  fill guard_hi (+8192, 128 B) = 0x5A5A
fill P        (L2 +384, 2048 B) = 0xDEAD
fill P guards (L2 +256 and L2 +2432, 128 B each) = 0x5A5A
zero the untouched bytes (+4096..+6015, +8320..+8447, +8960..+12287 = 5,376 B)
                          = 0x00   # makes the §8 untouched-bytes row checkable

# critical path - nothing else may run between these packets
mxclracc
MMA #1 packet:  activation.hf = mxmem(act) ; weight.hf = mxmem(wgt)
Convert #1:     cvt.hf = acc(Rs = 0x0001)          # bias set 0, shape 0, retain ACC
Store #1:       mxmem(X) = cvt                     # HMX store into X (+6144)
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
- The prologue order is part of the registration: operands are packed **before**
  the guard fills, because that ordering is what makes a guard on top of an
  operand fatal and therefore detectable (§3.1).

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
| Store #1 address | VTCM `+6144` | |
| Convert #2 | `Rs = 0x1000`, bias set 1, shape 4, `bo = 0` | clears ACC after |
| Store #2 address | VTCM `+6144` (same range) | |
| HVX copy | 1 loop, 16 vectors of 128 B, VTCM `+6144` -> L2 `+384` | |
| Readbacks | 2 (P then X) | |

Lane coverage comes from the frozen fixture, not from a new one: spatial lanes
`s0` (positive), `s1` (negative), `s2` (mixed / cancellation), `s3` (subnormal
edge) and `s4..s31` (realistic). The fixture's mutual-exclusivity property
(`both_nonzero = 0`) is preserved by the sign-separated convert pair, so every
output cell carries its sign in exactly one of the two buffers.

### 3.1 VTCM map, revision 2 (8,960 B of 12,288 B)

| Region | Offset | Size | Initial content |
| --- | ---: | ---: | --- |
| `act` | `+0` | 2,048 B | packed `A_plain` |
| `wgt` | `+2048` | 2,048 B | packed `W_plain` |
| *(gap)* | `+4096` | 1,920 B | must stay untouched |
| `guard_lo` | `+6016` | 128 B | `0x5A5A` |
| **`X`** | **`+6144`** | **2,048 B** | `0xA5A5` |
| `guard_hi` | `+8192` | 128 B | `0x5A5A` |
| *(gap)* | `+8320` | 128 B | must stay untouched |
| bias set 0 | `+8448` | 256 B | scale / shape 0 / bo 0 |
| bias set 1 | `+8704` | 256 B | scale / shape 4 / bo 0 |
| unused | `+8960` | 3,328 B | must stay untouched |

Both guards are adjacent to `X` and clear of every operand. `X` stays 2 KiB
aligned and the bias sets stay 256 B aligned.

**Why revision 1 was rejected (this is the defect revision 2 fixes).**
Revision 1 put `guard_lo` at `+3968` and `X` at `+4096`. `+3968..+4095` is the
last 128 B of the packed weight tile — crouton input-channel 30/31 of every
output channel — so the registered prologue filled `0x5A5A` over live weights
**before the MMA was issued**. The G8 fixture uses those channels on ordinary
cells, so the run would have corrupted its own input and no C1/C2 conclusion
would have been possible. The map is a defect of the *registration*, not of
Candidate C.

The CPU model now carries both maps and a pairwise overlap checker. Revision 2
reports 0 overlaps; revision 1 reports exactly 1 overlapping pair, `wgt`/
`guard_lo`, over 128 B (§12, `c2_rev1_*`). Revision 1's offsets remain in
`hmx_b3_r2.h` only so that regression keeps running.

This is still deliberately **not** the Candidate C map: one MMA needs no
`Alo`/`Wlo` and no operand reuse, so the experiment cannot be mistaken for a
layout test.

### 3.2 L2 arena for `P` (2,560 B)

| Region | Offset | Size |
| --- | ---: | ---: |
| *(headroom)* | `+0` | 256 B |
| `P` guard low | `+256` | 128 B |
| **`P`** | **`+384`** | **2,048 B** |
| `P` guard high | `+2432` | 128 B |

Revision 1 placed `P` at L2 `+0` with its lower guard at L2 `-128`, which is not
an allocation that can be bounds-checked. Revision 2 keeps both guards inside a
positive-size arena: the guard offsets, the `P` offset and the arena size are all
non-negative and are asserted to tile the arena exactly.

Allocation and pointer rules (added by the 2026-10-30 audit):

- The arena **base must be 128 B aligned**. `P` (`+384`) and both guards
  (`+256`, `+2432`) are 128 B multiples, so with an aligned base every 128 B
  vector access stays aligned; the 16 copy vectors tile `P` exactly and the
  first and last vector stores land inside `P`, never in the guards. The CPU
  model asserts all of this (`c2_l2_*`).
- Device pointers into the arena are formed only as `base + offset` with
  `0 <= offset < 2560`. Negative offsets and pointers outside the reservation
  are contract violations (revision 1's `L2 -128` guard is the example of what
  is now structurally impossible).

---

## 4. Pre-generated expected bits (all 1,024 cells)

The expected arrays are the Plain+R2 subcase outputs already frozen by the CPU
model. They are not regenerated for the device run, and revision 2 does not
change them: the arithmetic is identical, only the addresses moved.

| Artifact | Cells | Canonical serialisation | SHA-256 |
| --- | ---: | --- | --- |
| Expected `P` (positive) | 1,024 | `binary16` LE, decoded cell order `s*32+o` | `82107d6e53e2e6d7d07bf007357d722fdfcb4a17c00a3582c3b514e90f76ffb8` |
| Expected `X` final (negative) | 1,024 | idem | `3fcddb24e66920f21ff2818aea0f3003f344ac006b546047e3151e6bce1a3309` |
| Optional diagnostic recon | 1,024 | idem | `0e58d8f88c8ac00d005a3a93e743fedc90bdc3cb0e163d0a23b18c118e9336bb` |

All three are asserted every run by `scripts/run_hmx_b3_r2_sim.ps1` (the third
row **is** one of the six frozen streaming digests, the Plain recon
`sha_plain_expected_bits`, so the run asserts the two Candidate C arrays plus
the six streaming digests — 8 named digests, not 9; see the digest inventory in
`docs/hmx-v81-cand-c-c2-device-evidence-contract.md`), so a fixture change
cannot pass silently. The C2 model recomputes the first two from its own bits
and compares them with these pinned values (`c2_nominal_sha_p`,
`c2_nominal_sha_x`); those two checks are recomputations of rows 1–2, not
additional independent digests.

Derivable cell anchors (from the registered recon pins plus mutual exclusivity,
no new measurement):

| Cell | Expected `P` | Expected `X` final |
| --- | ---: | ---: |
| `(s0, o0)` | `0x5400` | `0x0000` |
| `(s1, o0)` | `0x0000` | `0x5400` |
| `(s2, o3)` | `0x0000` | `0x0000` (the single non-discriminating cell) |

### 4.1 Payload census (measured, revision 2)

This is the population the classifier has to work with, and it is the reason
§6 is more careful than revision 1 was:

| Cell population | Count | Meaning for attribution |
| --- | ---: | --- |
| discriminating (`pos != neg`) | 1,023 | only these can carry read-order information |
| non-discriminating (`pos == neg == 0x0000`) | 1 | no information either way |
| negative payload is a **non-zero** pattern | 475 | a match here is **unambiguous** C2 evidence (§5) |
| negative payload is `0x0000`, positive payload normal | 546 | a zero match here is **weak** evidence only |
| positive payload subnormal (negative payload `0x0000`) | 2 | a zero here is **ambiguous**: FTZ or C2 |

`c2_payload_census cells=1024 discriminating=1023 nonzero_neg=475
zero_neg_normal_pos=546 subnormal_pos=2`

---

## 5. What makes a cell attributable

The property that makes an observation attributable comes from the CPU model's
own convert law:

> A Convert/Store stores `cvt` only when the sign bit is clear, otherwise it
> stores `0x0000`. Therefore **every stored payload cell has bit 15 = 0.**

| Possible content of a `P` cell | Bit 15 | Distinguishable? |
| --- | --- | --- |
| Correct positive payload | 0 | — |
| Negative payload, non-zero pattern | 0 | yes: 475 cells carry a specific pattern |
| `0x0000` (agrees with a zero negative payload) | 0 | **no** — also what any zeroing law produces |
| `0xA5A5` (X's pre-Store-#1 sentinel) | **1** | always |
| `0xDEAD` (P's own sentinel, copy never ran) | **1** | always |

So bit 15 separates "no store landed here" from "a store landed here", and a
**non-zero** pattern separates "read before Store #2" from "read after Store
#2". A zero does not: 546 cells have a zero negative payload, so on all of them
`P == 0x0000` is simultaneously "read after Store #2" and "something zeroed the
payload". Revision 1 treated those as a C2 signature; revision 2 does not.

### 5.1 CPU and device success labels (audit 2026-10-30)

`C2_MINIMAL_SEQUENCE_SUPPORTED` is a **device** outcome. It is reserved for a
run that executed the registered sequence on hardware and passed every §8
contract row. The CPU model's nominal outcome is labelled
`C2_CPU_NOMINAL_MATCH`, and every CPU-side evaluation of a device label
(classifier controls, fault injection) is displayed with an explicit
`simulated` / `fault_injected` tag. The CPU model asserts both directions of
this contract every run (`c2_cpu_nominal_label`,
`c2_cpu_label_never_device_label`): no amount of CPU-only testing can promote
any result to `C2_MINIMAL_SEQUENCE_SUPPORTED`.

---

## 6. Keeping C1 and C2 apart

Both a C1 failure (Store #1 not visible to the HVX read) and a C2 failure
(Store #2 overtaking the read) corrupt `P`, and in both cases the final `X`
still holds the correct negative payload. `X` therefore **cannot** disambiguate;
only `P`'s content can, via §5.

A cell is classified in this order, first match wins:

| Order | Condition on a `P` cell | Class | Weight |
| ---: | --- | --- | --- |
| 1 | `== 0xDEAD` | copy did not execute | fatal |
| 2 | `== 0xA5A5` | C1 signature (saw X before Store #1) | strong |
| 3 | `== expected positive payload` | payload match | — |
| 4 | `== 0x0000`, expected positive non-zero, negative `0x0000`, positive **subnormal** | ambiguous (FTZ or C2) | none |
| 5 | `== 0x0000`, expected positive non-zero, negative `0x0000`, positive **normal** | zero agreement | weak |
| 6 | `== 0x0000`, expected positive non-zero, negative non-zero | payload loss | none |
| 7 | `== expected negative payload` and it is **non-zero** | C2 signature | **strong** |
| 8 | anything else | unexplained | none |

Note what row 4 does and does not claim: `vsub.hf` is not in this experiment, so
a zero observed where a subnormal payload was due **is not** evidence of
`B3_R2_HVX_SUB_FTZ_OBSERVED`. HMX Convert and the HVX load/store path are both
candidates and this experiment cannot separate them, so it is recorded as
`C2_SUBNORMAL_OR_ZERO_LOSS` and stops. Likewise, rows 6 and 8 are recorded, not
interpreted.

### 6.1 Controls, and what Mode 45 already covers

No second device run is registered (budget is one attempt), so the controls are
CPU-side or already-registered evidence:

| Control | Where it comes from | What it buys |
| --- | --- | --- |
| C1 direction (MX store → later HVX read of the same bytes) | **Mode 45**, `out_pos`/`out_neg` read by the `vsub.hf` loop, S1/S2 matched | C1 is `IMPLEMENTATION_SUPPORTED`; this experiment does not re-establish it, and a C1-looking result is labelled separately rather than reported as a C2 result |
| Store → later scalar byte read of the same bytes | Mode 45 FNV read-back loop | shows the easy direction works, nothing about C2 |
| `pos != neg` on 1,023 cells | CPU model, registered digest | makes a mixed vector detectable |
| 475 cells carry a non-zero negative pattern | CPU model, measured (§4.1) | gives the classifier unambiguous C2 evidence |
| All payloads have bit 15 = 0 | convert clamp law, CPU model | makes the sentinel signature unambiguous |
| Guard regions and the two gaps | this document | detects a store or copy that lands outside X / P |
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
| Region overlap count (declared map) | 0 |
| Guard fill vs. packed weight tail | weight tail unchanged (128 B) |
| `mxclracc` count | 1 |
| MMA packets | 1 |
| Convert count | 2 |
| Store count | 2 |
| HVX copy loops / vectors | 1 / 16 |
| Readbacks | 2 |
| Bias sets written | 2 |
| `Rs` history | `0x0001`, `0x1000` |
| Store addresses | `+6144`, `+6144` (identical) |
| Guard regions | unchanged (`0x5A5A`) |
| P guards (L2 `+256`, `+2432`) | unchanged (`0x5A5A`) |
| Bytes outside every declared region | `0x00` (5,376 B: the 1,920 B and 128 B in-extent gaps plus the 3,328 B tail; revision 2 first wrote "2,560 B" here, which mis-summed them) |
| Build ID | matches the stamped build under test |
| RPC status / close status | 0 / 0 |

The first two rows are new in revision 2. They are a *static* pre-check: an
overlapping map is rejected before the MMA is issued, so a run cannot spend its
one attempt on a corrupted input. The map row is checked twice over: once from
the declared regions and once from the byte ranges the prologue fills actually
wrote (`c2_written_ranges`, audit 2026-10-30), so a fill that overshoots its
declared range is caught even if the map table itself were wrong.

On the copy-vector row: the device contract checks the DSP's reported copy
count (1 loop / 16 vectors). The CPU model's contract gate deliberately does
**not** gate on its copy counter, so that the content-level per-cell `0xDEAD`
test (§9) stays exercisable — fault 24 models a copy that stops half-way, which
the counter of a correctly-built kernel would not report. The two mechanisms
are complementary: the count catches an honest overrun, the `0xDEAD` residue
catches a copy that did not deliver what its count claims.

Any deviation fails closed as `C2_PREREG_CONTRACT_FAILED` and **no numeric
interpretation is made** from that run.

---

## 9. Decision procedure (evaluated in this order)

```text
[ device run ]
      |
      +--> RPC / Build-ID / counts / Rs / store addresses / guards /
      |    untouched-bytes / map-overlap / weight-tail mismatch?
      |        yes -> C2_PREREG_CONTRACT_FAILED            (stop, no interpretation)
      |        no
      v
   ANY cell == 0xDEAD?  --yes--> C2_COPY_NOT_EXECUTED      (stop)
      |                           partial copy included; no signature is read
      no
      v
   X != expected negative payload? --yes--> C2_STORE2_PATH_BROKEN
      |                                       (not a C1/C2 result)
      no
      v
   C1 signature AND non-zero C2 evidence? --yes--> C2_CAUSE_INCONCLUSIVE
      |                                             (two read points in one tile)
      no
      v
   C1 signature present?
      | yes -> C1_STORE_VISIBILITY_SUSPECT  (or C2_CAUSE_INCONCLUSIVE if any
      |                                      zero/ambiguous/unexplained cell is
      |                                      also present)
      | no
      v
   non-zero C2 evidence present? --yes--> C2_ORDERING_VIOLATED
      | no
      v
   only ambiguous subnormal zeros? --yes--> C2_SUBNORMAL_OR_ZERO_LOSS
      | no
      v
   any zero-agreement / payload-loss / unexplained cell?
      | yes -> C2_CAUSE_INCONCLUSIVE
      | no
      v
   P == expected positive payload on all 1,024 cells?
      | yes -> C2_MINIMAL_SEQUENCE_SUPPORTED
      | no  -> C2_CAUSE_INCONCLUSIVE
```

Rules attached to the procedure:

- **The `0xDEAD` test is per cell, not per tile.** Revision 1 only recognised a
  tile that was entirely `0xDEAD`, so a copy that stopped half-way was read as a
  numeric disagreement and could have been attributed to C2. Any residual
  `0xDEAD` now stops the run at `C2_COPY_NOT_EXECUTED`.
- **Run-level order of the terminal classes is fixed and test-pinned**
  (audit 2026-10-30): the `0xDEAD` residue test precedes the `X` interpretation
  (`c2_control_x_broken_plus_dead` constructs both and must stop as
  `C2_COPY_NOT_EXECUTED`), and the signature classes (mixed / C1 / non-zero C2)
  precede the subnormal / zero-loss classes. The second ordering is not free
  style: a genuine post-Store-#2 read zeroes the two subnormal cells as part of
  its own negative payload, so a literal "special values first" would misreport
  a real ordering violation as an FTZ observation —
  `c2_control_full_store2_read` contains both patterns at once and must land on
  `C2_ORDERING_VIOLATED`. Conversely, a C1 signature coexisting with subnormal
  zeros is inconclusive, not an FTZ observation
  (`c2_control_c1_plus_subnormal_inconclusive`). The special-value classes
  claim a run-level verdict only when no signature evidence coexists.
- **Special values are classified before any signature at the cell level**
  (§6 rows 1–6 precede row 7). A zero never becomes a C2 claim on its own.
- **Attribution needs a non-zero pattern.** `C2_ORDERING_VIOLATED` is reported
  only when at least one cell carries the expected negative payload as a
  **non-zero** bit pattern, because no zeroing law can produce a specific 32-bit
  neighbourhood of payload values. Once that is established the zero-valued
  cells are explained by the same read and do not veto the conclusion.
- On `C2_ORDERING_VIOLATED`, `C2_CAUSE_INCONCLUSIVE`,
  `C1_STORE_VISIBILITY_SUSPECT` or `C2_STORE2_PATH_BROKEN`: preserve the raw
  `P`/`X`/guard bytes, the full cell dump, the digests of what was actually
  observed, the Build ID and the RPC transcript, then **stop**. No retry, no
  "run it again to see", no clamping or resetting of anything.
- These signatures are **evidence consistent with** a cause, not a unique proof
  of it. `0xA5A5` cells say the read landed on pre-Store-#1 contents; they do not
  by themselves prove *why* the store was late. The classification selects the
  most conservative action, it does not certify a mechanism.
- No numeric tolerance is registered: the comparison is raw-bit exact.

---

## 10. What each outcome means for Candidate C

| Outcome | Consequence for Candidate C |
| --- | --- |
| `C2_MINIMAL_SEQUENCE_SUPPORTED` | C2 is *not disproved* in this build and this interleaving. Candidate C stays a candidate; C1/C3/C4/C5 remain separate contracts; a full-tile device trial is still unauthorised |
| `C2_ORDERING_VIOLATED` | the single-shared-buffer premise is unsafe as written. Candidate B (`CPU_VALIDATED_PREFERRED_DESIGN`) becomes the only surviving CPU-validated design, and a synchronization mechanism would have to be found before Candidate C is revisited |
| `C1_STORE_VISIBILITY_SUSPECT` | contradicts Mode 45; treat as a build/measurement problem first, not as new physics |
| `C2_SUBNORMAL_OR_ZERO_LOSS` | says nothing about C2; records that a payload cell read as zero and that this experiment cannot attribute it to Convert or to the HVX path |
| `C2_COPY_NOT_EXECUTED` / `C2_STORE2_PATH_BROKEN` | contract failure; the sequence under test was not the registered one |
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
| Authorisation | **NOT GRANTED** — requires explicit user approval of revision 2 |

Revision 1's flat "forbidden regardless" list mixed two different things.
Revision 2 separates them.

**Forbidden now, by this document, regardless of approval of it.** These are not
part of the experiment and approving §11 does not approve them:

- DSP source changes, DSP builds, QAIRT invocation, ADB contact, device
  connection, performance measurement;
- `push`, `main` changes, PR merges.

**Required for the single registered attempt, and therefore part of what
approving §11 would approve** (each still has to satisfy Tier 2 at execution
time):

- 1 clean stamped DSP build containing only this 1-MMA sequence;
- 1 device session, 1 attempt, no retry;
- the APK audit, the pinned QAIRT root / Build ID and the
  `focus_takeover_count=0` monitoring that Tier 2 requires.

Even under Tier 2's conditions (single resolved online endpoint, pinned QAIRT
root and Build ID, APK audit PASS, `focus_takeover_count=0`, no root / SELinux /
system changes), execution requires the user to approve *this* revision first.
Nothing here may be run on the strength of the Candidate C CPU result.

---

## 12. CPU-side validation performed for revision 2

The map and the decision procedure are not prose only: both are executable in
`host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c` and are checked every
run by the Fast gate.

| Check family | What it establishes | Result |
| --- | --- | --- |
| `c2_rev2_*` (16 checks) | revision 2 map: 0 overlaps, alignment, guards adjacent to X and clear of `wgt`, 8,960 B used, 3,328 B margin; plus the **written-range ledger**: the 7 prologue fills are pairwise disjoint and each equals its declared region | PASS |
| `c2_rev1_*` (3 checks) | the same checker rejects revision 1: exactly 1 overlapping pair, `wgt`/`guard_lo`, over 128 B | PASS |
| `c2_l2_*` (8 checks) | the L2 arena tiles exactly: guard low + P + guard high with no negative offset; `P` and both guards are 128 B multiples and the 16 copy vectors tile `P` up to the upper guard | PASS |
| `c2_nominal_*` (21 checks, incl. the two 1,024-cell diffs) | the registered sequence produces 1 `mxclracc`, 1 MMA, 2 converts, 2 identical store addresses, 16 copied vectors, clean guards, untouched gaps, and classifies as `C2_CPU_NOMINAL_MATCH` (the CPU rendering of the all-positive-payload outcome; §5.1) | PASS |
| `c2_nominal_sha_p` / `_sha_x` | the observed `P` and `X` digests equal the registered positive/negative digests | PASS |
| `c2_payload_census` | 1,023 discriminating, 475 non-zero negative payloads, 546 weak cells, 2 ambiguous subnormal cells | PASS |
| label contract (`c2_cpu_nominal_label`, `c2_cpu_label_never_device_label`) | no CPU context can render `C2_MINIMAL_SEQUENCE_SUPPORTED`, for any classifier outcome (§5.1) | PASS |
| `c2_control_*` (17 checks) | every branch of §9, the whole-tile post-Store-#2 read that must be attributed, and the 2026-10-30 audit controls: X-broken+`0xDEAD` order pin, C1+subnormal inconclusive, unknown bit pattern, P/X swap, both-tile decode-order swap, P-only decode-order swap | PASS |

Fault injections (`b3_r2_fault_injection=PASS detected=27/27`), the five new
ones being 23–27:

| # | Fault | Required detection | Observed |
| ---: | --- | --- | --- |
| 23 | `guard_lo` fill lands inside the `wgt` tile (revision 1 map) | map overlap **and** destroyed weight tail, with `mma = 0` (stopped before the MMA) | `CONTRACT_FAILED`, overlaps=1, tail intact=0, mma=0 |
| 24 | only 8 of 16 vectors reach `P` (partial copy) | `C2_COPY_NOT_EXECUTED`, and *not* an all-`0xDEAD`-only test | `COPY_NOT_EXECUTED`, deadic=512 |
| 25 | half of `P` read before Store #1, half after Store #2 | `C2_CAUSE_INCONCLUSIVE`, never a C2 verdict | `INCONCLUSIVE`, c1=512, non-zero C2=217, weak=295 |
| 26 | FTZ zeroes the subnormal payload cells | must **not** become a C2 signature | `SUBNORMAL_OR_ZERO_LOSS`, 2 cells, `would_be_false_c2=2` |
| 27 | copy destination shifted one vector past `P` into the L2 P guard | `C2_PREREG_CONTRACT_FAILED` from guard content (payload bytes in the guard) and the 64-cell `0xDEAD` residue | `CONTRACT_FAILED`, p_guard=1, deadic=64 |

Fault 26 is a classifier probe rather than a memory fault: the CPU model takes
the nominal observation and zeroes the cells whose positive payload is subnormal,
then re-runs the registered procedure. The `would_be_false_c2` counter is the
point of the test — under revision 1's rule those two cells would each have
matched the negative payload and the run would have been reported as an ordering
violation.

Two things the classifier tests found while they were being written, both now
fixed in the procedure:

1. A "zero-loss first" rule that simply discards every zero where a payload was
   due makes `C2_ORDERING_VIOLATED` unreachable, because a genuine post-Store-#2
   read yields `0x0000` on the 546 cells whose negative payload is zero. The rule
   now requires a non-zero negative pattern before attributing, and the
   whole-tile control proves a genuine violation is still reported.
2. A subnormal zero must be classified before the "agrees with the negative
   payload" test, otherwise it lands in the weak-evidence bucket and the run
   reads as inconclusive instead of as an FTZ observation.

The 2026-10-30 audit added decode/label probes and measured the
coincidental-match question directly instead of assuming it: swapping the `P`
and `X` arrays at harvest fails closed on the `X` check
(`c2_control_p_x_swapped`), decoding both tiles with the identity map instead
of the registered store crouton also fails closed on the `X` check
(`c2_control_decode_order_swapped`), and decoding only `P` wrongly produces
**zero** coincidental non-zero-negative matches on this fixture
(`p_swap_c2_coincidences=0`), so the run is inconclusive rather than attributed
(`c2_control_p_decode_swap_fail_closed`). Sentinel coincidences are impossible
by construction: `0xA5A5` and `0xDEAD` both have bit 15 set, and no stored
payload cell can (§5).

---

## 13. Reproducing the CPU-side registrations

```powershell
# CPU only; no SDK, no device, no ADB.
.\scripts\run_hmx_b3_r2_sim.ps1
# cand_c_layout vtcm_used=10752 ... margin=1536 ...
# cand_c_l2_lifetimes peak_raw=10240 peak_fold=6144 ...
# cand_c_bits ... all six arrays 0/1024 diff
# c2_prereg revision=2 vtcm_used=8960 margin=3328 rev2_overlaps=0 rev1_overlaps=1
# c2_prereg_map act=+0 wgt=+2048 guard_lo=+6016 X=+6144 guard_hi=+8192 bias0=+8448 bias1=+8704
# c2_written_ranges count=7 bytes=6912 overlaps=0 mismatch_declared=0
# c2_nominal p_diff=0 x_diff=0 ... class=C2_CPU_NOMINAL_MATCH
# c2_payload_census ... nonzero_neg=475 zero_neg_normal_pos=546 subnormal_pos=2
# c2_classifier_controls full_store2=C2_ORDERING_VIOLATED(simulated) ... subnormal_zero=C2_SUBNORMAL_OR_ZERO_LOSS(simulated)
# c2_control_matrix x_plus_dead=C2_COPY_NOT_EXECUTED(simulated) ... p_decode_swap=C2_CAUSE_INCONCLUSIVE(simulated) p_swap_c2_coincidences=0
# b3_r2_fault_injection=PASS detected=27/27
# b3_r2_checks=12776 fail=0
```

Registered artifacts are `build/`-only and gitignored. The digests above are the
registration; the files are not. The `b3_r2_checks` count is a deterministic
function of the simulator source (same count at `-O0` and `-O2`); it moved from
12759 to 12776 with the 17 checks this audit added, and from 12704 (revision 1,
commit `b22836e`) to 12759 when revision 2 was written.

---

## 14. Unresolved items carried forward

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
8. **Revision 1's map** — rejected on paper and in the CPU model, never executed.
   The device-side address constraints of the revision 2 map (whether `+6016`
   and the 128 B gaps are acceptable to the allocator) are a separate audit that
   has **not** been done.
