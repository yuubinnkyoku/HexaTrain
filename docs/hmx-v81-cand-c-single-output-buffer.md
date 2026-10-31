# Candidate C — Single Shared Output Buffer (CPU-only evaluation)

Status: **CPU-ONLY HYPOTHESIS EVALUATED**, 2026-10-29.
Scope: design geometry, operand-immutability proof from recorded write events,
output-stage evidence preservation, L2 peak recomputed from lifetimes, the new
synchronization dependency chain, 1,024-cell parity with the registered G8/G9
expectations, and five new fail-closed fault models.
No DSP source changed, no DSP build, no QAIRT invocation, no ADB, no device
contact, no timing measured, no Mode 46 pre-registration update.

This document does **not** change any existing label:

| Item | Label (unchanged by this document) |
| --- | --- |
| D1–D3 (sequential streaming operand overwrite) | `DESIGN_ELIMINATED` |
| D8a (`Wlo` region reuse) | `DEVICE_ONLY` |
| D8b (`Ahi` region reuse) | `DEVICE_ONLY` |
| Candidate B | `CPU_VALIDATED_PREFERRED_DESIGN` |
| B3+R2 CPU integration | `B3_R2_FULL_TILE_CPU_READY` |
| B3+R2 synchronization | `B3_R2_SYNC_UNPROVEN` |
| Mode 46 | `NOT_RUN` |
| **Candidate C** | **`CAND_C_CPU_MODEL_CONSISTENT`** (CPU scope only) |

`CAND_C_CPU_MODEL_CONSISTENT` means the CPU model is internally consistent and
reproduces the frozen bits. It is not a design verdict, not a device result, and
not a promotion of Candidate C over Candidate B.

---

## 0. The question this document answers

Candidate B removed the D1–D3 hazard by phase-reusing operand regions, at the
cost of one narrower dependency: `out_neg`/`out_recon` overwrite `Wlo`/`Ahi`
after MMA #3 and both converts, so D8 asks whether MMA #2's operand reads are
finished by then. D8 is `DEVICE_ONLY`.

The open question was whether the *final* reuse of operand regions can be
removed as well, so that no operand region is ever written again after the
prologue. The answer is yes, and only in one shape inside 12 KiB, because:

| All regions live at the same time | Bytes | Fits 12,288 B? |
| --- | ---: | --- |
| 4 operands + 3 outputs + 2 bias sets | 8,192 + 6,144 + 512 = 14,848 | no |
| 4 operands + 2 outputs + 2 bias sets | 8,192 + 4,096 + 512 = 12,800 | no, over by 512 |
| **4 operands + 1 output + 2 bias sets** | 8,192 + 2,048 + 512 = **10,752** | **yes, margin 1,536** |

So "four resident operands, never rewritten" forces exactly one 2 KiB output
buffer. Candidate C is that layout.

---

## 1. VTCM map

| Region | Offset | Size | Writes per run | Lifetime |
| --- | ---: | ---: | --- | --- |
| `Ahi` | `+0` | 2,048 B | 2 — once in each subcase prologue | that subcase's MMA phase |
| `Alo` | `+2048` | 2,048 B | 1 — B3 prologue only | B3 MMA phase |
| `Whi` | `+4096` | 2,048 B | 2 — once in each subcase prologue | that subcase's MMA phase |
| `Wlo` | `+6144` | 2,048 B | 1 — B3 prologue only | B3 MMA phase |
| `OUT` (shared) | `+8192` | 2,048 B | 3 per subcase: pos, neg, recon in place | output phase |
| bias set 0 | `+10240` | 256 B | 2 — once in each subcase prologue | that subcase |
| bias set 1 | `+10496` | 256 B | 2 — once in each subcase prologue | that subcase |
| **used** | | **10,752 B** | | of 12,288 B, **margin 1,536 B** |

The two subcases share one physical VTCM, so a region's "written once" is **per
subcase**, not per run: the B3 prologue legitimately repacks `+0` and `+4096`
after the Plain subcase's MMAs have finished. Six operand-region writes occur in
the whole run (2 Plain + 4 B3), and none of them is between two MMAs.

MMA phase usage 8,704 B; peak 10,752 B — the same peak as the registered
streaming map and Candidate B. All three designs peak at 10,752 B; they differ
in *what* is reused and *when*, not in how much VTCM they need.

There is no `act`/`wgt` streaming buffer and no `Whi` restore: `Whi` stays at
`+4096` for the whole kernel, so the `stWhi` staging tile, its FNV checksum and
fault 14 have no counterpart here.

---

## 2. Execution schedule

### Plain + R2

```text
pack A_plain -> +0, W_plain -> +4096      (2 VTCM writes, operand regions)
build bias sets at +10240 / +10496
mxclracc
MMA #1: act=+0, wgt=+4096
sentinel fill OUT (+8192)
Convert #1 (Rs 0x0001, bias 0, shape 0) -> OUT      pos
HVX save  OUT -> L2 plain_pos                       16 vectors
Convert #2 (Rs 0x1000, bias 1, shape 4) -> OUT      neg (overwrites pos)
HVX save  OUT -> L2 plain_neg                       16 vectors
HVX in-place recon: pos(L2) - neg(OUT) -> OUT       16 vectors
HVX save  OUT -> L2 plain_recon                     16 vectors
```

### B3 + R2

```text
pack Ahi -> +0, Alo -> +2048, Whi -> +4096, Wlo -> +6144   (4 writes)
build bias sets
mxclracc
MMA #1: Ahi * Whi        (act +0,     wgt +4096)
MMA #2: Ahi * Wlo        (act +0,     wgt +6144)
MMA #3: Alo * Whi        (act +2048,  wgt +4096)
sentinel fill OUT
Convert #1 -> OUT ; HVX save -> L2 b3_pos ; Convert #2 -> OUT ;
HVX save -> L2 b3_neg ; HVX in-place recon (pos L2, neg OUT) -> OUT
host readback of OUT over the RPC boundary            recon
```

The B3 prologue writes only `+0..+8191`, so the Plain subcase's `OUT` content
survives it; the first write that destroys it is the B3 output phase's sentinel
fill, immediately before the B3 **Convert #1**. That gives the Plain evidence
copy a window of a whole prologue plus three MMA packets, instead of the single
instruction Candidate B has.

Both subcases are modelled on **one physical VTCM** (`vtcm_shared_by_both_subcases=1`),
not as two independent runs, so the cross-subcase ordering is checked and the
"the Plain result must be evacuated" claim is measured: see §4.1.

### Static copy accounting (counts only — no timing claim)

| Item | Count |
| --- | ---: |
| VTCM -> L2 tile copies, Plain | 3 (`pos`, `neg`, `recon`) |
| VTCM -> L2 tile copies, B3 | 2 (`pos`, `neg`; recon stays in VTCM) |
| Total copies | **5 tiles = 80 vectors of 128 B = 10,240 B** |
| Mid-kernel operand writes | **0** |
| Final operand-region reuse | **none** |

No cycle, nanosecond or percentage figure is registered. The same rule as the
pre-registration applies: the execution cost of these copies depends on issue
width, L2-to-VTCM bandwidth, store/load dependencies and VTCM bank behaviour,
none of which is established for this kernel.

---

## 3. L2 peak recomputed from lifetimes

The output stages cannot all be live in VTCM, and the plain results must leave
`OUT` before it is rewritten. The arena is derived from the lifetime table, not
from a sketch:

| Tile | Size | Live from | Live to |
| --- | ---: | --- | --- |
| `plain_pos` | 2,048 B | Plain pos save | end of kernel (evidence) |
| `plain_neg` | 2,048 B | Plain neg save | end of kernel (evidence) |
| `plain_recon` | 2,048 B | Plain recon save | end of kernel (evidence) |
| `b3_pos` | 2,048 B | B3 pos save | B3 in-place recon (as the positive input) |
| `b3_neg` | 2,048 B | B3 neg save | end of kernel (evidence) |

Stage liveness (bytes):

| Stage | Live | Bytes |
| --- | ---: | ---: |
| Plain pos saved | 1 | 2,048 |
| Plain neg saved | 2 | 4,096 |
| Plain recon saved | 3 | 6,144 |
| B3 pos saved | 4 | 8,192 |
| **B3 neg saved** | **5** | **10,240 (peak)** |
| B3 recon read from VTCM | 5 | 10,240 |

**Peak = 10,240 B (10 KiB)** under raw-bit retention of all plain evidence.

If the plain `pos` and `neg` tiles are folded into the mutual-exclusivity
counters during the Plain recon pass instead of being retained, only
`plain_recon` survives, and the peak becomes **6,144 B (6 KiB)**:

| | Raw retention | Folded |
| --- | ---: | ---: |
| L2 peak | 10,240 B | 6,144 B |
| Preserved raw arrays | 6 of 6 | 4 of 6 (plain pos/neg not retained) |

Both numbers are computed by the simulator from the two liveness tables and
asserted (`cand_c_l2_lifetimes peak_raw=10240 peak_fold=6144`). They are the
honest cost of Candidate C: **more L2 than Candidate B (2,048 B) and more than
the streaming design's 6,144 B operand staging** (which, note, does not include
the plain-evidence save it also needs).

---

## 4. Operand immutability, proved from the write events

The review requirement was to establish operand immutability from the writes
that actually happen, not from the intended schedule. The model logs **every
VTCM access** as `(seq, kind, offset, size, is_write)` and classifies each write
against the four operand ranges.

Nominal B3 run (`cand_c_write_census`):

| Key | Value |
| --- | ---: |
| VTCM writes, total | 24 |
| Writes touching an operand region | **4** (the four prologue packs) |
| Operand-region writes **after MMA #1 was issued** | **0** |
| Writes inside `OUT` | 19 (1 sentinel, 2 MX stores, 16 in-place vector stores) |
| Event-log overflow | 0 |

Nominal Plain run: 2 operand writes (the two packs), 0 after MMA #1.
Whole run: **6** operand-region writes (`whole_kernel_operand_writes=6`).

Scope of the claim, stated explicitly: "zero operand writes after MMA #1" is a
statement **about one subcase's MMA phase**. It does not forbid the prologue of
the next subcase from repacking the same regions — those writes happen before
that subcase's first MMA, when the previous subcase's MMAs and converts are long
finished. No write in the whole run lands between two MMA packets.
An independent FNV-1a check over each of the four operand tiles before and after
the whole MMA phase also reports zero mutation.

### 4.1 The between-subcase save is load bearing (measured)

Because both subcases run in one VTCM image, the model can show rather than
assert that evacuating the Plain result matters:

| Check | Result |
| --- | --- |
| C5 ordering: Plain recon copy seq < the first B3 write into `OUT` | holds |
| The B3 output phase changes the bytes that were in `OUT` | yes (snapshot differs) |
| The `plain_recon` L2 tile is unchanged by the B3 subcase (FNV-1a) | holds |
| Plain evidence deferred/absent | no |

### 4.2 This is where Candidate C does not inherit the Candidate B defect

The D8 audit found a real inconsistency in the Candidate B model: its
`memset` sentinel fill of the aliased `out_neg`/`out_recon` regions is the
**first actual write** into those operand regions, and it happens *before*
Convert #1, while the ordinal contract only orders the convert store
(`last_use(owner) < PR_ORD_NEG_STORE`). The model therefore commits the D8
hazard earlier than the contract it validates.

In Candidate C the sentinel target is `OUT` at `+8192`, which is disjoint from
every operand region by construction, so the fill is legal whenever it runs.
The check is on **all** write events, not on a store ordinal, so the class of
error is not representable. This is a property of the layout, not of a stricter
assertion bolted onto the same schedule.

Recorded as **CB-FIX-1** in §9 — it remains an open fix item for Candidate B and
is deliberately *not* applied here.

---

## 5. New synchronization dependency chain

Candidate C does not eliminate hardware ordering questions; it replaces them.
The chain is `MX store -> HVX save -> MX store -> HVX in-place recon`, and every
link is classified at the strength the project's evidence supports.

| # | Dependency | Class | Basis |
| --- | --- | --- | --- |
| **C1** | Convert #1 MX store -> HVX load of `OUT` for the pos save | `IMPLEMENTATION_SUPPORTED` | Mode 45: `mxmem(...) = cvt` store followed by an HVX `vsub` read of the same bytes, S1/S2 slots matched |
| **C2** | HVX pos save complete -> Convert #2 MX store over `OUT` | **`DEVICE_ONLY`** | Never exercised. No spec, no fence in the shipped object, no device run where a load is followed by an MX store to the same bytes |
| **C3** | Convert #2 MX store -> HVX load of `OUT` for the neg save | `IMPLEMENTATION_SUPPORTED` | Same direction as C1 |
| **C4** | In-place recon: per-vector load before store, same thread | `PROGRAM_ORDER` | The store's data dependency passes through the load's result register, so this is a register dependency, not a cross-agent memory fence; vectors do not alias each other |
| **C5** | Plain evidence -> L2 before the B3 Convert #1 | `PROGRAM_ORDER` | Interior to one thread's instruction order; the window is a whole prologue + 3 MMAs |

Exactly **one** dependency (C2) is a hardware question, and it is a different
question from D8:

| | Candidate B (D8) | Candidate C (C2) |
| --- | --- | --- |
| Which bytes | a **live operand tile** (`Wlo`, `Ahi`) | the **output buffer** |
| Who writes | MX store / HVX store after MMA #3 + 2 converts | MX store between Convert #1 and #2 |
| Distance from the read | one MMA + one convert later | immediately after the save |
| If the hardware reorders | an MMA consumes stale/overwritten operands | the saved positive tile is garbled |
| Failure signature | silently wrong arithmetic | silently wrong evidence and recon |

Two honest conclusions, neither of which is "C is safe":

1. **C2 is not weaker in kind than D8.** Both are write-after-read hazards
   between two agents on the same VTCM bytes, and both fail silently.
2. **C2 is far cheaper to test.** It needs one MMA, one convert, one HVX save
   and one more convert — no 3-product stream, no `Whi` restore, no full-tile
   operand supply. A single-MMA experiment that examines only the output buffer
   can bound it, whereas D8 requires the full B3 schedule to be running before
   the question can even be asked.

That is the actual research-risk argument for Candidate C: it does not remove
the undocumented hazard, it **moves it onto a buffer that can be tested in
isolation, with one MMA instead of three**.

---

## 6. 1,024-cell parity with the frozen G8/G9 expectations

The candidate C model harvests all six arrays in decoded cell order (`s*32+o`)
and compares them against the arrays the registered streaming model produces
under the registered `deploy_float_split` condition.

| Array | Cells | Differing cells |
| --- | ---: | ---: |
| Plain `out_pos` | 1,024 | **0** |
| Plain `out_neg` | 1,024 | **0** |
| Plain `out_recon` | 1,024 | **0** |
| B3 `out_pos` | 1,024 | **0** |
| B3 `out_neg` | 1,024 | **0** |
| B3 `out_recon` | 1,024 | **0** |

The six registered SHA-256 digests are unchanged. The two output digests are
recomputed from the candidate C bits inside the simulator and compared with the
pinned constants:

| Artifact | SHA-256 (unchanged) |
| --- | --- |
| Plain expected raw bits | `0e58d8f88c8ac00d005a3a93e743fedc90bdc3cb0e163d0a23b18c118e9336bb` |
| B3 expected raw bits | `687509a2c5999541ee15d5da21aecd28e41d37a4417ed22b2f0a13184e030939` |

Two further digests are registered **for Candidate C only**, because the C2
pre-registration observes the positive and negative buffers separately and needs
their expected bits, not just the reconstruction. They are the Plain subcase's
single-MMA arrays:

| Candidate-C artifact | Cells | SHA-256 |
| --- | ---: | --- |
| Plain `out_pos` raw bits | 1,024 | `82107d6e53e2e6d7d07bf007357d722fdfcb4a17c00a3582c3b514e90f76ffb8` |
| Plain `out_neg` raw bits | 1,024 | `3fcddb24e66920f21ff2818aea0f3003f344ac006b546047e3151e6bce1a3309` |
| Cells where `pos != neg` (classifier denominator) | 1,023 | — |

The single non-discriminating cell is the one where both halves store `0x0000`,
so it carries no information about which buffer was read. The six frozen
digests above are untouched by this addition.

(`b3_r2_sha256_input_truth`, `_oracle_double`, `_operands_deploy` and
`_operands_fp32` are untouched: the fixture and operand conditions were not
modified.) Because the bits are identical, the frozen metric table
(`1.307244e-03`, `7.084101e-04`, ratio `1.845321`, 427 / 597 / 0) applies to
Candidate C without recomputation — the layout changes the memory schedule, not
the arithmetic. That also means adopting Candidate C would cost no numerical
evidence, exactly as with Candidate B.

---

## 7. Fault injection: five new models, all fail closed

Added to the battery as faults 18–22 (battery total 22; the streaming set 1–14
and the Candidate B set 15–17 are unchanged and still detected). Each of the
five has a genuine numeric consequence, not only a bookkeeping flag:

| # | Fault | Injected defect | Observed consequence (this run) |
| --- | --- | --- | --- |
| 18 | `CC_POS_SAVE_MISSING` | positive tile never copied out before Convert #2 | 548/1024 `pos` cells and 548/1024 recon cells wrong; flag set |
| 19 | `CC_RESTORE_BEFORE_SAVE` | Convert #2 issued after 8 of 16 save vectors | 408/1024 `pos` cells wrong (second half filled with neg bytes), 400/1024 recon cells wrong; the save-completion landmark now lands **after** Convert #2, i.e. the C2 ordering is recorded as violated |
| 20 | `CC_NEG_EVIDENCE_LOST` | negative raw bits not copied before the in-place recon | 476/1024 `neg` cells wrong, recon **unchanged** — the signature of evidence loss rather than arithmetic loss |
| 21 | `CC_INPLACE_RW_CONFLICT` | one vector stored before its own neg load | exactly 64 recon cells (one 128 B vector) wrong |
| 22 | `CC_PLAIN_CLOBBERED` | Plain evidence copied after the shared buffer was rewritten | 428/1024 Plain recon cells wrong; deferral flag set |

Fault 20 is the one worth keeping in mind: the reconstructed output stays
correct while the evidence for the negative half is destroyed. A verifier that
only checks the recon bits would pass a run whose `out_neg` proof is gone.

`b3_r2_fault_injection=PASS detected=22/22`.

---

## 8. Three-way comparison

| Design | Operand overwritten between MMAs | Final operand reuse | Mid-kernel operand tiles | L2 | Output buffers | Unproven |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Streaming (registered) | yes, 2 regions | yes | 3 | 6,144 B (+ plain save) | 3 | D1, D2, D3 |
| Candidate B (phase reuse) | no | yes (`Ahi`, `Wlo`) | 0 | 2,048 B | 3 | D8 |
| **Candidate C (shared output)** | **no** | **none** | **0** | **10,240 B** | **1** | **C2** |

CPU parity: all three produce the same six 1,024-cell arrays and the same six
digests.

Costs of Candidate C, stated plainly:

- **+8,192 B more L2 than Candidate B** (10,240 vs 2,048), or +4,096 B with the
  folded-evidence policy;
- **five tile copies (10,240 B / 80 vectors)** that B does not need, on the same
  DRAM-free but not free-of-cost L2 path. Performance is **not** claimed to be
  better than B; it may be worse;
- the in-place recon means the negative half's VTCM copy is destroyed by design,
  so its evidence must be taken to L2 *before* the recon — a new, load-bearing
  ordering that B does not have;
- CPU consistency is not a design verdict. Candidate C has been evaluated, not
  endorsed.

---

## 9. Open items (unchanged priority order)

1. **CB-FIX-1 (Candidate B, not yet applied)** — the sentinel fill of
   `out_neg`/`out_recon` is the first real write into the aliased operand
   regions and is not covered by `last_use(owner) < PR_ORD_NEG_STORE`. Either
   register the fill as its own write event with an ordinal, or stop filling
   those regions and detect unwritten stores another way. Until fixed, the
   Candidate B model over-reports its own ordering guarantee. The 1,024-cell
   bits and the CPU results are unaffected.
2. **D8a / D8b (`DEVICE_ONLY`)** — unchanged. D8 is a diagnostic *hypothesis*
   about which region reuse corrupts the result; a pattern match in one run does
   not establish the internal cause.
3. **Candidate C C2 (`DEVICE_ONLY`, new)** — requires a device experiment that
   deliberately commits the hazard: MMA -> Convert -> HVX save of `OUT` -> Convert
   over `OUT` -> read back both. The failure signature is a garbled positive
   tile, so the experiment is small and directly attributable. **Not authorised,
   not implemented, not scheduled.**
4. **Mode 46** — `NOT_RUN`, no pre-registration update. Candidate B remains the
   preferred basis for any future Mode 46 work; Candidate C is a parallel
   hypothesis, not a replacement.

This document does not authorise a DSP build, a device trial, Mode 46, or any
change of preferred candidate. Approval for any device experiment remains with
the user.

---

## 10. Reproducing this

CPU only; no SDK, no device, no ADB, no QAIRT.

```powershell
.\scripts\run_hmx_b3_r2_sim.ps1
# b3_r2_sim_gcc_version=gcc.exe ...
# cand_c_layout vtcm_used=10752 vtcm_total=12288 margin=1536 ...
# cand_c_l2_lifetimes peak_raw=10240 peak_fold=6144 arena=10240 ...
# cand_c_bits plain_pos_diff=0 ... b3_recon_diff=0 cells=1024
# cand_c_sha256 plain=0e58...b3=6875... registered_unchanged=1
# cand_c_sha256_plain_pos_bits=82107d6e...
# cand_c_sha256_plain_neg_bits=3fcddb24...
# cand_c_plain_payload discriminating_cells=1023 non_discriminating=1 cells=1024
# cand_c_write_census b3_vtcm_writes=24 b3_operand_writes=4 b3_operand_writes_after_mma1=0 ...
#                       whole_kernel_operand_writes=6
# cand_c_sync ... (5 lines, C2 class=DEVICE_ONLY)
# design_compare ... (3 lines)
# b3_r2_fault_injection=PASS detected=22/22
# b3_r2_checks=12704 fail=0
# b3_r2_verdict=B3_R2_FULL_TILE_CPU_READY
```

Sources: `host_tests/hmx_acc_residual_rpc/hmx_b3_r2.h` (candidate C constants,
faults 18–22) and `host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c`
(`run_cand_c_evaluation`, `cand_c_fault_detected`). Generated artifacts stay
under `build/` and are not committed.

Environment limitation of the recorded run: `gcc 15.2.0 (MSYS2, x86_64-w64-mingw32)`
is the gate-owning compiler; the runner's `clang` cross-check printed
`SKIP (clang not found on PATH)` and is **not** treated as a pass. Only the gcc
result is registered here.

---

## 11. Relationship to the existing documents

- `docs/hmx-v81-b3-r2-full-tile-prereg.md` is the frozen pre-registration of the
  **streaming** design. Its 14 fault models, its layout, its digests and its
  verdict are untouched.
- `docs/hmx-v81-b3-r2-sync-static-audit.md` records the D1–D8 classification and
  the Candidate B evaluation, ending at 17 faults. That count was correct for its
  scope; the battery now runs 22 because Candidate C adds 5.
- This document adds a third, separately labelled hypothesis. It changes no
  frozen number and no existing label.

---

## 12. Verdict

**`CAND_C_CPU_MODEL_CONSISTENT` (CPU scope only).**

- Four resident operand tiles plus **one** 2 KiB shared output buffer fit in
  10,752 B of the 12,288 B VTCM window with a 1,536 B margin. Two output buffers
  with four resident operands do not fit (12,800 B, over by 512 B), so the single
  shared buffer is the only way to keep every operand region unwritten.
- Operand immutability is established from **recorded write events**: 4 operand
  writes in the whole kernel, all in the prologue, **0 after MMA #1**.
- All three output stages are preserved: pos and neg are copied to L2 before the
  buffer is reused or overwritten in place, and the plain evidence leaves `OUT`
  before the B3 output phase writes the shared buffer (verified on one physical
  VTCM image, including that the B3 phase really does destroy the bytes left
  there). L2 peak recomputed from lifetimes: **10,240 B** with raw retention,
  **6,144 B** with folded plain counters.
- All six 1,024-cell arrays are **bit-identical** to the registered model and the
  six SHA-256 digests are unchanged.
- The new chain `MX store -> HVX save -> MX store -> HVX in-place recon` has
  **exactly one** dependency that only a device can answer (**C2**); C1 and C3
  are the direction Mode 45 already exercised, C4 and C5 are same-thread program
  order.
- Five new fault models fail closed, each with a measured numeric consequence.

What it does not establish: that C2 holds on silicon, that Candidate C performs
better than Candidate B (it has more copies and more L2), or that Candidate C
should replace Candidate B. `B3_R2_FULL_TILE_CPU_READY`,
`B3_R2_SYNC_UNPROVEN`, `CPU_VALIDATED_PREFERRED_DESIGN` and Mode 46 `NOT_RUN` all
stand as recorded.
