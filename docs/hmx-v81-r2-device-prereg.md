# R2 device pre-registration: retained-ACC two-convert export

Status: **PRE-REGISTERED, NOT AUTHORISED**, 2026-10-24.
Revision 2 (2026-10-25): ambiguous S1/S2 observations now map to
`R2_CAUSE_INCONCLUSIVE` on the device path; the `Rs=0x0001` no-feedback
composition is called out as the main residual spec risk; the 11-slot host
contract is re-checked. Revision 3 (2026-10-25): slots 8..10 are fixed as
DSP-authored HVX results in a third VTCM region, the bias rows move to make
room for it, and instruction counts require a DSP counter plus a static
disassembly audit of the same build. No DSP code has been added, no build has
been made, no device has been contacted. This document fixes the fixture, the expected
raw bits and the decision procedure **before** any measurement. After the
first run the expectations in §6 may not be edited; a mismatch is a result,
not a reason to retune.

This pre-registration follows the CPU-only feasibility study
[`hmx-v81-r2-acc-reuse-feasibility.md`](hmx-v81-r2-acc-reuse-feasibility.md)
and the implementation plan
[`hmx-v81-r2-device-implementation-plan.md`](hmx-v81-r2-device-implementation-plan.md).
The feasibility verdict is `R2_SPEC_FEASIBLE`; this document is the device
gate that would turn it into (or refute) `R2_ACC_REUSE_SUPPORTED`.

## 0. Question

Can one HMX MMA's 37-bit accumulator be converted **twice** — first with
`shape 0` and then with `shape 4`, two bias sets, two stores — without a
second MMA or an ACC recompute, and produce the two non-negative buffers
whose HVX difference is the signed FP16 value?

The test is designed to answer three things in one run:

1. **Retention + bias-set switch (S1):** a positive second store that is
   different from the first only because a different bias set was selected.
2. **Sign split (S2):** positive and negative ACC lanes exported into the
   correct buffers, with an exact HVX reconstruction.
3. **Host contract:** exactly one MMA per subcase, two converts per subcase,
   two stores per subcase, and the registered `Rs` values; a second MMA or a
   feedback bit fails closed.

## 1. Why a positive second store is mandatory

The existing record has two-convert sequences (modes 0 and 22), but neither
has a positive second store from the same ACC:

* mode 22's second store is `0x0000`; a destroyed ACC with the same `-64`
  output bias also gives `0x0000`;
* mode 0's second store is `0x0000`; it is consistent with a negative shaped
  term from the retained ACC, but feedback is `NOT_CONFIRMED` and the mode-0
  detail is a session record, not a durable raw transcript.

S1 removes both ambiguities. Its second convert selects a different bias set
with a different **positive** scale (`+32` instead of `+64`), so a destroyed
ACC gives 0 and a failed bias switch gives the first store's `0x5420`, while
a retained ACC with the correct set gives `0x5020`. No clamp is involved on
either side.

## 2. Fixture (frozen)

```text
a  = 1 + 2^-6 = 1.015625              FP16 0x3C10 (exact)
x  = a*a      = 4225/4096 = 1.031494140625
64x = 66.015625                       0.25 ULP above 66.0
32x = 33.0078125                      0.25 ULP above 33.0
```

Both scaled values are exactly 0.25 ULP above an FP16 grid point, so RN16 and
ceiling differ:

| value | RN16 | ceiling |
| --- | --- | --- |
| `64x = 66.015625` | `0x5420` | `0x5421` |
| `32x = 33.0078125` | `0x5020` | `0x5021` |
| `-64x = -66.015625` | `0xD420` | `0xD420` (toward -inf is the same here) |

The only rounding step before the store is the convert's RN16; `x` and `2x`
are exact.

## 3. Tiles

One 32x32x32 K32 tile per subcase. Only input channel 0 is non-zero. All
unlisted activation/weight cells are zero.

### S1: retention + bias-set control

```text
A(0,0) = +a
W(0,0) = +a
ACC(0,0) = +x
```

All other ACC cells are 0.

### S2: sign split

```text
A(0,0) = +a      A(1,0) = -a      A(2,0) = 0
W(0,0) = +a
ACC(0,0) = +x    ACC(1,0) = -x    ACC(2,0) = 0
```

All other ACC cells are 0. This puts positive, negative and zero ACCs in the
same tile, as requested.

## 4. Bias rows and `Rs`

Three 256-byte bias rows are loaded once at the start of the invocation:

| set | address | all 32 channels |
| --- | --- | --- |
| 0 | `base + 8192` | scale `0x5400` (+64), shape 0, bi 0, bo 0 |
| 1 | `base + 8448` | scale `0x5400` (+64), shape 4, bi 0, bo 0 |
| 2 | `base + 8704` | scale `0x5000` (+32), shape 0, bi 0, bo 0 |

The per-channel cell encoding is the PRM Figure 15 / Table 4 layout:

```text
lo32 = bo_fp16 << 16 | scale_fp16
hi32 = shape[10:8] << 8 | bo_extra4[7:4] << 4 | scale_extra4[3:0]
```

The 256-byte row is 32 low words (bytes 0..127) followed by 32 high words
(bytes 128..255), not 32 interleaved 64-bit words. The helper is
`bias20_lo_cell` / `bias20_hi_cell` in
`host_tests/hmx_acc_residual_rpc/bias20.h`.

`Rs` values:

| subcase | step | `Rs` | meaning |
| --- | --- | --- | --- |
| S1 | convert #1 | `0x0001` | set 0, retain ACC, no feedback |
| S1 | convert #2 | `0x2000` | set 2, clear+swap after, no feedback |
| S2 | convert #1 | `0x0001` | set 0, retain ACC, no feedback |
| S2 | convert #2 | `0x1000` | set 1, clear+swap after, no feedback |

`Rs[3:2]=0` in all four converts. A feedback bit, a non-zero `Rs[8]` or a
non-zero `Rs[6]` is a host-contract failure. The retain variant `0x1001` for
S2's second convert is **not** part of this test; it is CPU-checked as
store-equivalent in the feasibility study, but the first device run uses the
PRM's literal no-feedback recipe `0x1000`.

### 4.1 The `0x0001` no-feedback composition

PRM §2.3.3 says that for a convert without feedback, set `Rs[bias_sel]` to the
bias index and set the rest of the register to 0. The literal set-0
no-feedback encoding is therefore `0x0000` (clear+swap), while this test uses
`0x0001` (retain). Chapter 5's retain example also enables feedback
(`Rs[8]=1`, `fb_dest=2`, `fb_op=1`), so "retain + no feedback" is a
composition of two documented behaviours, not a verbatim PRM recipe.

This is the main residual spec risk. If the device treats the no-feedback
setup rule as normative and ignores `Rs[0]=1`, convert #1 clears the ACC and
S1's second slot is `0x0000`. That observation is
`R2_CAUSE_INCONCLUSIVE`, not `R2_ACC_DESTROYED`. S1 is therefore mandatory;
the device run must not be skipped on the grounds that ACC retention is
"already in the PRM".

## 5. VTCM map

| region | offset | size | alignment |
| --- | --- | --- | --- |
| `act` | `+0` | 2048 B | 2 KiB |
| `wgt` | `+2048` | 2048 B | 2 KiB |
| `out_pos` | `+4096` | 2048 B | 2 KiB |
| `out_neg` | `+6144` | 2048 B | 2 KiB |
| `out_recon` | `+8192` | 2048 B | 2 KiB |
| `bias` set 0 | `+10240` | 256 B | 256 B |
| `bias` set 1 | `+10496` | 256 B | 256 B |
| `bias` set 2 | `+10752` | 256 B | 256 B |
| total | | 11008 B | of 12288 B |

`out_pos`, `out_neg` and `out_recon` are distinct and all 2 KiB aligned.
`out_recon` is written by the DSP HVX loop, not by a convert store. Before
each store, fill the destination with `0xA5`; a readback of `0xA5A5` means the
store did not execute. The all-1024-cell store bijection is
`store_u16(s,o) = (s>>1)*64 + o*2 + (s&1)`.

## 6. Frozen readback slots and expectations

`results[0..10]` = 11 zero-extended 16-bit values in 32-bit longs. Indices are
store u16 indices.

| slot | subcase | cell | source | expected |
| --- | --- | --- | --- | --- |
| 0 | S1 | `(s0,o0)` | first store | `0x5420` |
| 1 | S1 | `(s0,o0)` | second store | `0x5020` |
| 2 | S2 | `(s0,o0)` | positive buffer | `0x5420` |
| 3 | S2 | `(s1,o0)` index 1 | positive buffer | `0x0000` |
| 4 | S2 | `(s2,o0)` index 64 | positive buffer | `0x0000` |
| 5 | S2 | `(s0,o0)` | negative buffer | `0x0000` |
| 6 | S2 | `(s1,o0)` index 1 | negative buffer | `0x5420` |
| 7 | S2 | `(s2,o0)` index 64 | negative buffer | `0x0000` |
| 8 | S2 | `(s0,o0)` | DSP `out_recon` (+8192) | `0x5420` |
| 9 | S2 | `(s1,o0)` index 1 | DSP `out_recon` (+8192) | `0xD420` |
| 10 | S2 | `(s2,o0)` index 64 | DSP `out_recon` (+8192) | `0x0000` |

The HVX recon is computed on the DSP in `out_recon`, not on the host, so the
run also exercises `vsub.hf` and the pos/neg buffer read path. A host-side
subtraction is a fallback analysis only: `recon_source` must be `DSP_HVX`
for `R2_ACC_REUSE_SUPPORTED`. The DSP must not overwrite `out_pos` or
`out_neg`; the run checks that both inputs are unchanged after the HVX loop.

The expected pattern is frozen:

```text
slot:   0      1      2      3      4      5      6      7      8      9      10
        0x5420 0x5020 0x5420 0x0000 0x0000 0x0000 0x5420 0x0000 0x5420 0xD420 0x0000
```

## 7. Execution counts and fail-closed conditions

| item | count |
| --- | --- |
| clean stamped V81 build | 1 |
| deploy/run sessions | 1 (attempts = 1, no retry) |
| subcases | 2 (S1, S2) |
| `mxclracc` | 2 (one per subcase, before the MMA) |
| MMA packets | 2 (one per subcase) |
| bias loads | 3 (sets 0, 1, 2; once at the start) |
| converts | 4 (2 per subcase) |
| stores | 4 (2 per subcase) |
| HVX recon loops | 2 (one per subcase) |
| returned slots | 11 |
| `metadata[7]` HMX executed | 1 |
| `metadata[8]` trials | 2 |

This minimal test uses **one MMA packet per subcase** (a single dot in S1, three
spatial lanes in S2). It is not a B3 accuracy run: the 3-packet B3 synthesis
and the 1,024-cell comparison are CPU-only evidence in the companion document.
The static audit therefore must not accept a third packet, and the R2 cost
model's 3 packets refer to the B3 accuracy path, not to this test.

S1 also runs the HVX recon loop even though slots 0..1 come from the two
convert buffers. The loop is part of the measured sequence: it checks that the
recon writer leaves `out_pos` and `out_neg` unchanged. Only S2's recon values
are registered as slots 8..10.

The current host result buffer is four slots. The future implementation must
widen it to 11; the RPC IDL already uses an unbounded `sequence<long>`, so no
IDL change is needed. That host change, the new mode and the DSP code are all
part of the future authorised work, not this CPU-only study.

Fail closed before interpreting any slot if any of the following holds:

* `results.size() != 11`, or a slot is not a zero-extended 16-bit value
  (`0..0xFFFF`);
* build id mismatch between host and DSP, or either id is unavailable when
  the run claims to be stamped;
* `rpc_status != 0` or `dsp_status != OK`;
* `hmx_executed != 1`;
* `trials != 2`;
* `mma_count != 2`, `convert_count != 4`, `store_count != 4`,
  `bias_loads != 3`, or `recon_loops != 2`;
* `Rs` values are not the four registered values (`0x0001`, `0x2000`,
  `0x0001`, `0x1000`), or the DSP `Rs` echo is absent/mismatched;
* `out_pos`, `out_neg` and `out_recon` are not distinct 2 KiB-aligned regions;
* `recon_source != DSP_HVX` (a host-side subtraction is not accepted);
* the count evidence is host-planned, or the DSP counter and static
  disassembly audit do not both exist and agree;
* the store sentinel did not change, or any slot is `0xA5A5`;
* any of the 11 slots is not zero-extended 16-bit;
* close status is non-zero or resource cleanup fails.

A fail-closed condition is `R2_HOST_CONTRACT_FAILED` or an infrastructure
failure; it is never interpreted as success.

## 8. Decision procedure

Evaluate in this order. The labels are the pre-registered future verdicts.

1. **Host contract.** If any §7 condition fails, report
   `R2_HOST_CONTRACT_FAILED` and do not interpret the slots.
2. **Store regions.** If the two stores of either subcase share a physical
   region, report `R2_STORE_INTERFERENCE`.
3. **Rounding.** If slot 1 is `0x5021`, or slot 2/slot 6 is `0x5421`, report
   `R2_ROUNDING_MISMATCH`. Do not adjust the expectation.
4. **S1 retention/bias control.** Slot 0 must be `0x5420`; otherwise
   `R2_UNEXPECTED_OUTPUT` (the first-convert anchor itself failed).
   * slot 1 = `0x5020` -> S1 passes: under the registered instruction
     sequence, the output is consistent with ACC reuse and Bias set 2
     selection. This is **not** a bit-level proof that all 37 ACC bits were
     retained or that the internal bank state is unchanged;
   * slot 1 = `0x0000` -> `R2_CAUSE_INCONCLUSIVE`. Candidate causes: retain
     bit ignored in a no-feedback convert, physical ACC loss, bias load
     failure, convert not executed, or store anomaly. Do **not** report
     `R2_ACC_DESTROYED` from this slot alone;
   * slot 1 = `0x5420` -> `R2_CAUSE_INCONCLUSIVE`. Candidate causes: set 2 not
     selected, set 2 row not loaded, `Rs` ignored, or store anomaly. Do
     **not** report `R2_BIAS_SWITCH_FAILED` from this slot alone;
   * slot 1 = `0x5021` -> `R2_ROUNDING_MISMATCH` (the registered ceiling
     alternative);
   * anything else -> `R2_UNEXPECTED_OUTPUT` with the raw slot recorded; no
     cause assertion.
5. **S2 first-store check.** If slot 2 = `0x0000` and slot 3 = `0x5420`, the
   second store overwrote the first: `R2_STORE_INTERFERENCE`.
6. **S2 sign split.**
   * slot 2 must be `0x5420`; otherwise `R2_UNEXPECTED_OUTPUT`;
   * slot 3 must be `0x0000`; a non-zero value here with slot 2 correct is an
     unexpected export, not the store-overwrite pattern already handled in
     step 5: `R2_UNEXPECTED_OUTPUT`;
   * slot 5 must be `0x0000`; if it is non-zero, the shape/bias/store path is
     wrong but the slot alone does not identify which: `R2_CAUSE_INCONCLUSIVE`;
   * slot 6 must be `0x5420`; if it is `0x0000`, the negative half is
     missing and the cause is not localized (shape 4, `Rs`, store, or lane
     map): `R2_CAUSE_INCONCLUSIVE`; a different non-zero word is
     `R2_UNEXPECTED_OUTPUT` with the raw slot recorded;
   * slots 4 and 7 must be `0x0000` (zero ACC lane);
   * slots 8..10 must match the recon pattern; otherwise
     `R2_UNEXPECTED_OUTPUT`.
7. **Success.** If S1 and S2 both match and the host contract passed, report
   `R2_ACC_REUSE_SUPPORTED`.

`R2_ACC_REUSE_SUPPORTED` must not be returned from S2 alone, from a run
without S1, or from a run with a second MMA / feedback bit. S1 is the
same-ACC evidence; S2 is the sign-split evidence.

### 8.1 What S1 success means

Slot 1 = `0x5020` is a strong output-consistency result: the registered
sequence (one MMA, convert `0x0001`, store, convert `0x2000`, store) produced
the value expected from the same ACC with Bias set 2. It is not a bit-level
proof of the 37-bit ACC contents, and it does not prove that the internal
bank pointer or all 1,024 cells are unchanged. The IMPLICATION is "consistent
with ACC reuse and the registered bias selection", not "ACC retention is
proven". The same qualification applies to the S2 sign split.

### 8.2 Evidence provenance: counters and disassembly are separate facts

| source | what it establishes | limitation |
| --- | --- | --- |
| Host-planned counts | intended recipe only | no execution evidence |
| DSP instruction counters | the instrumented path executed | self-reported; not an instruction-identity proof |
| Static disassembly of the built binary | the registered instruction sequence exists | does not prove it ran |
| DSP `Rs` echo | the reported `Rs` values match | self-reported |
| Store sentinel | a store wrote the region | no count/order by itself |
| 11 result slots | the numerical output | not bitwise ACC retention |

The verdict requires the result pattern, sentinel change, build-id match,
`Rs` echo, DSP counters, and a static disassembly audit of the same build.
The counter counts and the static-audit counts must agree; a mismatch or a
missing source is `R2_HOST_CONTRACT_FAILED`, not a success. Host-displayed
planned values are never evidence. The CPU-only `r2.evidence.*` tests model
this contract; they do not produce device evidence.

## 9. Registered failure predictions

`R2_ACC_DESTROYED` and `R2_BIAS_SWITCH_FAILED` in the first two rows are
**CPU fault-injection classifications**, not device conclusions. On the
device, the same S1 patterns are `R2_CAUSE_INCONCLUSIVE`; the predictions
below describe the expected raw bits, not a proven cause.

| injected/observed condition | prediction | model verdict | device verdict |
| --- | --- | --- | --- |
| ACC cleared/swapped between store #1 and convert #2 | S1 slot 1 = `0x0000`; S2 slots 5/6/7 = 0 | `R2_ACC_DESTROYED` | `R2_CAUSE_INCONCLUSIVE` |
| bias set is ignored; all converts use set 0 | S1 slot 1 = `0x5420`; S2 slot 5 = `0x5420`, slot 6 = `0x0000` | `R2_BIAS_SWITCH_FAILED` | `R2_CAUSE_INCONCLUSIVE` |
| both stores target the same region | S1 slots 0/1 read the second store (`0x5020`); S2 positive buffer holds the negative data | `R2_STORE_INTERFERENCE` | `R2_STORE_INTERFERENCE` |
| store #1 is overwritten by convert #2 before it reads the convert state | S2 slot 2 = `0x0000`, slot 3 = `0x5420` | `R2_STORE_INTERFERENCE` | `R2_STORE_INTERFERENCE` |
| convert rounds toward +inf | S1 slot 1 = `0x5021`; S2 slots 2/6 = `0x5421` | `R2_ROUNDING_MISMATCH` | `R2_ROUNDING_MISMATCH` |
| output bias on set 1 is non-zero | S2 slot 6 = `0x5420 + bo` (for `bo=+4`, `0x5460`) | `R2_UNEXPECTED_OUTPUT` | `R2_UNEXPECTED_OUTPUT` |
| feedback enabled on convert #2 | `Rs` contract fails if `Rs[3:2]!=0`; if the hardware leaks feedback with the registered `Rs`, slot 6 changes | `R2_HOST_CONTRACT_FAILED` or `R2_UNEXPECTED_OUTPUT` | `R2_HOST_CONTRACT_FAILED` or `R2_UNEXPECTED_OUTPUT` |
| shape 4 not applied | S2 slot 5 = `0x5420`, slot 6 = `0x0000` | `R2_BIAS_SWITCH_FAILED` | `R2_CAUSE_INCONCLUSIVE` |
| a second MMA was executed | counts fail | `R2_HOST_CONTRACT_FAILED` | `R2_HOST_CONTRACT_FAILED` |
| an unrelated word is corrupted | slot mismatch outside the registered alternatives | `R2_UNEXPECTED_OUTPUT` | `R2_UNEXPECTED_OUTPUT` |

The feedback output detector is masked if the clamp is inside the convert
(the previous convert state is already 0). The `Rs` contract is the robust
guard and is mandatory.

## 10. Stop conditions

After the run: record the 11 raw slots, the metadata, the build id and the
verdict; stop.

* No retry, no second session, no third subcase, no expectation change.
* `R2_ACC_REUSE_SUPPORTED` certifies only the S1/S2 minimal configuration. It
  does not certify a full 32-output-channel tile, real-tile accuracy, HVX
  subnormal behaviour, or performance.
* `R2_STORE_INTERFERENCE` blocks R2 on this runtime; the next option is R1
  or a different export path, not a retune.
* `R2_CAUSE_INCONCLUSIVE` stops the decision: record the raw slots, the
  metadata, the `Rs` echo, the bias-load status and the sentinel pattern;
  do not assert ACC destruction or bias-switch failure. A follow-up needs a
  new pre-registration (for example, a feedback variant that separates the
  retain-bit composition from physical ACC loss).
* `R2_ACC_DESTROYED` and `R2_BIAS_SWITCH_FAILED` remain **model-only**
  labels; they are not emitted by the device classifier.
* `R2_ROUNDING_MISMATCH` means the RN16 assumption does not hold for this
  path; record and diagnose. Do not edit §6.
* `R2_UNEXPECTED_OUTPUT` is recorded verbatim; no after-the-fact explanation.
* `R2_HOST_CONTRACT_FAILED` invalidates the run; do not interpret the slots.

## 11. What this test does not prove

* A full 32x32 tile: S1/S2 use one output channel and three spatial lanes;
  the other 1,021 cells are zero.
* Mixed per-channel shapes: each set is uniform across 32 channels.
* HVX subnormal, NaN or Inf behaviour; the fixture uses normal `66.0`/`33.0`.
* `Rs=0x1001` (retain after the second convert); the run uses `0x1000`.
* The first convert's `Rs=0x0001` as a **verbatim** no-feedback recipe: the
  PRM's no-feedback setup rule says to zero the non-bias bits, and Chapter 5's
  retain example uses feedback. S1 tests the composition, not a documented
  verbatim sequence.
* The cause of an ambiguous S1/S2 failure. The device classifier reports
  `R2_CAUSE_INCONCLUSIVE`; it does not localize the fault.
* The reload-between-converts bias variant.
* Real-tile accuracy, throughput, or R1-vs-R2 cost.
* Any production QNN/Muon path.

## 12. Authorization

Device execution requires **explicit user approval** for the new mode, the DSP
code change, one build and one device run. Nothing in this document
authorises an implementation or a measurement. The current statuses remain:
`B2_THREE_PRODUCT_CONFIRMED`, `E3C1R_LAYOUT_CONFIRMED_CLAMP`,
`E3D_CLAMP_DOWNSTREAM_OF_SHAPE`, `BLOCKED_SIGNED_OUTPUT_FOR_CURRENT_CONVERT_PATH`,
Path B `CONDITIONALLY_POSSIBLE`, real-tile accuracy `INSUFFICIENT_EVIDENCE`,
E-3C2 `UNTRIGGERED`. PR #14 stays Draft.
