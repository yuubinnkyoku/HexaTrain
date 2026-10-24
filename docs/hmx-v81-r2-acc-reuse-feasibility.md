# R2 feasibility: one MMA, two converts over a retained V81 HMX accumulator

Status: **CPU_ONLY / SPEC_AUDIT**, 2026-10-24. No device contact, no ADB, no
DSP change, no DSP build, no new HMX mode, no performance measurement, no
training, no push, no PR merge. This document does not change any standing
research verdict.

**Verdict: `R2_SPEC_FEASIBLE`.** The PRM specifies every operation R2 needs:
the ACC can be retained across converts, a convert reads the ACC and writes
the convert state without consuming the ACC when `Rs[0]=1`, a second convert
can select a different bias set, and a store does not modify the convert
state. Device behaviour for the exact two-convert R2 recipe is **not**
verified. A CPU-model PASS is not the basis of the verdict; the PRM text and
the existing recorded device modes are (§2, §3).

The device question is pre-registered separately in
[`hmx-v81-r2-device-prereg.md`](hmx-v81-r2-device-prereg.md). It is **not
authorised** by this document.

## 0. Scope and non-claims

R2 is a way to recover a signed FP16 value from one HMX accumulator without a
second MMA:

```text
one MMA (B3: three product packets into one ACC)
        |
        +-- convert #1: bias set 0, shape 0, scale +64, bo 0, Rs[0]=1
        |       -> store positive buffer
        |
        +-- convert #2: bias set 1, shape 4, scale +64, bo 0
                -> store negative buffer
                          |
                    HVX positive - negative
                          |
                    signed FP16
```

The absolute condition is: **convert #2 must not need a new MMA or an ACC
recompute.** It reads the same retained 37-bit accumulator that convert #1
read.

This document does **not** claim:

* a device-confirmed R2;
* a signed HMX store (`BLOCKED_SIGNED_OUTPUT_FOR_CURRENT_CONVERT_PATH`
  remains);
* an accuracy improvement (the G9 1.845321 is B3's, carried over unchanged);
* a performance result (the cost model is static instruction/byte counting);
* real-tile accuracy (the 1,024-cell tile is a CPU model);
* any change to `B2_THREE_PRODUCT_CONFIRMED`,
  `E3C1R_LAYOUT_CONFIRMED_CLAMP`, E-3D
  `CLAMP_DOWNSTREAM_OF_SHAPE`, Path B `CONDITIONALLY_POSSIBLE`, or E-3C2
  `UNTRIGGERED`.

## 1. Why R2 is different from R1

E-3D established that a negative convert result is zeroed **downstream of the
shaping function** (`out[2]=0x0000` with a positive ACC and `shape 4`;
`out[3]=0x5420` with a negative ACC and `shape 4`). The sign can therefore be
recovered by exporting the positive and negative halves separately and
subtracting them in HVX.

There are two ways to obtain the two halves:

| | R1: duplicated output channels | R2: two passes over one ACC |
| --- | --- | --- |
| physical output channels | 2 per logical output | 1 per logical output |
| MMA work for 32 logical outputs | 2 tiles x 3 B3 packets = 6 packets | 1 tile x 3 B3 packets = 3 packets |
| converts / stores | 2 / 2 per 32 logical outputs | 2 / 2 per 32 logical outputs |
| needs a second MMA? | yes (the second tile) | **no** (same ACC) |
| output VTCM before HVX | 4 KiB | 4 KiB |
| output-channel duplication | yes | **no** |

R1 is the "one pass, duplicated channels" realization assumed by the G9
model. R2 is the realization that the PRM's `Rs[0]=1` retain bit makes
possible. R2 is not free — it still pays a second convert and a second store
— but it does not repeat the three B3 MMA packets. That is the whole question.

## 2. PRM evidence

Source: *Qualcomm Hexagon V81 HMX Programmer's Reference Manual*,
80-N2040-62 Rev AA, March 2026. The local text extraction used for the audit
is `build/privacy-audit/hmx_v81_prm.txt` (gitignored; not evidence by itself).
Page numbers below are the PRM's printed page numbers.

### 2.1 Accumulator, primary/secondary, and `Rs[0]`

PRM §2.2.3 (p.9–10):

> HMX has a pair of accumulators. Each accumulator is a matrix with a size of
> spatial dimension by output-channel, which has the size of 32 (spatial) x 32
> (output-channel) x 37-bit (cell).

> A matrix multiply operation stores the results in the primary accumulator,
> and a following convert transfer instruction can optionally clear and swap
> the accumulator. The convert instructions are the only way to accesses the
> accumulators and store the data back to VTCM.

Table 6 (p.15–16) defines the convert scalar bit:

| `Rs[0]` | Meaning |
| --- | --- |
| `0` | Clear accumulator values and swap primary accumulator |
| `1` | **Retain accumulator values** |

So `Rs[0]=1` is not a hint or a "best effort"; it is the specified retain
behaviour. It leaves the primary/secondary bank pointer untouched and does not
clear the values. `Rs[0]=0` is the only documented path that both clears the
primary values and flips the bank (`mxclracc` clears without flipping).

PRM §2.5 (p.19) adds an independent clear instruction
(`Q6_mxclracc_hf`), which clears both accumulators in a set. The two
documented clear operations are therefore the convert's `Rs[0]=0` and
`mxclracc`; no other instruction in the reviewed text clears one.

**Claim:** after convert #1 with `Rs[0]=1`, the same 37-bit ACC values are
still in the primary accumulator and are readable by a following convert.

**Class:** `SPEC_CONFIRMED` for the retain field; `SPEC_IMPLIED` for "the
following convert reads the primary" and for the bank not being swapped. The
table states value retention explicitly; the bank is not swapped because `0`
is the only behaviour that says "swap" (the complement is an inference, not a
second explicit sentence). The PRM's primary/swap model plus "convert
instructions are the only way to access the accumulators" supplies the read
path, but the PRM does not spell out the bank read in one sentence.

### 2.2 Convert semantics: read ACC, write convert state, optionally clear

PRM §2.3 (p.11–12):

> The convert state transfer instruction uses the bias register values to
> perform the convert operation on the accumulator data, then transfers the
> result to the convert state register. Every cell in the accumulator goes
> through the same convert operation and is stored in the corresponding
> location in the convert state register. The instruction then optionally
> clears the accumulator data. This instruction reduces the 37-bit floating
> point accumulator value to FP16.

The convert equation (Figure 11, p.12) is:

```text
cvt = s * shape(ACC + bi) + bo        then RN16
```

`Rs` is **not** an operand of that equation. It selects the bias set, the
feedback mode, the rounding/overflow mode, and the post-convert ACC action.
The ACC is an input; the convert state is an output; the optional clear/swap
is a post-convert side effect.

**Claim:** with `Rs[0]=1`, a convert does not modify the ACC values it read.
**Class:** `SPEC_CONFIRMED` (the only specified ACC write is the optional
clear/swap, which is disabled by `Rs[0]=1`).

### 2.3 Convert state and the previous-convert state

PRM §2.1 (p.3) lists three register sets: accumulator, convert state, and
bias registers. The convert state stores the result after the activation
function is applied. PRM §2.3.2 (p.15) describes it as a 32x32 array that
mirrors the accumulator cell for cell.

PRM Chapter 5 (p.32–34) adds feedback: a convert can use the **previous
convert state** as the scale or output-bias input, combined with the bias
register value by min/max. The first convert sets `Rs[0]=1` so the ACC is
not cleared; the second convert uses that same ACC and the previous convert
state as feedback. PRM §5.1:

> For the first convert instruction prior to the feedback, set the Rs[acc]
> field ... to keep the accumulator values from being cleared.

> For the second convert instruction using the values of the prior instruction
> as feedback, set the Rs[fb_dest] and Rs[fb_op] fields. Ensure the correct
> bias register index is selected.

The PRM's code example (p.33–34) is exactly a two-convert sequence over one
ACC. The fragment below is simplified from the PRM's address setup; the
converts and the retain bit are verbatim in substance:

```c
Q6_bias_mxmem2_A(bias_start_addr);       // bias set 0
int retain_acc = 1;
Q6_cvt_hf_acc_R((extra_precision << 8) | retain_acc);  // convert #1, ACC retained
Q6_bias_mxmem2_A(bias_start_addr + 256); // bias set 1 (index in low bits)
Q6_cvt_hf_acc_R((bias_idx << 12) | ...); // convert #2, same ACC
```

No MMA appears between the two converts. The example uses feedback and a bias
load between the converts, but it demonstrates the key fact: **two converts
may consume one accumulator**.

**Claim:** a second convert can read the same ACC after the first convert
without a new MMA.
**Class:** `SPEC_CONFIRMED` (PRM Chapter 5 example and §5.1). The example does
not contain a store between the two converts; that is a separate composition
question (§2.5).

### 2.4 Bias sets and `Rs[13:12]`

PRM §2.3.1 (p.13):

> Each output channel of the accumulator is associated with a bias register
> (64 bits). A set of bias registers include all bias registers of the 32
> output channels. There are four sets of bias registers available to the
> convert transfer instruction.

> A convert transfer instruction selects one of the four sets of bias
> registers to use, and the bias cells 0 through 31 within the set are
> applied to the corresponding accumulator output channel values.

> The bias register must be loaded from VTCM prior to the convert transfer
> instruction. ... The bias load instruction can be either before or after
> the multiplication instructions, as long as it is before the convert
> transfer instruction.

Table 6 (p.15) defines `Rs[13:12]` as **Bias sel: Select the bias register set
for this instruction.** The bias load instruction (Table 3, p.14) uses the low
two bits of its address register as the set index; the address must be 256-byte
aligned. A set is 256 bytes = 32 output channels x 64 bits.

The 64-bit cell layout (Figure 15 / Table 4, p.14) is:

```text
lo32 = output_bias_fp16 << 16 | scale_fp16
hi32 = shape[10:8] << 8 | output_bias_extra4[7:4] << 4 | scale_extra4[3:0]
```

The 256-byte row stores the 32 low words first (bytes 0..127), then the 32
high words (bytes 128..255), not 32 interleaved 64-bit words. This is the
layout in `host_tests/hmx_acc_residual_rpc/bias20.h` and in the E-3D bias row.

**Claim:** the two R2 passes can use two different, preloaded bias sets, and
`Rs[13:12]` selects them per convert.
**Class:** `SPEC_CONFIRMED`.

### 2.5 Store: read convert state, do not modify it

PRM §2.4 (p.17):

> The write convert state instruction stores the FP16 data in convert state
> age to VTCM at the specified address. It does not modify the data in the
> convert state.

The store address must be 2 KiB aligned; `Rt` carries the spatial mask. The
store maps the 32x32 convert state to the pair-interleaved VTCM layout:

```text
store_u16(s,o) = (s >> 1) * 64 + o * 2 + (s & 1)
```

**Claim:** a store between convert #1 and convert #2 does not change the ACC
or the convert state, so it cannot invalidate the second convert's input.
**Class:** `SPEC_CONFIRMED` for non-modification. The PRM does **not**
explicitly specify the convert-state age/queue protocol relative to a store,
so the instruction ordering guarantee is `SPEC_SILENT`; the safe sequence is
the program-order sequence convert #1 -> store #1 -> convert #2 -> store #2.
The existing device mode 22 exercises that sequence and its first store is
preserved (§3.2).

### 2.6 Ordering, packets, dependencies

The PRM states:

* activation and weight must be sequential in the program and together form a
  multiply instruction (§2.2, p.4);
* a bias load may be before or after the multiply, as long as it is before the
  convert (§2.3.1, p.13);
* a convert reads the ACC and writes the convert state, and optionally clears
  the ACC (§2.3, p.11-12);
* a store reads the convert state and does not modify it (§2.4, p.17).

The PRM text reviewed here does not publish a hazard table, latency, or a
formal packet rule for convert -> store -> convert. It does expose "Convert State Age 0" in Figure 16
and the feedback path requires the previous convert state, which implies at
least a previous/current distinction in the convert state. The store
instruction has no visible age-select field (Table 8), so the store reads the
current convert state. This is consistent with the sequence, but it is not a
separate guarantee.

**Class:** `SPEC_SILENT` for a formal ordering guarantee; `DEVICE_SUPPORTED`
for the observed mode 22 sequence (§3.2). This is the main item the future
device test must re-check, not a spec blocker.

### 2.7 Evidence summary

| Claim needed by R2 | PRM | Class |
| --- | --- | --- |
| ACC is 32x32x37-bit, primary/secondary pair | §2.2.3 p.9 | `SPEC_CONFIRMED` |
| MMA writes primary | §2.2.3 p.10 | `SPEC_CONFIRMED` |
| Convert reads ACC, writes convert state | §2.3 p.11-12 | `SPEC_CONFIRMED` |
| `Rs[0]=1` retains ACC values | Table 6 p.16 | `SPEC_CONFIRMED` |
| `Rs[0]=1` does not swap the primary bank | Table 6 p.16 (complement of `0`) | `SPEC_IMPLIED` |
| Only convert reads the ACC values (clear is the other ACC operation) | §2.2.3 p.10, §2.5 p.19 | `SPEC_CONFIRMED` |
| Two converts can consume one ACC without a new MMA | Ch.5 p.33-34, §5.1 | `SPEC_CONFIRMED` |
| 4 bias sets; per-channel 64-bit cells | §2.3.1 p.13 | `SPEC_CONFIRMED` |
| `Rs[13:12]` selects the bias set | Table 6 p.15 | `SPEC_CONFIRMED` |
| Bias load may precede the convert, even after the MMA | §2.3.1 p.13 | `SPEC_CONFIRMED` |
| Store reads convert state, does not modify it | §2.4 p.17 | `SPEC_CONFIRMED` |
| A store between two converts is safe | not stated explicitly | `SPEC_SILENT` + `DEVICE_SUPPORTED` at mode 22 |
| Convert #1 rounding cannot affect convert #2 with feedback off | Figure 11 + Table 6 | `SPEC_CONFIRMED` |
| Feedback uses the previous convert state | Ch.5 p.32-34 | `SPEC_CONFIRMED` |
| Negative result is clamped downstream of shape | not in PRM | `DEVICE_SUPPORTED` (E-3D; stage `DEVICE_UNVERIFIED`) |
| HVX FP16 subtract is available | HVX ISA (`vsub.hf` / `Q6_Vhf_vsub_VhfVhf`) | documented; rounding/subnormal not categorically documented (§8.3) |

## 3. Can two converts read the same ACC?

### 3.1 Spec answer

**Yes, by specification.** The chain is:

1. The MMA writes the primary ACC.
2. Convert #1 with `Rs[0]=1` reads the primary ACC, writes the convert state,
   and retains the ACC values (no clear, no swap).
3. Convert #2 with `Rs[13:12]=1` and `Rs[0]=1` (or `0`; see below) reads the
   same primary ACC and writes the convert state again. No MMA is needed.
4. The PRM's own Chapter 5 example performs two converts on one ACC with
   `Rs[0]=1` on the first; the second uses the same ACC plus feedback.

The second convert may clear the ACC after it runs (`Rs[0]=0`). That is safe
because the clear is a post-convert action; the convert state is already
written, and the store reads the convert state, not the ACC. The recommended
R2 encoding is therefore:

| convert | `Rs` | meaning |
| --- | --- | --- |
| #1 | `0x0001` | bias set 0, retain ACC, no feedback, RN16, overflow to inf |
| #2 | `0x1000` | bias set 1, clear+swap after the convert, no feedback |
| #2 alternative | `0x1001` | bias set 1, retain ACC after the convert |

`0x1000` is the PRM's literal no-feedback recipe (`bias_sel` only, all other
bits zero) and has an existing device precedent in mode 22's second convert.
`0x1001` is the user's candidate and differs only in the post-convert ACC
state; the two must store identical values. The CPU state model checks both
(§6).

`0x1001` is not a guessed value. Table 6 puts the bias set in bits [13:12]
and the retain bit in bit [0]; all other bits stay zero per the PRM's
no-feedback recipe. `0x1001` sets exactly those two documented bits. It is a
**PRM-confirmed encoding**, but the exact R2 sequence with `0x1001` has not
been executed on a device.

### 3.2 What the existing device record already shows

The existing residual probe (`host_tests/hmx_acc_residual_rpc/probe_dsp.c`)
already contains the R2 control skeleton. Modes 0–11 run:

```c
mxclracc;
MMA (one packet);
bias = mxmem2(set0);
bias = mxmem2(set1);
cvt.hf = acc(0x0001);          // bias set 0, retain
mxmem(hi) = cvt;               // store #1
cvt.hf = acc(second_rs);       // bias set 1, e.g. 0x1000 or 0x1004
mxmem(residual) = cvt;         // store #2
```

Recorded facts relevant to R2:

| record | sequence | observation | what it shows | what it does not show |
| --- | --- | --- | --- | --- |
| mode 22, battery 1 | one MMA, `Rs=1`, store, `Rs=0x1000` (set 1, scale 64, bo -64), store | `hi=0x5400`, `residual=0x0000` | the first store was not clobbered by the second convert in this configuration | the second store would also be `0x0000` if the ACC had been destroyed (`0*64 - 64` clamps to 0), so it does **not** prove retention |
| mode 0, battery 3 | one MMA, `Rs=1`, store, `Rs=0x1004` (set 1, shape 4, bo +65504, feedback min), store | `residual=0x0000` | the 0 is consistent with a retained negative term; a destroyed ACC with working feedback (previous=0) also gives 0, so this row does not distinguish retention | feedback is `NOT_CONFIRMED`; the mode-0 detail is a session record, not a durable raw transcript; no positive second store |
| E-3D, mode 44 | one MMA, one convert, one set with per-channel shapes 0/4 | `out[2]=0x0000`, `out[3]=0x5420` | `shape 4` works on a negative ACC; the clamp is downstream of shape | no second convert; no two-bias-set sequence |

**Important correction to a loose reading of the old README.** The README's
Phase E-2 table says "2-CVT chains retain the accumulator". Mode 22 is
*consistent with* retention, but its second store cannot discriminate
retention from destruction because the -64 output bias makes both paths
clamp to 0. The first store does prove that the second convert did not
overwrite the first store. The record therefore supports **store ordering**
and **consistency with retention**, not a positive proof of retention.

No recorded mode has a **positive second store from the same ACC**. Modes 1
and 6 (the exact shape-4 / positive-switch variants) were not in the
battery. That is the gap the pre-registered S1 control fills.

### 3.3 What is not proven

1. A second convert producing a **positive** FP16 store from the same ACC.
2. The R2 configuration specifically: bias set 0 shape 0 then bias set 1
   shape 4, both `bo=0`, both stores 2 KiB apart.
3. The store/convert ordering at the R2 configuration (mode 22 is close, but
   its second bias row is shape 0 / bo -64).
4. HVX `vsub.hf` subnormal behaviour (§8.3).
5. Full-tile, real-data behaviour.

None of these is a spec contradiction. They are device-verification items.

## 4. Bias set 0/1 control

### 4.1 Bias rows

Two 256-byte rows are prepared in VTCM:

| set | address | scale | shape | input bias | output bias | use |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `base + 8192` | `0x5400` (+64) | 0 (`x`) | 0 | 0 | convert #1, positive half |
| 1 | `base + 8448` | `0x5400` (+64) | 4 (`-x`) | 0 | 0 | convert #2, negative half |
| 2 | `base + 8704` | `0x5000` (+32) | 0 (`x`) | 0 | 0 | S1 control only |

The per-channel cell for every output channel is identical inside a set:

```text
lo32 = 0x00005400          (bo = +0, scale = +64)
hi32 = 0x00000000          (shape 0)
hi32 = 0x00000400          (shape 4)
```

All three addresses are 256-byte aligned, as required by the bias load. The
rows are written before the first bias load and are not modified by the
converts.

### 4.2 `Rs` encodings

| field | convert #1 | convert #2 (recommended) | convert #2 (retain variant) | S1 control #2 |
| --- | --- | --- | --- | --- |
| `Rs[13:12]` bias sel | 0 | 1 | 1 | 2 |
| `Rs[8]` rnd | 0 | 0 | 0 | 0 |
| `Rs[6]` ovf | 0 | 0 | 0 | 0 |
| `Rs[4]` fb op | 0 | 0 | 0 | 0 |
| `Rs[3:2]` fb dest | 0 | 0 | 0 | 0 |
| `Rs[0]` acc | 1 (retain) | 0 (clear+swap after) | 1 (retain) | 0 |
| value | `0x0001` | `0x1000` | `0x1001` | `0x2000` |

The PRM's no-feedback rule is "`Rs[bias_sel]` = the bias register index; set
the rest of the register to 0". `0x0001` is the exception required by R2:
`Rs[0]=1` is the retain bit, not a feedback bit. `0x1000` is exactly the
literal recipe. `0x1001` adds only the documented retain bit. The PRM's
warning "only the necessary bits" is satisfied by all three.

### 4.3 Preload both sets vs reload between converts

Both are spec-legal:

| | preload set 0 and set 1 before convert #1 | reload set 1 between convert #1 and #2 |
| --- | --- | --- |
| instructions between store #1 and convert #2 | none | one 256-byte bias load |
| bias sets resident | 2 of 4 | 1 of 4 |
| VTCM rows | 512 bytes | 256 bytes (if the row is rewritten) or 512 bytes (two rows) |
| per-tile cost | 0 extra loads after the prologue | 1 extra 256-byte load per tile |
| ordering risk | lowest (no instruction between store and second convert) | the load must complete before convert #2; the PRM allows it, but it serializes |
| PRM precedent | §2.3.1 allows loads before the multiply | Chapter 5 example and mode 10 load set 1 between converts |

**Recommendation for the first device test: preload both sets.** It keeps the
sequence convert #1 -> store #1 -> convert #2 -> store #2 contiguous, which is
the sequence the ordering question is about. It also lets a real kernel hoist
both loads out of the tile loop when the weights and bias are constant. The
reload variant is a fallback if bias-register pressure matters; it does not
change the arithmetic.

## 5. State transitions

The table is the R2 sequence with `Rs #2 = 0x1000` (clear after the second
convert). `+x` is the E-3D fixture `x = (1+2^-6)^2 = 4225/4096`; `64x =
66.015625`.

| step | primary ACC | secondary ACC | convert state | previous cvt | bias sets | output |
| --- | --- | --- | --- | --- | --- | --- |
| after MMA | `ACC(x)` | stale/zero | invalid | invalid | rows 0/1 loaded | none |
| convert #1 `0x0001` | **unchanged** | unchanged | `cvt0 = 64*shape0(ACC)` | `cvt0` | set 0 | none |
| store #1 | unchanged | unchanged | `cvt0` unchanged | `cvt0` | set 0 | `pos = clamp(RN16(cvt0))` |
| convert #2 `0x1000` | read, then cleared | old primary (now secondary) = 0; old secondary becomes primary | `cvt1 = 64*shape4(ACC)` | `cvt0` becomes previous | set 1 | none |
| store #2 | old secondary (stale; 0 in this tile) | 0 (old primary cleared) | `cvt1` unchanged | `cvt0` | set 1 | `neg = clamp(RN16(cvt1))` |
| HVX | - | - | - | - | - | `recon = RN16(pos - neg)` |

With `Rs #2 = 0x1001`, the last two rows keep the ACC values and the primary
bank instead of clearing/swapping. The stores are identical; only the
post-convert ACC state differs. A later tile would need an explicit
`mxclracc` (or a `0x1000` convert) before reusing the ACC.

For the S1 control, convert #2 uses set 2 (`+32`, `Rs=0x2000`). The expected
words are `0x5420` (first, `64x`) and `0x5020` (second, `32x`), both positive.
A zero second word means the ACC was destroyed; `0x5420` means set 2 was not
selected and the hardware fell back to set 0. This is how S1 separates the
two failure modes without relying on a clamp.

## 6. CPU state model and tests

### 6.1 Model

`hmx_signed_output_oracle.c` §12 adds an independent state machine:

* two 32x32 ACC banks, a primary index, and an `acc_valid` flag;
* the current and previous convert-state arrays (raw 20-bit value and RN16
  bits), with explicit `cvt_valid` / `cvt_prev_valid`;
* four bias sets, each 32 per-channel cells (`scale_bits`, `shape`,
  `obias_bits`), decoded through the same `bias20.h` helpers as the probe;
* two VTCM output regions, written in the PRM store layout
  `(s>>1)*64 + o*2 + (s&1)`;
* an ACC-read snapshot per convert, so "same ACC" is checked over all 1,024
  cells, not a single representative;
* counters for MMA, convert, store, and region writes;
* fault-injection switches for ACC destruction, forced bias-set lookup,
  forced feedback, rounding mode, and clamp placement.

The model implements the PRM equation plus the **separate, labelled**
empirical clamp (`negative stored value -> 0x0000`). It does not model
overflow to maxnorm, min/max feedback arithmetic beyond the contamination
test, NaN payloads, or HVX subnormal flushing. Those are outside R2's
registered conditions.

### 6.2 Required cases

Run by `scripts/run_hmx_signed_output_oracle.ps1`; all 63 R2 checks pass, and
the full oracle is 218 checks / 0 failures.

| case | model result | detector result |
| --- | --- | --- |
| positive ACC | `pos=0x5420`, `neg=0x0000`, `recon=0x5420` | `R2_ACC_REUSE_SUPPORTED` (simulated) |
| negative ACC | `pos=0x0000`, `neg=0x5420`, `recon=0xD420` | `R2_ACC_REUSE_SUPPORTED` (simulated) |
| ACC = 0 | both buffers 0, `recon=+0` | both-zero lane passes |
| ACC retention success | both converts' 1,024-cell ACC snapshots are identical; one MMA, two converts, two stores | `R2_ACC_REUSE_SUPPORTED` (simulated) |
| ACC destroyed after store #1 | second store 0; S1 control 0 | `R2_ACC_DESTROYED` |
| bias set selection failure | S1 second `0x5420` instead of `0x5020`; S2 `neg(s0)=0x5420`, `neg(s1)=0` | `R2_BIAS_SWITCH_FAILED` |
| feedback accidentally enabled | Rs with `fb_dest!=0` fails the host contract; a forced feedback leak with the registered Rs changes `neg(s1)` from `0x5420` to a different word | `R2_HOST_CONTRACT_FAILED` (Rs) or `R2_UNEXPECTED_OUTPUT` (output leak) |
| output bias non-zero | `neg(s1)=0x5460` instead of `0x5420` | `R2_UNEXPECTED_OUTPUT` |
| positive/negative store overlap | second store overwrites the first; region write counts become 4/0 | `R2_STORE_INTERFERENCE` |
| one-ULP ceiling rounding | S1 second `0x5021`, S2 `pos(s0)=0x5421`, `neg(s1)=0x5421` | `R2_ROUNDING_MISMATCH` |
| corrupted word | `0x5420 -> 0x5520` | `R2_UNEXPECTED_OUTPUT` |
| host contract (extra MMA/convert/store, wrong Rs) | counters or Rs differ | `R2_HOST_CONTRACT_FAILED` |
| negative-ACC lane leaked into the positive buffer | `s2_pos[1]=0x5420` with a correct `s2_pos[0]` | `R2_UNEXPECTED_OUTPUT` |

The nominal output line is:

```text
r2_cpu_model nominal=R2_ACC_REUSE_SUPPORTED acc_destroyed=R2_ACC_DESTROYED
             bias_switch=R2_BIAS_SWITCH_FAILED
             store_interference=R2_STORE_INTERFERENCE
             rounding=R2_ROUNDING_MISMATCH
             unexpected=R2_UNEXPECTED_OUTPUT
             contract=R2_HOST_CONTRACT_FAILED
r2_cpu_model_checks=63
r2_device_verdict=NOT_RUN
```

`R2_ACC_REUSE_SUPPORTED` in that line is a **classifier test on a simulated
observation**, not a device result. The oracle prints `r2_device_verdict=
NOT_RUN` unconditionally so the two cannot be conflated.

### 6.3 What the CPU model does and does not prove

It proves that the decision procedure does not accept the nominal pattern
when the ACC is destroyed, when the bias set is not switched, when the store
regions collide, when the rounding is off by one ULP, when the output bias is
non-zero, or when the Rs/count contract is wrong. It also proves that
`0x1000` and `0x1001` produce the same stores while leaving different
post-convert ACC states.

It does **not** prove that the device retains the ACC, that the device's
convert state survives the store, or that the device's second convert sees
the same value. Those are exactly the pre-registered device checks.

**Feedback caveat.** The output-based feedback detector works when the clamp
is at the store (the previous convert state is still negative and min-feedback
changes the second output). If the clamp is inside the convert, the previous
state is already 0 and the output detector is masked. The host contract's Rs
check is the robust guard: the registered `Rs #2` must be `0x1000` or `0x1001`
with `Rs[3:2]=0`. This limitation is printed as
`r2_feedback_output_detector=MASKED_AS_EXPECTED` and is not hidden.

## 7. G8/G9 1,024-cell accuracy

The existing G8 tile (seed `0xE3C0117E5EEDF00D`, 32x32x32, 1,024 cells) is
reused unchanged. The three variants are:

1. **Plain FP16**: one MMA, direct signed RN16 store.
2. **B3 direct**: three MMA packets into one ACC, direct signed RN16 store.
3. **B3 + R2**: three MMA packets into one ACC, convert shape 0 -> store,
   convert shape 4 -> store, HVX subtract.

The G9 `t8_split_store()` already implemented variant 3's encoding; the new
`t8_r2_metrics()` applies it in the R2 order and reports the full comparison
set. The numbers are CPU-model, not device measurements.

| variant | mean abs error | max abs error | ratio vs plain | bit match vs direct | numeric match |
| --- | --- | --- | --- | --- | --- |
| Plain FP16 | `1.307244e-03` | `1.113717e-02` | `1.000000` | `1024/1024` | `1024/1024` |
| B3 direct RN16 | `7.084104e-04` | `7.217338e-03` | `1.845321` | `1024/1024` | `1024/1024` |
| B3 + R2 | `7.084104e-04` | `7.217338e-03` | `1.845321` | `1024/1024` | `1024/1024` |

Other counts:

```text
improved=427 equal=597 worsened=0
neg_zero_diff_plain=0 neg_zero_diff_b3=0
subnormal_plain=2 subnormal_b3=2
overflow=0 inf=0 nan=0
both_nonzero=0 one_sided=1023 both_zero=1
intermediate_rounding=0
r2_state_matches_g9=1024
```

The previously reported `1.845321x` is preserved **exactly**; it was not
retuned. The R2 path is bit-for-bit identical to the B3 direct RN16 store on
this tile. That is the expected result: with `bo=0`, exactly one half is
non-zero, so there is one RN16 rounding and the HVX subtraction is exact.
The split recovers the sign; it does not add precision.

### 7.1 Arithmetic assumptions

* ACC values are the G8 float-chain values (`gT8P`, `gT8B`), not device ACC
  dumps. The B3-vs-plain comparison is a stored-domain encoding comparison.
* The CPU model uses double intermediates and the shared `fp16_util.h`
  binary16 helper: round-to-nearest, ties-to-even, subnormals at 2^-24,
  overflow to infinity above 65520, signed zero preserved.
* The runner compiles with `-ffp-contract=off`; no FMA contraction.
* The CPU model does not flush subnormals (no FTZ/DAZ). The HVX subtract is
  modelled as IEEE binary16 RN-even. The HVX caveat is in §8.3.
* `bo=0` on both halves. `bi=0` on both halves. One shared positive scale
  (+64). Both stores are 2 KiB apart.
* NaN/Inf are out of scope; the tile contains none (counts above). The oracle
  sweep separately exercises the subnormal and overflow bands.

### 7.2 Negative zero, subnormal, overflow, Inf/NaN, intermediate rounding

* **Negative zero**: a direct signed RN16 store can produce `0x8000` for a
  negative zero; the split produces `+0` because both halves are non-negative
  and `0 - 0 = +0`. On the G8 tile the difference count is 0 for both plain
  and B3. The oracle's wider sweep has 65 such bit differences (numerically
  equal). A bitwise device comparator must tolerate this.
* **Subnormal**: two cells per variant. The CPU model reconstructs them
  exactly (bit match 1024/1024). Whether HVX `vsub.hf` preserves subnormals is
  not fully documented; the first device test avoids subnormals by using
  `66.0` / `33.0` (normal FP16).
* **Overflow**: none on the tile. The oracle's `e ∈ [-32,11]` sweep covers
  the overflow band; the convert equation with `Rs[6]=0` overflows to
  infinity, not maxnorm, so an overflow would appear as `0x7C00`/`0xFC00`,
  not a silent clamp. Count 0 here.
* **Inf/NaN**: none on the tile. NaN payload preservation is out of scope;
  the R2 precondition excludes NaN in the ACC.
* **Intermediate convert rounding**: `both_nonzero=0`, so the two stores never
  both carry a rounded value; the subtraction is exact and there is no double
  rounding. If a non-zero output bias makes both halves non-zero, the result
  becomes `2y` (large bias) or `y+bo` (small bias); the oracle §11 pins both
  failure modes. `bo=0` is a configuration invariant, not a data-dependent
  hope.

## 8. VTCM, store path, and HVX

### 8.1 VTCM map

The existing 12 KiB probe layout already fits R2 without resizing:

| region | offset | size | alignment | contents |
| --- | --- | --- | --- | --- |
| `act` | `+0` | 2048 B | 2 KiB | activation crouton |
| `wgt` | `+2048` | 2048 B | 2 KiB | weight crouton |
| `out_pos` | `+4096` | 2048 B | 2 KiB | convert #1 store |
| `out_neg` | `+6144` | 2048 B | 2 KiB | convert #2 store |
| `bias` set 0 | `+8192` | 256 B | 256 B | scale +64, shape 0, bo 0 |
| `bias` set 1 | `+8448` | 256 B | 256 B | scale +64, shape 4, bo 0 |
| `bias` set 2 | `+8704` | 256 B | 256 B | S1 control: scale +32, shape 0 |
| total used (S1 with set 2) | | 8960 B | | 3328 B free of 12288 B |
| S2-only (sets 0/1) | | 8704 B | | 3584 B free of 12288 B |

All store addresses are 2 KiB aligned (`Rs[31:11]` address, Table 8). All bias
addresses are 256 B aligned (Table 3). The two output regions are disjoint;
the oracle §12.15 checks the offsets/alignment/disjointness and the 1,024-cell
store bijection.

### 8.2 Store layout is an activation crouton

The store layout is:

```text
store_u16(s,o) = (s >> 1) * 64 + o * 2 + (s & 1)
```

The activation crouton layout is:

```text
act_u16(s,i)   = (s >> 1) * 64 + i * 2 + (s & 1)
```

They are the same formula with the channel index renamed. Therefore the
32x32 signed result produced by the HVX subtract is already in the layout the
next MMA expects for a 32-input-channel activation crouton. The result can be
written in place over `out_pos` (or to a third 2 KiB buffer) and fed to the
next layer without a repack. This is an implementation advantage of the
32-channel tile, not a numerical claim.

### 8.3 HVX `vsub.hf`

The HVX ISA provides half-precision vector subtract (`vsub.hf`, intrinsic
`Q6_Vhf_vsub_VhfVhf`). The v79/v81 floating-point text reviewed here identifies
the half-precision add/subtract operations, but it does not give a categorical
FTZ/DAZ or rounding statement for every binary16 case. The CPU model therefore
assumes IEEE-754 round-to-nearest-even with subnormals preserved; that is a
model assumption, not a hardware guarantee. A 32x32 tile is 2,048 bytes = 16
HVX vectors of 64 FP16 lanes. The R2 post-processing is:

* 16 loads from `out_pos`;
* 16 loads from `out_neg`;
* 16 `vsub.hf` (one per 64-lane vector);
* 16 stores of the signed result (in place or to a third buffer).

The pos and neg buffers are disjoint, so the in-place form is safe per vector
(load both vectors, subtract, store over pos). No output channel is
duplicated; the 32 lanes of a vector are the 32 output channels of one
spatial pair, and the two spatials are the even/odd half of the vector.

**Subnormal caveat.** The HVX manual's "HVX unnormal" wording is not an IEEE
subnormal guarantee, and the indexed text does not give a categorical
FTZ/DAZ statement for every binary16 subnormal. The CPU model assumes IEEE
RN-even with subnormals preserved; the G8 tile's two subnormal cells are
therefore a CPU-model result, not a hardware guarantee. The first device test
uses `66.0` and `33.0` (normal), so the R2 ACC-reuse question is not exposed
to this ambiguity. A later kernel that can produce subnormal outputs must
measure `vsub.hf` on subnormal inputs/outputs before relying on the encoding.

**Negative zero.** `+0 - +0 = +0` in RN-even. The split cannot produce `-0`
from a zero ACC; a direct signed store could. This is the same ±0 caveat as
§7.2.

## 9. Static cost model

Unit: 1,024 logical outputs = one 32 spatial x 32 output-channel tile,
K=32, 3-product B3 synthesis. "Packet" = one activation+weight MMA packet.
"Convert" and "store" are one HMX instruction each (32x32 cells).

| item | Plain | B3 | R1 (duplicated channels) | R2 (retained ACC) |
| --- | --- | --- | --- | --- |
| MMA packets | 1 | 3 | 6 (2 physical tiles x 3) | **3** |
| MMA instructions | 2 | 6 | 12 | **6** |
| converts | 1 | 1 | 2 | **2** |
| stores (2 KiB) | 1 | 1 | 2 | **2** |
| bias sets resident | 1 | 1 | 1 | 2 |
| bias load bytes | 256 | 256 | 256 | 512 (prologue; hoistable) |
| output VTCM before HVX | 2 KiB | 2 KiB | 4 KiB | **4 KiB** |
| HVX `vsub.hf` (64-lane) | 0 | 0 | 16 | **16** |
| HVX loads / stores | 0 | 0 | 32 / 16 | **32 / 16** |
| output-channel duplication | no | no | yes (2x) | **no** |
| ACC recompute | n/a | none | none (two tiles) | **none** |

Conditions and caveats:

* R1's 6 packets assume the 32-output-channel ACC limit: 32 physical channels
  carry only 16 logical outputs when every output is duplicated, so 32 logical
  outputs need two tiles. If a future HMX variant had 64 output channels, R1
  would drop to 3 packets; the V81 PRM's ACC is 32 output channels.
* R2's 3 packets are the B3 synthesis, not a new MMA. The second convert reads
  the same ACC; no weight tile is duplicated.
* R2 is not free versus B3: it adds one convert, one store, one 2 KiB output
  buffer, and 16 HVX subtracts per 1,024 outputs. Versus R1 it saves three MMA
  packets (half the MMA work) and the duplicated weight tile, at the cost of
  one extra bias set (256 B).
* The convert/store path is not assumed cheaper than an MMA packet. The table
  counts instructions and bytes, not cycles. Convert latency, store bandwidth,
  bias-load latency, and HVX throughput are not measured. **No runtime or
  speedup is claimed.**
* E-3C2 remains `UNTRIGGERED`; this is a static feasibility model, not a
  performance experiment.

## 10. Verdict

### `R2_SPEC_FEASIBLE`

The spec-side conditions are met:

1. **ACC reuse:** `Rs[0]=1` retains the ACC; the PRM's Chapter 5 example runs
   two converts over one ACC without an intervening MMA.
2. **Bias-set switch:** four bias sets exist; `Rs[13:12]` selects one; both
   sets can be preloaded before the first convert.
3. **Store independence:** the store reads the convert state and does not
   modify it; the ACC is only accessed by converts.
4. **No second MMA:** nothing in the PRM requires an MMA between two converts
   of a retained ACC. The second convert reads the same primary ACC.
5. **Sign recovery:** E-3D established that the negative clamp is downstream
   of the shape, so the shape-0/shape-4 pair produces two non-negative
   buffers whose HVX difference is the signed value.

This is not `R2_SPEC_BLOCKED`: no PRM statement forbids any required step.
It is not `R2_INSUFFICIENT_SPEC_EVIDENCE`: the core operations (retain, bias
select, store non-modification) are explicitly specified. The remaining gaps
are device behaviour and one unspecified ordering detail, not missing
specification of ACC reuse.

The verdict is **not** based on the CPU model's PASS. The CPU model only
shows that the decision procedure can distinguish success from the registered
failure modes. The PRM text and the existing device modes are the basis.

### What would change the verdict

* `R2_SPEC_BLOCKED` if the PRM said a convert consumes/clears the ACC even
  with `Rs[0]=1`, or that a store consumes the convert state, or that a
  convert can only be issued once per MMA. None of those statements exists.
* `R2_INSUFFICIENT_SPEC_EVIDENCE` if the "convert reads the primary ACC"
  reading were contradicted, or if the two-convert Chapter 5 example were
  found not to apply to the no-feedback path. The example uses feedback, but
  the ACC-retention mechanism is the same; the no-feedback path is a subset.

## 11. Unresolved items and recommended next gates

1. **Minimal device test (pre-registered, not authorised).** S1 retention +
   bias-set control, S2 sign split. One build, one run, two subcases, 11
   readback slots. See [`hmx-v81-r2-device-prereg.md`](hmx-v81-r2-device-prereg.md).
2. **HVX subnormal behaviour.** Measure `vsub.hf` on subnormal inputs and
   results, or keep the first kernel in the normal FP16 range. Do not assume
   IEEE subnormal preservation from the current documentation.
3. **Store/convert ordering at the R2 configuration.** Mode 22 already
   preserves the first store, but its second bias row is shape 0. The S2
   first-store check is the R2-specific confirmation. If the first store is
   overwritten, the verdict is `R2_STORE_INTERFERENCE`, not a numerical
   mismatch.
4. **Clamp stage.** The clamp is downstream of shape but not localised between
   convert and store. R2's no-feedback path does not depend on the stage: both
   halves are non-negative, and the store clamp only removes the inactive
   half. The stage matters only for the output-based feedback detector (§6.3).
5. **Real-tile accuracy.** The 1,024-cell tile is a CPU model. Real-tile
   accuracy remains `INSUFFICIENT_EVIDENCE`.
6. **Cost/throughput.** The static model says R2 uses half the MMA packets of
   R1 and the same convert/store/HVX cell count. It does not say R2 is faster.
   A timed kernel remains E-3C2, `UNTRIGGERED`.
7. **R1 vs R2 selection.** R2 is strictly better on MMA count under the
   32-channel ACC limit and avoids duplicating the weight tile. R1 may still
   be simpler if a duplicated-channel layout is already available, but it
   pays 2x MMA. No timing or real-tile accuracy comparison exists.

## 12. Reproduction

```powershell
# CPU-only, no SDK, no ADB, no device.
.\scripts\run_hmx_signed_output_oracle.ps1
.\scripts\run_fp16x2_split_sim.ps1
.\scripts\verify.ps1 -Profile Fast
```

Expected keys:

```text
r2_cpu_model nominal=R2_ACC_REUSE_SUPPORTED acc_destroyed=R2_ACC_DESTROYED
             bias_switch=R2_BIAS_SWITCH_FAILED
             store_interference=R2_STORE_INTERFERENCE
             rounding=R2_ROUNDING_MISMATCH
             unexpected=R2_UNEXPECTED_OUTPUT
             contract=R2_HOST_CONTRACT_FAILED
r2_cpu_model_checks=63
r2_device_verdict=NOT_RUN
hmx_signed_output_oracle checks=218 fail=0
hmx_signed_output_oracle=PASS

r2_g9 cells=1024 plain_direct_mean_abs=1.307244e-03 ...
       plain_over_b3=1.845321 ... both_nonzero=0 ... intermediate_rounding=0
       r2_state_matches_g9=1024
r2_g9_verdict=R2_STATE_ENCODING_PRESERVED accuracy_change=none_vs_b3_direct
fp16x2_split_sim=PASS
```

The oracle and split-sim runners compile with gcc 15.2.0 (MSYS2 UCRT64,
`x86_64-w64-mingw32`) and clang 22.1.8; the oracle runner uses
`-ffp-contract=off`. Both compilers produced the same results.

## 13. Fast gate and repository state

At the time of this audit:

* branch: `experiment/hmx-acc-residual-feedback`;
* pre-change HEAD: `cb41b7c` (`test(hmx): evaluate shape-0/4 sign-separated
  output on CPU`), 7 commits ahead of
  `origin/experiment/hmx-acc-residual-feedback`;
* `cb41b7c` and `b0b7edf` were present and untouched;
* working tree was clean at the start;
* `scripts/verify.ps1 -Profile Fast` was run after the code changes:
  **PASS**, 543.1 s, `PASS=10 FAIL=0 SKIP=21`, exit 0. The run includes the
  R2 oracle (`checks=218 fail=0`, `r2_cpu_model_checks=63`), the split sim
  (`fp16x2_split_sim=PASS cases=115573`), and `host-fast-contract PASS`
  (60.8 s). The cold Gradle `unit-tests` step (466.4 s) dominated the
  runtime; the 1–3 minute target assumes a warm Gradle cache, and the
  longer runtime here is not a test failure. A duplicate confirmation run
  was cut off by the 600 s tool limit while re-running the same host tests;
  it did not report a failure, and the completed run above compiled and ran
  the final oracle (218 checks).

Changed files for this study:

* `host_tests/hmx_acc_residual_rpc/hmx_signed_output_oracle.c` — R2 state
  machine, fault injection, decision procedure;
* `host_tests/hmx_acc_residual_rpc/hmx_fp16x2_split_sim.c` — G9 R2 metrics;
* `docs/hmx-v81-r2-acc-reuse-feasibility.md` — this document;
* `docs/hmx-v81-r2-device-prereg.md` — future device pre-registration.

No DSP source, production QNN/Muon code, APK, checkpoint, logcat, device
identifier, or private corpus is part of this change. No push, no merge, no
PR state change; PR #14 remains Draft.

## 14. Bottom line

R2 is a coherent, spec-supported design: one B3 MMA, two converts over the
same retained 37-bit ACC, two 2 KiB stores, one HVX FP16 subtract. The second
convert needs no new MMA and no ACC recompute. The PRM specifies the retain
bit, the bias-set selection, and the store's non-modification; the existing
mode 22 shows the store/convert sequence works at one configuration; E-3D
shows the sign split's physics. What remains is device verification of a
**positive second store** from the same ACC and of the exact R2 bias rows.
That is a small, pre-registered experiment, not a redesign.
