# E-3D pre-registration: can the V81 HMX convert path store a negative FP16 value?

Status: **PRE-REGISTERED**, 2026-10-24, CPU-only. No DSP mode has been added,
no build has been made, no device has been contacted. This document freezes the
fixture, the expectations and the decision procedure **before** any measurement.
After the first device run the expectations in §8 may not be edited; a mismatch
is a result, not a reason to retune.

Scope of the question: *does the V81 HMX convert/store path preserve a negative
FP16 value at the store, or is the observed `0x0000` a general negative-value
clamp?* The experiment is designed to be decidable from four 16-bit store cells
produced by one MMA packet, one convert and one store.

## 0. Why another trial is needed

The existing record is consistent with two very different worlds:

| world | prediction for a negative convert result |
| --- | --- |
| 1. the store is signed; `0x0000` came from the *particular* modes | `0xD420` etc. stored verbatim |
| 2. every negative result is destroyed at/after the convert | always `0x0000` |

Every negative result measured so far sits inside world 2's prediction, and no
surveyed mode ever read back a cell whose *only* possible source was a negative
value produced from a positive accumulator. Mode 43 (`E3C1R`) came closest:
`results[1] = 0x0000` for `(s1,o0)`, but `0x0000` is exactly what the clamp law
predicts, so that cell cannot separate the worlds. Mode 16 is the only recorded
*contradiction* of a simple final-value clamp (its PRM value is **positive**,
`+61.984375 = 0x53C0`, yet the device stored `0x0000`), which is why the cause
analysis treats the negative-scale path separately
(`docs/hmx-v81-signed-output-cause-analysis.md`).

E-3D therefore adds exactly one thing the record lacks: **a cell whose expected
negative value is produced by the shaping function from a positive
accumulator**. If the store is signed, that cell is `0xD820`. If it is not, the
cell is `0x0000` *and* the positive twin cell is `0x5820`, which proves the
negation happened upstream and was then destroyed. Either outcome is decisive.

## 1. Fixture (frozen)

```text
a  = 1 + 2^-6 = 1.015625                 FP16 0x3C10 (exact)
x  = a * a   = 4225/4096 = 1.031494140625 (exact in binary; exact in the 37-bit ACC)
2x = 2.06298828125
64x  = 66.015625
128x = 132.03125
```

Both scaled magnitudes sit **exactly 0.25 ULP above** an FP16 grid point:

| value | lower grid point | ULP | distance |
| --- | --- | --- | --- |
| `66.015625` | `66.0` = `0x5420` | `2^-4` = `0.0625` | `0.015625` = 0.25 ULP |
| `132.03125` | `132.0` = `0x5820` | `2^-3` = `0.125` | `0.03125` = 0.25 ULP |

Consequences, all pinned in the CPU oracle:

* round-to-nearest gives `0x5420` / `0x5820`;
* round-toward-`+inf` gives `0x5421` / `0x5821`;
* round-toward-`-inf` gives `0x5420` / `0x5820` for positives and
  `0xD421` / `0xD821` for negatives;
* truncation toward zero is indistinguishable from RN at this offset (recorded
  as a limitation, not a discriminator).

The fixture is the same `q = 6` fixture used by modes 11–19, so the positive
anchor is directly comparable with the device-recorded value `0x5420` (mode 11
set-0 convert, commit `413d872` body: "Mode 11 (set0->set1, +64): both 0x5420").

## 2. Tile

One 32×32×32 K32 tile. Only input channel 0 is non-zero; only spatials 0 and 1
and output channels 0 and 1 carry data. `A(s,i)` is the activation, `W(o,i)`
the weight.

```text
A(0,0) = +a = 0x3C10        A(1,0) = -a = 0xBC10        all other A = 0
W(0,0) = +a = 0x3C10        W(1,0) = +a (variant A) or +2a = 0x4010 (variant B)
```

ACC is the exact dot product, so with only `i = 0` non-zero:

| cell | variant A | variant B |
| --- | --- | --- |
| `ACC(s0,o0)` | `+x` | `+x` |
| `ACC(s1,o0)` | `-x` | `-x` |
| `ACC(s0,o1)` | `+x` | `+2x` |
| `ACC(s1,o1)` | `-x` | `-2x` |

No rounding occurs anywhere before the convert: `x` is exact, `2x` is exact,
and the only rounding step is the final RN16 export.

### VTCM packing (PRM §2.2.1/§2.2.2, device-corroborated at store index 0/32)

```text
act_u16(s,i)   = (s>>1)*64 + i*2 + (s&1)
wgt_u16(o,i)   = (i>>1)*64 + o*2 + (i&1)
store_u16(s,o) = (s>>1)*64 + o*2 + (s&1)
```

| logical cell | VTCM u16 index |
| --- | --- |
| `A(0,0)` | 0 |
| `A(1,0)` | 1 |
| `W(0,0)` | 0 |
| `W(1,0)` | 2 |
| store `(s0,o0)` | 0 |
| store `(s1,o0)` | 1 |
| store `(s0,o1)` | 2 |
| store `(s1,o1)` | 3 |

The four returned slots are the first four `uint16` of the store crouton, i.e.
one 64-bit store word. The all-1024-cell pack/decode bijection and the
bounds/disjointness of the three structures are pinned by the CPU oracle
(`hmx_signed_output_oracle.c` §3) so that an index error cannot masquerade as a
numeric result.

### VTCM byte map (identical to mode 43)

| region | offset | size |
| --- | --- | --- |
| `act` | `+0` | 2048 B |
| `wgt` | `+2048` | 2048 B |
| `bias` set 0 | `+8192` | 256 B |
| `bias` set 1 | `+8448` | 256 B |
| `out` | `+10240` | 2048 B |
| total | | 12288 B (`VTCM_BYTES`) |

`out` is filled with `0xA5` before the store; a read of `0xA5A5` would mean the
store did not execute.

## 3. Bias row: one set, two shapes

The PRM (p.13) is explicit that the bias register is **per output channel**:

> A convert transfer instruction selects one of the four sets of bias registers
> to use, and the bias cells 0 through 31 within the set are applied to the
> corresponding accumulator output channel values.

So a per-channel Shape is legal, and the E-3D tile programs cell 0 with
`shape = 0` and cell 1 with `shape = 4` in the **same** set. This is the first
time any probe on this path does that: every existing mode uses `init_bias()`,
which writes one shape to all 32 cells.

One cell is a 64-bit register split across the 256-byte row as
`lo[o] = row + 4o`, `hi[o] = row + 128 + 4o` (PRM Fig. 13/15, Table 4):

| cell | `lo32` | `hi32` | decode |
| --- | --- | --- | --- |
| 0 | `0x00005400` | `0x00000000` | scale `+64` (`0x5400`), bo `+0`, shape 0 |
| 1 | `0x00005400` | `0x00000400` | scale `+64`, bo `+0`, **shape 4** |
| 2..31 | `0x00005400` | `0x00000000` | safe default (zero ACC → `+0`) |

`hi32[10:8] = shape`, `hi32[7:4] = bo extra4`, `hi32[3:0] = scale extra4`
(all extra4 nibbles 0). Cells 2..31 are programmed, not left as `0xA5`: a cell
left unprogrammed would make the row contents depend on the prefill.

## 4. Convert scalar (`Rs`)

PRM Table 6 (p.15):

| bits | field | value used | meaning |
| --- | --- | --- | --- |
| `[13:12]` | Bias sel | `0` | bias set 0 |
| `[8]` | Rnd | `0` | IEEE precision (not "extra four bits") |
| `[6]` | Ovf | `0` | overflow to ±infinity |
| `[4]` | Fb op | `0` | min (unused: no feedback) |
| `[3:2]` | Fb dest | `0` | **no feedback** |
| `[0]` | acc | `1` | retain accumulator values |

`Rs = 0x00000001`, exactly the value the mode-43 handler uses (`cvt.hf = acc(1)`
in the disassembled skel). The convert equation (PRM Figure 11) is

```text
cvt = s * shape(ACC + bi) + bo        then RN16, overflow per Rs[6]/USR[21:20]
```

`Rs` is not an operand of that equation, so no arithmetic term can differ
between `Rs = 1` and the PRM's literal no-feedback recipe `Rs = 0`. The only
difference is post-convert ACC state (retain vs clear+swap), which a single
convert cannot observe. **The trial therefore makes no claim about `Rs = 0` and
must not be read as evidence for or against it**; see §11 for why a twin trial
is excluded on cost grounds.

The `Rs = 1` choice also keeps the entire instruction sequence byte-identical to
the device-executed mode 43: `mxclracc`, one MMA packet, two bias loads (set 0
and set 1), one convert, one store. Only VTCM **contents** and the four returned
indices change. That is the whole point of the design.

## 5. Readback

`results[0..3]` = store u16 indices 0..3:

| slot | device cell | meaning |
| --- | --- | --- |
| `results[0]` | `(s0,o0)` | positive anchor, `shape 0`, `ACC = +x` |
| `results[1]` | `(s1,o0)` | `shape 0`, `ACC = -x` |
| `results[2]` | `(s0,o1)` | `shape 4`, `ACC = +x` (A) / `+2x` (B) |
| `results[3]` | `(s1,o1)` | `shape 4`, `ACC = -x` (A) / `-2x` (B) |

No IDL change: `sequence<long> results` is unbounded and the existing host
constant is 4. Values are returned zero-extended (`0x0000D420`, not a negative
`int`) and reported as exactly four hex digits. `0x0000` and `0x8000` are
distinct outcomes and must never be collapsed into a decimal `0`.

## 6. Two variants

The literal mission fixture is **variant A** (`W(o1) = +a`). It has a defect
for this purpose: `(s1,o0)` and `(s0,o1)` both expect `0xD420` under a signed
store and both expect `0x0000` under a clamp, so a **slot-1/slot-2 transposition
is invisible**. Mode 43 constrains a global 1↔2 swap (`0x0000` at slot 1 but
`0x5400` at slot 2), but not the combination "shape applied to the wrong output
channel + slots swapped", which produces the same four values.

**Variant B** (`W(o1) = +2a = 0x4010`) costs the same one MMA/convert/store and
makes all four expected cells distinct, so every permutation of the four slots
is visible. It is the recommended run.

Variant A is retained in §8 verbatim, because it is the fixture the mission
specified; if only A is authorised, the run is still decidable but a
transposed-slot result must be reported as `E3D_INCONCLUSIVE_UNMATCHED` rather
than interpreted.

## 7. What each cell can decide

| cell | expected (B, signed) | if `0x0000` | what it proves |
| --- | --- | --- | --- |
| `out[0]` | `0x5420` | precondition failure | tile/scale/bo/store at index 0 work; a wrong scale (32 or 128) moves this cell |
| `out[1]` | `0xD420` | predicted by the known law | **replication, not new evidence**: mode 43 stored `0x0000` at the same store cell with the same shape/scale and a negative ACC (`-64.0`); E-3D changes the magnitude only |
| `out[2]` | `0xD820` | a negative was created by `shape 4` and destroyed at/after the convert | **the decisive cell**: `ACC` is positive, so no "ACC cannot hold negatives" explanation applies |
| `out[3]` | `0x5820` | `shape 4` was not applied, or the clamp is before the shape | separates "shape applied then clamped" from "shape ignored" |

`out[2]` is the cell the whole design exists for. A negative `out[2]` cannot be
explained by clamping the accumulator, because the accumulator is positive; it
can only be explained by a sign-preserving store. Conversely `out[2] = 0x0000`
with `out[3] = 0x5820` proves the negation happened and the negative value was
then destroyed after the shaping function.

Device corroboration that `shape 4` itself works on a **positive** ACC with
positive scale: mode 17 (`+64`, `shape 4`, `bo +128`, `ACC=+x`) stored `0x53C0`
(= `62.0`), the RN16 of `-64x + 128 = 61.984375`, which is only reachable if the
shaped term was `-66.015625` (commit `d4d9935` body; oracle §1/§9). E-3D's
`out[2]` uses the same shape and a positive ACC, with `bo = 0` instead of
`+128`, so any deviation is attributable to the negative result, not the shape.

## 8. Frozen expectations (variant B — do not edit after measurement)

```text
out[0] out[1] out[2] out[3]
0x5420 0xD420 0xD820 0x5820    SIGNED
0x5420 0x0000 0x0000 0x5820    LATE_CLAMP        (clamp at/after the convert)
0x5420 0x0000 0x0000 0x0000    EARLY_CLAMP       (clamp at the ACC input + store clamp)
0x5420 0x0000 0x5820 0x0000    SHAPE_IGNORED     (shape 4 not applied per channel)
0x5420 0xD420 0x5820 0xD820    SHAPE_IGNORED_SIGNED
0x5420 0x0000 0xD820 0x8000    ACC_CLAMP_SIGNED  (clamp at ACC, signed store, -0 kept)
0x5420 0x5420 0x5820 0x5820    MAGNITUDE_STORE   (sign dropped, not zeroed)
0x5420 0xFBFF 0xFBFF 0x5820    NEG_MAXNORM       (negative saturates to -maxnorm)
0x5421 0xD420 0xD820 0x5821    ROUND_CEIL
0x5420 0xD421 0xD821 0x5820    ROUND_FLOOR
```

Verdicts, in this order:

| pattern | verdict |
| --- | --- |
| `SIGNED` | `E3D_SIGNED_STORE_CONFIRMED` |
| `LATE_CLAMP` | `E3D_CLAMP_DOWNSTREAM_OF_SHAPE` |
| `EARLY_CLAMP` | `E3D_CLAMP_AT_ACC_BEFORE_SHAPE` |
| `SHAPE_IGNORED` | `E3D_SHAPE_NOT_APPLIED_PER_CHANNEL` |
| `SHAPE_IGNORED_SIGNED` | `E3D_SHAPE_NOT_APPLIED_PER_CHANNEL` (store signed) |
| `ACC_CLAMP_SIGNED` | `E3D_SIGNED_STORE_WITH_ACC_CLAMP` |
| `MAGNITUDE_STORE` | `E3D_MAGNITUDE_STORE` |
| `NEG_MAXNORM` | `E3D_NEGATIVE_MAXNORM` |
| `ROUND_CEIL` | `E3D_ROUNDING_NOT_RN16` (toward `+inf`) |
| `ROUND_FLOOR` | `E3D_ROUNDING_NOT_RN16` (toward `-inf`) |
| anything else | `E3D_INCONCLUSIVE_UNMATCHED` |

All ten variant-B patterns were proven pairwise distinct (all 45 pairs differ in
at least one of the four cells) by the CPU oracle; the variant-A list below was
checked by hand and is also pairwise distinct. A tuple that matches none of them
is **not** interpreted; it is recorded verbatim and the trial is reported as
inconclusive.

Variant A (only if the literal fixture is authorised):

```text
0x5420 0xD420 0xD420 0x5420    SIGNED
0x5420 0x0000 0x0000 0x5420    LATE_CLAMP
0x5420 0x0000 0x0000 0x0000    EARLY_CLAMP
0x5420 0x0000 0x5420 0x0000    SHAPE_IGNORED
0x5420 0xD420 0x5420 0xD420    SHAPE_IGNORED_SIGNED
0x5420 0x0000 0xD420 0x8000    ACC_CLAMP_SIGNED
0x5420 0x5420 0x5420 0x5420    MAGNITUDE_STORE
0x5420 0xFBFF 0xFBFF 0x5420    NEG_MAXNORM
0x5421 0xD420 0xD420 0x5421    ROUND_CEIL
0x5420 0xD421 0xD421 0x5420    ROUND_FLOOR
```

## 9. Decision procedure (frozen)

1. **Precondition.** If `out[0] != 0x5420`, the run is `E3D_INCONCLUSIVE_TILE`
   and no statement is made about signedness. Record all four slots anyway.
   If `out[0] == 0xD420` and `out[2] == 0x5420`, the bias row was applied with
   shape 4 on channel 0 (or read from the wrong cell): record the tuple with
   the annotation `bias_shape_misapplied`. That is still
   `E3D_INCONCLUSIVE_TILE` for the signedness question, but it is a direct
   observation about the row.
2. **Exact match.** Compare the 4-tuple against §8 for the variant that was
   actually built. On a match, report the verdict and the raw slots.
3. **Single-cell fallback (pre-registered).** If the tuple matches nothing but
   `out[2] != 0x0000` and `(out[2] & 0x8000) != 0`, report
   `E3D_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY`: the store preserved a negative
   value (the only possible source is a sign-preserving store, since `out[2]`'s
   ACC is positive and shape 4 is device-confirmed), while `out[1]`/`out[3]`
   deviate from every registered model and are recorded as a separate finding.
   No further inference is made from those two cells.
4. **Otherwise** `E3D_INCONCLUSIVE_UNMATCHED`. Do not retry, do not re-tune, do
   not "explain" the tuple after the fact.

Nothing in this procedure depends on the value of `Rs`, the tile's zero cells,
or an assumed clamp stage.

## 10. Trial accounting and cost

| item | count |
| --- | --- |
| clean stamped V81 build | 1 |
| `-DeployAndRun` sessions | 1 (`attempts = 1`) |
| `metadata[7]` (HMX executed) | 1 |
| `metadata[8]` (trials) | 1 |
| `mxclracc` | 1 |
| MMA packets | **1** |
| bias loads | 2 (set 0 = E-3D row, set 1 = zeros; identical to mode 43) |
| converts | **1** |
| stores | **1** |
| returned cells | 4 |

"Trials = 1" is only honest while the counts above hold. If an implementer adds
a second convert (for example the `Rs = 0` twin) the run has **two** converts
and **two** stores, the four-cell table is void, and the trial count must be
reported as the number of convert/store pairs actually executed — not as 1.
The same rule applies to a second MMA packet for any reason.

## 11. Deliberate exclusions

1. **No `Rs = 0` twin.** `Rs` is not an operand of the convert equation, so the
   CPU model has exactly one value for both settings; the twin would test only
   post-convert ACC state, at the cost of a second convert+store (8 readback
   slots, or a second run) and a broken trial-count claim. It is excluded as
   non-decisional for the signed-store question.
2. **No `-0` cell.** With `bo = +0`, IEEE gives `s * (-0) + (+0) = +0`, so a
   `-0` cannot be forced through the convert equation at all. The 4-slot ABI is
   therefore unchanged and no IDL change is needed. The `ACC_CLAMP_SIGNED`
   pattern is the only registered route by which `0x8000` could legitimately
   appear, and it is reported as an anomaly at that point.
3. **No clamp-stage localisation.** With convert+store as the only observables,
   "clamp in the ACC", "clamp in the convert" and "clamp in the store" cannot be
   separated. E-3D only separates **before shape** from **after shape** (via
   `out[3]`), which is the strongest statement this ISA supports.
4. **No scale-vs-shape ordering test.** The scale is positive, so
   `s*shape(v)` and `shape(s*v)` are identical. Recorded as scoped out.
5. **No negative-scale coverage.** E-3D uses `s = +64` everywhere by design; the
   mode-16 negative-scale anomaly is a separate question and is not tested here.

## 12. Stop conditions

After the run: record the four raw slots and the verdict; stop. No retry, no
second session, no B3 expansion, no mode addition, no expectation change. An
`E3D_INCONCLUSIVE_TILE` or `E3D_INCONCLUSIVE_UNMATCHED` ends the investigation
until a *new* pre-registration is written. A `SIGNED` verdict does not by itself
authorise Path B or any production change; it removes one blocker and leaves the
real-tile accuracy question (`INSUFFICIENT_EVIDENCE`) untouched.

## 13. CPU-only pre-run evidence (2026-10-24)

`host_tests/hmx_acc_residual_rpc/hmx_signed_output_oracle.c`, run by
`scripts/run_hmx_signed_output_oracle.ps1` (also wired into
`run_host_contract_tests.ps1` before the Fast early return):

```text
hmx_signed_output_oracle checks=107 fail=0
hmx_signed_output_oracle=PASS
signed_oracle_clang=PASS (oracle exit 0, triple=x86_64-pc-windows-msvc19.44.35229)
signed_oracle=PASS
```

gcc 15.2.0 (MSYS2, `x86_64-w64-mingw32`) with
`-std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off`; clang 22.1.8 as the
second compiler. `-ffp-contract=off` is required because the mode-16 regression
pins an exact tie (`0x53BF`/`0x53C0`) that FMA contraction can move.

What the oracle pins, in order: the five existing PRM convert cases and their
device regression bits; fixture exactness (`a = 0x3C10`, `2a = 0x4010`,
`64x = 66.015625`, `128x = 132.03125`, both quarter-ULP offsets); the layout
functions and the all-1024-cell bijections; the per-channel bias row and its
byte offsets; the four discriminator cells for both variants and the
transposition degeneracy of A; the ten outcome patterns, their pairwise
distinctness for B and for the authorised variant A; the `Rs` field values; the two-convert split reconstruction over
2176 cases (`both_nonzero = 0`, `mismatches = 0`); and the mode 16/17
regression.

This section records CPU evidence only. It is not a device result and it is not
a substitute for the Fast gate.

## 14. Authorization

Device execution of E-3D requires explicit user approval (new mode, one build,
one device run). Nothing in this document authorises an implementation. The
existing statuses are unchanged by this pre-registration:
`B2_THREE_PRODUCT_CONFIRMED`, `E3C1R_LAYOUT_CONFIRMED_CLAMP`,
`BLOCKED_SIGNED_OUTPUT_FOR_CURRENT_CONVERT_PATH`, Path B
`CONDITIONALLY_POSSIBLE`, real-tile accuracy `INSUFFICIENT_EVIDENCE`, E-3C2
`UNTRIGGERED`.

## 15. Execution selection (2026-10-24, before any measurement)

The user authorised exactly one E-3D device run and selected **variant A** (the
literal mission fixture, `W(o1) = +a = 0x3C10`) with the four slots fixed
before implementation:

```text
out[0] out[1] out[2] out[3]
0x5420 0xD420 0xD420 0x5420    SIGNED            (expected)
0x5420 0x0000 0x0000 0x5420    LATE_CLAMP        (registered alternative)
```

The operative frozen table for the run is therefore the **variant-A block of
§8**; variant B stays pre-registered but is not built. `Rs = 0x00000001` is
used, `bo = +0`, `scale = +64`, bias cells 0/1 = shape 0/4, one MMA packet,
one convert, one store, four returned cells.

Two consequences are recorded in advance, not discovered afterwards:

1. Variant A is transposition-degenerate at slots 1/2 (`0xD420` vs `0xD420`),
   so a readback that differs from `SIGNED` only by a `slot1 <-> slot2` swap
   cannot be interpreted. Such a tuple matches no §8 row and is already
   `E3D_INCONCLUSIVE_UNMATCHED` under §9.4; it is not retried.
2. `LATE_CLAMP` differs from `SIGNED` in slots 1/2 only, which A can still
   see, because both negative cells go to `0x0000` while both positive cells
   stay `0x5420`. The A table's ten rows were re-checked for pairwise
   distinctness by the CPU oracle (`e3dA.model_collisions = 0`).

This section selects a scope; it does not edit an expectation. No variant-A
tuple in §8 changed after this point.
