# Why does a negative HMX V81 convert result store as `0x0000`?

Status: CPU-only cause analysis, 2026-10-24. No DSP change, no device run, no
new mode. This document classifies every explanation that the recorded evidence
supports, refutes or leaves open. It does not replace the pre-registration in
`docs/hmx-v81-signed-fp16-discriminator-prereg.md` (E-3D), which is the only
registered way to decide the remaining question.

Source for the PRM statements: *Qualcomm Hexagon V81 HMX PRM*, 80-N2040-62
Rev. AA (`build/privacy-audit/hmx_v81_prm.txt`). Device values are quoted from
the commit bodies `413d872`, `d18e375`, `d4d9935` and the mode-42/43 records in
`host_tests/hmx_acc_residual_rpc/README.md`; none of them is re-derived here.

## 1. The observation, stated precisely

The convert equation (PRM Figure 11) is

```text
cvt = s * shape(ACC + bi) + bo        then RN16
```

with `shape` from Table 5 (`0 = x`, `4 = -x`, …) and one 64-bit bias cell per
output channel. Nothing in the PRM describes a sign clamp. Yet every
**negative** stored value measured on this path is `0x0000`:

| mode | bias configuration | `ACC` | PRM value before RN16 | stored | source |
| --- | --- | --- | --- | --- | --- |
| 12 | `s=-64`, `bo=0` | `+x` | `-66.015625` | `0x0000` | `413d872` |
| 13 | `s=0`, `bo=-4` | `+x` | `-4.0` | `0x0000` | `413d872` |
| 19 | `s=0`, `bo=-4` | `+x` | `-4.0` | `0x0000` | `d4d9935` |
| 32–36 | `s=+64`, `bo=-64.5…`, nibble sweep | residual | `-0.0029…-0.0576` | `0x0000` | README §Phase D |
| 42 | `s=+64`, `bo=0` | `-30.125` (CPU pre-clamp `0xcf88`) | `-30.125` | `0x0000` | README mode 42 |
| 43 | `s=+64`, `bo=0`, `shape 0` | `-1.0000` | `-64.0000162 → -64.0` (`0xD400`) | `0x0000` | README mode 43 |

Positive controls store correctly, including two that are arithmetically
identical to a zero-storing case:

| mode | bias configuration | `ACC` | PRM value | stored | source |
| --- | --- | --- | --- | --- | --- |
| 11 | `s=+64`, `bo=0` | `+x` | `+66.015625 → 66.0` | `0x5420` | `413d872` |
| 15 | `s=+64`, `bo=-4` | `+x` | `+62.015625 → 62.0` | `0x53C0` | `d4d9935` |
| 17 | `s=+64`, `shape 4`, `bo=+128` | `+x` | `+61.984375 → 62.0` | `0x53C0` | `d4d9935` |
| 18 | `s=+64`, `bo=-66` | `+x` | `+0.015625` | `0x2400` | `d4d9935` |

**Mode 16 is the anomaly.** It is `s=-64`, `bo=+128`, `ACC=+x`, so the PRM
value is `-66.015625 + 128 = +61.984375`, which rounds (RN-even, exact tie) to
`+62.0 = 0x53C0` — the *same stored bits* as modes 15 and 17. The device stored
`0x0000`. A rule of the form "a negative final value becomes `0x0000`" cannot
explain a positive final value, so the clamp law alone is an incomplete model of
the path.

## 2. Classification vocabulary

| label | meaning |
| --- | --- |
| `SPEC_CONFIRMED` | stated explicitly in the PRM text or tables |
| `DEVICE_SUPPORTED` | measured on the device; no device datum contradicts it |
| `CPU_MODEL_ONLY` | reproduces the data on the CPU but has no independent hardware evidence |
| `DEVICE_UNVERIFIED` | consistent with all evidence, but the discriminating observation has never been made |
| `REFUTED` | contradicted by PRM text, by device data, or by exact arithmetic |

## 3. Candidate classification

| # | candidate | class | evidence / reason |
| --- | --- | --- | --- |
| C1 | **Sign clamp at the store** (any negative FP16 → `+0`) | `DEVICE_SUPPORTED` (behaviour); **stage `DEVICE_UNVERIFIED`** | explains 12, 13, 19, 32–36, 42, 43; the PRM documents no clamp, so it is undocumented behaviour, not a specified one |
| C2 | Clamp on the shaped value **before** `bo` is added (`s*clamp(shape(ACC)) + bo`) | **`REFUTED`** | mode 17 (`shape 4`, `bo +128`) stored `62.0`; a clamp before `bo` would give `0 + 128 = 128 = 0x5800` |
| C3 | Clamp **after** `s*shape(ACC)+bo` (convert output or store input) | `DEVICE_SUPPORTED` | same six modes; also the only variant that is consistent with mode 17 |
| C4 | **Negative-scale path defect**: when `s<0`, `bo` is dropped, its sign is inverted, or the convert fails closed | `DEVICE_SUPPORTED` as an anomaly; **sub-mechanism `DEVICE_UNVERIFIED`** | mode 16 is the controlled case: same expected bits as 15/17, only the scale sign differs, yet it stores `0`; mode 12 is consistent |
| C5 | ACC cannot hold a negative value | `DEVICE_UNVERIFIED`, disfavoured | mode 17 requires the shaped term `-66.015625` to produce the observed `+61.984375`; that is a negative intermediate, although it could be formed in the shaper rather than stored in the ACC |
| C6 | The shaper drops the sign / works on the absolute value | **`REFUTED` for `shape 4`** | mode 17's `62.0` requires `-x`, not `abs(x)` (`+64x + 128 = 194.0 = 0x5A10`) |
| C7 | Per-output-channel bias cells are not honoured (one shape per set) | `SPEC_CONFIRMED` (PRM p.13: cells 0–31 apply to the corresponding output channels); **`DEVICE_UNVERIFIED`** | no recorded mode has ever programmed two different shapes in one set; every mode uses the set-wide `init_bias()` |
| C8 | Operand packing is still wrong (row-major) | **`REFUTED`** | mode 43's corrected pair-interleaved packing reproduced 3/3 positive slots and index 32; the pre-mode-43 row-major replay is documented in the E-3C diagnosis |
| C9 | FP16 overflow / saturation | **`REFUTED`** | magnitudes 0.003–66 against a maximum of 65504; `Rs[6]=0`, no overflow is reachable |
| C10 | Feedback (`Fb dest`/`Fb op`) rewrites the result | **`REFUTED`** | `Rs[3:2]=0` (no feedback) in every surveyed mode; the PRM no-feedback recipe sets the rest of `Rs` to 0 |
| C11 | `Rs[0]=0` (clear+swap) would revive negatives | **`REFUTED`** (by PRM text) | `Rs[0]` is "accumulator behavior", not an operand of Figure 11; it cannot change a single convert's output bits |
| C12 | Rounding / quantization is the culprit | **`REFUTED`** for the recorded magnitudes | the clamp is orders of magnitude larger than any rounding step; mode 43's `-64.0` and mode 42's `-30.125` are not near-zero |
| C13 | Store index / readback misread (`0x0000` is an unwritten cell) | **`REFUTED`** for indices 0/1/2/32 | mode 43 returned `0x5400` at index 2 and `0x0000` at index 1 from the same store, and mode 42 matched index 32 to `(s0,o16)`; a slot permutation is separately excluded by E-3D variant B |
| C14 | An undocumented normalize stage between convert and store | `CPU_MODEL_ONLY` | indistinguishable from C1/C3 with a convert+store-only ISA; no way to observe an intermediate value |
| C15 | The clamp is specific to the *residual* path (fine converts) | `REFUTED` | mode 42's `-30.125` and mode 43's `-64.0` are full-magnitude converts, not residuals |

### 3.1 What is established

1. The convert/store path **does not preserve a negative FP16 value** in any
   configuration measured so far (`DEVICE_SUPPORTED`, ten recorded modes,
   magnitudes from `0.003` to `66`).
2. The clamp is **downstream of the shaping function**, not between the ACC and
   `bo` (mode 17 refutes C2).
3. The **negative-scale path is independently broken**: mode 16's PRM value is
   positive and RN16-rounds to the same bits as modes 15/17 (`0x53C0`), yet it
   stores `0`. This cannot be a final-value clamp (`DEVICE_SUPPORTED`).

### 3.2 What is not established

1. Which stage clamps **between the convert output and the store**. This ISA
   exposes only `cvt` and the store, both single instructions, and no
   intermediate readback exists, so convert-output vs store is unobservable in
   principle with the current surface. "ACC input" vs "after the shape" is
   separable, and E-3D does separate it (`out[3]`).
2. Whether a per-output-channel `shape` is honoured on device (PRM says yes,
   `DEVICE_UNVERIFIED`). E-3D is the first probe to program mixed cells.
3. Whether the `-0` bit pattern can ever be produced or is normalised (`+0`),
   because `bo=+0` makes `s*(-0)+(+0) = +0` by IEEE and no `-0` can be forced
   through the convert equation at all.
4. Whether the negative-scale defect is "bo dropped", "bo sign inverted" or
   "fail closed". A single convert cannot separate all three, because with
   `s<0` and a positive ACC both a dropped and an inverted `bo` produce a
   negative result that clamps to `0`. A *pair* of converts with `s=-1`,
   `ACC=-x` and `bo=+2` / `bo=-2` would: correct gives `[3.03, 0]`, `bo`
   dropped gives `[1.03, 1.03]`, `bo` inverted gives `[0, 3.03]`, fail-closed
   gives `[0, 0]`. That experiment is **not** authorised and is not part of
   E-3D, which uses `s=+64` throughout.

## 4. The mode-16 anomaly in detail

```text
mode 15: 64*(+x) - 4     = 62.015625  -> tie -> 62.0 = 0x53C0   device 0x53C0  OK
mode 17: 64*(-x) + 128   = 61.984375  -> tie -> 62.0 = 0x53C0   device 0x53C0  OK
mode 16: -64*(+x) + 128  = 61.984375  -> tie -> 62.0 = 0x53C0   device 0x0000  CONTRADICTION
```

The three cases were run in the same session family with the same fixture
(`x = a²`, `q = 6`), the same store, the same `Rs`, and produce the same
mathematical result — none of them is negative, so no final-value clamp can be
involved. Mode 17 proves the device can produce a negated product when the sign
comes from the shaper (`shape 4`) with a positive scale. Mode 16 computes the
same value with the sign in the scale and fails. Mode 15 confirms that a
nonzero `bo` is applied correctly when the scale is positive. The remaining
variable is the negative scale itself. Three sub-hypotheses fit the data
equally well:

| sub-hypothesis | mode 16 | mode 12 (`bo=0`) | mode 15 (`s>0`) |
| --- | --- | --- | --- |
| `bo` dropped when `s<0` | `-66.015625` → clamp → 0 | 0 | `62.0` |
| `bo` sign inverted when `s<0` | `-66.015625 - 128` → clamp → 0 | 0 | `62.0` |
| convert fails closed when `s<0` | 0 | 0 | `62.0` |

No recorded datum separates them. Note that "negative scale is silently treated
as `|s|`" is **refuted**: that would give `+64x + 128 = 194.0 = 0x5A10`, not 0.

## 5. What this changes for the two-convert split (Path B reconstruction)

Path B's synthesis needs the signed value of one accumulator split into two
non-negative stores, because the direct store cannot hold a negative. The
correct HMX realization is **`shape 2 = max(x,0)` for the positive part and
`shape 5 = -min(x,0) = max(-x,0)` for the negative part**:

```text
y = RN16(64 * max(y,0))  -  RN16(64 * max(-y,0))        (HVX subtract)
```

`shape 6 = -max(x,0)` is a trap: it is always `<= 0`, so it is exactly the kind
of value the clamp destroys (oracle §8 pins `shape 6` on a negative ACC →
`0x0000`, `shape 5` → `0x5420`).

Properties pinned by the CPU oracle (2176 cases, `e ∈ [-8,8]`, 64 mantissa
steps, both signs):

* the two halves are never simultaneously non-zero (`both_nonzero = 0`), so the
  subtraction never cancels a rounding error;
* the reconstruction is **bit-identical to a signed RN16 store**
  (`mismatches = 0`): the split recovers the sign, it does not buy precision;
* `shape 2` on a positive ACC and `shape 5` on a negative ACC both produce
  non-negative values, so neither triggers the measured clamp if the clamp is
  at or after the shape.

Decision table (this is why E-3D matters):

| E-3D verdict | consequence for the split |
| --- | --- |
| `E3D_SIGNED_STORE_CONFIRMED` | the split is unnecessary; a direct signed store exists |
| `E3D_CLAMP_DOWNSTREAM_OF_SHAPE` | the split **works** with no ISA change: 1 MMA, 2 converts, 2 stores, 1 HVX subtract |
| `E3D_CLAMP_AT_ACC_BEFORE_SHAPE` | the split **fails**: `shape 5` would receive a clamped `+0` and the negative half is unrecoverable |
| `E3D_SHAPE_NOT_APPLIED_PER_CHANNEL` | the split as designed is unavailable; the PRM's per-channel bias claim is falsified on device |
| `E3D_SIGNED_STORE_WITH_ACC_CLAMP` | the split fails for the same reason as `EARLY_CLAMP` |
| `E3D_ROUNDING_NOT_RN16` | orthogonal to the clamp; the split still works but on a different lattice |
| `E3D_INCONCLUSIVE_*` | no consequence; Path B stays where it is |

The standing cost caution is unchanged and independent of this document: against
the kernel-given-input reference, the W-only variant of the current
FP16-activation path regresses 1.4–1.5× at ~3× MMA cost, so a useful kernel has
to justify that cost. The CPU follow-up
(`docs/hmx-v81-sign-separated-output-cpu.md`) refines the split cost: if the
two shapes are carried by duplicated output channels in one pass, the split
costs ~2× MMA per logical output (a 32-output tile then carries 16 logical
outputs) in addition to the second convert and store. A two-pass variant that
re-reads a retained ACC would keep 1× MMA, but that sequence is unverified.
Neither realization has been timed.

CPU evaluation (2026-10-24): the shape-0/4 pair reproduces the signed RN16
value on all 5,632 sweep cases and on the 1,024-cell G8 tile
(`b3_bits_diff = 0`), and carries the B3-vs-plain ratio over exactly
(`1.845321`). It requires `bo = 0` and is a re-encoding, not a precision gain;
it is not device-confirmed for shapes 2/5 or outside the E-3D magnitudes.

## 6. What E-3D decides, candidate by candidate

| E-3D cell outcome | candidates affected |
| --- | --- |
| `out[2] = 0xD820` | signed store established; C1/C3 refuted for a positive-ACC cell; C5 refuted (ACC is positive); C7 refuted (shape 4 was applied to channel 1) |
| `out[2] = 0x0000`, `out[3] = 0x5820` | shape 4 applied (C7 refuted), so C1/C3 hold **after** the shape; C5 not decidable (a store clamp hides the ACC state) |
| `out[2] = 0x0000`, `out[3] = 0x0000` | C5 supported: the only way to zero a positive `out[3]` is to clamp the negative ACC before the shape; C7 refuted (`out[2]` would be `0x5820` if shape 4 were ignored) |
| `out[2] = 0x5820` | C7 supported: shape 4 was not applied to channel 1, refuting the PRM p.13 per-channel claim on device; `out[1]`/`out[3]` then decide the store |
| `out[2] = 0xD820`, `out[3] = 0x8000` | signed store + ACC clamp + `-0` preserved; C7 refuted |

C4 is out of E-3D's scope by construction (`s=+64`): the negative-scale defect
needs its own two-convert discriminator, and the mode-16 anomaly stays
`DEVICE_SUPPORTED` but mechanistically unresolved until then.

## 7. Bounded next steps (none of them authorised here)

1. **E-3D** as pre-registered: decides C1 vs C3 vs C5 vs C7 for the store and
   for the per-channel bias row. Requires explicit approval (new mode, one
   build, one run).
2. **Negative-scale discriminator** (`s=-1`, `ACC=-x`, `bo=+2` and `bo=-2`, two
   converts): separates "bo dropped" from "bo sign inverted" from "fail
   closed". Only worth doing if the negative-scale path becomes relevant to a
   research priority; it does not affect the split, which uses `s=+64`.
3. **Do not** re-run mode 43, expand B3, add a performance measurement, or
   change production QNN/Muon code on the basis of this analysis. E-3C2 stays
   `UNTRIGGERED`.

## 8. Status carry-over

Unchanged by this document: `B2_THREE_PRODUCT_CONFIRMED`,
`E3C1R_LAYOUT_CONFIRMED_CLAMP`,
`BLOCKED_SIGNED_OUTPUT_FOR_CURRENT_CONVERT_PATH`, Path B
`CONDITIONALLY_POSSIBLE`, real-tile accuracy `INSUFFICIENT_EVIDENCE`, E-3C2
`UNTRIGGERED`. The classification of this analysis is
`SIGNED_OUTPUT_CAUSES_CLASSIFIED`: one device-supported clamp with an unobservable
stage, one device-supported negative-scale anomaly with an unresolved
sub-mechanism, and one spec-confirmed-but-device-unverified per-channel bias
feature.
