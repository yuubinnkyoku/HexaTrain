# Sign-separated output reconstruction (shape 0 / shape 4): CPU-only evaluation

Status: `CPU_MODEL_ONLY`, 2026-10-24. No DSP change, no QNN change, no device
run, no new mode, no performance measurement. This document evaluates the
method suggested by the accepted E-3D result; it does not replace the E-3D
pre-registration (`docs/hmx-v81-signed-fp16-discriminator-prereg.md` §17) or
the device record (`b0b7edf`).

## 1. Premise taken from E-3D

The only device input to this evaluation is the accepted E-3D four-slot result
and its reading:

| ACC | shape | expected signed FP16 | measured |
| --- | --- | --- | --- |
| `+x` | 0 (identity) | `0x5420` | `0x5420` |
| `-x` | 0 (identity) | `0xD420` | `0x0000` |
| `+x` | 4 (`-x`) | `0xD420` | `0x0000` |
| `-x` | 4 (`-x`) | `0x5420` | `0x5420` |

Verdict: `E3D_CLAMP_DOWNSTREAM_OF_SHAPE`. A negative value is destroyed after
the shape is applied; the ACC itself can hold and shape a negative value. The
exact stage (inside the convert or at the store) remains unidentified.

Everything below is a CPU model of that law. It is not additional device
evidence, and the clamp law is measured only at the E-3D magnitudes
(`±66.015625`, `±132.03125` at scale 64).

## 2. Method

One logical output `y` is duplicated into two physical output channels that
receive the same ACC. The convert bias cells differ only in `shape`:

```text
y_rec = fp16( store(shape 0(y)) - store(shape 4(y)) )
      = fp16( store(y)          - store(-y) )
```

with one shared positive scale `s` on both channels and post-shape output bias
`bo = 0`. The subtraction is done in HVX after the store, not in HMX.

With `bo = 0` and the measured law:

* `y > 0`: the shape-0 half survives as `RN16(s*y)`, the shape-4 half is
  rounded to a negative value and discarded as `0x0000`;
* `y < 0`: the shape-0 half is discarded, the shape-4 half survives as
  `RN16(s*|y|)`;
* the surviving half carries the only rounding that reaches the result, and
  `x - 0` / `0 - x` are exact in FP16.

The method is therefore a **re-encoding**, not an arithmetic change. It
recovers the sign; it cannot add or remove precision relative to a signed RN16
store. That is the property the two CPU models below test.

## 3. CPU results

### 3.1 Encoding sweep (`hmx_signed_output_oracle.c` §11)

`e ∈ [-32, 11]`, 64 mantissa steps, both signs, scale 64, `bo = 0`
(5,632 cases; the band includes subnormal underflow and FP16 overflow):

```text
split04 sweep: cases=5632 both_nonzero=0 one_sided=5502 both_zero=130
               numeric_mismatch=0 neg_zero_bitdiff=65
```

* `both_nonzero = 0`: exactly one half is non-zero (or both are zero), so the
  subtraction never combines two rounded values.
* `numeric_mismatch = 0`: every reconstructed value equals the direct signed
  RN16 value.
* `neg_zero_bitdiff = 65`: the only bit-level difference is `RN16(-0) = 0x8000`
  on the direct path versus `+0 = 0x0000` after the split. Numerically equal;
  a bitwise output comparator must tolerate this.
* E-3D magnitudes reproduce exactly: `-x -> (0x0000, 0x5420) -> 0xD420`,
  `+x -> (0x5420, 0x0000) -> 0x5420`, `-2x -> (0x0000, 0x5820) -> 0xD820`.

### 3.2 Stored-domain accuracy (`hmx_fp16x2_split_sim.c` G9)

The G8 1,024-cell tile (seed `0xE3C0117E5EEDF00D`, same tile and oracle as the
published E-3C0 numbers) is re-encoded through the shape-0/4 pair:

```text
sign_split_g9 cells=1024 plain_recon_mean_abs=1.307244e-03
               b3_recon_mean_abs=7.084104e-04 plain_over_b3=1.845321
               plain_bits_diff=0 b3_bits_diff=0 plain_both_nonzero=0
               b3_both_nonzero=0 plain_one_sided=1023 plain_both_zero=1
               improved=427 equal=597 worsened=0
               plain_optimal_bits_frac=5.820312e-01
               b3_optimal_bits_frac=9.980469e-01
sign_split_g9_verdict=ENCODING_PRESERVED accuracy_change=none
```

| variant | stored mean abs error | ratio vs plain | oracle-optimal bits |
| --- | --- | --- | --- |
| plain FP16 | `1.307244e-03` | `1.000000` | `58.20 %` |
| B3 three-product | `7.084104e-04` | `1.845321` | `99.80 %` |

Both variants are reproduced **bit-for-bit** (`plain_bits_diff = 0`,
`b3_bits_diff = 0`). The previously reported ~1.85× improvement is therefore
preserved exactly, but it is also **not a property of the split**: it is the B3
vs plain improvement on this tile, unchanged by the encoding wrapper. The
split neither rescues nor degrades the three-product method.

The 1.71×/1.83× figures from the G7 internal-domain analysis are a different
metric and are unaffected.

## 4. Preconditions and failure modes

The method is valid only under all of the following. Each is a configuration
invariant, not a data-dependent hope.

| condition | why | status |
| --- | --- | --- |
| clamp law holds after the shape | this is what zeroes the inactive half | device-supported at E-3D magnitudes only |
| `bo = 0` on both duplicated channels | `bo` is added after the shape and does not cancel | required by the model |
| one shared positive scale `s` | different scales give `(s0+s4)·y`, not `y` | required |
| both channels receive the same ACC | the two halves must describe one value | required |
| no NaN in the ACC | NaN payload is not preserved by the model | out of scope |

Failure modes, pinned in oracle §11:

* **Signed store counterfactual.** If a future mode preserved the sign, shape 0
  would export `-32` instead of `0` and shape 4 would export `+32`; the
  reconstruction returns `2y` (`-64` instead of `-32` at scale 64). The E-3D
  slots `out[1] = 0x0000` and `out[2] = 0x0000` are therefore also the validity
  guard for this method and must be re-checked if the convert path changes.
* **`bo ≠ 0`.** A large bias makes both halves non-zero and the result becomes
  `2y`; the `both_nonzero` symptom detects that case. A small bias leaves one
  half clamped and the result becomes `y + bo` — still wrong, but *not*
  detectable from the data alone (`bo = 1` case: `32 + 1 = 33` stored on one
  side, `0` on the other). Only a configuration invariant `bo = 0` is safe.
* **`bi ≠ 0` is allowed.** The pre-shape input bias is applied to both halves
  and cancels exactly; only the post-shape `bo` is forbidden. Whether this
  convert path can use `bi` is not verified.
* **Bias-safe alternative.** `shape 2 / shape 5` (`max(y,0)`, `max(-y,0)`)
  cancels `bo` exactly because each shape zeroes the inactive half before the
  bias is added. It is the bias-robust variant, but it relies on PRM shape 2/5
  semantics that this device configuration has never exercised; E-3D used
  shapes 0 and 4 only. Oracle §8 pins its bit-identity on 2,176 cases.
* **Signed zero.** The split produces `+0` where a signed RN16 store would
  produce `-0`; numerically identical, bitwise different.
* **Clamp placement.** The model applies the clamp to the post-convert word.
  The E-3D evidence excludes an ACC clamp but does not separate a convert
  clamp from a store clamp; the method needs only "a negative final value
  becomes zero after the shape".

## 5. Cost model (static, not performance-measured)

Two realizations are possible, and they have different costs.

**R1 — one pass, duplicated output channels (assumed by the G9 model).**
Each logical output needs two physical channels, so a 32-output MMA tile
carries only 16 logical outputs:

* ~2× MMA work per logical output, duplicated weight tile;
* 2× convert + 2× store per logical output (one per duplicated channel);
* 1 HVX FP16 subtract per logical output;
* 2× output bias cells, `bo = 0` on all of them;
* ~2× output VTCM words before reconstruction.

**R2 — two passes over a retained ACC (unverified).**
If the ACC can be converted a second time after a store (`Rs[0]` retain), the
same 32-channel ACC could be exported twice with two bias rows (all shape 0,
then all shape 4):

* 1× MMA work, no duplicated weights;
* 2× convert + 2× store per logical output;
* 1 HVX subtract per logical output;
* requires the convert/store sequence to re-read the ACC after a store, which
  is not demonstrated in this configuration.

The earlier sentence in
`docs/hmx-v81-signed-output-cause-analysis.md` §5 that the split "adds store
bandwidth, not MMA work" describes R2 only; R1 doubles MMA per logical output.
Neither realization has been timed. E-3C2 remains `UNTRIGGERED`.

## 6. What this does not claim

* Not a signed HMX store. `BLOCKED_SIGNED_OUTPUT_FOR_CURRENT_CONVERT_PATH`
  stands; the sign exists only after an HVX subtraction.
* Not device-confirmed for shapes 2/5, per-channel bias with mixed shapes, or
  any magnitude outside the E-3D four slots.
* Not an accuracy improvement. The 1.85× is B3's, reproduced exactly.
* Not a performance result. No kernel was built, no cycle counter was read.
* Not real-tile evidence. The 1,024-cell tile is a CPU model; real-tile
  accuracy remains `INSUFFICIENT_EVIDENCE`.
* No change to `B2_THREE_PRODUCT_CONFIRMED`,
  `E3C1R_LAYOUT_CONFIRMED_CLAMP`, `E3D_CLAMP_DOWNSTREAM_OF_SHAPE`, Path B
  `CONDITIONALLY_POSSIBLE`, or E-3C2 `UNTRIGGERED`.

## 7. Reproduction

```powershell
.\scripts\run_hmx_signed_output_oracle.ps1   # §11 sweep + failure pins
.\scripts\run_fp16x2_split_sim.ps1           # G9 tile + E-3C0 gate
.\scripts\verify.ps1 -Profile Fast
```

Expected keys: `split04 sweep ... numeric_mismatch=0`,
`sign_split_g9 ... plain_over_b3=1.845321 ... b3_bits_diff=0`, and
`sign_split_g9_verdict=ENCODING_PRESERVED accuracy_change=none`.

If a future device test is ever approved, the minimal discriminator is one
tile with two channels carrying the same `±ACC`, bias cells `shape 0` and
`shape 4`, `bo = 0`, plus one channel with `bo ≠ 0` to confirm the failure mode.
No such test is authorised here.
