# V81 HMX accumulator residual-recovery probe (experimental)

This is a **separate FastRPC micro-experiment**. It does not change the
production HexaTrain Muon, HVX, QNN, checkpoint, or training code paths.

## Question

Can V81 HMX export an accumulator's low-order information through two
FP16 convert transfers, using `Rs[0]=1` to retain ACC and then
`Rs[3:2]=1` to feed the previous convert state into output bias?

The test constructs a single nonzero dot product:

```text
a = b = 1 + 2^-10                  (exact FP16: 0x3c01)
x = a*b = 1 + 2^-9 + 2^-20         (exact mathematical product)
hi = round16(64*x)                  expected 64.125 = 0x5402
residual = round16(-64*x + min(65504, prior_convert))
                                     expected -2^-14 = 0x8400
reconstructed = (hi - residual)/64  expected x
```

The code uses 32x32 FP16 croutons in VTCM, sets every element to zero
except `activation[0]` and `weight[0]`, and reads output tile element zero.
Therefore **it does not depend on a multi-element cancellation pattern**.

Two identical HMX products are run:

1. First convert `Rs=0x0001`: ACC retained, normal feedback rounding;
   second convert `Rs=0x1004`: bias set 1, `min` feedback as output bias,
   ACC cleared/swapped.
2. First convert `Rs=0x0101`: extra four feedback mantissa bits enabled;
   second convert again `Rs=0x1004`. This is a *diagnostic comparison*.

The first bias set has FP16 scale=64 and shape=x. The second has
scale=64, shape=-x and FP16 output-bias limit +65504. The bias payloads
follow the two-vector / 256-byte arrangement in V81 manual Tables 3–5.

**Why trial 2 matters:** If the previous convert state has more precision
than the FP16 `hi` exported to VTCM, subtracting it may not reconstruct
the same upper+residual decomposition. The host prints both trials.

## Building (Qualcomm Hexagon SDK, not Android app Gradle)

From a compatible Hexagon SDK installation:

```sh
source /path/to/Hexagon_SDK/setup_sdk_env.source
cd host_tests/hmx_acc_residual_rpc
build_cmake android
build_cmake hexagon DSP_ARCH=v81
```

Both builds require the corresponding Android/Hexagon toolchains; DSP
compilation explicitly enables `-mhmx`.

**Status:** built and executed on a nubia NX741J (Hexagon V81, CDSP) with
Hexagon SDK 6.6.0.0 / tools 19.0.07. See the PR #14 comments for the
measured results.

## Host-only self test

`probe_host_selftest.c` verifies the shared IEEE binary16 helpers
(`fp16_util.h`) and the numeric oracle without a device:

```sh
cc -std=gnu11 -O2 -o probe_host_selftest probe_host_selftest.c -lm
./probe_host_selftest
```

Clang is also supported (its native `_Float16` is used as an independent
reference when available):

```sh
clang -std=gnu11 -O2 -o probe_host_selftest probe_host_selftest.c -lm
./probe_host_selftest
```

The self test covers:

- known values (including the previously mis-rounded 64.0625, 64.1875 and
  1.0009765625),
- round-to-nearest-even tie behaviour for positive and negative values,
  including extra binades, subnormal boundaries and the largest finite
  value,
- subnormal boundaries (smallest subnormal, transition to smallest normal,
  largest subnormal) and the overflow threshold around 65504,
- a round trip of **all 65536 binary16 bit patterns** (63488 finite
  including signed zero, 2 inf, 2046 NaN),
- an independent cross-check against the compiler's own `_Float16`
  conversion when the toolchain provides it: all 63488 exactly
  representable values, all 2048 specials, a non-representable sweep
  (8 ULP fractions × 31743 adjacent pairs × both signs) and the overflow
  boundary — **573452 `_Float16` conversions, 639073 comparisons total**,
- the expected hi / remainder table for q=6..10 (positive *and* negative
  remainders — the negative ones are what the device actually clamps,
  see the battery results below) and the reconstruction identity.

`fp16_util.h` is shared by the host runner and the self test, so the
oracle and its test cannot drift apart.

**Compiler matrix (2026-10-20; raw evidence under `build/hmx_evidence/`,
gitignored):**

| toolchain | target | result |
| --- | --- | --- |
| gcc 15.2.0 (MSYS2) | x86_64-w64-mingw32 | PASS — 573452 `_Float16` comparisons, 0 mismatches, `-Wall -Wextra -Werror` clean |
| clang 22.1.8, vcvars64 env + `-rtlib=compiler-rt` | x86_64-pc-windows-msvc | PASS — identical counts |
| clang 22.1.8, MSYS2 libgcc | x86_64-w64-windows-gnu | **FAIL — reported, not masked**: 573443/573452 `_Float16` comparisons mismatch |

The failing clang triple is root-caused (llvm-objdump,
`selftest_clang_x86_64-w64-windows-gnu.txt`): the call site reads the
`_Float16` result from the Windows FP return register (`pextrw $0x0,
%xmm0`) while MSYS2 `libgcc.a __truncdfhf2` returns it in `%eax`, so the
conversion result is never observed; `-rtlib=compiler-rt` does not exist
for that triple. The default clang target without a vcvars environment
fails at `vcruntime.h`, and MSVC linking without `-rtlib=compiler-rt`
fails on `__truncdfhf2` — both constraints are recorded in the MSVC
evidence file. `scripts/run_hmx_selftest.ps1` encodes the working
invocations and is wired into the Fast verification profile.

## Identity check

Every run first calls the DSP with `mode=30` (identity mode) on the same
handle. The DSP echoes its own `PROBE_REVISION`, the received `q`, the
received `mode` and the DSP architecture through `results[0..3]`, and
reports its metadata (`HMX_MAGIC`, architecture, HMX-executed flag and
trial count) through the metadata buffer. The host aborts with exit code 9
when either the echo or the identity metadata does not match this source,
so a stale DSP skel can never be mistaken for a current one.

Deploy the generated `libhexatrain_hmx_acc_residual.so` (AArch64 stub),
`libhexatrain_hmx_acc_residual_skel.so` (V81 DSP skel) and
`hexatrain_hmx_acc_residual_test` (AArch64 executable) using the same
CDSP unsigned-PD / FastRPC procedure as the existing HexaTrain HVX probe.
Make the skel discoverable in your device's configured DSP library path.
Run `hexatrain_hmx_acc_residual_test` without arguments.

The host **rejects non-V81 CDSP** rather than silently emulating the
operation on a CPU. The DSP requests HMX plus 12 KiB of VTCM, and errors
out if allocation, 2-KiB alignment, or HMX locking fails. Teardown is
attempted and the host verifies the handle close status; a non-zero close
status fails the run.

## Interpreting the output

The host reports raw FP16 bits, FP16-decoded values, a CPU-double
reference, reconstruction error and the ordinary FP16 output error.

```text
extra4=0 hi_bits=0x5402 residual_bits=0x8400 ...
observed_reproduced=REPRODUCED
math_expected_hi=0x5402 math_expected_res=0x8400 math_match=MATCH
fp16_residual_recovery=SUPPORTED
```

Two verdicts are reported separately and must not be conflated:

- `observed_reproduced=REPRODUCED` — the raw bits equal a previously
  measured value. This says nothing about numerical correctness.
- `math_match=MATCH|MISMATCH` — the raw bits equal the mathematically
  correct value, derived from double arithmetic by the host.

Modes without a registered expectation print `oracle=NO_ORACLE`.
`reconstruction_valid=0` means the slot layout for that mode does not
define a reconstruction, so `reconstructed` carries no meaning.

If the first trial instead returns `residual_bits=0x0000` (or any other
value), then the hypothesis is **not confirmed** on that runtime. This
is no longer hypothetical: on the measured device the residual slot for
the mode-0 feedback recipe *does* come back `0x0000` for these operands
— see the battery results below — so `fp16_residual_recovery=NOT_CONFIRMED`
is the measured status of the original recovery recipe. A single failing
configuration does not prove universal impossibility either: the
quantization laws section describes exactly which corrections the
convert can and cannot represent. No downstream optimizer switches are
included.

## Sources / caveats

- Qualcomm, [Hexagon V81 HMX Programmer’s Reference Manual, Rev. AA]
  (https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf),
  especially Tables 3–6, Table 9 and Chapter 5.
- [Existing HexaTrain HVX FastRPC probe](../hvx_rpc/).
- The HMX experiment is an independent implementation from the public
  instruction specification; it does not incorporate htp-ops-lib source.
- Do not infer FP32 output support or a valid high-precision GEMM from
  this experiment unless on-device tests support that conclusion.

## Build id and identity fail-closed hardening

The shared validator lives in `hmx_identity.h` (no dependencies, usable
from both host and DSP). It turns the identity-mode echo into a
fail-closed check with one flag per failure condition:

| Flag | Failure meaning |
| --- | --- |
| `HMX_IDENTITY_FAILURE_MAGIC` | DSP metadata `HMX_MAGIC` mismatch |
| `HMX_IDENTITY_FAILURE_ARCH` | DSP architecture is not V81 |
| `HMX_IDENTITY_FAILURE_REVISION` | echoed `PROBE_REVISION` mismatch |
| `HMX_IDENTITY_FAILURE_MODE` | echoed mode is not the identity mode (30) |
| `HMX_IDENTITY_FAILURE_Q` | echoed `q` is not the requested `q` |
| `HMX_IDENTITY_FAILURE_BUILD_ID` | nonzero DSP/host build id mismatch |
| `HMX_IDENTITY_FAILURE_TRIAL_COUNT` | identity-mode trial count is not 0 |
| `HMX_IDENTITY_FAILURE_HMX_FLAG` | identity metadata HMX flag is set |
| `HMX_IDENTITY_FAILURE_CLOSE` | mode-30 handle close reported failure |

A build id of `0` on either side is reported as `BUILD_ID=UNAVAILABLE`
and is **not** a failure (`HMX_BUILD_ID_UNAVAILABLE`): it only means the
binaries were not stamped. A mismatch between two *nonzero* build ids
fails the run, so a stale DSP skel can never pair with a newer host
runner.

The DSP build id travels through `metadata[HMX_IDENTITY_BUILD_ID_SLOT]`
(`metadata[2]`, unused by mode 30; IDL ABI shapes — `results[4]` and
`metadata[9]` — are unchanged) and is stamped at compile time with
`-DHMX_PROBE_BUILD_ID=0xNNNNNNNN`. `hmx_identity.h` defines the macro to
`0` when unset, so unstamped builds still compile and run.

**Stale-skel incident (battery 2, 2026-10-20).** A device battery was
run after a host-only rebuild without redeploying the skel: host stamp
`0x19cc4c4f`, deployed skel still stamped `0x7dfcb259`. Every case
failed closed exactly as designed — `build_id_status=MISMATCH`,
`dsp_identity=MISMATCH` with the `BUILD_ID` identity flag printed, exit
9 *before any numerical experiment* (transcript
`build/hmx_evidence/b_battery2_20261020.txt`). The root cause was on the
build side: the compiler response file path was constant, so CMake never
reconfigured when the stamped id changed and the skel kept its old
stamp. Fix: the response file name now embeds the build id
(`b_build_id_<hex>.rsp`), forcing regeneration on every change. The
follow-up full battery rebuilt at stamp `0x0c630195` and every case
reported `build_id_status=MATCH`; the verdict-fix rerun at `0x539bf682`
likewise matched.

The identity self test (`probe_identity_selftest.c`) validates the
shared header on the CPU without a device, including all adversarial
payloads and the build-id-unavailable semantics:

```sh
gcc -std=c11 -O2 -Wall -Wextra -o probe_identity_selftest probe_identity_selftest.c
./probe_identity_selftest          # prints "all checks passed", exit 0
```

## DSP return codes

| Code | Name | Meaning |
| --- | --- | --- |
| 14001 | `HMX_ERR_ARG` | invalid argument |
| 14002 | `HMX_ERR_ALLOC` | rpcmem allocation failed |
| 14003 | `HMX_ERR_VTCM` | VTCM allocation/alignment failed |
| 14004 | `HMX_ERR_LOCK` | HMX power-up or lock failed |
| 14005 | `HMX_ERR_UNLOCK` | legacy unlock status (unused) |
| 14006 | `HMX_ERR_POWER_UP` | HMX power-up path failed |
| 14007 | `HMX_ERR_UNLOCK2` | cleanup: second unlock stage failed |
| 14008 | `HMX_ERR_RELEASE` | cleanup: HMX release stage failed |
| 14009 | `HMX_ERR_POWER_DOWN` | cleanup: power-down stage failed |

Cleanup precedence: after the compute path, teardown stages run in the
order mode-A unlock2 → mode-A release → mode-B unlock2 → mode-B release
→ power-down. The **first failing cleanup stage** wins and is returned
*before* an execution failure from the compute path, so resource
mis-management is never masked by a numeric error. Metadata is not
touched by cleanup reporting. Code 14006 is unchanged: nothing is
acquired yet on the power-up path, so the defensive power-down is kept
but not reported as a cleanup failure.

## `run_experiment.ps1` provenance

`run_experiment.ps1` orchestrates a fully local provenance build. It
stamps a build id, rebuilds both targets (Android AArch64 + Hexagon V81)
through the SDK CMake toolchain using a response file for
`-DHMX_PROBE_BUILD_ID` (the SDK `build_cmake`/`cmake_configure.bash`
path corrupts values containing `=` and drops env `CFLAGS`, so a
`@file` response is used instead; the response file name embeds the
build id itself — `b_build_id_<hex>.rsp` — so a changed stamp forces
CMake reconfiguration), runs the identity self test, and
records SHA-256 hashes of all three ship artifacts:

```powershell
cd host_tests/hmx_acc_residual_rpc
pwsh -NoProfile -File .\run_experiment.ps1 -Provenance
```

The script prints a `SHA256` artifact table and writes a transcript to
`build\hmx_evidence\b_provenance_<timestamp>.txt` (gitignored; never
committed). Useful parameters: `-BuildId 0xNNNNNNNN` (default: random),
`-SkipBuild`, `-DeployAndRun`, `-Q`, `-Mode`, `-DeployRetries`.

## Judgment matrix and exit codes

Every run ends with a judgment matrix so the verdicts cannot be
conflated:

```text
RPC_OK=1 IDENTITY_OK=1 HMX_EXECUTED=1 RESOURCE_CLEANUP_OK=1
OBSERVED_REGRESSION_MATCH=1 MATH_ORACLE_MATCH=N/A
DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED
exit_code=0 exit_code_description=...
| legend: 0=ok; 9=identity; 8=close; 7=rpcmem; 6=open; 5=uri; 4=pd; 3=arch; 2=args
```

- `RPC_OK` — FastRPC call and handle close both succeeded.
- `IDENTITY_OK` — every identity flag passed (see the table above;
  failures are printed individually as `identity_fail=<FLAG>`).
- `HMX_EXECUTED` — DSP metadata reports HMX execution.
- `RESOURCE_CLEANUP_OK` — all cleanup stages returned success (no
  14007/14008/14009).
- `OBSERVED_REGRESSION_MATCH` — raw bits equal previously measured
  literals (`observed_source_*`); independent of numeric correctness.
- `MATH_ORACLE_MATCH` — raw bits equal the double-derived expectation
  (`oracle_source=double_derived` for modes 24–29 with a literal
  cross-check in `oracle_selfcheck=`; `oracle_source=literal` for modes
  18/21/22). A math mismatch does **not** change the exit code; the
  matrix reports it as a fact.
- `DYNAMIC_FEEDBACK_CONFIRMED` — dynamic-feedback hypothesis status.

Per-mode host reconstruction is defined only for modes {0, 2, 20, 23};
other modes print `reconstruction=N/A recovered_error=N/A`. The printed
`scale=` line gives the per-mode scale (32→{24,27}, 128→{26,29},
otherwise 64). Single-CVT trials label the unused second-convert slot as
`unused_slot2` / `unused_val` rather than implying a second convert.
Modes 21/22 print a `zero_residual_control` note with the q binding.

## Phase D: extra4 nibble sweep (modes 31..38)

**Question.** For q=8 the double-exact residual is
`(1+2^-8)^2*64 - 64.5 = 2^-10` — **1/4** of the output-bias fine quantum
`2^-8` (2^6/16384, the 20-bit mantissa step at that binade) — yet mode 28
stored `0x1c00` = `2^-8`, i.e. **+1 fine ULP**. Does the stored convert
output track the bias's extra4 nibble one fine quantum per step
(bias-side), or does it stay put (convert-side rounding)?

**Encoding** (PRM 80-N2040-62 AA §2.3.1: Fig.13 p.13 — the 256-byte row
is 32 low words at bytes 0..127 then 32 high words at bytes 128..255;
Fig.15 / Table 4 p.14): `lo32 = {out_bias FP16 [31:16], scale FP16
[15:0]}`, `hi32 = {shape [10:8], out_bias extra4 [7:4], scale extra4
[3:0]}`; value = `(-1)^s * 2^(e-15) * (1 + frac10/1024 + n/16384)`, so on
the negative `-64.5` base (`0xD408`) nibble n gives
`out_bias = -(64.5 + n*2^-8)`. Worked example **n=8**:
`out_bias = -(64.5 + 8/256) = -64.53125`, lo cell `0xD4085400`,
hi cell `0x00000080`.

| mode | nibble | payload / purpose (`hypothesis_note=`) |
| --- | --- | --- |
| 31 | 0 | `extra4_sweep_n0_baseline` (mode-28 payload, Rs[8] forced 0) |
| 32 | 1 | `extra4_sweep_n1` |
| 33 | 2 | `extra4_sweep_n2` |
| 34 | 8 | `extra4_sweep_n8` |
| 35 | 15 | `extra4_sweep_n15` |
| 36 | 3 | `border_control_n3_between_2_and_8` |
| 37 | 0 | `exact_zero_control_fallback`: n=1 exact-zero alignment is off the 20-bit grid (0.25 nibble steps), so a=b=1.0 + out_bias=-64.0 gives target 0 exactly (proved exhaustively by `probe_bias20_test.c`) |
| 38 | 1 | `bias_readback_probe` (see below) |
| 39 | 0 | `e1_range_discriminator_base64p25`: base −64.25 (`0xD404`), target 64.25 grid units — E-1 range discriminator (see Phase E-1 below) |

All D modes: q=8 mandatory (DSP returns 14001, host exits 2 otherwise),
feedback disabled (single-CVT, Rs=1 only, §2.3.3 Table 6 no-feedback
recipe), scale 64 (`0x5400`), a=b=`0x3c04` (mode 37: `0x3c00`). Expected
observed bits = `fp16_round(scale*a^2 - (base + n*2^-8))` with base 64.5
(mode 37: 64.0; mode 39: 64.25) — the FP16
rounding is part of the math: §2.4 stores the FP16 convert state to
VTCM and Ch.5 says the extra four bits are internal to the convert state,
used only for feedback.

**Mode 38 readback.** After the convert the DSP stores bias set 0 back to
a 256B-aligned VTCM scratch (`mxmem2(Rs32)=bias`, `Q6_mxmem2_bias_A`,
§2.3.1 p.15) and returns `results[0]=lo(0)` (bytes 0..3),
`results[1]=hi(0)` (bytes 128..131 — Fig.13 splits the two halves into
the two vectors; they are not adjacent), `results[2]=results[3]=0`.

CPU oracle test (no SDK/device): `gcc -std=c11 -O2 -Wall -Wextra -Werror -o probe_bias20_test probe_bias20_test.c -lm && ./probe_bias20_test`.
Device runs: `.\run_experiment.ps1 -Provenance`,
then `.\run_experiment.ps1 -DeployAndRun -Q 8 -Mode 31` (likewise `-Mode 32`..`-Mode 39`). Executed results are in the battery section below.

## Device battery results (2026-10-20)

All numbers below are quoted from device transcripts
(`build/hmx_evidence/b_battery_console_full.txt` for battery 1 — the raw
per-case `[device]` console payload for all 27 cases — and
`b_battery3_20261020.txt` for battery 3; gitignored like all provenance —
the numbers and build ids are what matter). The battery-3 file is a
per-case summary (`CASE_EXIT ... exit=1` marks a failed case); the
probe-level `[device]` detail for the battery-3 reruns was streamed to
the run console and its repeated-case values agree with battery 1's raw
console. The mode-0 measurement detail (wanted residual negative,
`residual_bits=0x0000`, probe `exit_code=8`) is preserved in this
section and in the PR record; if any battery-3 row is ever formalized as
paper evidence, re-record it under the current harness for a durable raw
transcript. Provenance for every run (commit, dirty state, artifact
SHA-256 table) is in `build/hmx_evidence/b_provenance_*.txt`.

Run provenance:

- **Battery 1** (26 cases): built at commit `93738c5` with the Phase D
  code still uncommitted in the working tree (stamped `0x7dfcb259`;
  the tree was dirty — the Phase D probe changes were committed
  afterwards as `3bad274` / `19cc4c4`). Host-side identity +
  build-id fail-closed gate active from the start.
- **Battery 2** (8 cases): demonstrated the stale-skel fail-closed
  behaviour described above (every case exit 9, no experiment executed).
- **Battery 3** (11 cases): fresh committed build from a clean HEAD
  `0c63019` (stamped `0x0c630195`), every case `build_id_status=MATCH`.
- **Verdict-fix rerun** (commit `539bf68`, stamp `0x539bf682`): single
  case (q=8, mode 31) after the mode-31 selfcheck correction described
  below.

### Battery 1 — Phase C reproduction sweeps

`OBS` / `MATH` = `OBSERVED_REGRESSION_MATCH` / `MATH_ORACLE_MATCH`
(two separate facts; MATH is informational and does not change the exit
code). The math column quotes the *double-derived* expectation.

| mode | q (scale) | observed residual bits | value | OBS | MATH (math bits) |
| --- | --- | --- | --- | --- | --- |
| 30 | 8 | raw-convert control, no oracle | — | N/A | N/A |
| 20 | 6 | 0x2400 | 2^-6 = 4 grid units | PASS | PASS (0x2400) |
| 20 | 7 | 0x1c00 | 2^-8 = exactly 1 grid unit | PASS | PASS (0x1c00) |
| 20 | 8 | 0x1c00 | 1 unit (target +1/4 unit) | PASS | FAIL (0x1400) |
| 20 | 9 | 0x1c00 | 1 unit (target +1/16 unit) | PASS | FAIL (0x0c00) |
| 20 | 10 | 0x1c00 | 1 unit (target +1/256 unit) | PASS | FAIL (0x0400) |
| 23 | 6..10 | identical pattern to mode 20 | — | PASS | PASS at q=6,7; FAIL at q=8..10 |
| 21 | 6 | 0x0000 | zero-residual control | PASS | PASS |
| 22 | 6 | 0x5400, 0x0000 | zero-residual 2-CVT control | PASS | PASS |
| 24 | 6 (32) | 0x2000 | 2^-7 = 4×2^-9 on-grid | PASS | PASS |
| 25 | 6 (64) | 0x2400 | 2^-6 = 4×2^-8 on-grid | PASS | PASS |
| 26 | 6 (128) | 0x2800 | 2^-5 = 4×2^-7 on-grid | PASS | PASS |
| 27 | 8 (32) | 0x1800 | 2^-9 = 1 grid unit (target +1/4 unit) | PASS | FAIL (0x1000) |
| 28 | 8 (64) | 0x1c00 | 1 unit (target +1/4 unit) | PASS | FAIL (0x1400) |
| 29 | 8 (128) | 0x2000 | 1 unit (target +1/4 unit) | PASS | FAIL (0x1800) |
| 37 | 8 | 0x0000 | exact-zero control | PASS | PASS |
| 38 | 8 | readback lo=0xd4085400 hi=0x00000010 | configured payload echo | PASS | PASS (decode −64.50390625) |

Reconstruction facts worth recording: mode 20 q=6 reconstructs the
oracle **exactly** (`recovered_error=0`, residual on-grid); mode 20 q=8
reconstruction error is 4.58e-05 — **3× worse** than the plain FP16
baseline (1.53e-05) because the +1-unit ceiling overshoots the true
residual.

### Battery 3 — Phase D nibble sweep and the first mode-0 measurement

| mode | n | out_bias | math target | observed | hypothesis (bias grid) | OBS | MATH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 0 | −64.5 | +2^-10 = +1/4 unit | 0x1c00 (1 unit) | ceil | PASS | FAIL (0x1400) |
| 32 | 1 | −64.50390625 | −0.0029296875 | 0x0000 | clamp | PASS | FAIL (0x9a00) |
| 33 | 2 | −64.5078125 | −0.0068359375 | 0x0000 | clamp | PASS | FAIL (0x9f00) |
| 34 | 8 | −64.53125 | −0.0302734375 | 0x0000 | clamp | PASS | FAIL (0xa7c0) |
| 35 | 15 | −64.55859375 | −0.0576171875 | 0x0000 | clamp | PASS | FAIL (0xab60) |
| 36 | 3 | −64.51171875 | −0.0107421875 | 0x0000 | clamp | PASS | FAIL (0xa180) |
| 37 | 0 | −64.0 | 0 exactly | 0x0000 | 0 | PASS | PASS |
| 38 | 1 | −64.50390625 | readback | lo=0xd4085400 hi=0x00000010 | payload echo | PASS | PASS |

Battery 3 case exits: mode 30 → 0; (q6, mode 0) and (q8, mode 0) → 8;
mode 31 → 8 (selfcheck defect, see below); modes 32–38 → 0. The MATH
`FAIL` verdicts on modes 31–36 are **expected research facts**, not
harness failures: the whole point is that pure math does not derive the
observed values — the bias-grid model does.

The observed bits are identical between battery 1 and battery 3 for
every repeated case (two independent builds), e.g. battery 1 mode 31
`hi_bits=0x1c00` (judged FAIL there only because the host did not yet
carry the bias-grid expectation, added in `19cc4c4`).

**Mode-31 selfcheck defect found by battery 3.** The registered selfcheck
compared the *math* oracle (`expected_math[0]` = 0x1400) against the
measured plateau literal 0x1c00 — a comparison that by construction can
never pass, since the research result is precisely that math does not
derive the plateau. Fix (commit `539bf68`): the mode-31 selfcheck now
compares the hypothesis oracle `expected[0]` against the plateau
literal (`oracle_selfcheck=OK compare=hypothesis_vs_measured_plateau
literal=0x1c00 derived=0x1c00`), while modes 24–29 keep their original
math-pinning semantics. Re-verified on device at `539bf68`: exit 0,
`OBSERVED_REGRESSION_MATCH=PASS`, `MATH_ORACLE_MATCH=FAIL` (fact).

### Mode 0: first full device measurement of dynamic feedback

Mode 0 had never run against the hardened harness (battery 1 covered
modes 20–30 and 31–38 only). Result for both q=6 and q=8: both trials
return `residual_bits=0x0000`, so
`fp16_residual_recovery=NOT_CONFIRMED` and the case exits 8 after all
three deploy attempts — an honest failure, not masked. The residual
target the math oracle wants is **negative** (q=6: 0xa400 = −2^-6;
q=8: 0x9400 = −2^-10) because round-to-nearest places `hi` *below* the
scaled operand — and the device clamps non-positive fine convert
outputs to zero (the same law the controlled Phase D sweep established).
The original recovery recipe therefore fails *for a measured,
explainable reason*, not for a harness reason.
`DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED` remains the honest headline.

### Phase E-1 — range discriminator (mode 39)

Built at a clean HEAD `a7037c3` (stamped `0xa7037c38`), q=8, two trials
(`build/hmx_evidence/b_mode39_20261020.txt`):

| mode | n | out_bias | math target | observed | hypothesis | OBS | MATH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39 | 0 | −64.25 (`0xD404`) | 0.2509765625 = 64.25 units | **`0x3410`** (0.25390625 = 65 units), both trials | ceil | PASS | FAIL (`0x3404`) |

Oracle self-checks pinned both sides of the reviewer-corrected
discriminator pair: `oracle_selfcheck=OK literal=0x3410 derived=0x3410`
and `oracle_selfcheck_math_pin=OK literal=0x3404 derived_math=0x3404`
(identity MATCH, cleanup PASS, exit 0). The MATH `FAIL` is the expected
finding, not a harness failure: the device does not store the plain
correctly-rounded FP16 value.

**Verdict: the ceiling law is global in the measured range.** Every
target measured until now sat within ±1 grid unit of zero; the
64.25-unit target sits far from zero (two grid steps below the next
integer multiple of 64 units) and the device still stores
ceil(target)·grid, not the exact FP16. 0.2509765625 (= 64.25 units) is
itself off the 2^-8 grid, so storing `0x3404` would have falsified the
global-grid reading — it did not. The slot is a quantizer onto the
output-bias fine grid, not an encoder of arbitrary exactly-representable
FP16 values.

### Argument gate (host-side, device-verified)

`q` outside 6..10, `mode` outside 0..39, and phase-D modes with `q≠8`
are rejected before any device state is touched. Device-verified with
the same binary: arguments `5 0`, `11 31`, and `8 39` (invalid at the
time, before mode 39 was added in `db94439`) each print the usage
line and exit 2 with no capability query, no open, no HMX session
(`build/hmx_evidence/b_arggate_20261020.txt`).

## Observed quantization laws (device-derived hypothesis)

These laws summarize every measured convert output on this runtime.
They are a **model for the stored FP16**, double-checked per case above;
they are not claims about FP32 accuracy or HMX internal precision.

1. **FP16-only export.** The value stored to VTCM by a convert is FP16;
   the fine (20-bit, extra4) bits stay internal to the convert state and
   only feed the feedback path (PRM 80-N2040-62 AA, Ch.5 after Fig.25).
   Device data agrees: mode 38 readback shows the *configured* bias
   payload (including its extra4 nibble) is retained in the bias
   registers, while convert outputs never show sub-FP16 information.
2. **Output-bias fine grid.** The stored value lands on the fine grid of
   the *output bias*: 1 nibble LSB = 2^(e_obias − 14), where e_obias is
   the FP16 exponent field of the bias magnitude (−64/−64.5 → e=6 →
   2^-8). The scale sweep confirms the dependence on the bias magnitude:
   biases −32/−64/−128 give grids 2^-9/2^-8/2^-7, and the measured
   remainders 2^-11/2^-10/2^-9 all ceil to exactly one unit (modes
   27/28/29 → 0x1800/0x1c00/0x2000). The E-1 discriminator shows the
   same reading far from zero: bias −64.25 (`0xD404`, e=6 → 2^-8) with
   a 64.25-unit target stores 65 units (mode 39 → `0x3410`).
3. **Positive targets: ceiling**, not round-to-nearest and not
   round-half-up. A +1/4-unit target stores +1 unit (0x1c00 at e=6;
   nearest would store 0 — modes 20/23/28/31 at q=8..10). The
   far-from-zero mode-39 target also ceils (64.25 units → 65 units =
   `0x3410`), so the ceiling is not a small-magnitude artefact.
4. **Non-positive targets: clamp to zero.** 0x0000 for all negative
   fine targets (modes 32–36) and for the mode-0 feedback residual.
5. **On-grid targets pass through exactly.** q=6 (residual 2^-6 = 4
   units) reproduces math exactly, reconstruction error 0; the q=7
   remainder is exactly 1 unit (boundary case, math matches); mode 37's
   exact-zero target stores 0.
6. **The extra4 feedback bit does not change the stored value.** Mode 20
   trials 0/1 (Rs[8]=0/1) return identical bits at every measured q —
   the internal fine bits do not leak into the FP16 store either way.
7. **Reproducibility.** Every repeated case produced identical raw bits
   across battery 1 (dirty-tree build), battery 3 (clean committed
   build) and the `539bf68` rerun (three independent builds).

## Phase E feasibility memo

Where this leaves the original question (can two FP16 converts export an
accumulator's low-order information?):

- **Exact recovery is achievable only for corrections that are positive
  bias-grid multiples.** Measured: mode 20 q=6 (correction +4 units)
  reconstructs the oracle exactly (`recovered_error=0`). This is a real,
  reproducible capability of the two-convert recipe.
- **Sub-grid positives overshoot** (+1 unit; q=8..10 made reconstruction
  worse than the plain FP16 hi: 4.58e-05 vs 1.53e-05 — those residuals
  sat at r = 0.25g, below the 0.5g break-even of the ceil-error law
  formalized in the Phase E-2 section; sub-grid residuals with r > 0.5g
  *improve* under the same law), and
  **non-positive corrections vanish** (clamp to 0). Since round-to-
  nearest `hi` often sits *below* the scaled operand, the negative case
  is common — mode 0 hits it for both tested operands.
- **The fine bits never leave the convert state** through either route:
  not through the FP16 store (law 1), and not through the Rs[8] feedback
  bit (law 6).
- **Phase E-1 executed (2026-10-20): the ceiling is global.** Mode 39
  measured the only discriminating target defined so far (64.25 grid
  units, exactly representable in FP16 as 0x3404): the device stored
  0x3410 (65 units) in both trials. The quantization-scope question is
  closed for the measured range; what remains open is recipe
  feasibility (blockers above), not quantizer behaviour.
- **Generalization blockers** for a real GEMM/optimizer path: the grid
  varies with the per-tile bias magnitude (granularity is not uniform
  across accumulation ranges); corrections must be ≥ 1 unit and
  positive; the feedback recipe serializes converts (second convert
  consumes the first's state), which conflicts with tile-parallel
  accumulation; cancellation behaviour is untested by design
  (single-nonzero-element construction); VTCM budget is 12 KiB for
  32×32 croutons.
- **Minimal next experiment (Phase E-1 candidate, implemented as mode 39).**
  The current data pins the ceiling law only for *sub-unit* targets and
  pass-through for *integral* targets. One mode decides the scope: configure
  a bias base of −64.25 (same e=6, grid 2^-8) so the convert target is
  64.5009765625 − 64.25 = 0.2509765625 — i.e. **64.25 grid units**,
  exactly representable in FP16 as **0x3404** (exp field 13, frac10 = 4).
  Prediction set (bit values corrected 2026-10-20 after PR review: the
  original text wrote 0x3410/0x3450 here, which encode 0.25390625 (65
  units) / 0.26953125 (69 units) and are bitwise wrong here):

  | stored value | meaning |
  | --- | --- |
  | 0x3410 (0.25390625 = 65 units) | global ceiling law: the slot is a quantizer onto the bias grid |
  | 0x3404 (0.2509765625) | plain correctly-rounded FP16: the ceiling is only a small-magnitude effect; this value is itself off the 2^-8 grid (64.25 units), so a 0x3404 result falsifies a global grid reading |
  | anything else | third behaviour — analyze |

  One build, one run, no ABI changes. **Executed 2026-10-20 (mode 39,
  commit `a7037c3`, two trials): stored `0x3410` (0.25390625 = 65
  units) in both trials** (`b_mode39_20261020.txt`), with
  `OBSERVED_REGRESSION_MATCH=PASS` and `MATH_ORACLE_MATCH=FAIL`
  (`0x3404`, expected fact). Conclusion: the bias-grid ceiling is
  global; the stored FP16 is a quantizer output, and the scope of the
  quantization laws is not limited to small magnitudes.

## Phase E-2 feasibility assessment (2026-10-20)

Question: does a **practical FP16×2 computing path** exist on HMX V81
(this runtime) — either (A) harvesting the accumulator residual through
two converts, or (B) splitting the *inputs* into FP16 hi/lo pairs and
synthesizing the product from several plain HMX multiplies? Verdict
vocabulary: `POSSIBLE` / `CONDITIONALLY_POSSIBLE` / `NO_GO` /
`INSUFFICIENT_EVIDENCE`. This section adds no new probe mode and does
not touch any production path; device work beyond what is already
recorded requires explicit instruction.

### Evidence base (file-backed unless noted)

| fact | reference |
| --- | --- |
| Plain (no-output-bias) converts store round-to-nearest FP16 of the converted value: hi rows q=6..10 reproduce the double-derived RN bits exactly | `b_battery_console_full.txt` modes 20/23 hi rows; zero-residual controls 21/22 exact |
| 2-CVT chains retain the accumulator and store per convert-recipe order (64 then 64−64=0) | battery 1 mode 22 (both trials, raw console) |
| Stored = 0 for non-positive fine convert targets | modes 32–36 (battery 1 + battery 3), mode 0 both q (session-record detail; see battery-3 provenance note above) |
| Positive targets ceil onto the output-bias grid, including far from zero; grid g = 2^(e_obias−14) tracks the bias magnitude | modes 27/28/29 scale sweep; mode 39 E-1 (global boundary) |
| On-grid positive corrections reconstruct exactly (recovered_error = 0); sub-grid positive corrections overshoot q8: 4.58e-05 vs plain-FP16 1.53e-05 | mode 20 q=6 vs q=8 |
| `Rs[8]` fine feedback bit never changes the stored FP16; fine bits stay in the convert state | mode 20 both trials at every q; PRM Ch.5 after Fig.25 |
| Dynamic feedback (convert state → next convert's bias) never positively confirmed | `DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED` every case; mode 0 wanted residual negative → could not distinguish consumption-then-clamp from no-consumption |
| CPU split-precision simulation: the FP16 hi/lo split is NOT exact — lo-normal bound 2^-23·\|A\| holds (0 viol of 100058 normal-lo cases, max ratio 1.191511e-07 ≈ 2^-23·1.006); subnormal-lo degrades and at 2^-20..2^-15 lo is 0 in all 2400 sweep points (max rel 3.02e-02, bound breaks below ≈9.5e-07); float-arith split (G3, unit-corrected): double→float input rounding is the only floatization loss (e0 max ratio exactly 2^-24, 0/100058 exceed) and the float subtraction is Sterbenz-exact against the RN16-hi partner (e1 ≡ 0, 27 exact cancellations); the final recovery ratio exceeds 2^-23 in 942 lo-normal cases (max 1.760996e-07 ≈ 1.48×2^-23 — e0 + RN16(lo), not cancellation); the earlier 671-viol count was a units mismatch (relative vs absolute) and is withdrawn; reviewer example x = 1+2^-11−2^-23 → residual exactly 2^-23 (hi=0x3c00, lo=0x1000) | `hmx_fp16x2_split_sim.c` via `scripts/run_fp16x2_split_sim.ps1` (gcc gcc-15.2 mingw + clang-22.1.8 msvc byte-identical, seeds fixed) |
| B-1 device run (2026-10-20): ONE build + ONE run, mode 40 — two consecutive MMA operand packets into ONE accumulator, `mxclracc` once at start, no convert/store between batches, ONE final bias-0 convert at scale 64; trial 0 (+1)+(+2) stored **`0x5a00`** = 192 (accumulate branch of the pre-registered table), trial 1 (+2)+(−1) stored **`0x5400`** = 64 (mixed-sign batch additivity, positive sum); strict build-id all-agree gate PASS, exit 0, cleanup PASS — `TWO_MMA_ACCUMULATE_CONSISTENT` | `run_experiment.ps1 -DeployAndRun -Q 6 -Mode 40` (transcript `build/hmx_evidence/b1_run_20261020.txt`, gitignored) |
| B2 CPU stored-domain comparison (E-3A, G7): final-FP16-store applied to the SAME synthesis — plain stored vs B3 stored mean abs err 1.446613e-03 vs 8.484465e-04 @n16, 2.719522e-03 vs 1.489992e-03 @n64 (FP32-carried act: 1.71×/1.83×, worsened 0 %, B3 oracle-optimal bits 99.6–100 % vs plain 52.9–55.1 %); FP16-exact act (W-only): 1.22×, worsened 12.5–12.7 %; pinned B-2 scalar fixtures reproduce exactly (plain `0x5400` / B3 `0x5401` and `0x5400`/`0x53ff`; full-RN16 coincide — fixture-specific) | same sim, G7: `split_b2_stored_*`, `split_b2_fixture`, `b2_cpu_gate=PASS improves=4/4`; pre-store G5/G6 figures above are internal-domain, not end-to-end |
| B-2 device run (2026-10-20): ONE build + ONE run, mode 41 — three MMA packets (`Ahi·Whi`, `Ahi·Wlo`, `Alo·Whi`) into ONE accumulator, `mxclracc` once per subcase, no convert/store/clear between packets, ONE final bias-0 scale-64 convert + ONE store; trial 0 stored **`0x5401`** (64.0625), trial 1 stored **`0x53ff`** (63.96875) — both exactly the pre-registered 3-product bits, distinct from the plain context `0x5400`; strict build-id all-agree gate PASS, exit 0, cleanup PASS, `DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED` — `B2_THREE_PRODUCT_CONFIRMED` at 1-lane/1-run scope | `run_experiment.ps1 -DeployAndRun -Q 6 -Mode 41` (transcript `build/hmx_evidence/b2_run_20261020.txt`, gitignored) |
| CPU 3-GEMM synthesis (G5/G6): both-operands-split mean rel err 5.66e-07 @n=64 (4.30e-06 @n=16) vs plain 2.57e-03 / 1.53e-02 → ~4.5e3× mean improvement; the 4-term variant is not better (5.73e-07); chained single-accumulator vs separate float accumulators equivalent (6.79e-07; combine loss subdominant: mean 2.83e-08, max 1.39e-07); W-only split (Alo=0 activations) gains only 1.5–1.8× | same sim |

### Path A — harvest hi/lo from one accumulator through two converts

**Verdict: `NO_GO` as a general recipe.** Narrow static case below is
`POSSIBLE` — but it is already covered by the plain FP16 baseline, so
the two-convert dynamic-feedback variant adds nothing measured.

*Arithmetic of the export (from the measured laws):* the exported value
is the second convert's stored target, and every measured store obeys

```
store(x) = 0                      for x ≤ 0        (law 4)
store(x) = ceil(x/g)·g            for x > 0        (laws 2/3, E-1)
g       = 2^(e_b − 14)            substrate-tied   (g ∝ |b| ≈ substrate)
```

For the exported correction r = s − RN(s) the ceil step is a quantizer
onto the bias grid, and the reconstruction error after export is

```
E(r) = g·ceil(r/g) − r     (for r > 0; E = 0 exactly when r ≡ 0 mod g;
                            E ∈ [0, g) always, ties away from on-grid)
```

so the exportable-correction table with the exact error column is:

| wanted correction r | exported | error E(r) |
| --- | --- | --- |
| r ≤ 0 (common; mode 0 hit it for both q) | 0 | the whole correction is lost (clamp) |
| r = k·g, k ≥ 1 (on-grid, positive) | exactly r | 0 — exact (measured q=6: recovered_error = 0) |
| 0 < r < g | ceil → +1 unit | E = g − r |
| r ≥ g, off-grid | ceil | 0 ≤ E < g, and E < r always — the correction improves unconditionally (E = 0 exactly on-grid) |

**Sign of the ceil law (review corrections 2026-10-20, two rounds — the
first draft claimed sub-grid positives always worsen, which was wrong;
the second un-scoped the frac condition, which was also wrong):**

- For **0 < r < g**: E(r) = g − r. Improvement (E < r) exactly when
  frac(r/g) > 0.5; worsening when frac(r/g) < 0.5; break-even at 0.5.
  Example: r = 0.75g exports with E = 0.25g — the residual error
  shrinks to a third of the uncorrected one. The measured q=8 case
  sits on the worsening side: r = 0.25g ⇒ E = 0.75g ⇒ 4.58e-05
  (= 0.75g/64) reconstruction error vs the plain-FP16 baseline
  r/64 = 1.53e-05 (3× worse).
- For **r ≥ g**: E(r) < g ≤ r (strict for every off-grid r > g) — a
  positive residual of one full grid unit or more is therefore
  *unconditionally* improved by the ceil export; and on-grid r = k·g
  exports with E = 0 exactly. The improving/worsening distinction the
  frac(r/g) condition trades on exists ONLY inside the first grid
  cell (0 < r < g).
- A positive on-grid residual exports error-free; every other sign /
  sub-unit-fraction combination is at best partial (E < r) and at
  worst a regression (E > r) or a total loss (r ≤ 0).

So A keeps NO_GO as a *general* exact-recovery recipe — E is never 0
except on-grid, the negative half is clamped away outright, the
residual distribution of a real training GEMM is not known a priori,
and the recipe's ingredients (dynamic feedback) are unconfirmed. But
the honest characterization is: the ceil quantizer flips between
improving and worsening only where the residual sits inside its FIRST
grid cell (0 < r < g) and improves unconditionally above one unit
(E < g ≤ r) — so a *residual-distribution-aware* variant is not ruled
out by the mathematics alone; it is ruled out by the missing
mechanism (see below) and by the fact that steering residuals
(positive, ≥ one grid unit — the unconditionally-improving region) is
exactly what the static-bias baseline already controls.

*Why no additional instruction path fixes this* (all three constraints
are structural in the measured behaviour): (i) the store is FP16-only —
the fine (16+4) bits never leave the convert state except through the
feedback input of the *next* convert (law 1/6); (ii) that next convert
re-quantizes its own store onto its own bias lattice — and its bias must
cancel the same substrate, so its g is tied to the same magnitude (the
E-1 result closes the door on any far-from-zero escape); (iii) the
sign clamp is unconditional for the store. The V81 convert surface
offers no documented route that writes sub-lattice or signed residual
information to memory. Via the bias payload the *programmer* can
pre-align the substrate so that s − b lands on-grid and positive —
that is exactly the mode 39/20 q6 family — and it requires no dynamic
feedback at all, meaning the baseline already delivers every
correction A can deliver.

*What A would need to become general* (all unmet by the documented V81
surface): a signed store bypassing the clamp, a store lattice finer than
the substrate-tied grid, or a state-readout path other than
state→next-bias feedback. None exists in the PRM-documented convert
recipes measured here.

*Minimal experiment if anyone wants to revive A* (**A-2**, not
implemented this commit): program the recipe with a *positive* ≥1-unit
residual and bias set 1 empty (bias 0) with `Rs[3:2]=1` feedback
enabled on the second convert. Prediction matrix: if the convert state
is truly consumed as a bias, the second store differs from plain RN by
the state's contribution; if nothing changes (plain RN hi), the
dynamic-feedback hypothesis is dead for the store and A closes
entirely. Single mode, one build, no ABI change. Only run it on
explicit instruction.

### Path B — split inputs into FP16 hi/lo, synthesize from plain HMX multiplies

**Verdict: `CONDITIONALLY_POSSIBLE`** — the composition is standard
FP16 `hi/lo` splitting and *none of the measured HMX landmines apply to
it*; it is gated on confirming one HMX primitive and on a cost
measurement, not on new HMX knowledge.

*Model:* A = Ahi + Alo, W = Whi + Wlo. **The split is NOT exact in
general** (review correction 2026-10-20, with CPU simulation — see the
evidence table and `hmx_fp16x2_split_sim.c`). With Ahi = RN16(A) and
Alo = RN16(A − Ahi) computed with a ≥FP32 subtraction, recovery
satisfies |hi + lo − A| ≤ 2^-23·|A| only while lo lands in the FP16
*normal* range — simulated max ratio 1.191511e-07 ≈ 2^-23·1.006 with
zero violations in the lo-normal domain — and degrades as soon as the
residual A − Ahi falls below the FP16 smallest exponent: in the
2^-20..2^-15 magnitude sweep lo is 0 in **all** 2400 points (the split
degenerates to the plain FP16 value) with max relative recovery error
3.02e-02, and the smallest magnitude breaking the 2^-23·|A| bound is
≈9.5e-07. The reviewer's example is confirmed numerically: x =
1 + 2^-11 − 2^-23 splits to hi = 0x3c00 (ties-to-even), lo = 0x1000,
leaving recovery error exactly 2^-23 (rel 1.19e-07). Computing the
split subtraction itself in float instead of double is NOT an extra
cancellation hazard: the unit-corrected G3 decomposition shows the
float subtraction is Sterbenz-exact for the RN16-hi partner (e1 ≡ 0 in
all 100058 cases, 27 exact cancellations included) and the only
floatization loss is the double→float input rounding (e0, relative max
exactly 2^-24, 0 of 100058 exceed the RN bound). The float variant's
final recovery exceeds the relative 2^-23 bound in 942/100058
lo-normal cases (max 1.760996e-07 ≈ 1.48×2^-23): that half-ulp excess
is carried by the input floatization (e0) plus RN16(lo), NOT by
subtraction cancellation. Practical reading: the split SUBTRACTION itself introduces no extra
rounding (e1 ≡ 0); the 942 excesses mirror the input floatization that
a float-resident pipeline pays only when its source data is wider than
float (double→float first, e0 > 0; already-float operands pay e0 = 0).
The earlier "the subtraction must be ≥FP32" conclusion was an artefact
of the mis-united 671 count and is withdrawn; the G1 double-arith
bound above stays the strict reference, and the lo-normal domain
scoping of every 2^-23-class claim is unchanged. Then

```
Y ≈ Ahi·Whi + (Ahi·Wlo + Alo·Whi) [+ Alo·Wlo]      (3 or 4 plain multiplies)
```

The only convert in the whole path is the **final** commit with
**bias 0** — precisely the case measured to behave as plain RN-FP16
(hi rows q=6..10). No output-bias converts ⇒ no grid ceiling, no clamp
(laws 2–4 target the output-bias recipe only). The context of the
original problem statement "FP16×2 HMX arithmetic" is satisfied by
reusing the FAU's own accumulation instead of trying to smuggle bits
out of the convert state.

*Error model (CPU-simulated, deterministic; device numbers still
pending):* per-term products of two FP16 values are exact in FP32;
accumulation in the FAU is FP32 (PRM HMX chapter; accumulation format
citation to re-verify before implementation). Simulated synthesis
(`run_fp16x2_split_sim.ps1`, gcc gcc-15 mingw and clang-22 msvc agree
byte-identically, fixed seeds — n ∈ {16, 64}, elements in [−2, 2] plus
hard-case entries): plain-FP16 mean relative error 2.57e-03 (n=64) /
1.53e-02 (n=16) vs the both-operands-split 3-GEMM mean 5.66e-07 (n=64)
/ 4.30e-06 (n=16) — a ~4.5e3× (n=64) mean improvement, max errors
2.39 → 1.44e-04 (n=64). Adding the 4th product (Alo·Wlo) is not
measurably better (n=64 mean 5.73e-07), so 3 GEMMs are the baseline
recipe; accumulating all passes into ONE chained float accumulator
instead of separate ones is statistically equivalent (n=64 mean
6.79e-07), and the loss from committing partials to separate float
accumulators and combining in float is subdominant (mean 2.83e-08,
max 1.39e-07). This does **not** reach CPU double precision (~1e-16
rel) — the double oracle remains the reference, and Path B's honest
expectation is FP32-class accuracy, not double. Device-measured error
numbers are `INSUFFICIENT_EVIDENCE` until a run exists.

**Stored-domain correction (Phase E-3A, 2026-10-20 — reviewer-mandated
re-evaluation):** ALL figures above to this point are *pre-store*
(internal float/float-domain) — the measured V81 path commits the
output through one final FP16 convert, so they are NOT end-to-end
stored-output accuracy and the ~4.5e3× figure must not be quoted as the
delivered improvement. Re-running the SAME synthesis with the store
step applied (`split_b2_stored_*`, G7: same tensors, seeds and double
oracle, stored = RN16 before comparison):
`b2_cpu_gate=PASS improves=4/4 cells`. In the FP32-carried-activation
regime the stored 3-GEMM beats the stored plain pipeline by **1.71×
@n=16 / 1.83× @n=64 in mean absolute error** (1.446613e-03 →
8.484465e-04 and 2.719522e-03 → 1.489992e-03), with **zero worsened
elements** and **99.6–100 % oracle-optimal stored bits** (plain stored:
52.9–55.1 %); RMSE improves 1.47× / 1.59×. With FP16-exact activations
(Alo ≡ 0, W-only split) the stored gain shrinks to ~1.22× and
**12.5–12.7 % of elements worsen** — the dominant stored-domain
difference comes from the activation hi/lo pair, and B-1🔗's
FP16-exact case is where the store erases most of the pre-store edge.
The decisive stored-domain quantity is the oracle-optimal bit rate: in
the FP32-carried regime the B3 synthesis essentially always lands on
the exact FP16 the true value wants.

*Cost model (analytic):*

| item | Path B multiplier vs plain FP16 GEMM |
| --- | --- |
| MMA compute | 3× (optionally 4× with the negligible Alo·Wlo term) |
| weight storage | 2× (Whi, Wlo — static, split offline once) |
| activation traffic | 2× (Ahi, Alo) + the split op itself (elementwise, FP32 domain) |
| convert endpoints | unchanged vs baseline (one final RN commit; the baseline already converts its output) |
| VTCM | two FP16 operand tiles must co-reside per synthesis; the probe's 12 KiB figure is the *probe's* operating envelope, not a device statement |
| bias/registers | one plain bias row passed repeatedly; no per-pass output-bias tuning |

*Implementation requirement (correctness-critical):* the partial
results must NOT be committed to FP16 between passes. Either chain all
three multiplies into one accumulator (single final convert) or commit
them as FP32 tensors and combine on HVX/CPU; a half-measure that
converts partials to FP16 throws away exactly the precision the
splitting bought.

*Input-precision requirement (decisive for the accuracy claim):* the
split only helps where the operand tensors carry *more* than FP16
information. Weights do — they are static and split offline once from
the FP32 checkpoint. Activations in today's FP16 pipeline are already
RN-FP16: their hi/lo split yields Alo = 0 and recovers nothing. To
realize the FP32-class accuracy the graph must *carry FP32-class
activations between layers* (or FP16 hi/lo pairs — the actual "FP16×2
pipeline"), which moves the cost profile to activation traffic and
storage and turns the accuracy comparison into a pipeline-design
question, not just a kernel question. With FP16-exact activations and
split weights only (simulated W-only variant: Alo ≡ 0), the gain over
plain is small and quantitative: mean relative error 8.36e-03 vs
plain 1.53e-02 at n=16 (1.8×) and 1.69e-03 vs 2.57e-03 at n=64
(1.5×) — the activation-rounding term (2^-11) stays dominant, as the
conditional model predicted.

*Minimal experiment, now implemented and user-approved* (**B-1**, mode
40; approval PR #14 comment 6290840051 — 1 build, 1 device run, both
trials inside the one run): one mode runs two MMA batches into the same
accumulator within a single kernel call — `mxclracc` exactly ONCE per
trial, at the start; then two consecutive operand packets (second
activation row pre-built in a second tile; weight row = exact 1.0 so
each MMA contributes only its activation scalar); NO convert and NO
FP16 store between the packets; then ONE bias-0 convert (the measured
plain RN-FP16 case) with scale 64. The harness's two trials switch the
operand constants:

| trial | scalars | accumulate ⇒ stored | overwrite ⇒ stored |
| --- | --- | --- | --- |
| 0 (primary) | +1, +2 | (1+2)·64 = 192 = `0x5a00` | 2·64 = 128 = `0x5800` |
| 1 (contrast) | +2, −1 | (2−1)·64 = 64 = `0x5400` | −64 ⇒ `0xd400` or clamp `0x0000` (negative bias-0 convert is itself unmeasured — contrast only) |

Trial 0 (positive-only operands) is the primary judgment basis; trial 1
is the negative/cancellation contrast only (an overwrite result there
is indistinguishable from a clamp, by design of the contrast).
Prediction matrix: both trials ACCUMULATE ⇒ chained batch accumulation
works and Path B's core primitive is device-consistent; OVERWRITE ⇒ B
needs separate accumulators per pass (still possible, costlier
combine). The 2-CVT retention measured in mode 22 already shows
converts do not disturb the accumulator; what B-1 isolates is the
*MMA-batch↔ACC* additivity across batches, which no recorded mode
exercises. **B-1 success does not certify Path B for production**: the
next step is a cost/accuracy measurement of the real 3-GEMM synthesis
(separate work item, user-gated, not part of this run).

**B-1 result (mode 40 device run, 2026-10-20; one build + one run):**
**`TWO_MMA_ACCUMULATE_CONSISTENT`** — both stored bits landed exactly
on the accumulate predictions: trial 0 (+1)+(+2) → **`0x5a00`**
(192 = 3·64), trial 1 (+2)+(−1) → **`0x5400`** (64 = 1·64). The
negative second contribution is therefore carried inside the
accumulator (mixed-sign batch additivity, positive final sum), and the
single bias-0 convert read the signed sum as plain RN-FP16 exactly as
measured before. Strict build-id gate PASS (host = DSP = same id,
all-agree), exit code 0, resource cleanup PASS. Claim scope: ONE run,
ONE device, exact-representable scalar operands (each packet one
scalar, sums exactly representable on the convert grid) — the
*MMA-batch ↔ ACC* additivity across operand reloads is device-confirmed
at this minimal shape. General multi-tile/multi-channel additivity and
the negative FINAL store of the bias-0 convert remain unmeasured (the
ACC stayed positive in both trials, so the contrast's overwrite branch
never presented a negative store). Path B's B-1 gate is met at this
minimal scope; this still does not certify production Path B — the
next step is the cost/accuracy measurement of the real 3-GEMM
synthesis (separate, user-gated work item).

### Phase E-3 (B-2): does the improved synthesis survive the final FP16 store?

**E-3A (CPU only, mandatory gate — executed 2026-10-20, `PASS`):** the
simulator's G7 group re-runs the SAME G5/G6 synthesis tensors with the
final `cvt.hf`-committed FP16 store applied to every variant before
comparison. Verdict: **stored-domain B3 improves on stored plain in
4/4 (n, regime) cells — 1.71× @n=16 and 1.83× @n=64 mean absolute
error in the FP32-carried regime with zero worsened elements
(oracle-optimal stored bits 99.6–100 % vs plain 52.9–55.1 %); W-only
(FP16-exact activations) gains only 1.22× with 12.5–12.7 % worsened.**
The pre-store ~4.5e3× figure is internal-domain only and is expressly
NOT the end-to-end expectation (see the stored-domain correction in the
error model above). The E-3A gate for device work is therefore met.

**E-3B (mode 41, user-approved; exactly ONE build + ONE device run):**
three MMA packets accumulate `Ahi·Whi → Ahi·Wlo → Alo·Whi` into the
SAME ACC (no convert / FP16 store / clear between packets — the
additivity E-2/B-1 device-confirmed), then ONE bias-0 scale-64 convert
(feedback off) and ONE store. Pre-registered, user-fixed fixture table
(expected bits are pinned NOW; retuning them after the measurement is
forbidden):

| subcase | FP32 inputs A=W | Ahi=Whi | Alo=Wlo | plain 1-product (context) | 3-product expected store | full-product RN16 |
| --- | --- | --- | --- | --- | --- | --- |
| 0, positive | `1+2^-11` | 1 = `0x3c00` | `+2^-11` = `0x1000` | `0x5400` (64) | **`0x5401`** (64.0625) | `0x5401` |
| 1, cancellation-sensitive | `1−2^-12` | 1 = `0x3c00` | `−2^-12` = `0x8c00` | `0x5400` (64) | **`0x53ff`** (63.96875) | `0x53ff` |

Exact full scaled products: 64.0625152587890625 and
63.968753814697265625; the three-term approximation drops `lo·lo` and
still coincides with the full-product RN16 for THESE fixtures — a
fixture-specific coincidence, not evidence the `lo·lo` term is
universally dispensable. Interpretation rules (pre-registered): both
subcases land on the expected bits `0x5401`/`0x53ff` ⇒ **B-2 three-product internal synthesis → a distinguishably improved
single FP16 store works at 1-lane, 1-run scope**; a `0x5400` or other
result ⇒ diagnose ACC behaviour, data layout and rounding — NO
automatic NO_GO on Path B; negative final FP16 store remains untested
(no signed-range guarantee). E-3C (real multi-lane 3-GEMM accuracy +
cost) is decision-only, NOT authorized by this stage.

**B-2 result (mode 41 device run, 2026-10-20; one build + one run,
build id `0xe99839a1`, host = DSP all-agree):**
**`B2_THREE_PRODUCT_CONFIRMED`** — both subcases landed exactly on the
pre-registered bits: trial 0 → **`0x5401`** (64.0625), trial 1 →
**`0x53ff`** (63.96875), in both cases differing from the plain
1-product context (`0x5400`, 64) exactly as the CPU fixtures
predicted. The three MMA packets (`Ahi·Whi`, `Ahi·Wlo`, `Alo·Whi`,
including the mixed-sign cancellation accumulator of trial 1:
1 − 2^-12 − 2^-12) accumulated into the SAME ACC with no intermediate
convert / FP16 store / clear, and the single bias-0 scale-64 convert
stored `RN16(ACC·64)` exactly as simulated. Exit code 0, identity and
resource-cleanup PASS, `DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED`
maintained. Claim scope: ONE run, ONE device, ONE lane, scalar
fixtures with exact-representable store targets — what is confirmed is
that the HMX ACC synthesizes the three-product sum *through the final
FP16 store*, distinguishable from the plain FP16 pipeline output at
these fixtures. General multi-lane accuracy/cost (E-3C), performance,
and negative-final-store behaviour remain unmeasured; no
production/permission claim.

### Overall comparison against the plain FP16 baseline and the double oracle

| criterion | plain FP16 baseline (today) | Path A (2-convert harvest) | Path B (input split) | CPU double oracle |
| --- | --- | --- | --- | --- |
| accuracy vs double | input-rounding 2^-11 term dominates | equal-or-worse in general (measured); exactly equal on on-grid positives only; for 0 < r < g the ceil export improves iff frac(r/g) > 0.5, for r ≥ g it always improves (E < g ≤ r) — distribution-dependent in the sub-unit cell, mechanism (dynamic feedback) missing | internal pre-store: mean rel err 5.66e-07 @n=64 vs plain 2.57e-03 (~4.5e3× — internal-domain only, NOT end-to-end); STORED-domain (final FP16 store): 1.71×@n16 / 1.83×@n64 mean-abs vs stored plain (FP32-carried act, zero worsened, 99.6–100 % oracle-optimal bits); W-only 1.22× (12.5–12.7 % worsened); split not exact (2^-23-class residue + subnormal-lo degradation); device scalar-fixture stored bits CONFIRMED (mode 41: `0x5401`/`0x53ff` vs plain `0x5400`, 1-lane/1-run); statistical multi-lane device accuracy `INSUFFICIENT_EVIDENCE` (E-3C decision-only) | reference |
| negative residual corrections | falls out naturally of RN | lost (clamp, measured) | unaffected (no clamp path involved) | — |
| sub-lattice corrections | lost | lost (grid ceiling is global — E-1) | preserved inside FP32 domain | — |
| extra compute | 0 | 2 converts + bias re-inits | 3× MMA + split op | — |
| new HMX knowledge required | none | none — verdict derivable from laws | B-1 primitive — device-confirmed at minimal shape (mode 40, 2026-10-20); B-2 three-product-through-store — device-confirmed at 1-lane/1-run (mode 41, 2026-10-20) | — |
| verdict | baseline | general: `NO_GO`; on-grid static: `POSSIBLE` (= baseline capability); dynamic feedback: `INSUFFICIENT_EVIDENCE` (A-2 only) | `CONDITIONALLY_POSSIBLE` (B-1 and B-2 both confirmed at minimal/1-lane scope; E-3C multi-lane cost/accuracy, performance and production integration NOT measured and separately gated) | — |

### Decision

- Record Path A as **NO_GO (general)**; keep the exact-recovery
  positive-on-grid capability as the *already-established* static-bias
  behaviour; archive A-2 as the only worthwhile A follow-up, executed
  solely on explicit instruction.
- Path B is the only route consistent with every measured HMX law.
  (1) B-1 executed (mode 40, `TWO_MMA_ACCUMULATE_CONSISTENT` above);
  (2) Phase E-3 approved 2026-10-20 (PR #14 comment 6293114616):
  E-3A stored-domain CPU gate **PASS (4/4)**, E-3B executed the same
  day (mode 41, one build + one run, `B2_THREE_PRODUCT_CONFIRMED`
  above) — the three-product synthesis survives the final FP16 store
  at 1-lane/1-run scope; E-3C (multi-lane 3-GEMM accuracy + cost) is
  decision-only and NOT authorized by this stage; production
  integration, benchmarking, multi-tile implementation and long
  training remain out of scope.
- Mode 40 is added under the standing policy — new modes only for
  clearly discriminating minimal experiments, only on explicit
  instruction; this mode exists by exactly that gate.
