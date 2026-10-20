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

All D modes: q=8 mandatory (DSP returns 14001, host exits 2 otherwise),
feedback disabled (single-CVT, Rs=1 only, §2.3.3 Table 6 no-feedback
recipe), scale 64 (`0x5400`), a=b=`0x3c04` (mode 37: `0x3c00`). Expected
observed bits = `fp16_round(scale*a^2 - (64.5 + n*2^-8))` — the FP16
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
then `.\run_experiment.ps1 -DeployAndRun -Q 8 -Mode 31` (likewise `-Mode 32`..`-Mode 38`). Executed results are in the battery section below.

## Device battery results (2026-10-20)

All numbers below are quoted from device transcripts
(`build/hmx_evidence/b_battery_console_full.txt` for battery 1 and
`b_battery3_20261020.txt` for battery 3; gitignored like all provenance —
the numbers and build ids are what matter). Provenance for every run
(commit, dirty state, artifact SHA-256 table) is in
`build/hmx_evidence/b_provenance_*.txt`.

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

### Argument gate (host-side, device-verified)

`q` outside 6..10, `mode` outside 0..38, and phase-D modes with `q≠8`
are rejected before any device state is touched. Device-verified with
the same binary: arguments `5 0`, `11 31`, `8 39` each print the usage
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
   27/28/29 → 0x1800/0x1c00/0x2000).
3. **Positive targets: ceiling**, not round-to-nearest and not
   round-half-up. A +1/4-unit target stores +1 unit (0x1c00 at e=6;
   nearest would store 0 — modes 20/23/28/31 at q=8..10).
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
- **Sub-grid positives overshoot** (+1 unit; q=8..10 make reconstruction
  strictly worse than the plain FP16 hi: 4.58e-05 vs 1.53e-05), and
  **non-positive corrections vanish** (clamp to 0). Since round-to-
  nearest `hi` often sits *below* the scaled operand, the negative case
  is common — mode 0 hits it for both tested operands.
- **The fine bits never leave the convert state** through either route:
  not through the FP16 store (law 1), and not through the Rs[8] feedback
  bit (law 6).
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

  One build, one run, no ABI changes.
