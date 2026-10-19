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
- a round trip of **all 63488 finite binary16 values**,
- an independent cross-check against the compiler's own `_Float16`
  conversion when the toolchain provides it: all 63488 exactly
  representable values **plus** midpoints between adjacent values and
  points just below/above them, and the overflow boundary (about 158722
  comparisons),
- the expected hi / remainder table for q=6..10 and the reconstruction
  identity.

`fp16_util.h` is shared by the host runner and the self test, so the
oracle and its test cannot drift apart.

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

If the first trial instead returns `residual_bits=0x0000`, or another
value, then the hypothesis is **not confirmed** on that runtime.
A single passing example does not establish universal FP32 or 37-bit
recovery. Test negative results, underflow, cancellation, and random
matrices before applying this to Muon. No downstream optimizer switches
are included.

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
`@file` response is used instead), runs the identity self test, and
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
Device runs (none executed for Phase D): `.\run_experiment.ps1 -Provenance`,
then `.\run_experiment.ps1 -DeployAndRun -Q 8 -Mode 31` (likewise `-Mode 32`..`-Mode 38`).
