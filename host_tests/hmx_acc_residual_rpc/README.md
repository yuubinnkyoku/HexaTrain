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
