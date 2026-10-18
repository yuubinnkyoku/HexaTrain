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
compilation explicitly enables `-mhmx`. This experiment was **not built
or executed** in the authoring environment, so actual SDK compatibility
and device resource acquisition remain to be checked.

Deploy the generated `libhexatrain_hmx_acc_residual.so` (AArch64 stub),
`libhexatrain_hmx_acc_residual_skel.so` (V81 DSP skel) and
`hexatrain_hmx_acc_residual_test` (AArch64 executable) using the same
CDSP unsigned-PD / FastRPC procedure as the existing HexaTrain HVX probe.
Make the skel discoverable in your device's configured DSP library path.
Run `hexatrain_hmx_acc_residual_test` without arguments.

The host **rejects non-V81 CDSP** rather than silently emulating the
operation on a CPU. The DSP requests HMX plus 12 KiB of VTCM, and errors
out if allocation, 2-KiB alignment, HMX locking, or release fails.

## Interpreting the output

The host reports raw FP16 bits, FP16-decoded values, a CPU-double
reference, reconstruction error and the ordinary FP16 output error.

```text
extra4=0 hi_bits=0x5402 residual_bits=0x8400 ...
fp16_residual_recovery=SUPPORTED
```

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
