# Nicopedia CPU/HTP precision mitigation study (L19 seed1)

Status: **MEASURED-2026-08-08** (protocol fixed before measurement; private
protocol/state under
`build/private-diagnostics/nicopedia-htp-precision-mitigation/`).

Scope: Nicopedia L19 seed1 (V=256, T=32, D=16, FFN=32, H=2), checkpoints
`htp-seed1-l19-step320.ckpt` and `htp-seed1-l19-step1000.ckpt`, QAIRT
`2.48.40.260702` build `2.48.40.260702151143`, NX741J SM8850 HTP V81.
The legacy parity gate, checkpoints, CPU reference, and graph math are
unchanged; no additional training was performed.

## 1. Question

The step-1000 checkpoint fails the legacy parity gate on `parity_13`
(raw max abs 4.28e-2 > 2e-2; centered RMS 2.24e-2). Previous localization
(docs/qnn-nicopedia-htp-divergence-localization.md) established
GRADUAL_ACCUMULATION: a ~1e-3 relative element-wise rounding spread across
all matmul/LayerNorm arithmetic, accumulating through 19 residual blocks,
consistent with lower-effective-precision HTP kernel arithmetic. This study asks whether
any precision control exposed by QAIRT 2.48.40.260702 can reduce that error on
the HTP side itself, keeping the same checkpoint, CPU reference, prefixes, and
legacy parity policy.

## 2. QAIRT precision-control inventory

Audited the pinned SDK headers (`include/QNN/HTP/*.h`) and found the
following graph-level numeric-path controls:

| option | enum | value type |
|---|---|---|
| `QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION` | 2 | `Qnn_Precision_t` (FLOAT32/FLOAT16) |
| `QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION` | 18 | bool |
| `QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING` | 12 | bool |
| `QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION` | 15 | bool |

No per-op accumulation-precision parameter exists for MatMul / ReduceMean /
ElementWiseRsqrt in this SDK; no FP32-accumulation flag is exposed. The HTP
backend executes FP32-declared tensors at a backend-chosen precision; the
codebase previously documented (qnn_runtime_qairt.cpp) that HTP elementwise
kernels behave near the FP16 square limit even for FP32-declared tensors.

## 3. Implementation (numeric-path-only diagnostics)

Runtime options now carry the four graph configs plus an FP16-NATIVE-tensor
diagnostic switch; they are plumbed through `GenerateConfig`, the JNI bridge,
the debug intent (`phonelm.htp_graph_*` extras), and the PowerShell runner
(`-HtpGraphPrecisionMode`, `-HtpGraphPrecisionCompensation`,
`-HtpGraphWeightsPacking`, `-HtpGraphAdvancedActivationFusion`,
`-HtpNativeTensorFp16`). Default values (all unset) produce the exact
established graph (nullptr config array), so existing behavior is unchanged —
verified by bit-identical parity metrics against the previous milestone.

The config array is delivered twice: at `QnnGraph_create` and, when the
interface exposes it, again through `QnnGraph_setConfig` after creation, to
rule out a delivery-path artifact. The device report echoes the requested
settings (`htp_graph_precision_mode` etc.) and the runtime diagnostics echo
the delivered config (`htp_graph_precision=...`), so delivery is verified in
every run.

## 4. Measured outcome (step1000, parity_13)

All runs: fixed serial identity (NX741J SM8850), thermal 0, battery 87%,
checkpoint identity verified privately, `cpu_fallback=false`.

| candidate | precision | comp | packing | fusion | native fp16 | parity_13 raw | centered RMS | prob L1 |
|---|---|---|---|---|---|---|---|---|
| baseline (unset) | 0 | 0 | 0 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| FP32 config | 2 | 0 | 0 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| FP16 config | 1 | 0 | 0 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| precision compensation | 0 | 2 | 0 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| weights packing off | 0 | 0 | 1 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| activation fusion off | 0 | 0 | 0 | 1 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| FP32 via graphSetConfig | 2 | 0 | 0 | 0 | false | 4.281e-2 | 2.235e-2 | 2.977e-2 |
| NATIVE tensor FP16 | 0 | 0 | 0 | 0 | true | — (graphFinalize=1002) | — | — |

Every config option produced bit-identical parity metrics (all 20 prefixes and
the 8 AR steps, not just parity_13). Declaring the internal NATIVE tensors as
FP16 while keeping APP/STATIC tensors FP32 was rejected by the backend at
graph finalize (QNN error code 1002, mapped to `QNN_COMMON_ERROR_MEM_ALLOC` in
the pinned headers; interpreted here as an unsupported mixed-precision graph),
so an all-FP16 or mixed-precision graph is not a viable execution path either.
The internal kernel precision was not directly observed; the bit-identical
results across FP32/FP16/unset configs are consistent with (but do not prove)
FP16-equivalent internal execution.

## 5. Conclusion

The HTP backend in QAIRT 2.48.40.260702 produced bit-identical outputs for
all five tested graph-level precision controls (FP32 and FP16 precision modes,
precision compensation, weights packing, activation fusion), and the only
dtype change the backend accepted (all-FP32) is the status quo. The evidence
is consistent with the backend executing FP32-declared graphs at
FP16-equivalent internal precision, but the internal kernel precision was not
directly observed and no API-visible FP32 execution path exists in this SDK
version.

**Result: no precision mitigation was found through the tested online graph
precision controls in this SDK.** This does not exclude every context,
serialization, partitioning, or future-SDK execution route. A follow-up audit
of the documented context graph-splitting option is reported in
`qnn-nicopedia-htp-execution-path.md`. The legacy parity gate remains closed
for step1000, so generation stays BLOCKED with the exact reason previously
documented (KEEP_LEGACY_BLOCKED).

The precision-config plumbing added in this milestone is a private diagnostic
surface (default-off) that preserves existing behavior exactly; it documents
the delivery path and would exercise any future QAIRT release that honors
these options.

## 6. Private data

Raw prompts, logits, per-prefix tensors, device serials, and ADB endpoints
stay under `build/private-diagnostics/` and never enter public artifacts.
