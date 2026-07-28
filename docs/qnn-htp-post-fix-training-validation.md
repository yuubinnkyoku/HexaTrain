# QNN HTP post-fix training validation

## Scope

This report records the first device validation performed after commit
`0152775`, which corrected the Softmax-backward `ReduceSum` declaration:
`SOFTMAX_DOT` is now `tokens x 1` for `axis=LAST_AXIS` and
`keep_dims=true`.  The tested model is the fixed FP32 tiny Transformer
(`B=1, T=8, V=32, D=16, heads=1, layers=1, FFN=32`, causal pre-norm ReLU).

## Shape validation

`qnn_graph_shape_validator` is run before QNN graph construction for the
`tt_smd` ReduceSum.  It infers shapes and checks declarations, dtype,
element-count constraints, axes, and broadcast compatibility for ReduceSum,
ReduceMean, MatMul, Transpose, binary elementwise operations, Softmax,
Reshape, Concat, Split, and LayerNorm.

The regression test supplies the former invalid node:

```
input=[8,8], axis=1, keep_dims=true, declared output=[8,8]
```

It is rejected with `node=tt_smd` and `expected=[8,1]`; `[8,1]` passes.

## Passed post-fix checks

The QAIRT build-ID-audited APK built successfully.  The headless device probe,
Softmax backward, attention backward, LayerNorm backward, Adam one-step,
fixed-state reproducibility, and paired DPROBABILITIES-to-DSCORES tap suites
passed.  The fixed-state run reported no variability; the paired tap completed
100 QNN executes with zero execution failures, nonfinite elements, APP_READ
poison residual, or APP_WRITE changes.

## Long Adam result

The required conclusion cannot yet be that the shape defect fully explains
training instability.  On this post-fix run, the existing 1,000-step five-seed
Adam candidates failed late despite successful QNN execution:

| configuration | finite seeds | completed-step range |
| --- | ---: | ---: |
| lr=0.0003, clip=5 | 0/5 | 655--981 |
| lr=0.0003, clip=10 | 0/5 | 655--981 |

All failures ended in nonfinite loss.  The clip=5 run recorded zero clipped
steps and a maximum pre-clip norm of about 4.12, so the two clip thresholds
were not exercised.  QNN execute returned success throughout the failed
run (8,030 successes, zero nonzero returns); this isolates the observed
nonfinite result from a graph-execute error, but does not identify its root
cause.

## Classification

`SOFTMAX_SHAPE_DEFECT_FIXED_NONFINITE_BUT_CONVERGENCE_GAP_REMAINS`.
The structural defect and its fixed-state manifestation are addressed, but
the required five-seed long-training and phase01-generation success criteria
are not met.  No raw tensors, logs, APKs, runtime binaries, device endpoints,
or private paths are included in this document.
