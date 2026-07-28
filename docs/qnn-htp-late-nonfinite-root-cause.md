# QNN HTP Transformer: late-nonfinite root cause

## Outcome

`CAUSE_CLASSIFICATION=HTP_BACKWARD_NUMERIC_ERROR`.

The failure was deterministic on fixed last-finite training states. The first non-finite value is `tap_SQUARE2`, produced by `tt_ln2_square` in the training-graph LayerNorm2 path. Its immediate predecessor, `CENTERED_S2` from `tt_ln2_center_scale`, remains finite at roughly -257 to +257. The first-bad indices for seeds 1--5 are 22, 5, 10, 6, and 9; their predecessor values are -256.75, -256.5, -257.25, +256.5, and -256.25. Squaring after the application scale of 64 crosses the HTP FP16-like multiplication limit (65504), producing `+Inf`. This is a forward activation failure that subsequently contaminates the backward gradient and Adam state; it is not an Adam-origin failure.

The application fix reduces the centered scale from 64 to 8. It does not replace non-finite values, clamp tensors, alter epsilon, or lower the learning rate as a workaround.

## Formal results

The pre-fix `lr=0.003`, 320-step, unclipped Adam run failed on all five HTP seeds (first non-finite steps 99, 84, 95, 95, and 121). The independent CPU control was finite and deterministic on 5/5 seeds (`nan_inf_count=0`): its final evaluation losses were 0.16749, 0.18904, 0.16996, 0.24801, and 1.67823, with final evaluation accuracies 0.96875, 0.96875, 0.9375, 0.90625, and 0.84375. After the scale change, the same HTP condition is finite on 5/5 seeds, with final train loss 0.00148--0.00343 and final train accuracy 1.0 for all five seeds.

For the reproducible `lr=0.0003`, 1000-step, clip-5 condition, pre-fix last-finite/first-non-finite steps were 818/819, 655/656, 846/847, 698/699, and 997/998. The five post-fix seeds are finite through step 1000; their final losses are 0.135--0.457 and their final accuracies are 0.9375--1.0.

## Fixed replay and path isolation

Each pre-fix last-finite checkpoint was replayed 100 times. All 500 replays reached the same first-bad tensor and value class deterministically. The checkpoint manifest exposes only element counts and SHA-256 identifiers.

On the same pre-fix checkpoint, the four paths separate the fault:

| Path | Result before fix |
| --- | --- |
| A: CPU gradient + CPU Adam | finite |
| B: HTP gradient + CPU Adam | non-finite |
| C: CPU gradient + HTP Adam | finite |
| D: HTP gradient + HTP Adam | non-finite |

CPU and HTP Adam use the same beta values, epsilon, bias correction, step indexing, and clip order (global norm, gradient scale, then Adam). Path C being finite excludes the optimizer as the initiating cause. On all five legacy checkpoints after the scale fix, the full HTP-gradient/HTP-Adam path was finite and deterministic for 100 replays; the A/B/C/D separation was then run once per checkpoint from the identical state, and all four paths were finite.

## Public evidence

The generated bundle at [docs/results/qnn-htp-late-nonfinite-2026-07](results/qnn-htp-late-nonfinite-2026-07/README.md) records conditions, all seed results, hash-only checkpoints, replay counts, and 2x2 results. Its exporter validates fields and scans its outputs for restricted data.

## Remaining caveat

Post-fix finiteness resolves the late-nonfinite root cause. Numerical parity still merits follow-up: when starting from the legacy accumulated HTP state, path C/D maximum parameter differences can be substantial even though all tensors are finite, while the corresponding first- and second-moment differences are small. This is not hidden or treated as proof of bitwise CPU/HTP equivalence.
