# QNN HTP generic depth-quality diagnosis

This public bundle contains aggregate-only evidence from 320-step training on
the T8/D16/FFN32 L18/L19 HTP configurations. Each listed run used one explicit
seed, the legacy protocol or an explicitly named experimental stability mode,
and read-only phase-1 evaluation at the listed checkpoints.

The result is NO_STABILIZER_SELECTED. ZERO_OUTPUT_PROJ_BRANCH_INIT improved
the selected bad L19 seed 2 evaluation loss from 5.4304 to 2.2809 and seed 4
from 3.4454 to 1.9415, but worsened the selected good seed 1 from 0.2629 to
0.6863. DEPTH_SCALED_BRANCH_INIT improved seed 2 to 0.9089 but worsened
seed 1 to 9.1662. GRADIENT_CLIP_1 worsened all three selected L19 seeds.
No candidate improved all selected seeds, so LEGACY remains the default and
no stabilized mode is claimed as a fix.

All aggregate runs had zero nonzero QNN return counts and finite reported
tensors. FAILED in the comparison table denotes the existing generation
quality threshold result, not a QNN execution or finiteness failure.

Raw weights, optimizer states, private checkpoints, logs, device identifiers,
APK hashes, host paths, and app-private paths are intentionally excluded.
The numerical operation statement is limited to: training-step numerical
operations were executed on HTP; CPU-side scalars and reporting remain
explicitly identified by the app protocol.