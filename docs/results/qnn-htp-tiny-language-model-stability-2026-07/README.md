# QNN HTP tiny language-model numerical-stability results

This directory contains public, aggregate-only evidence for the synchronized checkpoint,
2x2 gradient/optimizer split, five-seed convergence, and four-pattern inference study.
The result is GOAL_PARTIAL_SUCCESS: all five HTP training seeds remained finite and met
the evaluation-loss convergence gate, while autoregressive continuation did not meet 3/4.

Raw weights, optimizer states, callback output, logs, device endpoints, binaries, APKs,
host paths, and app-private paths are intentionally excluded.
