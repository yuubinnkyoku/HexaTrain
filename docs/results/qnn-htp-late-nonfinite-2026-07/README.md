# QNN HTP late-nonfinite public result bundle

This directory is an allowlisted, generated summary of the July 2026 investigation. It contains aggregate metrics and SHA-256 identifiers only; checkpoint contents and platform traces are intentionally excluded.

Files:

- `summary.json`: classification, fix, formula audit, and residual numerical-parity caveat.
- `seeds.csv`: pre-fix and post-fix five-seed results.
- `cpu-control.csv`: independent CPU control with the actual train and evaluation metrics for all five seeds.
- `checkpoints.csv`: five last-finite checkpoint manifests (element counts and SHA-256 identifiers).
- `replay.csv`: each pre-fix checkpoint replayed 100 times, plus the post-fix same-checkpoint replay.
- `two-by-two.csv`: CPU/HTP gradient and Adam path isolation before and after the fix.

The public exporter validates source field counts, replay counts, and identifier syntax before writing these files. It refuses any output directory other than this directory and scans generated files for restricted data patterns.
