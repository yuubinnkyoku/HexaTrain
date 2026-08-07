# Nicopedia HTP step-320 vs step-1000 comparison

This bundle contains aggregate results comparing the L19 seed-1 checkpoint at
step 320 and step 1000 trained on the Nicopedia real-text corpus with QNN HTP.

## Teacher-forced metrics (allow-listed aggregates)

| checkpoint | source | validation NLL | selected step |
|------------|--------|---------------|---------------|
| L19 seed-1 step-1000 | CPU anchor | 2.510798771 | 1000 |
| L19 seed-1 step-1000 | HTP | 2.553278928 | 1000 |

The HTP step-1000 validation NLL is within ~0.04 of the CPU anchor.

## Generation status

- Step-320 HTP generation (greedy, sample): SUCCESS, parity gate PASS.
- Step-1000 HTP generation: BLOCKED by CPU/HTP parity gate.
  The fixed 2e-2 absolute logit-error gate is exceeded because logits grow in
  magnitude between step 320 and step 1000 (CPU logit RMS ~3.21 -> ~3.57 for
  prefix 0). Cosine similarity remains above 0.999998, argmax agreement is
  preserved, and top-k set overlap stays 5/5, so the ranking topology is intact.
- Step-1000 CPU generation is available privately for quality comparison;
  only aggregate byte/UTF-8 statistics are published here.

## Files

- generation-summary.csv: aggregate generation statistics per run.
- generation-parity.csv: CPU/HTP parity error magnitudes per prefix.
- generation-ar.csv: CPU/HTP autoregressive parity error magnitudes per step.
- training-status.csv: training run metadata.
- cpu-anchor.csv: CPU-selected anchor NLL from the public pilot bundle.
- limitations.csv: known limitations and blocked outputs.
- manifest.json: content hashes for the published files.

## Reproduction (manual)

```powershell
# Training
scripts/run_nicopedia_htp_training.ps1 -Model L19 -Seed 1 -Steps 1000 `
  -QairtSdkRoot "C:\Qualcomm\AIStack\QAIRT\2.48.40.260702" `
  -ExpectedBuildId "2.48.40.260702151143"

# Generation (step 320)
scripts/run_nicopedia_htp_generate.ps1 -Model L19 -Seed 1 `
  -Prompt "人工知能とは" -MaxNewBytes 64 -Mode Greedy -CheckpointStep 320 `
  -QairtSdkRoot "C:\Qualcomm\AIStack\QAIRT\2.48.40.260702" `
  -ExpectedBuildId "2.48.40.260702151143"

# Host eval
build/host_tests/htp_checkpoint_eval.exe `
  build/reports/nicopedia-htp-training/htp-seed1-l19-step1000.ckpt `
  build/private-data/nicopedia-real-text/caches/validation.bin `
  build/private-data/nicopedia-real-text/caches/development.bin 256 512
```
