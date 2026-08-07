# Nicopedia real-text HTP training, August 2026

This milestone ports the established Nicopedia real-text CPU pilot to the
Qualcomm QNN HTP backend. The transformer learning step's numerical work
(forward, cross-entropy backward, and Adam update) runs in an explicit QNN
HTP graph; CPU supplies tokenized pilot batches and host control. QNN
automatic differentiation is not used, and the claim is limited to executing
the training step's numerical operations on HTP.

## Configuration

Same input, initial parameters, and training conditions as the CPU pilot:
UTF-8 byte tokenizer (V=256), context 32, D=16, FFN=32, H=2, Adam lr=0.003.
The private tokenized pilot cache (1,995 articles, 8.4 MB, 8.39M target
tokens) is shared; the device receives only the minimal pilot input and never
publishes article text, token sequences, or checkpoints.

## Results

- L6: 320 training steps on HTP, finite for seeds 1/2/4. Validation NLL
  2.910/2.947/2.907 vs CPU anchors 2.911/2.874/2.872 (differences 0.001-0.073).
- L19: 320 training steps on HTP, finite. Validation NLL 2.908 vs CPU 2.819
  (difference 0.088); top-1 agreement is unchanged at 0.3005.
- One-step CPU/HTP parity: logits max abs error 4.1e-4-1.9e-3, gradient max
  abs error 6.8e-5-4.4e-3, probability 1.1e-5-1.5e-5 (all below the fixed
  tolerances 2e-2/5e-3/3e-2).
- Short-trajectory (2/4/8 step) parameter drift grows monotonically as
  expected from FP16 intermediates; all anchors finite.
- HTP timing on the nubia Z80 Ultra (SM8850, HTP V81): L6 ~185 ms/step,
  L19 ~483 ms/step at batch 8 (2-3 graph executes per step).

Thermal status was recorded only; no arbitrary-temperature cooldown was
introduced. Android thermal status stayed at 0 (normal) for all runs.

Attribution: This research used the "Nicopedia data" provided by Dwango Co.,
Ltd. through the IDR Dataset Provision Service of the National Institute of
Informatics (NII). Use is limited to non-commercial research. The dataset and
derived tokenizer/checkpoint artifacts are not redistributed.
