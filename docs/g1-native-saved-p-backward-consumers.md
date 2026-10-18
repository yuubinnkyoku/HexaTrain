# G1 Native Saved-P Backward Consumers — audit memo

Status: `UNRESOLVED` (host audit only; device phases BLOCKED).

Branch: `investigation/g1-native-saved-p-backward-consumers`
Base: `024aaef44784355e8e278ef642e2752ae79846bb` (clean tree at start)

## 1. Provenance gaps (do not certify)

- Mission research base `e05f353`: unknown in the local repo.
- Mission previous branch
  `investigation/g1-s2-double-qk-native-p-closure`: absent locally.
- Mission primary doc `docs/g1-s2-double-qk-native-p-closure.md`: missing.
- Mission hashes (`099baa5`, `962a98c6`, `e2df381e`, `911d66e8`,
  `d8e50728`, `456ad468`): unverifiable locally.
- Mission chain `S2 LN1 / Mixed QKV / Double QK / Custom Softmax`:
  absent from the current production source.
- Mission fixtures (44 forward, 25 gradients, 16 holdout, 8 holdout
  gradients): no matching files in the local repo.

Local repository is authoritative. No history was reconstructed from
GitHub main. No reset/rebase/amend/push/merge was performed.

## 2. Production Saved-P ownership (audited in source)

Graph builder:
`app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc`

- Per-head producer: attention softmax node into
  `headRecord.probabilities [T,T]`.
- Forward consumer: PV MatMul `probabilities x value -> context`.
- Backward consumer 1: dValue MatMul with `TRANSPOSE_IN0` only.
- Backward consumer 2: softmax_product
  `dProbabilities x probabilities`.
- Softmax backward order: product, ReduceSum-last keep_dims into
  `[T,1]`, subtract, multiply into dScores.
- dScores is dMaskedLogits; score-scale backward is a separate node.
- Diagnostic builds expose head P as APP_READ taps; production keeps P
  NATIVE. Both use the same graph tensor ID per head.
- HOST readout plus APP_WRITE reinjection is a control only.

## 3. Standalone references (audited, not executed here)

- `prepareSoftmaxBackward` / `executeSoftmaxBackward`:
  existing 4x4 micro graph in `qnn_runtime_transformer.inc`.
- `prepareAttentionBackward` / `executeAttentionBackward`:
  existing T4/D8 micro graph in the same file.
- CPU analytic references live in `qnn_transformer.cpp`.
- New host-only T32/Dh32 double plus ordered-FP32 replay added under
  `host_tests/fp32_custom/`; not compiled here because no host C++
  compiler (`g++`, `clang++`, `cl`) is on PATH.
- Prospective plan frozen in
  `host_tests/fp32_custom/saved_p_backward_native_plan.json`.

## 4. Blocked device work

Tier2 preconditions were not established in this session:

- same physical device identity
- awake / top-resumed
- no active training run
- pinned QAIRT root plus Build ID
- QNN-enabled build plus APK audit

Therefore all DSP consumer, native fan-out, observer, performance,
three-process, regression, and F1-F33 answer phases remain BLOCKED.

## 5. Protected state

No checkpoint, optimizer, dataset, training, 40k/60k/200k, 6031, or
35/40-node-tap state was changed. No Tier3 action was executed.
