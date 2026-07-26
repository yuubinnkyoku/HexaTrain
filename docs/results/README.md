# Experimental results

PhoneLMで再現・検証した研究結果の公開用indexです。生ログ、端末識別情報、ローカルパス、APK、QAIRT配布バイナリは含みません。

- [QNN HTP full training-step result (2026-07)](qnn-htp-full-training-step-2026-07/README.md) — 2層ReLU MLPのforward、MSE loss、backward、SGD updateを単一QNN HTP graphで実行したcorrectness・5 seed収束・性能結果。
- [QNN HTP tiny language model stability result (2026-07)](qnn-htp-tiny-language-model-stability-2026-07/README.md) — 同期checkpointと2×2経路分離によるAdam反復更新の診断、global-norm clipping後の5 seed収束、および4規則列autoregressive inferenceの未達結果。
- [QNN HTP tiny language model autoregressive-gap result (2026-07)](qnn-htp-tiny-language-model-generation-2026-07/README.md) — same-prefix parity、pattern-balanced phase sampling、oracle/free-running分離、および5 seed exact rolloutの部分成功結果。
- [QNN HTP fixed-state graph-prefix bisection (2026-07)](qnn-htp-graph-bisection-2026-07/README.md) — 同一input/stateのfull graphが2 fresh processで変動し、後続の`lm_dembedding`境界graphでは決定的になるgraph variant/Runtime順序依存条件。
