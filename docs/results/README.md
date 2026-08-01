# Experimental results

PhoneLMで再現・検証した研究結果の公開用indexです。生ログ、端末識別情報、ローカルパス、APK、QAIRT配布バイナリは含みません。

- [QNN HTP generic Transformer depth/head envelope (2026-07)](qnn-htp-generic-depth-head-2026-07/README.md) — generic layer/head graph、5 formal構成25/25 finite、L6/H8 combined、L18→L19およびFFN371→372数値境界、bitwise再現性、exclusive性能、実UI結果。
- [QNN HTP generic depth-quality diagnosis (2026-08)](qnn-htp-generic-depth-quality-2026-08/README.md) — L18/L19 phase-1 trajectory、明示的stability候補3種、finite/QNN return監査、および採用stabilizerなしの結果。
- [QNN HTP full training-step result (2026-07)](qnn-htp-full-training-step-2026-07/README.md) — 2層ReLU MLPのforward、MSE loss、backward、SGD updateを単一QNN HTP graphで実行したcorrectness・5 seed収束・性能結果。
- [QNN HTP tiny language model stability result (2026-07)](qnn-htp-tiny-language-model-stability-2026-07/README.md) — 同期checkpointと2×2経路分離によるAdam反復更新の診断、global-norm clipping後の5 seed収束、および4規則列autoregressive inferenceの未達結果。
- [QNN HTP tiny language model autoregressive-gap result (2026-07)](qnn-htp-tiny-language-model-generation-2026-07/README.md) — same-prefix parity、pattern-balanced phase sampling、oracle/free-running分離、および5 seed exact rolloutの部分成功結果。
- [QNN HTP Tiny LM staged scaling result (2026-07)](qnn-htp-tiny-lm-scaling-2026-07/README.md) — seed 2〜5のbitwise再現確認、sequence 32・dimension 32までの単一要素ごとの拡大、5 seed正式評価、layers 2の明示的な未対応停止条件。
- [QNN HTP fixed-state graph-prefix bisection (2026-07)](qnn-htp-graph-bisection-2026-07/README.md) — 同一input/stateのfull graphが2 fresh processで変動し、後続の`lm_dembedding`境界graphでは決定的になるgraph variant/Runtime順序依存条件。
- [QNN HTP fixed-state Runtime/context order study (2026-07)](qnn-htp-graph-order-2026-07/README.md) — 60 fresh processの順序直交化で、graph variant固有・固定position・先行graph必須ではないfresh instance関連の数値変動を集計。
- [QNN HTP fixed-state root-cause study (2026-07)](qnn-htp-root-cause-2026-07/README.md) — Softmax backwardの`SOFTMAX_DOT` ReduceSum出力shape不整合、first-changing区間、修正前後fresh-process対照、および修正後未再発を集計。
