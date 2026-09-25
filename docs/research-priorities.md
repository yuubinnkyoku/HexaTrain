# HexaTrain 技術候補・優先順位

> **更新基準: 2026-09-22 JST**
>
> 現在の実装・実測の基準は [`yuubinnkyoku/HexaTrain`](https://github.com/yuubinnkyoku/HexaTrain) `main`、確認時HEAD **`315d048`**（2026-09-22確認）。Muonの正式品質baseline、parameter metadata SSOT、headwise gated-attention実験経路まで含む現行状態を基準にする。
>
> **Limite注記:** 2026-09-22時点ではLimite 1B - Violettoの正式Technical Report本文は未公開。Limite関連の優先順位は、公式release / Hugging Face config・model card / Value Model文書 / 公式vLLM実装 / Paradigma speedrun回顧で確認できる範囲に限定し、未公開のtraining recipeは推測しない。
>

## 方針

1か月前と最も大きく変わったのは、研究の中心が「**学習演算をQNN HTP graphへ載せる**」だけではなく、**QNN HTP・HVX・必要最小限のCPUを、演算の数値特性とデータ移動コストに応じて配置する**段階へ進んだことである。

Forward / Loss / Backward / Adamは引き続きQNN opを明示的に構築する。一方、MuonのNewton–SchulzではQNN HTP MatMulの内部精度がボトルネックになり、同じ数学をHVX FP32で実装するとCPU oracleに極めて近い精度で成立した。したがって「すべてをQNN graphへ押し込む」こと自体を目的にしない。

最終的には次の7層を共同最適化する。

1. **Model Architecture** 
   - D / FFN / L / H
   - Query head / KV head / head dimension
   - Attention / recurrent operator / layer layout
   - tied embedding / memory layer / sparse structure
2. **Text Representation** 
   - byte-BPEを現行基準とし、raw byte / GBST / BLT等を比較
   - vocabulary sizeではなく、元UTF-8 byteあたりの品質・時間で比較
3. **Gradient / Optimizer** 
   - Exact BP / segmented BP / ZO / hybrid BP+ZO
   - Adam / Muon+Aux Adam / quantized / stateless / low-rank
4. **Training Graph / Memory** 
   - algebraically equivalent graph / kernel
   - graph partition / fusion / KEEP / spill / RECOMPUTE
5. **Precision / Representation** 
   - storage dtype と compute dtype を分離
   - tensor/opごとのFP / INT8 / INT4 / INT2
   - scale granularity / outlier transform / rounding
6. **Backend Placement** 
   - QNN HTP / HVX / CPUのどこで演算するか
   - FastRPC境界、pack/unpack、shared buffer、worker数、power voteも探索対象
   - 「QNNで実行できるか」だけでなく「要求精度を満たす最速経路か」で決める
7. **Hardware Objective** 
   - validation / development / frozen-final bpb
   - ms/update / original UTF-8 bytes/s
   - Forward / Backward / Optimizer / RPC / kernel時間
   - peak RAM / tensor bytes moved
   - graph create/finalize時間
   - thermal / energy
   - QNN/HVX failure / non-finite / fallback
   - CPU oracleとの差

## 2026-09-22 現在地

### 学習baseline

- **V1024 / T32 / D64 / FFN128 / L19 / H2**、**758,528 parameters**が現行の公式品質baseline。
- tokenizerは決定的**Byte-BPE V=1024**。tokenizer / cache / checkpoint / evaluationのidentityをSHA-256等で結び、互換しないartifactはfail-closedで拒否する。
- Adamは`LR=0.0022`をstep 4000まで維持し、step 8000で`0.0001`へlinear decayする **S4000** を独立した安定controlとして保持する。
- untouched final sampleは既存のread-once結果を研究候補の選別へ逆流させない。新候補はVal/Devで比較する。
- parameter identity / shape / optimizer role / fan-in・fan-out semanticsは`transformer_parameter_metadata`へ集約され、Muon / Aux Adamの分割をname substringへ依存しないSSOTとして扱える。

### Muon — 「候補」から正式baselineへ

- Matrix 114個 / 622,592 parametersを**Original Muon**、残り135,936 parametersを**Auxiliary Adam**へ分けるmixed optimizerを実装済み。checkpointは**NPRTCKPTV4**。
- Forward / Backwardは**QNN HTP**、Original Muonは**Hexagon HVX FP32 / 8 workers**、Aux AdamはCPUというheterogeneous training構成で8,000 stepを完走済み。
- 8,000-step × 2 seedのBalanced bpbは、Adam S4000平均`2.409512`に対してHVX Muon平均`2.343970`、差は**-0.065542 bpb**。両seedでVal / Devとも改善。
- HVX Muon側はRPC failure / fallback / non-finite=`0 / 0 / 0`。同一backendでのuninterrupted vs resumeはbitwise一致。
- seed 2のfresh 8,000-step wallは約`10,552.95 s ≈ 2.93 h`、2 seedの平均は約**1.313 s/update**。過去Adam S4000 tailとの単純なend-to-end実測比較は約3.66xだが、runner世代等が異なるためoptimizer primitive単体の倍率とはみなさない。
- **QNN HTP-native Muonは引き続きBLOCKED**。Newton–Schulz反復でQNN HTP MatMulの誤差が増幅し、NS5でrelative L2が約`5.2e-2`まで拡大する。Original Muonの品質baselineはHVX FP32経路とする。
- 単一stepではHVX FP32 MuonはCPU oracleへmaxAbs約`1.49e-8`、relative L2約`4.4e-8`で一致する一方、複数stepではFP32/CPU-doubleの軌道差が増える。長期near-bitwise parityは要件にしない。

### MiMo-V2.6を受けて新しく見えるoptimizer課題

MiMo-V2.6は、AdamW事前学習から大batch RLへ直行せず、**mid-trainingでhidden weight matrixをMuownへ移行してから大規模RLへ接続**する設計を採用した。Technical Reportでは、大batch化に伴うAdamWの最適化効率低下を問題として挙げ、Muon系のmatrix-aware updateを選びつつ、Muownのrow-norm controlでspectral-norm driftとweight-decay感度を抑える。

HexaTrainはすでにscratchからOriginal Muonを正式baselineとして持つため、ここから得るべきものは「Muon導入」ではない。次の問いは、

1. 現行8,000-step Muonで**semantic row norm / spectral norm driftが実際に起きているか**
2. 起きているなら**Muownで抑えるとbpb・学習率頑健性・後半の改善速度が上がるか**
3. effective batchを増やしたとき、Adam / Muon / Muownのどれが**同一token budgetあたりの品質**を維持するか

である。

### Limite 1B - Violettoから見えるarchitecture課題

2026-09-22時点でLimite 1B - Violettoの正式Technical Report本文は未公開で、pretraining optimizer / exact data mixture / SFT・RL actor recipe等は確定できない。一方、公式release、Hugging Faceのconfig / model card、Value Modelの`MODEL_DETAILS.md`、公式vLLM実装から、**network structureについてはかなり高い解像度で確認できる**。

Violettoで確認できる主要要素は、

- 48-layer / hidden 1280 / 10 query heads / 2 KV headsのGQA
- local sliding-window attentionと周期的full/global attention
- RoPE + QK RMS normalization
- per-head attention output gate
- **XSA (Exclusive Self Attention)**
- token-dependent **Value Embedding**
- 限定layerでの**MUDD dynamic dense residual mixing**
- learned residual / branch scales
- SwiGLU
- tied input/output embedding
- 131k context

である。

ただしHexaTrainは現在**T32 / D64 / L19 / H2 / 758k**であり、Violettoの形を縮小コピーするのは目的ではない。重要なのは、Paradigmaが1B級モデルでも採っている「**小さい追加parameterで情報経路を制御する**」「**長文脈コストをlocal/globalで分ける**」「**projection/layoutを融合して実行効率を稼ぐ**」という設計原理である。

HexaTrainへ直近で落とし込む問いは次の5つ。

1. 既存G1 gateの改善がstep 2000以降も残るか。残らない場合、Limite型の`2 * sigmoid` / reduced gate channels / identity initializationで改善するか
2. XSAをlearnable strengthかつzero-initで加えたとき、T32でもattentionの自己成分依存を減らしてbpbが改善するか
3. Value Embeddingを直接増設する前に、既存Vを再利用する**Value Residual**で同じ方向の効果を安く得られるか
4. MUDDのような重いdense residual mixingへ行く前に、learned residual scale / branch scaleで深さ19の情報流を改善できるか
5. Q/K/Vを別MatMul + selector/scatterで扱う現行graphを、**packed QKV + reshape/slice/concat**へ寄せて品質を変えずにwall timeを削れるか

また、Paradigmaのpretraining speedrun系で使われたsingle-head multi-token supervision、ReLU²、終盤RRE/Anderson extrapolationはVioletto本体の確定仕様と同一視しないが、Limiteの設計系譜としてHexaTrainの低コスト候補へ加える。

### 推論・Android基盤

- model configの一般化、Material 3 Expressive UI、学習表示、推論GUIは実装済み。
- FORWARD_ONLY graphとprepared generation engineにより、同一checkpointのwarm再生成ではcreate/finalizeを再利用する経路が成立。
- 長時間学習はWorkManager経路と4時間級soak検証、8,000-step級runまで進んでいる。

### 現在見えている本当のボトルネック

- Muon backend単体ではHVX kernel最適化だけでなく、**pack/unpack・validation・state copy・Aux Adam・FastRPC境界**を含めたend-to-end設計が重要。kernelだけの高速化では不十分。
- full評価はone-window-per-graph executionの固定費が大きく、**評価engineのbatch化・persistent graph化**が研究iteration速度を制限している。
- D64/FFN128は容量増で小さく一貫した改善を示すが、FFN幅だけをさらに増やすより、parameter-matched shape searchの情報量が高い。
- Muownの第一段階は新規学習ではなく、既存Muon checkpointからの**row geometry診断**で済む。ここは実装費用に対する情報量が非常に高い。
- 現行generalized HTP graphは各layerでQ/K/Vを**3本の別MatMul**として作り、H2分割でもselector/scatter MatMulを使う。Limite/vLLM系のpacked QKV layoutを参考に、optimizer/checkpoint上のWq/Wk/Wv identityは保ったまま、実行用packed cacheを持つ余地がある。
- T32では131k context / sliding-global attentionを移植しても固定費を回収しにくい。一方、G1 / XSA / residual scale / ReLU² / dense multi-token supervisionは**短文脈でも追加計算が小さく、現在の研究scaleに合う**。

# 優先順位

## P0 — 現在最優先

- **現行formal baselineを回帰基準として凍結**
  - V1024/T32/D64/FFN128/L19/H2
  - QNN HTP Forward/Backward + HVX W8 Original Muon + CPU Aux Adam
  - Adam S4000を独立controlとして保持
  - tokenizer / dataset / order / checkpoint / QAIRT identityを固定
  - final splitは既存read-once結果をcandidate tuningへ逆流させない
- **Muon row-geometry診断**
  - 既存step 0 / 500 / 1000 / 2000 / 4000 / 6000 / 8000 checkpointから、114 matrixのsemantic row norm・spectral norm・coherence・effective rank・angular updateを抽出
  - `max row norm ↑`と`spectral norm ↑`が同期するかを確認
  - W1/W2では保存上のrowではなく、SSOTの`fan_out_axis`に従った**出力neuron方向**をrowとして扱う
  - driftが弱い場合、Muown実装の優先度を下げる
- **評価engine高速化**
  - one-window-per-graphをbatch / persistent executionへ寄せる
  - windows/s、original UTF-8 bytes/s、graph prepare amortizationを測る
  - full-final相当を現実的な時間で読める経路を目標にする
- **QNN HTP / HVX Backend Placementの計測基盤**
  - op単位に「QNNで可能」ではなくprecision / latency / data movementで配置を決める
  - QNN HTP Newton–Schulzは新しいprecision evidenceが出るまで再試行しない
- **Gated Attentionの実機A/B継続**
  - 既存`headwise_g1_sigmoid`はCandidate2000まで完了し、Balanced差はstep 500 / 1000 / 1500 / 2000でそれぞれ`-0.032712 / -0.036686 / -0.021864 / -0.007113`。早期gainは強いが差が縮小している
  - まず同一recipeで4000まで延長し、「収束加速だけか / 最終品質差が残るか」を判定
  - gate parameterはAux Adamのまま扱い、Muon packing contractを無理に拡張しない
  - 次段候補としてLimite型の`2 * sigmoid`、`Wg=0` identity initialization、D64全体ではなく8〜16 channelsだけからgateを作るvariantを短期A/B
- **実験protocol / metrics固定**
  - bpb / ms/update / original bytes/s / RAM / thermal / parity
  - RPC external / DSP worker imbalance / optimizer geometryも標準telemetryへ

## P1 — P0の直後

- **Muown reference実装と短期A/B**
  - まずCPU oracleを論文のAlgorithm 1に忠実に実装
  - `g / r / m_g / v_g`のrow stateとdirection Muonを分離
  - 現行Original Muon identityを変更せず、新optimizer identity / checkpoint schemaでfail-closed
  - 最初はNS5固定。MiMo-V2.6のNS10等を同時に持ち込まず、row-norm controlの効果だけを分離
  - 1 / 8 / 32 step correctness → 500 / 2000 step qualityへ段階昇格
- **Effective Batch Search**
  - physical B8を固定し、gradient accumulation等でeffective batch 8 / 16 / 32 / 64を比較
  - optimizer step数ではなく**同一original-byte / token budget**でAdam / Muon / Muownを比較
  - quality/byteだけでなくwall-time、optimizer invocation数、RAMを記録
- **Dense Multi-Token Supervision / MTP-lite**
  - Paradigma speedrun系の「1つのLM headで複数future targetを同時教師化」を、full MTP head追加より先に試す
  - 現行HTP CE backwardは`(P - Y) / N`なので、`Y`をsoft targetへ変えるだけならlogits/head数とbackward shapeを維持できる
  - T32で`t+1/t+2/t+3`を使うにはcacheを`T+3`相当へ拡張し、uniform `1/3,1/3,1/3`とdecay `0.5,0.3,0.2`を比較
  - final metricは通常next-token bpbのまま。追加教師がnext-token性能を損なう場合は採用しない
- **ReLU² A/B**
  - 現行`ReLU(W1x)`を`ReLU(W1x)^2`へ変更する低コスト候補
  - parameter数・Muon matrix shapeは不変。SwiGLUより先に、elementwise squareと単純backwardだけで効果を測る
- **Learnable XSA**
  - per-layer/per-head強度`alpha`を0初期化し、初期forwardをbaselineと一致させる
  - T32では原論文の長文脈gainを前提にせず、6-layer subset → 全19層の順で短期A/B
  - XSA演算をFP32/HVXへ逃がす必要があるか、QNN HTPのnorm/dot/elementwiseで成立するかも同時に測る
- **Lightweight Residual / Value Routing**
  - `x' = λ_resid x + λ_branch f(x)`のlearned scalar residual scaleをidentity初期化
  - MUDDより先に、既存第1層V等を再利用するValue Residualを比較
  - token-dependent Value Embeddingは追加tableが大きいため、Value Residualで効果を確認してから
- **parameter-matched Architecture Search**
  - D / FFN / L / H / head_dim / Q-KVを総parameterを近づけて比較
  - shallow/wide vs deep/narrowをwall time・bytes moved込みで評価
- RMSNorm / RoPEの独立A/B。Gated FFNはReLU²を先に通し、その後SwiGLU/ReGLUへ進む
- QK Norm / Zero-Centered RMSNorm / norm-weight monitoring
- Z-Loss diagnostic
- HTP-Aligned GQA / tied embedding
- **Training Memory Planner**
  - FP KEEP / low-bit KEEP / spill / RECOMPUTE
  - QNN graph boundaryとHVX shared-buffer boundaryを一緒に扱う
- **Data-Movement / Host Boundary Optimization**
  - pack/unpack、transpose、candidate copy、validation、FastRPC、shared arena
  - **packed QKV execution cache**: optimizer/checkpoint上はWq/Wk/Wvを分離したまま、Muon更新後にQNN実行用`[Wq|Wk|Wv]`を更新
  - H2 selector/scatter MatMulをreshape / slice / concatへ置換できるかmicrobenchmarkし、node数と実wall timeで判定
  - direct RPC layout / persistent buffers / double buffering候補
- **QNN/HVX Superoptimization**
  - 数式rewriteだけでなくbackend placementまで探索
- NASの効率化
  - direct sweep → MatFormer / Puzzle型候補評価
- Attention normalizer比較
  - Softmax / Sigmoid / ReLU系

## P2 — Muown / architecture基礎実験の後

- **AngularMuown**
  - Muownのdirection row normが暗黙にangular step-sizeを減衰させるという解析をHexaTrainで検証
  - rowごとの更新角`theta`とbpb改善速度を追い、必要ならangular multiplierを明示制御
- **Adam → Muon / Muown移行protocol**
  - scratch baselineでは不要
  - 将来Adam checkpointを再利用する場合だけ、同一parentからAdam継続 / Muon / Muownをbranchし、optimizer mismatchを1/8/32/100/500 stepで測る
  - MiMo-V2.6のようなmid-training bridgeを候補にする
- **Text Representation Searchの第2段階**
  - 現行byte-BPE V1024を基準にraw byte / larger BPE / GBST / 簡易BLT
- Exact BP vs Apple MeBP / Qualcomm QZO-FF / Google Addax
- Quantized Adam / optimizer-state低bit化
- INT8 / INT4 / balanced INT2 / ternary training baseline
- Gated DeltaNet + Attention hybrid
- **MUDD-lite / dynamic dense residual mixing**
  - Violetto同様に全層へ入れず、まず2箇所程度・3 taps・小内部幅で試す
  - history activation保持、extra node/tensor、backward固定費がD64では相対的に大きいため、learned residual scale / Value Residualの後
- Gated Residual / multi-stream residual
- Hymba型 Hybrid-Head
- Attention Skip → recurrent operator置換
- Partial RoPE
- Full MTP / NEXTN heads
  - P1のDense Multi-Token Supervisionで学習gainを確認した後、追加headを持つMTPへ進む
  - training auxiliary + speculative decodingを一体評価
- Meta Token
- Mixture-of-Depths
- AltUp
- GaLore / SWAN / RACS / Alice / AdEMAMix比較
- COAT型activation/optimizer state低bit保存
- FPTQuant / QuaRot / SpinQuant / Hadamard
- stochastic rounding
- IR-QLoRA / LR-QAT / QZO-LoRA
- Weight Reuse / Minitron / pruning + distillation
- Quality-Diversity Architecture Search
- Memory Layer / UltraMem
- **MOPD / MOPD²型distillation**
  - external teacher利用を許す別研究trackとして、T32 suffix windowやprefix再利用へ小型化
- **Muon Turbo等の別algorithm**
  - Muownとの比較軸が揃った後に着手

## P3 — 長文脈・疎構造・RL / 独自architecture

- Cross-Layer KV Sharing / YOCO
- Local / Global hybrid
- NAMM / KeyDiff / KVP
- CommVQ / ShadowKV / LaCache
- Trellis / Lattice
- Block Sparse / FlexPrefill / Cross-Layer Index Sharing
- QSA / DSA型 indexed sparse attention
- sparse + linear/recurrent attention hybrid
- SimpleGDN / Gated DeltaNet詳細化
- N-gram associative memory
- HTP/HVX専用Delta / recurrent operator
- N\:M sparsity / Sparse-BitNet
- Direct Low-Bit MatMul
- PIE / STRING / long-context position処理
- HMX直接利用
- **GRS / GAR型groupwise grading**
  - long-horizon RLを始めた場合に、pass/fail以外のtrajectory品質・短さをrewardへ入れる
- **R3 / Router freeze**
  - 現行Dense Transformerには適用先なし。MoE化した場合だけtrain/inference routing整合性の優先課題にする
- 7k+ agent environment / multi-harness RLは、現行のscratch LM architecture研究とは別track

## 今はやらない / 優先度を落とすもの

- **同じQNN HTP MatMul経路でのNewton–Schulz再挑戦**
  - precision failureの原因が十分に局在しているため、新しいbackend/precision機構なしの再試行は情報量が低い
- **Muon scratchのVTCM化**
  - 実測でDDR scratchより大幅に遅かった
- **FFN幅だけをさらに増やす探索**
  - FFN128の追加容量は改善するが費用対効果が小さい
- **T32のまま本格sparse attention / Limiteの131k local-global構成を移植**
  - indexer・mask・layout固定費を回収しにくい。長文脈化後でよい
  - Violettoの36 local + 12 globalという層配置はT32の根拠にはしない
- **MUDD / Value EmbeddingをViolettoと同じ規模で一括導入**
  - D64では追加graph/tensor/tableの固定費が相対的に大きい
  - residual scale → Value Residual → XSA → 限定MUDDの順で寄与を分離する
- **Limiteのactor-critic / value-model recipeを現行scratch pretrainingへ持ち込む**
  - post-training目的と現行next-token pretrainingの目的が違う。RL track開始時まで保留
- **MiMo-V2.6のRL stackをそのまま移植**
  - 1,568 prompts × 16 rollouts、multi-agent sandbox、grader cluster等はHexaTrainの現スケールと目的が違う
  - 取り込むのはoptimizer / data-mixture / distillation / reward設計の**縮約可能な原理**に限定する

## 2026-09-22 直近LLMリリースからの差分

直近1か月の公開モデルをHexaTrainの観点で読み直すと、単発の新奇技術よりも、**同じ方向の技術が複数の大規模モデルで同時に採用され始めたこと**が重要。

| モデル | 確認できた技術 | HexaTrainでの扱い |
| --- | --- | --- |
| **MiMo-V2.6** | AdamW pretrain → Muown mid-training、large-batch fully-async RL、1,568 prompts × 16 rollouts、GRS/GAR、MOPD²、mixed-task ratio安定化、MoE router freeze | **Muon→MuownをP0/P1へ昇格**。まずrow-geometry診断。MOPD²はP2、GRS/GARとrouter系はP3 |
| **Limite 1B - Violetto** | 1B Dense、48L/D1280、10Q/2KV GQA、local/global attention、QK RMS norm、head-wise gate、XSA、Value Embedding、限定MUDD、learned residual scale、SwiGLU、tied embedding、131k context | G1継続をP0、XSA / residual scale / Value Residual / packed QKVをP1、MUDD-liteをP2、local/global長文脈をP3。正式Technical Report未公開のためtraining recipeは未確定として扱う |
| Qwen3.8-Flash-Next | Gated DeltaNet + QSA、4-way Gated Residual、N-gram Embedding、Muon | GDN hybridはP2。Gated ResidualをP2。QSAとN-gram memoryはP3で小型化 |
| GLM-5.3-Flash | sparse attention + linear attention、mHC | Hybrid Sequence Operatorの根拠を強化。mHCはQNN graph複雑性のためP3据え置き |
| Hy4-preview | Gated DSA、IndexCache、iHC、MTP | indexed sparse attention + index reuseをP3へ。MTPのtraining/inference両用を強化 |
| DeepSeek-V4.1-Flash | sparse attention、compressed indexer、N-gram memory、Hyper-Connection、MTP | QSA/DSA系・N-gram・residual拡張・MTPが別系統でも収束している証拠として扱う |
| dots3-note Preview | 13 DSA + 33 SWA、shared MTP | sparse/dense混成layoutとMTP speculative decodingの実用例 |
| MiniCPM5-2B | 2.5B級Dense、on-device志向、学習データ群公開 | HexaTrainの小型モデル比較・data pipeline・再現性のreference |
| K2 Horizon | 0.9B〜375Bのscale family、data/code/method/intermediate checkpoints公開方針 | scale ladderと再現可能な実験protocolのreference |
| Granite 4.2 | controllable thinking、tool use | architectureではなくinference/evaluation側のbudget制御候補 |
| Ornith-1.5 / Smaug | self-generated task loop、agent trajectory中心のpost-training | P3以降のpost-training/data generation候補 |

Limiteから特に重要なのは、**1Bモデルの個々の部品をコピーすることではなく、追加parameterや追加MatMulを小さく保ちながらattention / residual / value経路を制御し、実行layoutも同時に最適化すること**である。

HexaTrainでは、

- 既存G1を4000まで延長し、必要ならLimite型`2 * sigmoid` / reduced gate channelsを比較
- packed QKVとhead split/concatのlayout最適化を品質非変更のspeed trackとして独立評価
- ReLU² / Dense Multi-Token Supervisionを低コストquality trackとして追加
- XSAをzero-init learnable strengthで小さく導入
- learned residual scale / Value ResidualをMUDDより先に比較
- MUDDは全層ではなく限定2箇所程度から
- tied embeddingはV1024×D64で65,536 parameters（現行総数の約8.6%）を節約できる候補として、品質とのtrade-offを実測
- 131k / local-global attentionはT32の間は優先しない

へ落とす。

MiMo-V2.6から特に重要なのは、**Muon系を使ったこと自体ではなく、「大batchへ移る前にmid-trainingでoptimizer状態を作る」「row magnitudeとdirectionを分離する」「複数task/harnessを一つのrunへ混ぜるときsample ratioとgrader signalまで設計する」**という橋渡しの発想である。

HexaTrainではこれをそのまま巨大RLへ拡張せず、

- Original Muon checkpointからrow-norm driftを診断
- Muownをsemantic fan-out rowで実装
- effective batchを小さく段階拡大
- 複数corpusを混ぜる場合はsource ratioをidentity化
- 外部teacherを使うtrackではprefix再利用型distillationを検討

へ落とす。

参考:

- [MiMo-V2.6 Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
- [MiMo-V2.6 Official Release](https://mimo.mi.com/docs/en-US/news/latest/v2-6)
- [Muown: Row-Norm Control for Muon Optimization](https://arxiv.org/abs/2605.10797)
- [Muown Implicitly Performs Angular Step-size Decay](https://arxiv.org/abs/2606.23637)
- [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468)
- [MOPD](https://arxiv.org/abs/2606.30406)
- [R3: Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)
- [Qwen3.8-Flash-Next GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [GLM-5 GitHub](https://github.com/zai-org/GLM-5)
- [Hy4-preview](https://huggingface.co/tencent/Hy4-preview)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [dots3-note Preview](https://github.com/studio-dots-ai/dots3-note-prev)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [IFM K2 Horizon](https://ifm.ai/)

## 最終的なHexaTrain独自技術候補

- **Heterogeneous Training Backend Planner**
  - QNN HTP / HVX / CPUをop・tensor・phase単位で選択
  - precision gateとdata-movement costを同時に扱う
- **Matrix-Optimizer Geometry Lab**
  - Original Muon / Muown / 将来AngularMuownを、row geometry・spectral norm・angular update・qualityで比較
  - semantic fan-out orientationをparameter metadata SSOTから取得
- **HVX Optimizer Backend**
  - Original Muonを最初の正式実証とし、persistent RPC arena / multi-HVX / fail-closed state publicationを一般化
  - Muownが有望ならdirection updateをHVXへ拡張
- **HTP Training Architecture Search**
  - D / F / L / H / Q-KV / head_dim / attention placement
  - G1 / XSA / residual scale / Value Residual / ReLU²を独立factorとして扱う
- **Dense Supervision Planner**
  - next-token one-hotだけでなく、same-head multi-future soft targetの重み・horizon・cache spanを探索
  - full MTP headを増やす前に、既存HTP CE graphを再利用できる範囲を最大化
- **Projection / Head Layout Planner**
  - Wq/Wk/Wvのoptimizer identityとQNN実行layoutを分離
  - packed QKV、reshape/slice/concat、selector/scatter除去をdevice latencyで選択
- **HTP/HVX Gradient Strategy Search**
  - BP / segmented BP / ZO / hybrid BP+ZO
- **Training Memory Planner**
  - FP KEEP / Quantized KEEP / spill / RECOMPUTE / backend boundary
- **QNN Graph + HVX Kernel Superoptimizer**
  - algebraic rewrite + graph partition + backend placement + device autotune
- **Tensor Representation Planner**
  - storage dtype / compute dtype / group size / transform / rounding
- **Evaluation Execution Planner**
  - batch size / graph reuse / checkpoint reuse / input packingを自動選択
- **Direct Low-Bit MatMul**
- **Hybrid Sequence Operator**
  - Attention / Skip / Gated DeltaNet / recurrentをlayer/head単位で選択
- **Reproducible Data-Mix Planner**
  - source ratio / order / cursor / curriculumをcheckpoint identityへ含める
- **FAST / BALANCED / LOW MEMORY / BEST QUALITY presetの自動生成**

---

# 基礎

## GUI / Android実行基盤

### 実装済み

- Material 3 Expressive化
- 推論GUI
- 学習中のphase / seed / step / loss / CPU・HTP担当表示
- Fwd+Backward / Adam時間表示
- model configの一般化
- checkpointを使ったHTP generation
- prepared generation engine + FORWARD_ONLY graph
- 長時間WorkManager training経路
- 4時間級soak / reliability検証

### 追加したいtelemetry

- peak / current RAM
- 累積original UTF-8 bytes/s
- RPC / kernel / pack / unpack / Aux Adam内訳
- HVX worker imbalance / pcycles
- Z-Loss / grad norm / Attention logit max

## アプリ単体化

長時間training runnerやcheckpoint/resume基盤はかなり成熟したが、**dataset原本からtokenizer/cache生成までを完全に端末UIだけで完結する部分**は引き続き別課題として扱う。

- `nicopedia_2024`等の入力directoryをアプリ内で変換
- dataset選択 → tokenizer/cache生成 → training
- checkpoint / resume / evaluation / generationを同じUIから扱う

## D / FFN Capacity — 一段落

→ **milestone達成。以後はP1のparameter-matched shape search**

現行の研究baselineは **V1024/T32/D64/FFN128/L19/H2 = 758,528 parameters**。

D64/FFN64 → D64/FFN128ではparametersが約25.8%増えた一方、256+256 held-outのBalanced改善は比較方法により約`0.0054〜0.0100 bpb`だった。改善は小さいが一貫しており、FFN64に多少のcapacity bottleneckがあった可能性はある。ただし次にFFN160/256へ直進する根拠は弱い。

次は総parameterを近づけて、

- D
- FFN
- L
- H
- head dimension
- Attention placement

を入れ替え、**quality / wall-time / original-byte throughput / RAM**のParetoで見る。

## Text Representation Search

→ **現行baseline確立済み。追加探索はP2**

現在の基準は決定的な**Byte BPE V=1024**。tokenizer SHA-256をcache/checkpoint/eval/generationへbindingし、異なるtokenizerのartifactはfail-closedで拒否する。

今後の候補:

### A. Raw Byte

- V=256
- 未知文字なし
- 日本語UTF-8ではsequenceが長くなる

### B. Byte BPE

- **V=1024: 現行baseline**
- V=512 / 2048 / 4096は必要なら再探索
- [BPE論文](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Byte-level BPE](https://arxiv.org/abs/1909.03341)

### C. GBST / Charformer系

- byte入力を維持しながらlearned downsampling
- global Transformerへ渡すsequence lengthを減らす
- [Google: Efficient Sequence Modeling for On-Device ML](https://research.google/blog/efficient-sequence-modeling-for-on-device-ml/)

### D. BLT系

- entropyに応じた可変長byte patch
- 現状では実装・backward・HTP costが大きいため、GBST/簡易patchの後
- [Byte Latent Transformer](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/)

評価指標はtokenizer間で直接比較できるよう、

- **bits per original UTF-8 byte (bpb)**
- original UTF-8 bytes / model step
- original UTF-8 bytes / second
- global Transformer sequence length
- output projection cost
- embedding / optimizer-state bytes
- end-to-end training wall time

を中心にする。

---

# Core Transformer Block Candidates

ここは既存の外部モデルへの互換化を目的とせず、**現行V1024/D64/F128 baselineから一要素ずつ変え、QNN HTP上の速度・RAM・数値安定性・学習品質を比較する独立候補**として扱う。Forward/Backwardは原則QNN HTPを基準にし、QNN内部精度が要件を満たさない演算だけHVX等へ配置する。

## RMSNorm

→ **P1**

- [RMSNorm](https://arxiv.org/abs/1910.07467)
- Violettoはblock normとQ/Kの正規化にRMS系を使う。HexaTrain側にも`QNN_OP_RMS_NORM`の存在確認はあるが、正式baseline経路ではdevice-testedではないため、まずmicrotestを通す

比較:

- 現行LayerNorm
- RMSNorm
- weightless RMSNorm
- Zero-Centered RMSNorm

見るもの:

- Forward / Backward node数
- Reduce / centering opの数
- ms/step
- parity error
- gamma / scaleの最大値
- 深層化時の数値安定性

Limiteを根拠にQK Normまで同時変更せず、**block normだけを先に分離A/B**する。

## FFN Activation / Gating

→ **P1**

Violetto本体はSwiGLUだが、Paradigmaのspeedrun系ではReLU²も使われている。HexaTrainでは追加projectionを持つSwiGLUより、**matrix shapeを変えないReLU²を先に試す**。

比較順:

1. 現行ReLU FFN
2. **ReLU²** = `relu(x)^2`
3. Gated ReLU / ReGLU
4. SwiGLU

ReLU²はparameter数とMuon matrix packingを変えず、QNN側にelementwise squareと対応backwardを足すだけで比較できる。

SwiGLU / ReGLUはprojectionが増えるため、品質だけでなくHTP上のMatMul増加・Muon/Aux Adam role・checkpoint payloadを必ず測る。

参考:

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Paradigma: A Retrospective on Our World Records](https://paradigma.inc/blog/a-retrospective-on-our-world-records/)

## RoPE

→ **P1**

比較:

1. 現行position方式
2. Full RoPE
3. Partial RoPE
4. 将来: layer-wise RoPE / NoPE

RoPEを採用すること自体は前提にしない。

## Zero-Centered RMSNorm

→ **P1〜P2**

Qwen3-NextでQK-Normのnorm weightが大きくなる問題への対策として採用された。

- [https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d)

比較時は、

- norm gamma max
- gamma RMS
- gradient norm
- validation NLL
- HTP/CPU差

を記録する。

---

# Hardware-Aware Architecture Search

## 基本探索

→ **P1**

現行`yuubinnkyoku/HexaTrain`のV1024/T32/D64/FFN128/L19/H2をcontrolに、総parameterを近づけたshape比較を行う。

探索parameter:

- D
- FFN
- L
- H
- Query head数
- KV head数
- head dimension
- Attentionを置くlayer
- residual / compute width
- context length
- text compression rate

評価:

- validation / development NLL
- ms/step
- Forward / Backward / Optimizer個別時間
- tokens/s / original UTF-8 bytes/s
- peak RAM
- tensor bytes moved
- graph node / tensor count
- graph create/finalize時間
- thermal / energy
- finite / QNN failure / CPU fallback
- HTP/CPU parity

### Meta MobileLLM-Flashからの教訓

- deep-and-thinが品質上有利でも、実機latencyでは不利な場合がある
- FLOPs/parameter数はlatency proxyとして弱い
- Attentionを一部layerから丸ごと外す`Skip Attention`も探索対象になり得る

参考:

- [https://arxiv.org/abs/2603.15954](https://arxiv.org/abs/2603.15954)

最初はparameter数を近づけて、

- shallow / wide
- medium
- deep / narrow

をV81で比較する。

## Attention Placement Search

→ **P0〜P1**

最初の安価な探索:

- Full Attention
- Skip Attention

例:

- AAAAA...
- AASAA...
- ASASA...
- 3層以上連続Skipは別途品質確認

その後、

- Gated Attention
- GQA
- Local
- Gated DeltaNet
- recurrent operator

へ拡張する。

## Gated Attention

→ **P0**

現行HexaTrainにはすでに`headwise_g1_sigmoid`が実装され、HTP Forward/Backward、NPRTCKPTV5、resume、FORWARD_ONLY generationまで通っている。Candidate2000はcontrolに対しBalanced `-0.007113`で、早期500〜1000 stepでは約`-0.03`台のgainがあった。

現行:

`O' = O * sigmoid(LN1(X) W_g)`

Limite/Violetto実装ではper-head gateをoutput projection前へ置き、`2 * sigmoid(...)`を使う。HexaTrainではまず現行G1を4000まで延長し、その後必要なら次を分離比較する。

- scale 1 vs 2
- random init vs `Wg=0` identity init（scale 2なら初期gate=1）
- input channels D64全部 vs 8 / 16 channel subset
- gate trajectoryのsaturation / sparsity

追加演算が比較的小さいため、GDN/Sparse Attentionより先に扱う。

測定:

- Forward / Backward ms
- gate projectionコスト
- validation / development bpb
- activation RMS / max
- gate mean / std / <0.1 / >0.9
- attention sink指標
- parity error

参考:

- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
- [Limite vLLM implementation](https://github.com/paradigma-inc/limite-violetto)
- [HexaTrain G1 experiment](headwise-g1-gated-attention.md)

## HTP-Aligned GQA

→ **P1**

既存LLMのhead比率をコピーせず、V81実測で決める。

候補:

- MHA
- GQA
- MQA

探索:

- Q heads
- KV heads
- head dimension
- selector/scatter/reshapeコスト
- KV RAM
- Forward / Backward双方

参考:

- [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

## Tied Input / Output Embedding

→ **P1**

小型Qwen / Apple AFM / MobileLLMに加え、Violettoの公開vLLM実装でもLM headはinput embeddingを共有する。

`Embedding[V,D]` と `Output[D,V]` を共有し、

`2VD → VD`

へ近づける。

現行V1024/D64では**65,536 parameters**を削減でき、758,528 parametersに対して約**8.6%**。ただしModded-NanoGPT系ではuntied headが品質改善に使われた例もあるため、節約量だけで採用せず、

- tied baseline
- untied baseline
- 浮いた65,536 paramsをFFN / depthへ再配分したparameter-matched model

を比較する。

## MatFormer / Elastic Model Search

→ **P1〜P2**

Google MatFormer / Gemma 3n、NVIDIA Flextron / LLaMaFlexの方向。

1 checkpoint内に複数FFN幅やsubmodelをnestし、

- FAST
- BALANCED
- QUALITY

を切り出せる可能性がある。

まず独立モデルのsweepをbaselineとして作り、その後MatFormerが順位を再現できるか確認する。

参考:

- [https://arxiv.org/abs/2310.07707](https://arxiv.org/abs/2310.07707)
- [https://proceedings.mlr.press/v235/cai24e.html](https://proceedings.mlr.press/v235/cai24e.html)

## Puzzle / Distillation-Based NAS

→ **P1〜P2**

NVIDIA Puzzle:

- block候補をteacherからlocal distillation
- 候補ごとのhardware costを実測
- 全モデルをfull pretrainせず組合せ探索

HexaTrain候補block:

Attention:

- Full
- Skip
- Gated
- GQA
- later GDN

FFN:

- ReLU
- Gated ReLU
- width F32 / F64 / F96 / ...

最初はOptuna/direct sweep、その後探索空間が大きくなったら導入する。

参考:

- [https://arxiv.org/abs/2411.19146](https://arxiv.org/abs/2411.19146)

## AltUp / Representation-Compute Width Separation

→ **P2**

Google AltUp / Recycled-AltUpから、

`D_repr != D_compute`

を探索する。

例:

- D_repr=64, D_compute=16
- D_repr=64, D_compute=32
- D_repr=128, D_compute=32

表現容量とHTPが実際に処理するMatMul幅を分離できる可能性がある。

参考:

- [https://research.google/blog/alternating-updates-for-efficient-transformers/](https://research.google/blog/alternating-updates-for-efficient-transformers/)

## Data-Movement-Aware Search

→ **P1**

FLOPsだけでなく、

- transpose
- reshape
- scatter / gather
- APP_READ / APP_WRITE
- parameter upload
- activation spill
- graph boundary
- bytes read / write

を重視する。

参考:

- PFN TensorLayout
- Microsoft Ladder
- NVIDIA hardware/model co-design

## 多目的探索

→ **P1**

最初はOptuna等。

目的:

- maximize: held-out quality
- minimize: step latency
- minimize: peak RAM
- minimize: energy / thermal cost
- minimize: graph init/finalize cost

最終的にPareto frontを保存する。

---

# 学習安定化

## QK Norm

→ **P1**

Qwen3 / PLaMo等で学習安定化目的に採用。

比較:

- no QK Norm
- RMS QK-Norm
- per-head QK Norm
- Zero-Centered RMSNorm版

Qwen3-Nextでは一部norm weightが大きくなる問題が観測され、Zero-Centered RMSNorm + norm-weight decayへ移行している。

必ず記録:

- Q norm gamma max
- K norm gamma max
- gamma RMS
- max attention logit
- grad norm
- non-finite
- HTP/CPU差

## Z-Loss

→ **P1 / 最初はdiagnosticのみ**

PLaMo-100B等を参考に、

- CE
- Z-Loss
- grad norm
- max attention logit
- non-finite count

を記録。

必要なら後から`CE + λ Z`を比較。

## QK-Clip / Attention Logit Stabilization

→ **P2**

MuonClip / QK Clip等はQK-Norm・Gated Attentionで不足した場合に追加。

---

# Gradient Strategy / On-Device Training Method

HexaTrainではQNNのBackwardを明示実装できる点が強みだが、**全てをExact BPに固定する必要もない**。

## A. Exact Backpropagation

→ **baseline**

現在のHexaTrain training baselineの中心。

- exact gradient
- scratch training
- architecture比較の基準

## B. Apple MeBP型 Segmented Backprop

→ **P2（RAMが先に制約になった場合はP1へ昇格）**

`Memory-Efficient Backpropagation for Fine-Tuning LLMs on Resource-Constrained Mobile Devices`を参考に、

- layer単位にForward/Backward graphを分割
- checkpoint
- mmap / spill
- 必要layerだけload
- activationをmemory hierarchyへ逃がす

という経路を検討する。

HexaTrainでは、

`巨大1 graph`

vs

`per-layer / segment graph`

を比較する。

参考:

- [https://arxiv.org/abs/2510.03425](https://arxiv.org/abs/2510.03425)

## C. Qualcomm QZO-FF / sign-m-SPSA

→ **P2 / personalization優先**

Backwardを使わず、perturbationした2回のForwardからgradient方向を推定。

HexaTrainでは、

- Exact BP
- SPSA
- sign-SPSA
- sign-m-SPSA

を同一checkpoint / datasetで比較する。

見るもの:

- loss decrease / step
- loss decrease / second
- RAM
- energy
- total convergence time

fine-tuning / personalization用途では特に有力。

参考:

- [https://arxiv.org/abs/2411.04036](https://arxiv.org/abs/2411.04036)

## D. Google Addax

→ **P2**

First-order gradientとZeroth-order gradientを混ぜる。

HexaTrainでは将来的に、

- sample-wise BP/ZO
- layer-wise BP/ZO
- tensor-wise BP/ZO

へ拡張できるか検討。

参考:

- [https://research.google/pubs/addax-utilizing-zeroth-order-gradients-to-improve-memory-efficiency-and-performance-of-sgd-for-fine-tuning-language-models/](https://research.google/pubs/addax-utilizing-zeroth-order-gradients-to-improve-memory-efficiency-and-performance-of-sgd-for-fine-tuning-language-models/)

## QZO-LoRA / LR-QAT / IR-QLoRA

→ **P2 / personalization**

全weightではなく低rank adapterだけを更新することで、

- ZOの探索次元削減
- low-bit base model維持
- on-device personalization

を狙う。

参考:

- Qualcomm LR-QAT: [https://arxiv.org/abs/2406.06385](https://arxiv.org/abs/2406.06385)
- ByteDance IR-QLoRA: [https://arxiv.org/abs/2402.05445](https://arxiv.org/abs/2402.05445)

---

# Training Memory / Recomputation

## 目標: HTP Memory Planner

→ **P1**

各tensorについて、

```text
KEEP FP
KEEP low-bit
SPILL / mmap
RECOMPUTE


```

から実機cost最小のものを選ぶ。

統合する先行研究:

- PFN FastSA / MN-Core recomputation
- Meta MODeL
- Apple MeBP
- NVIDIA COAT
- PrismMLのmean-centered low-bit storage

## Selective Activation Recompute

候補tensor:

- Q / K / V
- attention score
- probability
- context
- Norm intermediates
- FFN pre/post activation

単純に「RAMが減るか」だけでなく、

- save/write時間
- read時間
- recompute時間
- HTP kernel効率

を測る。

## Xiaomi MiMo

- MiMo-V2.6
- AdamW → Muown mid-training bridge
- large-batch fully asynchronous RL
- Groupwise Reward Synthesis / Advantage Redistribution
- MOPD / MOPD²
- stable mixed-task sample ratio
- router freeze / R3系のtrain-inference consistency
- 7k+ RL task environments / composable harness公開方針

HexaTrainでは特に**Muown・effective batch・MOPD²の縮約**を取り込む。MoE router技術は現行Denseモデルへは適用しない。

## PFN Transfer-Aware Recomputation

PFN MN-CoreではDRAM転送より再計算した方が速いケースを利用している。

HexaTrainでも、

`memory bytes saved / extra HTP recompute time`

だけでなく、**step latency自体が改善するか**を見る。

参考:

- [https://tech.preferred.jp/ja/blog/mncore-compiler-optimization-with-recompute/](https://tech.preferred.jp/ja/blog/mncore-compiler-optimization-with-recompute/)

## FastSA型自動探索

計算graph上の保存/再計算配置を自動探索。

HexaTrainでは最初はheuristic、その後simulated annealing等へ。

参考:

- [https://openreview.net/forum?id=fbpTObq6TW](https://openreview.net/forum?id=fbpTObq6TW)

## Meta MODeL型 Lifetime / Placement最適化

tensor lifetime、placement、spill、rematerialization、reduced precisionを一つの問題として扱う。

参考:

- [https://ai.meta.com/research/publications/model-memory-optimizations-for-deep-learning/](https://ai.meta.com/research/publications/model-memory-optimizations-for-deep-learning/)

## Quantized Activation Checkpointing

候補:

- FP32 / FP16相当
- INT8
- INT4
- 将来INT2
- RECOMPUTE

さらに、

- per-tensor
- per-channel
- per-group

を比較する。

NVIDIA COATでは非線形/Norm周辺で細粒度group量子化を使うため、HexaTrainでもtensor種類ごとにgranularityを変える。

参考:

- [https://research.nvidia.com/labs/eai/publication/coat/](https://research.nvidia.com/labs/eai/publication/coat/)

## Mean-Centered Activation Quantization

PrismML KV cacheのmean-centeringをactivation保存へ応用。

`x - mean → quantize → save → dequantize → +mean`

をA/Bする。

## Token-Selective / Mixed Precision Activation

Qualcomm STaMPの思想を応用し、

- 大部分token: low-bit
- 重要token: higher precision

を将来比較する。

## Graph Partition Search

巨大1 graphが必ず最速とは仮定しない。

比較:

- 1 graph / step
- Forward | Backward | Optimizer
- per-layer
- several-layer segment
- sequence chunk × layer segment

評価:

- execute latency
- APP tensor traffic
- peak RAM
- QNN internal scheduling
- create/finalize
- resume/checkpointとの相性

---

# QNN / HVX Graph・Kernel Optimization

## Heterogeneous Superoptimization

→ **P0〜P1**

HexaTrainではForward/Backward/Adamの数式をQNN opへ明示的に落とせる一方、Muon Newton–SchulzではQNN HTPの内部精度が要求を満たさず、HVX FP32が有効だった。したがって探索空間を

```text
math
  → equivalent mathematical forms
  → QNN graph candidates / HVX kernel candidates
  → boundary & layout candidates
  → V81 profiling

```

へ広げる。

対象:

- RMSNorm forward/backward
- Softmax forward/backward
- Gated Attention
- Attention backward
- FFN backward
- Adam
- Muon / matrix optimizer
- Gated DeltaNet

各variantについて、

- compile/finalize成功
- CPU oracleとの差
- non-finite / fallback
- execute / RPC / kernel latency
- pack/unpack / candidate copy
- RAM / tensor traffic
- node count / worker数

を測る。

### Muonから得た実測上の教訓

- QNNでFLOAT32 tensorを宣言しても、反復MatMulの実効数値精度がCPU FP32相当とは限らない。
- backend precision gateは「最終lossが有限」より前に置く。
- HVXは8 workerまでmatrix-level parallelismが効き、kernelではW1→W8で約6.31x、RPC込みでも約3.33xまで伸びた。
- VTCMは常に速いわけではなく、今回のMuon scratchではDDRより遅かった。
- accelerator kernelを速くした後は、host pack/unpack・FastRPC・state copyが次のボトルネックになる。

## Qwen FlashQLAからの教訓

FlashQLAではGated DeltaNetのForward/Backwardを数式のまま実装せず、hardware-friendlyになるよう代数的に再定式化している。

HexaTrainでも「同じ数学 = 同じgraph/kernel」と考えない。ただし数式rewriteでCPU oracleとの同値性を失わないことを最優先する。

参考:

- https://github.com/QwenLM/FlashQLA
- [https://qwen.ai/blog?id=flashqla](https://qwen.ai/blog?id=flashqla)

## Prism / Equality Saturation

Prismのrewrite / equivalence class / target profilingを参考にする。CUDA固有scheduleをコピーするのではなく、QNN graph・HVX kernel・host boundaryを含む候補集合に適用する。

参考:

- [https://arxiv.org/abs/2604.15272](https://arxiv.org/abs/2604.15272)

## Ladder型 Storage / Compute dtype分離

- storage dtype
- compute dtype
- conversion location
- memory hierarchy

を分けて探索する。QNNとHVX間の共有bufferでも同じ原則を使う。

## Tensor Layout / Data-Movement Rewrite

対象:

- transpose
- reshape
- head split/concat
- selector/scatter
- Q/K/V projection fusion / packed execution weight
- KV layout
- contiguous / non-contiguous representation
- QNN APP_READ/APP_WRITE
- RPC wire layout
- HVX canonical matrix layout

Muonでpack planeの不要なresize初期化を除いただけでactual optimizer update medianが`126.333 → 104.137 ms`まで短縮したため、layout/data movementは補助最適化ではなくP0/P1の研究対象とする。

Violettoの公開vLLM実装はQKVを1つのprojectionへまとめる。HexaTrainではoptimizer/checkpoint identityを壊さず、更新後にpacked execution cacheを作る方式を優先して検証する。また現行H2経路のhead selector/scatter MatMulがreshape/slice/concatより本当に速いかをV81で直接測る。

---

# Multi-Token Prediction

## Dense Multi-Token Supervision / Same-Head MTP-lite

→ **P1**

Paradigmaのspeedrun系では、**1つのlanguage-model headを複数の連続future targetへ同時に学習させる**方式が使われている。これは追加MTP headを持つ方式と分けて評価する。

現行HexaTrainのHTP cross-entropy gradientは、

`dlogits = (P - Y) / N`

なので、`Y`をone-hotから確率分布へ変えるだけなら、softmax / logits / output projection / backward tensor shapeを増やさずに複数future targetのlossを合成できる。

T32の候補:

- `Y = 1/3 * (onehot(t+1) + onehot(t+2) + onehot(t+3))`
- `Y = 0.5 * onehot(t+1) + 0.3 * onehot(t+2) + 0.2 * onehot(t+3)`

必要変更:

- cacheを現在の`context+1`から少なくとも`context+3`を表現できる形式へ拡張
- CPU target builderをsoft-target対応
- scalar loss/evaluatorは通常next-token bpbとmulti-target training lossを分離記録
- final quality判定は通常next-token bpbで行う

これでgainが無い場合、full MTP headへ進まない。

参考:

- [Paradigma: A Retrospective on Our World Records](https://paradigma.inc/blog/a-retrospective-on-our-world-records/)

## MTP-1 / NEXTN head

→ **P2**

Dense Multi-Token Supervisionで「未来tokenを補助教師にする」価値を確認した後、追加headを持つMTPを独立比較する。

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3)
- [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)
- [Step 3.5 Flash](https://arxiv.org/abs/2602.10604)
- [GLM-5](https://arxiv.org/abs/2602.15763)
- [Hy4-preview](https://huggingface.co/tencent/Hy4-preview)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [dots3-note Preview](https://github.com/studio-dots-ai/dots3-note-prev)

2026年8〜9月のHy4、DeepSeek-V4.1-Flash、dots3-noteでもMTP系が継続して使われているため、単発の研究候補ではなく**training auxiliaryとinference accelerationを接続できる技術**として評価する。

## Shared MTP

→ **P2後半**

GLM-5では複数段MTPでparameterを共有し、段数増加によるparameter memory増加を抑える。

## Staged MTP Expansion / Weight Copy

→ **P2後半**

Step 3.5 Flashでは大半の学習をMTP-1で行い、後半にMTP-1からMTP-2/MTP-3へweight copyして共同学習する。

最初から多段MTPを持つより学習負荷を抑えられる。

## MTP → Speculative Decoding

→ **P2 / inference連携**

dots3-note Previewはshared MTP / NEXTNを投機的生成へ使い、SGLang構成で**TPOTを50%以上削減できる場合がある**と説明している。

HexaTrainでは、

1. training中はMTP-1を補助lossとして使う
2. inferenceでは同じheadをdraft token生成へ流用
3. acceptance rate / extra head cost / end-to-end latencyを測る

という一体評価を行う。

MTPの価値を`validation NLLが下がるか`だけで判定せず、**学習時の追加costを推論時に回収できるか**まで見る。

参考:

- [dots3-note Preview](https://github.com/studio-dots-ai/dots3-note-prev)

---

# Optimizer

Optimizerは**quality / state RAM / accelerator placement / end-to-end wall time / update geometry**で比較する。現在はAdamとOriginal Muonが二本の実証済みbaselineであり、MiMo-V2.6を受けてMuownを次の主要研究候補へ上げる。

## Adam baseline

→ **production-quality control**

現行V1024/T32/D64/FFN128/L19/H2では、S4000 schedule

```text
LR 0.0022 through step 4000
→ linear decay
→ LR 0.0001 at step 8000
```

を凍結controlとして保持する。Muon/Muownやarchitecture変更の比較では、Adam側を追従チューニングして基準を動かさない。

## Muon + Auxiliary Adam

→ **formal quality baseline / production**

### Parameter split

- Muon: Wq / Wk / Wv / Wo / W1 / W2、19層で114 matrices、622,592 parameters
- Aux Adam: embedding / output / norm等、135,936 parameters
- total: 758,528 parameters
- checkpoint: NPRTCKPTV4、Adam V3とのcross-resumeは拒否
- role / shape / fan-in / fan-outはparameter metadata SSOTから取得

### Quality evidence

8,000-step × 2 seed:

| optimizer | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: |
| Adam S4000 Balanced | 2.413116 | 2.405908 | **2.409512** |
| HVX Muon Balanced | **2.337821** | **2.350119** | **2.343970** |
| Muon − Adam | -0.075295 | -0.055789 | **-0.065542 bpb** |

- 両seedでVal / Devとも改善
- RPC failure / fallback / non-finite=`0 / 0 / 0`
- HVX uninterrupted vs HVX resumeはbitwise PASS
- 2 seed平均約`1313.29 ms/update`

Muonは「将来候補」ではなく、architecture研究に使える正式baselineである。

### QNN HTP-native Muon

→ **BLOCKED / 同経路の再試行は停止**

QNN graph自体は成功しfinite/no-fallbackだが、Newton–SchulzのMatMulでprecision errorが増幅する。

- normalized 64x64 relative L2: 約`8.3e-4`
- NS1: 約`2.15e-3`
- NS5: 約`5.22e-2`

新しいprecision機構や別backend evidenceなしにthresholdを緩めて採用しない。

### HVX FP32 Original Muon

→ **formal baseline**

Original Muonの数学を保ったHVX FP32 W8実装。single-updateでは114/114 weight/momentum comparisonを通過し、worst maxAbs約`1.49e-8`、worst relative L2約`4.4e-8`。

長期ではCPU double oracleと同一trajectoryにはならないため、評価基準を

1. single-update correctness
2. training quality / stability
3. same-backend resume reproducibility
4. end-to-end wall time

に分離する。

## Muown

→ **P0 diagnostic / P1 implementation**

MiMo-V2.6で最重要の追加候補。MuownはMuonのdirection updateを残しつつ、weight matrixを

`W = Diag(g / ||R||row) R`

とみなし、**row magnitude `g`をAdam、direction `R`をMuon**で更新する。Muonで観測されるspectral-norm driftの主因をrow magnitude側へ分離して制御する。

### まず実装せずに診断する

既存HVX Muon checkpointで、各114 matrixについて以下をstep別に計測する。

- spectral norm
- max / median / RMS semantic row norm
- row-coherence factor
- weight / gradient / update effective rank
- row-wise angular update
- update-to-parameter ratio

Muown論文と同様に`spectral norm ↑`が主に`max row norm ↑`で説明できるなら実装根拠が強い。driftがほぼ無ければ優先度を下げる。

### Semantic rowの定義

Muownのrowは**出力neuron**に対応する。保存layoutの単純な先頭dimensionをrowとみなさない。

現行metadataでは`fan_out_axis` / `fan_in_axis`がSSOT化されているため、それを唯一のorientation sourceにする。

特に:

- `ffn_w1`: stored `[MODEL, FEED_FORWARD]` → output countはFFN側
- `ffn_w2`: stored `[FEED_FORWARD, MODEL]` → output countはMODEL側

Muon Newton–Schulzの計算都合によるtranspose orientationと、Muown magnitudeのsemantic orientationを分離する。

### 追加state

現行114 matricesの出力row総数は、

`19 × (Wq64 + Wk64 + Wv64 + Wo64 + W1 128 + W2 64) = 8,512`

Muown Algorithm 1のrow state `g / r / m_g / v_g`をFP32で持つと、

`8,512 × 4 × 4 bytes = 136,192 bytes ≈ 133 KiB`

程度。現行Muon momentumに対して小さく、端末RAM上は第一の障害ではない。

### 実装方針

1. CPU referenceを論文Algorithm 1に忠実に実装
2. Original Muon `keller_original_64560829_fp32`を変更しない
3. Muown専用optimizer identityを追加
4. NPRTCKPTV4/V5の既存state意味論へ無理に詰め込まず、新schemaで`g/r/m_g/v_g`を明示
5. 1 / 8 / 32 step correctness
6. 500 / 2000 step Val/Dev
7. 有望ならHVX direction pathへ接続

最初のA/Bでは**NS5固定**とする。MiMo-V2.6の大規模RL設定やMuown論文の別NS回数を同時変更すると、row controlの寄与が分からなくなる。

参考:

- [Muown: Row-Norm Control for Muon Optimization](https://arxiv.org/abs/2605.10797)
- [MiMo-V2.6 Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)

## AngularMuown

→ **P2**

Muownの後続解析では、direction parameterのrow normが育つことで、同じ加算updateでも実際のrow方向回転角が小さくなり、**暗黙のangular step-size decay**が生じると解釈される。

HexaTrainでは先にMuown/Original Muonへangular telemetryを追加し、

- row norm
- update angle
- bpb改善速度

の関係を見る。明確な減衰が品質停滞と結び付く場合のみAngularMuownを実装する。

参考:

- [Muown Implicitly Performs Angular Step-size Decay](https://arxiv.org/abs/2606.23637)

## Adam → Muon / Muown移行

→ **P2 / checkpoint transferを使う場合のみ**

現行のscratch Muon baselineにはoptimizer mismatch問題は直接当たらない。一方、将来Adam-pretrained checkpointをMuon/Muownへ切り替える場合、naive switchでpretrained knowledgeを崩す可能性が報告されている。

比較:

- Adam continue
- Adam → Muon
- Adam → Muown
- 必要なら短いmid-training bridge

同一parent checkpointからbranchし、1/8/32/100/500 stepでbpb、parameter displacement、forgetting指標を測る。

MiMo-V2.6はAdamW pretrainingからMuown mid-trainingを挟んでlarge-batch RLへ移っており、**optimizer変更を独立した移行stageとして扱う**実例として参考にする。

参考:

- [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468)

## Effective Batch × Optimizer

→ **P1〜P2**

MiMo-V2.6 Technical Reportは、大batch側でAdamWの最適化効率が落ち、matrix-aware optimizerが有利になる観察をMuown移行の理由にしている。

HexaTrainでは規模が桁違いなので、その結論をコピーしない。physical B8を基準に、

- B8 × accumulation 1
- B8 × accumulation 2
- B8 × accumulation 4
- B8 × accumulation 8

を候補にし、**同一token / original-byte budget**でAdam / Muon / Muownを比較する。

見るもの:

- bpb / byte
- bpb / wall time
- optimizer update回数
- Forward/Backward回数
- peak RAM
- gradient norm / noise proxy
- row geometry

## Quantized Adam / COAT-like

→ **P2**

Muon/Muownと比較基準を揃えた後、Adam state RAMがscale制約になった時点でFP16/INT16/INT8 stateとDynamic Range Expansionを比較する。

参考:

- [NVIDIA COAT](https://research.nvidia.com/labs/eai/publication/coat/)

## Gradient Multi-Normalization / SWAN

→ **P2**

stateless training候補。Adam / Muon / Muownの実測基準が揃った後、loss/secondとstate RAMで比較する。

## RACS / Alice

→ **P2**

低state optimizer候補。

## GaLore / 8-bit GaLore

→ **P2**

low-rank投影でstate削減。SVD/basis updateをQNN/HVX/CPUのどこへ置くかもコストに含める。

## AdEMAMix

→ **P2**

Adam系の比較候補。state増加とのtrade-offを見る。

## Turbo Muon / 別Muon variants

→ **P2後半以降**

Original Muonの正式baselineが成立したため実験可能ではあるが、MiMo-V2.6を受けた次の情報量はMuownの方が高い。Muon → Muown → 必要ならAngularMuownを先に比較し、その後に別Muon familyへ進む。

---

# Meta Token

→ **P2**

- [Step 3.5 Flash §A.3](https://arxiv.org/abs/2602.10604)

training sample先頭へ、

- content type
- language
- domain
- source

等をmetadata tokenとして付与。

例:

`<type=code><lang=ja><domain=cpp> ...`

metadata自身のlossをmaskし、本文予測の条件として利用する方式も比較。

追加計算量が小さいため、モデル基本構造確定後に試す。

---

# Attention Block

## Attention Normalizer Search

→ **P1〜P2**

比較:

### Softmax

baseline。

### Sigmoid Attention

Appleの研究。

- row-wise normalization/reductionが不要
- HTPで有利かは実測

参考:

- [https://machinelearning.apple.com/research/sigmoid-self-attention](https://machinelearning.apple.com/research/sigmoid-self-attention)

### ReLU / FLARE系

Google Research。

- SoftmaxをReLU系へ
- mobile/edge向けの演算簡略化

参考:

- [https://research.google/pubs/flare-fine-tuned-long-context-acceleration-with-relu-enhanced-fire/](https://research.google/pubs/flare-fine-tuned-long-context-acceleration-with-relu-enhanced-fire/)

測定:

- Forward
- Backward
- node数
- reduction数
- NLL
- 数値安定性

## Per-Head QK Norm

QK Normの粒度をhead単位でも比較する。

## Partial RoPE

→ **P1〜P2**

候補:

- 100%
- 50%
- 25%
- 0%

Qwen3-Nextはhead dimensionの一部のみRoPEを使う。

短いTでは効果が見えにくいため、速度だけ先に測り、

長文脈品質評価はT拡大後。

## Attention Output Gate / Head-wise Gate

→ **P0**

詳細は上のGated Attention節と[実験記録](headwise-g1-gated-attention.md)を正とする。Limiteを受けた追加比較は`2 * sigmoid`、identity init、reduced gate channels。

## Exclusive Self Attention / XSA

→ **P1**

XSAはattention outputから現在token自身のvalue方向成分を除き、context由来成分を使わせる方向の補正。

HexaTrainでは固定100%除去ではなく、

`Y' = Y - tanh(alpha[l,h]) * proj_vself(Y)`

のlearnable strengthを使い、`alpha=0`初期化でbaselineと同じforwardから開始する。

順序:

1. CPU oracle + gradient check
2. 6層subset
3. 全19層
4. G1との組み合わせは単独効果確認後

T32では長文脈論文のgainを外挿せず、bpb / wall time / parityで採否を決める。

参考:

- [Exclusive Self Attention](https://arxiv.org/abs/2603.09078)
- [Limite vLLM implementation](https://github.com/paradigma-inc/limite-violetto)

---

# Residual / 層間接続

## Learned Residual / Branch Scale

→ **P1**

Violettoは単純な`x + branch`ではなく、attention / MLPの各joinにlearned coefficientを持つ。

HexaTrainではまず各layerごとに、

`x' = lambda_resid * x + lambda_branch * branch`

を使い、`lambda_resid=lambda_branch=1`でidentity initializationする。attention側2 scalar + FFN側2 scalarなら19層で76 scalar程度。

MUDDやmulti-stream residualより圧倒的に安いため、深さ19の情報流改善を調べる第一候補とする。

## Value Residual / Value Routing

→ **P1〜P2**

Violettoのtoken-dependent Value EmbeddingをそのままD64へ入れる前に、既存layerのVを再利用する軽量版を試す。

候補:

- `V_l' = V_l + beta_l * V_0`
- `V_l' = lambda_l * V_l + beta_l * V_0`
- 3〜4層おきだけvalue residual

V1024/D64で別value embedding tableを全面追加すると相対コストが大きいため、まず追加tableなしで効果を判定する。

参考:

- [Value Residual Learning](https://arxiv.org/abs/2410.17897)
- [Limite vLLM implementation](https://github.com/paradigma-inc/limite-violetto)

## MUDD-lite / Dynamic Dense Residual Mixing

→ **P2**

ViolettoではMUDDを全48層へ入れず限定layerで使う。HexaTrainでも、

- 2箇所程度
- 3 taps
- 小さいinternal width

から始める。

D64ではhistory activation保持とextra graph node/tensorの固定費が相対的に大きいため、**Learned Residual Scale → Value Residual → MUDD-lite**の順で進む。

参考:

- [MUDDFormer](https://arxiv.org/abs/2502.12170)

## Gated Residual / Multi-Stream Residual

→ **P2**

Qwen3.8-Flash-Nextではresidual streamを**4分岐**し、read / writeを動的gateで制御するGated Residualを採用している。

HexaTrainでは4-wayをそのままコピーせず、まず:

- ordinary residual
- 1-stream + scalar/channel gate
- 2-stream gated residual
- later 4-stream

の順で増やす。

見るもの:

- extra projection / elementwise cost
- Forward / Backward node数
- activation RAM
- gradient flow / norm
- validation NLL
- HTP/CPU parity

mHCやAttnResより実装を小さく始められるため、**P2のresidual候補**として先に比較する。

参考:

- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## Attention Residuals / AttnRes

→ **P3**

- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [Moonshot Repo](https://github.com/MoonshotAI/Attention-Residuals)
- [Kimi K3](https://arxiv.org/abs/2607.24653)

固定`x = x + layer(x)`ではなく、過去layer出力を深さ方向Attentionで混合。

Full / Block AttnResを候補にする。

## Manifold-Constrained Hyper-Connections / mHC

→ **保留 / P3以降**

- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)

単一residual streamを複数経路へ拡張し、混合行列に制約を与える。

2026年8月公開のGLM-5.3-FlashでもmHC採用が確認でき、巨大モデル側では実用例が増えた。ただしHexaTrainではGated Residualよりgraphが重くなる可能性が高いため、優先順位は上げない。

HexaTrainではAttnResよりさらに後。

---

# Attention Architecture

## Full / Skip

→ **P0〜P1**

Meta MobileLLM-Flash型。

特殊sparse kernelなしで、Attentionを置くlayerそのものを減らす。

## Multi-Scale Local Attention

→ **P3**

Tが大きくなってから。

## Local / Global Hybrid

→ **P2〜P3**

Gemma 3の`5 local : 1 global`等をbaselineにする。

現在T=32では優先しない。

## Shared K=V / MQA系

GQA / MQAの一部として比較。

## Cross-Layer KV Sharing

→ **P2**

Apple AFM / Gemma 3n / Hymbaで実用例。

一部layerでK/V projectionとcacheを共有する。

効果:

- KV RAM
- K/V projection MatMul削減
- prefill短縮可能性

## YOCO

→ **P2〜P3**

Microsoft。

global KVを一度だけ作り、後段layerで共有。

Cross-Layer KV Sharingのさらに強い比較対象。

参考:

- [https://www.microsoft.com/en-us/research/publication/you-only-cache-once-decoder-decoder-architectures-for-language-models/](https://www.microsoft.com/en-us/research/publication/you-only-cache-once-decoder-decoder-architectures-for-language-models/)

## Hybrid Attention Layout

→ **P2〜P3**

layer単位:

- Full
- Skip
- Local
- Gated DeltaNet
- recurrent

## Search-Based Attention Layout

→ **P2〜P3**

最初:

- Full / Skip

次:

- Full / GDN

その後:

- Local / recurrent / sparse

## QSA / DSA型 Indexed Sparse Attention

→ **P3**

2026年8〜9月の公開モデルでは、単なる固定window sparseより**小さなindexerで重要blockを選ぶ方式**が目立つ。

- Qwen3.8-Flash-Next: Gated DeltaNet + Qwen Sparse Attention（QSA）、micro-block選択
- Hy4-preview: Gated DSA + IndexCache
- DeepSeek-V4.1-Flash: sparse attention + compressed indexer
- dots3-note Preview: 13 DSA + 33 sliding-window attention

HexaTrainでは長文脈化後、最初から複雑なlearned sparsityを作らず、

1. fixed block size
2. fixed Top-K budget
3. cheap score/indexer
4. block Gather
5. Full Attention teacherとのA/B

から始める。

測定:

- indexer Forward / Backward ms
- TopK / Gather / Scatter ms
- selected KV bytes
- attention本体ms
- recall / validation NLL
- `indexer overhead > saved attention cost`になっていないか
- layer間でindexを再利用した場合の効果

IndexCacheは**Cross-Layer Index Sharingの実モデル側の根拠**として扱い、QNN上でindex再計算よりcache/reuseが速いかを測る。

参考:

- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Hy4-preview](https://huggingface.co/tencent/Hy4-preview)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [dots3-note Preview](https://github.com/studio-dots-ai/dots3-note-prev)

## FlexPrefill型 Head-Adaptive Attention

→ **P3**

ByteDance Seed。

head / inputごとにsparsity patternとbudgetを変える。

Tが十分大きくなってから。

参考:

- [https://arxiv.org/abs/2502.20766](https://arxiv.org/abs/2502.20766)

## Mixture-of-Depths

→ **P2**

Google DeepMind。

固定capacity `k` tokenだけをlayerで処理し、残りをskipする。

static tensor shapeを維持できるのがQNN向き。

ただしTopK / Gather / ScatterのHTP costを必ず測る。

参考:

- [https://arxiv.org/abs/2404.02258](https://arxiv.org/abs/2404.02258)

## Evolutionary Data-Flow Search

SakanaのData Flow Spaceの考え方を転用。

各layer/headのsequence operatorを探索する。

## Cross-Layer Attention

長文脈時の候補。

---

# Linear / Recurrent Attention

2026年8月のQwen3.8-Flash-Nextは`Gated DeltaNet + QSA`、GLM-5.3-Flashは`sparse + linear attention`を採用した。巨大モデル側でも**Attentionを全layerで同じ形に固定せず、線形/recurrent系と疎Attentionを組み合わせる**方向が強まっている。

HexaTrainではこの事実を「GDNを採用すべき」という結論ではなく、**Hybrid Sequence Operator Searchを優先する根拠**として扱う。

## recurrent memory head

→ **P2〜P3**

Attention以外の固定state型sequence operatorを比較する入口。

## SimpleGDN

→ **P2**

簡易Gated Delta系baseline。

## Gated DeltaNet

→ **P2**

Qwen3-Next / Qwen3.5/3.6に加え、Qwen3.8-Flash-Nextでも主要sequence operatorとして採用。

元技術はNVIDIA Research等のGated Delta Networks。

比較:

- Full Attention
- Gated Attention
- Gated DeltaNet
- 3 GDN : 1 Attention hybrid

見るもの:

- Forward / Backward
- state RAM
- sequence scaling
- recall
- parity

参考:

- [https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule](https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule)
- [https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d)

## Hymba Hybrid-Head

→ **P2**

同一layerのheadを、

- Attention heads
- recurrent / SSM heads

に分ける。

layer-wise hybridより細かい探索。

さらにHymbaではmeta tokens / cross-layer KV sharingも併用される。

参考:

- [https://research.nvidia.com/publication/2025-04_hymba-hybrid-head-architecture-small-language-models](https://research.nvidia.com/publication/2025-04_hymba-hybrid-head-architecture-small-language-models)

## Nemotron-H型 Hybrid

→ **P2〜P3**

少数Attention + 多数Mamba/recurrent layer。

`Attention Skip`のskip部分をrecurrent operatorへ置換する発展として扱う。

## Griffin / RecurrentGemma

→ **P2**

Google DeepMind。

gated linear recurrence + local attentionの強い比較対象。

参考:

- [https://arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427)

## HTP専用Delta Attention

→ **P3 / 独自候補**

Gated DeltaNet等の標準baselineを先に実装・比較した後、

- QNN op数
- recurrent state layout
- elementwise/reduction構成
- V81 shape

に合わせて簡略化した独自variantを設計する。

---

# Learned Memory Eviction / KV Memory

長文脈では「一つのKV圧縮法」にまとめず、問題を分解する。

```text
WHAT to store?     → Cross-Layer KV / YOCO
WHICH to keep?     → KeyDiff / NAMM / KVP
HOW to compress?   → INT4 / CommVQ / low-rank / Trellis
WHERE to place?    → ShadowKV / RAM hierarchy
WHEN to reposition?→ PIE / STRING


```

## NAMM

→ **P3**

Sakana AI。

attention historyからtokenをkeep/forgetするlearned policy。

## HTP-NAMM Lite

まずは、

- EMA(attention received)
- recent attention
- token age

等の安いscoreで比較。

## KeyDiff

→ **P3**

Qualcomm。

Key同士のcosine similarityから特徴的なKeyを残すtraining-free eviction。

NAMMより先に試せる。

## KVP

Appleのper-head policy系。

学習policyを使うKV evictionとして比較。

## CommVQ

→ **P3**

Apple。

RoPEと可換になるようなvector quantizationでKVを強く圧縮。

## ShadowKV

→ **P3**

ByteDance Seed。

- hot/low-rank/index情報
- cold value cache

を異なるmemory階層へ置く考え。

HexaTrainではHTP / RAM / 将来UFSの階層へ読み替える。

## LaCache

→ **P3**

NVIDIA。

token方向だけでなくlayer方向にもcache budgetを配分する。

## Trellis / Lattice

→ **P3**

Google。

- Trellis: KVを固定サイズmemoryへlearned recurrent compression
- Lattice: 重複方向を抑え、新規情報中心にmemory更新

## PIE / STRING

→ **P3**

ByteDance。

- PIE: RoPE付きK cacheのpositionを再変換して再利用
- STRING: relative position分布を再配置して長文脈外挿を改善

---

# Multi-Matrix / Output Projection候補

## Multi-Matrix Factorization Attention

→ **P3以降**

- [MFA](https://arxiv.org/abs/2412.19255)
- [Step-3](https://arxiv.org/abs/2507.19427)
- [Step-3 GitHub](https://github.com/stepfun-ai/Step3)

## Grouped Low-Rank Output Projection

→ **P3以降**

- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)

head groupごとにlow-rank projectionを挟んでからDへ戻し、`Wo`計算を減らす候補。

---

# Compressed Attention

## Compressed Sparse Attention / CSA

→ **P3**

- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)

`複数token KV → compressed KV → indexer → Top-K → Sparse Attention`

直近tokenは生KVを別branchで保持。

## Heavily Compressed Attention / HCA

→ **P3**

強くKVを圧縮し、Indexerを使わず圧縮KV全体へDense Attention。

候補:

- 4 token → 1 compressed KV
- 16 token → 1 compressed KV
- 直近32 token → 生KV

---

# Sparse Attention

Qwen3.8-Flash-Next、GLM-5.3/5.3-Flash、Hy4-preview、DeepSeek-V4.1-Flash、dots3-note Previewまで、**sparse/indexed attentionは長文脈の実用系モデルで連続して採用**されている。

ただしHexaTrainの現在の短いTではindexerやGatherの固定費が支配的になりやすい。したがってP3据え置きで、Tを伸ばした段階で**Full / Local / Block Sparse / Indexed Sparse**を同一HTP graph条件で比較する。

## Streaming-Aware Attention

→ **P3**

- [LongCat Sparse Attention](https://arxiv.org/abs/2608.01662)
- [LongCat-2.0](https://github.com/meituan-longcat/LongCat-2.0)

理論FLOPsよりmemory accessの連続性を重視。

## Block Sparse GQA

→ **P3**

- [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
- [MSA](https://github.com/MiniMax-AI/MSA)

token単位でなく連続block単位で選択し、不規則Gatherを減らす。

## Hierarchical Indexing

→ **P3**

1. 粗いblock選択
2. block内部を詳細選択

## Cross-Layer Index Sharing

→ **P3**

K/Vそのものではなく、`どのtoken/blockを見るか`をlayer間共有。

- [You Only Index Once](https://arxiv.org/abs/2606.06467)

## Indexer Distillation

→ **P3**

1. Full Attention modelを学習
2. 本体固定
3. Indexerだけ蒸留
4. Sparseへ切替
5. 全体追加学習

## Spectral-Aware Block Selection

→ **P3後半**

RoPE後のQ/Kを単純mean poolingすると位置の高周波成分を失う可能性があるため、low/high frequencyを分けてblock importanceを推定する方向。

- [Prism: Spectral-Aware Block-Sparse Attention](https://arxiv.org/abs/2602.08426)

Tが十分大きくなってから。

---

# Memory

## N-gram Embedding / Engram

→ **P3**

- [Engram](https://arxiv.org/abs/2601.07372)
- [Engram GitHub](https://github.com/deepseek-ai/Engram)
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

Qwen3.8-Flash-Nextは大容量N-gram Embeddingを**計算量をあまり増やさず容量を増やす外部的memory**として使い、host memoryへの退避・prefetchも想定している。DeepSeek-V4.1-FlashでもN-gram系memoryが確認できる。

HexaTrainでは巨大tableをコピーせず、

`token embedding + small bigram/trigram table`

から始める。

比較軸:

- table bytes
- hit/access pattern
- APP/HTP transfer bytes
- mmap / host-backed table
- prefetch有無
- NLL改善 / added latency
- tokenizer方式（byte/BPE）との相互作用
- Engram / N-gram → 静的 / 連想memory
- recurrent memory → 動的memory

として併用可能。

---

# 低精度化

## 原則

「モデル全体をINT4」のような1個のdtypeで考えない。

探索対象:

- tensor/op
- storage dtype
- compute dtype
- scale granularity
- group size
- outlier transform
- rounding
- QAT recipe

## Operation / Tensor-Wise Precision Search

→ **P2**

候補例:

- QKV projection
- QK MatMul
- Attention normalizer
- AV MatMul
- FFN
- RMSNorm
- LM head
- gradient
- optimizer m/v
- checkpointed activation

Apple / PLaMo / NVIDIAの結果から、SoftmaxやLM head等は高precisionに残した方がよい場合がある。

## INT2 / Ternary Baseline

→ **P2**

比較:

### Apple Balanced 2-bit

`{-1.5s, -0.5s, +0.5s, +1.5s}`

- learnable scale
- balanced zero-centered levels
- weight decay等のrecipeも比較対象

### BitNet b1.58

`{-s, 0, +s}`

training from scratch向けの強いternary baseline。

### Ordinary INT2

QNN/HTPで実際に使える表現へ合わせる。

重要:

低bit「保存形式」とHTPが実際に使うcompute kernelは分けて検証する。

## Progressive Quantization

Qualcomm UPQを参考に、

`FP → INT4 → INT2`

の段階的学習を比較する。

## QAT Oscillation Monitoring

QualcommのQAT oscillation研究から、

- quantized value flip count
- flip rate / parameter group

を記録。

必要ならdampening / iterative freezingを比較。

## Scale Granularity

候補:

- per-tensor
- per-channel
- group 16 / 32 / 64 / 128 / 256
- micro-block

Meta mobileはgroup=32、PrismML等では128の例があるが、V81で探索する。

## Quantization-Friendly Reparameterization

→ **P2**

比較:

- mean center
- Hadamard
- QuaRot
- SpinQuant
- FPTQuant

目的:

outlierを分散/変換し、同じbit数でも量子化誤差を減らす。

## Stochastic Rounding

→ **P2**

NVIDIA NVFP4 training等を参考に、

- nearest
- stochastic

を比較。

小さいgradientが低bit丸めで系統的に消える問題を抑えられるか見る。

## COAT-Style Activation / Optimizer Compression

低bitをMatMulだけに限定せず、

- saved activation
- Adam m/v

にも適用。

## STaMP / Token-Wise Mixed Precision

→ **P2〜P3**

一部tokenだけ高precisionに残す。

## IR-QLoRA / LR-QAT

→ **P2**

低bit base + low-rank adaptation。

ByteDance IR-QLoRA:

information retentionを基準にquantizerを補正。

Qualcomm LR-QAT:

low-rank parameterだけでQAT。

## Ladder-Style Storage/Compute dtype

storage bit数とcompute bit数を分離し、変換位置を探索。

## HTP Direct Low-Bit MatMul

→ **P3 / 独自候補**

比較:

A. packed low-bit → FP/INTへ展開 → MatMul

B. packed low-bit → native higher-bit primitiveへ変換

C. custom packed/LUT/bitwise path

Microsoft T-MAC / Ladderを設計参考にする。

## N\:M Sparsity × Ternary

→ **P3**

Sparse-BitNet / SlideSparseの方向。

QNN HTPで対応primitiveが確認できる場合のみ優先度を上げる。

---

# Model Scaling / Weight Transfer

## Weight Reuse

→ **P2**

小→大へ既存weightを埋め込んで継続学習。

例:

`D16 F32 → D32 F64 → D64 F128`

architecture searchで小モデルを先に試し、有望構造を育てる用途。

## Scale Ladder / Intermediate Checkpoints

→ **P2**

K2 Horizonは0.9B / 3.7B / 7B / 32B / 36B-MoE / 375B-MoEの複数scaleを同時に出し、data / method / code / intermediate checkpointsまで追跡可能にする方針を示した。

HexaTrainでは絶対scaleは桁違いだが、研究方法はそのまま使える。

- `tiny → small → medium`のshape ladderを固定
- 各scaleで同じA/Bを短く実施
- 小scaleでの順位が大scaleでも保たれるかを見る
- architecture変更前後のintermediate checkpointを保存
- failed runもconfig付きで残す

これにより、単一の「最終best model」だけでなく、**どの技術がscaleして効いたか**を検証できる。

参考:

- [IFM K2 Horizon](https://ifm.ai/)

## Structured Pruning + Distillation

→ **P2**

PFN PLaMo / Meta MobileLLM-Flash / NVIDIA Minitron系を参考にする。

大きいparentから、

- D
- FFN
- layer
- attention block

を削減し、teacherで回復。

## Puzzle / Minitron / miniPuzzle

候補architectureをscratchから全て長時間学習せず、

- parent
- prune / local distill
- short continuation
- HTP profile

で探索コストを下げる。

## Top-K / Sparse Logit Distillation

PLaMo / Gemma等。

vocabularyが大きくなった場合、全teacher logitsを保存せず、

- Top-K
- sampled logits
- residual mass

で蒸留memoryを削減。

## Model Merging during Pre-training

→ **P2 / 実験**

ByteDance Seed。

shape互換checkpointを複数短時間学習してmergeし、

有望性評価や学習短縮に使えるか検証。

端末間の疎結合checkpoint mergeはHexaTrain独自応用候補。

## Flextron / LLaMaFlex / MatFormer

1 checkpointから複数submodelを切り出すelastic modelとして比較。

---

# Quality-Diversity / Evolutionary Search

## CycleQD型Quality-Diversity Architecture Search

→ **P2**

- [Sakana AI CycleQD](https://sakana.ai/cycleqd/)

CycleQDは「一番良い1個」ではなく、異なるBehavior Characteristicsを持つ高品質な個体群を維持する。

HexaTrainへの転用:

Behavior Characteristics例:

- step latency
- peak RAM
- energy
- parameter count
- long-context score

Quality例:

- validation NLL
- generation quality

結果として、

- FAST
- BALANCED
- LOW MEMORY
- BEST QUALITY

のような複数nicheを残す。

最初からCycleQDを使わず、通常の多目的探索が成立してから導入する。

## Evolutionary Data-Flow Search

SakanaのEvolutionary Model Mergeの「Data Flow Space」探索をarchitectureへ転用。

各layerの種類を遺伝子とする。

---

# Data / Post-Training Strategy

HexaTrainの中心はHTP上のscratch trainingだが、2026年8〜9月の小型・open modelでは**architectureだけでなくdata / post-trainingの公開度と設計**が性能差として大きい。

## Reproducible Data Pipeline

→ **P0〜P1 / research protocol**

MiniCPM5-2BはUltra-FineWeb / UltraData-Math / UltraData-Code / SFT-Agent / RLなど、モデルと結び付いたdata群を公開している。K2 Horizonもpretrain / midtrain dataを含む再現性重視の公開方針を採る。

HexaTrainではNicopediaを基準にしつつ、

- raw source manifest
- preprocessing version
- tokenizer model/hash
- train/dev split hash
- sample order / seed
- curriculum stage
- exact checkpoint parent

を固定する。

外部dataを使う場合も、**データ追加とarchitecture変更を同時に行わない**。

参考:

- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [UltraData](https://ultradata.openbmb.cn/)
- [IFM K2 Horizon](https://ifm.ai/)

## Stable Multi-Source Data Mix

→ **P1〜P2 / 複数corpusを混ぜる時点**

MiMo-V2.6はmulti-task RLでtask sample ratioを安定化させる仕組みを明示している。HexaTrainでも日本語本文 / code / math / conversation等を混ぜる場合、単純shuffleだけに依存しない。

checkpoint identityへ、

- source manifest/hash
- target source ratio
- source cursor
- source-specific order seed
- curriculum stage

を含め、resume後もmix比率を再現する。

## MOPD / MOPD²

→ **P2〜P3 / external-teacher track**

MOPDはdomain-specific teacherを別々に作り、**student自身のon-policy rollout**上でteacherのdense token-level signalを使って能力を統合する。MiMo-V2-Flashで実運用され、MiMo-V2.6ではMOPD²へ発展している。

MOPD²ではstudentの通常rolloutだけでなく、Teacher-Prefix / SFT-Prefixの履歴を再利用したsingle-turn rolloutを混ぜ、重要なdecision pointを学習するためにprefix全体を毎回再生成する費用を減らす。

HexaTrainへの縮約候補:

1. stronger teacherでtrajectoryをオフライン生成
2. 重要位置の直前T32 prefixを切り出す
3. teacher next-token / Top-K distributionを保存
4. studentをT32の既存training pathでdistill
5. ordinary SFT / offline distillation / on-policy-liteを比較

これは「端末だけで知識を獲得する」scratch-training trackとは研究目的が違うため、明示的に別trackとする。

参考:

- [MOPD](https://arxiv.org/abs/2606.30406)
- [MiMo-V2.6 Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)

## GRS / GAR型 Groupwise Grading

→ **P3 / long-horizon RL開始後**

MiMo-V2.6ではbinary pass/failだけでなく、同一promptの成功trajectory同士を比較してreward/advantageを再配分し、より短いpath・少ないtokenで成功するtrajectoryを優先する。

現行next-token LMには直接の挿入点がない。将来reasoning / code agent / Android tool-use RLへ進んだ場合に、

- correctness
- path length
- token count
- tool-call count
- verifier confidence

を分離してreward化する。

## MoE Router Consistency / R3

→ **現状対象外 / MoE化時P0級**

MiMo-V2.6は大規模RLでMoE routerをfreezeし、MiMo系R3研究ではinference時routingをtrainingへreplayしてtrain/inference不整合を抑える。

現行HexaTrainはDenseなので実装しない。将来tiny MoEへ進む場合は、expert selectionを離散stateとしてcheckpoint / rollout / backward間で整合させる。

参考:

- [R3](https://arxiv.org/abs/2510.11370)

## Self-Improvement / Synthetic Curriculum

→ **P3**

Ornith-1.5は、

`task proposal → scaffold generation → rollout → RL`

のself-improvement loopを採用する。

HexaTrainでは端末上で全ループを回す必要はなく、将来、

- small modelが苦手例を抽出
- teacher / stronger modelが課題・解答を生成
- HexaTrainでshort SFT / RL
- 次のhard-example mining

という外部teacher併用版を比較できる。

## Agent Trajectory Post-Training

→ **P3 / personalization以降**

Smaug系は人手選別した実環境agent trajectoryと難例由来のsynthetic dataを重視する。

HexaTrainがtool use / Android操作へ進む場合、

- result token
- tool call
- observation
- hidden reasoning相当部分

のどこへlossを掛けるかを明示し、通常SFTと比較する。

現時点では**HTP training architecture探索より後**。

---

# Task Adaptation / Personalization

## LoRA baseline

on-device adaptationでfull weight更新より軽い基準。

## QZO-LoRA

Qualcomm QZO-FFをlow-rank parameterへ限定し、ZOの探索次元を減らす。

## LR-QAT

Qualcomm。

low-rank parameterでquantization-aware adaptation。

## IR-QLoRA

ByteDance Seed。

量子化時のinformation retentionを改善。

## Transformer² / Singular Value Finetuning

→ **P3**

SVDのsingular value方向だけをtask adapterとして更新。

LoRA/QZO-LoRAと比較。

## Titans / Test-Time Memory

→ **P3〜P4**

Google。

推論中にneural memory自体を更新する。

HexaTrainは端末上training pathを持つため、将来的には普通のinference-only frameworkより検証しやすい可能性がある。

---

# Inference側

学習研究とは分けるが、同じmodel/graph技術を再利用できるものを残す。

## Thinking / Compute Budget Control

→ **P2〜P3 / inference policy**

Granite 4.2はthinkingの有無や低い推論量を指定できる形を標準化している。

HexaTrainでは専用architectureを作る前に、

- direct answer
- short scratchpad / limited reasoning tokens
- tool call allowed
- max generation budget

を同一modelで切り替え、

- quality
- wall-clock latency
- energy
- generated tokens

のParetoを見る。

小型端末モデルでは「常に長く考える」より、**taskごとに計算budgetを制御する方が実用上重要**な可能性がある。

参考:

- [Granite 4.2 language models](https://github.com/ibm-granite/granite-4.2-language-models)

## Quantized KV Cache

候補:

- FP16相当
- INT8
- INT4
- CommVQ
- mean-center calibration

PrismMLの4-bit KVはmean-centering biasを利用。

## Cross-Layer KV / YOCO

KVを全layer分保持しない。

## KV Eviction

- KeyDiff
- NAMM
- KVP

## KV Placement

- ShadowKV
- LaCache

## Speculative Decoding

候補:

- separate drafter
- **MTP / NEXTN head** 
  - training時のauxiliary headをそのままdraftへ再利用できるか比較
- LayerSkip/self-speculative
- OmniDraft

## LayerSkip / Early Exit

Meta。

training時にearly-exitしやすい表現を作り、

同一modelの前半layerをdraftとして使う。

## Fast BLT / multi-byte generation

byte-level/latent tokenizationを採用した場合の将来候補。

---

# 比較実験の実行順

## Phase 0 — 2026-09 baseline凍結【済】

固定済み:

- V1024/T32/D64/FFN128/L19/H2
- Byte BPE tokenizer identity
- dataset split / training order
- Adam S4000 control
- Val/Dev protocol
- untouched final sampleのread-once評価
- QAIRT `2.48.40.260702151143` / V81

final結果を今後のcandidate tuningへ使い戻さない。

## Phase 1 — HVX Original Muon formal baseline【済】

- QNN HTP Forward / Backward
- HVX FP32 W8 Original Muon
- CPU Aux Adam
- 1-step correctness
- same-backend resume reproducibility
- 8,000 step × 2 seed
- Adam S4000とのmatched Val/Dev比較

結果: Muon平均`2.343970` vs Adam平均`2.409512`、**-0.065542 bpb**。Muon+Aux Adamをformal quality baselineへ昇格済み。

## Phase 2 — Muon Geometry Audit【最優先・低コスト】

**完了（2026-09-22）。** 結果は [muon-row-geometry-audit.md](muon-row-geometry-audit.md)。
seed1/seed2 とも `max row norm ↑` と `spectral norm ↑` が corr≈0.99 で同期し、
step 250→8000 で約2.4〜2.7倍に成長。late の angular update も縮む。
**Muown の優先度は下げない。** Phase 4 へ進んでよい。

既存checkpointだけを使い、

- semantic row norm
- spectral norm
- row coherence
- effective rank
- row-wise angular update

をstep 0〜8000で抽出する。

目的は「Muownを実装したい」ではなく、**現行MuonにMuownが解くべき現象が存在するか**を先に判定すること。

## Phase 3 — Evaluation Engine + Low-Cost Architecture Lane【P0並行】

Evaluation:

- persistent evaluator graph
- multi-window batching
- checkpoint/graph reuse
- original UTF-8 bytes/s
- full-final相当の推定所要時間

並行して、長いimplementation chainを必要としない候補を早期に判定する。

- G1 candidateを4000まで延長
- ReLU² smoke → 500 / 2000
- Dense Multi-Token SupervisionのCPU oracle / cache設計 → 500 / 2000
- packed QKV / selector-scatter除去はquality-neutral microbenchmarkとして別lane

研究iteration速度そのものを改善しながら、低コストarchitecture候補を先に刈り込む。

## Phase 4 — Muown CPU Reference / Short A/B

geometry auditが肯定的なら進む。

- semantic fan-out row orientation
- `g/r/m_g/v_g` state
- 新optimizer identity / checkpoint schema
- NS5固定
- 1 / 8 / 32-step correctness
- 500 / 2000-step Val/Dev
- Original Muon / Muown / Adam control

ここで改善が無ければHVX化しない。

## Phase 5 — Muown HVX Promotion

CPU/reference Muownが有望な場合のみ。

- direction Muonの既存HVX primitive再利用可能性を検証
- row-state updateをCPU/HVXのどこへ置くかprofile
- pack/RPC/state trafficを含むactual path
- resume reproducibility
- 8,000-step seed 1 → seed 2

## Phase 6 — Effective Batch × Optimizer

- effective batch 8 / 16 / 32 / 64
- Adam / Original Muon / Muown
- same original-byte budget
- quality / wall time / RAM / optimizer invocation count

MiMo-V2.6のlarge-batch observationがtiny on-device trainingでも再現するかを見る。

## Phase 7 — Architecture A/B

一度に一つ変更。Phase 3で早期判定した候補は、ここで正式な長期比較へ昇格する。

- Full vs Skip Attention
- Gated Attention（現行G1 / Limite-style scale2・identity-init）
- RMSNorm
- ReLU² → 必要ならSwiGLU / ReGLU
- Learnable XSA
- Learned Residual / Branch Scale
- Value Residual
- RoPE
- QK Norm / Zero-Centered RMSNorm
- tied embedding

MUDD-liteは上記の軽量residual/value候補が不十分な場合のみ追加する。

最初は現行D64/F128/L19/H2を固定する。

## Phase 8 — Parameter-Matched Shape Search

- D / FFN / L / H
- MHA / GQA / MQA
- head dimension
- Attention placement
- shallow/wide vs deep/narrow

parameter数/FLOPsではなく、bpb・wall time・original-byte throughput・RAMのParetoを保存する。

## Phase 9 — Memory / Data Movement

- KEEP / low-bit KEEP / RECOMPUTE / spill
- graph partition
- persistent APP buffers
- direct RPC layout
- shared arena / double buffering
- pack/unpack transpose除去

## Phase 10 — Text Representationの再探索

現在のByte BPE V1024をcontrolに、必要なら

- raw byte
- BPE V512/2048/4096
- GBST
- simplified BLT

をbpbとoriginal-byte/sで比較する。

## Phase 11 — Gradient / Optimizer alternatives

- MeBP
- QZO-FF
- Addax
- Quantized Adam
- SWAN / Gradient Multi-Normalization
- GaLore / RACS / Alice / AdEMAMix
- AngularMuown（Muownで暗黙angle decayが確認された場合）

Adam / Original Muon / Muownのcontrolを残す。

## Phase 12 — Low Precision

- INT8 / INT4 / balanced INT2 / ternary
- storage dtype != compute dtype
- activation checkpoint precision
- optimizer-state precision
- stochastic rounding / outlier transform

QNN HTP内部精度の実測を無視して「FP32 tensorだから安全」と仮定しない。

## Phase 13 — Distillation / Data-Mix

外部teacher trackを有効にする場合:

- deterministic source mix
- Top-K / sparse-logit distillation
- MOPD-lite
- prefix reuse / MOPD²-lite
- domain teacher integration

scratch-training trackと結果を混同しない。

## Phase 14 — Hybrid Sequence Operators

- Gated DeltaNet
- Attention/GDN hybrid
- Gated Residual
- Hymba hybrid-head
- Mixture-of-Depths
- MTP + speculative decoding

## Phase 15 — Long Context / Sparse / RL

Tを十分増やして固定費を回収できる段階で、

- Local / Global
- Cross-Layer KV / YOCO
- QSA / DSA
- FlexPrefill
- Cross-Layer Index Sharing
- KV compression / eviction

へ進む。

long-horizon RLを始める場合のみGRS/GAR型reward設計を追加し、MoE化する場合のみR3/router consistencyを追加する。

## Phase 16 — Automated Co-Search

探索parameter:

- architecture
- backend placement
- attention placement
- precision
- optimizer
- memory/recompute
- graph/kernel rewrite
- tensor/RPC layout
- effective batch
- data mix

最終的にはPareto / Puzzle / MatFormer / QD / Evolutionary Searchへ接続する。

---

# 研究機関別に取り込んだ主な技術

## Paradigma

- Limite 1B - Violetto
- head-wise attention gate / reduced-channel gate
- Exclusive Self Attention (XSA)
- Value Embedding
- MUDD selective placement
- learned residual / branch scale
- packed/fused QKV implementation
- pretraining speedrun系のReLU² / same-head multi-token supervision / RRE
- 「モデル構造と実行layoutを同時に詰める」設計のreference

## Qualcomm AI Research

- QZO-FF / sign-m-SPSA
- LR-QAT
- UPQ: INT4→INT2
- QAT oscillation / freezing
- FPTQuant
- STaMP
- DONNA hardware-in-loop NAS
- KeyDiff

## Apple Machine Learning Research

- MeBP
- balanced 2-bit QAT
- Cross-Layer KV Sharing
- Cut Cross-Entropy
- Sigmoid Attention
- AdEMAMix
- CommVQ / KVP / EpiCache
- tied embedding

## Microsoft Research / MSRA

- BitNet b1.58 / a4.8
- Gradient Multi-Normalization / SWAN
- RACS / Alice / COSMOS
- Ladder
- T-MAC
- QuaRot
- YOCO
- Q-Sparse / Sparse-BitNet
- Cross-Layer Shared Routing

## ByteDance Seed

- IR-QLoRA
- UltraMem
- FlexPrefill
- ShadowKV
- PIE / STRING
- pretraining checkpoint merge

## Meta FAIR / Meta GenAI

- MobileLLM-Flash
- Attention Skip
- BLT / Compute-Optimal Tokenization
- MODeL
- GaLore
- Memory Layers
- LayerSkip
- MobileLLM quantization
- tied embedding

## Google Research / DeepMind

- Addax
- MatFormer / Gemma 3n
- AltUp
- GBST / Charformer
- FLARE
- Mixture-of-Depths
- Griffin / RecurrentGemma
- Trellis / Lattice
- Titans

## NVIDIA Research

- COAT
- NVFP4 training recipeの設計思想
- Puzzle
- Hymba
- Gated DeltaNet
- Nemotron-H
- Minitron / Flextron
- LatentMoE
- LaCache

## Alibaba Qwen Team

- Gated Attention
- QK Norm
- Zero-Centered RMSNorm
- Partial RoPE
- MTP
- Gated DeltaNet + Attention hybrid
- FlashQLA
- tied embedding / GQA
- long-context chunking / head-wise sparse config

## DeepSeek

- Compressed / indexed sparse attention
- N-gram associative memory / Engram direction
- Hyper-Connection / mHC
- MTP
- V4.1-Flashの`memory + sparse indexer + MTP`統合例

## Tencent

- Hy4-preview
- Gated DSA
- IndexCache
- iHC
- MTP

## IFM

- K2 Horizon scale family
- data / method / code / intermediate checkpoint公開
- reproducible training protocolのreference

## OpenBMB / ModelBest

- MiniCPM5-2B
- on-device small model設計
- UltraData群
- pretrain → midtrain/SFT → RLまでの公開pipeline

## IBM Granite

- controllable thinking / reasoning budget
- tool-use evaluation
- small/medium Dense familyのdeployment reference

## dots studio / Xiaohongshu

- DSA + sliding-window hybrid
- shared MTP
- MTP / NEXTN speculative decoding

## Ornith AI / Abacus.AI

- self-improvement loop
- agent trajectory / synthetic hard-example post-training
- architectureだけでなくpost-training data設計を比較するreference

## PFN

- FastSA
- transfer-aware recomputation
- TensorLayout
- QK Norm / Z-Loss（PLaMo）
- weight reuse
- structured pruning/distillation

## Sakana AI

- CycleQD
- Evolutionary Data-Flow Search
- NAMM
- Transformer²
- adaptive computeの発想

## Prism / PrismML

- Quantized activation/KV保存のmean-center
- ternary/1-bit deploymentの設計
- symbolic tensor-program superoptimization
- speculative drafter

---

# 最終的に目指す形

```text
Dataset / Text Representation
  Byte BPE baseline / Raw byte / GBST / BLT
        ↓
Model Search
  D / F / L / H / Q-KV / head_dim
  Attention placement / recurrent / sparse
        ↓
Gradient + Optimizer
  Exact BP / segmented / ZO / hybrid
  Adam / Muon+Aux Adam / low-state
        ↓
Mathematical Rewrite
  equivalent forward/backward/optimizer forms
        ↓
Backend Placement
  QNN HTP  |  HVX  |  CPU
       ↘ precision gate ↙
        ↓
Graph / Kernel / Boundary Search
  fusion / partition / worker count
  APP tensors / FastRPC / pack / shared buffers
        ↓
Memory + Precision Planner
  KEEP / low-bit / spill / recompute
  storage dtype / compute dtype / rounding
        ↓
Evaluation Engine
  persistent graph / batch / checkpoint reuse
  bpb / original UTF-8 bytes/s
        ↓
V81 Device Profiling
  latency / RAM / transfer / power / thermal / parity
        ↓
Automated Search
  Pareto / Puzzle / QD / Evolution
        ↓
Presets
  FAST / BALANCED / LOW MEMORY / BEST QUALITY

```

HexaTrainの研究主張は、

> **Android上で学習を成立させるだけでなく、モデル構造・optimizer・数式表現・QNN graph・HVX kernel・backend境界・memory・precisionを、Snapdragon実機の品質とwall timeに基づいて共同最適化する**

ことへ広げる。

Muonで得られた結果は、この方向をかなり明確にしている。QNN HTPが得意なForward/Backwardと、数値精度を明示的に制御できるHVXを組み合わせる方が、単一backendへ統一するより実機上の最適解に近い可能性が高い。MiMo-V2.6がMuownで示したように、次はbackendだけでなく**optimizer内部のgeometry（row magnitude / direction / angular step）**まで共同最適化の対象へ広げる。

Limite 1B - Violettoからは、さらに**attention / value / residualの情報経路と、QKV・head layoutの実行形を別々に最適化する**視点を加える。HexaTrainでは巨大モデルの構成を縮小コピーせず、G1・ReLU²・Dense Multi-Token Supervision・XSA・Residual/Value routingの順に、追加costの小さい候補からV81実機で選別する。

---

# 2026-09-22 一次資料

## HexaTrain

- [HexaTrain repository](https://github.com/yuubinnkyoku/HexaTrain)
- [current main HEAD `315d048`](https://github.com/yuubinnkyoku/HexaTrain/commit/315d0487d98584e9f172dc77972e3fee52ded3d4)
- [HVX FP32 Muon / formal baseline](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/nicopedia-hvx-muon.md)
- [Muon quality pilot](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/nicopedia-v1024-d64-ffn128-muon-pilot.md)
- [HTP-native Muon investigation](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/nicopedia-htp-muon.md)
- [Headwise G1 gated-attention experiment](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/headwise-g1-gated-attention.md)
- [Transformer parameter metadata SSOT](https://github.com/yuubinnkyoku/HexaTrain/blob/main/metadata/transformer_parameter_metadata.json)
- [V1024/D64/FFN128 frozen final evaluation](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/nicopedia-v1024-d64-ffn128-final-evaluation.md)
- [D64/FFN128 capacity experiment](https://github.com/yuubinnkyoku/HexaTrain/blob/main/docs/nicopedia-byte-bpe-v1024-d64-ffn128.md)

## Limite / Paradigma

- [Limite 1B - Violetto release](https://paradigma.inc/blog/limite-1b-violetto/)
- [Limite 1B - Violetto model](https://huggingface.co/paradigma-inc/limite-1b-violetto)
- [Limite Value Model MODEL_DETAILS.md](https://huggingface.co/paradigma-inc/limite-1b-value-model/blob/main/MODEL_DETAILS.md)
- [Limite Violetto vLLM implementation](https://github.com/paradigma-inc/limite-violetto)
- [Paradigma: A Retrospective on Our World Records](https://paradigma.inc/blog/a-retrospective-on-our-world-records/)
- [Exclusive Self Attention](https://arxiv.org/abs/2603.09078)
- [MUDDFormer](https://arxiv.org/abs/2502.12170)
- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
- [Value Residual Learning](https://arxiv.org/abs/2410.17897)

## MiMo / Optimizer

- [MiMo-V2.6 Official Release](https://mimo.mi.com/docs/en-US/news/latest/v2-6)
- [MiMo-V2.6 Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
- [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)
- [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
- [Muown: Row-Norm Control for Muon Optimization](https://arxiv.org/abs/2605.10797)
- [Muown official implementation](https://github.com/kcc-lion/muown)
- [Muown Implicitly Performs Angular Step-size Decay](https://arxiv.org/abs/2606.23637)
- [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468)
- [MOPD: Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2606.30406)
- [Stabilizing MoE RL by Aligning Training and Inference Routers / R3](https://arxiv.org/abs/2510.11370)

