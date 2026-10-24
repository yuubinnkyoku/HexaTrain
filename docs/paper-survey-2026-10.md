# 外部論文サーベイ 2026-10 — AlphaXiv / arXiv 中心

> **初回作成: 2026-10-25 JST。branch `research/paper-survey-2026-10`（専用 worktree）で管理する。**
>
> この文書は [research-priorities.md](research-priorities.md) のオープンな研究課題に対して、
> AlphaXiv / arXiv 等の外部論文から使える知見を拾い、**HexaTrain のどの課題へどう対応するか**を
> まとめるための living document である。論文そのものの網羅的な紹介が目的ではなく、
> 「現在の優先順位に情報を足すかどうか」を基準に選ぶ。
>
> 各エントリの構成:
>
> - **要約**: abstract / 本文読みによる 1 段落要約
> - **HexaTrain との関連**: 具体的にどの課題・phase に対応するか
> - **次に読む / 試すこと**: 採用する場合の最初の一歩（実験とは分けて記載）
>
> **注意**: この文書は文献調査が目的であり、いかなる standing verdict や device 実験の
> 判断も変更しない。数値結果の引用は元論文の条件によるもので、V81 実機への外挿ではない。

## クラスタ一覧

| ID | クラスタ | 対応する優先順位項目 |
| --- | --- | --- |
| A | 低精度学習（FP8/FP4/MX、INT8 backprop、stochastic rounding） | 低精度化 / Phase 12 / HMX accumulator 数値 |
| B | on-device / NPU training systems | Gradient Strategy / MeBP / 端末学習基盤 |
| C | Optimizer（Muon 系・低 state・scaling） | Optimizer / Muown / Phase 4-6 |
| D | Accumulator / GEMM kernel 数値（residual fusion・on-chip reuse） | HMX ACC residual feedback (R2) / graph fusion |
| E | 小規模 LM architecture（gated attention・residual・MTP・MQA） | Core Transformer Block Candidates / Phase 7 |
| F | Linear / recurrent・sparse attention | Phase 14-15 |
| G | Training memory / recompute / activation 圧縮 | Training Memory Planner / Phase 9 |
| H | その他（評価engine・data・tokenizer・推論） | 評価engine高速化 / Data / 推論側 |
| I | Data curriculum（difficulty-aware sampling） | Difficulty-Aware Sampling / Phase 13 |
| J | 小規模学習 recipe の系譜（modded-nanogpt speedrun） | Muon identity / 低コスト候補の收割順 |

（各クラスタの節は調査進行に合わせて追記される。まだ調査していないクラスタは空。）

---

## A. 低精度学習（FP8/FP4/MX、stochastic rounding）

**調査日: 2026-10-25。** AlphaXiv / arXiv 検索から、MX block-scaled 系・FP4 pretraining 系・
stochastic rounding 系の中心論文を拾った。HexaTrain の V81 HMX が対象とする block-scaled
format（MX 系）の学習側文献として、このクラスタは Phase 12 と HMX 数値実験の両方に直接対応する。

### A-1. Microscaling Data Formats for Deep Learning

- [arXiv:2310.10537](https://arxiv.org/abs/2310.10537) / [AlphaXiv](https://www.alphaxiv.org/abs/2310.10537)
  — Rouhani, Zhao, Schulte, Micikevicius, Chung ほか33名（Microsoft / AMD / Intel / NVIDIA consortium）、2023。
  [OCP MX v1.0 仕様](https://www.opencompute.org/en/projects/data-center/doc/wg/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)の評価論文。
- **要約**: 32要素 block + 8-bit shared exponent の上に FP8/FP6/FP4/INT8 を載せる MX format を、
  推論・学習25種以上の workload で評価。master weights / optimizer state / reduction を高精度に
  保った構成で、generative LM の学習を weights / activations / gradients すべて 8-bit 未満で
  回した初の実証とされる。
- **HexaTrain との関連**: HMX（V81）が実装している block-scaled MMA の format family がまさにこれ。
  「MX は drop-in replacement ではなく、何を高精度に残すかの設計が本体」という枠組みは
  低精度化の原則（tensor/op ごとの分離）と同一。group size 32 は Phase 12 の
  scale granularity 候補（group 16/32/64）の直接的な参照点。
- **次に読む**: master weight / reduction をどこへ置いたかの構成表（Table 系）と、
  gradient 側の block scale 扱い。

### A-2. Characterization and Mitigation of Training Instabilities in Microscaling Formats

- [arXiv:2506.20752](https://arxiv.org/abs/2506.20752) — Su, Kwun, Gil, Kakade, Anand（Harvard Kempner Institute）、2025-06。
  [Kempner research page](https://kempnerinstitute.harvard.edu/research/deeper-learning/characterization-and-mitigation-of-training-instabilities-in-microscaling-formats/) / code: `KempnerInstitute/systems-scaling`。
- **要約**: OLMo 系 LM を FLOPs 2e17〜5e19 で約1,000本学習し、MX format の組合せごとの
  不安定化を特徴づけ。すべての MX 組合せが十分長い学習で不安定化し、原因は
  (1) 通常の stochastic-optimization divergence と (2) 低精度 quantum 化による
  **系統的な gradient bias** の2種類。特に MXFP8 E4M3 の overflow clamp が、
  LayerNorm affine weight（1.0 近傍に密に分布）の 32要素 block 全体を同一値へ clamp して
  分散を消す現象を synthetic proxy で分離した。緩和策は「forward だけ量子化 / activation は
  bfloat16 のまま / LayerNorm affine weight を量子化対象から除外」。早期介入は有効だが
  直前介入は divergence を遅らせるだけ。
- **HexaTrain との関連**: 現行 baseline が LayerNorm + gamma を持つ点で、
  「norm 系 weight を低精度化対象から外す」根拠となる一次資料。また QNN HTP 内部精度を
  FP32 と断定しない運用原則と整合する「quantization bias は loss spike の前に gradient norm で
  検知できる」という観測は、Phase 12 を始める前の telemetry 設計（grad norm 連続監視）に
  そのまま使える。
- **次に読む**: synthetic student-teacher proxy の設定（HexaTrain の CPU oracle 診断と同じ型の
  実験設計として参照価値が高い）。
  → **【2026-10-25 AlphaXiv AI overview 経由で追記】**: overflow は
  **FFN 側 layer norm の方が attention 側より多く発生**（proxy activations の約1%・
  OLMo activations の約0.5%が最終 bin に到達）。E4M3 では block max の 87.5% 超
  （max normal 448）で clamp する。in-situ 介入実験（step 4500 / 5080 で
  精度切替、divergence は 5090〜5100 近傍）では早期の backward 量子化無効化は
  有効、直近介入は延期効果のみ。**HexaTrain の tensor 除外表作成では
  「FFN 側 norm を最初に除外候補にする」優先順位の根拠になる**。

### A-3. Pretraining Large Language Models with NVFP4

- [arXiv:2509.25149](https://arxiv.org/abs/2509.25149) / [AlphaXiv](https://www.alphaxiv.org/abs/2509.25149)
  — NVIDIA（Catanzaro, Micikevicius, Mishra, Rouhani, Waleffe ほか90名近く）、2025-09（v2 2026-03）。
- **要約**: 12B model / 10T token を NVFP4 で pretrain した最多規模の 4-bit 学習実証。
  安定化の柱は (1) block outlier を抑える random Hadamard transform、(2) forward/backward 一貫した
  2次元 quantization scheme、(3) unbiased gradient estimation としての stochastic rounding、
  (4) 選択的な高精度 layer。結果は FP8 baseline と同等の loss / downstream accuracy。
- **HexaTrain との関連**: 優先順位文書の P2「stochastic rounding」が参照している NVFP4 recipe の
  正体がこれ。RHT は P2「Quantization-Friendly Reparameterization（Hadamard / QuaRot 系）」と
  同一手法なので、低精度化の各候補（outlier transform / SR / 高精度 layer 選択）は
 独立 factor ではなく一つの recipe として共同設計するべきだという示唆。
- **次に読む**: どの layer を高精度に残したかの内訳（HexaTrain の
  Operation/Tensor-Wise Precision Search の初期候補表になる）。

### A-4. Optimizing Large Language Model Training Using FP4 Quantization

- [arXiv:2501.17116](https://arxiv.org/abs/2501.17116) / [AlphaXiv](https://www.alphaxiv.org/abs/2501.17116)
  — Wang, Gong, Liu ほか（Microsoft / USTC）、ICML 2025。
- **要約**: 初の FP4 LLM training framework とされる。(1) differentiable quantization estimator による
  weight update、(2) activation collapse を防ぐ outlier clamping + compensation。mixed precision +
  vector-wise quantization と組み合わせ、13B / 100B token まで BF16・FP8 と同等精度を維持。
- **HexaTrain との関連**: QAT 系（学習時に量子化誤差を gradient へ通す経路の設計）の代表例。
  手書き backward で量子化演算の gradient を自分で定義できる HexaTrain の構造上、
  straight-through estimator と differentiable estimator の違いを独自に A/B できる。
- **次に読む**: differentiable quantization estimator の定義部。

### A-5. Training with Fewer Bits: Unlocking Edge LLMs Training with Stochastic Rounding

- [arXiv:2511.00874](https://arxiv.org/abs/2511.00874) / [AlphaXiv](https://www.alphaxiv.org/abs/2511.00874)
  — Liu, Andronic, Gündüz, Constantinides（Imperial College London）、2025-11。
- **要約**: edge LLM training を対象に、stochastic rounding を使った低精度 SGD の理論+実証解析。
  mini-batch SGD + SR について、**batch size を大きくすると backprop の精度低下を分散の観点から
  補償できる**こと、weight 量子化と activation 量子化が gradient variance に非対称に効くことを示す。
- **HexaTrain との関連**: physical B8 固定・accumulation 1/2/4/8 という Phase 6 の設計と
  直接組み合わせられる。「低精度 gradient × accumulation」を独立 factor として測る根拠になる。
  HexaTrain の現行 B8 はこの理論で不利な側（batch が小さい）なので、SR 導入時は
  accumulation との同時設計が必須という読み。
- **次に読む**: gradient variance の理論式（どの tensor の量子化が効くかの予測に使える）。

### A-6. A Stochastic Rounding-Enabled Low-Precision Floating-Point Training

- [arXiv:2404.14010](https://arxiv.org/abs/2404.14010) / [AlphaXiv](https://www.alphaxiv.org/abs/2404.14010)
  — Ali ほか、2024-04。
- **要約**: rounding に使う乱数 bit 数を変えた場合の hardware cost と精度への影響を調査。
  最適化した SR により **accumulator の word length を削っても精度を維持できる**ことを示す。
- **HexaTrain との関連**: 「37-bit ACC に留まる積和で十分か」という HMX accumulator 系の問いに
  対して、accumulator bit 幅 × rounding 方式の観点からアプローチする文献。R2 の signed 出力
  実験とは独立に、ACC bit 幅の妥当性を CPU model で測るときの对照枠になる。
- **次に読む**: accumulator word length を削った具体的な実験設定。

### A-7. 古典・補助参照

- Gupta et al., *Deep Learning with Limited Numerical Precision*（ICML 2015）— fixed-point + SR の
  原典。低精度学習の baseline 引用として保持。
- [Improved Stochastic Rounding (arXiv:2006.00489)](https://arxiv.org/abs/2006.00489) —
  SR の unbiased 化の改良。丸めの実装候補比較時に参照。
- [FP8-LM: Training FP8 Large Language Models (arXiv:2310.18313)](https://www.alphaxiv.org/abs/2310.18313)
  — FP8 導入の3段階（delayed scaling 等の granularity 設計）の整理。Phase 12 の
  scale granularity 探索の整理枠。
- MX+（ACM 2025-10）: MX format 自体の拡張。HMX の format 拡張候補を考える際の参照。
- **【2026-10-25 AlphaXiv 推薦経由で追加】**:
  - [Small-scale proxies for large-scale Transformer training instabilities
    （arXiv:2309.14322）](https://arxiv.org/abs/2309.14322) /
    [AlphaXiv](https://www.alphaxiv.org/abs/2309.14322)
    — Wortsman et al.（Google DeepMind）、ICLR 2024（引用220+）。
    **小規模 model で大規模の学習不安定化（attention logit growth・loss divergence）を
    再現・予測する**手法論。LR と安定性の関係、max update size・attention logit growth
    等の指標で規模間予測。HexaTrain の stress harness（P0 共通基盤）の
    指標設計（update norm / update-to-parameter ratio / logit max）と同じ指標系で、
    「小規模診断の結果をどう読むか」の参照になる。
  - *Scaling FP8 Training to Trillion-Token LLMs*（Fishman et al.）—
    trillion-token 級 FP8 学習の安定化。Phase 12 の大規模側参照。要確認。

### A-8. FP4 training recipe の起源系（AlphaXiv related 推薦から追加、2026-10-25）

AlphaXiv の NVFP4 論文ページの related 推薦経由で、**RHT + SR recipe の起源論文**を確認した。

- [Training LLMs with MXFP4（arXiv:2502.20586）](https://arxiv.org/abs/2502.20586) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2502.20586)
  — Tseng, Yu, Park、2025-02（引用55+）。
- **要約**: MXFP4 による 4-bit 学習で高精度学習と ほぼ同等（near-lossless）を達成。
  32要素 group を max magnitude で scale し rounding。ablation で
  **SR と RHT の両方が不可欠**と示し、SR+RHT の分散 bound の理論的正当化も提示。
  適用は **RHT を3種の training GEMM 全部へ、SR は dY（weight gradient）のみ**へ絞る。
- [Quartet（arXiv:2505.14669）](https://arxiv.org/abs/2505.14669) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2505.14669) / code: `IST-DASLab/Quartet`
  — Castro, Panferov, Tabesh, Sieberling, Chen, Nikdan, Ashkboos, Alistarh
  （IST Austria DASLab）、2025-05（NeurIPS 2025 採録、v4 2026-01）。
- **要要約**: 線形層の主要計算をすべて hardware FP4 で実行する end-to-end FP4 学習。
  bit 幅と性能を結ぶ**低精度 scaling law** を Llama 系で同定。FP16/FP8 学習の
  competitive alternative となることを主張。NVFP4 論文（A-3）が採用した
  Hadamard + SR 系技術の直接の先行・並行研究。
- **HexaTrain との関連**: P2「stochastic rounding」の設計粒度がここで確定する。
  すなわち (1) RHT は forward/backward の MatMul 全般、(2) **SR は weight gradient
  （dY）だけ**で near-lossless になる。HexaTrain が SR を試す場合、
  全 gradient ではなく dY（Wq〜W2 の 114 行列分）だけが第一候補になる。
  また AlphaXiv ページ上の人的議論はまだ無いが、related 推薦 graph が
  系譜追跡に有効なことを確認した。
- **次に読む**: MXFP4 論文の SR 分散 bound（A-5 の理論と接続するか）。

### クラスタAの統合所見（2026-10-25時点、A-8 追記後更新）

1. **MX 系 block-scaled 学習は「何を高精度に残すか」で決まる。** Kempner の結果は
   LayerNorm affine weight 除外・activation 高精度維持・forward のみ量子化という
   段階的緩和で FP8 相当まで回復する。HexaTrain の Phase 12 は
   「全 tensor 一括低精度化」ではなく、最初から tensor 除外表を作って始めるべき。
2. **FP4 recipe = RHT + (2D) scale + SR + 高精度 layer 選択 のセット**で、
   起源は Tseng et al.（MXFP4, A-8）→ Quartet（A-8）→ NVFP4（A-3）の系譜。
   SR の適用先は **dY（weight gradient）のみ**で near-lossless になる点が設計上重要。
   HexaTrain の P2 候補（Hadamard 系 reparam / SR / op-wise precision search）は
   個別の候補としてではなく 1つの recipe として扱う方が文献と整合する。
3. **SR × batch size の相互作用**（A-5）は Phase 6 の accumulation 設計と結合する独立研究課題。
   B8 固定の HexaTrain では SR の恩恵が理論上小さくなる可能性があり、accumulation を
   変えない SR 単体の A/B は情報量が低いかもしれない。

---

## B. on-device / NPU training systems

**調査日: 2026-10-25。** 「smartphone / edge で NN の学習を回す」系の文献は
fine-tuning 前提のものが多く、**scratch 言語モデル pretraining を accelerator 上の
exact backprop で回している公開文献は見つからなかった。** HexaTrain の
「QNN HTP Forward/Backward + HVX optimizer + CPU Aux の scratch training」という構成は、
公開文献の範囲では近隣が存在しない位置にある。したがってこのクラスタは
「直接の競合」ではなく、**Gradient Strategy（P2）と memory planner の設計参照**として読む。

### B-1. PockEngine: Sparse and Efficient Fine-tuning in a Pocket

- [arXiv:2310.17752](https://arxiv.org/abs/2310.17752) / [AlphaXiv](https://www.alphaxiv.org/abs/2310.17752)
  — Zhu, Hu, Lin, …, Gan, Song Han（MIT Han Lab）、**MICRO 2023**。
- **要約**: edge 端末上の fine-tuning 用 compact training framework。
  (1) **compile-time に forward/backward/optimizer graph を生成**し runtime overhead を削減、
  (2) sparse backpropagation（backward graph の枝刈り + sparse update）、
  (3) operator reordering / backend switching。Raspberry Pi で最大 15x 高速化、
  Jetson AGX Orin で backprop memory 5.6x 削減、LLaMAv2-7B fine-tune を 550 token/s。
- **HexaTrain との関連**: 「training graph を compile time に固定する」という設計は
  HexaTrain の明示的 QNN op 構築と同型。また「backward graph の枝刈り」は
  Gradient Strategy の segmented BP / hybrid BP+ZO（Phase 11）と連続的な中間解で、
  「exact BP か ZO か」の二択ではなく gradient の枝を選ぶ設計が成立する実例。
  HexaTrain で backward の一部 op（例えば gate 系の小 MatMul）だけ ZO 化する場合の
  先行例として位置づけられる。
- **次に読む**: sparse backprop の精度評価（どの枝を刈れるかの分類）。

### B-2. MeBP — Memory-Efficient Backpropagation for Fine-Tuning LLMs on Mobile

- [arXiv:2510.03425](https://arxiv.org/abs/2510.03425) — Song, Tang（Apple）、2025-10。
  code: `apple/ml-mebp`。repo の P2 にも参照済み。
- **要約**: mobile 上の LLM fine-tuning 向け memory-efficient backprop。
  ZO は memory は軽いが 10〜100x step を要するため、その中間を狙う。
  iPhone 15 Pro Max で 0.5B〜4B model を 1GB 未満の memory で fine-tune。
- **HexaTrain との関連**: repo 側の segmentation 設計（per-layer graph / spill / mmap）の
  一次参照として既に採用済み。追記事項として、Apple 系の mobile ML 論文は
  memory 削減手法と on-device scheduler の記述が細かいので、Phase 9（memory / data movement）の
  実装設計時に本文を読み直す価値がある。
- **次に読む**: layer 分割と activation 保持の粒度（Phase 9 direct）。

### B-3. ZO / forward-only 系の最新動向（MeZO follow-ups）

- 原典: [MeZO (arXiv:2305.17333)](https://arxiv.org/abs/2305.17333) —
  in-place ZO-SGD、memory footprint = inference と同一。
- benchmark: [Revisiting ZO for LLM Fine-tuning (arXiv:2402.11592)](https://www.alphaxiv.org/abs/2402.11592)。
- 新しい系譜（2024-2026）:
  - [LOZO (arXiv:2410.07698)](https://arxiv.org/abs/2410.07698) — gradient の low-rank 構造を
    ZO estimator で利用
  - Sparse MeZO（NeurIPS 2025）— 更新対象 parameter の部分選択
  - LOREN（AAAI 2026）— curvature adaptation + variance reduction を持つ ZO optimizer
  - ZO-Finetuner（ICML 2026, [OpenReview](https://openreview.net/forum?id=2Dn4yHYLQJ)）—
    layer ごとの perturbation scale を学習
  - [Activation-Guided ZO (arXiv:2601.17261)](https://arxiv.org/abs/2601.17261) —
    activation 統計で perturbation を案内
  - Bio-FO（MLSys 2025）— biologically-plausible な forward-only on-device learning
- **HexaTrain との関連**: repo の QZO-FF（sign-m-SPSA）/ Addax（P2）はこの系譜の端点。
  LOZO の「gradient の low-rank 構造」は GaLore 系と同じ観察に由来するため、
  ZO × low-rank × HexaTrain の手書き backward の組み合わせ（一部 layer だけ exact BP、
  残りを low-rank ZO）は Phase 11 の設計空間として文献の裏付けが付いた。
  特に activation-guided 系は、QNN graph から activation 統計を安く観測できる
  HexaTrain 構成と相性が良い。
- **次に読む**: LOREN（curvature を ZO に入れる具体法。Muon との組み合わせ可能性）。

### B-4. その他の周辺システム

- [MELTing point（EuroSys 2025, arXiv:2403.12844）](https://arxiv.org/abs/2403.12844) —
  Android / iOS / Jetson 上の LLM 推論 benchmark 基盤。headless 実行・energy 計測の
  infrastructure 設計は、HexaTrain の device runner / 評価engine の参照になる（学習は対象外）。
- PocketLLM（PrivateNLP 2024）— derivative-free な on-device fine-tuning。
  OPPO Reno で OPT-1.3B 級。QZO-FF との比較対照。
- LiteRT (TFLite) の on-device training 公式経路は CPU/GPU 中心で、NPU training は
  framework 側で未対応。**「NPU 上の学習」が framework 標準から出ていない**という状況は
  HexaTrain の直接 QNN op 構築路線の妥当性（と独自性）の傍証。

### B-5. mobile NPU inference の計測系（2026-10-25 追加）

- [Is Your NPU Ready for LLMs?（arXiv:2607.05475）](https://arxiv.org/abs/2607.05475) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2607.05475)
  — Cai, Tian, Yang ほか、2026-07（v2 2026-10、**SIGMETRICS 2027 採録**）。
- **要約**: 5 framework × CPU/GPU/NPU backend の横断計測（PowerBench による
  backend 別 energy 帰属）。主要発見は (1) framework 間で NPU 性能が最大10倍違う
  （offload 戦略・演算子の HW 活用・activation 量子化戦略の差）、
  (2) **NPU は compute-bound の prefill で勝ち、memory-bound の decode では
  CPU が全 backend を上回る**（NPU は大きい固定 shape 向けで、decode は小さい
  動的 kernel だから）、(3) scheduling 無駄（thread 設定・NPU sleep latency・
  CPU polling）が energy を食い、CPU 周波数調整だけで最大 39.7% 削減。
  energy-oriented 設定で NPU backend の energy を最大 53.2% 削減。
- [T-MAN（arXiv:2511.11248）](https://arxiv.org/abs/2511.11248) — 低bit LLM の
  NPU end-to-end 推論。Qualcomm QNN を industry SOTA として llm.npu と比較する視点。
- **HexaTrain との関連**: 「NPU decode は弱い」という計測は、
  P1「Incremental KV-cache generation」の設計に直接影響する。
  1-token decode を HTP 固定 graph で回すか、decode だけ CPU/HVX に置くかの
  backend placement 判断（7層共同最適化の Backend Placement）に
  外部計測からの根拠が加わった。HexaTrain の rolling T32 再計算方式は
  「固定 shape を保つ」方向で NPU 特性には合っているが、per-token 固定費が
  大きいという別問題を持つ。
- **次に読む**: decode の kernel size と NPU 効率の関係（incremental KV cache の
  shape 設計の参照）。

### クラスタBの統合所見（2026-10-25時点）

1. **HexaTrain の位置は公開文献上も空白。** 「mobile NPU 上の exact-BP scratch LM training」は
   調査範囲で存在せず、近隣（PockEngine / MeBP / ZO 系）はすべて fine-tuning 前提。
   対外説明では「fine-tuning 系の先行例と区別した scratch training」という表現が正確。
2. **Gradient Strategy の設計空間が広がっている。** exact BP ↔ ZO の二択でなく、
   low-rank ZO / sparse ZO / curvature ZO / activation-guided ZO / sparse backprop が
   2024-2026 に集中している。Phase 11 は MeBP / QZO-FF だけでなく
   LOZO・LOREN 型の選択肢も含めて再設計する価値がある。

---

## C. Optimizer（Muon 系・低 state・理論）

**調査日: 2026-10-25。** Muon 系は 2025 後半〜2026 前半に
「orthogonalization 後の row/column norm 不均衡」を直接的に直す variant が集中している。
これは HexaTrain の row-geometry audit（corr≈0.99 の norm drift）と
Muown lane の前提を**独立系統から裏付ける**動向であり、Phase 4 の比較設計に直接反映すべき情報。

### C-1. MUON+ — One Additional Normalization Step

- [arXiv:2602.21545](https://arxiv.org/abs/2602.21545) / [AlphaXiv](https://www.alphaxiv.org/abs/2602.21545)
  — Zhang ほか、2026-02（v3 2026-05）。
- **要約**: Newton–Schulz polar iteration は特異値 spectrum を 1 へ潰しても
  **row/column 単位の norm 不均衡は解消しない**（むしろ実用 polar step では悪化し得る）と指摘し、
  「post-polar imbalanced update problem」と命名。**orthogonalization の後に
  1 回だけ normalization を挿入する one-line fix** を提案。optimizer state を追加しない。
  GPT / LLaMA 系 60M〜7B、token-to-parameter 比 ~200 までで Muon を一貫して上回る。
- **HexaTrain との関連**: row-geometry audit が観測した「row norm / spectral norm の同期 drift」
  と同一現象への**state 不要**の対処。Phase 4 の比較（Original Muon / Muon+WD / Muown）に
  **Muon+ 型の post-NS row normalization arm を加える**価値が高い。Muown Algorithm 1 は
  `g/r/m_g/v_g` state を追加するのに対し、Muon+ は state ゼロで同種の問題に当たるため、
  「row-norm 制御に state が要るか」を分離できる。
- **次に読む**: 挿入する normalization の正確な形（RMS 系か max-norm 系か）と
  HexaTrain の `keller_original_64560829_fp32` との差分。
  → **【2026-10-25 精読済み】**: normalization は **row / column 単位の L2 norm による除算**
  （`D_col^{-1}` / `D_row^{-1}`、双方逐適用の `col_row` / `row_col` が最良）。
  挿入位置は NS の直後・`η√(m/n)` scale の前。hyperparameter は eps=1e-8 のみ、
  state 追加なし。LLaMA-350M で Muon 14.02 → col_row 13.41。**全 polar 系
  （You / Jordan / PolarExpress）で有効で exact SVD でも 29.17 → 27.90。**
  LR はほぼ不変で、Frobenius norm を match しても gain が残るため
  暗黙の LR rescale ではない。runtime overhead ≤ 2%。HexaTrain の HVX Muon
  （NS5 後）への追加は小変更で、**semantic fan-out row 方向（`row_col` 相当）から
  試すのが自然**。

### C-2. NorMuon — neuron-wise adaptive + orthogonalization

- [arXiv:2510.05491](https://arxiv.org/abs/2510.05491) / [OpenReview](https://openreview.net/forum?id=m1IRWFAMsa)
  — Li, Liu, Liang, Chen, Zhao、2025-10。
- **要約**: Muon update の neuron（row）norm が不均一で一部の neuron が学習を支配する問題へ、
  **per-neuron second moment 統計を持ち、orthogonalization 後に row-wise normalization を適用**。
  1.1B pretraining で Adam 比 21.74% / Muon 比 11.31% 効率改善。state は Muon と同程度
  （row 単位の second moment を追加）。
- **HexaTrain との関連**: Muown（row magnitude を Adam、direction を Muon へ分離）と
  問題設定が同じで解き方が違う系統。**Muown vs NorMuon vs Muon+ の3点比較**は、
  「row 制御をどの位置（weight parameterization / post-NS normalization / per-neuron 2nd moment）で
  入れるか」の分解実験として自然に設計できる。HexaTrain の 8,512 row 規模なら
  CPU reference で全案を安く試せる。
- **次に読む**: per-neuron second moment の update 則（Aux Adam との共存方法）。
  → **【2026-10-25 精読済み】**: Algorithm 1 の正確な形は
  (1) `M = β1 M + (1-β1) G`、(2) `O = NS5(M)`、
  (3) **`v = β2 v + (1-β2) mean_cols(O⊙O)`**（NS 後の update 自身の行 RMS を EMA）、
  (4) `Ô = O / (√V + ε)` で row-normalize、(5) `η̂ = 0.2 η √(mn)/‖Ô‖_F`、
  (6) `W ← W − ηλW − η̂ Ô`（decoupled weight decay 内蔵）。
  **state は行列あたり m scalar のみ**（Muon の ~1/n 増、Adam 比で半減以下）。
  β1=β2=0.95。1.1B/5.4B で Muon の step 削減 10.4%→21.7% / 6.1%→13.9% に倍増。
  step 時間 overhead 2.9%。HexaTrain 向けの要点は、**gradient でなく
  orthogonalized update 自身の行統計を持つ**点で、Muown の
  「weight 側 g/r state」とは state の持ち方が根本的に違う。
  3案比較（Muon+ = stateless L2 norm / NorMuon = update RMS EMA / Muown =
  weight parameterization）が row 制御 lane の自然な設計になる。
  code: `zichongli5/NorMuon`。

### C-3. Preconditioning Benefits of Spectral Orthogonalization in Muon

- [arXiv:2601.13474](https://arxiv.org/abs/2601.13474) — Ma, Huang, Chi, Chen（CMU 系）、2026-01。
- **要要約**: simplified Muon の理論解析。matrix factorization と linear transformer の
  in-context learning で、spectral domain に dynamics が独立な scalar 列へ decouple し、
  **condition number に依存しない iteration complexity** で linear convergence することを証明。
  GD / Adam を理論上上回る。
- **HexaTrain との関連**: 「Muon が効く条件 = ill-conditioning がボトルネックの matrix 最適化」
  という理論的把握。HexaTrain の 758k 規模で Muon が Adam を -0.066 bpb 下回った実測の
  解釈枠になる。また P2「Muon Orthogonalization Accuracy / Polar Express」lane で
  NS 係数を変える意味の理論的文脈になる（完全な NS でなくても preconditioning として成立する）。
- **次に読む**: linear transformer 設定の assumptions（T32/D64 の現行モデルとの距離）。

### C-4. 低 state / stateless optimizer 系

- [SWAN (arXiv:2412.13148)](https://arxiv.org/abs/2412.13148)（ICML 2025）—
  GradNorm + whitening 前処理のみの stateless SGD。Adam と同品質、SGD と同 memory。
- [Gradient Multi-Normalization (arXiv:2502.06742)](https://arxiv.org/abs/2502.06742)
  （Microsoft）— SWAN 動機の stateless optimizer 設計framework。repo 参照済み。
- [RACS / Alice (arXiv:2502.07752)](https://arxiv.org/abs/2502.07752)（ICLR 2026）—
  structured Fisher approximation。Alice は Adam 比 2x 収束を主張。repo の P2 名称对照を確認。
- [AdaMuon (arXiv:2507.11005)](https://arxiv.org/abs/2507.11005)（v3 2025-12）—
  element-wise second moment を Muon へ統合。
  **【2026-10-25 確認済み】**: (a) orthogonalized update への element-wise second moment、
  (b) orthogonalization 前の momentum への sign 変換、(c) Adam と update magnitude を
  揃える RMS-aligned rescaling（既存 LR schedule を再利用可能）の3構成。
  「Adam 比 40% 効率改善」を主張。
- [Mousse (arXiv:2603.09697)](https://arxiv.org/abs/2603.09697)（2026-03）—
  Muon へ structural curvature を明示統合。
- [ARO（arXiv:2602.09006）](https://arxiv.org/abs/2602.09006) / [AlphaXiv](https://www.alphaxiv.org/abs/2602.09006)
  （2026-02、Gong, …, Chao Ma）— **gradient rotation を第一設計原理**とする
  matrix optimizer。orthogonalization を「回転座標系での normed steepest descent」の
  特殊例と位置づけ直し、residual stream の回転対称性を意識した rotation policy。
  8B までで orthogonalization 系比 1.1〜1.15x / AdamW 比 1.3〜1.35x。
- **Sinkhorn 系との接続**: repo 参照の GMN（2502.06742）と同じ番号だが、
  SinkGD として読むと **Sinkhorn algorithm の square-root iterate を
  stateless optimizer として正確に復元する**ことが示されている
  （[ARO 文中の related 紹介](https://arxiv.org/html/2602.09006v1) 経由で確認）。
  これは repo の2課題、V4.1 系 **Sinkhorn-balanced embedding update**（HOLD中:
  scale 非保存が問題）と stateless optimizer 設計が**同じ理論的土台**に乗る
  ことを示す。scale-preserving Sinkhorn の再設計は SinkGD 系の理論から
  借りられる可能性がある。
- **HexaTrain との関連**: 端末 RAM 上、optimizer state は Adam m+v が最大の固定費。
  Muown 導入の動機の一つ（embedding/head の 512 KiB 削減）と同型の削減が
  stateless 系では全行列対象になり得る。ただし whitening / Fisher 系の演算が
  HVX/CPU でどこまで低コストかは未検証なので、Phase 11 の candidate として
  「CPU reference 1-step cost」から測るのが現実的。
- **次に読む**: Alice の low-dimensional subspace 推定（HexaTrain の 64x64 行列群への適合性）。

### C-5. 索引

- **【2026-10-25 直接確認済み】Muown 原論文**:
  [arXiv:2605.10797](https://arxiv.org/abs/2605.10797) / [AlphaXiv](https://www.alphaxiv.org/abs/2605.10797)
  — Lion, Hübler, Li, Orvieto, He（ETH 系）、2026-05。
  spectral norm を **row-magnitude factor と row-coherence factor へ分解**し、
  drift の主因は row-magnitude 側だと特定。row-magnitude は
  **explicit optimizer variable として分解が誘導する ℓ∞ geometry で更新**し、
  direction は Muon 不変（repo の Algorithm 1 読みでは `g/r/m_g/v_g` state）。
  124M〜2.7B で Muon / SOAP / AdamW / Lion を上回り、**near-optimal LR plateau を拡大**、
  weight-decay 感度を低減。理論は non-convex 収束 rate + stochastic noise 係数が
  Muon より小さいことを示す。
  → row 制御3案（Muown = weight parameterization / Muon+ = post-NS L2 norm /
  NorMuon = update RMS EMA）のうち、**Muown だけが LR plateau 拡大と WD 感度低減を
  明示的に主張**しており、Phase 6（LR×batch×optimizer）との接続が最も強い。

- [Awesome-Muon](https://github.com/zhanghualin0/Awesome-Muon)（2026-06 更新）—
  Muon 系の網羅 list。以後の追跡はこれを定期確認するのが最も安い。
- repo 参照済み: Muown 2本（2605.10797 / 2606.23637）、Muon is Scalable（2502.16982）、
  Polar Express（2505.16932）、Can Muon Fine-tune Adam-Pretrained（2605.10468）。
  本調査で新規に重要と判断したのは MUON+ / NorMuon / スペクトル理論の3点。

### クラスタCの統合所見（2026-10-25時点）

1. **「post-NS row norm 不均衡」が独立3系統（HexaTrain audit / Muon+ / NorMuon）で確認された。**
   Phase 4 は Muown 単体ではなく、**row 制御の位置（parameterization vs post-NS normalize vs
   per-neuron moment）を分離する3-4 arm 比較**へ拡張する情報的価値が高い。
   深掘り後の整理として、適応の **granularity 軸**で並べると
   「Muon（適応なし）→ Muon+（row/col L2・stateless）→ NorMuon（row RMS EMA・m scalar）→
   AdaMuon（element-wise second moment・mn scalar）」の spectrum になり、
   それぞれの state 増分と引き換えに得る適応粒度が読み取れる。
   Muown は spectrum の外側（weight 側の parameterization）に位置する。
2. **Muon+ は state 追加ゼロの one-line fix**なので、HexaTrain の HVX Muon 実装への
   変更が最小（NS5 の後に row normalize を 1 演算）であり、最初の arm として最も安い。
3. 理論（C-3）は「NS 係数の妥協が常に致命的とは限らない」ことを示唆し、
   P2 の Polar Express / NS8 比較lane と QNN HTP NS の再試行条件の設計に使える。

---

## D. Accumulator / GEMM kernel 数値（bit-accurate model・epilogue fusion・整数学習）

**調査日: 2026-10-25。** HMX ACC residual feedback（R2）と G1 fusion lane に対応する、
「行列演算器の数値を model で再現する」「accumulator / epilogue 上で要素演算を吸収する」
「整数演算だけで学習する」の3系統。

### D-1. Accurate Models of NVIDIA Tensor Cores

- [arXiv:2512.07004](https://arxiv.org/abs/2512.07004) / [AlphaXiv](https://www.alphaxiv.org/abs/2512.07004)
  — 2025-12。V100/A100/H100 系 tensor core の inner product 挙動を
  **bit-accurate に再現する software model** を構築・公開。
- **要約**: 低精度・混合精度の matrix multiplier の丸め・accumulator 挙動を test vector で
  特徴づけし、数値解析・debug・algorithm 設計に使える正確な emulator を作る研究。
  GPU tensor core の rounding mode / accumulator bit 幅 / 正規化位置が
  仕様公開されないまま実装依存で残る問題への対処。
- **HexaTrain との関連**: **HexaTrain の HMX CPU model / oracle 手法と同型の研究 programme。**
  `hmx_signed_output_oracle.c` 等の「PRM 記載を test vector で device 実挙動と突き合わせ、
  CPU model で再現する」進め方の GPU 側対抗馬であり、(1) 手法の正統性の外部事例、
  (2) model 検証に使う discriminating vector の設計パターン、の両方を参照できる。
  HMX PRM (80-N2040-62) の記述が曖昧な項目（convert の丸め・bias set 挙動）を
  どう特徴づけるかの設計参照にする価値が高い。
- **次に読む**: 論文中の「仕様未記載挙動の特定に使った test vector 設計」の節。

### D-2. CODA — Rewriting Transformer Blocks as GEMM-Epilogue Programs

- [arXiv:2605.19269](https://arxiv.org/abs/2605.19269) / [AlphaXiv](https://www.alphaxiv.org/abs/2605.19269)
  — Guo, Zhang, Menon, Guessous, Thakkar, Kim, Tri Dao（MIT / NVIDIA 系）、2026-05。
- **要約**: norm・activation・residual update・reduction 等の memory-bound 演算を
  **代数的に再 parameter化し、GEMM 出力 tile が on-chip に留まっている間（epilogue で）実行**する
  GPU kernel abstraction。scaling / reduction / pairwise / accumulation の小さな
  epilogue primitive を組合せ、標準 Transformer block の forward/backward の
  非attention計算のほぼ全部を GEMM+epilogue 形へ書き直す。
- **HexaTrain との関連**: R2「retained ACC を使い切ってから store する」という発想の
  GPU 側一般形。HMX では epilogue に相当する convert/stage 命令の自由度が違うが、
  「どの演算なら ACC / 変換段に吸収できるか」の分類表はそのまま HMX 版の設計問いになる。
  また G1 fusion lane（`1-G` / `G(1-G)` 等の intermediate 削減）の
  候補リストを algebraic reparameterization の観点で見直す材料。
- **次に読む**: epilogue primitive の完全リストと、吸収できなかった演算の理由分類。

### D-3. Memory-Efficient Integer-Only Training of Deep Neural Networks

- [arXiv:2610.16298](https://arxiv.org/abs/2610.16298) / [AlphaXiv](https://www.alphaxiv.org/abs/2610.16298)
  — Shalby, Pittorino, Roveri、**2026-10-12（13日前の新着）**。
- **要約**: weights / gradients / activations / optimizer velocity を**すべて INT8 で保存し、
  forward / backward / weight update / optimizer update をすべて整数演算で実行**する学習法。
  符号付き整数 mantissa + 2の冪 exponent の表現で、積・和・定数除算・非線形・weight update の
  5つの整数-only building block に分解。**velocity は group 単位の独立 exponent 追跡**で
  neuron/channel 間の magnitude 差を保持。ResNet/MobileNet/ViT で FP32 と競合する精度、
  学習 memory は約 1/4。
- **HexaTrain との関連**: P2「INT8/INT4/ternary training baseline」「Quantized Adam /
  optimizer-state 低bit化」の最新対照。特に「velocity の per-group exponent 追跡」は
  Muon momentum / Adam v を低bit化する場合の granularity 設計に直接使える考え方。
  HMX が整数 MMA を基本とする構成と相性が良い（HexaTrain は FP16 系だが
  HTA/整数経路の検討時に一次参照）。
- **次に読む**: 非線形（GELU/softmax）の整数近似の誤差評価（HexaTrain の
  QNN sigmoid/softmax 相当演算の低bit化と比較できる）。

### D-4. Block floating point (BFP) 系

- [FAST: DNN Training Under Variable Precision Block Floating Point（HPCA 2022）](https://www.eecs.harvard.edu/htk/static/files/2022-hpca-zhang-mcdanel-kung.pdf)
  — BFP mantissa 幅を学習中に可変にする。FP→BFP 変換の gradient に SR を使う。
- [BiE: Bi-Exponent Block Floating-Point（ICML 2024）](http://www.cse.cuhk.edu.hk/~byu/papers/C229-ICML2024-BiE.pdf)
  — block 内に 2本の exponent を持たせ精度/效率を改善。
- [F-BFQ（arXiv:2510.13401）](https://arxiv.org/abs/2510.13401) — LLM matmul 向けの
  flexible BFP quantization。
- Drumond et al.（EPFL）— end-to-end BFP 学習 accelerator（TETRIS 系譜）。
- **HexaTrain との関連**: MX が shared exponent 1本/k=32 に対し、BFP 系は
  「block 内 exponent 構成の自由度」を探索する系統。HMX の block scale 実装が
  固定のとき、**scale を固定したまま mantissa 側表現を変える**候補の理論的整理に使える。
- **次に読む**: BiE の 2-exponent 割当規則（outlier 1要素だけ特別扱いする類の話か）。

### D-5. 整数学習の古典

- WAGE（arXiv:1803.04483, ICLR 2018）— W/A/G/E すべて整数の学習の出発点。
- [NITI（TPDS 2022, arXiv:2009.13108）](https://arxiv.org/abs/2009.13108) — integer-only 学習の
  実装体系。INT8 parameter / ≤5bit gradient。
- [Is Integer Arithmetic Enough for Deep Learning Training?（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/file/af835bd1b5b689c3f9d075ae5a15bf3e-Paper-Conference.pdf)
  — ResNet-50/ImageNet を integer-only で学習。
- **HexaTrain との関連**: Phase 12 の INT 系 baseline を組むときの再現対照。
  HexaTrain 規模（758k）なら WAGE 級の再現実験も CPU oracle で現実的。

### D-6. 周辺参照

- [Fasi et al., Numerical Behavior of NVIDIA Tensor Cores（2021）](https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/)
  — tensor core 数値挙動の canonical reference。
- Block FMA（Blanchard, Higham, Mary ら）— tensor core matmul の rounding error 解析。
- CUTLASS [EVT（Epilogue Visitor Tree）](https://research.colfax-intl.com/epilogue_visitor_tree) /
  [vLLM fusion passes](https://docs.vllm.ai/en/latest/design/fusions) —
  実運用での epilogue fusion（residual+RMSNorm 等）の実装例。HMX 側で同種の fusion を
  設計するときの「何を fusion するのが普通か」の相場観。

### クラスタDの統合所見（2026-10-25時点）

1. **「accelerator 行列演算器の bit-accurate CPU model 構築」は GPU 側で確立した研究分野**
   （D-1）。HexaTrain の HMX model / oracle / discriminator 手法は孤立した工作ではなく
   この系統の Hexagon 版として位置づけられ、test vector 設計の外部参照が得られた。
2. **CODA（D-2）は R2 の一般形を与える。** 「ACC/tile が on-chip にあるうちに
   何を吸収できるか」を演算分類で考える枠組みは、R2 の signed 出力（convert 2回）だけでなく
   G1 backward fusion・LayerNorm 逆伝搬の HMX 吸収候補の整理に再利用できる。
3. **整数-only 学習（D-3, D-5）は velocity 含む全 state の低bit化に到達した。**
   HexaTrain の P2 低精度化は weight/activation に片寄っているため、
   「optimizer state の group 単位 exponent」を独立候補として足す検討価値がある。

---

## E. 小規模 LM architecture（gate・value residual・MTP・sub-B recipe）

**調査日: 2026-10-25。** repo が既に参照している Gated Attention / MUDDFormer / Value Residual /
MTP の直接 follow-up 系を中心に、G1 lane・P1 Candidate Pool（Value Residual, MTP-lite）へ
情報を足す論文を選んだ。

### E-1. Gated Attention for LLMs — NeurIPS 2025 採録版

- [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) / [AlphaXiv](https://www.alphaxiv.org/abs/2505.06708)
  — Qiu ほか（HKU/UW 系）。repo 参照済みだが **NeurIPS 2025 採録・引用 390+** に到達。
  code: `qiuzh20/gated_attention`。
- **要約（更新点）**: head-specific sigmoid gate（HexaTrain G1 相当）に加えて
  **G2 = value-layer gating**（V 側を gate する変種）も perplexity を改善すると明記。
  また gate 導入による sparsity / attention sink 抑制の観察を報告。
- **HexaTrain との関連**: G1 lane の `PROMOTE_G1_GATE_VARIANTS` 継続探索に対して、
  「query/key 側 gate（G1）vs value 側 gate（G2 相当）」の比較が原文側でも実験されている。
  HexaTrain の reduced-channel gate 計画と並列に、**gate を掛ける位置の分解**
  （output 乘算位置と channel 幅の2軸）として整理するとよい。
- **次に読む**: G2 の正確な挿入位置と backward cost。

### E-2. Value-State Gated Attention（VGA）

- [arXiv:2510.09017](https://arxiv.org/abs/2510.09017) / [AlphaXiv](https://www.alphaxiv.org/abs/2510.09017)
  — Bu, Zhong, Chen, Li、2025-10（v3 2026-01）。
- **要約**: attention sink / value-state drain は「near-zero value への attention 集中が
  no-op 挙動を固定する」相互増強ループから生じると分析。**V 自身から計算した
  learnable gate で attention 出力を modulate** する構成が、input embedding 側から
  gate する既存手法より value と attention-score の credit assignment を
  decouple できることを gradient 解析で示す。sink 抑制・value norm 安定・
  **quantization fidelity 改善**を報告。
- **HexaTrain との関連**: 現行 G1 は `LN1(X) W_g`（input embedding 側）から gate を
  計算しており、VGA の理論は「V 側から gate した方が良い場合がある」ことを直接示唆する。
  G1 parent 配下の gate variant として **V-gate（gate 入力を V へ替える）**は
  graph 変更が小さく理論的動機が明確な候補。quantization fidelity の観察は
  Phase 12 と G1 の相互作用（gate が低bit化を楽にするか）という新規軸。
- **次に読む**: gate の関数形（sigmoid か softplus か）とパラメタ数。
  → **【2026-10-25 精読済み】**: gate は `g = σ(V W_g)`（`W_g ∈ R^{d×h}`、head 数出力）で
  **attention 出力へ乗算** `O' = g ⊙ O`。HexaTrain G1 との差は**gate 入力が `LN1(X)` か
  `V` かだけ**（構造はほぼ同一、param 追加も同等）。理論は
  sink token で `α→1, g→0` のとき gradient が両経路で消えて
  attention weight と gradient magnitude の decoupling が起きる点。
  実験は BERT / OPT-125m（小規模・HexaTrain に近い帯域）で IGA（=Gated Attention）を
  上回り、**8-bit PTQ での劣化も最小**（BERT +0.12）。大規模検証は未実施と明記。
  HexaTrain では G1 parent 配下の variant として
  「gate 入力を V へ替える」だけで既存 graph 資産をほぼ再利用できる。

### E-3. Depth-Attention — Cross-Layer Value Mixing

- [arXiv:2606.05014](https://arxiv.org/abs/2606.05014) / [AlphaXiv](https://www.alphaxiv.org/abs/2606.05014)
  — Zeng, Hao, …, Lin、2026-06。
- **要約**: self-attention の前に、同 token 位置で**過去 layer の key へ query が attend し
  その value を混合**する仕組み。hidden state を跨ぐ従来の cross-layer 系と違い
  **parameter 追加ゼロ・KV cache 以外の persistent state なし**（depth 混合済み value を
  通常の value と置き換えるだけ）。360M〜3B で一貫改善、arithmetic FLOPs +0.01% 未満。
- **HexaTrain との関連**: P1 Candidate Pool の **Value Residual**（`V_l' = V_l + beta_l V_0`）は
  固定係数の足し算だが、Depth-Attention はその「attention による選択的混合」版で、
  parameter も state も増えない。T32/L19 では depth 方向の attention が小さい固定費で
  収まるか実測する価値がある。Cross-Layer Attention Reuse 診断（NOW 5 番）とも同じ
  「layer 間で何を共有するか」の問いで、reuse 診断の延長として natural に試せる。
- **次に読む**: depth-attention の key/value はどの layer から取る設計か（全層 or 直近のみ）。
  → **【2026-10-25 精読済み】**: source は** stride s=L/2 の疎集合**
  （layer ℓ は自身 + {0, s, 2s, ...}）。mixed value は再帰伝播
  `ṽ_ℓ = α_ℓℓ v_ℓ + Σ α_ℓj ṽ_j` で、**通常の V を ṽ で in-place 置換**するのみ
  （KV cache 形状不変・GQA では KV-head 解像度で動作）。overhead は
  FLOPs +0.004%、学習 wall +9〜11%（未 fusion の PyTorch）、推論 +1.18%。
  DenseFormer / Attention Residuals / mHC の +48〜321% と比べると最小級。
  **backward は未解析**（著者明記）で、HexaTrain が手書き backward で最初に
  設計する部分になる。再帰伝播により早期 layer の V へ gradient が戻るため、
  HySparse2 系と同じ cross-layer gradient accumulation が必要。
  L19 なら stride 9 で source ≤ 4 層、QNN graph には per-token の
  微小 attention（要素数 ≤4）として載る。

### E-4. GateSkip — residual gate による layer skip

- [arXiv:2510.13876](https://arxiv.org/abs/2510.13876) — residual stream への gate で
  token 単位に layer 計算を skip する枠組み。
- **HexaTrain との関連**: Attention Skip / Mixture-of-Depths（P2）と同問いへの gate ベース接近。
  HexaTrain では static shape を保てる構成（gate が 0 でも graph は実行）なので、
  「品質を保ったまま HTP 実行を軽くする」のでなく「品質を保ったまま深さを増す」
  MobileLLM 的 weight sharing との組合せで読む方が有益。
- **次に読む**: 省略ではなく表現力としての skip gate の効果分離。

### E-5. AdaMTP — Adaptive horizon の multi-token prediction

- [arXiv:2608.00434](https://arxiv.org/abs/2608.00434) / [AlphaXiv](https://www.alphaxiv.org/abs/2608.00434)
  — Cui, Shi, He ほか、2026-08。
- **要約**: 固定 horizon の MTP は高エントロピーの semantic 境界を跨ぐ予測に
  noisy gradient を生み、backbone 共有ゆえ本体を劣化させる。entropy-based segmentation で
  **token ごとに prediction depth を可変化**し、境界を跨ぐ loss を動的に mask。
  7〜12B backbone で標準 MTP を上回り speculative decoding 速も改善。
- **HexaTrain との関連**: Horizon-Specific MTP-lite（P1）の設計に直接使える考え方。
  HexaTrain の `t+2 adapter` は horizon が固定だが、AdaMTP の知見は
  「t+2 予測の loss を常に掛けるのではなく、entropy 高位の位置だけ/境界を跨がない位置だけ
  掛ける」方向に拡張できる。future-mixture supervision とも相性が良い（混合重みを
  entropy で変調）。
- **次に読む**: entropy threshold の決め方（base model 自身の entropy を使う再帰性）。

### E-6. MobileLLM-R1 — sub-billion reasoning の open recipe

- [arXiv:2509.24945](https://arxiv.org/abs/2509.24945) / [AlphaXiv](https://www.alphaxiv.org/abs/2509.24945)
  — Zhao, Chang, Liu, …, Chandra（Meta）、**ICLR 2026**。code/data/recipe 公開。
- **要約**: sub-billion（950M 中心）でも data curation + resampling により
  reasoning が成立することを示す。2T token 級の高品質 pool から 4.2T を resample して
  pretrain し標準 post-training。Qwen3-0.6B と複数 benchmark で同等以上。
- **HexaTrain との関連**: HexaTrain（758k）とは2桁違うが、
  「小規模では data 質・配分が token 量より支配する」ことの最新傍証。
  data pipeline identity を checkpoint に結ぶ現行運用（Reproducible Data Pipeline）の
  方向性と整合。将来的に corpus を増やす場合の resampling 設計の参照。
- **次に読む**: data mixing ratio の公開表（Phase 13 の stable multi-source mix 設計の参照）。
  → **【2026-10-25 精読済み】**: data pipeline の核心は
  (a) **leave-one-out ablation**（Code/Math/Knowledge の capability probing set 上の
  NLL delta）、(b) **influence score**（AutoMixer 系の Hessian-free 近似で
  self / cross influence を算出し dataset 単位の sampling weight へ集約）、
  (c) mid-training での**影響度正の sample だけ残して再 mix する
  iterative data-model co-evolution**。140M/360M/950M の3 size、QK-norm 採用、
  **input/output embedding の weight sharing（tied）で parameter efficiency**。
  SFT 蒸留 data が小規模 model の直接 RL より勝つという結果も。
  HexaTrain との対比: tied embedding を採る sub-B 系（MobileLLM-R1）と
  untied が勝つ speedrun 系（J-1）が対立しており、**tied/untied は規模・data 依存**
  の判断。influence-score 系の data 評価は Phase 13 の候補手法として参照価値が高い
  （ただし HexaTrain の corpus 規模では leave-one-out の費用対効果は要検討）。

### E-7. survey・その他

- [SLM survey（arXiv:2501.05465, 2026-05 更新版）](https://arxiv.org/abs/2501.05465) —
  約160論文の SLM 系 survey。architecture / data / efficiency の地図として保持。
- [Matryoshka Language Model Suites（arXiv:2608.09703）](https://arxiv.org/abs/2608.09703) —
  共有学習で model suite を作る方向。MatFormer / elastic model 系（P1〜P2）の近縁。
- **XSA（arXiv:2603.09078, repo 参照済み）の追跡状況（2026-10-25）**:
  Apple（Shuangfei Zhai）による二次資料（EmergentMind / 著者 X 投稿）の確認では、
  XSA の gain は** sequence length とともに増大**するとされる。
  T32 の HexaTrain では gain が小さい可能性が高く、repo の
  「長文脈論文の gain を外挿しない」原則と整合する方向。
  明確な follow-up 論文はまだ無し。

### クラスタEの統合所見（2026-10-25時点）

1. **G1 lane の次の variant は「gate の入力側」が有望。** VGA（E-2）の gradient 解析は
   input embedding 側 gate より value 側 gate の優位を理論づけ、原著（E-1）の G2 も
   value 側 gate の改善を報告している。HexaTrain は gate 位置×channel 幅の2軸で
   variant を整理し直すと探索が閉じやすい。
2. **Value Residual lane に「parameter ゼロ・KV 中立」の選択肢（E-3）が加わった。**
   固定係数 `beta_l V_0` より表現力が高く、 inference cost 増もない。
   Cross-Layer Reuse 診断と同じ instrumentation で評価できる。
3. **MTP-lite は horizon 可変化（E-5）が次の設計軸。** adapter 構造はそのままに、
   loss 掛ける位置の entropy-based 選択だけを先に試せる。

---

## F. Linear / recurrent attention（GDN 系の理論統一と実装基盤）

**調査日: 2026-10-25。** repo は Gated DeltaNet / KDA / SimpleGDN migration を
Phase 14 に持つ。新規に拾えたのは (1) TTT 系と linear attention の理論的統一、
(2) chunkwise 並列実装の共通基盤が library 化している点。edge 整数 attention 系の
収穫はクラスタ D/H へ回した。

### F-1. Test-Time Training with KV Binding Is Secretly Linear Attention

- [arXiv:2602.21204](https://arxiv.org/abs/2602.21204) / [AlphaXiv](https://www.alphaxiv.org/abs/2602.21204)
  — Liu, Elflein, Litany, Li, Gojcic（NVIDIA Research ほか）、**ICML 2026**。
- **要約**: TTT（test-time training）+ KV binding 系の広い class が
  **learned linear attention 演算と数学的に等価**であることを系統的な reduction で証明。
  「TTT は test-time の暗記ではなく、表現力の高い linear attention である」と再解釈し、
  逐次 test-time loop を回避する fully parallel 形式と principled な構造簡略化を導出。
- **HexaTrain との関連**: Phase 14 の Hybrid Sequence Operator Search に対して、
  「GDN / KDA / TTT 系は同じ数理的土台の上の variant」という整理を与える。
  HexaTrain が独自に「HTP専用 Delta Attention」（P3）を設計するとき、
  この等価性を経由して「既存 attention graph の部品（MatMul / elementwise / gate）を
  そのまま再利用する形」で設計できる可能性がある。SimpleGDN weight-preserving
  migration の数式変形を考える際の理論的整理枠。
- **次に読む**: reduction の仮定（chunk 境界・gate 形式）と HexaTrain SimpleGDN 設計の対応。

### F-2. chunkwise parallel 系の実装基盤

- [flash-linear-attention（GitHub, fla-org）](https://github.com/fla-org/flash-linear-attention) —
  GLA / DeltaNet / GDN 等の Triton 実装を集めた共通 library。chunkwise algorithm の
  参照実装として、QNN graph へ落とすときの「演算単位の切り出し」に使える。
- GLA（arXiv:2312.06635）/ GDN（arXiv:2412.06464, repo 参照済み）の chunkwise 形式は
  「sequence を chunk C に割り、intra-chunk は MatMul、inter-chunk state は
  小さい再帰」に分解する。**この分解は HTP 向き**（MatMul 主体、state が小さい）で、
  HexaTrain の SimpleGDN 実装計画の前提と整合する。
- **次に読む**: GDN の Triton kernel 1個分の演算粒度（QNN node 数の見積り材料）。

### F-3. 索引（未検証・要確認）

- [Bridging KV-Cache Quantization and Linear Attention（arXiv:2610.11214）](https://arxiv.org/abs/2610.11214)
  — 2026-10。KV cache 量子化（entry 圧縮・全保持）と linear attention
  （multi-KV を固定 size state へ集約）を RAM-Net の soft address assignment で
  統一する理論。**pretrained weight からの移行経路**（soft-quantized 中間段階）を持ち、
  0.3B〜7B の 9 model で 500M token の budget で teacher gain の 87.1% を回復。
  要確認だが、HexaTrain の low-disruption migration protocol と同じ
  「既存 weight を壊さず移行」の設計思想を持つ。
- [Native Hybrid Attention（arXiv:2510.07019）](https://arxiv.org/abs/2510.07019) —
  検索結果では recall-intensive / commonsense で transformer 系 hybrid を上回るとの由。
  要 abstract 確認。
- [RNN as Linear Transformer（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_RNN_as_Linear_Transformer_A_Closer_Investigation_into_Representational_Potentials_CVPR_2026_paper.pdf)
  — 線形 attention vs Mamba 系の表現力比較（vision 中心）。P14 の sorted 順序決めの補助資料。
- [Awesome-State-Space-Models](https://github.com/radarFudan/Awesome-state-space-models) —
  追跡用 index。

### F-4. Native Top-k 学習の予備研究・UNIQUE（2026-10-25 追加）

- [A Preliminary Study on Native Top-k Sparse Attention（arXiv:2512.03494）](https://arxiv.org/abs/2512.03494)
  / [AlphaXiv](https://www.alphaxiv.org/abs/2512.03494)
  — Xiu, Tang, Rong ほか（Meituan 系）、2025-12。
- **要約**: (1) 推論時の exact Top-k context は full attention と同等以上
  （HELMET / LongBench v2）、(2) **学習時から Top-k attention でやる
  （train/inference 一致）と性能がさらに伸びる**、(3) 近似 Top-k の
  fidelity と下流性能は正相関、(4) Top-k は低-entropy 状態と相性が良い、
  という entropy 解析。DeepSeek-V3.2-Exp の Lightning Indexer 精度の統計評価も。
- [UNIQUE（arXiv:2605.27740）](https://arxiv.org/abs/2605.27740) —
  training-free deploy と sparsity-aware pretraining の両対応を1つの
  top-k framework で扱う後続研究（2026-05）。
- **HexaTrain との関連**: repo の sparse 実装順序
  （oracle sparse simulator → microbenchmark → 最小 block → Full teacher A/B）に対し、
  「近似 fidelity ↔ 品質の正相関」と「native 学習の增益」は
  診断の評価軸としてそのまま使える。また低-entropy 状態との相性は
  H-2（byte LM）・E-5（AdaMTP）の「entropy が構造境界に集中」観察と
  同じ entropy 視点の系譜で、sparse 選択・MTP horizon・curriculum が
  同じ entropy 地図を共有するという統一的読みが可能。
- **次に読む**: k の選び方と model 規模（HexaTrain の T32 診断に近い条件があるか）。

### クラスタFの統合所見（2026-10-25時点）

1. **linear attention 系は理論統一（F-1）と実装共通化（F-2）が同時に進んだ。**
   HexaTrain が SimpleGDN を作るとき、「独自改造」より「chunkwise 分解を QNN graph に
   忠実に落とす」第一歩の妥当性が上がった。
2. TTT 等価性は「HTP での recurrent operator = attention 部品の再構成」という
   実装戦略を正当化する。Phase 14 の graph 設計言語として使える。

---

## G. Training memory planner（KEEP / 低bit / RECOMPUTE の自動選択）

**調査日: 2026-10-25。** repo の P1「HTP Memory Planner」
（tensor ごとに KEEP FP / KEEP low-bit / SPILL / RECOMPUTE を実機 cost 最小で選ぶ）に
直接対応する外部研究が見つかった。

### G-1. Adacc — compression と recomputation の統一的適応選択

- [arXiv:2508.00806](https://arxiv.org/abs/2508.00806) / [AlphaXiv](https://www.alphaxiv.org/abs/2508.00806)
  — Chen, Deng, Li, He, Zhu, Zheng, Wang, Huai, Guo（SJTU / 华为 系）、2025-08。
- **要約**: activation の再計算・保持・圧縮を**tensor 単位で動的に選択**する適応 framework。
  (1) LLM activation の outlier を扱う layer 特化 compression algorithm、
  (2) layer 横断で memory 戦略を大域最適化する MILP-based scheduling、
  (3) 学習中の data 分布変化へ追従する policy evolution。
  SOTA 比で throughput 1.01〜1.37x、精度は baseline 相当。
- **HexaTrain との関連**: **repo の Training Memory Planner 設計の外部対応物。**
  「recompute vs keep vs compress を tensor 特性 + hardware 制約の関数として選ぶ」
  という問題定義が同一。HexaTrain 版の差分は、hardware 制約の入力が
  V81 実機 telemetry（HTP kernel 時間 / APP tensor traffic / RPC 境界）である点と、
  候補に spill (DDR/UFS) と backend 移動（HVX/CPU で再計算）が入る点。
  MILP による大域 scheduling は Phase 9 の heuristic からの出発点として参照できる。
- **次に読む**: tensor 特徴量（どの統計で recompute/compress を予測するか）の設計。

### G-2. CompAct — 低ランク射影による activation 圧縮

- [NAACL 2025](https://aclanthology.org/2025.naacl-long.71/) — Shamshoum, Hodos, Sieradzki, Schuster（Technion）、2025-04。
- **要約**: backward 用に保存する activation を**random projection による低ランク圧縮**で保存
  （量子化ではない）。projection は random なので追加 memory なし。
  pretraining で peak memory 25〜30%、fine-tuning で 50% 削減。
- **HexaTrain との関連**: 「Quantized Activation Checkpointing（INT8/INT4 保存）」と
  「低ランク保存」は同じ KEEP slot の別実装。D64 と小さい隠れ次元では
  低ランク圧縮の効率が落ちる可能性が高いが、T32 window × B8 の小さい tensor 群では
  「保存コストより projection MatMul コスト」が逆転するか実測する価値がある。
  QNN graph に random projection を足すのは node 数小で済む。
- **次に読む**: projection の rank 選択規則。

### G-3. 既知系統の整理（repo 参照済み）

- **COAT**（NVIDIA）: activation / optimizer state の細粒度 group 量子化。
- **MODeL**（Meta）: tensor lifetime / placement / spill / rematerialization /
  reduced precision を一つの scheduling 問題として扱う。Adacc と同じ「統合」路線の先駆。
- **FastSA**（PFN）: 計算 graph 上の保存/再計算配置の自動探索。
- **MeBP**（Apple, cluster B-2）: layer 分割と memory hierarchy への逃がし。
- **HexaTrain との関連**: Memory Planner の設計要素（lifetime 解析 / granularity /
  大域 scheduling / 適応更新）はすべて外部に先行例がある。HexaTrain 独自の価値は
  「QNN HTP / HVX / CPU の3 backend と FastRPC 境界を含めた cost model」と
  「端末実測での検証」にある。

### クラスタGの統合所見（2026-10-25時点）

1. **Memory Planner は「新規概念」ではなく「確立した問題領域の HTP 版」**として設計できる。
   Adacc の MILP scheduling と policy evolution は Phase 9 の実装柱として直接参照する。
2. HexaTrain の差別化要素は backend 異質性（HTP/HVX/CPU + RPC 境界）と spill 先の
   memory 階層。外部論文の cost model にこの2つを足すのが設計の本体。

---

## H. Text representation・softmax 数値・整数推論（評価/推論側）

**調査日: 2026-10-25。** HexaTrain の bpb（bits/original UTF-8 byte）指標と
直接同じ土俵に立つ tokenizer スケーリング研究、QNN softmax 精度に直結する
quantized softmax 研究、整数-only 推論系。

### H-1. Compute Optimal Tokenization

- [arXiv:2605.01188](https://arxiv.org/abs/2605.01188) / [AlphaXiv](https://www.alphaxiv.org/abs/2605.01188)
  — Limisiewicz, Pagnoni, Iyer, Lewis, Mehta, Liu, Li, Ghosh, Zettlemoyer（Meta）、2026-05。
  code: `facebookresearch/compute-optimal-tokenization`。OpenReview 2026-09。
- **要約**: BLT の latent tokenization で compression rate（平均 bytes/token）を連続制御し、
  **50M〜7B・988 model**を学習して tokenizer の compute-optimality を測定。
  (1) compute-optimal 設定では parameter は** byte 測りの data size に対して**scale する
  （token 測りの Chinchill 流儀と矛盾）、(2) **最適 compression rate は compute 増に対して
  低下する**（大規模ほど細かい token が最適）、(3) 結果は subword tokenizer と
  非英語にも一般化。
- **HexaTrain との関連**: HexaTrain は評価を bits/original-byte で行っており、
  この研究と同じ byte 基準の尺度に立っている。**HexaTrain の compute budget（端末）は
  この実験の最小域（50M）よりさらに小さい**ため、理論外挿上は「より高い compression
  （大きい bytes/token = より少数の dense token）側」が最適になりうる。
  Phase 10 の Text Representation Search（V512/V1024/V2048, GBST, BLT）で
  「compression rate を compute budget の関数として扱う」設計根拠になる。
  V1024 byte-BPE 固定ではなく、日本語 corpus での bytes/token 分布と
  compression rate の掃引を作る意義が体系的に裏付けられた。
- **次に読む**: 最適 rate の回帰式（50M 近傍の外挿に使えるか）。
  → **【2026-10-25 精読済み】**: 最適 compression は `T* = T₀ / C^δ`
  （subword: T₀=1577, δ=0.129 / latent: T₀=18.2, δ=0.035）。検証範囲は
  C ≈ 5e18〜2e21 FLOPs で、その下端ですでに T*≈4〜4.5（BPE 並み）。
  **HexaTrain の compute（〜1e13 FLOPs 級）は fit 範囲より5桁以上小さい**ため
  定量的外挿は不可能だが、方向としては「より大きい bytes/token（高圧縮）側」が
  最適になりうる。多言語では parity と T* が相関し（English 3.71 / Russian 5.67 /
  Hindi 8.09）、**既存 tokenizer は高資源言語を過圧縮する**との指摘は日本語 V1024
  byte-BPE の位置づけ検討に直接関係する。

### H-2. Byte Language Models — scaling・emergent abstraction

- [arXiv:2610.05978](https://arxiv.org/abs/2610.05978) / [AlphaXiv](https://www.alphaxiv.org/abs/2610.05978)
  — Wang, Luo, Zhang, Wu、2026-10-05（3週間前）。
- **要約**: byte 直接入力 Transformer について、token-superposition training + hash embedding
  で subword Transformer を規模とともに逆転できること、byte model が
  外部 tokenizer 相当の局所抽象を創発すること（中間層の最大25%をその局所表現に
  制限しても性能維持）、生成の不確実性が構造境界近傍に集中すること
  （speculative decoding で subword 比 3.4x の受理）を報告。
- **HexaTrain との関連**: Phase 10 の raw byte 候補（V256）に必要な技術要素
  （superposition training / hash embedding）の最新処方。「byte は固定費が大きい」という
  HexaTrain の暫定見立てに対して、「学習法を変えれば scale で逆転する」反例。
  また「entropy が構造境界に集中」は AdaMTP（E-5）と独立に同じ観察をしており、
  HexaTrain の difficulty-aware sampling（P2）設計の裏付けが2系統揃った。
- **次に読む**: token-superposition training の学習則（QNN graph で実装可能な形か）。

### H-3. Pretraining Transformers with Quantized Softmax in Attention

- [arXiv:2609.33591](https://arxiv.org/abs/2609.33591) / [AlphaXiv](https://www.alphaxiv.org/abs/2609.33591)
  — Zhu, Hu, Kozlowski（Hugging Face 系）、2026-09（v2 2026-10-16）。
- **要要約**: attention softmax の exp を **K+1 点の grid（K-interval attention）で近似**し、
  scratch pretraining から学習。行ごとの calibration（min-max vs 固定窓）、
  補間 vs hard rounding、straight-through surrogate を正規化の前後どこへ置くかを系統掃引し、
  backward 則（calibration 微分含む）を導出。124M / 2.5B token で
  **K=16・pre-normalization surrogate なら exact softmax 比 +0.004 nats**。
  一方 surrogate 位置と calibration の組み合わせを誤ると大きく劣化
  （detached row extrema は forward 不変でも遅れて loss 悪化）。
- **HexaTrain との関連**: **QNN HTP softmax の演算コスト・精度問題（qnn-htp-softmax 系）へ
  最も直結する文献。** Attention MatMul は低bit化が進む一方 softmax が高精度に残る、
  という状況認識は HexaTrain の「Softmax / LM head は高precision残置」という
  想定と同一。K-interval 化は QNN graph 上では要素演算（grid 参照 + 補間）で構成でき、
  HMX/HTP 側で softmax を近似する選択肢の1つになる。
  Attention Normalizer Search（sigmoid / ReLU 系）と同じ「softmax を演算器に合わせて
  変形する」設計空間に属する。
- **次に読む**: grid 構成（等間隔か log 間隔か）と K の感度曲線。
  → **【2026-10-25 精読済み】**: grid は row 単位 calibration で (a) **MinMax**（行 min-max 全域を
  K+1 等分）か (b) **FWM: 行 max から τ=6 nats の固定窓**（窓外は 0）。再構成は
  LERP 補間か Nearest 丸め。STE は正規化前（Weight-STE）か正規化後（Prob-STE）。
  **MinMax + Weight-STE + K=4 は +0.89 nats と大きく劣化**するが、
  **FWM と surrogate 位置のどちらかを直すだけで大半が解消**し、
  **FWM + Weight-STE + K=16 で +0.004 nats**。劣化は grid 間隔 h と
  相関（Spearman +0.95）で、calibration gradient（行 max 依存経路）を切ると
  forward 不変でも 25〜30M token 後に発散する。hardware motivation は
  **exp の LUT 化（2〜3 bit table read）**で、HexaTrain の QNN softmax
  （softmax を高精度演算に残す現状）への直接的な置換候補。

### H-4. IntAttention — fully integer attention pipeline

- [arXiv:2511.21513](https://arxiv.org/abs/2511.21513) — Zhong ほか（SUSTech）、
  **MLSys 2026**。code: `WanliZhong/IntAttention`。
- **要約**: attention の全 pipeline を整数演算化（integer softmax 含む）。
  edge 推論で約 3.7x の効率改善を報告。softmax 近似の頑健性 ablation あり。
- **HexaTrain との関連**: H-3 が「学習時に softmax を量子化して慣らす」、
  IntAttention は「推論 pipeline 全体を整数化する」。HexaTrain は学習側なので H-3 が主だが、
  学習済み checkpoint を HMX 整数経路で推論するときの対照。
- **次に読む**: integer softmax の丸め設計（H-3 の K-interval との比較）。

### H-5. その他の索引

- [P3-LLM（arXiv:2511.06838）](https://arxiv.org/abs/2511.06838) — NPU-PIM 統合 accelerator、
  32-bit fixed-point。HTP ではなく PIM 前提だが低bit KV / weight-act quant の SOTA 対照。
- [Faster Superword Tokenization（arXiv:2604.05192）](https://arxiv.org/abs/2604.05192) —
  BPE tokenization の高速化（4.7 CPU日→600 CPU秒）。アプリ内で tokenizer/cache を
  端末生成する「アプリ単体化」課題の工敷削減に使える可能性。
- [Bridging KV-Cache Quantization and Linear Attention（arXiv:2610.11214）](https://arxiv.org/abs/2610.11214)
  — KV 量子化と linear attention の統一理論。P3 の KV 系と Phase 14 の接続部。
  要確認。
- MELTing point（EuroSys 2025）はクラスタ B-4 参照。評価engine（NOW 1番）への
  直接の外部文献は薄いが、serving 系の設計が最も近い実務参照:
  [Multi-Bin Batching（ACM 2026-08）](https://dl.acm.org/doi/10.1145/3843225)
  （出力長での group 化 batching）、[Sarathi-Serve（OSDI 2024）](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  （chunked prefill による throughput-latency 制御）、vLLM の continuous batching /
  PagedAttention。HexaTrain の評価は同型 window（T32 固定）なので可変長対応が不要で、
  「window 単位の batch 化 + graph 再利用」に絞れば serving 系より単純な設計で済む。

### H-6. softmax 代替・量子化の系譜（2026-10-25 追加）

H-3（学習時慣らし）・H-4（整数 pipeline）の周辺系統を整理する。

- [Taming the Exponential: HCCS（arXiv:2604.02292）](https://arxiv.org/abs/2604.02292) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2604.02292)
  — Danopoulos, Lupi, Kagan, Pierini（CERN 系）、2026-04。
- **要約**: softmax の exp を**max-centered logit への clipped linear 写像**で置換
  （有界・単調で順序保存、非負分布が安定）。per-head calibration parameter を
  offline で最適化。AMD Versal AI Engine の int8 MAC 向けに実装し、
  bfloat16/LUT 依存の reference 実装を大きく上回る速度。
  **quantization-aware retraining が必要**と明記。
- [EXAQ（OpenReview）](https://openreview.net/pdf?id=AuJ6gDjcZK) — softmax 入力の
  最適 clipping 値を解析的に決め sub-4-bit 量子化。exp 計算と累積 phase を分離。
- **IndexSoftmax**: H-4 IntAttention（2511.21513）の softmax 成分の名称。
  LUT ベース・整数実行互換の softmax 近似 + 整数 normalization。
- **HexaTrain との関連**: HexaTrain の Attention Normalizer Search 候補
  （Softmax / Sigmoid / ReLU 系）に加え、
  「max-centered clipped linear（HCCS）」「K-interval grid（H-3）」が加わり、
  softmax 代替の選択肢は計5系統になった。いずれも**学習時（または
  quantization-aware retraining）での慣らしが前提**で、
  学習 graph を自前で持つ HexaTrain では導入障壁が相対的に低い。
  T32 / H2 では per-head calibration の固定費が問題になるため、
  head 共有 calibration からの簡略化が HexaTrain 向けの最初の変形になる。
- **次に読む**: HCCS の clipped linear の clip 点決定規則。

---

1. **byte 基準の tokenizer スケーリング則が確立しつつある（H-1, H-2）。**
   HexaTrain の bpb 指標設計はこの潮流と整合しており、Phase 10 の compression rate
   掃引に理論的 Anchorage ができた。小 compute 側の外挿は HexaTrain 自身が測る価値がある。
2. **softmax の低bit化は「学習時に慣らす」流れが出てきた（H-3）。**
   QNN softmax 精度問題への新しい解決方向であり、Attention Normalizer Search と
   統合した候補表に加えるべき。
3. 評価engine 高速化（NOW 1番）に直結する学術文献は薄い。serving 系の実務設計
   （batching / graph reuse）の借用が現実的。

---

### クラスタHの統合所見（2026-10-25時点、H-6 追記後更新）

1. **byte 基準の tokenizer スケーリング則が確立しつつある（H-1, H-2）。**
   HexaTrain の bpb 指標設計はこの潮流と整合しており、Phase 10 の compression rate
   掃引に理論的 Anchorage ができた。小 compute 側の外挿は HexaTrain 自身が測る価値がある。
2. **softmax の低bit化・代替は「学習時に慣らす」が前提の流れ（H-3, H-6）。**
   選択肢は Softmax / Sigmoid / ReLU 系（repo 既存）に
   K-interval grid と max-centered clipped linear が加わり計5系統。
   QNN softmax 精度問題への解決方向であり、Attention Normalizer Search と
   統合した候補表に加えるべき。
3. 評価engine 高速化（NOW 1番）に直結する学術文献は薄い。serving 系の実務設計
   （batching / graph reuse）の借用が現実的。

---

## I. Data curriculum（difficulty-aware sampling の外部証拠）

**調査日: 2026-10-25（追加クラスタ）。** P2「Difficulty-Aware Sampling /
Loss Curriculum」への文献裏付けを探した。

### I-1. Beyond Random Sampling: Efficient Pretraining via Curriculum Learning

- [arXiv:2506.11300](https://arxiv.org/abs/2506.11300) / [AlphaXiv](https://www.alphaxiv.org/abs/2506.11300)
  — Zhang, Mohamed, Abdine, Shang, Vazirgiannis、2025-06（v2 2026-01）。
- **要約**: LLM pretraining における curriculum learning の最初の体系的調査とされる。
  **200+ model / 最大 100B token**で、(a) easy-to-hard の vanilla curriculum、
  (b) pacing-based sampling、(c) interleaved curricula を random sampling と比較。
  difficulty 指標は言語学的・情報理論的6種で、**compression ratio / lexical diversity
  (MTLD) / readability (Flesch) が最良**。結果は早期〜中盤で 18〜45% step 削減、
  **warmup-then-random の形だと最後まで +3.5% 持続**。data ordering は
  既存の data selection 手法と orthogonal。
- **HexaTrain との関連**: 現行 P2 設計は「直近 model の loss EMA で easy/medium/hard を
  分ける」案だが、本論文の効く指標は**モデル非依存の言語統計**（compression ratio 含む）
  だった。HexaTrain では loss EMA と bytes/token（bpb 測定で既に持っている量）の
  両方を難易度指標候補として並べる根拠になる。「uniform 成分を残す」設計（20/50/30）は
  warmup-then-random と同じ方向で文献整合的。
- **次に読む**: continual training scenario の結果（resume が多い HexaTrain 運用に近い）。

---

## J. 小規模学習 recipe の系譜（modded-nanogpt speedrun）

**調査日: 2026-10-25（追加クラスタ）。** HexaTrain の Original Muon optimizer identity
（`keller_original_64560829_fp32`）の系譜元である modded-nanogpt speedrun の
現状と技術内訳を確認した。

### J-1. modded-nanogpt speedrun（124M / FineWeb / 8xH100）

- [KellerJordan/modded-nanogpt](https://github.com/kellerJordan/modded-nanogpt) —
  2026-10 時点の record は **90秒**（1年前は 8.2分）。
  [Speedrun Optimization Writeup（discussion #211）](https://github.com/KellerJordan/modded-nanogpt/discussions/211)
  / [LessWrong 記事](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/) /
  [Prime Intellect auto-speedrun](https://www.primeintellect.ai/auto-nanogpt)（AI agent が
  speedrun 改善を競う benchmark）。
- **要約（writeup の伸び内訳）**: Muon 77.5% > FlexAttention 54.5%（後に FA3 へ） >
  初期値・LR tuning 46.1% > architecture 現代化（RoPE / QK-norm / zero-init projection /
  ReLU²）45.2% > **U-Net skip + value embeddings 36.5%** > logit softcap 15 + offset
  logits 20.9% > untied embeddings 11.1% > bf16 activation + fp8 head 8.0%。
  speedup は規模依存という注意が共同作業者から明示されている。
- **HexaTrain との関連**:
  - **U-Net skip + value embeddings が36%級の巨大 gain**という記録は、P1 pool の
    Value Residual / Depth-Attention（E-3）を優先する傍証（124M と HexaTrain の中間規模）。
  - **logit softcap 15 は HexaTrain が未検討の低コスト候補**。CE graph への
    変更が小さく（logits の clamp + offset）、学習安定化（attention logit 系 telemetry）
    との関連で short A/B に向く。
  - untied > tied という speedrun 側の観察は、P1「tied embedding」候補の
    期待値を相対的に下げる方向（repo も Modded-NanoGPT の untied 事例は注記済み）。
  - ReLU² が効いたという speedrun 記録と HexaTrain の ReLU² 否定（matched A/B）の
    差は「規模・data・他技術との相互作用」の問題であり、speedrun 記録の
    直接移植可能性が低いことの実例。
- **次に読む**: momentum warmup と分散 overhead 削減の具体的形
  （HVX Muon の W8 構成への流用可否）。

### J-2. logit softcap の系譜と QK-norm との関係

- [Gemma 2（arXiv:2408.00118）](https://arxiv.org/abs/2408.00118) —
  `tanh(x/cap)*cap`（attention cap=30 / 最終層 cap=50）で logit 成長を抑え
  学習安定化・蒸留可能性を確保した原典。
- 後継（Gemma 3 / Qwen3 / OLMo 2）は softcap をやめて **QK-norm へ置換**した流れ
  （hardware-friendly な理由）。[QK-norm + soft-capping の段階的併用を提案する研究
  （OpenReview, Rybakov et al.）](https://openreview.net/pdf?id=RL6R5ryuL5) もある。
  Karpathy nanochat でも「softcap を外して QK-norm だけにする」議論
  （[issue #362](https://github.com/karpathy/nanochat/issues/362)）が行われている。
- **HexaTrain との関連**: repo の学習安定化候補（QK Norm / Z-Loss / QK-Clip）は
  この系譜の上にある。HexaTrain は QK-norm 未導入なので
  「softcap 単体 → QK-norm 単体 → 併用」の順序が文献上自然。
  logit softcap は CE graph の logits への小変更（tanh + scale）で、
  QNN graph 追加 node が小さく短 A/B 向き。
- **次に読む**: Rybakov et al. の「段階が違う」主張の内容（どの phase に
  それぞれ効くか）。
  → **【2026-10-25 確認済み】**: [Rybakov et al.（arXiv:2410.16682）](https://arxiv.org/abs/2410.16682)
  （NVIDIA、2024-10）は QK-norm × soft-capping × z-loss の組合せを体系比較し、
  **QK-norm + (softcap / z-loss) で LR を 1.5x まで上げても diverge しない**ことを示した。
  なお QK FC norm（全結合 variant）は plain QK-norm を上回らなかった。

### J-3. Output Embedding Centering（OEC）— z-loss の代替（2026-10-25 追加）

- [arXiv:2601.02031](https://arxiv.org/abs/2601.02031) / [AlphaXiv](https://www.alphaxiv.org/abs/2601.02031)
  — Stollenwerk, Lokrantz, Hertzberg、2026-01（v3 2026-09）。
- **要約**: 学習終盤の **output logit divergence** の原因を output embedding の
  anisotropy と特定。z-loss / softcap は対症療法と位置づけ、
  (a) **μ-centering**: output embedding を中心化する決定論的操作、
  (b) **μ-loss**: 中心化を促す正則化項、の2 variant を提案。
  安定性で z-loss を上回り、softcap と同等。**weight tying の有無両方で動作**し、
  μ-loss は z-loss より hyperparameter 感度が低い。
- **HexaTrain との関連**: repo の Z-Loss diagnostic（P1）と同問題への
  より安価な対照候補。`output_projection`（65,536 params）の行平均を引く
  μ-centering は QNN graph への追加 node が小さく、
  **logit 系 telemetry（Z-Loss / attention logit max）と同じ計測枠で A/B できる**。
  weight decay との相互作用の実験が v3 で追加されており、
  Muon+WD lane との併存確認にも使える。
- **次に読む**: μ-centering の実装詳細（logits から引くか embedding を前処理するか）と
  実験規模。

---

# 全体統合（2026-10-25、クラスタ A〜J 完了時点）

## 1. repo の研究方向を外部文献が独立に支持しているもの

| repo の現行計画 | 外部からの裏付け | 重要度 |
| --- | --- | --- |
| Muown / row-geometry 制御（Phase 4） | MUON+（C-1）と NorMuon（C-2）が同一現象を指摘・修正。state ゼロの Muon+ 型が最安 arm | **高** |
| HMX CPU model / oracle 手法（R2 周辺） | bit-accurate tensor core model 構築が GPU 側で確立分野（D-1） | 高 |
| Text Representation Search・bpb 指標（Phase 10） | byte 基準 compute-optimal tokenization が 988 model 規模で確立（H-1, H-2） | **高** |
| Training Memory Planner（Phase 9） | Adacc / MODeL が同一問題の外部先行例（G-1, G-3） | 高 |
| Softmax 高精度残置の想定 | softmax 自体の低bit化は「学習時に慣らす」必要があるという結果（H-3）。Kellogg の LN weight 除外（A-2）とも整合 | 中 |
| Cross-Layer Reuse 診断 → Value Residual（NOW 5 / P1 pool） | Depth-Attention が parameter ゼロ・KV 中立の cross-layer value mixing を 360M から検証（E-3） | 中 |
| Training Stability Stress Harness（P0 共通基盤） | Wortsman の小規模 proxy 手法論（A-7 追記）が同一指標系（update norm・logit growth）で規模間予測を実証 | 中 |

## 2. この調査で新しく見えた設計軸（repo にまだ無いもの）

1. **gate の入力側（E-2 VGA）**: 現行 G1 は input embedding 側。V 側 gate の理論的優位が
   示された。G1 variant 探索に安い追加 arm。
2. **MTP の horizon を entropy で可変化（E-5 AdaMTP）**: adapter 構造不改変で
   loss 掛ける位置を選択するだけの拡張。future-mixture supervision とも接続。
3. **optimizer state の group 単位 exponent（D-3 integer-only training）**:
   低bit化の対象を optimizer まで広げる具体的手法。
4. **post-NS row normalization（C-1）**: Muown と同問題への state ゼル対処。
   HVX Muon への変更が最小（NS5 後に 1 演算）。
5. **K-interval softmax（H-3）**: QNN softmax を要素演算の grid 近似へ替え、
   学習から慣らす。Attention Normalizer Search と統合。
6. **logit softcap（J-2）**: `tanh(x/cap)*cap` による最終 logits の抑制。
   CE graph への変更が最小で、QK-norm 未導入の HexaTrain では
   「softcap 単体 → QK-norm 単体 → 併用」の最初の arm になる。
7. **難易度指標を bytes/token 系と言語統計へ拡張（I-1）**: loss EMA 案と並べて
   compression ratio / readability を candidate 指標に加える。

## 3. 外部動向として記録すべき構造変化

- **FP4 pretraining が 10T token 規模で FP8 並みに到達**（A-3）。recipe の中核は
  RHT + 2D scale + SR + 高精度 layer 選択の set。HexaTrain の P2 候補は個別ではなく
  recipe として設計すべき。
- **「NPU 上の exact-BP 学習」は公開文献に存在しない**（B）。HexaTrain の
  scratch training 構成の独自性は fine-tuning 系先行例と区別して主張できる。
- **linear attention 系の理論統一**（F-1 TTT ≡ linear attention）。
  HTP 実装は attention graph 部品の再構成として設計できる。

## 4. すぐ読むべき論文の優先順位（本文精読リスト）

1. **MUON+（2602.21545）** — Muown lane の比較 arm 設計に直結。挿入する normalization の
   正確な形を確認する必要がある。
2. **Compute Optimal Tokenization（2605.01188）** — Phase 10 の設計を compression rate
   掃引へ寄せるかの判断材料。最適 rate の回帰式。
3. **VGA（2510.09017）** — G1 variant の V-gate arm。gate の関数形と backward cost。
4. **Quantized Softmax（2609.33591）** — QNN softmax 置換候補の実装可能性
   （grid 構成・K の感度）。
5. **Depth-Attention（2606.05014）** — Value Residual lane の別実装。
   どの layer から key/value を取るかの設計。

## 5. 本調査の限界（2026-10-25 時点）

- abstract ベースの要約が中心。数値の詳細（特に A-3 の高精度 layer 内訳、
  C-2 の second moment update 則）は本文未確認。ただし精読リスト上位5本
  （MUON+ / tokenization / VGA / K-interval softmax / Depth-Attention）は
  本文確認済みで「次に読む」に追記済み。
- 「AlphaXiv」は paper page（AI overview / related 推薦 graph）として利用した。
  人的 discussion は主要論文（MUON+ / NVFP4）で空だったため、
  議論内容の参照はまだ無い。related 推薦経由で FP4 recipe 起源論文（A-8）を発見。
- 検索は英語圏・US 検索経由。中国語圏の model card / blog 記事は repo 側参照と
  重なるため本調査では省いた。
- 実機での検証は一切行っていない。すべて文献ベースの候補整理である。
- Qualcomm HMX 自体の学術論文・Hot Chips 詳細記事は公開されておらず、
  正式な情報源は V81 HMX PRM（repo が既に使用）のみ。HMX 内部構造の外部文献
  对照は現状不可能で、repo の CPU model 手法が唯一の公開検証経路。

---
