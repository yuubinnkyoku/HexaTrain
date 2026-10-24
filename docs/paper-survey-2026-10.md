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

### クラスタAの統合所見（2026-10-25時点）

1. **MX 系 block-scaled 学習は「何を高精度に残すか」で決まる。** Kempner の結果は
   LayerNorm affine weight 除外・activation 高精度維持・forward のみ量子化という
   段階的緩和で FP8 相当まで回復する。HexaTrain の Phase 12 は
   「全 tensor 一括低精度化」ではなく、最初から tensor 除外表を作って始めるべき。
2. **NVFP4 recipe = RHT + 2D scale + SR + 高精度 layer 選択 のセット。** HexaTrain の
   P2 候補（Hadamard 系 reparam / SR / op-wise precision search）は個別の候補としてではなく
   1つの recipe として扱う方が文献と整合する。
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
- [AdaMuon (arXiv:2507.11005)](https://arxiv.org/abs/2507.11005) — element-wise second moment を
  Muon へ統合。
- [Mousse (arXiv:2603.09697)](https://arxiv.org/abs/2603.09697)（2026-03）—
  Muon へ structural curvature を明示統合。
- **HexaTrain との関連**: 端末 RAM 上、optimizer state は Adam m+v が最大の固定費。
  Muown 導入の動機の一つ（embedding/head の 512 KiB 削減）と同型の削減が
  stateless 系では全行列対象になり得る。ただし whitening / Fisher 系の演算が
  HVX/CPU でどこまで低コストかは未検証なので、Phase 11 の candidate として
  「CPU reference 1-step cost」から測るのが現実的。
- **次に読む**: Alice の low-dimensional subspace 推定（HexaTrain の 64x64 行列群への適合性）。

### C-5. 索引

- [Awesome-Muon](https://github.com/zhanghualin0/Awesome-Muon)（2026-06 更新）—
  Muon 系の網羅 list。以後の追跡はこれを定期確認するのが最も安い。
- repo 参照済み: Muown 2本（2605.10797 / 2606.23637）、Muon is Scalable（2502.16982）、
  Polar Express（2505.16932）、Can Muon Fine-tune Adam-Pretrained（2605.10468）。
  本調査で新規に重要と判断したのは MUON+ / NorMuon / スペクトル理論の3点。

### クラスタCの統合所見（2026-10-25時点）

1. **「post-NS row norm 不均衡」が独立3系統（HexaTrain audit / Muon+ / NorMuon）で確認された。**
   Phase 4 は Muown 単体ではなく、**row 制御の位置（parameterization vs post-NS normalize vs
   per-neuron moment）を分離する3-4 arm 比較**へ拡張する情報的価値が高い。
2. **Muon+ は state 追加ゼロの one-line fix**なので、HexaTrain の HVX Muon 実装への
   変更が最小（NS5 の後に row normalize を 1 演算）であり、最初の arm として最も安い。
3. 理論（C-3）は「NS 係数の妥協が常に致命的とは限らない」ことを示唆し、
   P2 の Polar Express / NS8 比較lane と QNN HTP NS の再試行条件の設計に使える。

---
