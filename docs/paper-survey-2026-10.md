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
