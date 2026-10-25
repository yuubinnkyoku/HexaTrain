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

（全クラスタA〜Jの第一 pass を 2026-10-25 に完了。以後の追記は
「精読済み」注記・追加論文・新着掃描の形式で行う。）

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
  → **【2026-10-25 精読済み】**: 高精度残置の内訳は
  (1) **最初の2 block + 最後の8 block を BF16**（線形層の約16%。ablation では
  最後の4 block だけでも安定）、(2) embedding / 出力 projection head /
  normalization / 非線形 / **attention の softmax・QK・score-value GEMM**、
  (3) 主要 weight・weight gradient・optimizer state は FP32。
  RHT は **Wgrad 入力のみ**に適用（Fprop/Dgrad 入力への適用はむしろ有害）、
  tile は 16x16、**乱数 sign は全 layer 共通の固定 seed 1本**。
  2D scale は weight を 16x16 block（入出力双方）にして
  **forward と backward で同一の量子化表現を保証**するのが目的
  （無いと backprop の転置で scale 次元が変わり chain rule が壊れる）。
  SR は gradient（Dgrad+Wgrad）のみ。**forward tensor への SR は divergence した**。
  12B/10T で FP8 比の相対 loss 誤差は安定期 <1%、LR decay 後 ~1.5%。
  vs MXFP4 は同 loss に 36% 多い token が必要。
  HexaTrain の tensor 除外表（Phase 12 初手）としてほぼそのまま使える構成。

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
  → **【2026-10-25 精読済み】**: 理論の要点は
  (1) RTN は QAT 目的に対して**有biased**、SR は unbiased、
  (2) **weight 量子化の bias `B_W ∝ L√d Δ_W` は batch size で消えない**
  （weight は batch 全体で共有され丸めが1回だから誤差 floor になる）、
  (3) activation/gradient の SR noise は per-sample i.i.d. で `T^Q ∝ (1/b)·2^(-2B)`。
  **1 bit 落とすと variance が4倍 → batch 4倍で相殺**（実測では約2倍で足りる）。
  収束 bound は `4(L(w0)−L*)/ηT + (2+4ηL)B_W² + 2ηL(σ_S²+σ_Q²)/b`。
  実験は WideResNet / Llama-3.2-3B SFT / BERT-GLUE で E4M0〜E4M2 × batch 8〜32。
  **forward は高精度維持・backward だけ量子化**が前提。
  また SR の hardware cost は LFSR で FMA 比無視できる規模（66 register 級）と試算。
  HexaTrain 適用の含意: (a) B8 固定では `1/b` 項が効かないため SR 導入は
  **accumulation 増とセット**が理論上必須、(b) weight 側低bit化は
  batch で救えない bias floor を作るため Phase 12 では activation/gradient 側から、
  (c) backward のみ量子化方針は Kempner（A-2）の緩和策と一致。

### A-6. A Stochastic Rounding-Enabled Low-Precision Floating-Point Training

- [arXiv:2404.14010](https://arxiv.org/abs/2404.14010) / [AlphaXiv](https://www.alphaxiv.org/abs/2404.14010)
  — Ali ほか、2024-04。
- **要約**: rounding に使う乱数 bit 数を変えた場合の hardware cost と精度への影響を調査。
  最適化した SR により **accumulator の word length を削っても精度を維持できる**ことを示す。
- **HexaTrain との関連**: 「37-bit ACC に留まる積和で十分か」という HMX accumulator 系の問いに
  対して、accumulator bit 幅 × rounding 方式の観点からアプローチする文献。R2 の signed 出力
  実験とは独立に、ACC bit 幅の妥当性を CPU model で測るときの对照枠になる。
- **次に読む**: accumulator word length を削った具体的な実験設定。
  → **【2026-10-25 abstract 再確認】**: 具体構成は **FP8 乗算入力 × FP12 累算**の
  MAC unit で、swamping error を **最適化 SR（eager SR unit）で緩和**し
  subnormal 対応を削除。CV タスク群で近 baseline 精度を維持。
  random bit 数別の hardware/精度トレードは本文が必要。
  → HexaTrain との対比が鮮明: **HMX の 37-bit ACC は FP12 累算 + SR の
  数十倍の余裕**がある設計であり、「37-bit に留まる積和で十分か」の問いは
  「FP12+SR で回る literature が存在する以上、37-bit は amplitude headroom 的に
  過剰でない限り安全」と読める。R2 の CPU model で headroom の
  実使用率を測ることがこの対比の数値化になる。

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
  - [Scaling FP8 training to trillion-token LLMs（arXiv:2409.12517）](https://arxiv.org/abs/2409.12517) /
    [AlphaXiv](https://www.alphaxiv.org/abs/2409.12517)
    — Fishman, Chmiel, Banner（Intel/Habana）, Soudry（Technion）、2024-09（引用78）。
    trillion-token 級 FP8 混合精度学習の安定化。COAT・MOSS への直接的な先行。
    Phase 12 の大規模側参照として保持。
- **【2026-10-25 新着掃描で追加】loss spike 系**:
  - [Stabilizing the Pre-training of LLMs（arXiv:2312.16903、v4 2025-07）](https://arxiv.org/abs/2312.16903)
    — loss spike の理論解析（少数の bad batch による **gradient norm の制御不能な成長**
    が原因）。**小規模 model で大規模の不安定を予測する**手法は Wortsman（上記）と
    同型で、spike の起きる機構の理論面を補う。stress harness の
    「gradient norm 監視」項目の理論的根拠。
  - [Spike-Aware Adam with Momentum Reset（arXiv:2501.06842）](https://arxiv.org/abs/2501.06842)
    — 複数 architecture で gradient spike が普遍的に発生することを示し、
    spike 検出 + **momentum reset** による緩和 optimizer を提案。
    HexaTrain で spike が観測された場合の緩和候補（QNN graph 外の CPU 側で
    適用可能な形状）。abstract 確認済み・本文未読。

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
  → **【2026-10-25 NVFP4 精読との対比で注記】**: RHT の適用範囲について、
  Tseng ら MXFP4（本 entry）は「3種の training GEMM 全部」、NVFP4（A-3）は
  「Wgrad 入力のみ（他は有害）」と**逆向きの結論**。規模・format 差
  （MXFP4 の k=32 vs NVFP4 の 16x16 2D scale）依存の可能性があり、
  HexaTrain では RHT 適用範囲を独立 factor として掃引する根拠になる。

### A-9. Quant.npu — NPU 向け fully static 量子化（2026-10-25 追加）

- [arXiv:2605.20295](https://arxiv.org/abs/2605.20295) / [AlphaXiv](https://www.alphaxiv.org/abs/2605.20295)
  — Zhang, Xu, Wang, …, Mengwei Xu, Huang（北大系）、2026-05。
- **要約**: mobile NPU 上の LLM 推論向け integer-only・**fully static** 量子化。
  通常の PTQ が activation 量子化 parameter を runtime で動的再計算するのに対し、
  NPU は**完全に静的な量子化 parameter** を要求するという制約に対処。
  learnable な静的量子化 parameter + rotation matrix、rotation 対応の
  2段階選択的最適化、sensitivity-guided mixed precision。
  実機 NPU で SOTA PTQ と同等精度、latency 最大 15.1% 削減。
- **HexaTrain との関連**: Phase 12 の HTP 向け量子化設計に直接響く制約の明文化。
  QNN HTP graph も固定 shape・静的 parameter 前提なので、
  「runtime で量子化 parameter を計算する方式は HTP に載らない」。
  HexaTrain の学習は自前 graph なので学習中に scale を更新する余地はあるが、
  推論 graph（FORWARD_ONLY / prepared generation）へ持ち込む量子化は
  静的 parameter 化が前提になる。rotation 系（Hadamard/RHT）との併用も
  A-3/A-8 の recipe と整合する。
- **次に読む**: rotation 対応 2段階最適化の分離基準（rotated / unrotated tensor）。

### A-10. Full-Stack FP4 — optimizer state まで FP4 化（2026-10-25 追加・重要）

- [arXiv:2607.04422](https://arxiv.org/abs/2607.04422) / [AlphaXiv](https://www.alphaxiv.org/abs/2607.04422)
  — Ding, Ma, Tong, Xing, Wang, Li、2026-07（v2 2026-08）。
- **要約**: NVFP4 量子化を線形層から **optimizer state と attention まで拡張**。
  (1) AdamW momentum の NVFP4 保存を **sqrt / tile-mean / Hadamard 変換の
  順序付き pipeline** で安定化、(2) **Muon 系（Root 反復）の計算自体も NVFP4 で実行**
  （shape 依存係数と clipping で直接 FP4 Root 反復を安定化）、
  (3) LoRA-SVD で小さい BF16 部分空間を残し loss gap を 1.40% → 0.61% に削減。
  3B / 64B token で BF16 比 loss gap 0.838%、zero-shot PPL ほぼ tie。
  RTX 5090 で **Root optimizer 2.50〜2.83x 高速化・AdamW peak memory
  37.9〜42.5% 減**。
- **HexaTrain との関連**: repo の P2「Quantized Adam / optimizer-state 低bit化」の
  最新・最強 instance。特に **Muon（Root/NS 反復）自体の低bit化**は
  HVX FP32 Muon の wall time 問題（約55〜59 ms/update）への
  「反復を低bit tensor core で回す」方向の存在証明になる。
  HexaTrain では HVX に FP4 tensor core はないが、
  (a) momentum 保存の低bit化（sqrt/tile-mean/Hadamard pipeline）は
  HVX/CPU 実装に直接移植可能、(b) **NS 反復の数値安定化は QNN HTP NS 再試行の
  precision gate 条件を満たす新経路**（A-3 の forward 高精度残置との対比で、
  「反復側を安定化する」アプローチ）。
- **次に読む**: FP4 Root 反復の clipping 係数の決め方（NS 係数との関係）。
  → **【2026-10-25 本文確認済み（HTML）】**: NS 反復は3項2次型
  `X_{k+1} = aX_k + bX_k(X_k^TX_k) + cX_k(X_k^TX_k)^2`。
  「shape-dependent」の中身は **Root（He et al. 2025）の最適化係数を
  行列の aspect ratio 別に特殊化したもの**で、連続 NS step 間の
  誤差増幅を抑える役割。**p99 outlier clipping** を併用、
  量子化は RTN + adaptive 4/6 scaling、**1x16 block に Hadamard 回転**を挟む
  pipeline。効果は NVFP4 NS の誤差 **~10% → ~3%**。
  係数の具体値・導出は本論文に載らず Root 側に委ねられる。
  → HexaTrain への含意: (a) **aspect ratio 別係数**の発想は Phase 5 で
  非正方行列（G1 gate 等）を NS に掛ける場合の係数選択に直結
  （64x64 正方では影響小）、(b) **低精度 NS では p99 clipping が
  標準的な安定化手段**という知見は HTP NS 再試行の precision gate 設計に
  使える（clipping 前提の gate 条件を書ける）。

### A-11. Quartet II — MS-EDEN、SR を超える unbiased 量子化（2026-10-25 新着掃描）

- [arXiv:2601.22813](https://arxiv.org/abs/2601.22813) / code: `IST-DASLab/Quartet-II`
  （Quartet / MXFP4 と同系、2026-01）。
  **【2026-10-25 abstract 確認済み】**: NVFP4 全量子化 pre-training を対象に、
  micro-scaled format 向け **unbiased 量子化 routine MS-EDEN** を提案。
  主張: **SR より量子化 error が 2x 以上低く**、forward/backward の主要な
  全 matrix multiply で gradient 推定が一貫して向上。
  1.9B / 38B token で end-to-end 検証、Blackwell kernel で **BF16 比最大 4.2x**。
- **HexaTrain との関連**: A-3（NVFP4 recipe = SR）と A-5（SR 理論）に対する
  **第3の選択肢の出現**。SR は「unbiased だが variance が増える」、MS-EDEN は
  「unbiased かつ error 小さい」を狙う。HexaTrain の Phase 12 で SR A/B を回すなら
  「SR vs MS-EDEN 型 unbiased estimator」の比較設計が最新の正解系谱になる。
  ただし MS-EDEN の演算コストが HVX に乗るかは未確認（SR は LFSR でほぼ zero-cost、
  A-5 参照）。abstract 確認済み・本文未読。

### A-12. Transposition-Invariant 2D block FP4 — 前後で scale を不変に（2026-10-25 新着掃描）

- [arXiv:2607.24953](https://arxiv.org/abs/2607.24953) — Rahimifar ほか、2026-07。
  **【2026-10-25 abstract 確認済み】**: FP4 training の不安定の原因を
  **tensor 転置に起因する scale 不一致**と特定（1D block では forward と backward で
  同じ weight に別の scale が付き、gradient が biased・不安定になる）。
  **転置不変な 2D block FP4** + truncation-free scaling + SR で解決。
  Q/K projection だけ MXFP8 に残す mixed-precision。7B dense + 30B MoE、
  最大 100B token で BF16 比 **perplexity / downstream 1.3% 未満の劣化**。
- **HexaTrain との関連**: A-3 の NVFP4 recipe が **weight scale を 2D (16x16) にする**
  理由の理論化と読める。HexaTrain の V81 HMX は MX 系 1D block（32 要素）であり、
  HMX block-scaled MMA の scale 計算が forward/backward で一致しているかは
  **R2 系数値実験の新しい観測点**（dX と dW で同 tensor の scale が一致しない
  構造は HexaTrain にも存在し得る）。Q/K だけ高精度残す選択は
  Phase 12 の tensor 除外表の文献側実例がまた1つ増えたことになる。

### A-13. ScaleSearch — BFP scale を max でなく探索する（2026-10-25 新着掃描）

- [arXiv:2605.12464](https://arxiv.org/abs/2605.12464) — Gupta, …, Tri Dao, Fu,
  De Sa ほか、2026-05。
  **【2026-10-25 abstract 確認済み】**: BFP の scale を「block 最大値」でなく
  **microscaling format の mantissa bit を使った fine-grained 探索**で選び、
  分布ごとに量子化 error を最小化。NVFP4 で error 27% 減、
  PTQ で MATH500 最大 +15pt（Qwen3-8B）、Llama 3.1 70B の Wikitext-2 PPL 0.77 改善。
  ScaleSearchAttention（NVFP4 attention 高速化）も提案。
- **HexaTrain との関連**: HexaTrain の HMX block scale は MX 系の
  power-of-2 E8M0 共有 scale が前提で、ScaleSearch の free mantissa scale 探索は
  **HMX hardware の制約と非互換**（scale は format で固定）。
  適用余地は QNN graph 外の CPU/HVX 側 tensor（optimizer state 保存など）のみ。
  「scale 選択は最適でない」という問題提起自体は Phase 12 の静的 scale 設計
  （学習中 scale 更新の A/B）と接続する。abstract 確認済み・本文未読。

### A-14. Outlier-Safe Pre-Training — Muon が outlier を作らない（2026-10-25 新着掃描）

- [Outlier-Safe Pre-Training for Robust 4-Bit Quantization of LLMs（arXiv:2506.19697）](https://arxiv.org/abs/2506.19697)
  / code: `dmis-lab/Outlier-Safe-Pre-Training` — 2025-06。
  **【2026-10-25 abstract 確認済み】**: extreme activation outlier の原因を
  **channel-wise 演算と adaptive gradient scaling** に帰属させ、
  事後緩和でなく「outlier を作らない学習」を提案。3技術:
  (1) **Muon optimizer**（privileged basis の除去）,
  (2) **Single-Scale RMSNorm**（channel 別 gain をやめる）,
  (3) **learnable embedding projection**（embedding 出力の magnitude 再配分）。
  1.4B / 1T token で「outlier 無しの production-scale LLM は初」、
  excess kurtosis **0.04**（通常 1818.56）、4-bit 量子化で 35.7 vs Adam 26.5
  （10 benchmark 平均）、training overhead 2%。
- **HexaTrain との関連**:
  (1) **Muon 採用自体が 4-bit 配備耐性の有利な先行条件**になるという外部証拠。
  HexaTrain（HVX Muon）の checkpoint は Adam 学習の同規模 model より
  量子化しやすい可能性が高く、Phase 12 の出発点として不利ではない。
  (2) **Single-Scale RMSNorm は model 構造の変更**を伴う（現行 RMSNorm の
  per-channel gain を外す）— G1/Phase 7 の構造 arm との交差点だが、
  758k model での利得は unverified。
  (3) 「outlier は LLM 固有でなく学習戦略の帰結」という主張は、
  Phase 12 を「低bit化の許容表を作る co-design」として設計する根拠になる。

### クラスタAの統合所見（2026-10-25時点、A-8〜A-10 追記後更新）
---

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
4. **NPU は fully static な量子化 parameter を要求する**（A-9）。
   動的な activation scale 計算を前提にした手法は HTP graph に載らない。
   Phase 12 の候補は最初から「静的 scale（学習中に更新・graph へ焼き込み）」で設計する。
5. **低bit化の対象は optimizer state まで及んだ**（A-10）。
   Muon 系 Root 反復自体の FP4 実行が成立しており、
   「momentum 保存の低bit化」は HVX/CPU 実装に移植可能、
   「反復側の安定化」は QNN HTP NS 再試行の新たな precision gate 経路になる。
6. **【A-11〜A-13 追加後】unbiased 量子化は SR 一強でなくなった。**
   MS-EDEN（A-11）が SR 比 2x 低 error を主張し、Phase 12 の estimator A/B は
   「SR vs MS-EDEN 型」に更新すべき。また forward/backward の
   **scale 一致性**（A-12）は HexaTrain の HMX R2 数値実験の新観測点であり、
   「scale 探索」（A-13）は HMX 固定 scale と非互換だが CPU/HVX 側 state には適用余地。

---

## B. on-device / NPU training systems

**調査日: 2026-10-25。** 「smartphone / edge で NN の学習を回す」系の文献は
fine-tuning 前提のものが多く、**QNN HTP graph 上の exact-BP 言語モデル scratch
training を行っている公開文献は見つからなかった**（DSP offload による
mixed-precision DNN 学習の先行例 Mandheling は B-7 に追加済み）。
HexaTrain の「QNN HTP Forward/Backward + HVX optimizer + CPU Aux の scratch
training」という構成は、公開文献の範囲では近隣が存在しない位置にある。
したがってこのクラスタは「直接の競合」ではなく、**Gradient Strategy（P2）と
memory planner の設計参照**として読む。

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
  - **【2026-10-25 abstract 確認済み】[Steering the Noise（arXiv:2601.04710）](https://arxiv.org/abs/2601.04710)**
    — 候補 perturbation pool を評価し、loss の低/高 group の対比から
    guiding vector を作る **MeZO-GV** と、budget 内で最良1本を残す **MeZO-Greedy** を提案。
    OPT-13B の 11 benchmark で全 ZO baseline を上回り 9 個で gradient 系を超える。
    perturbation pool + 選択という variance reduction は QZO-FF（sign-m-SPSA）にも
    追加可能な plug-in 形（追加 forward が HTP で相対的に安い HexaTrain とは相性がある）。
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
- [When NPUs Are Not Always Faster（arXiv:2605.27435）](https://arxiv.org/abs/2605.27435) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2605.27435)
  — Li, Qi, Chen、2026-05。
  **【2026-10-25 確認済み】**: CPU-NPU heterogeneous SoC 上の LLM 推論を
  OPMASK-based pipeline 分解で通信・量子化・計算に分離した初の体系的 benchmark。
  **Prefill で CPU が NPU より最大 1.6x 速い stage-level 逆転**、
  decode では NPU でも 1.05〜1.2x 程度、scheduling overhead と cross-backend
  fallback が gain を侵食、NPU offload 増で energy が最大 51% 増える場合もある、
  という結果。**SIGMETRICS 2027 論文（NPU は prefill で勝つ）と逆向きの結論**で、
  NPU 利得が framework・SoC・workload 依存で不安定であることを示す。
- **【2026-10-25 追加】Hexagon kernel layer 直叩き系の新着**:
  - **ParallelPrefill（ACM, DOI 10.1145/3821725.3821806、2026-09）** —
    LLM prefill を **Hexagon DSP kernel layer で 2D 並列化**。
    QNN / AI Engine Direct SDK が **HMX/HVX level の制御を露出しない**ことを
    低 layer 最適化の動機として明記。HVX Muon を手書きする HexaTrain と同じ
    「QNN から見えない層」を突く研究で、kernel layer の並列化設計の参照。
    ACM のみで arXiv 版未確認（詳細は要本文）。
  - **A Systematic Study of LLM Serving on Multi-core NPUs（arXiv:2510.05632）** —
    multi-core NPU 上の tensor parallelism 戦略の体系解析。
    HexaTrain の単一 NPU 前提とは異なるが、HTP 内部の並列構造を扱う数少ない公開資料。
- **HexaTrain との追加関連**: この矛盾自体が「実機 telemetry を正本とする」
  HexaTrain の運用原則（G1 runtime lane の `NO_SAFE_GAIN` 教訓）を
  外部から支持する。backend placement は graph 理論だけでなく
  実測の通信・量子化・scheduling 分解で決めるべきという同系統の主張。
- **HexaTrain との関連**: 「NPU decode は弱い」という計測は、
  P1「Incremental KV-cache generation」の設計に直接影響する。
  1-token decode を HTP 固定 graph で回すか、decode だけ CPU/HVX に置くかの
  backend placement 判断（7層共同最適化の Backend Placement）に
  外部計測からの根拠が加わった。HexaTrain の rolling T32 再計算方式は
  「固定 shape を保つ」方向で NPU 特性には合っているが、per-token 固定費が
  大きいという別問題を持つ。
- **次に読む**: decode の kernel size と NPU 効率の関係（incremental KV cache の
  shape 設計の参照）。

### B-6. heterogeneous co-execution scheduling（2026-10-25 追加）

- 代表例: [Band（MobiCom 系 2022, 多-DNN を mobile processor 間で調整）](https://snu.elsevierpure.com/en/publication/band-coordinated-multi-dnn-inference-on-mobile-proc)、
  Thermal-Aware Scheduling for DL on Mobile with NPU（2026,
  [ResearchGate](https://www.researchgate.net/publication/379297343_Thermal-Aware_Scheduling_for_Deep_Learning_on_Mobile_Devices_With_NPU)）、
  fine-grained CPU-GPU partitioning（2026, USC, [PDF](https://qed.usc.edu/paolieri/papers/2026_epew_mobile_inference_cpu_gpu.pdf)）。
- **要約**: mobile の CPU/GPU/NPU co-execution は **推論側**で確立した領域。
  subgraph 分割・thermal-aware な配置・real-time graph 解析に基づく適応 scheduling、
  線形 op の fine-grained co-execution で最大 1.89x 等の結果がある。
- **HexaTrain との関連**: repo の「Heterogeneous Training Backend Planner」は
  これらの**学習版**に相当し、やはり公開文献に直接の先行例がない
  （cluster B の統合所見と同じ構図）。thermal-aware scheduling の
  評価軸（温度と throughput の履歴を scheduling 入力にする）は
  HexaTrain の thermal telemetry を活かす placement 判断の参照になる。
- **次に読む**: thermal-aware の報酬設計（thermal throttle をどう cost 化するか）。
  → **【2026-10-25 書誌確認済み】**: Tan et al., *Thermal-Aware Scheduling for Deep
  Learning on Mobile Devices With NPU*, **IEEE TMC 23(12), 2024-12**。
  DNN op を subgraph へ分割し GPU/NPU 間で thermal throttle を避けるよう配分。
  後続に IceEdge（ACM 2026, [dl.acm.org/doi/full/10.1145/3801093](https://dl.acm.org/doi/full/10.1145/3801093)）、
  OPTiC（CPU-GPU collaborative, thermal constraint）がある。
  HexaTrain では thermal telemetry を placement 判断へ入れる際の
  先行評価軸として参照。本文の cost 設計は IEEE 有料壁のため未確認。

- **FastRPC 周辺の学術文献は希薄（2026-10-25 掃描）**: ML 推論を Hexagon DSP へ
  offload する際の FastRPC overhead を分析した論文（faster host CPU で
  orchestration overhead が 10〜29% 減という観察）、DSPBooster
  （アプリの DSP offload の power-performance 最適化）程度で、
  **学習 workload での FastRPC 境界設計を扱う公開研究は見つからない**。
  FastRPC 内部構造の公開資料としては
  [Checkpoint の Pwn2Own Qualcomm DSP 解析](https://research.checkpoint.com/2021/pwn2own-qualcomm-dsp)
  （FastRPC / QuRT RTOS の内部）と
  [Qualcomm FastRPC GitHub](https://github.com/qualcomm/fastrpc) が最も詳しい。
  HexaTrain の RPC 境界・pack/unpack 最適化も公開文献上の空白であることを記録。
  **【2026-10-25 再掃描】**「RPC-bound training / accelerator-aware optimizer 配置 /
  host-accelerator 通信 overlap」で再検索したが、GPU 集合通信の offload（HOCCL,
  arXiv:2609.34334）や multi-node 通信 scheduling（TRANSIT, arXiv:2610.07593）が
  近いのみで、**学習 optimizer 境界としての RPC 設計を扱う文献は依然不存在**。
  空白判定は維持。

### B-7. Mandheling — DSP offload による mixed-precision 端末学習（2026-10-25 追加・重要）

- [arXiv:2206.07509](https://arxiv.org/abs/2206.07509) / [AlphaXiv](https://www.alphaxiv.org/abs/2206.07509)
  — Daliang Xu, Mengwei Xu, Wang, Wang, Ma, Huang, Huang, Jin, Xuanzhe Liu
  （北大 / Mengwei Xu group、LLM.npu / Quant.npu と同 group）、2022（引用91）。
- **要約**: **on-chip DSP offload と mixed-precision 学習を組み合わせた
  端末学習 system の最初のもの**とされる。4技術:
  (1) DSP が不得意な operator の **CPU-DSP co-scheduling**、
  (2) backprop の dynamic rescaling cost を減らす **self-adaptive rescaling**、
  (3) DSP cache 効率を上げる batch-splitting、
  (4) DSP 準備 overhead を減らす **DSP-compute subgraph reuse**。
  TFLite / MNN 比で per-batch 学習時間 5.5x 減・energy 8.9x 減、
  end-to-end で最大 10.7x 高速収束・13.1x 省energy、精度劣化 1.9〜2.7%。
  workload は vision DNN（言語モデル学習は対象外）。
- **HexaTrain との関連**: **HexaTrain に最も近い公開学習 system 文献**。
  ただし (a) vision DNN 中心で LM ではない、(b) HTP/QNN graph ではなく
  DSP 直接 offload、(c) mixed-precision（FP32 対比精度劣化あり）で
  exact-BP scratch LM 学習ではない、の3点で HexaTrain の構成とは別物。
  特に (2) self-adaptive rescaling は低精度 backprop の dynamic scaling 問題への
  先行対処であり、Phase 12 の静的 scale 設計（A-9 の制約）と対になる知見。
  (4) subgraph reuse は QNN graph 再利用設計と同型。
- **クラスタB冒頭の主張の修正**: 「accelerator 上の exact-BP 学習は文献に存在しない」は
  「**HTP/QNN graph 上の exact-BP 言語モデル scratch 学習**は存在しない」へ
  限定して正確化する。DSP offload による mixed-precision DNN 学習は
  Mandheling が先行する。

### B-8. Hexagon-MLIR — Hexagon NPU 向け AI compile stack（2026-10-25 追加・確認済み）

- [arXiv:2602.19762](https://arxiv.org/abs/2602.19762)（2026-02）—
  Absar, Baskaran, …, Lethin ほか25名（Qualcomm 系）。
  **【2026-10-25 確認済み】**: MLIR ベースの **open-source** compile stack で、
  **Triton kernel と PyTorch 2.0 model の双方を Hexagon NPU 向け binary へ
  自動 lowering** する。中心最適化は **TCM（tightly coupled memory）上の
  data locality を最大化する mega-kernel 生成**（library call 構成より
  bandwidth 瓶颈を減らす）。work-in-progress と明記。
  注意: 対象は Hexagon **NPU**（HVX ベクタ命令の直接 compile ではない）。
- **HexaTrain との関連**: (a) 手書き QNN graph と同じ「固定 graph + locality 最大化」
  方向を compiler が担う路線であり、QNN/HVX Superoptimization（P1）の
  将来的な自動化 backend 候補、(b) mega-kernel / TCM locality の設計思想は
  HMX retained-ACC（R2）や Memory Planner（Phase 9）と同型の
  「on-chip に留める」原則の compile 側実装、(c) open source のため
  中期的な検証材料になる。

### B-9. First-Order LoRA Training with Phone NPUs — V81 HMX 上の LoRA 学習（2026-10-25 追加・最重要近隣）

- [arXiv:2610.24816](https://arxiv.org/abs/2610.24816)
  — Zheng, Cheng, Chen, Xu, Sha, Ni, Luo ほか、**2026-10-17**（8日前の新着）。
  **【2026-10-25 abstract+HTML 確認済み】**: phone NPU 上で first-order LoRA training を
  行う system。frozen weight の3行列積（forward / activation recompute /
  **input gradient**）をすべて NPU へ置き、training graph / adapter / optimizer は CPU。
  **HexaTrain と同世代の実機**: Snapdragon 8 Elite Gen 5（**Hexagon V81**、OnePlus 15）と
  8 Elite（V79、OnePlus 13T）。backend は QNN でなく **改造 llama.cpp ggml Hexagon backend**。
  技術要素: (1) input-gradient kernel が forward の **resident weight tile と
  dequantizer を再利用**（転置 weight copy 不要）、(2) 精度を **gradient sensitivity で
  選択**（敏感な projection は FP16、残りは **32 weights/block + FP16 scale の Q8
  block 量子化**）、forward の dequantizer を backward でも再利用,
  (3) **FP16 出力への per-row range guard**、(4) NPU/adapter overlap, CPU operator fusion,
  (5) **resumable 実行と thermal pacing** による夜間 background training。
  workload は端末内 label による nightly personalization と LaMP-2。
  Qwen / Llama 系 4 model。CPU trainer / phone-GPU / forward-only / GPU 参照と比較。
- **HexaTrain との関連**: **これまでで最も近い隣接文献**（同 V81 HMX・同
  「NPU が行列積を担い CPU が optimizer」分割）。ただし **LoRA first-order で
  exact-BP scratch training ではなく**、weight gradient を NPU で出さない点、
  ggml backend（QNN graph でない）の点で HexaTrain とは別系統。
  クラスタB冒頭の正確化主張（HTP/QNN graph 上の exact-BP scratch LM 学習の不存在）は
  この論文によっても**維持される**（むしろ最新研究も first-order LoRA 止まりという
  ことは HexaTrain の空白の実在を補強する）。
  交差点は3つ:
  (a) **D クラスタ / R2 系**: 「HMX は FP16 operand を受け wider cell で accumulate し
  FP16 result を書く」という外部記述と、**FP16 出力の per-row range guard** は
  HMX signed FP16 出力（R2 の retain bit / late-clamp 系）と同じ問題意識の独立実装。
  対照表を作る価値がある。
  (b) **Phase 12**: gradient sensitivity による tensor 別精度配分（FP16 vs Q8 block）は
  NVFP4 recipe（A-2）の tensor exclusion table と同型の**第2の実例**。
  (c) **Cross-Layer Reuse / Phase 9**: resident weight tile + dequantizer 再利用は
  「forward が既に持つものを backward で再計算しない」原則の具体的実装で、
  retained-ACC（R2）と同じ方向。
  さらに本文で GDN（recurrent）系 projection が gradient-sensitive 側に挙げられており
  Phase 14（GDN migration）とも接する。推論 stack が llama.cpp 系のため
  QNN graph の precision 制約（A-9）を回避している点も設計比較の材料。
- 次に読むこと: 精度配分の sensitivity 推定方法（§実装）、per-row range guard の
  clamp 値と HMX R2 の late-clamp 閾値の比較、V79 と V81 の性能差。
  → **【2026-10-25 本文確認済み】**（HTML 版 §2-3 / Table 5-6）:
  (1) **sensitivity は runtime 推定ではなく一回限りの offline 計測**: LoRA gradient の
  FP32 参照比 relative L2 error を projection type ごとに測定し、
  **recurrent/GDN 系 projection だけ FP16、残り Q8 に固定**。
  根拠は「recurrent GDN state が weight 扰動を以後の全 token に運ぶ」という
  誤差増幅の構造（uniform Q8 18.8% → GDN だけ FP16 で 9.6%）。
  (2) **per-row range guard の実数値**: partial result が nonfinite または
  **60,000**（FP16 max 65,504 未満に意図的な margin）に達した行を flag →
  client が 2 の冪 scaling `c_i` で max |c_i·dY_ij| ≈ **2^12** に置いて再送、
  retry で `c_i × 2^-8`、最大3回。**「partial result は FP32 累算の前に丸められ、
  後段の cancellation では overflow を修理できない」**という順序依存の理屈が明記。
  実運用ではほぼ発火しない（全 epoch で 52 行 retry）。
  (3) **「HMX accumulates in 37-bit cells」「8 MB VTCM」の外部明記** —
  HexaTrain の HMX PRM 解釈（37-bit accumulator）の**独立した外部確認**。
  (4) V81 kernel: input-gradient 13.0 TFLOPS（FP16@840 token）/ 17.0（Q8@2048）、
  V79 比で GDN qkv 5.60→5.00 ms 等 V81 が概ね優位。tile 再利用の cycle 実測
  （activation operand 76 cycles vs weight operand 展開 152-182）。
  → R2 系への含意: HexaTrain の late-clamp / retain bit と同じ問題（**部分和の
  FP16 convert で overflow が修理不可**）を「margin 付き threshold + 2の冪
  row scaling + retry」で解く実装例が実在する。R2 の閾値設計（Rs[0] retain と
  clamp 値）はこの 60,000 / 2^12 の選択と**数値的な突き合わせができる**。
  また (1) の「GDN state が誤差を運ぶ」は Phase 14（GDN migration）の
  精度設計に直接の注意喚起として記録。

### B-10. 近隣 fine-tuning / 推論 追補（2026-10-25 新着掃描）

- **PieBridge（ACM SenSys 2024, DOI 10.1145/3666025.3699327）** —
  Yin, Wei, Liu ほか（北大 / B-7〜B-9 と同 group）。
  frozen backbone を学習中 **より小さい proxy network で動的に近似**し、
  full model を backprop しない on-device fine-tuning。
  HexaTrain 関連は薄い（scratch training では proxy 化対象が無い）が、
  「frozen 部分の backprop を回避する」もう一つの極（B-3 の ZO 系と同列）として記録。
  ACM のみで arXiv 版未確認。
- **shadowNPU / shadowAttn（arXiv:2508.16703, MobiSys'26）** — NPU-centric の
  on-device **推論** co-design。attention の quantization sensitivity が原因で
  attention operator が CPU/GPU へ fallback する問題を、NPU 上 pilot compute で
  重要 token 推定を隠蔽する sparse attention で解く。
  training 論文ではないが「quantization sensitivity による operator 落下」は
  HexaTrain の Phase 12（低精度化が HTP graph から落ちる operator を作る問題）の
  推論側 mirror として B-5 に並べる価値あり。abstract 確認済み・本文未読。

### クラスタBの統合所見（2026-10-25時点、B-7 追記後修正）

1. **HexaTrain の位置は公開文献上も空白。** 「mobile NPU 上の exact-BP scratch LM training」は
   調査範囲で存在せず、近隣（PockEngine / MeBP / ZO 系）はすべて fine-tuning 前提。
   対外説明では「fine-tuning 系の先行例と区別した scratch training」という表現が正確。
2. **Gradient Strategy の設計空間が広がっている。** exact BP ↔ ZO の二択でなく、
   low-rank ZO / sparse ZO / curvature ZO / activation-guided ZO / sparse backprop が
   2024-2026 に集中している。Phase 11 は MeBP / QZO-FF だけでなく
   LOZO・LOREN 型の選択肢も含めて再設計する価値がある。
3. **【B-9 追加後】近隣最先端は「NPU+CPU 分割の first-order LoRA」で一致、
   しかし exact-BP には届いていない。** 2026-10 時点の最新 system（B-9）が
   HexaTrain と同世代 V81 HMX で input-gradient まで NPU に置いている一方、
   weight gradient を NPU で計算する構成は依然存在しない。
   「frozen 3行列積 NPU + adapter/optimizer CPU」が実用解のデファクトになりつつあり、
   HexaTrain が NPU/HVX 側に置く範囲（weight gradient・optimizer 前段）の
   差分が研究価値の所在を示す。

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
  **【2026-10-25 確認済み】**: Ma, Gong, Scetbon, Meeds。
  gradient へ (1) normalization（分布安定化）と (2) whitening（局所 curvature 相殺）を
  **stateless に per-step 適用**して SGD で更新。LLaMA 350M/1.3B で
  Adam と同等以上の perplexity を **2x token efficiency**（半分の token で同 perplexity）、
  end-to-end memory 約50%減。Adam と同品質、SGD と同 memory。
  なお **SWAN / GMN / RACS / Alice / ARO は Gong・Chao Ma ら同一グループの
  系統的研究**（stateless・FIM 構造・回転設計）であり、
  この group の新論文を追うのが低 state optimizer 系追跡の最短経路。
- [Gradient Multi-Normalization (arXiv:2502.06742)](https://arxiv.org/abs/2502.06742)
  （Microsoft）— SWAN 動機の stateless optimizer 設計framework。repo 参照済み。
  **【2026-10-25 確認済み】**: Scetbon, Chao Ma, Gong, Meeds（SWAN と同 group）。
  複数の norm で同時に gradient を正規化する交替 scheme の一般枠組みで
  **SWAN は特定の norm 選択の特殊例**と示される。SWAN の whitening は
  大規模で高コストなため、性質を緩めた安価な variant を導出。
  交替法の不動点収束理論あり。LLaMA 1B までで Adam 比 3x 学習高速化。
- [RACS / Alice (arXiv:2502.07752)](https://arxiv.org/abs/2502.07752)（ICLR 2026）—
  structured Fisher approximation。Alice は Adam 比 2x 収束を主張。repo の P2 名称对照を確認。
  **【2026-10-25 確認済み】**: Gong, Scetbon, Chao Ma, Meeds。
  optimizer 設計を **Fisher information matrix の構造付き近似**として統一定式化し、
  既存の効率 optimizer を構造仮定の違いとして整理。
  **RACS = Row and Column Scaled SGD**（row/col 構造で SGD を scale、SGD 並みの memory）、
  **Alice = 低次元部分空間で構造を保持**（memory 低減、Adam 比 2x 収束）。
  LLaMA pretraining 1B まで検証。
  repo の「Muon 系 row 構造」と同じ FIM 視点の系譜上にあり、
  row/col scale が NorMuon・Muon+ と同型の操作である点が接続点。
- [AdaMuon (arXiv:2507.11005)](https://arxiv.org/abs/2507.11005)（v3 2025-12）—
  element-wise second moment を Muon へ統合。
  **【2026-10-25 確認済み】**: (a) orthogonalized update への element-wise second moment、
  (b) orthogonalization 前の momentum への sign 変換、(c) Adam と update magnitude を
  揃える RMS-aligned rescaling（既存 LR schedule を再利用可能）の3構成。
  「Adam 比 40% 効率改善」を主張。
- [Mousse (arXiv:2603.09697)](https://arxiv.org/abs/2603.09697)（2026-03）
  — Zhang, Xing, Huang, Lv, Zhou, Qiu, Guo, Chen（上海 AI Lab 系）。
  **【2026-10-25 確認済み】**: Muon の Stiefel manifold 制約が
  「等方的な landscape」を暗黙仮定し、heavy-tail な curvature spectrum と
  不整合である点を指摘。**Shampoo 系 Kronecker 因子統計で whitened 座標系を
  作り、その中で spectral steepest descent（異方的 trust region）**を解く。
  更新は whitened gradient の polar decomposition。160M〜800M で Muon 比
  約12% step 削減、overhead 小。
  「NS5 を hung する」HexaTrain HVX 構成への置換候補として、
  NS5 自体を替える系統（Polar Express lane と同列）の代表例。
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
  → **【2026-10-25 abstract 再確認】**: Alice は「general な optimizer 構造への
  low-rank extension framework」の適用例として提示され、**Adam 比 2x 超の収束**を
  LLaMA ≤1B で達成（memory overhead は小）。ただし **subspace の推定方法・
  次元の具体値は abstract に無く本文が必要**。64x64 行列群への適合性の判断は
  保留（64x64 は既に小さく、subspace 化の利益は state 削減よりも
  「Adam m+v を捨てられる」構造転換側にあると見るのが自然）。

### C-5. Polar Express（repo 参照の直接確認、2026-10-25）

- [arXiv:2505.16932](https://arxiv.org/abs/2505.16932) / [AlphaXiv](https://www.alphaxiv.org/abs/2505.16932)
  — Amsel, Persson, Musco, Gower（NYU Courant 系）、2025-05（v5 2026-05）。repo 参照済み。
- **要約（確認要点）**: Newton–Schulz と同じ matmul-only 構成を保ちつつ、
  **各 iteration の多項式更新則を minimax 最適化で適応決定**
  （Chen & Chow、Nakatsukasa & Freund 系）。固定係数 NS と違い
  early phase・漸近の両方で worst-case optimal に収束するため、
  同一 accuracy に対し必要 iteration 数が減る。**有限精度演算（bfloat16 級）での
  解析もあり**、GPT-2 / 1〜10B token で一貫した loss 改善。
- **HexaTrain との関連**: Phase 5 の「Muon Orthogonalization Accuracy lane」
  （NS5 固定 / Polar Express 系係数 / NS8 相当の同一 budget 比較）の一方の pole。
  重要な含意は「**係数適応化で iteration 数を削れる**」ことで、
  HVX FP32 では NS5 より少ない matmul で同等精度が狙える可能性があり、
  Muon update の wall time 直接低減候補になる。
  また QNN HTP NS 再試行の条件（backend precision gate）を満たす別経路としても
  記録価値がある。
- **次に読む**: minimax 係数の計算 cost（offline 事前計算で済むか、
  毎 step 必要か）。
  → **【2026-10-25 本文確認済み】**: 係数は **offline で1回だけ計算し保存**する。
  依存変数は singular value bounds (ℓ, u)・多項式次数 d・反復数 T のみで
  **行列次元や per-step data に依存しない**（全入力で再利用）。
  solver は SDP ではなく **Remez 型 equioscillation を 4x4 連立一次の反復解に
  縮約したもの**（3次は closed-form、5次は `optimal_quintic` が
  `np.linalg.solve` を反復）。**公開 code にハードコードされた係数表**
  （`coeffs_list`、(a,b,c) 三組の列）が同梱され、online 段は低精度
  （bf16）で実行可能、最終係数以降は NS 型 tuple (1.875, -1.25, 0.375) の
  反復に落着する。推奨設定は d=5, T=5〜6, ℓ1=1e-3。
  → HexaTrain 実装上の含意: **NS5 の係数を Polar Express 表に差し替える変更は
  「定数の差し替え」だけで per-step cost はゼロ**。T=5 が NS5 と直接比較可能な
  budget であり、Phase 5 の係数 arm は実装上最も安い部類。

### C-5b. NAMO / NAMO-D — Adam Improves Muon（2026-10-25 追加・AlphaXiv 推薦経由）

- [arXiv:2602.17080](https://arxiv.org/abs/2602.17080) / [AlphaXiv](https://www.alphaxiv.org/abs/2602.17080)
  — Minxin Zhang, Yuxuan Liu, Hayden Schaeffer（UCLA）、2026-02（引用11）。
- **要約**: orthogonalized momentum と Adam 型 noise adaptation の
  **原理的統合**を標榜する2 optimizer。
  **NAMO** は orthogonalized update へ element-wise 適応を付けず
  **単一の adaptive stepsize（scalar）**でスケール
  （直交性を保存したまま Muon を改善、追加 cost は無視できる程度）。
  **NAMO-D** は clamping 付き diagonal 行列を右から掛け
  neuron-wise 適応を入れる（block-diagonal Hessian 構造に整合）。
  決定的設定で最適収束 rate、確率的設定で noise level に適応する保証。
  GPT-2 pretraining で NAMO-D > NAMO > Muon / AdamW。
- **HexaTrain との関連**: 適応 granularity spectrum を埋める中間点。
  「**element-wise 適応は直交性を壊す**」という設計判断が
  AdaMuon（element-wise）との対立軸を明示的にしており、
  HexaTrain の row 制御 lane に **scalar adaptive stepsize arm**（最安）を
  加える根拠になる。spectrum は
  Muon（なし）→ Muon+（stateless norm）→ **NAMO（scalar adaptive）** →
  NorMuon/NAMO-D（neuron-wise）→ AdaMuon（element-wise）へ拡張。

### C-8. Why Muon Outperforms Adam — curvature perspective（2026-10-25 追加）

- [arXiv:2606.04662](https://arxiv.org/abs/2606.04662) / [AlphaXiv](https://www.alphaxiv.org/abs/2606.04662)
  — Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, Zhuoran Yang、2026-06。
- **要約**: loss の 2次 Taylor 展開由来の curvature penalty を
  **update norm の2乗 + Normalized Directional Sharpness（NDS）** に分解し、
  同一 validation loss で Adam と Muon を比較。両者の update norm は同程度だが
  **Muon は NDS が一貫して小さい**（より平坦で well-conditioned な領域へ収束）。
  Zipf-PCFG data での制御実験では data imbalance が Muon の NDS 優位を増幅。
  within/cross-layer 分解では中期〜後期の NDS 優位が within-layer curvature の
  小ささに由来し、異質 curvature の二次問題の理論では
  「update energy を curvature group 間で均す」ことで平均 NDS が GD より小さくなる
  ことを証明。
- **HexaTrain との関連**: Muon が Adam に勝つ理由の理論的説明が
  「row norm 制御（Muown lane）」とは独立の **curvature 軸**で与えられた。
  Phase 4 の比較で bpb 差だけでなく
  **NDS 相当量（update 方向の曲率）を telemetry に入れる**と、
  Muown / Muon+ / NAMO のどの arm が効いているかの帰属分析ができる。
  また C-3（condition number 不変の収束）と合わせ、
  Muon 系の理論面が「spectral」「curvature」の2軸で揃った。

### C-9. Spectral Allocation / SAMuon — Muon 改善の第4系統（2026-10-25 確認済み）

- [Spectral Allocation: Why Muon Outperforms Adam, and How to Improve Muon
  （arXiv:2608.25990）](https://arxiv.org/abs/2608.25990)
  — Xiaodong Wu, Wenyi Yu, Chao Zhang, Philip Woodland、2026-08。
- **要約**: momentum buffer を特異方向へ分解し方向ごとの loss-optimal step size を
  out-of-sample 推定すると、**Edge-of-Stability で動く volatile な head 方向と
  それよりはるかに大きい step を許容する tolerant な bulk 方向**という
  安定した anisotropic profile が現れる。Muon > Adam > SGD の順位は
  この profile への step size の整合度で説明でき、
  **Muon の uniform scaling は bulk の step を使い損ねている**という限界を指摘。
  修正案 **SAMuon** は head は Muon scale のまま bulk を static spectral prior で増幅。
  full 版は low-rank randomized SVD、**SAMuon-lite は rank-one power iteration の
  2段近似で persistent state 追加ゼロ・FLOPs 追加ほぼゼロ**。
  modded-nanogpt 124M〜1B で tuned AdamW / Muon（Scion）を全設定で上回り、
  **同 loss に 13.3〜24.0% 少ない token**。
- **HexaTrain との関連**: row 制御（Muown / Muon+ / NorMuon）とは別軸の
  **spectral axis 方向の Muon 改善**で、しかも SAMuon-lite は
  「state 追加ゼロ・1 power iteration」なので HVX Muon への追加コストが
  Muon+ 同様に最小級。Phase 4 / 5 の Orthogonalization Accuracy lane に
  「NS 後の spectral reweighting」arm として加える選択肢が新たにできる。
  Muon の uniform scaling の限界という指摘は、
  HexaTrain 実測で Muon が Adam を -0.066 bpb 下回った実績の上に
  さらに上ががあるかを測る問いを立てる根拠になる。

### C-10. Dion / Dion2 / Dion3 — orthogonalization cost 系譜（2026-10-25 追加）

- [Dion: Distributed Orthonormalized Updates（arXiv:2504.05295）](https://arxiv.org/abs/2504.05295)
  / code: `microsoft/dion` — Ahn, Xu, Abreu ほか（Microsoft）、2025-04（v3 2025-09、引用79）。
  **【2026-10-25 確認済み】**: Newton–Schulz を **momentum buffer 上の amortized
  power iteration + rank-fraction parameter（error feedback 付き）の低 rank 更新**へ置換。
  full-matrix 再構成を避け weight sharding と自然に整合。
  160M〜3B で orthonormal update の利点を保ったまま
  **大規模での wall-clock を顕著に削減**。NorMuon の比較でも Dion は
  Adam 比 10.43%（Muon と同値）の step 削減を達成（C-2 参照）。
- [Dion2: A Simple Method to Shrink Matrix in Muon（arXiv:2512.16928）](https://arxiv.org/abs/2512.16928)
  / Ahn, Amsel, Langford（Microsoft）、2025-12。
  **【2026-10-25 abstract 確認済み】**: Muon の orthonormalization の超線形コストに対し、
  **毎反復で行または列の一部 fraction を選び、その部分だけを orthonormalize する**
  最小構成の行列縮小法。update を sparse 化し compute / communication を削減。
  Dion 本体の rank-fraction（momentum buffer 上 power iteration）と異なり
  「選んだ行列だけを直接 orthonormalize する」点が素朴。abstract に実験数値なし
  （理論・手法論文。実験は Dion3 側にあると推測）。
- [Dion3: Full-Stack Orthogonal Updates（arXiv:2608.11612）](https://www.alphaxiv.org/abs/2608.11612)
  （2026-08、37頁）。**【2026-10-25 abstract 確認済み】**: Muon の直交化 overhead を
  stack の全層で攻める3+1構成:
  (1) **Gram Newton–Schulz**（対称性を利用し orthogonalization の FLOP を削減）,
  (2) 対称性を活かす **CuteDSL kernel**, (3) sharded weight の通信を減らす
  **megabatching**, (4) 毎 step momentum 行列の **行 fraction のみ orthogonalize**
  する修正 update rule（Dion2 の発想を Dion 本体へ統合）。
  主張: **Muon と同等以上の loss を達成しつつ optimizer step time を最大 6x 削減**。
  `dion` package で drop-in Muon replacement として提供。
- **HexaTrain との関連**: **HVX Muon の optimizer wall（約55〜59 ms）に対する
  最も直接的な低減候補群。** HexaTrain の 64x64 行列は小さいので NS5 の
  絶対コスト自体は大規模系より小さいが、Dion 系の「power iteration を
  amortize する」発想は NS5 反復を毎 step 完走する現行実装と代替関係にあり、
  Phase 5 の Orthogonalization Accuracy lane に
  「低 rank 近似で NS 反復数を削る」arm を立てる根拠になる。
  error feedback 付き rank-fraction は state 増分が小さく HVX 実装とも相性が良い。
  また Dion3 の Gram Newton–Schulz（対称行列の対称性を FLOP 削減に使う）は、
  HexaTrain の 64x64 NS5 実装にも作用面だけなら **反復あたり演算量を削れる
  無変更の micro 最適化候補**（ correctness は不変）。
  Dion2 の「行列を shrink する理論」単体は 64x64 では効かない（既に最小規模）。

### C-11. SOAP / Muon 大規模実証 — NVIDIA Megatron 系（2026-10-25 追加・NorMuon 関連経由）

- [SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales（arXiv:2607.20548）](https://arxiv.org/abs/2607.20548)
  — Khona ほか（NVIDIA / Megatron 系）、2026-07、code:
  `NVIDIA-NeMo/Emerging-Optimizers`。**【2026-10-25 abstract 確認済み】**:
  multi-billion param × trillion token × **batch 最大 100M token** で
  SOAP / Muon / AdamW を比較。LR transfer は **update-RMS matching** で統一。
  主張: SOAP と Muon がテスト全 scale で AdamW を一貫して上回り、
  **最大 batch では Muon/SOAP が安定を保つのに対し AdamW は劣化**。
  Muon の orthogonalization quality を経験的に評価。
  SOAP の大 batch 不安定を特定し、**per-step QR orthogonalization** と
  改良 preconditioning で loss spike を除去。optimizer 計算を近似しない
  layer-wise distributed 実装（Megatron-LM 対応）でメモリ/通信を隠蔽。
- **HexaTrain との関連**:
  (1) **大 batch での Muon 安定性優位**は Phase 6（LR×batch）の batch 拡張 arm に
  直接効く外部証拠（SR×batch 補償則 A-5 と同方向: batch を上げるほど
  Muon 側の相対利点が増える）。
  (2) **update-RMS matching による LR transfer** は Phase 6 の cross-optimizer
  比較（Muon vs HVX Muon vs Adam aux）で使うべき測定プロトコルの規範。
  (3) 「orthogonalization quality を経験評価する」手法自体が Phase 5
  （Orthogonalization Accuracy lane）の測定設計の参照になる。
  規模が 3桁以上違うため数値の直接移植は不可、プロトコル参照が主価値。

### C-7. MACRO — manifold constraint 理論（2026-10-25 追加・NorMuon 関連経由）

- [Demystifying Manifold Constraints in LLM Pre-training（arXiv:2605.04418）](https://arxiv.org/abs/2605.04418)
  — An, Li, Goldfarb, Ma、2026-05。
  **【2026-10-25 abstract 確認済み】**: Muon 系の「orthogonalization + row-wise
  normalization」を **weight 行列の manifold 正規化**（spectral norm ball 系）と
  みなして学習動力学を解析（activation scale / 回転動力学 /
  **update-to-weight ratio** の3観点）。Riemannian spectral steepest descent
  optimizer **MACRO** と radius 選択原理を提案。主要主張:
  (1) **RMSNorm と manifold 制約は重複した役割**を果たす（相互代替）,
  (2) manifold 制約下では **weight decay を完全に除去できる**,
  (3) update-to-weight ratio 制御で **update cancellation を大幅緩和**,
  (4) **MACRO は低精度計算に頑健**で標準 LLM pre-training の既存手法と競合。
- **HexaTrain との関連**: (4) が最重要 — manifold 制約 optimizer の
  **低精度頑健性の理論的裏付け**は HMX 低精度 Muon（Phase 12 / R2 系）と直結する。
  Muon の orthogonalization 自体が既に spectral ball 制約の離散近似だとする視点は、
  「HVX FP32 Muon の役割をどう数式で位置づけるか」の理論参照になる。
  (1)(2) は model 構造（RMSNorm 削除・WD 削除）の削減余地を示すが、
  758k param model で RMSNorm を外す効果は unverified であり、
  Phase 6（LR×batch×optimizer）で update-to-weight ratio を計測している場合の
  解釈補強に留める。MACRO 本体の採用は Riemannian 演算が HVX に乗るか未知。
- 次に読むこと: radius 選択原理の数式（本文未読）。NuMuon（下記索引）との関係。

### C-6. 索引

- **【2026-10-25 abstract 確認済み】MiMuon（arXiv:2605.19619）** —
  **Muon と momentum SGD の hybrid**: orthogonalization を「cautiously」
  （選択的に）だけ適用する。理論: Muon の generalization error は
  `O(1/(Nκ^T))`（κ = gradient 推定の最小 singular value gap、一般に小さい）
  に対し MiMuon は `O(1/N)`、収束 rate は Muon 同値。
  Qwen3-0.6B / YOLO26m で検証。
  **orthogonalization cost を削る第4の軸**（Dion=amortize、Dion2=row fraction、
  Gram NS=FLOP 削減、MiMuon=**適用を間引く**）として、Phase 5 の
  「NS を毎 step 完走しない」arm 群に加わる。abstract 確認済み・本文未読。

- **【2026-10-25 abstract 確認済み】Beyond Adam: SOAP and Muon for Label-Efficient
  Training（arXiv:2607.02499）** — 対象は **LLM ではなく MLIP**（NequIP/Allegro）だが、
  「Muon の利点は domain 依存で、SOAP / SOAP-Muon の方が一貫して頑健」という
  反証データ point。Muon 汎用性の限界を測る際の対照群。abstract 確認済み・本文未読。

- **【2026-10-25 abstract 確認済み】Pion（arXiv:2605.12492）** —
  **multiplicative optimizer**: weight 行列を左右の orthogonal 変換で更新し
  singular value spectrum（= spectral norm）を学習中一定に保つ
  （Adam/Muon を「additive」と characterize して対比）。
  LLM pre-training / finetuning で「stable and competitive」と主張
  （spherelab.ai/pion、引用8）。scale 制御問題（Muown / MACRO / NorMuon と同じ土台）への
  「additive をやめて spectrum を凍結する」という極端な解。HVX では
  64x64 正方行列なら orthogonal 変換2回/step で実装可能だが weight 更新の意味論が変わり、
  現行 Muon 系の検証資産と非互換。思想的参照止まり。abstract 確認済み・本文未読。

- **【2026-10-25 abstract 確認済み】NuMuon（arXiv:2603.03597）** — update direction への
  **nuclear norm 制約**で weight を低 rank 化しやすくし、標準 LLM 圧縮 pipeline 後の
  品質を billion scale で改善（Muon の full-rank update でも学習 weight は既に
  低 rank 構造を持ち圧縮しやすい、という観察が前提）。
  on-device 学習→on-device 配備の HexaTrain では「学習した weight が圧縮・低 bit 化に
  耐えるか」は Phase 12 と同根の関心事。abstract 確認済み・本文未読。

- **【2026-10-25 確認済み】ROOT（arXiv:2511.20626）** — dimension-robust な
  orthogonalization で安定性問題に対処（引用13）。行列の形状差による
  NS 挙動差は HexaTrain の 64x64 一様形状では比較的問題にならないが、
  形状探索（Phase 8）で非正方行列が増えた場合の参照。
- **【2026-10-25 確認済み】TEON（arXiv:2601.23261）** — layer-wise Muon を
  tensor 単位の orthonormalization へ拡張（MUON+ と同 group、2026-01）。
  HexaTrain では 4D tensor（head 次元含む）をどう orthogonalize するかの
  設計問いに対応。abstract 確認済み・本文未読。
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
- **【2026-10-25 直接確認済み】Muon is Scalable（Moonshot / Kimi）**:
  [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) — Liu, Su, Yao ほか、2025-02。
  Muon の大規模化に必要だったのは **(1) weight decay 追加、(2) per-parameter update
  scaling の調整（RMS 整列）** の2点だけで、hyperparameter 再調整なしで
  out-of-box に動作。3B/16B (A3B) MoE を 5.7T token で学習し、
  scaling law 実験で AdamW 比 約2x の compute efficiency。
  HexaTrain との関連: **WD 追加と RMS 整列が大規模化の必須条件**という観察は、
  repo の「Original Muon+WD control」（Phase 4 比較 arm）の理論的位置づけの裏付け。
  Muon+WD arm は大規模側で必要とされた修正そのものであり、HexaTrain 規模では
  「効果があるか」ではなく「不要で済むか」を検証する arm と読める。

- [Awesome-Muon](https://github.com/zhanghualin0/Awesome-Muon)（2026-06 更新）—
  Muon 系の網羅 list。以後の追跡はこれを定期確認するのが最も安い。
- repo 参照済み: Muown 2本（2605.10797 / 2606.23637）、Muon is Scalable（2502.16982）、
  Polar Express（2505.16932）、Can Muon Fine-tune Adam-Pretrained（2605.10468）。
  本調査で新規に重要と判断したのは MUON+ / NorMuon / スペクトル理論の3点。

### C-12. Awesome-Muon 差分整理 — 未収録の Muon 系論文マップ（2026-10-25 追加）

[Awesome-Muon](https://github.com/zhanghualin0/Awesome-Muon)（キュレーション list）と
本調査の収録差分を整理する。**以下の ★ は list 経由の確認で abstract 未読**
（要確認）を明示する。

- **【2026-10-25 abstract 確認済み】Small Batch Size Training for LMs
  （arXiv:2507.07101、NeurIPS 2025）** — 小 batch（**batch 1 まで**）で
  momentum 無しの vanilla SGD でも LM 学習が安定、**gradient accumulation は
  「step を effective batch に交換するだけ」で per-FLOP が等しいか悪くなるため、
  multi-device/replica でない限り非推奨**。移行則は Adam の second moment の
  **half-life を token 固定**で持つ（step 固定でない）。
  small batch + 小 state optimizer が LoRA 相当の memory で full fine-tuning に匹敵。
  → **Phase 6 の accumulation 1/2/4/8 掃引への反証仮説**: accumulation は
  HexaTrain の単一端末構成では「無駄」な可能性があり、
  「accumulation なし + aux Adam の half-life を token 基準で再調整」を
  対照 arm として立てる根拠。B8 固定の現行設計自体は本論文の推奨側に立つ。
- ★ **Phase 5（orthogonalization 削減・近似の理論）方向**:
  - **【2026-10-25 abstract 確認済み】MuonBP: Block-Periodic Orthogonalization
    （2510.16981、引用34）** — 動機は分散時の直交化通信 overhead（AdamW 比
    5-10% の throughput hit）。**各 device の shard で独立に orthogonalize し、
    定期的に full orthogonalization を挟む**。理論は
    **blockwise step と full step で別の step size を使うこと**を要求し、
    収束保証付き（baseline Muon と競合する iteration complexity）。
    8B × 8-way TP で throughput +8%、性能劣化なし。
    → 単一端末の HexaTrain では通信動機は無いが、**計算動機の読み替えが可能**:
    64x64 を 2x2 の 32x32 block に分割して NS を回すと block 毎の cost は 1/8、
    4 block で計 1/2。定期 full orthogonalization + 2 LR が理論条件。
    Phase 5 の「NS を毎 step 完走しない」arm 群の第5候補として正式化。
  - **【2026-10-25 abstract 確認済み】Drop-Muon: Update Less, Converge Faster
    （2510.02239、KAUST / Richtárik group）** — **各 step で layer の部分集合だけを
    更新する randomized progressive training**（全 layer 更新が最適でないことを
    layer-wise smoothness の理論で示し、stochastic/non-smooth 域の収束保証は初）。
    CNN で full-network Muon 同 accuracy を最大 1.4x 速い wall-clock で達成。
    → **optimizer cost 削減の第6軸**（update 自体を間引く — forward/backward は
    変わらず optimizer wall のみ削減）。HexaTrain の HVX Muon wall（約55-59 ms）に
    対し「L19 のうち半分だけ更新」のような schedule が理論保証付きで存在する。
    ただし **QNN graph は静的**なので実装は「未更新 layer の update を 0 mask」
    （計算は節約できない）か graph 分岐が必要で、実装コストの壁は大きい。
    Phase 5 arm として理論参照に置く。
  - **【2026-10-25 abstract 確認済み】Beyond the Ideal: Analyzing the Inexact Muon
    Update（2510.19933）** — 実用 Muon の**不正確な orthogonalization** の最初の理論解析
    （LMO + 加法誤差 model）。劣化を oracle 誤差の関数として明示化し、
    **「oracle 精度を下げるなら step size を小さく・momentum を大きくする共調整が本質」**
    という coupling を導出（NanoGPT で LR が精度に応じてシフトすることを実測確認）。
    → **Phase 5 の「低 rank 近似で NS 反復数を削る」arm には本論文の coupling が
    実験設計の必須条件**: 反復数を下げた arm は LR↓/momentum↑ の再調整をセットで
    行わないと、近似誤差と LR 未調整を区別できず A/B が無効化される
    （numerical-evidence の単一要因原則と整合）。
  - **【2026-10-25 abstract 確認済み】Low-rank Orthogonalization（2509.11983）** —
    学習中の gradient が近似的に低 rank であることを利用し、**低 rank 内で
    orthogonalization を完結**させる low-rank MSGD / low-rank Muon を提案。
    GPT-2 / LLaMA pretraining で vanilla Muon に概ね同等以上
    （大規模ほど利得が明確、v2 で主張を控えめに修正済み）。
    iteration complexity の理論 bound 付き（heavy-tailed noise 下）。
    code: `dengzhanwang/Low-rank-Muon`。
    → Phase 5 の「低 rank 近似で NS を削る」arm の**直接の実装系譜**
    （Dion=amortized power iteration / 本論文=gradient 低 rank 性の直接利用）。
  - **【2026-10-25 abstract 確認済み】MuonEq: Lightweight Equilibration
    （2603.28254）** — **NS 直交化の前に momentum 行列を row/column 正規化**する
    軽量 equilibration（R variant を hidden 行列の default とする）。
    「whitening の zero 次 surrogate」として finite-step NS が見る
    spectrum（stable rank / condition number）を改善。理論は Muon の
    `O(T^-1/4)` 保証を保持し **finite-step NS5 まで明示的な inexactness 定数付きで拡張**。
    LLaMA2/C4 の 130M/350M/1B で Muon に一貫して勝つ。
    → **row 正規化の「位置」が3つ揃った**: Muown = weight 側 parameterization、
    **Muon+ = NS 後 post-normalization（C-1）、MuonEq = NS 前 pre-normalization**。
    Phase 4（Muown lane）の arm 設計は「位置」を独立要因にでき、
    MuonEq は state ゼロ・1演算系で Muon+ と同コスト帯。
    NS5 への inexactness 定数付き保証は 2510.19933 の coupling 理論と噛み合う。
  - **【2026-10-25 abstract 確認済み】CANS: Chebyshev-type Polynomials による NS 加速
    （2506.10935、math.NA）** — 固定係数の Newton–Schulz に対し
    **Chebyshev 交互定理で 3 次の最適係数を理論導出、高次は Remez 法で計算**。
    matrix multiply のみで制御付き近似直交化（Muon 直交化と Stiefel 多様体の
    retraction を応用先に明記）。
    → C-5 Polar Express（minimax 最適係数）と同 family の別解。
    Phase 5 の「NS5 の係数を差し替えて同反復数で精度を上げる/同精度で反復数を削る」
    arm の係数源候補が Polar Express + CANS の2系統になる。
  - **【2026-10-25 abstract 確認済み】SUMO（2505.24749、NeurIPS 2025）** —
    **動的に適応する低次元部分空間内で exact SVD 直交化**を行う
    （full 空間での NS 近似を置換）。moment の condition number 依存の
    近似誤差を回避し、低 rank optimizer 群比で memory 20% 減・
    収束/安定性/性能を改善。
    → Phase 5 の別解系譜: 「近似 NS in full space」対「**exact SVD in subspace**」。
    64x64 規模なら CPU 側 Jacobi SVD は現実的なので、
    HVX/CPU 構成の HexaTrain では NS5 以外の直交化器の選択肢として
    理論参照になる（HVX に高速 SVD がない点は実装壁）。
  - **【2026-10-25 abstract 確認済み・初版記載を訂正】AMUSE: Anytime Muon
    （2605.22432）** — 「anytime」は直交化 cost でなく **Schedule-Free 系の
    anytime 学習**の意味。Muon の更新 geometry に Schedule-Free averaging を
    時変補間係数で統合（序盤は高速 Muon 列、後半は平均列へ漸移、
    LR schedule 不要）。river-valley loss 解析で「orthogonalization は
    bulk 進行を速める一方 dominant 方向の noise を増幅して振動する」
    という Muon の弱点整理。vision + LLM pretraining で
    Schedule-Free AdamW / Muon の Pareto 改善。
    → HexaTrain は checkpoint 保存型の学習で schedule-free 化の要請は薄いが、
    「Muon の振動 = dominant 方向 noise 増幅」は Phase 4/5 の telemetry
    （update の dominant singular 成分の揺れ）という観測軸を与える。
  - **【2026-10-25 abstract 確認済み】HTMuon: Heavy-Tailed Spectral Correction
    （2603.10067）** — HT-SR 理論に基づき「Muon の直交化 update は heavy-tailed
    weight spectrum の出現を抑え、noise 支配方向を過大評価する」と診断し、
    **より heavy-tailed な update** を作る補正を提案。
    理論は **Schatten-q norm 制約下の最急降下**に対応（Freon の Schatten 系と同 family）。
    LLaMA/C4 pretraining で perplexity 最大 0.98 改善、既存 Muon variant への
    plug-in として動作。
    → **Kaon（2605.11181「spectrum の形は無関係」）と正反対の主張**が同時期に併存。
    Phase 5 の spectrum 設計 arm は「flat（Muon 標準）vs heavy-tailed（HTMuon）vs
    random（Kaon）」の 3way が文献上の対立構図で、小規模 HexaTrain での優劣は未確定。
- ★ **Phase 12（Muon 低 bit 化）方向**: A-10（Full-Stack FP4）と同主題の群。
  - **【2026-10-25 abstract 確認済み】MuonQ: Directional Fidelity Optimization
    （2605.11396、COLM 2026）** — **Muon の state は Adam より量子化に敏感**
    （orthogonalization は方向情報のみを保持するため、singular vector 方向の
    小さな量子化誤差が update で増幅される）。対処3技術:
    量子化前 normalization / power iteration で支配的 singular 成分を分離量子化
    （誤差を方向でなく magnitude に向かわせる）/ **μ-law companding**
    （密な momentum 値域に高分解能を割く）。**state 4-bit が安定**、
    memory 7.3x 減で loss ほぼ回復（COLM 2026、code: `YupengSu/MuonQ`）。
    → HVX Muon の momentum を低 bit 保存する場合の**方向保護が必須**という
    設計制約。μ-law companding は HVX/CPU で安価。A-10 は「計算を FP4 化」、
    MuonQ は「state を 4-bit 化」で対策の階層が違う。
  - **【2026-10-25 abstract 確認済み】Effective Quantization of Muon Optimizer States
    （2509.23106）** — state の **blockwise 8-bit** 量子化で optimizer state を
    62% 削減。主張: 「Muon の update 機構は**単純な linear 量子化と相性が良い**
    （AdamW が要求する動的 scale 設計が不要）」+ 量子化 noise への頑健性の理論解析。
    2.7B までの Chinchilla-optimal pretraining + fine-tuning で parity。
    → **MuonQ（4-bit は方向保護が必須）との張り付き合いに注意**:
    8-bit linear なら方向 fidelity が保てるが、4-bit に落とすと MuonQ の
    normalization / 分離 / μ-law が要る、という2段階の整合的読み。
    HVX Muon state の低 bit 化（P2）は「まず 8-bit blockwise」から。
  - **Achieving low-bit Muon through subspace preservation and grid quantization
    （ICLR 2026 Poster、arXiv ID 未確認）**。
  - **【2026-10-25 abstract 確認済み】Error Feedback for Muon and Friends
    （2510.00643、KAUST、ICLR 2026、引用16）** — 圧縮（= biased compressor）下の
    Muon 型 orthogonalized update に対する**最初の厳密な error feedback 理論**。
    EF で biased 圧縮でも収束保証が復元することを示し NanoGPT-124M 等で検証。
    注意: 後続の sign 圧縮解析（2607.29674）は「**直交化後の出力を sign 圧縮した
    場合は EF でも救済されない**」という限界を報告。
    → Phase 12 で HVX Muon の momentum/update を低 bit 化する際の理論支柱
    （量子化 = biased 圧縮、EF で補償 — Dion の rank-fraction EF と同じ発想）。
    ただし「直交化後 output の aggressive 圧縮」は EF でも不可という boundary がある。
- ★ **理論・批判系（C-3/C-8 の補強と反証）**:
  - **【2026-10-25 abstract 確認済み】Muon is Not That Special（2605.11181）** —
    singular value を **random noise に置換する Kaon** が Muon と同等性能で
    収束保証も保持（「正確な幾何構造の遵守は実質無関係」）。
    Schatten (quasi-)norm 系 **Freon**（QDWH 反復）は SGD↔Muon を補間し、
    GPT-2 の最適 Schatten 参数は quasi-norm 域（unitarily invariant LMO では表現不可）。
    説明: 性能を支配するのは **alignment と descent potential の局所量**であり、
    Muon は「大域幾何の追跡でなく step-size 最適性の保証」で成功している。
    → C-3（preconditioning 利得理論）への**反証仮説**。Phase 5 の NS 精度 A/B の
    解釈を「直交性誤差そのもの」から「方向別 effective step size の制御」へ
    補正する材料（2510.19933 の LR/ momentum coupling と整合する読み）。
  - **Isotropic Curvature Model / Is Gradient Orthogonalization Optimal?
    （2511.00674）** — C-8（NDS 曲率）と同型の問いの別枠。
  - **【2026-10-25 abstract 確認済み】Muon in Associative Memory Learning
    （2602.05725、ICML 2026、53頁）** — 線形連想 memory + 階層 frequency spectrum の
    model で Muon の学習動力学を解析。**GD は低周波成分が bottleneck の
    不均一学習、Muon は frequency 間の進行を均し noiseless で GD に指数的加速**。
    label noise + power-law spectrum 下で **Muon の scaling law を導出し
    GD 上回る scaling 効率**を示す。解釈: Muon = adaptive task alignment と
    block-symmetric gradient 構造からの implicit matrix preconditioner。
    → 「data spectrum の偏りが大きいほど Muon の相対利得が大きい」メカニズムで、
    Byte-BPE V1024 の長尾 token 分布を持つ HexaTrain の Muon 利得の
    理論的な一説明。noise 下の scaling 効率は C-11（大 batch 実証）と
    補完する理論面。
  - **【2026-10-25 abstract 確認済み】Convergence Bound and Critical Batch Size of
    Muon（2507.01598）** — 実用4構成（±Nesterov、±WD）の Muon 収束証明。
    WD ありで「bounded-gradient 仮定なしに parameter/gradient norm が
    almost-sure 有界」、WD 係数と LR の相互作用も整理。
    **critical batch size の下界は定性的**（momentum β と WD λ が scaling を支配、
    絶対値予測は不可）で画像+LM で検証。
    → Phase 6 の batch/accumulation 掃引では **β と WD を batch と独立に固定しない**
    （critical batch size が β/λ 依存で動くための理論的注意）。
  - **【2026-10-25 abstract 確認済み】Delving into Muon and Beyond（2602.04669）** —
    Muon を `U Σ^p V′` family の **p=0 端点**として統一的に解析（momentum 系と
    RMS-normalized gradient 系の双方に適用）。p=1/2, 1/4, 1 を検証し
    **SVD を避ける coupled Newton iteration** を開発。
    主張: RMS-normalized の方が安定、spectral 圧縮は first-moment を安定化するが、
    **Muon（p=0）は Adam に一貫して勝つわけではない**。
    結論は「Muon は有効な spectral normalization の一形態で、万能ではない」。
    Kaon/Freon（上記）と同じ反万能系。code: `Ocram7/BeyondMuon`。
    （NorMuon 関連リストの残項目「SOAP-Muon-Beyond」は本論文では解消されず、
    C-11 の NVIDIA 論文が該当項目と判断するのが妥当。）
- ★ **適用範囲の設計問い**: **【2026-10-25 abstract 確認済み】Symmetry-Compatible
  Principle for Optimizer Design（2605.18106、NeurIPS 2026）** —
  「update 規則は対象 weight block の対称群に対して equivariant であるべき」原理。
  Muon / spectral / Scion は一般行列層の bi-orthogonal equivariance として統一説明。
  **embedding / LM head は permutation + shared-shift 対称**であり
  full orthogonalization は不適、one-sided spectral / row-norm / hybrid /
  centered row-norm / left-spectral 等の対称適合 update を導出
  （SwiGLU MLP・MoE router も個別対称）。dense/sparse MoE LM（0.6B〜1B 級）で
  val loss 改善・語彙 logit 成長の制御・router 安定化を報告。
  → **HexaTrain の「Muon を2次元隠れ行列のみ、embedding/output は aux Adam」分割の
  理論的正当化**。さらに「centered row-norm」は J-3 OEC（μ-centering）、
  「row-norm」は Muown（C-6）と family が一致し、
  row 制御系譜（Phase 4）の位置づけを補強する。
- **HexaTrain との関連（全体）**: C-12 は収録差分の索引であり、
  個別精読は Phase 5/6/12 の各 lane 開始時に★項の abstract 確認から始める。

### C-13. Bi-Maxwell — Muon の 2 時定数 momentum 理論（2026-10-25 新着掃描）

- [A Physical Response-and-Memory Model for Muon（arXiv:2608.22994）](https://arxiv.org/abs/2608.22994)
  — 2026-08。
  **【2026-10-25 abstract 確認済み】**: weight 行列を「memory を持つ応答媒質」として
  モデル化。Muon の semi-orthogonalized update を「出力側 safety budget 下の
  最大散逸応答」、momentum を「媒質に溜まる内部応力」と解釈し、
  応力緩和に **fast + slow の2時定数 memory kernel（Bi-Maxwell）**を提案
  （Muon の単一時定数 momentum の置換）。
  予測: 「gradient 方向は学習初期に速く後期に遅く変わる → 最適 memory 長は
  学習 stage とともに伸びる」— **read-only probe による memory 長 proxy の計測**
  （8 独立 trajectory）で検証。kernel の置換だけで
  公開 LLM optimizer benchmark で target loss までの step 数を削減。
- **HexaTrain との関連**:
  (1) **J-5 ANVIL II の twin momentum（fast + slow rail）に理論的裏付けを与える**
  — community の発見と独立導出の理論が一致している（本論文は PowerCool を
  直接扱わないが同 benchmark 系）。
  (2) HVX Muon の momentum は単一 buffer。**2本目（slow rail）の state 追加は
  64x64 buffer 1個/param で、CPU/HVX 実装の増分は小さい**。
  Phase 4/6 の momentum schedule 設計に「2時定数化」arm を追加する根拠。
  (3) **memory 長 proxy probe** は HexaTrain の telemetry（update norm 系と同じ枠）に
  追加可能な診断で、Phase 0 の stress harness 計測項目の候補。
  理論の「output-side safety budget」解釈は、NS 直交化の役割を
  step-size 制御（2605.11181 Kaon/Freon と同じ読み）に寄せる第3の視点。

### クラスタCの統合所見（2026-10-25時点）

1. **「post-NS row norm 不均衡」が独立3系統（HexaTrain audit / Muon+ / NorMuon）で確認された。**
   Phase 4 は Muown 単体ではなく、**row 制御の位置（parameterization vs post-NS normalize vs
   per-neuron moment）を分離する3-4 arm 比較**へ拡張する情報的価値が高い。
   深掘り後の整理として、適応の **granularity 軸**で並べると
   「Muon（適応なし）→ Muon+（row/col L2・stateless）→ NAMO（scalar adaptive stepsize）→
   NorMuon / NAMO-D（row RMS EMA・m scalar / neuron-wise diagonal）→
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
  → **【2026-10-25 精読済み】**: primitive は5 class —
  (1) elementwise/pairwise map（residual・activation・RoPE・SwiGLU gate）、
  (2) vector load/store（row/col broadcast）、
  (3) tile load/store（residual stream・saved activation・backward 中間）、
  (4) tile 還元（部分 row/col sum を小 kernel で合流）、
  (5) stateful transform（online log-sum-exp / CE 統計）。
  forward は `GEMM→residual→RMSNorm→GEMM` を中核に、**RMSNorm の row scale を
  次の GEMM の向こう側へ代数的に移す**。backward は **Theorem 1:
  tile-local な forward epilogue は tile-local な backward epilogue を誘導する**
  （fusion が勾配生成側 GEMM へ反転する）。RMSNorm backward の行統計は
  `s = (1/d)Σcols(∇y⊙y)` で GEMM 境界を越えられる。
  吸収不可: 大域 reduction・attention・embedding。
  **LayerNorm（mean 減算あり）は明示的に扱っていない**（LLaMA 系 RMSNorm のみ）。
  HexaTrain への含意: (a) 現行 LayerNorm→RMSNorm 移行（P1）は
  品質以外に **HMX epilogue 吸収可能性を解放する**副次効果を持つ、
  (b) Theorem 1 の原則は手書き QNN backward にも適用でき
  「勾配生成 MatMul の convert 段に要素演算を付ける」設計指針になる
  （R2 の retained-ACC と同型の発想）、(c) CE の stateful transform は
  CCE（H-7）と同じ lse 統計方式で、vocab chunk 逐次処理と組み合わせられる。

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
- [BiE: Bi-Exponent Block Floating-Point（ICML 2024, PMLR 235:62978）](https://proceedings.mlr.press/v235/zou24d.html)
  — Zou ほか（CUHK）。
  **【2026-10-25 全文精読済み】**: 6-bit BFP が有望だが activation の outlier 悪化が課題、
  という動機。**block（16要素・行方向）ごとに2本の共有指数** `e_n`（|x|≤T の正常部）と
  `e_o`（|x|>T の outlier 部）を持ち、要素ごとに 1-bit type flag で所属を区別。
  threshold `T` は **tensor ごとに offline で Bayesian 最適化**
  （GP + Matérn 5/2 kernel + Expected Improvement、探索空間は |x| 分布の
  75〜95 percentile に剪定）。W4A4 で OPT-66B WikiText2 9.82（FP16 9.12、
  BFP-W4A4 は 14.16 と崩壊）。hardware は comparator tree 追加 + bit shifter で
  BFP とほぼ同 throughput（FP16 比 3.51x）。**OPT-125m〜2.7B の小規模でも
  BiE > BFP が一貫**（125m: FP16 27.50 → BiE W4A4 28.20 vs BFP 30.17）で、
  HexaTrain 規模寄りの検証もある。推論 PTQ 専用（Softmax/LayerNorm は FP16 残置）。
  HexaTrain との関連は「outlier 分離 family の block 内2指数版」として、
  RHT（変換で分散）・Adacc Z-score（channel 分離）と対になる
  「分離位置」の第3の選択肢（block 内 value 分離）。
- [F-BFQ（arXiv:2510.13401, LG-ARC@ISCA 2025）](https://arxiv.org/abs/2510.13401) —
  Haris, Cano。**【2026-10-25 確認済み】**: BFP 量子化 LLM のための edge 用
  hardware accelerator。layer ごとに異なる BFP variant が混在する前提で
  **2つの BFP variant を再構成なしで動的切替**できる matmul を実装。
  AMD Kria 上で Arm NEON CPU 比平均 1.4x、5.2 token/s。推論専用。
  HexaTrain との関連は「HTP が固定 block scale を持つ状況での代替表現研究」の
  位置づけ（HMX 直接には非適用）。
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

### D-7. Bit-Accurate MMA Model / CLFP — Microsoft MMA-Sim（2026-10-25 新着掃描）

- [Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate（arXiv:2511.10909）](https://arxiv.org/abs/2511.10909)
  / code: `microsoft/MMA-Sim` — 2025-11（v3 2026-09）。
  **【2026-10-25 abstract 確認済み】**: **closed-loop feature probing (CLFP)** と名付けた
  体系的 probing framework で、NVIDIA Volta〜RTX Blackwell 10 architecture +
  AMD CDNA1〜3 の **全 MMA 命令**の bit-accurate 数値 model を構築（世界初と主張）。
  未文書化の内部浮動小数点挙動を入出力 probe から逆推定し、
  4種の precision bottleneck と 1種の **numerical asymmetry** を特定。
  cross-vendor / cross-generation の数値差が training instability の原因になり得ると指摘。
  html 版には FP16 subnormal 防止の入力 scale-up / 出力 scale-down の扱いもある（要本文）。
- **HexaTrain との関連**: D-1（2512.07004）と**姉妹関係にある方法論論文**で、
  CLFP という名前がついた体系化は HexaTrain の
  `hmx_signed_output_oracle.c` の discriminating vector 設計に
  **名前と構成を与える**（probe → model → closed-loop 検証）。
  「numerical asymmetry」の分類は HMX の signed FP16 出力（R2）の
  正負で挙動が変わり得る問題を分類する語彙になる。
  GPU 10世代の probing 設計（どの入力分布で内部丸めを区別するか）は
  HMX oracle の vector 拡充の参照。abstract 確認済み・本文未読。

### D-8. Adaptive Block-Scaled Data Types（IF4）— block 内最大値の誤差問題（2026-10-25 新着掃描）

- [Adaptive Block-Scaled Data Types（arXiv:2603.28765）](https://arxiv.org/abs/2603.28765)
  / code: `mit-han-lab/fouroversix` — 2026-03。
  **【2026-10-25 abstract 確認済み】**: NVFP4 の誤差が **各16値 group の
  near-maximal 値に集中する**という診断から、group ごとに FP4 / INT4 を選択する
  **IF4** 形式を提案（選択は NVFP4 の scale factor の **未使用 sign bit** に格納）。
  IF3 / IF6 へ拡張、quantized training で既存 4-bit block-scaled より低 loss、
  PTQ でも高精度。IF4 MAC unit の hardware 実装可能性も示す。
- **HexaTrain との関連**: **MX（E8M0 power-of-2 固定 scale）は per-group の
  FP/INT 選択を持たない**ため HMX には直接適用不可。ただし「block 内
  near-maximal 値に誤差が集中する」診断は、HMX R2 系で signed FP16 出力が
  最大振幅近くで飽和する挙動（B-9 の per-row range guard と同じ問題群）と
  同根の分布論的な問題であり、HMX 側の saturation 解析の対照として参照。
  A-13 ScaleSearch（scale 探索）と合わせて「scale / format の選択自由度」系の
  文献群をなす。

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
4. **【D-7 追加後】bit-accurate MMA model の方法論には体系名（CLFP）と
   大規模な公開実例（MMA-Sim）が揃った。** HMX oracle の設計は
   「GPU 系の確立手法の Hexagon 版」として整理でき、
   probe 設計・model 検証の参照が D-1 / D-7 の2本になった。

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
  → **【2026-10-25 本文確認済み（HTML）】**: G2 は **value projection の直後に
  `V ⊙ σ(XW_θ)` を掛ける** variant（gate score は「現在 query でなく
  過去 key/value に対応する hidden state」由来、非線形を attention weight 総和の
  **前**に挿入 = G1 は後）。PPL は G1 5.761 < G2 5.820 < baseline 6.026 で
  **G2 は sink を除去できない**（massive activation は減るが first-token
  attention score は下がらない）。cost は gate 1投影 + 要素積で
  latency < 2%（MoE 15A2B で G2 elementwise 25M / headwise 0.2M params）。
  → **E-2 VGA との構造区分に注意**: 原文の G2 は「**value map 自体**への
  X 由来 gate」だが、VGA は「**attention 出力**への **V 由来** gate」であり
  位置と入力の両方が違う。原文の「G2 < G1」は value map 位置の否定であって
  VGA 型（出力位置 + V 入力）の否定ではない。HexaTrain の V-gate arm は
  「出力位置は G1 のまま・入力だけ V に替える」であり、
  この区別を preregistration に明記する価値がある。

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
  → **【2026-10-25 本文確認済み（HTML）】**: 既定は **stride s = L/2**
  （layer ℓ は自分 + {0, s, 2s, … で ℓ 未満} の V を source にする）。
  3B(L=48) は s=24、360M/500M(L=24) は s=12、710M(L=32) は s=16、
  平均 source 数 ~2.5/layer。**stride ablation（500M）は inverted-U**:
  half-depth が最良、first-layer-only は information 足りず僅劣、
  **quarter-depth も full-source（stride 1）も僅劣**（冗長な歴史 V が混入）。
  mixing 規則は **value-only per-head softmax 2.2115 < uniform 2.2153 < vanilla
  2.2348 < key-only 2.2354**（key だけ混ぜると vanilla より悪い）。
  → HexaTrain L19 への適用値は s=9（source ≤3〜4 層）で既存計画と一致し、
  ablation は「s=9 を中心に s=4（密）と s=18 併用（疎）を外側に置く」設計を支持。
  **mixing は value-only** を第一 arm に（key 混在は否定）。
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
  → **【2026-10-25 追記】**[Adaptive Multi-Token Prediction: How far should we go?
  （arXiv:2610.28619）](https://arxiv.org/abs/2610.28619)（2026-10）は**推論側**の
  horizon 適応（speculative decoding の draft 長を online control 問題として解き、
  「acceptance survival 確率」から期待 tokens/sec 最大の horizon を選ぶ controller、
  再訓練不要・最大 4.1x 高速化）で、「単一の最適 horizon は存在しない」ことを
  MTP 学習側（AdaMTP）と独立に支持。HexaTrain が MTP-lite を将来 spec decoding に
  使う場合の推論側補完であり、entropy telemetry（H-2 / E-5）と
  acceptance survival の相関を測る診断として接続できる。abstract 確認済み・本文未読。

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
- **【2026-10-25 abstract 確認済み】mHC: Manifold-Constrained Hyper-Connections
  （arXiv:2512.24880、DeepSeek-AI、引用~120）+ xHC（arXiv:2607.14530）** —
  Hyper-Connections（residual stream を複数 stream に広げて接続 pattern を変える拡張）が
  **identity mapping 性質を損ない、大規模で学習不安定 + memory access overhead** を
  生む問題に対し、residual 接続空間を manifold へ射影して identity 性質を復元する
  mHC framework（infrastructure 最適化込み）。大規模で性能向上と高い scalability。
  → HexaTrain 関連は薄い（multi-stream 化は activation memory を L19 分増やすため
  端末の memory 制約と衝突）が、「residual 拡張系の安定化には identity mapping の
  保持が本質」という診断は、Value Residual / Depth-Attention（E-3）系の
  「identity からの離れ方」設計の対照軸として記録。follow-up の xHC は
  LM head 前で多 stream を1本に戻す設計。abstract 確認済み・本文未読。
- **XSA（arXiv:2603.09078, repo 参照済み）の追跡状況（2026-10-25）**:
  Apple（Shuangfei Zhai）による二次資料（EmergentMind / 著者 X 投稿）の確認では、
  XSA の gain は** sequence length とともに増大**するとされる。
  T32 の HexaTrain では gain が小さい可能性が高く、repo の
  「長文脈論文の gain を外挿しない」原則と整合する方向。
  明確な follow-up 論文はまだ無し。
- **KV head 構造の潮流（2026-10-25 追記）**: 業界整理は
  MQA（最大共有・品質低下あり）→ GQA（現行主流）→ **MLA / low-rank / latent 圧縮**
  （2025-2026 の中心）。[LoRKVA（low-rank KV attention）](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity)
  は MQA/GQA の head 多様性低下を避けつつ KV を削る方向。
  [MLTA（arXiv:2505.13544）](https://arxiv.org/abs/2505.13544) は latent 時間圧縮。
  HexaTrain の Hq2/Hkv1 MQA A/B（P1）に対しては、H2 と head 数が少なく
  MQA での head 多様性低下が相対的に大きいため、**low-rank K/V を
  比較 arm に加える意義**が出る（D64 なら rank 16 など）。

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
  → **【2026-10-25 AlphaXiv AI overview 経由で追記】**: 6-step ablation で
  LaCT / ViTTT を standard linear attention まで帰着させた具体的な簡約手順を提示。
  反直観的観察として **inner-loop step 増は inner-loop fit を良くするが
  下流指標を悪化させる**（1→64 step で PSPR 低下・PPL 上昇）。
  機構は「最終層が bias-free linear だと更新が outer-product として state に
  蓄積され query が読む = learned linear attention」。
  parallel 化には static feature map + weight normalization 除去が必要で、
  推論 4.0x / 学習 1.19x。**「内部最適化の深さ」と「下流品質」が乖離する**
  可能性は HexaTrain の GDN/T Titans 系（P3）を評価する際の注意点になる。

### F-2. chunkwise parallel 系の実装基盤

- [flash-linear-attention（GitHub, fla-org）](https://github.com/fla-org/flash-linear-attention) —
  GLA / DeltaNet / GDN 等の Triton 実装を集めた共通 library。chunkwise algorithm の
  参照実装として、QNN graph へ落とすときの「演算単位の切り出し」に使える。
- GLA（arXiv:2312.06635）/ GDN（arXiv:2412.06464, repo 参照済み）の chunkwise 形式は
  「sequence を chunk C に割り、intra-chunk は MatMul、inter-chunk state は
  小さい再帰」に分解する。**この分解は HTP 向き**（MatMul 主体、state が小さい）で、
  HexaTrain の SimpleGDN 実装計画の前提と整合する。
- **次に読む**: GDN の Triton kernel 1個分の演算粒度（QNN node 数の見積り材料）。

### F-3. 索引（簡易記載・主要項目は本文確認済み）

- [Bridging KV-Cache Quantization and Linear Attention（arXiv:2610.11214）](https://arxiv.org/abs/2610.11214)
  — 2026-10。**【2026-10-25 全文精読済み】**:
  RAM-Net は **product-softmax による soft address assignment**
  （U=5 factor × d_p=4 code の離散 address 空間、Top-K=32）と
  **Gated Sparse Update による連続 slot state**（per-slot mass と write-weighted mean、
  忘却 scalar γ）の組み合わせ。Theorem 1 は
  **circular codebook + rank-2 projection の局所 regime で
  address overlap が exp の dot-product kernel を近似する**こと
  （`log κ_A ≈ (β²/2)qᵀk`、最適集中度 `β*=√(2/τ)`）を示し、
  hard-write 極限で nearest-code VQ（quantized attention）、
  分離性から recurrent slot 集約（linear attention）を回収する。
  **移行 protocol は3段階**（低rank adapter + positional alignment →
  circular soft quantization の fold-in → staged recovery
  （head NMSE → attention block → 全 model、固定 500M token budget））で、
  9 teacher（0.3B〜7B）で teacher gain の平均 87.1% を回復。
  HexaTrain との関連: **low-disruption migration protocol（P1〜P2）の
  別分野実装例**であり、(a) 移行 class を段階分けする設計、
  (b) 固定 token budget 内での staged recovery、
  (c) attention map KL での calibration、が HexaTrain の
  `exact_identity` / `weight_preserving_non_identity` 区分と対応する。
  Phase 14 の SimpleGDN migration 設計の参照として有用。
- [Native Hybrid Attention（arXiv:2510.07019）](https://arxiv.org/abs/2510.07019) /
  code: `JusenD/NHA` — Du, Hu, Zhang, Sun, Cheng、**ACL 2026**。
  **【2026-10-25 確認済み】**: linear RNN で更新する長期 KV slot + sliding window の
  短期 token を単一 softmax attention で併合。layer 積み上げ型の hybrid と違い
  **全 layer を構造的に同一に保ち、sliding window size という単一 hyperparameter で
  linear ↔ full attention を内挿**する設計。recall-intensive / commonsense で
  transformer 系 hybrid を上回り、**pretrained LLM への retrofit
  （structural hybridization）も可能**。
  HexaTrain の Phase 14（Hybrid Sequence Operator Search）に対して、
  「layer ごとの離散選択」ではなく「連続 hyperparameter で内挿する」設計軸と、
  low-disruption migration との親和性を示す例。
- [RNN as Linear Transformer（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_RNN_as_Linear_Transformer_A_Closer_Investigation_into_Representational_Potentials_CVPR_2026_paper.pdf)
  — 線形 attention vs Mamba 系の表現力比較（vision 中心）。P14 の sorted 順序決めの補助資料。
- [LeapQuant（arXiv:2609.38166）](https://arxiv.org/abs/2609.38166)
  （2026-09）— **【2026-10-25 全文精読済み】**:
  **linear attention（GDN / KDA）の recurrent state 行列**を量子化する
  training-free 手法（推論時のみ、vLLM + TileLang 実装）。
  課題は (1) **step ごとの再量子化誤差が再帰的に蓄積**する
  （per-step BF16 でも Qwen3.5-9B AIME 87.9% → 72.1%、INT8 は 7.1% に崩壊）、
  (2) state の outlier が少数 row/col に集中（中央値の最大176倍）。
  手法は (a) **per-window 量子化**（window p=16 ごとに state を量子化し直し、
  窓内は高精度 rank-1 update を buffer）、(b) **compensator tokens**
  （支配的な rank-1 outlier 構造を FP16 で保持し通常 token と同じ経路で処理）、
  (c) diagonal residual smoothing。8-bit で FP32 と完全一致（75.5% avg）、
  6-bit で 72.4%（baseline は 29.6〜31.9% に崩壊）、kernel 2.05〜3.70x。
  HexaTrain との関連: Phase 14 の GDN 導入時、
  **recurrent state の保存精度が誤差蓄積の主戦場になる**ことの存在証明。
  HexaTrain は学習時に state を HTP graph 内で保持するため per-step 更新が
  再量子化に当たるが、「窓境界でだけ低bit化・窓内は高精度」という
  per-window 原理は Memory Planner（Phase 9）の KEEP/低bit KEEP 設計と
  同型でそのまま転用可能。
- [Erase-then-Delta Attention（arXiv:2606.26560）](https://arxiv.org/abs/2606.26560) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2606.26560)
  — Li, Zhang, Luo ほか（Alibaba 系）、2026-06。
  **【2026-10-25 確認済み】**: delta-rule の corrective write は現在の write address
  しか訂正できないため、別場所の stale data が消えない問題へ、
  **learned erase direction に沿った erase step を write と分離して追加**。
  dense 2.5B / MoE 25B で pretraining、4k〜128k long-context でも最良。
  Phase 14 の GDN 後続拡張候補として索引。
- **FlashQLA の出典確認（2026-10-25）**:
  [Qwen blog（2026-04）](https://qwen.ai/blog?id=flashqla) +
  [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  の FlashQLA backend。GDN chunked prefill の forward/backward を
  hardware-friendly に代数再定式化した kernel 群。repo 参照済みだが、
  **F-2 の chunkwise 分解と同じ source** であることを確認。
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

### F-5. Gated DeltaNet-2 — erase/write gate 分離、GDN と KDA の上位集合（2026-10-25 新着掃描）

- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention（arXiv:2605.22791）](https://arxiv.org/abs/2605.22791)
  / code: `NVlabs/GatedDeltaNet-2` — Hatamizadeh ほか（NVIDIA）、2026-05（引用30）。
  **【2026-10-25 abstract 確認済み】**: 従来の delta rule 系が1つの scalar gate で
  「key 側の erase 量」と「value 側の write 量」の両方を制御していたのを、
  **channel-wise erase gate b_t と channel-wise write gate w_t に分離**。
  「両 gate が同一 scalar に潰れると KDA、decay も潰れると Gated DeltaNet」に
  それぞれ帰着する一般化。fast-weight update view、
  **channel-wise decay を asymmetric erase factor に吸収した chunkwise WY**、
  並列学習を保つ **gate-aware backward** を導出。
  1.3B / 100B FineWeb-Edu で Mamba-2 / GDN / KDA / Mamba-3 中の最強、
  利得は RULER multi-key retrieval（長文脈）で最大。
- **HexaTrain との関連**: **Phase 14 の migration 設計空間を更新する論文。**
  repo は GDN / KDA / SimpleGDN migration を検討中だが、GDN-2 はその両方の
  上位集合であり、「GDN-2 の2 gate を実装してしまえば GDN と KDA は
  gate を潰す ablation で回収できる」＝ **実装1つで ablation 2点**。
  gate-aware backward は HTP graph 上で delta rule 系の backward を
  手書きする HexaTrain の実装設計（学習側の並列性維持）の直接参照になる。
  758k param / T32 では RULER 級の長文脈利得は出ないと想定されるが、
  retrieve 系タスクで GDN vs KDA vs GDN-2 の比較を Phase 14 の評価に足す根拠。

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
  → **【2026-10-25 精読済み】**: 設計の要点は
  (1) per-tensor 決定指標は **「optimization bandwidth」= 単位時間あたり節約 memory**。
  tensor size・再計算時間・圧縮/解凍時間・圧縮率から算出、
  (2) MILP は Boolean 変数 `R_i ∈ {recompute, compress, retain}`、
  目的 = 総 overhead 最小、制約 = memory 容量 + **layer 先頭 op は retain/compress
  固定**（checkpoint 役割）。**同一構造の Transformer block を利用して
  1 block 分だけ解いて全 layer へ適用**（PuLP で 0.5 秒未満）—
  HexaTrain の L19 同一 block 構成にそのまま流用できる最適化、
  (3) 圧縮は **outlier 分離方式**: channel-wise Z-score>3 の outlier channel は
  生値 + index で保存し、残りを FP16→INT4 対称量子化（group scale）。
  RHT（A-3/A-8）・BiE（D-4）・mean-center と同じ「outlier 分離」family の別実装、
  (4) policy は学習中に**指数 backoff 間隔（1,2,...,2^n iter）で統計を再評価**して
  MILP を解き直す。naive 一括量子化は精度 -39% かつ収束不能になり得るため、
  tensor 適応 policy が本質的であることも実証済み。

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

### G-4. PlaneZip — lossless bit-plane 圧縮 + compression-aware planner（2026-10-25 新着掃描）

- [arXiv:2610.25033](https://arxiv.org/abs/2610.25033) — 2026-10-18（1週間前の新着）。
  **【2026-10-25 abstract 確認済み】**: GPU↔host swap 対象の training activation を
  **lossless bit-plane codec** でオンザフライ圧縮。128-byte block ごとに
  zero-value / delta bit-plane / bit-plane / zero-masked bit-plane の最小を選択し、
  前反復の圧縮率で per-tensor 予測。BF16 activation 1.71x / FP32 1.39x（幾何平均）。
  本質は **planner 側**: cost-model planner（MegTaiChi 等）が swap に「生 size 分の
  転送時間」を課すため swap がほぼ出ないことを指摘し、**planner の転送 cost を
  圧縮後 size 基準に変える**だけで rematerialization overhead を
  bandwidth-aware planner 比 47%/31% 減・MegTaiChi 比 73%/54% 減（trace シミュ）。
- **HexaTrain との関連**: **Phase 9 の planner 設計への直接的な design rule**。
  Adacc（G-1）の MILP は {recompute, compress, retain} を選ぶが、
  PlaneZip の教訓は「**compress 選択肢の cost は圧縮後転送量で評価しないと
  MILP が compress をほぼ選ばない**」という objective 設計の落とし穴。
  HexaTrain の FastRPC 境界は PCIe より帯域が細く、転送 cost 圧縮の効きはさらに大きい。
  また bit-plane 系 lossless codec は bit 演算中心で **HVX の得意領域**であり、
  lossy 量子化圧縮（Adacc / CompAct 系）に対する NPU 側の対抗手段候補になる。
  対象が GPU swap であり NPU/TCM そのものではない点は注意。abstract 確認済み・本文未読。

### クラスタGの統合所見（2026-10-25時点）

1. **Memory Planner は「新規概念」ではなく「確立した問題領域の HTP 版」**として設計できる。
   Adacc の MILP scheduling と policy evolution は Phase 9 の実装柱として直接参照する。
2. HexaTrain の差別化要素は backend 異質性（HTP/HVX/CPU + RPC 境界）と spill 先の
   memory 階層。外部論文の cost model にこの2つを足すのが設計の本体。
3. **【G-4 追加後】planner の objective は圧縮後転送量で書く。**
   生 size 基準の cost model は compress 選択肢を構造的に過小評価する
   （PlaneZip の MegTaiChi 批判）。Phase 9 の MILP 実装時に最初から
   「compress 後の転送 cost」を objective に入れる。FastRPC 境界は細帯域なので
   この設計則の効きは GPU/PCIe より大きいと予測される。

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

### H-1b. Getting the Most Out of Your Tokenizer（2026-10-25 追加）

- [arXiv:2402.01035](https://arxiv.org/abs/2402.01035) / code: `gautierdag/tokenizer-bench`
  — Dagan, Synnaeve, Rozière（Meta FAIR）、ICML 2024。
- **要約**: pretraining / domain adaptation における tokenizer 選択を
  **average bytes per token**（UTF-8 byte ÷ token 数）を主指標に比較。
  domain 特化の tokenizer 学習とその initialization が、通常想定より
  学習効率に大きく効くことを示した。
- **HexaTrain との関連**: HexaTrain の bpb 指標と同じ bytes/token 視点の先行研究。
  Nicopedia domain 特化 V1024 byte-BPE という現行設計に対して、
  (a) tokenizer の初期化（既存 BPE からの warm start）の効果、
  (b) domain 特化 vs 汎用 tokenizer の差の定量化、という
  Phase 10 の評価項目を先取りしている。
- **次に読む**: warm start の具体的手続き。
  → **【2026-10-25 精読済み】**: warm start は **Fast Vocabulary Transfer（FVT）**
  （既存 model の embedding 空間へ新 vocabulary を写像）で行う。FVT なしでは
  HumanEval Pass@1 が 20.5% → 18.4% に劣化。tokenizer 交換は
  **fine-tuning data が ~50B token 以上**ある場合にのみ推奨（それ以下では
  baseline に届かない）。tokenizer 交換後、embedding 以外を凍結する方法は無効。
  また「最も圧縮の良い tokenizer（Identity pre-tokenization）は明確に性能を害する」
  「GPT-4 pre-tokenization regex が圧縮と性能のバランス最良」
  「vocab size 増は圧縮を指数的に飽和改善、コード性能への影響は弱い」
  「**inference-optimal と memory-optimal の vocab size は別々に定式化できる**
  （embedding params vs attention cache の trade。GQA はこれを変える）」
  という設計指針の数々は、HexaTrain の V1024/V2048/V4096 掃引
  （Phase 10）の評価軸（bpb・RAM・推論時間）と一致する形でそのまま使える。

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
  端末生成する「アプリ単体化」課題のコスト削減に使える可能性。
- **【2026-10-25 abstract 確認済み】Decoupling the Benefits of Subword Tokenization
  （arXiv:2604.27263）** — 制御された byte-level pretraining pipeline で
  subword tokenization の利点を分解: **(1) training throughput 増大**
  （同一 compute でより多くのテキスト）と **(2) subword 境界の言語学的 prior /
  inductive bias** の2因子が本質で、後者は byte-level 学習に prior として
  注入可能と示唆。HexaTrain の Byte-BPE V1024 は (1) を採る選択だが、
  embedding table が 758k param の大きな割合を占める以上、
  「byte まま + 境界 prior で語彙を削る」arm は Phase 10 の理論的対立仮説として
  記録に値する（実現コストは data pipeline 変更が大）。abstract 確認済み・本文未読。
- Bridging KV-Cache Quantization and Linear Attention（arXiv:2610.11214）は
  クラスタ F-3 に詳細あり（RAM-Net 統一 + weight migration）。
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
  → **【2026-10-25 精読済み】**: max 減算後の距離 `δ_i = min(max_j x_j − x_i, D_max,h)`
  （`D_max,h ≤ 127`、int8 範囲に収める）を clip し、surrogate score は
  単調減少の線形形 `s_i = B_h − S_h δ_i`（非負性は calibration 制約で保証）。
  正規化は Z = Σs_i、ρ = ⌊T/Z⌋ の**整数除算のみ**（leading-bit 近似で int8 版は
  最大 15.1x 高速化）。per-head parameter `θ_h = (B_h, S_h, D_max,h)` は
  64 sample の代表 data で KL divergence を grid search 最小化して offline 決定。
  **head-wise granularity が per-layer / global を大きく上回る**（MNLI 0.723 vs 0.545）。
  QAT retraining 必須（なしでは −20.6 pt 級、ありなら −0.3〜1.9 pt）。
  HexaTrain T32 では per-head calibration 3 scalar × 38 head で済み、
  QNN graph では「max 減算 + min + 乗減算 + 総和 + 整数除算」の要素演算列で構成できる。
  head-wise 優位の観察は「head 共有 calibration からの簡略化」より
  最初から per-head で試すべきことを示唆。

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

### H-8. KV cache 低bit化の学術的錨（2026-10-25 追加）

repo の P3「KV low-bit storage + dequantize-before-attention」が参照する
Prism/PrismML の mean-centering は blog level の情報で arXiv 論文としては見つからない。
**学術的な対応物（錨）**は次の系譜:

- [KVQuant（arXiv:2401.18079, NeurIPS 2024）](https://arxiv.org/abs/2401.18079) —
  非 uniform 量子化 + pre-layer rotation で sub-4-bit KV。
- KIVI — per-channel outlier 処理（mean/median centering 相当）の 4-bit KV。
  **PrismML 系 mean-centering の学術的最近縁**。
- [RotateKV（arXiv:2501.16383）](https://arxiv.org/abs/2501.16383) —
  rotation による 2-bit KV。[Kitty（arXiv:2511.18643）](https://arxiv.org/abs/2511.18643)
  も 2-bit。
- [SAW-INT4（arXiv:2604.19157, 2026-04）](https://arxiv.org/abs/2604.19157) —
  system 制約下で成立する 4-bit KV 手法の最小集合を特定。
- **HexaTrain との関連**: incremental KV-cache decode が成立した後の
  low-bit KV lane（P3）の比較 anchor。HexaTrain の KV は H2/T32 と小さいため
  粒度（per-channel vs per-token）の固定費比が大規模系と逆転する点に注意。

### クラスタHの統合所見（2026-10-25時点、H-6/H-8 追記後更新）

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

### H-7. Cut Cross-Entropy（CCE）— logits 非実体化（2026-10-25 追加）

- [Cut Your Losses in Large-Vocabulary LLMs（arXiv:2411.09009）](https://arxiv.org/abs/2411.09009) /
  [AlphaXiv](https://www.alphaxiv.org/abs/2411.09009)
  — Wijmans ほか（Apple）、2024-11。code: `unslothai/cut-cross-entropy`。
- **要約**: CE loss 計算で全 vocabulary の logits を global memory へ実体化しない
  （vocabulary chunk ごとの逐次計算）。大語彙（128K）では CE が学習 memory の
  最大90%を占め得る問題へ、Gemma 2 で数 GB → 20 MB 未満、数値 parity 維持。
- **HexaTrain との関連**: 現行 V1024/T32/B8 では logits は 8×32×1024 ≈ 1 MB（FP32）
  で memory 動機は小さい。ただし (a) Phase 10 で V2048/V4096 へ伸ばす場合、
  (b) MicroEngram で出力側が増える場合、(c) CE backward `dlogits=(P-Y)/N` を
  future-mixture で再利用する設計、の3点で chunked CE の原理が
  QNN graph 構成の選択肢になる。HMX 逐次 vocab chunk 処理は
  retained-ACC 手法（R2）との組合せ余地もある。

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

### I-2. Curriculum の学習動力学解析 — 小規模ほど安定化利得（2026-10-25 新着掃描）

- [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics（arXiv:2601.21698）](https://arxiv.org/abs/2601.21698)
  — Elgaar, Amiri、2026-01（v2 2026-05、引用9）。
  **【2026-10-25 abstract 確認済み】**: 14M〜1B × 300B token で、
  Age-of-Acquisition / word frequency / Verb Variation の3 curriculum を random と比較し、
  latent phase・**gradient noise scale (GNS)**・output head の **singular entropy** で解析。
  主張: (1) 学習は共有の latent phase 列を辿り、curriculum は **phase 内の滞在時間を変えるだけ**,
  (2) **random ordering は 14M〜70M で GNS が高く、160M まで singular-entropy spike**
  （noisier gradient + output head saturation）、(3) 下降順は上行順の利得を大きく失う,
  (4) **大規模になるほど安定性の差は縮む**。
- **HexaTrain との関連**: **HexaTrain（0.76M）はこの解析で「安定化利得が最大」の
  規模帯よりさらに小さい端**にいる。Phase 13（Difficulty-Aware Sampling）の
  仮説「小規模 model では data ordering が効く」に対する直接的な外部証拠であり、
  規模が上がるほど curriculum 利得が減るという逆相関も、
  HexaTrain の scale 実験（Phase 0 の model suite）での予測を与える。
  測定項目（GNS・output head singular entropy）は HexaTrain の診断 instrumentation に
  そのまま組み込める（output head は H2 次元なので SVD コストが小さい）。
  指標は言語学的3種で I-1 の情報理論指標と異なるため、
  難易度指標の選択は I-1（compression ratio / MTLD / Flesch）を主、本論文を対象語彙の補助とする。

### I-3. LR decay × curriculum の非両立 — decay を averaging で置き換える（2026-10-25 新着掃描）

- [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining（arXiv:2511.18903）](https://arxiv.org/abs/2511.18903)
  — Luo ほか（清華）、2025-11、**ICLR 2026 Oral（Top 1.2%）**、引用10。
  **【2026-10-25 abstract 確認済み】**: **ascending data quality order と decaying LR の
  非両立**を特定 — 高品質 data が学習終盤（LR が落ちた後）に配置されるため、
  curriculum の利得が random shuffle と同等まで消える（constant LR なら利得が持続）。
  対処2種: (1) **final LR を peak より「緩やかに小さい」程度に留める緩やかな decay**,
  (2) **LR decay を最終 checkpoint 群の重み付き平均で置き換える**。
  1.5B / 30B token で検証、組合せで random shuffle 比 benchmark 平均 +1.64%。
  「data curriculum と optimization の co-design」を呼びかけ。
- **HexaTrain との関連**: **Phase 13（curriculum）と LR schedule（Phase 6）の
  co-design 制約を与える論文。** HexaTrain が curriculum を導入する場合、
  現行の decay 型 schedule（S4000 凍結 control 系）のままだと
  「easy→hard」順の利得が終盤で打ち消されるリスクがあり、
  (a) 緩やか decay arm、(b) J-4 の checkpoint averaging 置換 arm、が
  curriculum A/B の **必須の交互作用要因**になる。
  J-4 の「高 LR × averaging」観察と統合すると、
  「高 LR + 緩やか decay + checkpoint averaging」が文献整合的な1セットになる。
  G1 LR stress grid の checkpoint 群を使った averaging A/B（統合所見 9）に
  「decay 置換」という評価軸が加わる。

### I-4. data mixing 系譜 — mixture 重みの最適化（2026-10-25 新着掃描）

- **Data Mixing Laws（[arXiv:2403.16952](https://arxiv.org/abs/2403.16952)、ICLR 2025、引用218）** —
  **validation loss を mixture 比率 `p` と model/data size の関数として
  `f̂ᵢ(p) = cᵢ + kᵢ·exp(Σⱼ tᵢⱼ pⱼ)` で予測**し、訓練せずに最適 mixture を選ぶ。
  小規模 fit → 大規模 extrapolation。follow-up に Aioli / OLMIX / IDEAL。
- **CausalMix（[arXiv:2607.01104](https://arxiv.org/abs/2607.01104)、2026-07）** —
  data pool の統計 feature を covariate、domain mixture を treatment とする
  **因果推論 framing** で CATE を推定し state 依存の最適 mixture を動的推定。
  **proxy model を 512 run の Qwen2.5-0.5B で fit** して 800K pool → 7B へ外挿。
  RegMix 等 proxy 系 baseline を上回る。「静的分布仮定の proxy 系は
  pool 変化で再訓練が必要」という限界指摘。
- **HexaTrain との関連**: Phase 13（Difficulty-Aware Sampling）の隣接系譜で、
  curriculum（順序）と mixture（比率）は独立軸。**Data Mixing Laws の
  「小規模 run 群で loss 関数を fit → 外挿」は、HexaTrain の安価な
  host-side 試行（A-7 Wortsman proxy と同じ思想）を mixture 設計に適用する
  型紙**になる。CausalMix の「0.5B × 512 run」も同型で、
  HexaTrain が multi-source corpus を増やす場合の設計参照。
  ただし HexaTrain の現行 corpus は単一 source 系のため
  「mixture 最適化」自体の効用は corpus 拡大時まで保留。

---

## J. 小規模学習 recipe の系譜（modded-nanogpt speedrun）

**調査日: 2026-10-25（追加クラスタ）。** HexaTrain の Original Muon optimizer identity
（`keller_original_64560829_fp32`）の系譜元である modded-nanogpt speedrun の
現状と技術内訳を確認した。

### J-1. modded-nanogpt speedrun（124M / FineWeb / 8xH100）

- [KellerJordan/modded-nanogpt](https://github.com/kellerJordan/modded-nanogpt) —
  **【2026-10-25 再確認・更新】record #92 = 0.665 分（約40秒、2026-08-30、DevenPzak）**。
  本調査初版時点の「90秒」記載は陳腐化していた（8月末の #90〜#92 で 68秒→41秒級へ急加速）。
  [Speedrun Optimization Writeup（discussion #211）](https://github.com/KellerJordan/modded-nanogpt/discussions/211)
  / [LessWrong 記事](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/) /
  [Prime Intellect auto-speedrun](https://www.primeintellect.ai/auto-nanogpt)（AI agent が
  speedrun 改善を競う benchmark、~10k optimizer-track run）。
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
  → **【2026-10-25 確認済み（blog level）】**: Muon の momentum warmup は
  **update-to-weight ratio を学習序盤で schedule しながら nesterov momentum buffer を
  立ち上げ、warmup 期間中は bias correction を効かせる**形
  （[Muon 原典 blog](https://kellerjordan.github.io/posts/muon) /
  [Varun Neal 解説](https://varunneal.github.io/essays/muon)）。
  「update-to-weight ratio の制御」は C-7 MACRO の解析軸と同じ量であり、
  HexaTrain の HVX Muon（CPU 側 optimizer）でも**序盤 momentum warmup +
  bias correction は実装コスト小**で試せる（state は既存 buffer のみ）。
  分散 overhead 削減側は 8 GPU 前提の話が多く、単一端末の HexaTrain には非適用。

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
  → **【2026-10-25 確認状況】**: HTML 版（v1/v3）は 404 で公開されておらず、
  PDF は tool で parse 不可。abstract の範囲では「μ-centering = 決定論的操作、
  μ-loss = 正則化」の2実装形のみ確認済みで、**数式 level の実装詳細は未検証**。
  A/B 設計時に PDF 原文（2601.02031）の人間側精読を要る状態を維持する。

### J-4. checkpoint merging 系譜（P2「Model Merging during Pre-training」の外部動向、2026-10-25 追加）

- [Model Merging in Pre-training of LLMs / PMA（arXiv:2505.12082）](https://arxiv.org/abs/2505.12082)
  — pretraining 中の model merging の最初の包括調査とされる（ByteDance Seed 系）。
- [Practical Guidelines for Model Merging in LLM Pre-Training（ACL 2026 Industry）](https://aclanthology.org/2026.acl-industry.105)
  — training phase 横断の checkpoint merging。**単純平均が一貫して改善**。
- [Early Weight Averaging meets High Learning Rates（OpenReview, 引用42）](https://openreview.net/forum?id=IA8CWtTnUr)
  — **高 LR 学習ほど checkpoint averaging の恩恵が大きい**。
- NeurIPS 2025 tutorial では **decay-free LR schedule を merging で代替**する
  方向も紹介。
- **HexaTrain との関連**: repo の P2「Model Merging during Pre-training」と
  端末間疎結合 merge 応用の外部裏付け。特に「高 LR × averaging」の相互作用は
  G1 LR stress lane（1.0x〜2.0x grid）の観測と直結し、
  **stress grid の checkpoint 群をそのまま averaging A/B に流用できる**。
  LR decay schedule の設計空間（S4000 凍結 control との関係）にも
  「decay を merging で置き換える」変種が加わる。
- **次に読む**: PMA の merge timing（どの phase 間で merge したか）。

### J-5. ANVIL II — Muon 後継 optimizer による 39.9秒 record（2026-10-25 追加）

- [Training NanoGPT in 39.9 seconds（hyperstition.cc）](https://hyperstition.cc/training-nanogpt-in-39-9-seconds)
  / repo の [Optimization Benchmark Track 3](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md)
  — ANVIL = **Averaged, Normalized Velocity with Isotropic Lanes**（2026-08〜09）。
  **【2026-10-25 確認済み】**: record 39.914 秒 / val loss 3.28（従来 73.889 秒から
  46.0% 改善「過去45 record 合計を上回る削減」）。一部手法は frontier 向けとして
  **非公開に留保**されている（「redacted methods that scale to the frontier」）。
  公開内訳: **(1) twin momentum（fast + slow の2本）**、
  **(2) update の singular spectrum を平坦化する 6 本の polynomial map**
  （Polar Express 系の適応直交化と同型）、(3) **slow momentum との一致で
  cautious weight decay を gating**、全体を **EMA-Nesterov で wrap**。
  ablation（秒）: sampled softmax 8.41 > **n-gram table（84.6M rows、bigram+trigram）7.06** >
  **ANVIL II 5.09** > FP8 4.95 > model 縮小 3.82 > CUDA graphs 2.99 >
  sparse embedding updates 1.20 > layer mixing 0.49。
  weight averaging は readout を **final 35% / EMA 65%** で blend（loss 3.2966→3.2784）。
  sampled softmax は candidate 10,240→14,336→24,576→最終87 step で full 50,304、
  gradient error bound 2(1−r_C)。
- **HexaTrain との関連**:
  (1) **ANVIL II の3要素は Muon 改良系譜（C-1/C-2/C-5）の community 実装合流点**で、
  HVX Muon の Phase 5（直交化）・Phase 4（row 制御）の候補を速度面から支持する。
  特に「polynomial map 6 本」は C-5 Polar Express の **NS 反復数を変えず係数を
  変える**方向の community 側採用実績。
  (2) **weight averaging の 35/65 readout blend** は J-4 の「高 LR × averaging」に
  具体的な実装形（EMA 保存 + readout 時 blend）を与える。HexaTrain の
  G1 LR stress grid checkpoint 群の averaging A/B（統合所見 9）で
  「単純平均」対比の arm にできる。
  (3) sampled softmax / n-gram table / FP8 は HexaTrain の V1024・T32・D64 規模では
  利得が出ない（語彙が小さく softmax がボトルネックでない）ため非適用と明記する。
  (4) 非公開部分があるため「ANVIL II は Muon より完全に優る」という一般化は
  **しない**（record は 8xH100・124M・短時間設定に固有）。
  → **【2026-10-25 追記】Track 3（step 数 benchmark）との関係**:
  ANVIL は wallclock track の記録であり、[Track 3 README](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md)
  （target val loss 3.28 を最少 step で）には現れない。Track 3 の現行 record は
  **#46 = 2690 step**（tuned Muon baseline 3250 step から 20.8% 削減）で、
  構成は Muon + **SOAP-Muon preconditioning** + radial brake + PowerCool LR +
  **EMA-Nesterov wrap** + Tail-EMA readout（README 自身が「独立に利得証明済みでない」
  と注意）。ecosystem には MUON+ / NorMuon / **Muown** / DynMuon / Contra-Muon /
  Aurora 等が並び、C クラスタの Muon 改良系譜が community の A/B 対象に
  なっていることの実測確認になる。EMA weight averaging（J-4/I-3）は
  Track 3 でも #45（150-step EMA への final blend）として採用済み。

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
| **【2026-10-25 午後追加】Muon 採用の大規模実証（Phase 4-6 の前提）** | multi-billion × 100M token batch で Muon が AdamW を一貫上回り、**大 batch で AdamW が劣化する中 Muon は安定**（C-11）。LR transfer は update-RMS matching が規範 | **高** |
| **【2026-10-25 午後追加】小規模 curriculum の安定化利得（Phase 13）** | 14M〜1B で random ordering が GNS 増大・singular-entropy spike、**規模が小さいほど差が大**（I-2）。HexaTrain は利得最大帯のさらに下端 | 高 |
| **【2026-10-25 午後追加】V81 HMX 上の学習 system の実在（R2/D クラスタの外部対照）** | 同世代 V81 HMX（OnePlus 15）で FP16 operand→wider accumulator→FP16 出力を用いた LoRA 学習 system が公開（B-9）。**per-row range guard** は R2 の signed 出力問題と同根の独立実装 | **高** |
| **【午後追加】「Muon を2次元隠れ行列のみに適用、embedding/output は別系」の分割** | Symmetry-Compatible 原理（C-12）が「embedding/LM head は permutation 対称で full orthogonalization 不適」を理論化。HexaTrain の現行分割はこの原理と整合 | **高** |

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
8. **checkpoint averaging（J-4）**: 高 LR 学習ほど averaging の恩恵が大きいという
   外部観察があり、G1 LR stress grid の checkpoint 群を averaging A/B へ
   流用できる。LR decay を merging で置き換える変種も候補。
9. **【午後追加】planner objective を圧縮後転送量で書く（G-4 PlaneZip）**:
   生 size 基準の cost model は compress 選択肢を構造的に過小評価する。
   Phase 9 MILP の最初からの設計則。
10. **【午後追加】orthogonalization を間引く 4 軸（C-10/C-6 MiMuon）**:
    Dion=amortize / Dion2=row fraction / Gram NS=FLOP 削減 / MiMuon=適用を間引く。
    Phase 5 の「NS を毎 step 完走しない」arm 群の候補が4つに。
11. **【午後追加】GDN-2 の統一 gate 実装（F-5）**: erase/write の channel-wise 2 gate で
    GDN と KDA を1実装+ablation で回収。Phase 14 の設計空間を更新。
12. **【午後追加】unbiased estimator の SR 代替（A-11 MS-EDEN）**:
    Phase 12 の estimator A/B を「SR vs MS-EDEN 型」に更新。
13. **【午後追加】forward/backward の scale 一致性の観測点（A-12）**:
    転置による scale 不一致が FP4 不安定の原因という診断は、HMX R2 系の
    block scale が dX/dW で一致しているかという新観測点を与える。
14. **【午後追加】weight averaging の readout blend（J-5 ANVIL II）**:
    final 35% / EMA 65% の blend が record 設定の構成要素。J-4 の arm に
    「単純平均対比の blend」を追加できる。

## 3. 外部動向として記録すべき構造変化

- **FP4 pretraining が 10T token 規模で FP8 並みに到達**（A-3）。recipe の中核は
  RHT + 2D scale + SR + 高精度 layer 選択の set。HexaTrain の P2 候補は個別ではなく
  recipe として設計すべき。
- **「NPU 上の exact-BP 学習」は公開文献に存在しない**（B）。HexaTrain の
  scratch training 構成の独自性は fine-tuning 系先行例と区別して主張できる。
- **linear attention 系の理論統一**（F-1 TTT ≡ linear attention）。
  HTP 実装は attention graph 部品の再構成として設計できる。

## 4. すぐ読むべき論文の優先順位（本文精読リスト）

0. **【午後追加】First-Order LoRA Training with Phone NPUs（2610.24816）** —
   同世代 V81 HMX の学習 system。per-row range guard の閾値・sensitivity 推定法・
   V79/V81 差は R2 と Phase 12 の直接対照になるため最優先。
1. **MUON+（2602.21545）** — Muown lane の比較 arm 設計に直結。挿入する normalization の
   正確な形を確認する必要がある。
2. **Compute Optimal Tokenization（2605.01188）** — Phase 10 の設計を compression rate
   掃引へ寄せるかの判断材料。最適 rate の回帰式。
3. **VGA（2510.09017）** — G1 variant の V-gate arm。gate の関数形と backward cost。
4. **Quantized Softmax（2609.33591）** — QNN softmax 置換候補の実装可能性
   （grid 構成・K の感度）。
5. **Depth-Attention（2606.05014）** — Value Residual lane の別実装。
   どの layer から key/value を取るかの設計。

## 5. 研究優先順位への反映提案（2026-10-25 時点・提案のみで verdict 変更なし）

`research-priorities.md` の管理者向けの素材。採否・順位は既存の運用規則
（NOW 最大5件・昇格時の入れ替え決定・preregistration）に従って判断されるべきもので、
本節は何も変更しない。

1. **Phase 4（Muown 比較）へ post-NS row-normalization arm の追加検討**（根拠: C-1
   MUON+ が state なし・overhead ≤2% で LLaMA 全規模に有効、C-2 NorMuon、
   repo row-geometry audit）。**Muon+ 型（`row_col` L2 norm）が最安の初手**。
   Muown Algorithm 1 の CPU reference と同時進行可能。
2. **G1 gate variant に V 入力 gate の追加検討**（根拠: E-2 VGA の gradient 解析 +
   E-1 原著 G2）。既存 G1 graph からの差分が projection 入力だけで
   `PROMOTE_G1_GATE_VARIANTS` の継続探索と整合。
3. **P1 Candidate Pool の Value Residual に Depth-Attention 型を併記**
   （根拠: E-3。parameter ゼロ・KV 中立・L19 では stride 9 で source ≤4 層。
   J-1 speedrun の value embeddings 36% gain も傍証）。
   ただし backward は未解析で cross-layer gradient accumulation が必要。
4. **Phase 12 開始条件として tensor 除外表を先に作成**（根拠: A-2 Kempner
   FFN 側 norm 优先除外、A-3 NVFP4 の先頭2+末尾8 block 残置、A-5 SR 理論の
   weight bias floor、A-9 静的 scale 制約）。
5. **QNN softmax 置換候補に K-interval grid を追加**（根拠: H-3。
   FWM(τ=6) + Weight-STE + K=16 で +0.004 nats。Attention Normalizer Search と統合）。
6. **学習安定化候補へ OEC（μ-centering）と logit softcap の追加検討**
   （根拠: J-3, J-2。いずれも graph 変更最小。QK-norm は softcap と置換関係）。
7. **Phase 6 に SR × accumulation 掃引を結合**（根拠: A-5。
   1 bit ↔ 4x batch（実測2x）の補償則。B8 固定での SR 単体 A/B は情報量小）。
8. **Muon+WD arm の位置づけ明確化**（根拠: C-6 Moonshot。
   WD+RMS 整列は大規模で必須だった修正。HexaTrain 規模では
   「不要で済むか」の検証として意義づけ可能）。
9. **G1 LR stress grid checkpoint 群の averaging A/B 流用**（根拠: J-4。
   高 LR ほど averaging の恩恵大という外部観察。新規 run 不要で試せる）。
10. **VGA の quantization fidelity 改善観察を Phase 12 との交差検証に**
    （根拠: E-2。8-bit PTQ 劣化最小。G1 導入済み model で低bit化耐性を測る意義）。
11. **Muon 系理論 telemetry の追加と spectral reweighting arm**（根拠: C-8 NDS、
    C-9 SAMuon。Phase 4/5 の評価に NDS 相当量（update 方向曲率）を入れると
    arm 間の効き所の帰属が取れる。SAMuon-lite は state ゼロ・rank-1 power iteration
    で Orthogonalization Accuracy lane の arm 候補）。
12. **【午後追加】Phase 13 curriculum A/B に LR 側の交互作用 arm を必須化**
    （根拠: I-3。ascending order × decay の非両立。緩やか decay arm と
    checkpoint averaging 置換 arm を curriculum A/B と同時走查する。
    J-4/I-2 とセット）。
13. **【午後追加】B-9（V81 LoRA system）との対照表作成**
    （根拠: B-9。HMX signed FP16 出力の per-row range guard vs R2 late-clamp /
    sensitivity 別精度配分 vs Phase 12 tensor 除外表 / resident weight tile 再利用 vs
    Cross-Layer Reuse。同一世代実機の独立実装は R2 系の数値合理解釈の
    交差検証材料になる）。
14. **【午後追加】Phase 5 に orthogonalization 削減 arm 群を4軸で整理**
    （根拠: C-10 Dion 系譜 + C-6 MiMuon。amortize / row fraction / Gram NS /
    適用間引きの4候補を、NS5 反復数と数値誤差の sweep に並べて比較設計する）。

## 6. 本調査の限界（2026-10-25 時点）

- abstract ベースの要約が中心。数値の詳細（特に A-3 の高精度 layer 内訳、
  C-2 の second moment update 則）は本文未確認。ただし精読リスト上位5本
  （MUON+ / tokenization / VGA / K-interval softmax / Depth-Attention）は
  本文確認済みで「次に読む」に追記済み。
  **午後追加の B-9（2610.24816）は HTML 本文確認済み**（sensitivity 推定・
  range guard 実数値・V81 kernel 数値まで）。B-9 の学習速度の end-to-end 数値は
  abstract に載っておらず未記録。
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
