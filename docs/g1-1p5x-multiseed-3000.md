# G1 1.5x multi-seed replication（3000 step / seed 2・4）— 実験設計

## Status

**PLANNED / 未実行。** 長時間 training は Tier 3 であり、ユーザーの明示指示なしには開始しない
（[docs/agent/device-test-tiers.md](agent/device-test-tiers.md)）。
本文書は実験設計と引き継ぎ指示を固定するもので、**結果を含めない**。

## 目的

stress grid（20 cell）と 1.5x long run は seed 1 のみで、次の 2 点を区別できない。

1. 初期（〜1000 step）の Val / Dev 同時改善が seed を跨いで残るか
2. step 1500–3000 で Dev だけ符号が反転する現象が再現するか
   （反転 step が seed ごとにずれるだけか、top-1 悪化が同じ checkpoint 帯に載るか、
   gate saturation の進行と対応するか）

2 / 2 の新 seed で「初期は勝つが 1500–3000 で Dev が反転」が再現するなら、G1 は
**早期 sample-efficiency 部品**として位置付けられ、high-LR の final quality lane は閉じる。
新 seed で 1 も再現しないなら、seed 1 の split divergence は弱い証拠として記録し、
これ以上予算を割らない。

## best LR arm の選定（一次データから）

`docs/results/g1-lr-stress-2026-09/quality-paired.csv` と `time-to-bpb.csv`、
`gate-static.csv` 由来（Δ = G1 − Control、負 = G1 良い）。

| LR | mean ΔBal (100–500) | mean ΔBal (400–500) | ΔVal @500 | ΔDev @500 | min ΔDev (5 cell) | 完全飽和 head @500 |
|---:|--------------------:|--------------------:|----------:|----------:|------------------:|-------------------:|
| 1.00 | −0.063171 | −0.042610 | −0.026774 | −0.038650 | −0.038650 | 0 / 38 |
| 1.25 | −0.058631 | −0.021859 | −0.028130 | −0.002972 | −0.002972 | 1 / 38 |
| **1.50** | **−0.065756** | **−0.046563** | **−0.036408** | **−0.023564** | −0.023564 | 12 / 38 |
| 2.00 | −0.054615 | −0.025566 | −0.023769 | −0.030359 | −0.027555 | 18 / 38 |

primary time-to-bpb は target 2.90 で **G1 1.5x = step 400 / Control 1.25x = step 500
（Δstep −100）**、target 2.95・3.00 は Δstep 0、2.85 は 500 step 内で未達。

**選定 = 1.5x。** 根拠は (a) mean ΔBalanced と late-step mean が 4 つ中最も深い、
(b) step 500 で ΔVal が最大かつ ΔDev が 1.25x の 8 倍の利得を保つ、
(c) step-base の time-to-bpb 利得が唯一 1.5x 由来、(d) 既存 seed 1 の 2000 / 8000 証拠と
同一 arm で繋がる。除外: **2.0x** は `g<0.1` 完全飽和が 18 / 38 head まで進み、
飽和が本質的な利得を隠す（高LR耐性の限界確認としては別途有価値）、
**1.25x** は step 500 で Dev 利得が `−0.002972` に消える局所現象を追う設計になる、
**1.0x** は high-LR の問いではなく historical anchor。

## seed 1 参照軌道（既存 committed evidence、Δ = G1 − Control bpb）

`g1-1p5x-long-2000-2026-09` と `g1-1p5x-full-8000-2026-09/quality-paired.csv` より。

| step | ΔVal bpb | ΔDev bpb | 備考 |
|-----:|---------:|---------:|---|
| 500 | −0.036408 | −0.023564 | stress grid と一致 |
| 750 | −0.040141 | −0.026004 | |
| 1000 | −0.047118 | −0.036221 | |
| 1250 | −0.063891 | −0.028541 | |
| 1500 | −0.031254 | −0.035283 | |
| 1750 | −0.017118 | **+0.011375** | ΔVal NLL −0.030889 / ΔDev NLL +0.018780、top-1 Val −76 / Dev −53 tokens |
| 2000 | −0.021819 | −0.028462 | |
| 2500 | −0.022967 | −0.020733 | |
| 3000 | −0.029737 | **+0.015448** | ΔVal NLL −0.053659 / ΔDev NLL +0.025505、top-1 Val −45 / Dev −16 tokens |

500–8000 の 16 eval point で ΔBalanced は 16 / 16 負、Dev bpb が正になるのは
**1750 と 3000 の 2 点だけ**で、3500（−0.032546）・4500（−0.053880）に戻る。
つまり seed 1 の事象は「後半の恒久的な Dev 崩壊」ではなく **断続的な Dev 反転**である。
判定は「1500–3000 で Dev 反転が 1 点でも出るか」で定義し、崩壊とは書かない。

gate 側は 500–2000 の完全飽和 head 数が `12 / 6 / 4 / 5 / 2 / 4 / 2` で、
1750 の反転にピークはない。したがって R5 の事前期待は「対応なし」側であり、
対応が出たらそれはそれで新規の観察として扱う。

## Protocol

| 項目 | 値 | seed 1 との関係 |
|---|---|---|
| arms | Control `attention_gate=none` / NPRTCKPTV4 / 758,528 params<br>G1 `headwise_g1_sigmoid` / NPRTCKPTV5 / 760,960 params | 同一 |
| seeds | **2, 4**（事前登録。通常実行で許可されるのはこの 2 つだけ） | 新規 |
| LR 1.5x triple | Muon `0.0075` / Aux Adam `0.0033` / target `0.00015` | 同一 |
| schedule | `linear_decay`、decay_start 4000 / decay_end 8000 / schedule_total 8000 | 同一 → **step ≤ 3000 は peak 一定** |
| model | V1024 / T32 / D64 / FFN128 / L19 / H2、batch 8 | 同一 |
| optimizer | Muon + Aux Adam、Muon backend HVX、momentum 0.95、ns_steps 5 | 同一 |
| start | **fresh step-0**（seed 1 checkpoint を resume しない） | 相違点（意図的） |
| steps | **3000**、`CheckpointInterval = 250` | 3000 点是新 |
| eval steps | 500, 1000, 1500, 1750, 2000, 2500, 3000（予算が許せば 250 / 750 / 1250 を追加） | 1750 と 3000 を必ず含む |
| eval | Val first 256 + Dev first 256 chunk、HTP native、canonical original UTF-8 byte bpb | 同一 |
| data | `train_pilot.bin`（dataset `fnv1a64:0c7b2826f5f26fea`）、tokenizer `byte-bpe-v1024` | 同一 |
| Device | NX741J 1 台、QAIRT `2.48.40.260702151143`（pinned、fallback 禁止） | 同一 |
| device 前後の awake helper | `svc power stayon` / `dumpsys deviceidle disable` を **endpoint 明示**で実行し、失敗は `device_awake_and_idle_disabled=false failures=...` として記録する（unscoped な `adb shell` は transport が 2 本以上あると無言で失敗する） | 同一 |
| arm order | seed2 Control → seed2 G1 → seed4 G1 → seed4 Control（1 session = 1 arm） | 交互配置で thermal / battery drift を分散。Mode All / Smoke は seed ごとに開始 arm を入れ替える |

**3000 step にする理由:** seed 1 の反転は 1750 と 3000 の 2 点。2000 で切ると 1 点しか
再現できず「反転時期がずれるのか」を測れない。4000 超は LR decay が効き始めるので
split 挙動の原因分離が難しく、まず constant-peak 域で決着をつける。

**seed 契約（事前登録の強制）:** 結果を見た後に seed を足すのは protocol event なので、
runner は 2 / 4 以外を `-AllowExploratorySeed` なしで拒否し（`SEED_NOT_PREREGISTERED`）、
`seed-registry.json` に role を累積記録する。analyzer はその registry を根拠に
`preregistered` / `exploratory` / `reference`（seed 1）を区別し、exploratory と reference は
R1–R5 の主判定に自動では混ざらない。registry に無い seed ディレクトリは exploratory として
fail closed に扱う（provenance 不明を preregistered と推測しない）。

## データオーダーについての前提（実装前に確認済み）

`app/src/main/cpp/qnn/qnn_transformer_training.cpp` の `nprtTrainingOrder()` は
`state = kNprtCanonicalTrainingOrderSeed`（20260806、model seed とは無関係）から
`state = nprtSplitMix(state + i)` を i ごとに進める**要素単位の関数**であり、
i 番目の batch は総 step 数に依存しない。つまり order は horizon に対して prefix 安定で、
fresh 3000-step run は seed 1 が 500→2000→8000 で見た step 1–3000 と同じ batch を受け取る。

`training_order_hash` は列全体の FNV なので、

- steps=3000 の hash は seed 1 の `ce1000529cb0eff4` (500) / `9d944f9e8f43f19a` (2000) /
  `37fe7bac20c91642` (8000) と**一致してこない**（一致しないほうが正しい）
- 検証すべきは (a) 同一 protocol 内の全 run で order hash が一致すること、
  (b) dataset hash が一致すること、(c) host 側で order(3000) が order(8000) の prefix
  であることを再計算で確認する test（`scripts/nicopedia_real_text_pipeline.py` の
  `training_order_identity` を流用できる）

model seed は `tiny_lm::initialParameters(config, seed)` による初期化にのみ使われ、
data order と eval window は seed 非依存。よって seed 差は初期化差だけを意味する。

fresh start なので checkpoint round-trip は経路に入らないが、seed 1 の continuation との
等価性は未検証のまま残る。これは下記の fresh seed 1 run で検証する。

## 主判定（実行前に固定する）

新 seed ごとに step 単位で split-level（bpb / NLL / top-1,2,5 / mean rank）を計算し、
Control と within-seed で pair する。

- **R1 初期の同時改善:** step 250–1000 で ΔVal < 0 かつ ΔDev < 0 が過半の checkpoint で
  成立するか（seed 1 参照: 5 / 5 成立）
- **R2 Dev 反転:** step ∈ [1500, 3000] で ΔDev NLL > 0（bpb でも報告）が 1 点でも出るか
  （seed 1 参照: 1750 と 3000 で成立）
- **R3 反転時期:** R2 が成立した step 番号を seed ごとに報告し、固定 step で揃うか
  1250 / 2500 などへずれるかで「構造的な位相」と「trajectory noise」を分ける
- **R4 top-1 の同所性:** R2 と同じ checkpoint で Val / Dev top-1（token 数、8192 分母）が
  Control を下回るか（seed 1 参照: 1750 と 3000 で両 split とも低下）
- **R5 gate 対応:** R2 step 前後の `g<0.1` 完全飽和 head 数と mean-of-means を並べ、
  反転と saturation が対応するか記述する（因果の主張はしない）

**fresh seed 1 run（本 runner の対象外）**

当初は seed 1 を fresh で 1 本走らせ、既存 continuation 軌道との決定論的一致を見る案が
あった。しかし seed 1 は参照軌道であり replication sample ではないため、本 runner は
`SEED_NOT_FRESH` で seed ≤ 1 を fail closed に拒否する。fresh == continuation の確認は
必要になった時点で別 runner と protocol 追記を明示的に行うものであり、この 4 arm run の
開始条件ではない。主判定は committed 済み seed 1 軌道を `reference` として参照するだけに
留め（analyzer は reference を R1–R5 に混ぜない）、新 seed の結果が出る前に protocol 差を
読み込む必要はない。

**Decision rule**

| 結果 | 帰結 |
|---|---|
| 2 / 2 の新 seed で R2 再現 | G1 は早期 / 中期 sample-efficiency 部品として位置付け、high-LR final quality lane を閉じる。`2*sigmoid` 等の新 gate は saturation 機構の probe としてのみ継続 |
| 1 / 2 のみ | 曖昧。tie-breaker として seed 3 を `-AllowExploratorySeed` 付きで 1 本だけ増やす。exploratory として報告され R1–R5 の自動判定には入らないため、決着は人間の明示判断で行う。それ以上の追加はしない |
| 0 / 2 | seed 1 の split divergence は再現せず。現行 claim（bpb / NLL が両 split で改善）を維持し、split lane は予算を割らず閉じる |

**統計上の禁止事項:** seed 高々 3 の sign consistency であり、検定・有意差・
"consistent across seeds" は主張しない。率だけでなく token 数（8192 / split）を併記する。
単一 device の単一 night であることを各報告に添える。

## 実装が必要なもので、今は無いもの

既存 runner は seed を通さない。`scripts/run_headwise_g1_lr_stress.ps1` は
`seed = 1`（identity 定義）と `Seed = 1`（training / eval 呼び出し）を直書きし、
`seed1-l19-…-result.txt` / `htp-seed1-l19-…-step<step>.ckpt` の文字列リテラルで
成果物名を決めている（276 / 282 / 481 / 554 / 595 行目）。`run_g1_1p5x_long.ps1` と
`run_g1_1p5x_full8000.ps1` も `htp-seed1-…` を同じ şekilde固定している。

| ID | 必要 | 方針 |
|---|---|---|
| P1 | seed 対応 runner | `scripts/run_g1_1p5x_multiseed.ps1` を新規作成（lr-stress runner を踏襲）。`-Seed`、`-Arm`、`-Mode Plan\|Smoke\|Seed\|All\|Analyze`、`-Steps 3000`、`-CheckpointInterval 250`、eval steps 固定、`-AllowExploratorySeed`（2 / 4 以外の追加 seed 用）。**既存 runner の `seed1` リテラルは seed 1 evidence の再現性を支えるので変更しない** |
| P2 | split-level analyzer | `scripts/g1_multiseed_analyze.py`。入力はその arm の一次レポートのみ。出力は seed × step の split-level paired（bpb / NLL / top-1,2,5 を rate と token 数の両方 / mean rank）、gate trajectory（mean-of-means、min / max head mean、mean `g<0.1`、完全飽和 head 数）、R1–R5 の per-seed verdict、README 雛形 |
| P3 | analyzer の回帰 self-test | 既存 committed tree から既知値を再計算して一致を確認する。最低: 1.5x full-8000 の step 1750（ΔVal NLL −0.030889 / ΔDev NLL +0.018780）・step 3000（−0.053659 / +0.025505）、stress grid 20 cell の ΔBalanced。2026-09-28 の split-level audit は `build/g1-lr-stress-audit/*.ps1`（ignored）で行ったので、**同じ集計を scripts 側に移植して残す**ことが条件 |
| P4 | order prefix test | `order(3000)` が `order(8000)` の prefix であることと、run 間の order / dataset hash 一致を確認する host 側 check（python で足りなければ `host_tests` に追加）。`kNprtCanonicalTrainingOrderSeed` と data order を seed に依存させる変更は禁止 |
| P5 | gate diagnostics | 既存の host tool `host_tests/headwise_g1_gate_diagnostics.cpp`（`build/host-tests/headwise_g1_gate_diagnostics.exe`）が checkpoint 単位で `gate-static-step<step>.txt` を生成する。seed 依存がなく path 展開だけで使える。追加実装は不要 |

## 予算（実測根拠）

- eval: `evaluation_total_seconds = 233.1857515`（256 + 256 chunk 1 回）→ 7 eval ≒ 27 min / arm
- training: continuation 実測 0.40–0.52 s/update、fresh 500-step grid の wall 770–904 s から
  固定オーバーヘッド ≒ 550–680 s → 3000 step ≒ 33 min / arm
- **arm あたり ≒ 60 min**。seed 2・4 の 4 run ≒ 4.0–4.5 h、1 / 2 tie で exploratory の
  seed 3 を足す場合は ≒ 6–6.5 h
- disk: checkpoint 12 本 / arm ≒ 7 MB／本（grid tree 実測 0.27 GB / 40 本）→ 4 arm で ≒ 0.35 GB、
  すべて ignored な `build/g1-1p5x-multiseed-3000/` 以下
- device session は 2 分割を想定（session A: seed 2 の 2 arm、session B: seed 4 の 2 arm。
  session C は 1 / 2 tie のときの exploratory seed 3 のみ、`-AllowExploratorySeed`）。
  device lock と active-run check は session ごとにやり直す

## 安全条件と検証

1. 長時間 training は **Tier 3**。ユーザーの明示指示なしに開始しない。UI 前面化、
   `EXCLUSIVE_BENCHMARK`、通知 / permission 変更、app data 削除は行わない
2. 実機前に online endpoint を安定識別子で 1 台に解決し、正式 endpoint を 1 つ選ぶ。
   active training が無いことを確認して fail closed。`am force-stop` / `pm clear` / reboot を
   勝手に実行しない。device lock（`.hexatrain-device-lock`）を取り、owner を残す
3. QAIRT は pinned root / Build ID `2.48.40.260702151143` のみ。自動 fallback と 2.47 との
   混在を禁止。QNN 有効 build 後は APK audit を通す
4. 検証順: script / analyzer 変更ごとに `verify.ps1 -Profile Fast` → analyzer・host test・
   order prefix test を `Host` → packaging を触ったら `Android` → `-Mode Smoke`（8 update）で
   Tier 2 の device smoke → その後に本 run。QNN graph を触らないので `Qnn` は通常不要
5. **QNN return code の成功と tensor の有限性は別々に確認**する。全 run で
   `status=SUCCESS`、`completed_steps=3000`、両 split の `*_nonfinite_chunks=0`、
   `api_trace_graph_execute_failure_count=0`、HVX failure / fallback / nonfinite = 0、
   `cpu_fallback=false`、thermal status 0、focus takeover 0 を一次レポートから確認して書く
6. 表現: 「学習 step の数値演算を HTP で実行した」まで。NPU-only と主張しない。
   wall ratio を speedup と読まない。per-update loss / gradient-norm は出ないため
   checkpoint 間の spike は観測不能であり「spike なし」とは書かない
   （muon-hybrid report の既知の測定制限）

## 成果物と commit

- 一次 evidence: `docs/results/g1-1p5x-multiseed-3000-2026-09/seed{2,4}/{control,g1}/` の
  `eval256-step*-htp.txt`、`seed<N>-l19-v1024-t32-d64-f128-steps3000-result.txt`、
  G1 の `gate-static-step*.txt`
- analyzer 生成 CSV と README は同じ tree の root に置く（一次ではない旨を README に書く）
- raw checkpoint、gate-diagnostic 入力、logcat、per-run log、ADB endpoint、絶対 path は
  commit しない。検証生成物は `build/` 以下だけにあること
- commit message は `docs(research): …` / `feat(scripts): …` の形、自分が今回の変更だけ stage

## 実装状況と実行コマンド

上記の 2–4（runner・analyzer・order prefix / hash assert）は実装済み。
5 の device smoke と 6 の本 run はそれぞれ Tier 2 / Tier 3 なので未実行のまま。

Tier 2 smoke の初回（seed 2 / Control / 8 step）は QNN `6031`（`QAIRT_GRAPH_ERROR_ABORTED`）で
FAILED し、原因未同定の incident として
[docs/g1-1p5x-multiseed-tier2-incident.md](g1-1p5x-multiseed-tier2-incident.md) に記録した。
同文書に graphExecute call の意味づけ、single-flight 2 系統の対応、unscoped adb の全列挙、
再実行の条件と分岐を書いている。**Tier 3 は incident が閉じるまで開始しない。**

analyzer は一次レポート（`eval256-step*-htp.txt`、`seed<N>-l19-v1024-t32-d64-f128-steps3000-result.txt`、
G1 の `gate-static-step*.txt`）だけを読み、`quality-split-level.csv`、`gate-trajectory.csv`、
`run-health.csv`、`run-identity.csv`、`verdicts.csv` を out dir に書き、split-level paired 表と
R1–R5 verdict と decision を markdown で stdout に出す。exit code は 0 = problem なし、
2 = identity / health problem あり（claim 不可）、3 = 一次 evidence の欠損または不正。
`--allow-problems` は報告のために 0 に下げるフラグなので、通常は付けない。

回帰 self-test（実機不要）は、training order の prefix 安定性、`g1-lr-stress-2026-09` の
20 cell の ΔBalanced bpb、`g1-1p5x-full-8000-2026-09` の 16 checkpoint の paired delta、
step 1750 / 3000 の ΔVal / ΔDev NLL と top-1 token 差、および gate aggregate の README anchor
を commit 済み一次データから再計算して照合する。

```powershell
$qa = @{ QairtSdkRoot = 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702'
         ExpectedBuildId = '2.48.40.260702151143' }

# 0) commit 前の gate: 回帰 self-test（device を触らない）
.\scripts\run_g1_1p5x_multiseed.ps1 -SelfTest @qa

# 1) plan 確認のみ（device を触らず、ファイルも書かない）
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Plan -Seeds 2,4 @qa

# 2) Tier 2 device smoke（8 update、eval なし。Control だけでなく G1 も 1 arm 通し、
#    G1 固有の gate 診断と analyzer 入力が実機で出ることを先に確認する）
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Smoke -Seeds 2 -Arm Control @qa
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Smoke -Seeds 2 -Arm G1 @qa

# 3) Tier 3 本 run（ユーザーの明示承認後。1 session = 1 arm ずつ）
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Seed -Seeds 2 -Arm Control @qa
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Seed -Seeds 2 -Arm G1 @qa
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Seed -Seeds 4 -Arm Control @qa
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Seed -Seeds 4 -Arm G1 @qa
# 全 arm の一次レポートが揃ったら解析のみ実行。Mode All は Seeds x arm の全 run を続けて走らせる
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Analyze @qa

# 生成物: docs/results/g1-1p5x-multiseed-3000-2026-09/{seed2,seed4}/{control,g1}/ と
#        その root の CSV + analysis.md
```

`-Seeds` / `-EvalSteps` は配列リテラル（`-Seeds 2,4`）でもカンマ文字列（`-Seeds '2,4'`）でも
受け付ける。`powershell -File` 経由では配列リテラルを渡せないため文字列形式を使う。
`-Mode Smoke` は 8 update で eval を走らせないので R1 / R2 band チェックを省略する。

seeds は事前登録の 2 / 4 だけが通常実行の対象である。追加 seed（例: 3）は protocol event
なので、`-AllowExploratorySeed` を明示的に付けたときだけ実行できる:

```powershell
# 1 / 2 tie の tie-breaker 専用。exploratory として記録され、R1–R5 には入らない
.\scripts\run_g1_1p5x_multiseed.ps1 -Mode Seed -Seeds 3 -Arm Control -AllowExploratorySeed @qa
```

runner は `seed-registry.json`（`build/g1-1p5x-multiseed` と results tree の両方）に role を
累積記録し、各 arm の `arm-identity.json` にも `seed_role` を残す。analyzer はその registry を
根拠に role を決め、registry に無い seed ディレクトリは exploratory として扱う。

runner は次に該当すれば fail closed で止まる: `SEED_NOT_FRESH`（seed ≤ 1）、
`SEED_NOT_PREREGISTERED`（2 / 4 以外を `-AllowExploratorySeed` なしで指定）、
`ARM_REQUIRED`（`-Mode Seed` で `-Arm All`）、`STEPS_EXCEED_DECAY_START`（peak LR 区間が
崩れる）、`EVAL_STEPS_MISSING_R1_BAND` /
`EVAL_STEPS_MISSING_R2_BAND`（verdict band が観測できない eval steps）、
`CHECKPOINT_MISSING` / `EVAL_REPORT_MISSING` / `GATE_REPORT_MISSING`、
`MULTISEED_HARD_STOP`（status・finiteness・HVX・focus takeover を一次レポートから別々に確認）、
`DEVICE_LOCK_HELD`（既存 lock の owner を残して停止）。device lock は run 内で 1 度だけ取り
`finally` で解き、thermal status が高い間は arm を開始しない。arm の開始順は seed ごとに
入れ替える（Control 先行 / G1 先行の交互）。

## Codex への引き継ぎ指示

```text
G1 1.5x の multi-seed replication を設計どおり準備し、実行は Tier 3 承認待ちで止めること。

1. docs/g1-1p5x-multiseed-3000.md を読み、Protocol / 主判定 R1–R5 / Decision rule を守る。
   arm は 1.5x に確定済み（一次データからの選定根拠も同文書）。seed は 2 と 4。
2. scripts/run_g1_1p5x_multiseed.ps1 を新規作成し、seed を param で通す。
   既存 run_headwise_g1_lr_stress.ps1 / run_g1_1p5x_long.ps1 / run_g1_1p5x_full8000.ps1 の
   seed1 リテラルと安全分岐は変更しない。Steps=3000、CheckpointInterval=250、
   eval steps {500,1000,1500,1750,2000,2500,3000}、LR は Muon 0.0075 / Aux Adam 0.0033 /
   target 0.00015、linear_decay 4000→8000 / total 8000、fresh step-0、QAIRT pinned。
3. scripts/g1_multiseed_analyze.py を新規作成し、一次 eval / result / gate-static だけから
   seed × step の split-level paired 表（bpb、NLL、top-1/2/5 を rate と token 数の両方、
   mean rank）、gate trajectory、R1–R5 verdict を出す。回帰 self-test として
   g1-1p5x-full-8000-2026-09 の step 1750 / 3000 の ΔVal / ΔDev NLL と
   g1-lr-stress-2026-09 の 20 cell ΔBalanced を再計算して既存値と照合する。
   build/ 以下の使い捨てスクリプトに依存したまま終わらせない。
4. order prefix check を host 側で追加し、run 間の training_order_hash / dataset_hash 一致を
   assert する。data order を model seed に依存させる変更は禁止。
5. 検証: Fast → Host → device smoke（-Mode Smoke、8 update、Tier 2。Control と G1 の両 arm）。
   FAIL を PASS にしない。
6. ここまでを commit して報告し、実機 3000 step run は開始しない。長時間 training は
   Tier 3 なのでユーザーの明示指示を待つ。指示が来たら session 分割
   （seed 2 → seed 4、1 session = 1 arm、開始 arm は seed ごとに交替）で実行し、
   device lock と active-run check を session ごとに確認する。第 3 サンプルは 1 / 2 tie の
   ときだけ seed 3 を -AllowExploratorySeed で追加する（exploratory、主判定には混ぜない）。
```

