# G1 1.5x multi-seed: Tier 2 smoke incident（6031 ABORTED）— 原因未同定

Status: **BLOCKED / 未解決。** Tier 3（3000 step 本 run）は開始していない。
本文書は incident 記録であり、**結果 evidence ではない**。R1–R5 の判定材料に使わない。

## 事象

`-Mode Smoke -Seeds 2 -Arm Control`（8 update、eval なし、Tier 2）が FAILED。

| 項目 | 値 |
| --- | --- |
| run id | `20261001-020339-627` |
| 失敗 | `status=FAILED` / `error=generalized tiny training graphExecute=6031` |
| 分類 | `test=nicopedia_muon_fwd_bwd`、`failure_code=AssertionError` |
| QAIRT | pinned `2.48.40.260702151143`（`check_qairt` = build id match / core complete、inventory は exit 3 の advisory） |
| APK audit | `status=SUCCESS`（arm64/V81、Stub/Skel hash 一致、2.47 文字列なし、host path なし） |
| device 側 health | `focus_takeover_count=0`、activity 系 0、`cpu_fallback=false` |
| 一次レポート | private（ignored）。`build/g1-1p5x-multiseed/incident/` に事後コピー済み。device 側 `files/headless/reports/20261001-020339-627-result.txt` |

`6031` は installed QAIRT header（`include/QAIRT/QairtGraph/QairtGraph.h`）で
`QAIRT_GRAPH_ERROR_ABORTED` = 「observed signal object に `QnnSignal_trigger` が発行されたため早期 abort」。
`6033 TIMED_OUT` ではない。

## 調査 1: graphExecute call index 24 の意味（確定）

L19 Muon 経路（`app/src/main/cpp/qnn/qnn_transformer_training.cpp`）:

- `runtime.prepareTinyTransformerTraining` は graph create/finalize のみで **execute を出さない**（7305–7313）
- 学習ループは `for (uint32_t step = resumeStep + 1; step <= steps; ++step)`（7345）、
  その内側に `for (uint32_t batch = 0; batch < 8; ++batch)`（7350）があり、
  **1 micro-batch = 1 graphExecute**。`qnnExecuteCount` も batch 単位で増える（7370）
- 各 batch の execute は `executeTinyTransformerTraining` → generalized FULL execute
  （`qnn_runtime_transformer_training_generalized_execute.inc:185` の単一 `graphExecute`）
- HVX Muon 更新は RPC（7437–7460）であり graphExecute を消費しない

したがって `steps=8` は 8 step × 8 batch = **64 execute** を予定し、trace の
`attempt_count=25 / success=24 / failure=1 / first_failure_call=24` は

- call 0–23 = **step 1–3 の全 batch（3 step × 8 batch）が成功**
- call 24 = **step 4 の最初の micro-batch** で 6031

を意味する。中止は「終了処理直前」ではなく**学習ループの途中**で起きた。
診断文字列も成功 call の `generalized_tiny_training_qnn_return=0`（+ poison 0 / nonfinite 0 /
all_written=true）の直後に `=6031` が続く形で、abort は QNN 内部で発生し
PhoneLM 側の finite/poison 検査は走っていない。

## 調査 2: `single_flight_result=ALREADY_RUNNING` と host clean gate の関係（説明済み）

食い違いではない。`app/src/androidTest/java/com/yuubinnkyoku/phonelm/HeadlessDeviceTestRunner.kt`
では、lease 取得失敗（実競合）は 51–53 行で `throw AssertionError("ALREADY_RUNNING existing_status=...")`
になり、**report を書かずに終わる**。report に残る `single_flight_result=ALREADY_RUNNING`
は 74–79 行の**自己排他プローブ**（`state.acquire()` を 2 回目に呼び、lease が返らず
自分の run_id が見えることを確認する）が成功した証拠ラベルであり、定数として出力される。

実際、committed 済みの全 run（`docs/results/g1-1p5x-full-8000-2026-09/*/eval256-*-htp.txt`、
`g1-lr-stress-2026-09`、`g1-identity-init-500-2026-09` など）にも同じ
`single_flight_result=ALREADY_RUNNING` が入っている。host 側
`headless_single_flight_gate=True reasons=CLEAN_NO_ACTIVE_EVIDENCE status=terminal` は
直前 run の terminal status + process/service/activity 不在を見ており、両者は同じ状態
（他に active run なし）を指している。したがってこの点は再実行を止める理由にならない。

## 調査 3: `-s` なし adb の全列挙

`scripts/` と `support/` の全 `.ps1` を走査した生の（endpoint 未指定の）adb 呼び出し:

| 種別 | 箇所 | 本実験への影響 |
| --- | --- | --- |
| **awake / stayon helper（training path）** | `scripts/run_g1_1p5x_multiseed.ps1:272–277`（4 コマンド） | **あった**。offline transport が 1 本でもあると 4 行とも `more than one device/emulator` で失敗し、`stayon` / `deviceidle disable` が無言で効かなくなっていた。本 incident の修正で endpoint 明示 + 失敗可視化に変更 |
| 同型 helper（別 runner） | `scripts/run_headwise_g1_lr_stress.ps1:209–216` | なし（本実験では未使用）。同型の潜在欠陥として残置、follow-up 候補 |
| device 解決（`& $adb devices`） | `run_qnn_*` の各 runner（`devices` のみ） | なし。transport 一覧取得は設計上 unscoped で、`-s` を必要としない |
| formal runner の reattach ループ | `scripts/run_qnn_resumable_formal.ps1:1185–1186`（`shell pm list packages` / `run-as ... cat`） | なし（Tier 3 formal 専用、本実験では未使用）。follow-up 候補 |

training path（`run_g1_1p5x_multiseed.ps1` → `run_nicopedia_htp_training.ps1` /
`run_nicopedia_htp_eval.ps1` / `nicopedia_runner_common.ps1`）の他の全呼び出しは
`Invoke-PhoneLmAdb -Device ...` 経由で `-s` が付く。修正後の active runner に raw な
`& $adb` は 0 件。

## 未同定のまま残るもの（推測と事実の区別）

- **6031 のトリガは未同定。** PhoneLM は全 `graphExecute` で signal 引数を
  `nullptr, nullptr` で渡しており（generalized execute 185 行ほか）、PhoneLM 自身が
  signal を発行する経路は無い。これは「PhoneLM 側の signal 発行ではない」ことの
  コード上の確認であり、abort の原因特定ではない。
- 仮説（いずれも未検証）: HTP 内部の自走 abort、DSP 側の一時的状態、Doze / idle 遷移と
  の相互作用（awake helper が無言で失敗していた期間と重なる）。仮説を結論として書かない。
- ADB transport 断は数値失敗として扱わない規約があるが、本件は
  device が terminal な `FAILED` report を書いており transport failure ではない。

## 再実行の条件と分岐（ユーザー合意）

条件: call 24 の意味づけ済み（本文書）／`ALREADY_RUNNING` の説明済み（本文書）／
training に関係し得る unscoped adb が 0（本 incident の修正）／seed-registry 修正後の Fast PASS／
device lock・process・terminal status が clean。

分岐:

- 同一 Control 8-step で **6031 再現** → G1 実験から切り離し、QNN / runner incident として
  継続調査。Tier 3 は BLOCKED のまま
- **6031 非再現** → 単発 abort として記録し、G1 Smoke（`-Arm G1`）へ進む
- **Control と G1 の両 Smoke 成功** → そこで初めて Tier 3（3000 step 本 run）へ進む

offline emulator は切断しない（先に消して「直った」ことにすると、原因が
adb 多重接続か signal abort か single-flight 状態か分からなくなるため）。

## 結果: 再実行は 6031 を再現せず（2026-10-01）

上記 5 条件（host epoch と device epoch の skew 実測 ≒ 0 秒を含む cleanliness 確認）を
満たしたうえで同一 Control 8-step を 1 本だけ再実行したところ:

- `PASS NICOPEDIA_HTP seed=2 layers=19 steps=8`、`SMOKE1_EXIT=0`
- `api_trace`: attempt / success / failure が全 64 execute（8 step × 8 batch）で 64 / 64 / 0、
  `last_result=0`
- 修正後のおかげで `more than one device/emulator` は **0 件**、
  `device_awake_and_idle_disabled=true`（以前は同じ 4 コマンドが無言で失敗していた）

したがってこの 1 本は **単発 abort として記録**する。**原因は未同定のまま**で、
「修正で治った」とは書かない（証明できない）。分岐に従い G1 Smoke へ進む。

再現しなかった残り候補（未検証）: Doze / idle 遷移（awake helper が無言失敗していた期間と
重なる）、HTP 内部の一時状態。6031 が再発した場合の再開条件は上記分岐をそのまま踏む。

## 2 回目の発生: Tier 3 arm 1 で再現（2026-10-01）→ Tier 3 は再度 BLOCKED

事前登録どおりの順序の 1 本目（seed 2 Control、3000 step、eval 7 点、role=preregistered、
APK は本日 audit 済み SHA-256 を `-SkipBuild -SkipInstall` で固定、起動時
`Assert-PhoneLmInstalledApkMatches` 通過）が同一エラーで失敗した。

| 項目 | 値 |
| --- | --- |
| run id | `20261001-135414-572` |
| run status | `status=FAILED`（device）／host は `TIER3_ARM1_CAUGHT: NICOPEDIA_HTP_FAILED` |
| error | `generalized tiny training graphExecute=6031`（**同一**） |
| execute | `attempt_count=609 / success=608 / failure=1`、`first_failure_call=608`（0-based） |
| 時刻 | start から最終 heartbeat 80.1 s、host progress 62 s 行の直後。**checkpoint は 1 本も無し**（interval 250 で最初の checkpoint 以前） |
| 原因 | **未同定**（下記の観察は因果を主張しない） |

**call 608 の意味づけ（同じ算術）**: `608 = 76 step × 8 batch` なので、call 0–607 は
step 1–76 の全 batch が成功し、**call 608 = step 77 の最初の micro-batch**。中止は学習
ループの途中で、これも「batch 0」。

**失敗レポートに存在する項目（健全性として確認できるもの）**: runtime QAIRT identity は健全 —
`compile_time_sdk_build_id=2.48.40.260702151143`、`backend_build_id_match=true`、core API
2.37.0 / HTP 5.48.0、backend/device/context/graph create・finalize すべて result=0、
`qnn_skel_expected == qnn_skel_actual`（`reused`）、`cpu_fallback=false`、
`api_trace_fallback_attempted=false`、`failure_injection_enabled=false`、
`focus_takeover_count=0`。host 側では起動時の pinned 引数一致と APK SHA-256 一致（本日 audit
と同一）が通過している。

**失敗経路では書かれない項目（absent = 済みではない。成功時のみ host/device が書く）**:
`qnn_return_code_success`、`output_tensors_finite` / `all_steps_finite` / `final_finite`、
`nan_detected` / `inf_detected`、`hvx_rpc_failure_count` / `hvx_fallback_count` /
`hvx_nonfinite_count`、`dataset_hash` / `training_order_hash` / `training_order_seed`、
`completed_steps`、host 追記の `compile_time_qairt_build_id` / `attention_gate` /
`parameter_count` / `checkpoint_format`。**これらの absent を成功と読まないこと。**

**観察（因果ではない）**: 2 回とも (a) ある step の **batch 0** で失敗（step 4 → step 77）、
(b) 30 s 境目の progress / heartbeat 行の直後（32 s 後 → 62 s 後）、(c) いずれも seed 2
Control、(d) いずれも checkpoint 以前、(e) 直前までの 609 中 608 execute が成功し
create / finalize / skel / backend は正常。偶然の一致かもしれず、batch 0 か 30 s 境界かを
原因と断定しない。

**分岐の適用**: 合意された分岐「6031 再現 → G1 実験から切り離し QNN / runner incident として
継続調査、Tier 3 は引き続き BLOCKED」に該当。よって本 incident は **2 回目の発生で
「単発 abort」という前回の分類を破棄**し、arm 2–4 は開始していない。R1–R5 は計算していない。
途中の Val / Dev / gate の品質値は一切読んでいない（health メタデータのみ）。

**証拠保全**: `build/g1-1p5x-multiseed/incident/tier3-blocked-20261001-135414-572-*`
（一次レポート、status.json、host log）。private のまま、results tree には昇格していない。
次に必要なのは 6031 の原因切り分けであり、実機 run 再開は新しい指示があるまで行わない。


