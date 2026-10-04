# G1 1.5x multi-seed: Tier 2 smoke incident（6031 ABORTED）

Status: **`UNRESOLVED / DORMANT / WATCH`**

本 incident は **`RESOLVED` でも「修正済み」でもない**。root cause は未同定のまま。
再発時は即 incident lane へ戻る。本文書は incident 記録であり、**結果 evidence では
ない**。R1–R5 の判定材料に使わない。

## watch 状態への移行（2026-10-04）

**追加の原因究明を主作業とする継続は停止する。** 当面の優先順位は
**G1 multi-seed 研究 > 6031 追加原因究明** とする。ただし **6031 が再発した瞬間に
この優先順位を反転する**。

### 移行の根拠（実測した事実）

- 6031 の発生は历史上 **2 回**（step 4 batch 0 / step 77 batch 0）
- HexaTrain 側の QnnSignal trigger site は **0 件**、対象 graphExecute は
  signal 引数を `nullptr, nullptr` で呼んでいる
- **Control 128-step 診断が 4 本連続成功**（full trace 2 本 + flight trace 2 本）。
  各 run とも `graphExecute 1024/1024` 成功、6031 = 0、finite、CPU fallback なし、
  HVX failure なし、**signal invariant 違反 0**
  （`qnn_signal_argument_nonnull_count=0` / `hexatrain_signal_trigger_count=0` /
  `trace_overflow_count=0`）
- **4 本では 6031 は非再現**

### 移行後の運用

- **「直った」「修正済み」「原因解決」とは記載しない。** 非再現は negative evidence
  であり解決の根拠ではない。root cause は未同定のまま保持する
- 追加の QAIRT profiling / event trace、heartbeat A/B、FastRPC A/B は、**6031 が再発して
  新しい timing evidence が得られた場合に初めて検討する**。現時点では追加 incident run を
  無目的に増やさない
- **6031 が 1 回でも再発したら即座に Tier 3 を BLOCKED に戻す**。後続 arm を開始せず、
  品質値を読まず、当該 run を private incident evidence として保存する
  （failure execute index / step / batch / api trace / status / health /
  preceding checkpoint・progress を先に記録）。その後、本 incident で完成した
  flight recorder と analyzer をそのまま使う incident diagnostic lane へ戻る
- 6031 以外の QNN error / nonfinite / fallback / identity mismatch も同様に fail closed とし、
  **6031 とは別の failure として分類する**
- 通常 quality run では incident trace を**常時有効化しない**。G1 multi-seed Tier 3 は
  **従来の通常条件で実行**し、6031 を避けるために quality experiment の timing を変えない

### incident tooling は削除・簡略化しない

以下は watch 状態に遷った後も**そのまま保持**する。再発時の切り分け能力の源泉であり、
簡略化は行わない:

- full trace
- flight recorder
- incident analyzer（`scripts/incident_6031_analyze.py`）
- signal invariant
- synthetic fixture / selftest（`scripts/incident_trace_selftest.py`）
- incident diagnostic runner（`scripts/run_incident_6031_diagnostic.ps1`）
- 6031 legacy evidence（2 件の過去 primary report）
- fail-closed guards

## 分類の根拠

- 過去に 6031 を **2 回**観測済み（step 4 batch 0 / step 77 batch 0）
- **root cause は未同定**
- full trace Control 128-step ×2 本で **非再現**
- flight trace Control 128-step ×2 本で **非再現**
- 計 **4 本連続成功**。うち flight run は **1024/1024 graphExecute 成功**
- **signal invariant = 0**（`qnn_signal_argument_nonnull_count` /
  `hexatrain_signal_trigger_count` の両方）
- **trace overflow = 0**
- CPU fallback / HVX failure / nonfinite なし
- **再発時に利用できる full/flight recorder・analyzer・fixture が完成済み**

**「instrumentation が 6031 を隠していた」は否定も肯定もされていない。** flight mode
でも再現しなかったため、instrumentation が無実犯人であった可能性と、真の原因が
依然として稀である可能性の両方が残る。4 本の連続成功は negative evidence であり、
解決の根拠ではない。

## 決定（2026-10-02）

**原因未同定だが、instrumentation 整備後 4 本連続診断が成功したため、本研究を再開する。
再発時は即 incident lane へ戻る。**

追加の 128-step 繰り返し、QAIRT profiling、heartbeat A/B 等は**現時点では実施しない**。

優先順位は当面 **`G1 multi-seed 研究 > 6031 追加原因究明`** とする。ただし
**6031 が再発した瞬間にこの優先順位を反転する**。

再開した G1 1.5x multi-seed Tier 3 は **通常の quality run** であり、
**incident trace を有効化しない**（6031 回避目的で quality experiment の
timing や条件を変更しない）。

### 再発時の手順（fail closed）

6031 が **1 回でも再発**したら即座に Tier 3 を BLOCKED へ戻す:

- 後続 arm を開始しない
- quality 値を読まない
- 失敗 run を成功扱いしない
- private incident evidence を保存
- 本研究で完成した **flight recorder / analyzer** を使う incident lane へ戻る

6031 以外の QNN failure、nonfinite、fallback、identity mismatch 等も同様に
fail closed とし、**別 incident として扱う**。

---

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

## 静的調査（実機 run なし）: QnnSignal / step 境界 / 30 s 境界（2026-10-01）

### 1. QnnSignal lifecycle: HexaTrain 側の trigger 経路は 0 件

- `app/src`（cpp / inc / h / kt）全体で `QnnSignal`、`signalCreate`、`signalTrigger`、
  `signalDestroy`、`signalHandle` の識別子は **0 件**。SDK 側に `QnnSignal.h` はあるが
  repo は include していない
- `api.graphExecute(` の呼び出しは **24 箇所**。機械照合で **24 / 24 が signal 引数に
  `nullptr, nullptr`** を渡している（`nullptr, nullptr` 以外を渡す site は 0）
- したがって「`QAIRT_GRAPH_ERROR_ABORTED` を HexaTrain 自身の signal trigger が起こした」
  という経路は**コード上存在しない**。6031 の trigger が実在するなら、それは backend /
  HTP runtime の内部（または別主体）である
- stop / cancel は QNN API ではなくプロセス内の `std::atomic_bool`
  （`native_bridge.cpp` の `gRunning` / `gStopRequested`）と Kotlin 側
  `NativeRunArbiter` / WorkManager / UI Stop。headless suite ではこれらの set 経路は動かない。
  学習ループは step 先頭で atomic を load して break（`interrupted`）するだけで、
  QNN handle には触れない
- 次の run で示すべきは「trigger が無いこと」: 各 execute に signal handle が null である
  ことを記録し、report に `signal_trigger_count=0` を持つ（emission ではなく invariant として）

### 2. step 境界の実行順（step N-1 最終 batch → step N batch 0）

`app/src/main/cpp/qnn/qnn_transformer_training.cpp` の L19 Muon ループ（7345–7520）:

1. 最終 batch の `graphExecute` が返る → registry identity 検査 + 勾配累積（CPU、758,528 要素）
2. （G1 のみ）gate aggregate 更新
3. `auxLr` / `muonLr` / `updateConfig` を計算（7420–7429）
4. **HVX Muon update（7432–7460）** = `nicopedia_hvx_muon::update`。`nicopedia_hvx_muon.cpp` は
   `<remote.h>` / `<rpcmem.h>` を使う **FastRPC（DSP）経路**で、custom skel
   `hexatrain_hvx_probe_configure/transport` を persistent session + `std::mutex` で呼ぶ。
   **HTP execute と HTP execute の間に DSP 上で走る唯一の外来コード**
5. `update` の move、`current` / `momentum` / `adamM` / `adamV` 差し替え（7469–7477）
6. finiteness AND 連結（7478–7482）
7. `++completed`、meanLoss、（`step % 25 == 0` のときだけ）curve 追記（7483–7488）
8. **telemetry 書き込み**（毎 step、buffered ofstream、7489–7490）
9. checkpoint 書き込み（**step % 250 == 0 か最終 step のみ**、7492–7504）
10. **progress emission**（`step == resumeStep + 1`、`step % 8 == 0`、checkpoint 時、最終 step、
    7506–7517）→ JNI upcall で Kotlin へ
11. ループ先頭: `stopRequested` atomic load（7346）、`zeroLanguageParameters` で **3 MB 確保**（7348）
12. batch 0: `nprtBatch`（order 読み・one-hot 生成）→ `executeTinyTransformerTraining` →
    registry 検査、APP_WRITE bind 更新（ポインタ再設定 + 128 KB スナップショット）、
    APP_READ 約 30 MB の poison fill → **`graphExecute`** ← 6031 はここ（2 / 77 回）

失敗は 2 回とも手順 12 の最初の execute で、手順 4（DSP 上の FastRPC）と手順 10（JNI upcall）を
含む境界の直後である。

### 3. 30 s 境界: 共有している状態・mutex・lifecycle object は無い

- **device 側 writer 1**: `HeadlessDeviceTestRunner` の heartbeat thread（`Thread.sleep(30_000)`）
  が `HeadlessTestState.write()` を呼ぶ
- **device 側 writer 2**: 同じファイルの progress callback。native が step 1 / 8 step ごと /
  checkpoint / 最終 step で `progress(status.str())` を呼び（7506–7517）、Kotlin は
  `PROGRESS_STATUS_INTERVAL_MS = 1_000L` の throttle で `state.write()` を呼ぶ。
  この経路は **native training thread 上の JNI upcall** として実行される
- `HeadlessTestState.write` は `@Synchronized` + 4096 B 固定長書き込み + `fd.sync()` +
  ATOMIC_MOVE（`Files.move`）。lock は `single-flight.lock` の FileLock のみ
- **QNN handle / context / signal と共有する object・mutex は無い**（native 側は Kotlin の
  state に参照を持たない。native→Kotlin の接触は progress upcall 1 経路だけ）
- host 側: 2 s ごとに `run-as cat status.json`、30 s ごとに checkpoint を `run-as ls`。
  host プロセス側の操作で、QNN と状態を共有しない
- Android watchdog / WorkManager: headless suite は WorkManager を使わない（UI Stop 経路も無し）
- **観察**: 失敗は heartbeat 書き込みの約 2 s 後（32 s / 62 s）。ただし heartbeat は 30 s 周期で
  あり、無作為な失敗が 2 s 以内に落ちる確率は各約 7%。2 例では原因と断定しない

### 4. 608 回の成功境界と 76 → 77 の比較: 既存 artifact では不能

- instrumentation の `stdout.txt` / `stderr.txt` は JUnit の枠（INSTRUMENTATION_STATUS と stack）だけで
  **per-step 行が無い**（41 行）
- host progress は 30 s 粒度、checkpoint 一覧は 250 step 毎、device の
  `learning-rate-telemetry.csv` は per-step 行を持つが **timestamp が無い**
- よって「失敗直前の境界だけ特別だったのか」「heartbeat が重なったのか」を既存 evidence から
  切り分けることはできない。これが instrumentation を足す理由である
- **logcat も使えない**: 事後の read-only dump（`logcat -d -t 6000`、private）では buffer の最新が
  `10-01 00:03:00` までで、`PhoneLMBench` 行 0・`6031` 行 0。失敗時刻 13:54 の backend ログは
  残っていない。次回の incident run では run 前に `logcat -c`、失敗時に `logcat -d` を
  private（`build/`、commit しない）へ退避する

## 4. QAIRT 2.48.40 header / docs による一次資料調査（実機 run なし）

SDK root は `docs/agent/qairt-policy.md` のピン `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`
（build id `2.48.40.260702151143`）。二次情報は使っていない。

### 4.1 6031 の公式定義と_errno_ の食い違い（重要）

`include/QAIRT/QairtGraph/QairtGraph.h`:

```
/// Call aborted early due to a QnnSignal_trigger call issued
/// to the observed signal object.
QAIRT_GRAPH_ERROR_ABORTED = 6031,
```

しかし **同じファイル内の API 契約コメントは 3 箇所とも別の言葉を使う**:
`QairtGraph_execute`（465 行目）/ `executeAsync`（617 行目）/ `finalize`
（310 行目）いずれも

> `QAIRT_GRAPH_ERROR_ABORTED: Execution aborted due to user cancellation.`

**事実**: 列挙体のコメントは「観測された signal object に対する
`QnnSignal_trigger`」、API 契約コメントは「user cancellation」と書いてある。
両者は同じ code を指す一つの enum であり、片方が誤記である可能性と、
backend 内部の signal が「user cancellation」として現出する可能性のどちらも
SDK 内だけでは排除できない。**本 incident ではこの不一致自体が未解決事項**。

### 4.2 signal 引数が NULL のとき 6031 は返り得るか（最重要）

`QairtGraph.h` の `signalHandle` 引数欄（421–423 行目）:

> `signalHandle` Optional signal object used to control execution. **If NULL,
> execution proceeds uninterrupted.**

つまり **ヘッダの記述どおりなら、`nullptr` を渡した execute は uninterrupted に
進む**。我々は 24/24 の call site で `nullptr, nullptr` を渡している。

したがって次の 2 仮説は排他的ではなく、どちらも残る:

- **H1**: backend / runtime が内部 signal を持ち、client signal が NULL でも
  internal abort を 6031 として返す（ヘッダの「If NULL, uninterrupted」は
  client-supplied signal のみを指す、という解釈）
- **H2**: 同一 build の backend がヘッダ契約に反して signal 未指定 execute に
  6031 を返す（契約違反として捕捉する価値がある）

**注意**: `QAIRT_GRAPH_ERROR_SIGNAL_IN_USE = 6030` が別の code として存在する
ため、「signal 関連，但不是 trigger 由来」という route は code 的に可能。
`6033 TIMED_OUT`（timeout 由来）でもない。

### 4.3 SSR（DSP crash）とは別 code

`docs/.../QNN/general/htp/htp_ssr.html`:

> The error code `QNN_COMMON_ERR_SYSTEM_COMMUNICATION` tells the client that
> the CDSP has crashed, and QNN has successfully recovered a connection.

**事実**: DSP crash / SSR は 6031 ではなく `QNN_COMMON_ERROR_SYSTEM_COMMUNICATION`
（recoverable）または `..._FATAL` を返す。したがって

- 2 件の 6031 は **SSR / CDSP crash ではない**（少なくとも報告 code は別）
- SSR なら context / graph handle が invalid になるが、我々の 2 件は
  graph create / finalize / backend / skel identity がすべて正常で、
  直前まで 608 回と 24 回の execute が成功している。**handle invalidation の
  証拠もない**

### 4.4 HTP yielding / pre-emption（HVX FastRPC 仮説に最も関連する）

`docs/.../QNN/general/htp/htp_yielding.html`:

> Yielding and pre-emption of Hexagon clients (QNN Graphs or non-ML use-cases)
> are based on HexagonOS client thread priority. **Every graph in QNN will
> acquire VTCM with a priority specified by `Qnn_Priority_t`.**

つまり QNN graph は優先度付きで VTCM を取得する。higher priority client は
lower priority client を **pre-empt** し得る。我々の step 境界は

```
HTP graphExecute × 8  →  HVX Muon FastRPC（CDSP domain, custom skel）  →  次 step batch 0
```

であり、**CDSP 上の外部コードが HTP graph の合間に、排他状態で走る唯一の
外来コード**。pre-emption が 6031 の内部 abort を誘発しうるかは SDK 記述からは
確認できないが、**候補を「構造的に不可能」と主張できる根拠はない**。

### 4.5 確定していないこと（断定しない）

- QnnSignal の ownership が client / backend のどちら持有的か: header は
  「signal handle は client が create/free」と書く（`QairtSignal.h`）が、
  6031 の実際の emission 主体は不明
- `QnnGraph_execute` の thread safety 記述: `execute` は「synchronous and blocks
  until completion」「if other executions are enqueued, this call will wait in
  queue」とだけ。**我々は同一 graph handle を単一 thread から逐次呼んでいる**
  ので、明示的な並行性違反の証拠はない
- power / context lifecycle の 6031 への寄与: SDK に該当記述なし

## 5. 実装した incident instrumentation（opt-in）

**通常 run の挙動・成果物・コストを変えない**ことを第一条件に設計した。

### 5.1 有効化と無効化

marker file `incident_trace_enabled` が **app-private** の
`files/headless-input/incident_trace_enabled` に存在するときだけ有効。
native 側（`incident_trace::configure`）は cachePath とその親を lookup し、
Kotlin 側（`IncidentTrace.forRun`）は `files/headless-input/` を見る。両者は
同じディレクトリを指す（cachePath は `files/headless-input/<runId>`）。

marker 無しでは各 call site は 1 回の予測可能分岐のみで、ログも生成しない。

### 5.2 記録内容

| 層 | 記録 |
| --- | --- |
| native | per-execute: `execute_id` / `step` / `batch` / monotonic ns / kernel tid / `execute_begin` / `execute_end` / QNN return code / `signal_arg1=null` / `signal_arg2=null` |
| native | step 境界: `stop_check`（`stop_requested` 値つき）/ `zero_parameters` / `batch_prepare` / `poison_fill` / `optimizer`（`rpc_status` `fallback` `output_finite`）/ `parameter_move` / `telemetry` / `checkpoint` / `progress_jni` |
| native | FastRPC: `hvx_lock_acquired` / `hvx_rpc_begin` / `hvx_rpc_end` / `hvx_kernel_metadata` / session open-configure-ready / `hvx_domain_control` |
| Kotlin | `heartbeat_wake` / `heartbeat_write_begin` / `heartbeat_write_end` / `progress_callback_enter` / `progress_callback_exit` / `progress_status_write_begin` / `progress_status_write_end` / `state_write_begin` / `state_write_end`（`reason=startup\|progress\|terminal_status\|terminal_failure`）/ `native_call_begin` / `native_call_end` / `report_write_begin` / `report_write_end` |
| 両方 | `trace_start`: pid / **run_id** / wall clock anchor / monotonic anchor / clock 種別 |

### 5.3 overhead と perturbation caveat（正直に）

- **1 行 = 1 append + 1 flush**。`fsync` / JSON / parse / Java callback は hot path に一切無い
- abort 時に **直前までの行が必ず残る** ことを優先して flush している
- **native と Kotlin は **両方 CLOCK_MONOTONIC**（`steady_clock` /
  `elapsedRealtimeNanos`）を使うので、host 側はオフセットを当てはめずに
  1 本の timeline に載る。加えて **wall clock** も記録してあるので、pull 時刻が
  異なっても skew を後から復元できる
- **残存リスク**: tracing 有効時は per-step あたり数十行の追記が入る。
  flush は page cache への書き込みで HTP execute そのものには触れないが、
  **「tracing が 6031 の発生率を変える可能性」は排除できていない**。
  したがって 2 本が成功しても「instrumentation 附带で 6031 が出なくなった」とは
  書かない（下記停止条件 C 参照）
- Kotlin 側は heartbeat thread / instrumentation thread / native training thread が
  同じファイルに追記するため synchronized。native 側 mutex は contention しない設計

### 5.4 signal invariant

報告と analyzer の両方に明示:

- `qnn_signal_argument_nonnull_count` = **期待 0**
- `hexatrain_signal_trigger_count` = **期待 0**

これらは**観測ではなく invariant** である（`app/src` に trigger site が 0 件なので）。
0 でない場合は analyzer が即 `SIGNAL_*_VIOLATION` として fail closed する。

## 6. incident analyzer（`scripts/incident_6031_analyze.py`）

ログ収集で終わらせない。人間の手計算を不要にするため、以下を自動出力する:

- failure execute の `execute_id` / `step` / `batch` / QNN return code
- 直前の step 境界処理（optimizer / RPC / parameter move / telemetry / checkpoint）
- **HVX RPC 終了 → failure execute の時間差**（us 単位）
- **heartbeat 書き込み終了 → failure の時間差**（ms 単位）
- **progress/status write → failure の時間差**
- `stopRequested` 状態
- signal invariants
- 最も近い外部イベント群（窓は前後 nearest 件に制限）

出力は machine-readable（`incident-timeline.csv` / `incident-findings.json`）と
human-readable（`incident-report.md`）の両方。

**proximity は因果として表現しない。** analyzer の出力には
「shared monotonic clock 上の近接であり、near == cause ではない」旨が
常に明記され、Markdown の該当行も observation として列挙される。

### 6.1 fail-closed 条件

`problems` が非空なら exit 非ゼロ。以下をすべて problem 扱いにする:

| code | 意味 |
| --- | --- |
| `NATIVE_TRACE_MISSING` / `KOTLIN_EVENT_MISSING` | incident mode で必須 trace / anchor が無い |
| `RUN_IDENTITY_MISMATCH` / `RUN_IDENTITY_ABSENT` | trace と status.json の run id 不一致 / 照合不能 |
| `EXECUTE_ID_MISSING` / `EXECUTE_ID_DUPLICATE` | execute id 欠落 / 重複 |
| `EXECUTE_BEGIN_WITHOUT_END` / `EXECUTE_END_WITHOUT_BEGIN` | begin/end 不対（failure execute は特別扱い） |
| `TIMESTAMP_DISORDER` / `TIMESTAMP_UNPARSABLE` | ファイル内 timestamp 逆転 |
| `STEP_MISSING` / `BATCH_OUT_OF_RANGE` | step / batch 不整合 |
| `PHASE_*` | step 境界 phase の pairing 崩れ |
| `SIGNAL_ARGUMENT_NONNULL` / `SIGNAL_*_INVARIANT_*` | signal invariant 違反 |
| `SIGNAL_INVARIANT_NOT_RECORDED` | invariant が「観測」されていない（未違反ではなく未確認） |

timestamp 逆転の検査は **merge 前の file 順**で行う。timestamp でソートした
timeline では検出できないため。

### 6.2 legacy evidence（過去 2 件）

`--legacy` / `--legacy-dir` は instrumentation 以前の primary report を解析する。
**新 instrumentation 相当の値を捏造しない。** 復元可能なのは
`status` / `error` / execute counts / `first_failure_call` / QNN identity /
fallback / focus のみで、step/batch は記録された execute index と
文書化された 1 step = 8 micro-batch から**導出**する（data から推測したのではなく
loop 構造の定数）。

記録されなかった項目は `NOT RECORDED` として列挙し、absent を 0 と読ませない。

実測: 2 件とも **step 4 batch 0** / **step 77 batch 0** を復元（本ドキュメントの
既存記述と一致）。

## 7. 合成 fixture と self-test（`scripts/incident_trace_selftest.py`）

実機不要で analyzer の fail-closed 規則と時間差演算を固定する 14 ケース。
`verify.ps1 -Profile Fast` に組み込まれている。

| fixture | 期待 |
| --- | --- |
| `normal_success` | problem 0 件 |
| `failure_step4_batch0` | 6031 / step 4 / batch 0 / problem 0 |
| `failure_step77_batch0` | 6031 / step 77 / batch 0 / problem 0 |
| `failure_missing_end` | `EXECUTE_BEGIN_WITHOUT_END`（failure を捏造しない） |
| `failure_after_heartbeat` | heartbeat delta ≈ 4 ms + execute duration |
| `failure_after_progress` | progress write delta ≈ 2 ms + execute duration |
| `failure_after_hvx_rpc` | HVX RPC delta ≈ 200 us + execute duration |
| `failure_no_external_event` | 該当 Δ は `n/a`（近接が無いことを捏造しない） |
| `timestamp_disorder` | `TIMESTAMP_DISORDER` |
| `duplicate_execute_id` | `EXECUTE_ID_DUPLICATE` |
| `signal_nonnull` | `SIGNAL_ARGUMENT_NONNULL` |
| `signal_trigger_violation` | `SIGNAL_TRIGGER_INVARIANT_VIOLATION` |
| `identity_mismatch` | `RUN_IDENTITY_MISMATCH` |
| `missing_trace_file` | `NATIVE_TRACE_MISSING` |

結果: **14/14 PASS**。

## 8. 帰宅後の診断プロトコル（実機 1 本）

**この指示期間中に実機 run は 0 本**。以下は接続後の手順。

前提: `.\scripts\verify.ps1 -Profile Fast` の PASS（唯一の FAIL が
`g++ not found on PATH` なら本 commit で_FIX した 6031 analyzer self-test が PASS すること）
と analyzer self-test PASS を確認済み。

### 8.1 1 本だけ: seed 2 / Control / 128 step / incident trace ON

```powershell
.\scripts\run_incident_6031_diagnostic.ps1 -Mode Run `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Seed 2 -Steps 128
```

固定条件（意図的に「怪しい機能を消して成功させる」ことを禁じている）:
batch 8 / Control（gate なし）/ 現行 Muon + HVX FastRPC / heartbeat ON /
progress + status write ON / host polling ON。品質評価なし、checkpoint 無効。

Runner が fail closed で守るもの:

- incident namespace が `docs/results/` 配下に解決したら **拒否**
- `build/` 外なら **拒否**
- trace marker をプロセス起動**前**に配置し、終了**後**に削除
- logcat を run 前 `logcat -c`、run 後（成功/失敗regardless）に `logcat -d` で private へ
- 完了後に analyzer を自動実行（trace があれば）
- G1 quality Tier 3 には一切触れない

### 8.2 1 本目の結果による分岐

**A. 6031 再現** → incident evidence として保全。analyzer の timeline で確定。
追加 3000-step run は禁止。trace で仮説が 1 つに絞れた場合のみ、
Phase G の範囲（最大 2 本の 128-step Control A/B、1 変数ずつ）で切り分け。

**B. 128 step 成功** → それだけで Tier 3 に戻らない。同一条件の
Control 128-step diagnostic run をもう 1 本だけ許可。2 本とも成功なら
「instrumentation 付き 128-step Control が 2 本連続成功」と記録するが、
**既存 6031 の消失原因は未同定**のまま、Tier 3 再開は行わない。

**C. 別の failure mode** → fail closed。6031 と混ぜず別 incident として分類。

### 8.3 解析で見るもの（品質値ではない）

`-Mode Analyze` で同じ directory を再解析できる。見るのは health と
incident timeline のみ:

- status / attempt / success / failure
- QNN return code と failure execute の step / batch / execute id
- signal invariants
- CPU fallback / HVX return code / DSP kernel metadata
- FastRPC timeline / heartbeat / progress timeline
- logcat の backend evidence
- process 残留・lock 解放

## 9. 診断 run 結果（2026-10-01、実機 2 本）

**6031 は再現しなかった。2 本とも 128 step Control が完全成功。**
これは停止条件 C（128-step 診断 run が成功）に該当し、**解決扱いしない**。

| run | steps | status | execute | 6031 | signal invariants | analyzer |
| --- | --- | --- | --- | --- | --- | --- |
| `incident-20261001-222455-298-44272` | 128 | SUCCESS | 1024 / 1024 | **0** | nonnull=0 / trigger=0 | problem 127 件（instrumentation bug、下記 9.2） |
| `incident-20261001-223002-448-7584` | 128 | SUCCESS | 1024 / 1024 | **0** | nonnull=0 / trigger=0 | **problem 0 件（exit 0）** |

device health（両 run 共通）: `completed_steps=128` / `qnn_return_code_success=true` /
`all_steps_finite=true` / `final_finite=true` / `nan_detected=false` /
`inf_detected=false` / `cpu_fallback=false` / `hvx_rpc_failure_count=0` /
`hvx_fallback_count=0` / `hvx_nonfinite_count=0` / `focus_takeover_count=0` /
`api_trace_last_qnn_result=0`。

trace 側の整合性: `execute_begin=1024 / execute_end=1024`（欠落・重複なし）、
`hvx_rpc_begin=128 / hvx_rpc_end=128`、非 0 の `qnn_result` は **0 件**、
`signal_arg1/2` の非 null は **0 件**。

### 9.1 品質値は見ていない

本 2 本は品質評価なし（eval runner を呼んでいない）。読んでいないのは品質値のみで、
health メタデータと incident timeline のみ。R1–R5 未計算、seed 3 未着手、
G1 quality Tier 3 は **BLOCKED のまま**。

### 9.2 この run で instrumentation 自身の欠陥が 3 件見つかった

**実行して初めて分かったもの**であり、コードレビューでは出てこなかった:

1. **namespace 破れ**: `.inc` は `namespace phonelm::qnn` の**内側**で
   include されるため、`incident_trace.h` を `.inc` から include すると
   `::phonelm::qnn::phonelm::incident_trace` と名前空間が二重になり、
   libc の `getpid` も二重宣言されて **10 件の compile error**。
   修正: header は file scope（`qnn_runtime_qairt.cpp`）で 1 回だけ include。
2. **checkpoint phase の非対称**: `checkpoint_begin` は checkpoint を書いた step のみ、
   `checkpoint_end` は毎 step → 127 件の `PHASE_END_WITHOUT_BEGIN`。
   analyzer が正しく検出した。修正: begin/end を checkpoint **判定**を囲む形で毎 step 出し、
   `written=true/false` を持たせる。
3. **trace pull 先の誤り**: native trace は `cachePath`（= `files/headless-input/<runId>/`）に
   書かれるが、pull は `files/headless/` を見ていた。**trace は最初から device 上にあった**。
   修正: 両パスを probe し、`event=` を含む payload のみ採用。

さらに runner 側の引数 2 件も実機動作で露見した（`EvalOnly` は存在しないパラメータ、
`checkpointInterval` の device 側上限は 10000）。いずれも build/install を 1 回消費后才に
判明したため、host 側で fail fast するようにした。Seed も 2 に pin した。

### 9.3 logcat は evidence にならない（本機では）

`logcat -g` は全 buffer が **"0 B readable"**（`adb logcat -g`:
main/system/kernel すべて 0 B）。`logcat -d` / `-b all` も実質 55 行しか出ず、
run の痕跡は残らない。**本機では logcat が有効な evidence channel にならない**ことが
確定した（§4 の「logcat も使えない」の拡張確認）。これは 6031 の原因の手がかりでは
ないが、**logcat 依存の切り分けはこの device では成立しない**ため、次の最小実験は
logcat 以外の channel（trace の Δ、QAIRT profile/event trace、SSR カウンタ）を
使う必要がある。

### 9.4 非再現の分類（解決ではない）

- 「instrumentation 付き 128-step Control が 2 本連続成功」は **事実**
- 既存 6031（step 4 / step 77）が消えた原因は **未同定のまま**
- instrumentation 自身が 6031 の発生率を変えていた可能性は**排除できていない**
  （1 step あたり数十行の per-line flush。§5.3 の caveat がそのまま残る）
- したがって **「解決」と書かない**。停止条件 C として記録し、
  Tier 3 再開は行わない

### 9.5 残る未同定事項（変化なし）

§4 の SDK 一次資料が示す 2 仮説（H1 backend 内部 abort / H2 ヘッダ契約違反）は
**今回の 2 本では切り分けられなかった**。非再現だからである。
batch 0 共通・signal 不変条件 0 は 3 例すべてで一致しており、これは
「batch 0 で起きる」観測を 1 例増やしただけで、原因には何も加えない。

## 10. flight recorder mode（低摂動）— 実装と結果

### 10.0 なぜ flight mode を追加したか

- full trace 2 本（Control 128-step）が **連続成功**、**6031 発生 0**
- 一方 **非 instrumented の Control では 6031 が 2 回**（step 4 / step 77）
- したがって次の問いは「再現の消失了」ではなく **「instrumentation が 6031 を
  隠していないか」**

full mode は 128-step で **10,152 行**を出し、**その 81% が micro-batch 毎 8 行**で
1 行ごとに `fwrite` + `fflush`。Kotlin 側は 1 行ごとに `FileOutputStream` を
open/write/close。**これが 6031 を消していた可能性は現時点で最も有力な説明**であり、
観測能力を保ったまま検証できる。

### 10.1 設計

marker `incident_trace_enabled` は従来通り。有効化は marker の**存在のみ**で、
mode は隣接 sidecar `incident_trace_mode` で選ぶ:

| ファイル | mode |
| --- | --- |
| marker のみ | `full`（従来動作。default） |
| marker + `incident_trace_mode` に `flight` | `flight` |
| sidecar 不在/読めない/未知 | **`full`**（`disabled` には**決して**しない） |

native `flight` は**固定容量 ring に POD record**を書く。hot path に
`std::string` 構築も `to_string` も syscall も mutex もない（lock は `dump()` 内のみ）。
容量 65536（128-step で必要行数 ~4.5k）。満杯時は**最新 record を捨てて損失を計上**する。
最古を上書きすると失敗直前の記録が消えるため。`trace_overflow_count` は dump header に
載り、analyzer が 0 以外を problem 扱いする（**穴のある trace の沈黙は証拠にならない**）。

dump 地点 3 つ:
1. **`graphExecute` が非 0 を返した直後**（ループ unwind 前に必ず通る唯一の地点）
2. 正常 terminal
3. `nicopediaHtpTraining` の forwarder（Muon loop の ~15 early return を 1 箇所で網羅）

Kotlin は preallocated list に buffer し `native_call_end`（`finally` 内）で 1 回だけ書出。
**観測以外の I/O は一切変えない** — `HeadlessTestState.write` / heartbeat thread /
progress・status write は元のまま。

### 10.2 実機結果（seed 2 / Control / 128 step / flight）

| run | status | execute | 6031 | signal | overflow | analyzer |
| --- | --- | --- | --- | --- | --- | --- |
| `…-003843-904-29120` | SUCCESS | 1024/1024 | **0** | nonnull=0 / trigger=0 | **0** | problem 2 件（instrumentation bug） |
| `…-005203-688-6264` | SUCCESS | 1024/1024 | **0** | nonnull=0 / trigger=0 | **0** | **problem 0 件 / exit 0** |

health（flight）: `completed_steps=128` / `all_steps_finite=true` /
`final_finite=true` / `nan_detected=false` / `inf_detected=false` /
`cpu_fallback=false` / `hvx_rpc_failure_count=0` /
`api_trace_graph_execute_success_count=1024` / `api_trace_last_qnn_result=0`。

### 10.3 full vs flight の additional overhead 差

| 指標 | full（2 本） | flight（2 本） | 差 |
| --- | --- | --- | --- |
| native trace 行数 | 10,025 / 10,152 | 4,482 | **−56%** |
| native trace bytes | 1,132,980 / 1,144,319 | 484,102 | **−57%** |
| trace_event_count | — | 4,481 | — |
| overflow | 該当なし | **0 / 0** | — |

diagnostic wall time（**instrumentation cost のみ。品質指標ではない。G1 run と比較しない**）:

| run | mode | wall (s) | ms/update |
| --- | --- | --- | --- |
| `…-222455-298-44272` | full | 264.8 | 2068.9 |
| `…-223002-448-7584` | full | 247.9 | 1936.8 |
| `…-003051-037-37072` | flight | 264.8 | 2068.9 |
| `…-003843-904-29120` | flight | 247.9 | 1936.8 |
| `…-005203-688-6264` | flight | 190.0 | 1484.1 |

flight は確かに速い。ただし **build/install を含む総 time の差**であり、
**HTP execute 単体の時間差として提示できるものではない**（同じ run の内訳がないため）。
**「flight は full より速い」ことは言えるが、「6031 の再現率を左右する perturbation が
消えた」とは言えない**（§10.4）。

### 10.4 結論（重要）

- **`full trace 2/2 success`、`flight trace 2/2 success`** を negative evidence として記録
- **6031 は flight mode でも再現しなかった**
- **原因: unresolved**。「instrumentation が 6031 を隠していた」は **否定も肯定もされていない**
  — flight でも出ないため、instrumentation は **無実犯人** であった可能性と
  **真の原因が依然として稀** である可能性の両方が残る
- したがって §9.4 の停止条件 C は **依然適用**。**G1 quality Tier 3 は BLOCKED のまま**
- 品質値・R1–R5・seed 3 には一切触れていない

> **2026-10-02 追記**: 本節が記録した BLOCKED は、その後**上位の決定
> （`UNRESOLVED / DORMANT / WATCH` として本研究を再開）により解除**された。§9 の
> full/flight 4 本連続非再現は negative evidence として有効であり、原因未同定は
> 変わらない。再発時は本節の recorder/analyzer をそのまま使う。

> **2026-10-04 追記**: 本節の「§9.4 の停止条件 C は依然適用」という記述は、
> 冒頭の「watch 状態への移行」節に**置き換えられた**。現在の運用は:
> full trace 2 本 + flight trace 2 本の**計 4 本連続非再現**（各 1024/1024 execute、
> signal invariant 違反 0）を根拠に **G1 multi-seed 研究を優先**し、6031 追加原因究明は
> 再発時にのみ行う。**「直った」とは記載しない。** root cause は未同定のまま。

### 10.5 flight mode を実装して実機で動かして分かった欠陥（4 件）

実行して初めて見えたもので、レビューでは出てこなかった:

1. **dump header を末尾に書いていた**（しかも現在時刻）→ file 内 timestamp 逆転で
   analyzer が健全な trace を `TIMESTAMP_DISORDER` で拒否。**先に書き、anchor 時刻**に修正
2. **`run_id` が trace のファイル名だった**（`incident-native-trace.log`）→
   identity cross-check が機能しない。`cachePath` basename に修正（full mode と統一）
3. **両 trace を append で open して truncate していなかった** → 3 本目の run の
   Kotlin trace が **398 行・flush header 3 本**になり、前 run の timestamp が
   後に来到 `TIMESTAMP_DISORDER`。**run ごとに trace ファイルを所有し truncate** するよう修正
4. **signal invariant を `invariant=<name> value=<n>` 形式で出していた** →
   analyzer は parsed **key** として invariant 名を探すので wrapper に隠れ
   健全な trace が `SIGNAL_INVARIANT_NOT_RECORDED` に。**bare key=value** に修正

1 と 4 は analyzer の fail-closed 判定が**自分の実装のバグを検出した**例であり、
fail-closed を捨てずに維持した根拠になる。




`-Mode Analyze` で同じ directory を再解析できる。見るのは health と
incident timeline のみ:

- status / attempt / success / failure
- QNN return code と failure execute の step / batch / execute id
- signal invariants
- CPU fallback / HVX return code / DSP kernel metadata
- FastRPC timeline / heartbeat / progress timeline
- logcat の backend evidence
- process 残留・lock 解放

