# G1 Shared Backend/Device Lifetime — Monitor / Transport Recovery and Formal Re-entry Blockade

**Status: `MONITOR_REPAIR_HOST_QUALIFIED`, `TRANSPORT_REPAIR_HOST_QUALIFIED`, `DEVICE_ENTRY_GATE_PASS`, `FORMAL_PREREQUISITE_BLOCKED`, `FORMAL_COMPLETED = 0/3`.**
**`NUMERICAL_PILOT_VERIFIED` is retained. `MONITOR_CONTRACT_FAILED` and `POST_EXIT_TRANSPORT_INCOMPLETE` remain the historical classifications and are not rewritten as PASS.**

本文書は `docs/g1-backward-shared-lifetime-formal.md` の続編である。前回の five-role pilot は数値・resource としては PASS したが、strict 2 秒 monitor 契約と post-exit evidence transfer の 2 点で protocol qualification が BLOCKED となった。今回の mission はその 2 点だけを host-side で修復し、Host 検証、dry run、独立 assessor、公開集計までを閉じた。2026-10-23 に同じ物理 device の endpoint が復帰し、entry gate / dry run / fresh Qnn / entry assessor は PASS したが、v3 runner が要求する independent prerequisite（`certify_v2 --entry formal-1`）が保持済み `mission-stop-private.json` を理由に FAIL したため、formal OS process は 1 つも起動していない。

- 対象: research clone（`PhoneLM-g1-long-linear-200k`）
- branch: `investigation/g1-backward-shared-lifetime-formal`
- base: `900e949dc5eee62cb230af39136f96a45df1824e`
- 候補: `shared-lifetime-five-role-final`（probe `6ffeec21…`、DSP `658bd900…`、Host `0aea5fe5…`、compiled source `9ceb0324…`、gradients `c742bfa2…`、plan `d42e0ecc…`）
- 変更したのは host orchestration のみ。算術、graph、fixture、observer、Backend/Device lifetime、APK、QAIRT 2.48.40.260702151143 は不変。

---

## Re-entry observation and formal prerequisite gate (2026-10-23)

同じ物理 device の wireless-debug endpoint が復帰したため、read-only で再入口を確認した。device の数値実行は行っていない。

| 項目 | 結果 |
| --- | --- |
| endpoint | `100.108.106.81:39723` と mDNS の 2 経路。両方 `device` |
| serial | 両経路とも `324753221196`。pilot の `recovery-identity-private.json` と一致 |
| model / device / product | `NX741J` / `PQ85A01` / `PQ85A01-UN` |
| device state | Awake、top は target 外、owned probe / target app なし |
| device mutation | 0（read-only のみ） |
| entry gate | `PASS`（同一物理 device 検証済み、`new_formal_processes=0`） |
| dry run | 10 samples、max start-to-start `1916.0566 ms`、`contract_met=true`、missed/timeouts/invalid/unknown/aux 0、11/11 fail-closed cases、`device_calls=0` |
| entry assessor | `PASS`（exit 0、`device_calls=0`） |
| fresh Qnn | `PASS`、`total 972.2s PASS=4 FAIL=0 SKIP=0` |
| `certify_v2 --entry formal-1` | **FAIL: `mission Device stop active`** |

`certify_v2` は pilot の `mission-stop-private.json` が残存していることを active stop とみなす。この file は recovery assessor が `mission_stop_retained=true` として保持を要求している historical evidence であり、削除・rename しない。v3 runner は line 54 でこの prerequisite を hard requirement として呼ぶため、`formal-1` directory を作る前に `INDEPENDENT_PREVIOUS_STAGE_NOT_PASS` で停止する。したがって device probe は 1 回も起動していない。

さらに `certify_v2` は `orchestration_sha256 == sha(v2 runner)` を要求するため、たとえ entry を通っても v3 evidence を正式認定できない。v3 runner に bind し、保持された historical mission-stop と host-qualified repair の関係を明示的に扱う v3 prerequisite / certifier が未実装である。これは device の安全性や 2 秒・transport 契約を緩和する話ではなく、独立認定経路の追加が必要という blocker である。formal は **0/3**、新規 formal process は **0**、one-process allowance は未使用のまま保持する。

---

## A. Provenance / frozen identities

| 項目 | 値 | 状態 |
| --- | --- | --- |
| base commit | `900e949dc5eee62cb230af39136f96a45df1824e` | 開始時 clean |
| frozen five-role plan | `d42e0ecc6c96f8bda43d0155d4d809481cc683cc2f0de15179c262be14fdba8d` | byte-identical |
| probe | `6ffeec218c2532e6b54c433dbe963f9210ed1beb6c3da4234001a67b7b03126f` | 再照合 PASS |
| `libHexaBackwardRepairDsp.so` | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` | 再照合 PASS |
| `libHexaBackwardRepairHost.so` | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` | 再照合 PASS |
| compiled source | `9ceb0324c2a8d82396d597feb0ac46cd1e5ac3c9` | 不変 |
| gradients | `c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b` | 再照合 PASS |
| v1 runner（欠陥源） | `a0a776659d786f53009f6b65bfac63c5f40769f97800edc1ed9de5a046badd42` | byte-identical で保存 |
| v3 runner（formal-1 用） | `2ac326fcd02db5993190584cbeed0049e0367124eb00982c665fb2f229d69b79` | 新規 |

candidate を再 build していない。したがって pilot の数値 evidence はそのまま有効であり、monitor/transport の失敗によって数値 PASS を失効させていない。

## B. Scope and prohibitions

変更したもの:

1. in-run foreground monitor の scheduling（`scripts/backward_formal_monitor.ps1`）。
2. post-exit evidence transfer（`scripts/backward_formal_transport.ps1`）。
3. 上記 2 つを組み合わせる formal-1 専用 runner（`scripts/run_backward_shared_lifetime_formal_v3.ps1`）。
4. host 検証、host-only dry run、独立 assessor、公開 exporter、Fast gate 配線、本文書。

変更していないもの: 算術 worker、oracle、word gates、observer 契約、digest 契約、graph topology、fixture、Backend/Device の生成・保持・teardown、Context/Graph の lifetime、role 順、APK、SDK。

明示的に禁止したこと: 4 秒観測の PASS 昇格、2 秒契約の緩和、waiver 追加、最大 interval の引上げ、in-run safety monitor を止めるための transport 利用、remote raw / 既存 private evidence の上書き・削除、自動 retry、新規算術、candidate rebuild。

## C. Evidence freeze and integrity

失敗解析の入力は前回 attempt の private raw と remote raw である。今回の mission はそれらを read-only で扱い、freeze 記録だけを追加した。

| 記録 | 内容 |
| --- | --- |
| `evidence-freeze-private.json` | 前回 run の file 一覧、hash、remote 側の状態、変更禁止の宣言 |
| `monitor-cadence-attribution-private.json` | top 37 記録と 33 interval の attribution |
| `transport-attribution-private.json` | `adb pull` 失敗、partial、remote identity、復元性の attribution |
| `device-entry-gate-raw-private.json` | 実機入口の read-only 観測（devices 空、get-state exit 1、TCP unreachable） |

remote 側の原本、partial、logcat、tombstone は削除も再送もしていない。private evidence は append-only で扱った。

## D. Monitor cadence root cause

前回の v1 は `$process.WaitForExit(2000)` を **serial な query block の前後に足し算**していた。したがって周期は 2 秒ではなく「2000 ms 待ち + competition/awake/top の実行時間」になっていた。

| 指標 | 値 |
| --- | --- |
| top 記録総数 | 37（pre 2 + loop 34 + post 1） |
| loop interval | 33 |
| minimum | 3.250952 s |
| median | 4.029617 s |
| maximum | 5.244854 s |
| 2 秒超過 | 33 / 33 |

median の内訳は `4.029617 s = 2.379454 s（top 完了→ps 完了。2000 ms wait を含む）+ 0.99 s awake + 0.36 s top`。maximum は同じ iteration で 10 秒 thermal block も走った場合である。コード経路と monotonic timestamp の算術が一致しており、host-side の root cause は **`ADDITIVE_WAIT_PLUS_SERIAL_QUERY_BLOCK`** と分類する。

これは device 上での A/B 実験による証明ではなく、実測 phase の整合による定量 attribution である。4 秒観測を 2 秒契約の証拠に読み替えない。

## E. Monitor repair design

- 固定絶対 deadline: `deadline_k = previous_sample_start_k + Period - Lead`。
- `Period = 2000 ms`、formal-1 の `Lead = 100 ms`。late completion で deadline を re-base しないため、遅れが次の interval を短くしない。
- serial fast lane は competition → awake → top の順で、1 sample 内で重複しない。
- 各 command に 8 秒の個別 timeout。command timeout、invalid sample、unknown state、missed deadline を別々の counter として記録する。
- thermal は owned child process の auxiliary lane（period 6000 ms、collect bound 4000 ms）。fast lane を待たせない。
- 契約判定は、counter が全て 0、start-to-start が全件 2000 ms 以下、invalid sample が 0 の場合だけ `contract_met=true`。
- device runner では timing violation でも probe を殺さない。readback と identity を先に書き、`safety_complete=false` を残した後で `MONITOR_CONTRACT_VIOLATION` を投げる。asleep / takeover / competing workload / unknown / auxiliary failure は従来どおり `Stop-Owned`。

## F. Monitor host qualification

| 検証 | 結果 |
| --- | --- |
| `scripts/test_backward_formal_monitor.ps1` | case 16/16 PASS |
| module self-test | 14 cases PASS |
| realistic 2000 ms run（captured log） | max start-to-start **1958.36 ms** |
| same test inside Fast gate（fresh rerun） | max **1960.972 ms** |
| host-only dry run | 10 samples、max **1916.2355 ms**、missed 0、timeouts 0、invalid 0、unknown 0、aux failure 0、`contract_met=true`、`monitoring_status=PASS` |
| dry-run fail-closed cases | 11/11 PASS |
| device calls | 0 |

dry run の latency は attribution の median（competition 379 ms、awake 994 ms、top 361 ms）に基づく **emulation** であり、device 実測ではない。その旨を private record に明記した。dry run は 2 秒契約を緩和するためではなく、deadline 算術と fail-closed を device 前に固定するために使う。

## G. Transport failure attribution

前回の readback は `adb pull saved-p-words.bin.gz` の途中で host wrapper の 180 秒 bound に殺された。分類は **`ADB_WRAPPER_TIMEOUT`**。

| 指標 | 値 |
| --- | --- |
| failure operation | `adb pull` of the device-gzipped 5-role container |
| configured timeout | 180 s |
| local partial | 332,333,056 B、sha256 `20cbd99e…` |
| remote gzip | sha256 `430d763a…` |
| remote raw | sha256 `5ba99db6…`、1,401,659,004 B |
| remote raw intact | true |
| five-role raw/gzip identity | 全件 verified |
| container reconstruction | lossless |
| post-exit evidence complete | false |

つまり欠けていたのは **container gzip の完全な local copy** であり、remote raw でも数値でもない。五 role の raw と gzip は個別に identity を照合でき、container は lossless に再構成できた。したがって transport 失敗を数値失敗へ読み替えない。

## H. Transport repair design

- `adb exec-out dd` による binary-safe な固定 byte range 読み出し。
- chunk 8 MiB、chunk timeout 120 s、chunk attempts 2、file budget 2700 s。
- chunk 完了ごとに SHA256 を記録し、再開時は完了 part を再利用する。失敗しても 0 からやり直さない。
- local file は exclusive create。partial と part は失敗時に残す。
- identity は **remote before → host after assembly → remote after** の 3 点で一致を要求する。
- gzip readback を保持する（`certify_v2` が raw と `.gz` の両方を要求するため）。
- transfer は owned probe の exit 後にだけ走る。in-run 2 秒 monitor を pause しない、置換しない、緩めない。

## I. Transport host qualification

| 検証 | 結果 |
| --- | --- |
| `scripts/test_backward_formal_transport.ps1` | case 16/16 PASS |
| module self-test | 23 cases PASS |
| binary fixture roundtrip | PASS |
| elapsed（captured log / Fast rerun） | 1591.3 ms / 2429.2 ms |
| device calls | 0 |

case には short write、retry、hash mismatch、remote raw missing、destination exists、global timeout、remote changed before/after、zero size、unsafe path、per-attempt evidence を含む。

## J. Runner v3 identity and scope

`scripts/run_backward_shared_lifetime_formal_v3.ps1` は `Stage='formal-1'` 以外を拒否する（`RUNNER_V3_SCOPE_FORMAL1_ONLY`）。pilot / formal-2 / formal-3 / performance は既に凍結された runner に束縛されたままであり、v3 で暗黙に置き換えない。

| 検証 | 結果 |
| --- | --- |
| runner self-test | `saved_p_runner_v3_selftest=PASS staging_cases=3 monitor_cases=14 transport_cases=23 device_calls=0` |
| staging collision guard | 3 cases PASS |
| plan record | `runner_sha256 = 2ac326fc…`、`plan = host_tests/fp32_custom/backward_shared_lifetime_monitor_transport_plan.json`、`sha256 = e3a86756…` |
| monitor / transport module sha256 | `2434c833…` / `fa4eded6…` |

v3 は数値・graph・fixture を変更しない。v1 との差は monitor と transfer の 2 点だけである。

## K. Independent assessor and entry gate

`scripts/assess_monitor_transport_recovery.py` は read-only で、device call を一切しない。5 つの verdict を **merge しない**。

| verdict | 現在値 |
| --- | --- |
| `numerical_pilot_verified` | true |
| `monitor_contract_failed`（historical） | true |
| `post_exit_transport_incomplete`（historical） | true |
| `host_repair_qualified` | true |
| `formal_eligible` | false |
| `formal_blocker` | `FORMAL_PROCESS_NOT_RUN`（assessor。runner の independent prerequisite は別途 `mission Device stop active`） |

`--self-test` は `PASS rejection_cases=34 device_calls=0`。`--entry formal-1` は再入口後 **PASS**（exit 0、`device_calls=0`）。`--out` の再評価は `formal_eligible=false` / `FORMAL_PROCESS_NOT_RUN` を返す。順序は host logs → dry run → pilot → device gate → prior attempt → committed source → runner identity → plan identity であり、以前は device gate で止まっていた。ただし v3 runner はさらに line 54 で `certify_v2 --entry formal-1` を hard requirement として呼ぶため、entry assessor の PASS だけでは formal-1 を起動できない。

plan record は private evidence であり、commit 対象の plan file と runner の hash を固定する。plan file の内容は「2 秒契約、Lead 100 ms、deadline 式、chunk 契約、host 検証、dry run、entry 条件、1 process 上限、retry 0」を列挙する。

## L. Device entry gate (initial repair-mission observation)

この節は修復 mission 開始時（device offline）の記録である。2026-10-23 の再入口は冒頭の「Re-entry observation and formal prerequisite gate」を参照。

| 項目 | 観測値 |
| --- | --- |
| result | `BLOCKED_DEVICE_OFFLINE` |
| devices listing entries | 0 |
| `get-state` | exit 1 |
| wireless endpoint TCP | unreachable |
| reconnect attempted | false |
| wake attempted | false |
| device mutations | 0 |
| read-only device calls | 2 |
| new formal processes | 0 |

endpoint は ADB の device table にも現れず、TCP も到達不能だった。reconnect、wake、unlock、reboot、force-stop、foreground takeover は行っていない。状態不明として fail closed した。

## M. Numerical pilot preservation

pilot の数値・resource evidence は不変である。

| 項目 | 値 |
| --- | --- |
| roles | 5（standalone_softmax / standalone_dvalue / native_softmax / native_dvalue / dual） |
| consumer pairs | 58,944 |
| executes | 24,560 |
| graphs | 20 |
| Backend create/free | 1 / 1 |
| Device create/free | 1 / 1 |
| Context create/free | 5 / 5 |
| inter-role Prepare reload | 0 |
| arithmetic word mismatch | 0 |
| observer word mismatch | 0 |
| legacy digests | 5 / 5 anchors 一致 |
| actual exit | 0 |
| teardown | PASS |
| duration | 141.5153525 s |

前回の readback 不完全は、この数値 PASS を失効させない。一方で、five-role formal candidate の protocol 認定にも進めない。

## N. Formal process accounting

| 項目 | 値 |
| --- | --- |
| formal completed | 0 / 3 |
| 今回起動した formal OS process | 0 |
| 今回許可した新規 formal process | 最大 1 |
| automatic retry | 0 |
| pilot の再利用 | 数値証拠のみ。formal へ昇格しない |

entry assessor が PASS しても、v3 runner の independent prerequisite（`certify_v2 --entry formal-1`）が保持済み `mission-stop-private.json` を理由に FAIL するため、formal-1 は起動しない。今回は entry assessor は PASS したが、この prerequisite で停止し、1 process の枠は未使用のまま保持している。

## O. Verification executed

| gate | 結果 |
| --- | --- |
| Fast | **PASS**、145.5 s、PASS=13 FAIL=0 SKIP=21 |
| Fast 内の新規 3 step | monitor host 9.9 s、transport host 3.9 s、dry-run self-test 23.8 s（全て PASS） |
| Host | 今回は再実行していない（候補の数値/checkpoint 変更なし） |
| Android | 今回は再実行していない（Kotlin/JNI/Gradle 変更なし） |
| Qnn | 再入口後に **PASS**、972.2 s、PASS=4 FAIL=0 SKIP=0（fresh log） |
| Device | read-only entry gate は PASS（online / same-device / Awake / no owned probe）。formal probe は未実行 |
| Formal | 未実行（v3 の independent prerequisite が `mission Device stop active` で FAIL） |

Fast は `scripts/verify_local.ps1` に monitor/transport host test と dry-run self-test を組み込んだ状態で実行した。したがって次回以降の通常開発でも同じ host 検証が回る。Qnn の freshness は formal 実行直前の前提条件であり、今回の host 修復によって省略可能にはならない。

## P. Public aggregate projection

`docs/results/g1-backward-shared-lifetime-formal-summary.json` は 55 key から **56 key** へ、`monitor_transport_recovery` を 1 つだけ追加した。

| 項目 | 値 |
| --- | --- |
| previous summary sha256 | `a18ef7d11ba2170f95f13938bccd9aca62c48820f7ca3fc264f043a9840d2713` |
| new summary sha256 | `1d7efeacb9034d9c9249ea01e30cf29a8644f1f41f48e87d2f29607d1900c741` |
| base 55 key の一致性 | 全件 byte-equivalent（独立比較） |
| status | `HOST_REPAIR_QUALIFIED_DEVICE_UNVERIFIED`（再入口前の export。現在の private assessor は gate PASS / `FORMAL_PROCESS_NOT_RUN`） |
| formal eligible | false / `FORMAL_PROCESS_NOT_RUN` |
| device cadence established | false（formal-1 未実行） |
| device transfer qualified | false（formal-1 未実行） |

exporter は closed allow-list で、`historical_cases=45` と `new_cases=40` の rejection を含む self-test が PASS。absolute path、device/app identifier、endpoint、APK hash、raw tensor bytes、ActivityManager dump、crash address を含まない。monitor/transport/numerical の verdict は別 field のまま維持した。

再入口後の private 評価は更新したが、公開集計は再発行していない。現行 exporter は既存の `monitor_transport_recovery` key を拒否するため、formal 後の公開更新には後段 exporter が必要である。公開 file は再入口前の export のままである。

## Q. What is and is not qualified

Qualified（host）:

- monitor の deadline 算術、command timeout、state 分類、evidence 出力。
- transport の chunk、resume、per-chunk hash、3 点 identity、binary roundtrip。
- v3 runner の formal-1 scope と self-test。
- 独立 assessor の rejection 34 cases。
- dry run の contract_met と fail-closed 11 cases。

Not qualified:

- device 上の 2 秒 cadence。endpoint がないため未検証。
- device 上の transfer。未検証。
- formal-1 / formal-2 / formal-3。0/3。
- performance、fresh device regression。
- Full Attention Backward、Full Attention、Full1-layer、Full19-layer、TRAINING_READY。

## R. Blocked status and exact re-entry condition

**BLOCKED_FORMAL_PREREQUISITE。** 数値 PASS と host 修復 PASS、device entry PASS は保持するが、formal は 0/3 のままである。修復 mission 開始時は `BLOCKED_DEVICE_OFFLINE` であったが、2026-10-23 に同じ物理 device の endpoint が復帰した。

再入口で完了したこと:

1. `scripts/run_backward_monitor_dryrun.ps1 -Force` を再実行し、`contract_met=true`（max 1916.0566 ms、counters 0、11/11 fail-closed cases）を確認した。
2. `python -B scripts/assess_monitor_transport_recovery.py --entry formal-1` を PASS（exit 0、`device_calls=0`）させた。
3. device entry gate を read-only で再観測し、`PASS` / `online_endpoint=true` / `same_physical_device_verified=true` / `new_formal_processes=0` を記録した。
4. fresh Qnn gate log を取得した（`total 972.2s PASS=4 FAIL=0 SKIP=0`）。
5. committed tree を確認した。

残る blocker は、v3 runner が line 54 で要求する `certify_v2 --entry formal-1` が `mission Device stop active` で FAIL することである。`mission-stop-private.json` は pilot の historical evidence として保持が要求されており、削除・rename しない。さらに `certify_v2` は v2 runner に bind しているため、v3 evidence を正式認定できない。v3 prerequisite / certifier の実装と独立検証が必要である。これは device 安全条件、2 秒契約、transport 契約を緩和する変更ではない。

wake、unlock、reboot、force-stop、reconnect の自動実行は行っていない。formal-1 は起動しておらず、automatic retry も 0 である。

## S. Next frontier / training

直近の engineering 作業は「v3 prerequisite / certifier の実装・独立検証 → formal-1 1 回」である。formal-1 が独立 assessor で PASS して初めて formal-2、formal-3 へ進む。performance と fresh regression はその後。

数学的 frontier は従来どおり、実 dP/dContext（PV/Gate/Output 由来）、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregation。Full Attention Backward / Full Attention / Full1-layer / Full19-layer / TRAINING_READY は未認定のまま。

40k→60k / 200k training は **NO-GO 維持**。今回 training、checkpoint decode/write、final_test、dataset/fixture 生成、optimizer/LR 変更、production backend 切替、push/PR/merge は行っていない。

---

## Final questions — all 50

| # | Question | Answer |
| --- | --- | --- |
| 1 | 正しい clone / base から開始したか | Yes。research clone、`investigation/g1-backward-shared-lifetime-formal`、HEAD `900e949`、開始時 clean。 |
| 2 | Original DSP/Host/probe 不変か | Yes。`658bd900…` / `0aea5fe5…` / `6ffeec21…` を再照合。再 build なし。 |
| 3 | New five-role ARM SHA256 | `6ffeec218c2532e6b54c433dbe963f9210ed1beb6c3da4234001a67b7b03126f`（不変）。 |
| 4 | Two-role からの diff | candidate の diff は前回のまま。今回の差分は host orchestration のみで、role/arithmetic/lifetime に触れていない。 |
| 5 | Math / graph / fixture / observer 不変か | Yes。DSP/Host bytes、graph builder、fixture、oracle、word gates、observer 契約は不変。 |
| 6 | Original formal plan 保存と別改訂 | Yes。`d42e0ecc…` を byte-identical で保持し、monitor/transport 用の別 plan `e3a86756…` を追加。 |
| 7 | Five-role Host lifetime tests | 前回 PASS を保持。今回再実行はしていない（lifetime 変更なし）。 |
| 8 | Backend create/free | pilot 実測 1 / 1。今回 device 実行なし。 |
| 9 | Device create/free | pilot 実測 1 / 1。今回 device 実行なし。 |
| 10 | Context create/free | pilot 実測 5 / 5。今回 device 実行なし。 |
| 11 | Role 間 Prepare reload | 0。総 load/unload 各 1。今回変更なし。 |
| 12 | All 20 graphs 構築・解放 | 20 create/finalize return 0。Context free で関連 graph/tensor を解放。今回変更なし。 |
| 13 | Full five-role pilot 完走 | 数値・resource は完走、actual exit 0。protocol は historical に BLOCKED。host 修復は完了したが device 未検証。 |
| 14 | Standalone Softmax maxAbs/digest | maxAbs 6.3353929016685129e-05、digest は legacy anchor 一致。 |
| 15 | Standalone dValue maxAbs/digest | maxAbs 1.6803001869902801e-05、digest は legacy anchor 一致。 |
| 16 | Native Softmax total maxAbs | 0.00010078856647055545。 |
| 17 | Native dValue total maxAbs | 1.3850631340517339e-05。 |
| 18 | Dual 両 total maxAbs | Softmax 0.00010078856647055545、dValue 1.3850631340517339e-05。 |
| 19 | 全 4 observer accuracy | PASS。Core/Holdout を独立再計算。 |
| 20 | Arithmetic word mismatch | 0。 |
| 21 | Observer final word mismatch | 0。 |
| 22 | Same-P NATIVE dual ownership | pilot 内 PASS。formal では未認定。今回の修復対象外。 |
| 23 | P lifetime / immutability | pilot 内 PASS。Context 間の tensor 持越しなし。今回の修復対象外。 |
| 24 | NoReceipt / NoTap 自身 | 両 consumer の実出力で accuracy/word PASS。不可視 P の直接読出しとは主張しない。 |
| 25 | Pilot actual exit / teardown | actual exit 0、Context→Device→Backend teardown PASS（既知 Logger leak 除外）。readback は不完全だった。 |
| 26 | Formal 3 独立 process | 未実行、**0 / 3**。今回の新規 formal process は 0。 |
| 27 | 各 formal role coverage | 未実行。pilot は全 5-role 完全 coverage。 |
| 28 | Legacy digest agreement | pilot 5 roles は anchor 全一致。formal 3-process agreement は未実行。 |
| 29 | Supplemental digest agreement | pilot 5 role-contract digests は算出済み。formal 比較は未実行。 |
| 30 | Exit139 再発 | この pilot では未観測（exit 0）。今回 device 実行なし。 |
| 31 | API 6031/6033 | 実 API error は未観測。api_us に 6031.* 4 行、6033.* 3 行。計時と error を分離したまま。 |
| 32 | Awake / top safety | 全観測 Awake/known、takeover 0。ただし historical runtime interval は 33/33 が 2 秒超過（median 4.029617 s）。修復版は host で契約内。 |
| 33 | NPU 競合 | 37 snapshots で認識可能な別 workload は未観測。完全排除とはしない。 |
| 34 | Backward 追加 cost | 未測定。performance 未開始。 |
| 35 | Forward/single/dual median/p10/p90 | すべて null / NOT_MEASURED。pilot の診断 api_us を正式性能値に昇格しない。 |
| 36 | Setup/teardown 分離 | process/Context 別 timer は実装・診断記録済み。正式比較測定は未実行。 |
| 37 | Fresh HVX FFN2/QKV | NOT_EXECUTED。 |
| 38 | Fresh Device smoke | NOT_EXECUTED（formal-1 は prerequisite で停止。device entry は online / same-device / gate PASS）。 |
| 39 | Fresh W8 114/114 | NOT_EXECUTED。historical PASS を流用しない。 |
| 40 | Fresh Forward P joint | pilot 内 native P の joint gate は PASS。別の required fresh device regression は NOT_EXECUTED。 |
| 41 | Fast/Host/Qnn | 今回 **Fast PASS 145.5 s（PASS=13 FAIL=0 SKIP=21）**。再入口後に **fresh Qnn PASS 972.2 s（PASS=4 FAIL=0 SKIP=0）** を取得。Host/Android/Device/Formal は再実行していない。前回 candidate の Host 232.4 s は保持するが、今回の fresh とは表現しない。 |
| 42 | Old negative controls | 全保全。expected FAIL を PASS に書き換えていない。 |
| 43 | 35/40 Tap / 6031 の扱い | 不変。Tap は REPRODUCED / CAUSE UNRESOLVED、旧 6031/DMA6033 も原因未解決。monitor の計時誤検出とは別問題。 |
| 44 | Protected checkpoint/refs/evidence | hash 読取のみ。旧 ledger と private files を個別照合し、加算しない。remote raw / private evidence の上書き・削除なし。 |
| 45 | Shared lifetime 正式 candidate 認定 | 不可。数値/host 修復は PASS、device entry も PASS だが、formal prerequisite（`mission Device stop active`）で formal-1 未起動、formal 0/3。 |
| 46 | 根本原因証明か支持仮説か | monitor は実測 phase の整合による **quantified host-side root cause**（`ADDITIVE_WAIT_PLUS_SERIAL_QUERY_BLOCK`）。device A/B による証明ではない。transport は wrapper の 180 秒 bound による `ADB_WRAPPER_TIMEOUT` と直接分類。 |
| 47 | Saved-P Backward 正式 closure | 不可。`BLOCKED_FORMAL_PREREQUISITE`、formal 0/3。performance / fresh regression も未完了。 |
| 48 | 残る frontier | endpoint 復旧・dry run・entry assessor は完了。次は v3 prerequisite / certifier の実装・独立検証 → formal-1 1 回 → formal-2/3。その後 performance/fresh regression。数学は実 dP/dContext、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregation。 |
| 49 | Full1-layer / Full19-layer | ともに NOT_CERTIFIED。Full Attention / Full Attention Backward / TRAINING_READY も未認定。 |
| 50 | 40k→60k / 200k training | NO-GO 維持。今回 training は実行していない。 |

---

## Final validation / closeout

- 前回 commit までの変更は host orchestration、host 検証、assessor/exporter、Fast 配線、文書、公開集計の 1 key に限定した。今回の再入口は read-only 観測と private evidence の追加のみで、code / candidate / 公開集計を変更していない。
- candidate / 算術 / graph / fixture / observer / Backend-Device lifetime の hash は不変。
- v1 runner は byte-identical で保存し、v3 は formal-1 専用として分離した。
- 2 秒契約は緩和していない。4 秒観測は historical FAIL のまま。dry run は emulation と明記した。
- transport 修復は in-run monitor を止めない。remote raw と既存 private evidence は append-only。
- Fast は PASS。再入口後の fresh Qnn も PASS。Android / Device / Formal は今回実行していない。
- 公開集計は 56 key、追加は `monitor_transport_recovery` 1 つだけ。base 55 key は独立比較で全一致。private 情報を含まない。
- formal は **0/3**、新規 formal process 0、automatic retry 0。
- 最終状態は **BLOCKED_FORMAL_PREREQUISITE**。device は online / same-device で entry gate は PASS だが、保持済み mission-stop を扱う v3 prerequisite / certifier が未実装である。この file は削除・rename していない。
