# AGENTS.md

このリポジトリでは、実機上のQualcomm QNN / HTPを用いた言語モデル学習を扱う。
安全性、再現性、証拠の保存を優先しつつ、開発中の検証コストが過剰にならないよう
変更範囲に応じた段階的な検証を行う。

関連する詳細規則は `docs/agent/` 以下を正本とする。

誤り、重複、陳腐化を発見した場合は更新してよい。
ただし、ユーザーの明示指示なしに、安全条件、検証条件、実機Tier、
QAIRT固定条件を弱めてはならない。
変更理由を完了報告に記載する。


## 絶対規則

- Tier 3 はユーザーの明示指示なしに実行しない。
- QAIRT の自動 fallback、2.47との混在、固定値と異なるSDKの使用を禁止する。
- QNN return code の成功と tensor の有限性を別々に確認する。
- raw checkpoint、logcat、ADB endpoint、ローカル絶対pathをcommitしない。
- reset、rebase、amendは禁止する。
- ユーザーの既存変更を破棄・stashしない。
- force push、force-with-lease、履歴書き換えを禁止する。
- 「NPUだけで学習した」「CPUを完全に使用していない」
  「QNNが自動微分した」と誇張しない。


## 完了前の検証

変更や検証報告の前に
[verification.md](docs/agent/verification.md) を読む。

検証は変更範囲と作業段階に応じて
**Fast / Targeted / Full**
の3段階で行う。

重い全体gateを局所修正や途中commitごとに反復してはならない。


### Fast verification

開発・反復中は原則としてFast verificationを使う。

最低限、変更範囲に応じて以下を行う。

- `git diff --check`
- PowerShell parser / preflight
- 変更箇所に直接対応するunit / host / self-test
- 必要な最小build / validator

`scripts/verify_local.ps1 -Fast` が利用可能な場合はこれを優先する。

Fast verificationは開発中の素早いフィードバックを目的とし、
変更していない領域のformal training batteryや
重いAndroid/QNN検証を無条件に含めない。

共通前提ツールや環境の欠落によって、
複数の後続testが同じ理由で失敗すると事前に判断できる場合は
fail fastとする。

例えば、

- `pwsh`
- Java / JDK
- Python
- C++ compiler
- Gradle実行環境

などの必須前提が欠落している場合、
同一原因による大量のFAILを生成するためだけに後続testを実行しない。

ただし、その前提に依存しない独立testまで不必要に停止させない。


### Targeted verification

変更範囲に応じて必要な検証だけを追加する。

例:

#### Android / Compose / UI変更

- 関連unit test
- 関連Compose test
- 必要に応じて `lintDebug`
- `assembleDebug`

#### C++ / QNN変更

- 関連host test
- shape validator
- QNN graph関連self-test
- 必要なQNN build

#### PowerShell runner変更

- parser check
- 対象runnerのself-test
- 直接関連するhost test

#### model config / preset変更

- config test
- model identity test
- checkpoint compatibility test
- resume mismatch negative test

変更していない領域のformal training batteryを毎回再実行しない。


### Full verification

`verify_local.ps1` のfull gateは原則として以下の場合に実行する。

- マイルストーン完了時
- 統合作業の完了時
- mainへの統合前
- release / formal result確定前
- ユーザーがformal verificationを明示的に要求した場合

通常の途中commitや、
専用 `codex/*` 作業branchへの保存pushでは、
変更範囲に十分なFast / Targeted verificationがPASSしていれば
Full gateを必須としない。

途中commitや局所修正ではFull gate未実施でも完了可能だが、

- Full gate未実施であること
- 実行済みのFast / Targeted test
- 残っているformal gate

を完了報告に記載する。

Full gateでFAILした場合は、
原因修正のたびにFull suite全体を再実行しない。

まずFAILした対象を個別に再現・修正し、
そのtargeted testがPASSした後、
必要な最終Full gateを1回実行する。

例:

- `assembleDebug` FAIL → `assembleDebug` を個別に修正・再検証
- host test FAIL → 関連host testのみ再検証
- runner self-test FAIL → 当該runnerのみ再検証

最終formal verificationでは対象に応じて、

- QNN build
- 実機試験
- APK audit
- 公開bundle

等の追加gateも実行し、
必須工程がPASSするまで
formal verification完了とは報告しない。

検証不能なら、

- 未実行工程
- 原因
- 再現コマンド

を示して `BLOCKED` とし、
環境を勝手にインストール・変更しない。

`-SkipAndroidBuild` は docs / scripts-only変更の最終確認に使用できるが、
Android / JNI / Gradle / CMake / APK packaging変更では
targetedなbuild / testと併用する。

incremental buildを既定とし、
必要な場合だけ `-Clean` を使う。

QNN node / tensor変更では
`run_host_tests.ps1` のshape validatorを必須とし、
shape変更時はgraph-map exporterとnegative testも更新する。

GitHub Actionsも `verify_local.ps1` を使い、
CI専用の別検証列を追加しない。

pinned MNN sourceはignoredな `third_party/MNN/` へ取得し、
QAIRT SDK、ADB端末、secrets、APK artifactをCIで使わない。


## QAIRT固定

QNN / QAIRT build、APK監査、実機操作の前に
[qairt-policy.md](docs/agent/qairt-policy.md) を読む。

設定正本は `scripts/qairt_version.ps1` である。

- SDK root: `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`
- Build ID: `2.48.40.260702151143`

QNN有効操作ではrootとBuild IDを明示し、
正本との一致をfail closedで確認する。

自動探索の結果、

- 別version
- 既存APK
- build cache

へfallbackしてはならない。

固定rootの不在、Build ID不一致、core required item不足では、
SDKをインストール・移動・変更せず停止する。

`QAIRT_CORE_INCOMPLETE` はfatal。

optional tools / samplesだけの
`QAIRT_INVENTORY_INCOMPLETE` はadvisoryである。

QNN / QAIRT変更の最終formal gateは、
固定引数付き

`verify_local.ps1 -WithQairt`

と

`audit_qnn_apk.ps1`

の

- ABI
- hash
- path
- 2.47混入

監査である。

advisoryだけでは停止しない。

開発中の局所変更では、
影響範囲のtargeted QNN build / testを先に用い、
このformal gateを変更ごとに反復する必要はない。

引数なし自動探索は読み取り専用inventory調査に限る。
self-testのtemp内偽SDKは例外とする。


## 実機Tier

実機またはdevice runnerを扱う前に
[device-test-tiers.md](docs/agent/device-test-tiers.md) を読む。


### Tier 1

実機不要の

- unit test
- host test
- build
- static validation

のみ。

自動実行できる。


### Tier 2

既存headless correctness runnerに限る。

以下をすべて満たす場合だけ自動実行できる。

- 固定QAIRT確認
- QNN build
- APK audit
- 物理端末同一性確認
- focus takeover 0
- 非破壊条件

activeなユーザーrunを妨害してはならない。


### Tier 3

以下はユーザーの明示指示なしに実行しない。

- UI前面化
- `EXCLUSIVE_BENCHMARK`
- `UI_VALIDATION` のformal runner
- notification / permission変更
- app data削除
- firmware変更
- SDK変更
- 長時間training
- 外部公開
- mainへのmerge
- mainへのpush

検証済みcommitを専用 `codex/*` 作業branchへ
通常のfast-forward pushで保存する操作は
実機Tierには分類せず、Git節の規則に従う。

`adb devices` のendpoint数を物理端末数として扱わない。

安定識別子でUSB / TCP aliasを重複排除し、
正式endpointを1つだけ選ぶ。

同一性を確認できなければ停止する。

root化、SELinux変更、system領域変更を禁止する。

ADB transport中断を数値失敗に分類しない。


## Active run保護

実機操作前には、既存のactive trainingが存在しないことを確認する。

少なくとも必要に応じて、

- training heartbeat
- run lock
- foreground service
- resumed / focused Activity
- instrumentation process
- process state

を確認する。

単なるcached processやinactive taskだけを
active trainingと誤判定しない。

一方で状態が不明な場合はfail closedとする。

ユーザーのactive runを通すために、

- `am force-stop`
- `pm clear`
- process kill
- app data削除
- reboot

を勝手に実行しない。


## QNN graph

- producerの推論shapeと宣言output shapeを一致させる。
- Reduceのaxis、keep_dims、output shapeをvalidatorで確認し、
  broadcastを暗黙に仮定しない。
- APP_READ / APP_WRITEの方向と全面書き込みを維持する。
- QNN node / tensor変更後は `run_host_tests.ps1` で
  `qnn_graph_shape_validator` testsを実行する。


## 数値・Evidence

数値変更、回帰判定、結果報告の前に
[numerical-evidence.md](docs/agent/numerical-evidence.md) を読む。

- HTP内部精度をFP32と断定しない。
- 平方、exp、divide前のdynamic rangeを監査する。
- clamp、epsilon増加、learning rate低下だけで原因修正扱いにしない。
- CPUとの差と、同一HTP実行の非決定性を区別する。
- 数値主張にはseed数、step数、発生率を記載する。
- regressionは承認済みcanonical anchorとの一致で判定する。
- 既知の低品質はquality shortfallとして分離する。
- runnerのterminal `FAILED`だけでコード回帰FAILにしない。
- QNN return code successとfinite tensorを別々に確認する。
- fallbackの有無を独立して記録する。
- 異なるevaluation条件の数値を直接比較しない。
- diagnostic用の短いchunk評価をformal quality比較へ混ぜない。
- 公開結果はallow-list exporterを使い、
  private evidenceや端末識別情報を含めない。


## 実験結果の解釈

単一runの観測値を複数の独立した再現証拠として扱わない。

同じ環境条件や同じrunner経路を共有する複数の失敗は、
独立した証拠とは限らない。

以下を区別する。

- model / native executionの成功
- runner / instrumentationの成功
- Android lifecycleの成功
- environment / harnessの成功

runnerの `Process crashed` やtimeoutだけから、

- QNN failure
- HTP failure
- OOM
- LMK
- native crash

を断定しない。

必要に応じて、

- native report
- checkpoint
- heartbeat
- ApplicationExitInfo
- logcat
- process state
- RSS / PSS
- MemAvailable

等を回収してから原因を分類する。


## Git

開始時に、

- 現在branch
- HEAD
- upstream
- `git status`

を確認する。

mainで最新 `origin/main` を取得できる場合だけHEAD一致を確認し、
不一致時はreset / rebase / mergeせず報告する。

現在branchを勝手に切り替えない。

ユーザーの未commit変更を保持する。

開始時から存在したユーザー変更を、

- discard
- reset
- stash
- commit

しない。

commit権限があるタスクでのみcommitし、
自分が今回変更したpathだけをstageする。

専用の `codex/*` 作業branchでは、
自分が作成し必要なFast / Targeted verificationを通したcommitを、
origin上の同名branchへ通常のfast-forward pushで保存してよい。

この保存pushのためだけにFull verificationを必須としない。

ただし、以下はユーザーの明示指示なしに行わない。

- mainへのpush
- mainへのmerge
- force push
- force-with-lease
- rebase
- reset
- amend
- history rewrite

既存remote branchがfast-forwardできない場合は、
rebase / reset / force pushせず停止して報告する。

終了時に `git status` を確認し、
検証生成物が原則として `build/` 以下だけであることを確かめる。

commit messageは

`type(scope): 概要`

とする。

type:

- `feat`
- `fix`
- `test`
- `build`
- `docs`

scope例:

- `qnn`
- `android`
- `lm`
- `scripts`


## 表現・成果物

「学習stepの数値演算をHTPで実行した」と表現する。

以下を主張しない。

- NPU-only
- CPU未使用
- QNN自動微分

QAIRTライブラリ、Stub、Skel、MNN source treeをGitへ追加しない。

以下をcommitしない。

- raw checkpoint
- private corpus
- private cache
- raw logcat
- APK
- QAIRT runtime
- device-specific private evidence

数値結果や実験手順を変更したら対応する `docs/` も更新する。


## Private training data

学習用private corpusと派生物は、
原則としてignoredな `build/private-data/` 以下に置く。

元dataset、clean text、token cache、human-readable text dump等を
Gitへcommitしない。

学習cacheから人間可読のtext dumpを生成する場合も、
実際の選択済みtraining subsetと対応することを明示し、
元dataset全体と混同しない。

validation / development / final_testを
training dataとして混入させない。

final_testはformal evaluation時まで開かない方針がある場合、
そのpolicyを維持する。


## Windows編集

Windowsで編集する前、
または組み込み `apply_patch` が失敗した場合は
[windows-editing.md](docs/agent/windows-editing.md) を読む。

restricted-token sandboxエラー時に、
同じ `apply_patch` や `apply_patch.bat` を反復しない。

代替は

`git apply --check -`

で検証後に同じdiffを適用する。

曖昧な置換や、
`Set-Content` によるファイル全体の無条件上書きを行わない。

編集後は、

- `git diff --check`
- 対象差分
- `git status`

を確認する。

workspace外や既存ユーザー変更に触れない。


## 作業の基本フロー

通常の開発は原則として以下の順で行う。

1. branch / HEAD / status確認
2. 関連policy文書を読む
3. 変更
4. Fast verification
5. 必要なTargeted verification
6. 差分確認
7. commit
8. 必要なら `codex/*` branchへ通常push
9. 次の変更へ進む
10. マイルストーン完了時にFull verification
11. 最終報告

つまり、

`編集 → Fast / Targeted → 編集 → Fast / Targeted → 完成 → Full`

を基本とする。

小さな修正ごとにFull suiteを反復して
開発時間を浪費しない。

一方で、検証時間短縮を理由として
安全条件、QNN correctness、QAIRT固定、実機保護を省略してはならない。