誤り、重複、陳腐化を発見した場合は更新してよい。
ただし、ユーザーの明示指示なしに、安全条件、検証条件、実機Tier、QAIRT固定条件を弱めてはならない。
変更理由を完了報告に記載する。

## 絶対規則

- Tier 3 はユーザーの明示指示なしに実行しない。
- QAIRT の自動 fallback、2.47との混在、固定値と異なるSDKの使用を禁止する。
- QNN return code の成功と tensor の有限性を別々に確認する。
- raw checkpoint、logcat、ADB endpoint、ローカル絶対pathをcommitしない。
- reset、rebase、amendは禁止する。ユーザーの既存変更を破棄・stashしない。
- 「NPUだけで学習した」「CPUを完全に使用していない」「QNNが自動微分した」と誇張しない。

## 完了前の検証

変更や検証報告の前に [verification.md](docs/agent/verification.md) を読む。
`verify_local.ps1` は全変更に共通するローカル基礎ゲートであり、
対象に応じてQNN build、実機試験、公開bundleの追加gateが必要である。

```powershell
.\scripts\verify_local.ps1
```

全必須工程がPASSするまで完了と報告しない。
検証不能なら未実行工程、原因、再現コマンドを示して `BLOCKED` とし、
環境を勝手にインストール・変更しない。
`-SkipAndroidBuild` は docs/scripts-only 変更の最終確認に使用できるが、
Android/JNI/Gradle/CMake/APK packaging変更では途中確認専用である。
incremental buildを既定とし、必要な場合だけ `-Clean` を使う。

QNN node/tensor変更では `run_host_tests.ps1` のshape validatorを必須とし、
shape変更時はgraph-map exporterとnegative testも更新する。
GitHub Actionsも `verify_local.ps1` を使い、CI専用の検証列を追加しない。
pinned MNN sourceはignoredな`third_party/MNN/`へ取得し、QAIRT SDK、
ADB端末、secrets、APK artifactをCIで使わない。

## QAIRT固定

QNN/QAIRT build、APK監査、実機操作の前に [qairt-policy.md](docs/agent/qairt-policy.md) を読む。
設定正本は `scripts/qairt_version.ps1` である。

- SDK root: `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`
- Build ID: `2.48.40.260702151143`

QNN有効操作ではrootとBuild IDを明示し、正本との一致をfail closedで確認する。
自動探索の結果、別version、既存APK、build cacheへfallbackしてはならない。
固定rootの不在、Build ID不一致、core required item不足では、
SDKをインストール・移動・変更せず停止する。
`QAIRT_CORE_INCOMPLETE` はfatal、optional tools/samplesだけの
`QAIRT_INVENTORY_INCOMPLETE` はadvisoryである。
正式gateは固定引数付き `verify_local.ps1 -WithQairt` と
`audit_qnn_apk.ps1` のABI/hash/path/2.47混入監査であり、advisoryだけでは停止しない。
引数なし自動探索は読み取り専用inventory調査に限る。self-testのtemp内偽SDKは例外とする。

## 実機Tier

実機またはdevice runnerを扱う前に
[device-test-tiers.md](docs/agent/device-test-tiers.md) を読む。

- Tier 1は実機不要のunit/host/build/static検証だけで、自動実行できる。
- Tier 2は既存headless correctness runnerに限り、固定QAIRT確認、QNN build、
  APK audit、物理端末同一性、focus takeover 0、非破壊条件を
  すべて満たす場合だけ自動実行できる。
- Tier 3（UI前面化、`EXCLUSIVE_BENCHMARK`、UI_VALIDATIONのformal runner、
  通知/permission、app data削除、firmware/SDK変更、長時間training、
  公開、push）は明示指示なしに実行しない。

`adb devices` のendpoint数を物理端末数として扱わない。
安定識別子でUSB/TCP aliasを重複排除し、正式endpointを1つだけ選ぶ。
同一性を確認できなければ停止する。
root化、SELinux変更、system領域変更を禁止する。ADB transport中断を数値失敗に分類しない。

## QNN graph

- producerの推論shapeと宣言output shapeを一致させる。
- Reduceのaxis、keep_dims、output shapeをvalidatorで確認し、broadcastを暗黙に仮定しない。
- APP_READ/APP_WRITEの方向と全面書き込みを維持する。
- QNN node/tensor変更後は `run_host_tests.ps1` で
  `qnn_graph_shape_validator` testsを実行する。

## 数値・Evidence

数値変更、回帰判定、結果報告の前に [numerical-evidence.md](docs/agent/numerical-evidence.md) を読む。

- HTP内部精度をFP32と断定しない。平方、exp、divide前のdynamic rangeを監査する。
- clamp、epsilon増加、learning rate低下だけで原因修正扱いにしない。
- CPUとの差と、同一HTP実行の非決定性を区別する。
- 数値主張にはseed数、step数、発生率を記載する。
- regressionは承認済みcanonical anchorとの一致で判定し、
  既知の低品質はquality shortfallとして分離する。
  runnerのterminal `FAILED`だけでコード回帰FAILにしない。
- 公開結果はallow-list exporterを使い、private evidenceや端末識別情報を含めない。

## Git

開始時に現在branch、HEAD、upstreamとの差分を確認する。
mainで最新origin/mainを取得できる場合だけHEAD一致を確認し、
不一致時はreset/rebase/mergeせず報告する。
現在branchを勝手に切り替えず、ユーザーの未commit変更を保持する。
commit権限があるタスクでのみcommitし、自分が今回変更したpathだけをstageする。
開始時から存在したユーザー変更を含めない。
pushは明示指示時だけfast-forwardで行う。reset、rebase、amendは禁止する。
終了時に `git status` を確認し、検証生成物が `build/` 以下だけであることを確かめる。
commit messageは `type(scope): 概要` とし、typeは `feat` `fix` `test`
`build` `docs`、scopeは `qnn` `android` `lm` などを使う。

## 表現・成果物

「学習stepの数値演算をHTPで実行した」と表現する。
NPU-only、CPU未使用、QNN自動微分を主張しない。
QAIRTライブラリ、Stub、Skel、MNN source treeをGitへ追加しない。
数値結果や実験手順を変更したら対応する `docs/` も更新する。

## Windows編集

Windowsで編集する前、または組み込み `apply_patch` が失敗した場合は
[windows-editing.md](docs/agent/windows-editing.md) を読む。
restricted-token sandboxエラー時に同じ `apply_patch` や `apply_patch.bat` を反復しない。
代替は `git apply --check -` で検証後に同じdiffを適用し、
曖昧な置換や `Set-Content` によるファイル全体の無条件上書きを行わない。
編集後は `git diff --check` と対象差分を確認し、workspace外や既存ユーザー変更に触れない。
