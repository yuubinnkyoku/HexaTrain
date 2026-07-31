必要があれば、このAGENTS.mdを自由に編集して良い。

## 完了前の検証

コードを変更したら、完了を報告する前に必ず次を実行する。

```powershell
.\scripts\verify_local.ps1
```

既定の `verify_local.ps1` は実機不要・QAIRT不要・tracked ファイルを変更しない。
`git-diff-check`、`tracked-binary-audit`、`secret-path-audit`、
`qairt-selection-self-test`（temp内の5ケース）、公開結果exporter self-test、
`resumable-formal-runner-self-test`（temp内のsynthetic fixture、実機不要）、
JVM unit tests、C++ host tests、`assembleDebug`、`assembleDebugAndroidTest` の
10工程が唯一のローカル検証ゲートである。
全工程が PASS になるまで完了を報告しない。失敗を隠して完了しない。

```powershell
.\scripts\verify_local.ps1                               # 通常
.\scripts\verify_local.ps1 -SkipAndroidBuild             # Android build を省略（速い）
.\scripts\verify_local.ps1 -Clean                        # clean build（遅いので必要時のみ）
.\scripts\verify_local.ps1 `
  -WithQairt `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

incremental build が既定。clean が必要な場合だけ `-Clean` を使う。

GitHub Actions は `.github/workflows/verify.yml` から同じ
`.\scripts\verify_local.ps1` を実行する。CI専用のテスト列は追加せず、
Android build依存のpinned MNN sourceはignoredな `third_party/MNN/` へ取得する。
QAIRT SDK、ADB端末、repository secrets、APK artifactを使用しない。

## QAIRT SDK の固定

PhoneLM の QNN 有効 build、APK 監査、実機試験では、次の SDK を明示的に使用する。

- QAIRT SDK root: `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`
- Expected Build ID: `2.48.40.260702151143`

QNN 有効操作では、SDK root と Expected Build ID を省略した自動選択を使用しない。
複数の QAIRT version がインストールされていても、最初に見つかった SDK を暗黙に採用しない。

既存の QNN build、APK 監査、実機 runner にも、対応する SDK root と
Expected Build ID を必ず明示して渡す。

禁止事項:

- QAIRT 2.47 を QNN build、APK 監査、実機試験に使用しない
- 2.48 SDK が存在しない、不完全、または Build ID 不一致の場合に別 version へ fallback しない
- 自動探索で見つかった SDK を、そのまま PhoneLM の build 対象として採用しない
- SDK root と Build ID を確認せず、既存 APK や build cache だけを根拠に実機試験を開始しない

明示した SDK を利用できない場合は、SDK のインストール、移動、version 変更を行わず、
次のいずれかを報告して停止する。

- `QAIRT_SDK_ROOT_UNAVAILABLE`
- `QAIRT_BUILD_ID_MISMATCH`
- `QAIRT_SDK_INCOMPLETE`

引数なしの自動探索は、インストール済み SDK の読み取り専用 inventory 調査に限って使用できる。
`check_qairt.ps1 -SelfTest` は偽 SDK を使う選択ルールの回帰試験であり、この制限の対象外とする。

## 実行Tier

### Tier 1: 自動実行可（実機不要）

- `.\gradlew.bat :app:testDebugUnitTest --no-daemon` — JVM unit tests
- `.\scripts\run_host_tests.ps1` — C++ host tests。`qnn_graph_shape_validator` の検証を含む。g++ が必要
- `.\gradlew.bat :app:assembleDebug --no-daemon` — QNN無効 build
- `.\gradlew.bat :app:assembleDebugAndroidTest --no-daemon`
- `.\scripts\verify_local.ps1` — 上記一括 + git/binary/secret 監査
- `.\scripts\check_qairt.ps1` — QAIRT SDK の読み取り専用検査。明示 `-SdkRoot` は排他的に優先され、存在しない/不完全な場合は別 version へ fallback せず FAIL。選択ルールの回帰は `-SelfTest`
- `.\scripts\export_qnn_tiny_lm_graph_map.ps1` — graph map の静的 export

### Tier 2: 条件を満たせば自動実行可

対象: 既存の headless QNN suite（`.\scripts\run_qnn_headless_tests.ps1`）、README に記載の既存回帰 runner（`run_qnn_device_tests.ps1`、`run_qnn_training_tests.ps1`、`run_qnn_htp_*_tests.ps1`）、fixed-state 試験、device probe、Skel reuse/recovery。TestMode は `BACKGROUND_CORRECTNESS` を使う。

条件（全て満たすこと）:

- オンライン物理端末を明示的に選択し、`adb devices` で online が 1 台のみであることを確認する
- QAIRT SDK root を `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`、Expected Build ID を `2.48.40.260702151143` として明示している
- `check_qairt.ps1` で明示 root と Build ID の一致を確認している
- QNN 有効 build 後に `.\scripts\audit_qnn_apk.ps1` を実行し、runtime、V81 Stub、V81 Skel の hash が同じ 2.48 SDK と一致し、`forbidden_2_47_strings=false` である
- Activity/focus takeover が 0（headless runner が 2 秒ごとの top-resumed 監視と `focus_takeover_count=0` 確認で自動検出する）
- root 化、SELinux 変更、システム領域変更を伴わない

### Tier 3: 明示指示が必要

- UI を前面化する試験。`EXCLUSIVE_BENCHMARK` を含む。
  `.\scripts\run_qnn_resumable_formal.ps1` も既定 TestMode=UI_VALIDATION で
  Activity を意図的に前面化するためこの Tier。seed 単位で atomic に保存・再開し、
  ADB transport 中断は数値失敗に分類しない。
- 通知・permission の操作
- app data の削除
- firmware や QAIRT SDK のバージョン変更
- 長時間 training
- 公開（`docs/results` への export や外部共有）や push

## QNN graph rules

- producer の推論 shape と宣言 output shape を一致させる
- Reduce の axis、keep_dims、output shape を validator で確認する
- broadcast を暗黙に仮定しない
- tensor shape 変更時は graph-map exporter（`.\scripts\export_qnn_tiny_lm_graph_map.ps1`）と negative test を更新する
- APP_READ/APP_WRITE の方向と全面書き込みを維持する
- QNN node/tensor を変更したら shape validator tests を必ず実行する。
  `.\scripts\run_host_tests.ps1`（`host_tests/qnn_sdk_independent_test.cpp` が `qnn_graph_shape_validator.cpp` を直接検証する）

## Numerical rules

- HTP 内部精度を FP32 と断定しない
- 平方、exp、divide 前の dynamic range を監査する
- clamp、epsilon 増加、learning rate 低下だけで原因修正扱いにしない
- CPU との差と、同一 HTP 実行の非決定性を区別する

## Evidence rules

- QNN return code 成功と tensor 有限性を別々に確認する
- 数値結果を主張する場合は seed 数、step 数、発生率を記載する
- private report から公開結果を作るときは allow-list exporter（`scripts\export_public_*.ps1`）を使う
- raw checkpoint、logcat、ADB endpoint、絶対 path を commit しない

## Git 運用

- 作業開始時に `HEAD == origin/main` であることを確認する
- milestone ごとに commit する
- 現在の branch を勝手に切り替えない。ユーザーの未 commit 変更を stash しない
- 指示されていないタスクでは push しない。push する場合は fast-forward のみ
- reset、rebase、amend は禁止
- 作業終了時に `git status` を確認し、検証生成物が `build/` 以下だけであることを確かめる
- commit message は `type(scope): 概要` とする。
  type は `feat` `fix` `test` `build` `docs`、scope は `qnn` `android` `lm` などを使う

## 表現とドキュメント

- 「学習 step の数値演算を HTP で実行した」と表現する。「NPU だけで学習した」「CPU を完全に使用していない」「QNN が自動微分した」とは表現しない
- QAIRT 2.47 との混在は禁止。QNN有効操作では固定した2.48 rootとBuild IDを必ず明示する
- QAIRT ライブラリ、Stub、Skel、MNN source tree を Git へ追加しない
- 数値結果や実験手順を変更したら、対応する `docs/` も同じタスクで更新する

## Windows でのファイル編集

Codex Desktop の Windows 環境では、組み込みの `apply_patch` が次の sandbox
初期化エラーで失敗することがある。

```text
windows unelevated restricted-token sandbox cannot enforce split writable root sets directly
```

このエラーはリポジトリや patch 内容の問題ではない。同じ呼び出しを何度も再試行しない。
また、`apply_patch.bat` を shell から実行すると WindowsApps 配下の実体が
`Access is denied` になることがあるため、この経路も繰り返さない。

組み込み `apply_patch` が上記理由で利用できない場合に限り、次の順で安全な代替を使う。

1. unified diff を `git apply --check -` で検証する。
2. 同じ diff を `git apply --whitespace=error -` で適用する。
3. diff 適用が構文上困難な場合だけ、PowerShell で一意な完全一致アンカーを検証して置換する。
4. 編集後に必ず `git diff --check` と対象差分を確認する。

代替編集では、対象が workspace 内にあることを確認し、既存のユーザー変更を保持する。
`Set-Content` によるファイル全体の無条件上書きや、曖昧な正規表現置換は行わない。
