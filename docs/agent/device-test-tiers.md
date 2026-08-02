# 実機試験Tier

実機試験、ADB操作、device runnerの実行前に本書を読む。端末を変更する操作は、該当Tierの許可範囲を超えてはならない。

## 物理端末とADB endpoint

`adb devices` の1行を物理端末1台と数えない。同一端末がUSB serialとTCP `host:port`の複数endpointを持つ場合がある。

1. online endpointごとに `ro.serialno`、`ro.boot.serialno` など端末側の安定識別子を読み、emulatorでないことも確認する。
2. 同じ安定識別子のUSB/TCP endpointを1台へgroup化する。空値、衝突、取得失敗などで同一性を証明できなければ停止する。
3. 対象の物理端末が1台だけであることを確認し、runnerが使う正式endpointを1つ明示的かつ決定的に選ぶ。
4. reconnect時も元の安定識別子と一致するendpointだけへreattachする。単にonline endpointが1つという理由で別端末へ接続しない。

endpointと安定識別子はprivate reportだけに置き、公開bundleやcommitへ含めない。ADB transport切断、endpoint rotation、alias変更はtransport failureであり、数値失敗へ分類しない。

## Tier 1: 実機不要・自動実行可

- JVM unit tests、C++ host tests
- QNN無効 `assembleDebug`、`assembleDebugAndroidTest`
- `verify_local.ps1`、`check_qairt.ps1` の読み取り専用inventory/self-test
- graph-mapなどのstatic exporter

## Tier 2: 条件付き自動実行可

既存headless correctness suite、README記載の既存回帰runner、fixed-state試験、device probe、Skel reuse/recoveryが対象で、`BACKGROUND_CORRECTNESS`を使う。次をすべて満たす。

- 上記手順でonline物理端末を1台に解決し、正式endpointを1つ選択済み
- 固定QAIRT root/Build IDを明示し、coreとBuild IDを確認済み
- QNN有効build後のAPK auditでABI/hash/path/2.47混入検査がPASS
- 2秒ごとのtop-resumed監視で `focus_takeover_count=0`
- root化、SELinux変更、system領域変更を伴わない

## Tier 3: 明示指示が必要

- UIを前面化する試験、`EXCLUSIVE_BENCHMARK`
- 既定がUI_VALIDATIONでActivityを前面化する `run_qnn_resumable_formal.ps1`
- 通知、permission、app data削除
- firmwareまたはQAIRT SDKのversion/配置変更
- 長時間training
- `docs/results`へのexport、外部共有、commit済み結果の公開、push

Tier 3はユーザーの明示指示なしに実行しない。formal runnerはseed単位でatomicに保存・再開し、ADB transport中断を数値失敗に分類しない。
