# PhoneLM QNN HTP — QAIRT 2.48正式構成

## 対象と固定版

- nubia NX741J / SM8850 / HTP V81 / Android 16
- QAIRT `2.48.40.260702151143`
- QNN Core API `2.37.0`
- HTP Backend API `5.48.0`
- Android NDK `26.2.11394342`（r26c）
- CMake `3.22.1`、ABI `arm64-v8a`

QAIRT `sdk.yaml`自身がAndroid NDK r26cを要求するため、PhoneLMもr26cを使用する。

## 初期化契約

QAIRT 2.47と2.48の双方で、host runtime、V81 Stub、unsigned V81 Skelを同一配布物から揃えると
HTP初期化とMatMul実行に成功した。2.48固有の回帰ではなかった。端末既定のDSP探索だけでは
`QnnDevice_create=14001`が再現する。

PhoneLMはローカルQAIRT SDKからビルド時に必要物を取り込み、Skelを
`files/qnn-dsp/2.48.40.260702151143/libQnnHtpV81Skel.so`へ展開する。assetと展開済みコピーの
SHA-256を比較し、不一致時は同じ版ディレクトリの一時ファイルへ書き、fsync後にatomic renameする。
再検証にも失敗した場合は、QNNライブラリを初回ロードする前に起動を停止する。

`ADSP_LIBRARY_PATH`は次の順序で、セミコロン区切り・重複除去によりプロセス内だけで設定する。

1. PhoneLMの完全Build ID別Skelディレクトリ
2. プロセスに既に設定されていた値
3. `/vendor/lib/rfsa/adsp`
4. `/vendor/dsp/cdsp`
5. `/system/lib/rfsa/adsp`

root、SELinux変更、system/vendor領域の書換え、端末全体の永続設定は不要である。

## 版混在防止

Gradle configurationで`SDK root/sdk.yaml/QnnSdkBuildId.h`を読み、完全Build ID、r26c要求、必須
header/runtime/Stub/unsigned Skelを検査する。正式期待値は`2.48.40.260702151143`で、2.47 SDKは
configuration段階で拒否する。CMakeもヘッダーBuild IDを再検証する。

APK完成後は`scripts/audit_qnn_apk.ps1`が全QNN runtimeとSkelをSDK原本のSHA-256と照合し、
2.47文字列、別Vxx Stub/Skel、ローカルSDK絶対パスを拒否する。レポートは
`build/reports/qnn-apk-audit-2.48.txt`へ生成し、Gitには追加しない。

実行時はヘッダーの`QNN_SDK_BUILD_ID`と`QnnBackend_getBuildId`を比較する。providerが返すCore APIと
Backend APIも別々に記録し、不一致時はbackend作成前に失敗する。CPU fallbackは実装・成功扱いしない。

## 再現

```powershell
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME

.\scripts\run_qnn_device_tests.ps1 `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

オンラインADB端末がちょうど1台でなければ停止する。スクリプトはclean build、APK監査、更新
インストール、probe 10回、forward 10回、probe→forward→force-stop 3巡、Skel再利用、アプリ専用
コピーの破損復旧を検証する。結果とlogcatは`build/reports/qnn-device-tests-2.48/`へ保存するが、
ADB endpointは記録しない。

## 2026-07-18 実機回帰結果

正式な一括スクリプトをclean buildから実行し、probe 10回、forward 10回、
probe→forward→force-stop 3巡を完走した。追加の破損復旧probeを含む全14 probeで
`QnnBackend_create=0`、`QnnDevice_create=0`、`QnnContext_create=0`、handle non-null、
`cpu_fallback=false`を確認した。全13 forwardでgraph create/finalize/execute、runtime weight update、
second executeが成功し、最大絶対誤差は全回`6.36559e-06`だった。

展開済みSkelの再利用と、PhoneLM自身のアプリ専用コピーを4 byte破損させた後のSHA-256検出・
再展開も成功した。復旧後のSHA-256はSDK/assetの期待値
`a0b9750d900a0afbba240e3506ba63ba91ec0e7f1b3a44de9afbb6818a665a32`と一致した。
## 成功範囲とライセンス

確認対象はHTP forward、CPU側loss/gradient、runtime weight update、更新後のHTP再実行である。
HTP上でbackwardやoptimizerまで実行したとは表現しない。

ローカル開発・実機検証のためのAPK同梱と、公開APK/GitHub Releaseでの第三者再配布は分離する。
QAIRT runtime、Stub、Skelの再配布可否は未確認であり、公開可能とは結論づけない。Qualcomm配布物は
Gitへ追加せず、ライセンス確認済みの各開発者のローカルSDKからビルド時に取り込む。