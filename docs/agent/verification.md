# 検証方針

`scripts/verify_local.ps1` は、全変更に共通するローカル基礎ゲートである。既定では実機・QAIRT SDKを使わず、tracked fileを変更しない。`git-diff-check`、tracked binary/secret path監査、QAIRT選択self-test、公開exporter self-test、resumable runner self-test、JVM unit tests、C++ host tests、`assembleDebug`、`assembleDebugAndroidTest`を一括実行する。

```powershell
.\scripts\verify_local.ps1
.\scripts\verify_local.ps1 -SkipAndroidBuild
.\scripts\verify_local.ps1 -Clean
.\scripts\verify_local.ps1 -Fast
.\scripts\verify_local.ps1 -PrGate
```

incremental buildが既定であり、cache不整合などcleanが必要な場合だけ `-Clean` を使う。

## 検証プロファイル

| プロファイル | 位置づけ | heavy diagnostic full runs |
| --- | --- | --- |
| `-Fast` | 開発・反復中の軽量gate | すべてSKIP |
| `-PrGate` | CI / pre-integration向け | path / semantic dependency に応じて選択（fail-closed） |
| 引数なし Full | milestone / formal evidence / 明示的formal validation | すべて実行 |

`Fast < PrGate < Full` である。Fullの挙動は維持し、heavy full runの削除・skipは行わない。`-PrGate` が指定された場合にのみ、heavy diagnostic full runをdependency-awareに選択する。`-SkipAndroidBuild` / `-Clean` / `-WithQairt` との互換性は維持する。`-Fast` と `-PrGate` は排他。

### PrGateで常に実行するcheap / correctness層

- git diff check / tracked binary / secret path audit
- PowerShell parser（`-Fast` と同様） / QAIRT selection self-test
- public exporter SelfTest一式（fixture-contained。live diagnostic ReportRoot / PrivateRootを消費しない）
- resumable runner self-test
- JVM unit tests / host contract suite / host diagnostic suite
- nicopedia parity policy host battery
- PrGate policy classifier self-test
- Android build（`-SkipAndroidBuild` 指定時のみskip）

### PrGate policyの原則

policy実装は `scripts/pr_gate_policy.ps1` が正本。changed-path discoveryはローカルでも動き、base解決は次の優先順。

1. 明示 `-PrGateBaseRef`
2. GitHub PR context（`GITHUB_BASE_REF` → `origin/<base>`）
3. 既存refの `origin/main` との merge-base

baseを安全に決定できない場合やclassifierが安全に分類できない場合は **heavy full runs全部** へfallbackする。「unknownだからskip」は禁止。exporter implementation / synthetic fixture / exporter SelfTestだけの変更では対応full diagnostic regenerationを要求しない。共有production core / shared dataset / trajectory / multi-diagnostic library変更ではPrGateもheavy fullへfail-closedする。tracked scientific result / public evidenceの実データ変更は対応するfull diagnostic regenerationを要求する。通常docs-only変更はheavy full不要。

## 変更種別ごとのgate

| 変更 | 最終確認に必要なgate |
| --- | --- |
| docs/scripts-only | 通常PRは `verify_local.ps1 -PrGate -SkipAndroidBuild` を使用可。milestone / formal evidence確定ではFull |
| Android/Kotlin/JNI/Gradle/CMake/APK packaging | 通常PRは `verify_local.ps1 -PrGate`。`-SkipAndroidBuild` は途中確認だけ |
| QNN node/tensor/shape | PrGateに加え `run_host_tests.ps1` のshape validator。shape変更はgraph-map exporterとnegative testも更新 |
| QNN有効build/APK | PrGateに加え固定引数付き `verify_local.ps1 -WithQairt`。`qairt-policy.md` のAPK auditを含む |
| 実機試験 | PrGate / QNN gateに加え `device-test-tiers.md` の該当Tier gate |
| 公開bundle | 対応するallow-list exporter self-test、source evidence照合、公開物監査。formal evidence確定時はFull |
| PR / pre-integration | `verify_local.ps1 -PrGate`。shared core / gate-policy / unknown relevant pathはheavy-allへfail-closed |

```powershell
.\scripts\verify_local.ps1 `
  -WithQairt `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

QNN build、実機試験、公開bundleは基礎gateだけでは完了しない。追加gateを省略した結果をPASSと表現しない。

## 検証不能・失敗

必須工程が未実行またはFAILなら完了と報告しない。次を含めて `BLOCKED` と報告する。

- 未実行または失敗した工程
- 原因と、確認できた範囲
- 再現コマンド
- 再開に必要な外部条件

不足tool、SDK、Android componentを勝手にインストール・移動・更新しない。失敗を別の軽いgateで置き換えない。

## CI

`.github/workflows/verify.yml` も同じ `verify_local.ps1` を使い、通常のpull requestでは `-PrGate` を実行する。main pushでは push全体のref更新を分類するため、GitHub push eventの `github.event.before`（push直前のmain SHA）を `-PrGateBaseRef` に渡す。これにより複数commit direct pushやmerge後でも、最後の1commitだけでなくpush全体をchanged-path discoveryの対象にし、gate-policy変更・shared production core・unknown relevant pathはheavy-allへfail-closedする。CI専用の別テスト列は作らない。

引数なしFullはCIの通常PRごとに無条件実行せず、milestone / formal evidence / release / explicit formal validationで維持する。Android build依存のpinned MNN sourceはignoredな `third_party/MNN/` に取得し、QAIRT SDK、ADB端末、repository secrets、APK artifactを使わない。
