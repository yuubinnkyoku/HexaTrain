# 検証方針

意味の明確なプロファイル入口は `scripts/verify.ps1` である。どの追加プロファイルを使うかは、変更内容の意味を読んだ AI が選ぶ。ファイルごとの検証対応表は作らない。

```powershell
.\scripts\verify.ps1 -Profile Fast
.\scripts\verify.ps1 -Profile Host
.\scripts\verify.ps1 -Profile Android
.\scripts\verify.ps1 -Profile Qnn
.\scripts\verify.ps1 -Profile Device
.\scripts\verify.ps1 -Profile Formal
.\scripts\verify.ps1 -Profile PrGate
```

内部実装は共有する（`scripts/verify_common.ps1` と既存 runner）。`scripts/verify_local.ps1` は Fast / PrGate / Full (Formal) の共有実装として残る。

| プロファイル | いつ使うか | 含むものの意味 |
| --- | --- | --- |
| `Fast` | 全ての通常開発・反復 | 明らかな破壊の短時間検出（目標 1〜3 分） |
| `Host` | C++ / 数値 / checkpoint / shape など PC で完結する変更 | 既存 host contract + diagnostic、parity battery |
| `Android` | Kotlin / JNI / Gradle / UI / packaging | JVM test、`assembleDebug`、`assembleDebugAndroidTest` |
| `Qnn` | QNN graph / tensor / QAIRT package 周辺 | pinned QAIRT、host contract（shape validator 含む）、QNN 有効 build、APK audit |
| `Device` | 実機でしか確認できない変更 | Tier 2 headless smoke（`BACKGROUND_CORRECTNESS`） |
| `Formal` | milestone / 正式 evidence / release / 明示 formal | 従来 Full（heavy diagnostic full runs をすべて実行） |
| `PrGate` | CI / pre-integration | 従来 PrGate（cheap correctness 常時 + heavy を fail-closed 選択） |

`Fast` は日常の軽量 gate であり、Host / Android / Qnn / Device は意味ごとの独立入口である。AI は変更に必要なものを選んで呼ぶ。Formal の保証を Fast や PrGate で縮小しない。

## Fast

通常の編集・PRで毎回走る高速 gate。目標 1〜3 分。

含む:

- `git diff --check`
- tracked binary / secret path 監査
- PowerShell parser
- metadata 生成物の stale 確認
- runner / policy の小さい SelfTest（QAIRT selection、PrGate policy）
- 高速な JVM unit test
- 高速で重要な C++ contract（metadata exporter + CPU reference）

原則含まない:

- APK の完全 build / androidTest APK build
- QAIRT build / APK audit
- 実機
- 長時間 training
- 重い研究 diagnostic full run

「何でも確認する mini Full」にしない。実装は `verify_local.ps1 -Fast`。

## Host

PC 上で確認可能な C++・数値処理。既存 host test 資産を再利用する。

- C++ contract tests（optimizer、checkpoint round-trip、shape validator、CPU reference、deterministic fixture）
- diagnostic / research host suite
- nicopedia parity policy host battery

実装: `run_host_tests.ps1`（contract + diagnostic）+ parity battery。実機・QAIRT・APK build は含めない。

## Android

Android / Kotlin / JNI / Gradle 等の変更向け。

- `:app:testDebugUnitTest`
- `:app:assembleDebug`
- `:app:assembleDebugAndroidTest`

Android と無関係な通常変更では実行しない。ファイルパスから自動判定する例外表は作らない。

## Qnn

QNN graph / tensor / QAIRT package 周辺の変更向け。固定条件は弱めない。

- pinned QAIRT root / Build ID 確認（明示引数必須）
- QNN 関連 host correctness（shape validator 含む host suite）
- QNN 有効 `assembleDebug`
- `audit_qnn_apk.ps1`（ABI / hash / path / 2.47 混入）

```powershell
.\scripts\verify.ps1 -Profile Qnn `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

## Device

実機でしか確認できない変更向け。小さく始める。`device-test-tiers.md` を先に読む。

- 既定は headless smoke（`-DeviceSuite device-probe`）
- `BACKGROUND_CORRECTNESS` のみ。`EXCLUSIVE_BENCHMARK` / UI 前面化 / 長時間 training は Tier 3 で、このプロファイルから自動実行しない
- 必要なら smoke 系 suite（例: `scale-sequence-16-smoke`）を `-DeviceSuite` で明示
- 8 / 32 step などへ進む判断は AI が変更内容を見て行う

明示的な pinned QAIRT 引数が必須。物理端末同一性・active run 保護・非破壊条件は `device-test-tiers.md` に従う。

## Formal

以下の場合のみ。

- milestone
- 正式な研究結果の確定
- 公開用 evidence の確定
- release
- ユーザーが formal verification を明示した場合

従来の引数なし Full gate を維持する。heavy diagnostic full run の削除・skip は行わない。formal evidence に QNN / 実機が必要な場合は、Formal に加えて `Qnn` / `Device` を AI が組み合わせる。

```powershell
.\scripts\verify.ps1 -Profile Formal
.\scripts\verify.ps1 -Profile Formal -WithQairt `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

## PrGate について

PrGate は「無関係な変更で重い研究 diagnostic を実行しないため」の役割を維持する。

**PrGate を、Android build / JVM / host / QNN / 実機まで自動判定する巨大な仕組みに拡張しない。**

policy 実装は `scripts/pr_gate_policy.ps1` が正本。changed-path discovery はローカルでも動き、base 解決は次の優先順。

1. 明示 `-PrGateBaseRef`
2. GitHub PR context（`GITHUB_BASE_REF` → `origin/<base>`）
3. 既存 ref の `origin/main` との merge-base

base を安全に決定できない場合や classifier が安全に分類できない場合は **heavy full runs 全部** へ fallback する。「unknown だから skip」は禁止。exporter implementation / synthetic fixture / exporter SelfTest だけの変更では対応 full diagnostic regeneration を要求しない。共有 production core / shared dataset / trajectory / multi-diagnostic library 変更では PrGate も heavy full へ fail-closed する。tracked scientific result / public evidence の実データ変更は対応する full diagnostic regeneration を要求する。通常 docs-only 変更は heavy full 不要。

`Fast` と `PrGate` は排他。

### PrGate で常に実行する cheap / correctness 層

- git diff check / tracked binary / secret path audit
- PowerShell parser（Fast と同様） / QAIRT selection self-test
- public exporter SelfTest 一式（fixture-contained。live diagnostic ReportRoot / PrivateRoot を消費しない）
- resumable runner self-test
- JVM unit tests / host contract suite / host diagnostic suite
- nicopedia parity policy host battery
- PrGate policy classifier self-test
- Android build（`-SkipAndroidBuild` 指定時のみ skip）

## 変更にどのプロファイルを足すか（AI 判断の指針）

対応表ではなく意味で選ぶ。例:

| 変更の意味 | 使うプロファイル |
| --- | --- |
| docs / scripts の軽微な修正 | Fast |
| 数式・optimizer・checkpoint codec・shape・dataset split | Fast + Host |
| Kotlin / Compose / JNI / Gradle / APK packaging | Fast + Android |
| QNN node / tensor / QAIRT package | Fast + Qnn（shape 変更時は Host も） |
| 実機固有（HTP / HVX / FastRPC / lifecycle / 実機速度） | 必要な下位 + Device |
| 正式結果・公開 evidence・release | Formal（必要なら Qnn / Device） |

新規ファイル追加時に検証設定へ追記する必要はない。意味が新しい領域なら既存プロファイルを呼ぶか、まだ無い意味なら新しいプロファイルを足す（ファイル名マッチ表は足さない）。

```powershell
# 従来互換（CI / 既存 caller）
.\scripts\verify_local.ps1
.\scripts\verify_local.ps1 -Fast
.\scripts\verify_local.ps1 -PrGate
.\scripts\verify_local.ps1 -SkipAndroidBuild
.\scripts\verify_local.ps1 -Clean
.\scripts\verify_local.ps1 -WithQairt -QairtSdkRoot '...' -ExpectedBuildId '...'
```

incremental build が既定であり、cache 不整合など clean が必要な場合だけ `-Clean` を使う。

## 検証不能・失敗

必須工程が未実行または FAIL なら完了と報告しない。次を含めて `BLOCKED` と報告する。

- 未実行または失敗した工程
- 原因と、確認できた範囲
- 再現コマンド
- 再開に必要な外部条件

不足 tool、SDK、Android component を勝手にインストール・移動・更新しない。失敗を別の軽い gate で置き換えない。

## CI

`.github/workflows/verify.yml` も同じ `verify_local.ps1` を使い、通常の pull request では `-PrGate` を実行する。main push では push 全体の ref 更新を分類するため、GitHub push event の `github.event.before` を `-PrGateBaseRef` に渡す。CI 専用の別テスト列は作らない。

引数なし Full は CI の通常 PR ごとに無条件実行せず、milestone / formal evidence / release / explicit formal validation で維持する。Android build 依存の pinned MNN source は ignored な `third_party/MNN/` に取得し、QAIRT SDK、ADB 端末、repository secrets、APK artifact を使わない。
