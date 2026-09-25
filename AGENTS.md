# AGENTS.md

PhoneLM / HexaTrain は、実機上の Qualcomm QNN / HTP を使った言語モデル学習を扱う。
安全性・再現性・証拠の保存を優先し、変更範囲に応じて段階的に検証する。

詳細な規則の正本は `docs/agent/` 以下である。**必要なときだけ**読む。

| 文書 | 読むタイミング |
| --- | --- |
| [docs/agent/verification.md](docs/agent/verification.md) | 検証プロファイルの使い分け・失敗報告 |
| [docs/agent/qairt-policy.md](docs/agent/qairt-policy.md) | QNN / QAIRT build、APK audit、実機操作の前 |
| [docs/agent/device-test-tiers.md](docs/agent/device-test-tiers.md) | 実機・ADB・device runner の前 |
| [docs/agent/numerical-evidence.md](docs/agent/numerical-evidence.md) | 数値変更・回帰判定・結果報告の前 |
| [docs/agent/windows-editing.md](docs/agent/windows-editing.md) | Windows で編集する前、apply_patch 失敗時 |

---

## 絶対規則

これらは詳細文書に書いてあっても、常に守る。

- **Tier 3**（UI 前面化、`EXCLUSIVE_BENCHMARK`、通知/permission 変更、app data 削除、firmware/SDK 変更、長時間 training、外部公開、main への merge/push）は**ユーザーの明示指示なしに実行しない**。
- QAIRT の自動 fallback、2.47 との混在、固定値と異なる SDK の使用を禁止する。
- QNN return code の成功と tensor の有限性を**別々に**確認する。
- 「NPUだけで学習した」「CPUを完全に使用していない」「QNNが自動微分した」と誇張しない。
- raw checkpoint、logcat、ADB endpoint、ローカル絶対 path を commit しない。
- **reset / rebase / amend / force push / force-with-lease / 履歴書き換えを禁止する。**
- ユーザーの既存変更を discard / stash / 勝手に commit しない。
- main への push / merge はユーザーの明示指示なしに行わない。
- QAIRT version を勝手に変更しない。長時間の実機 training を勝手に開始しない。
- 安全条件、検証条件、実機 Tier、QAIRT 固定条件を、ユーザーの明示指示なしに弱めない。

---

## 検証の入口

意味の明確なプロファイルは `scripts/verify.ps1` が正本。**どの追加検証が必要かは、変更内容の意味を読んで AI が判断する。** ファイルごとの検証対応表は作らない。新規ファイル追加時に検証設定へ追記する必要はない。

```powershell
.\scripts\verify.ps1 -Profile Fast      # 通常開発の毎回 gate（目標 1〜3 分）
.\scripts\verify.ps1 -Profile Host      # C++ / 数値 / checkpoint / shape（PC）
.\scripts\verify.ps1 -Profile Android   # Kotlin / JNI / Gradle / APK
.\scripts\verify.ps1 -Profile Qnn       # QNN graph / QAIRT / APK audit
.\scripts\verify.ps1 -Profile Device    # 実機 headless smoke（Tier 2）
.\scripts\verify.ps1 -Profile Formal    # milestone / 正式 evidence / release のみ
```

| プロファイル | いつ使うか |
| --- | --- |
| **Fast** | 全ての通常開発。明らかな破壊の短時間検出だけ |
| **Host** | 数式・optimizer・checkpoint・shape・dataset split など PC で完結する変更 |
| **Android** | Kotlin / Compose / JNI / Gradle / packaging の変更 |
| **Qnn** | QNN node / tensor / QAIRT package の変更（shape 変更は Host も） |
| **Device** | HTP / HVX / FastRPC / lifecycle / 実機速度など実機でしか確認できない変更 |
| **Formal** | 正式な研究結果の確定、公開 evidence、release、ユーザー明示の formal のみ |

内部の実際の test 一覧は書かない。実行内容は `scripts/verify.ps1` と [docs/agent/verification.md](docs/agent/verification.md) が管理する。

QNN / 実機 / 公開 bundle は Fast だけでは完了しない。追加 gate を省略した結果を PASS と表現しない。必須工程が未実行なら `BLOCKED` と報告する（[verification.md](docs/agent/verification.md)）。

---

## 通常の作業手順

```text
1. 関連コードと必要な詳細文書を読む
2. 変更する
3. Fast を実行する
4. 変更内容を見て、必要なら Host / Android / Qnn / Device を追加する
5. 差分を確認する
6. commit する
7. 必要なら PR を作る
8. Formal は正式結果・milestone・release 等でのみ使う
```

開発中の軽微な修正ごとに Full / Formal を反復しない。
一方で、検証時間短縮を理由に安全条件や QNN correctness を省略しない。

開始時に branch / HEAD / upstream / `git status` を確認する。
終了時に `git status` を確認し、検証生成物が原則 `build/` 以下だけであることを確かめる。

---

## 開発中の指針

- **PC は安価で高速な正しさの確認場所。** PC で確認できるものを理由なく実機で確認しない。
- **Active run 保護:** 実機操作前に既存の active training がないことを確認する。状態不明なら fail closed。`am force-stop` / `pm clear` / reboot を勝手に実行しない（[device-test-tiers.md](docs/agent/device-test-tiers.md)）。
- **QNN graph:** producer の推論 shape と宣言 output shape を一致させる。Reduce の axis / keep_dims を validator で確認する。APP_READ / APP_WRITE の方向を維持する。node / tensor 変更後は Host の shape validator を実行する。
- **数値:** HTP 内部精度を FP32 と断定しない。clamp / epsilon / LR 低下だけで原因修正扱いにしない。単一 run を独立した再現証拠としない（[numerical-evidence.md](docs/agent/numerical-evidence.md)）。
- **Private data:** 学習用 corpus と派生物は ignored な `build/private-data/` 以下。validation / final_test を training に混入させない。
- **表現:** 「学習 step の数値演算を HTP で実行した」と表現する。NPU-only と主張しない。
- **commit message:** `type(scope): 概要`。type は `feat` / `fix` / `test` / `build` / `docs`。scope 例: `qnn` / `android` / `lm` / `scripts`。
- **commit 範囲:** 自分が今回変更した path だけを stage する。QAIRT ライブラリ、Stub、Skel、MNN source、APK、private corpus、raw logcat を commit しない。
- **専用 `codex/*` 作業 branch** への通常 fast-forward push は保存として可。main への push / merge は禁止のまま。

Windows 編集で組み込み `apply_patch` が失敗したら [windows-editing.md](docs/agent/windows-editing.md) を読む。曖昧な置換や `Set-Content` による無条件上書きをしない。

---

## 作業に応じた詳細文書

| 作業 | 読む文書 |
| --- | --- |
| 検証を選ぶ / FAIL を報告する | `docs/agent/verification.md` |
| QNN / QAIRT / APK / 実機 | `docs/agent/qairt-policy.md` |
| 実機・ADB・device runner | `docs/agent/device-test-tiers.md` |
| 数値・回帰・evidence | `docs/agent/numerical-evidence.md` |
| Windows 編集 | `docs/agent/windows-editing.md` |

誤り・重複・陳腐化を見つけたら文書を更新してよい。
ただし絶対規則と安全条件は、ユーザーの明示指示なしに弱めない。変更理由を完了報告に書く。
