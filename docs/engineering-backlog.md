# HexaTrain: Engineering / Refactoring Backlog

## 目的

この文書は、HexaTrainの研究課題ではなく、開発基盤・コード構造の改善候補を管理する。

`docs/research-priorities.md` とは分離する。

ここにある項目は「順番に全部実装するTODO」ではない。

実際の研究開発で、

- AIが同じ構造を何度も調査している
- 同じ処理の複製が増えている
- 検証待ちが研究反復を妨げている
- 巨大ファイルのため局所変更が難しい
- 新しい実験を追加するたびに専用コードが増える

など、具体的なボトルネックとして現れた時点で着手する。

リファクタリング自体を目的にしない。

---

## 現在の状態

2026-09-25時点で、AI開発基盤のPhase 1〜3は完了済み。

- Fast / Host / Android / Qnn / Device / Formal の意味別検証入口
- Fastは約1分
- `AGENTS.md` を短い入口へ整理
- 詳細規則を `docs/agent/` へ分離
- `.agents/skills` を管理元としたskills SSOT
- `.commandcode/skills` のstale検出をFastへ統合
- `.commandcode/settings.json` のセッション残骸を除去
- path → test の巨大な対応表は作らない方針

この部分は、実際の問題が発生しない限り再設計しない。

---

# 優先候補

## E1: host test の標準ビルド化

### 現状

host testではPowerShellが、

- source一覧
- include一覧
- compiler invocation
- object生成
- object再利用
- link
- test実行

など、ビルドシステムに近い役割を持っている。

直近ではobject再利用によって反復速度を改善したが、
独自ビルド処理自体の複雑さは残っている。

### 候補

CMake + Ninja + CTestへ段階的に移行できるか調査する。

`verify.ps1 -Profile Host` というAI向け入口は維持し、
内部実装だけ標準ビルドへ移すことを検討する。

### 着手条件

以下のいずれかが実際の問題になった場合に着手する。

- host test追加時にsource/include列挙の修正が頻発する
- compile依存関係の管理でミスが発生する
- PowerShell側のcompile/cache実装の保守負担が増える
- C++研究の反復時間が明確なボトルネックになる

### 進め方

全面移行しない。

1. 現行host test構造を監査
2. 小さいcontract testを1つだけCMake/Ninjaで再現
3. 現行結果との一致を確認
4. contract suiteへ拡大
5. 必要ならdiagnosticへ拡大
6. 不要になったPowerShell compile/cache処理を削除

新しい仕組みを追加するだけで旧処理を残し続けない。

---

## E2: 新規実験の `runner + experiment spec` 化

### 現状

多数の `run_*` PowerShellが存在し、
実験条件と実行処理が同じscript内に混在しているものがある。

過去実験の再現性のため、既存scriptを一括変換しない。

### 候補

次に新しい実験を追加するとき、

```text
experiment spec
      ↓
common runner
```

方式を試す。

実験specには、

- model
- optimizer
- steps
- seed
- schedule
- checkpoint policy
- evaluation

など、実験ごとの差分を宣言する。

training / resume / checkpoint / evaluation / evidence処理など、
共通部分だけrunner側に置く。

### 着手条件

- 新しい実験のために既存runnerをコピーしようとしている
- 類似する `run_*` がさらに増える
- 実験間の違いを知るために巨大PowerShellのdiffが必要
- checkpoint / resume / evaluation処理を再実装している

### 重要

過去の正式実験scriptは必要ならfrozen/historicalとして残す。

最初から56本等を一括移行しない。

新しい実験1件で有効性を確認してから広げる。

---

## E3: `qnn_transformer_training.cpp` の段階分割

### 現状

`app/src/main/cpp/qnn/qnn_transformer_training.cpp` は巨大で、

- production training
- checkpoint
- validation
- diagnostics
- formal experiment
- evaluation
- reproducibility
- report/result処理

など複数の責務が同居している。

AIが局所的な変更を行う際、
必要以上に広い文脈を理解する必要がある。

### 目的

行数削減自体ではなく、

> 1回の変更でAIが理解する必要のある意味の範囲を小さくする

こと。

### 着手順候補

比較的独立しているものから分離する。

1. report / result serialization
2. checkpoint helper
3. validation / finite checks
4. diagnostic-only処理
5. formal experiment orchestration
6. evaluation helper

その後に必要なら、

- graph construction
- tensor / parameter binding
- execution
- training math

を検討する。

### 重要

同じ変更で、

```text
ファイル分割
+
数式変更
+
QNN graph変更
+
新機能追加
```

を行わない。

挙動不変の分割と機能変更を分ける。

---

# 低優先度 / 必要になった場合のみ

## E4: historical evidence の検索ノイズ削減

`docs/results/` には大量の過去evidenceがあり、
AIのrepository検索でproduction sourceと混ざることがある。

ただし、再現性のため削除しない。

最初から大量移動もしない。

必要になった場合は、

- current情報への索引
- historical evidenceの扱いの明文化
- AI向け検索方針

など、低リスクな方法から試す。

---

## E5: QAIRT/toolchain設定の機械可読SSOT化

現在の設定共有が実際に問題になった場合のみ検討する。

JSON / TOML等への移行そのものを目的にしない。

着手候補となる条件:

- 設定項目が増えて複数parserの維持が難しくなった
- PowerShell / Gradle / CMake間の読み取り不整合が発生した
- regexによる読み取りが実際に壊れた

現在のQAIRT固定条件は弱めない。

---

## E6: `.commandcode/skills` のGit追跡廃止

現在は、

```text
.agents/skills        ← 管理元
      ↓ sync
.commandcode/skills   ← 生成物
```

として二重手編集は解消済み。

Fastでstaleも検出する。

Git追跡を外すことで明確な利益が得られ、
clone直後のツール互換性にも問題がないと確認できるまで現状維持とする。

---

# 非対象

## PhoneLM → HexaTrain 全面rename

現時点では優先しない。

package / JNI / Gradle / CMake / checkpoint / evidence等への影響が広い割に、
研究開発速度への改善が小さい。

historical/internal identifierとして扱う。

---

# リファクタリング着手時の判断基準

新しい改善に着手する前に、以下を確認する。

1. 現在どの研究作業が遅くなっているか
2. AIが具体的にどこで迷っているか
3. 問題を実測または具体例で示せるか
4. 改善後に既存の複雑なコードを削除できるか
5. 新しい仕組み自体が次の負債にならないか

特に、

> 新しい抽象化を追加しただけで、既存の複雑さが残る

変更は避ける。

理想は、

> 新しい構造へ移行した結果、旧構造を削除できる

ことである。

---

# 研究との関係

研究優先順位は `docs/research-priorities.md` で管理する。

このengineering backlogは研究課題の順番を変更しない。

通常は研究を優先し、
この文書の項目が研究速度を実際に妨げた場合だけ一時的に着手する。
