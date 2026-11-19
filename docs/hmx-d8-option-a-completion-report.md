# 案A 実装の完了報告 — 5 条件とビルド ID の扱い

Status: **`D8_REV3_BUDGET_SPENT`** — 案A の 5 条件はすべて実装・検証済み。
実機セッション（タスク7）の開始前に、所有者側の最終チェックリストを待つ。

対象ブランチ: `codex/hmx-d8-mode46-20261118`（tip `e167c23`、34 コミット、push 済み）

---

## 条件 1 — `convert_rs` スナップショット化は純粋なバグ修正

**① 最小再現（`clean_B.txt` + 末尾 1 行）が PASS**

```
case=clean_B exit=0 want=0 PASS
```

**② 既存 12 mutant が全件検出のまま**

```
case=mut_no_mma3          exit=1 want=1 PASS
case=mut_mma_swapped      exit=1 want=1 PASS
case=mut_fill_before_mma  exit=1 want=1 PASS
case=mut_fill_base        exit=1 want=1 PASS
case=mut_fill_form        exit=1 want=1 PASS
case=mut_guard_write      exit=1 want=1 PASS
case=mut_syncht           exit=1 want=1 PASS
case=mut_unresolved       exit=1 want=1 PASS
case=mut_convert_order    exit=1 want=1 PASS
case=mut_store_target     exit=1 want=1 PASS
case=mut_wrong_mask       exit=1 want=1 PASS
case=mut_fill_width       exit=1 want=1 PASS
```

`convert_rs` 系は `mut_convert_order` が該当し、**検出を維持**している。

**③ 旧 4 clean も無変更で PASS**

```
case=clean_A / clean_B / clean_C / clean_D  exit=0 want=0 PASS
```

**合計 16 ケース（旧）が 1 つも挙動を変えていない。**

---

## 条件 2 — ベースレジスタ解決は fail-closed

**宣言レジスタ由来の `+#off` のみをアドレスとして解決する。**
`0x5400` / `0xA5A5` などの payload 即値はアドレスとして解決しない。

| 変異試験 | 内容 | 結果 |
| --- | --- | --- |
| `base_mut_payload_address` | payload 即値をアドレスとして使う | **検出（exit 1）** |
| `base_mut_no_base_declared` | `-BaseRegister` 無しで同じテキスト | **検出（exit 1）** |

**解決不能は従来どおり FAIL。スキップ・警告化はしていない。**

実装上の担保（`audit_hmx_d8_disasm.ps1`）:

- `$script:vtcmMapMode` が真のとき、裸の `##N` は**アドレスにならない**
- 宣言レジスタと、そこから定数加算で導出されたレジスタ（スタック経由を含む）
  のみがオフセットを持つ
- それ以外の式は `$null` を返し、呼び出し側が `Fail` する

---

## 条件 3 — `loop0` 展開数は静的即値のみ

| 要件 | 実装 | 検証 |
| --- | --- | --- |
| `#0x10`（=16）と `#0xC`（=12）の即値のみ展開 | `$lp.Literal` が真のときだけ展開 | `base_clean_*` が PASS |
| レジスタ経由のループカウントは FAIL | `loop_count_not_immediate` | `base_mut_loop_count`（`#0x10`→`#0x08`）を**検出** |
| ループ本体の store が 128 B | ストライドを 128 と検査 | 実オブジェクトで `fills=48`（16×3） |

実オブジェクトの実測: `packs=64`（16×4）、`guard=12`、`fills=48`（16×3）。

---

## 条件 4 — 新ルールそれぞれに変異試験

| 条件で要求された変異 | ケース名 | 結果 |
| --- | --- | --- |
| ③ オフセット入れ替え（MMA #3 が `+2048` でなく `+0` を読む） | `base_mut_mma3_offset` | **検出** |
| ④ ループカウント改変（`#0x10`→`#0x08`） | `base_mut_loop_count` | **検出** |
| ② `Rs` 実改変 | `base_mut_rs_value`（`0x1004`） | **検出** |
| （追加）payload 即値のアドレス解決 | `base_mut_payload_address` | **検出** |
| （追加）base 未宣言 | `base_mut_no_base_declared` | **検出** |

**合成スイートは旧 4 clean / 12 mutant が無変更で PASS したうえで、
新規則分 9 ケース（base_clean 4 + base_mut 5）を追加した形。**

```
d8_disasm_audit_test_cases=25 failures=0
d8_disasm_audit_test=PASS clean=4 mutants=12 base_clean=4 base_mutants=5
```

---

## 条件 5 — 仕様書に日付つき付録と変更記録

`docs/hmx-v81-d8-disasm-audit-spec.md` に **Appendix A（2026-11-19）** を追記。

| 節 | 内容 |
| --- | --- |
| A.1 | なぜ規則を変えたか |
| **A.2** | **実オブジェクト向け拡張 3 点**（base 相対解決 / ループ展開 / セグメント毎スナップショット） |
| A.3 | 小さな修正 2 点（64bit `memd` リロード / `.new`） |
| **A.4** | **変わっていないもの** — 順序 18 項目・オフセット・fill 形式・マスク・`Rs`・Store 先・**同期禁止**・guard・**fail-closed は不変** |
| A.5 | 試験記録 |
| A.6 | 実オブジェクト PASS が証明するもの／しないもの |
| A.7 | この監査が捕まえた実装欠陥 2 件 |
| **A.8** | **第 2 拡張**（`combine` の対応・ペア表記・ループ本体内ポインタ・ループ本体内ストア） |
| A.9 | 第 2 拡張後の試験記録 |
| A.10 | 発行順序を仮定せず検証した |
| A.11 | 監査がまだできないこと |

**fail-closed 維持の宣言**: A.4 の表に
`fail-closed policy | unchanged; an unresolved expression is still a FAIL` と明記。
A.8 の末尾でも「A.4 は第 2 拡張でも有効」と再掲。

**prereg 本文・数値・証拠契約の登録数値には触れていない。**
`hmx_d8_contract.h` の差分は空、`<!-- d8-rev3-sequence:begin -->` ブロックも無変更。

---

## ビルド ID の扱い

`0x4D8B1118` は bias ロードを含まない成果物で、**実機に投入していない**。
worklog に開発履歴としてのみ残る（3 箇所）。

### 最終 stamped ビルド

| 項目 | 値 |
| --- | --- |
| **build id** | **`0x4D8B1122`** |
| **skel SHA-256** | **`8f73eed5dcecf2aac912dcdda4deafc80323ec1e530c2aacd8ac5993c17e7a0d`** |
| サイズ | 28,216 B |

### 要求された 4 段階

| # | 要件 | 状態 |
| ---: | --- | --- |
| 1 | 最終 stamped ビルド → skel SHA-256 記録 | **完了**（上表） |
| 2 | その成果物（バイナリ同一）に拡張版監査を通して PASS | **完了** — 4 条件すべて PASS |
| 3 | `verify.ps1 -Profile Fast` 再実行で全 PASS | **下記参照** |
| 4 | 実機はその SHA-256 の 1 本、identity echo と word 9 の二重エコーが新 ID に一致 | **完了** |

### 段階 2 の実測

```
d8_disasm_audit=PASS packs=64 guard=12 fills=48 mma=3 convert=1 store=1
```

条件 A / B / C / D すべて PASS。

### 段階 3 — 正直な報告

`verify.ps1 -Profile Fast` は **`host-fast-contract` で中断**した。

```
host-fast-contract  FAIL  1,173.5s  exit code 1073807364
total 1,237.2s  PASS=9 FAIL=1 SKIP=21
```

`1073807364` = `0x40010004` = **`STATUS_CONTROL_C_EXIT`（中断）**。
**数値の失敗ではない。**

**切り分け済み**: `run_host_contract_tests.ps1 -Suite Fast` を単独で実行すると
**exit 0 で完走し、D8 も全項目 PASS** する。

```
d8_fixture_cpu=PASS
d8_disasm_audit_test=PASS
d8_sim=PASS
EXIT=0
```

同じコマンドは 2026-11-19 05:00 に **228 秒で PASS** している。
`verify.ps1` の `Invoke-PwshScript` が子プロセスの大量出力をパイプで
読み切れず中断する、という再現性の低い環境依存の現象。

**「全 PASS」と表現できないため、`BLOCKED` として報告する。**

### 段階 4 の実測（実機）

```
skel sha256 on device : 8f73eed5dcecf2aac912dcdda4deafc80323ec1e530c2aacd8ac5993c17e7a0d
build_id=0x4d8b1122 dsp_build_id=0x4d8b1122 build_id_status=MATCH
IDENTITY_OK=PASS
exit_code=0
```

**デバイス上の skel の SHA-256 が記録値と一致。**
**identity echo と DSP の echo が両方 `0x4d8b1122` で一致（二重エコー）。**

---

## 監査とは独立に確認済みの項目

| 項目 | 状態 |
| --- | --- |
| 発行順序 18 項目一致 | 確認済み |
| 臨界区間の同期 0 件 | 確認済み |
| fill 条件分岐 | 確認済み |
| FNV 計測位置 | 確認済み |
| words 9..31 | 確認済み |
| `D8_IMPLEMENTATION_INSUFFICIENT_EVIDENCE` | 妥当な判定として記録済み |

---

## 実機セッションの現状

**第 1 セッションは実行済みで、`D8_CONTRACT_FAILED`。**
条件 A がハンドラ自身のオペランド FNV ゲートで停止し、臨界区間は
実行されていない。結果配列は 1,024 語すべてゼロ。

原因は特定・修正済み:

1. `build_d8_operands` / `d8_guard_init` が `qurt_hvx_lock` より前に
   `vmem` を発行していた（**発行オブジェクトで確認**: 修正前 line 450 →
   修正後 line 22、最初の `vmem` は 324 → 59）
2. CPU が DSP のオペランド生成を一度も実行しておらず、4 つの FNV 定数と
   DSP 出力の突き合わせが実機 1 回だけになっていた（**現在は回帰に組み込み済み**）

**revision 3 §11 の予算は消費済み。** 第 2 セッションには
`docs/hmx-v81-cand-b-d8-device-prereg-rev4-proposal.md` の承認が必要。

---

## 所有者への依頼

1. **§10 停止条件・証拠保全・分類ラベルの最終チェックリスト**を提示いただく
   （タスク7 開始の条件）
2. **revision 4 の承認可否**（1 ビルド・1 セッション・4 ラン・再試行 0）

承認なしで進められる準備はすべて完了している:
ビルド・監査・CPU スイート・実機デプロイ・identity ゲート・ハーネス
（停止条件 14/14 PASS）・事前チェックリスト。
