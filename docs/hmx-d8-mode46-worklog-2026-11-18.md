# Phase 55 補足: HexaTrain HMX D8 Mode 46 — 作業記録

対象: 2026-11-18
worktree: `D:\ghq\github.com\yuubinnkyoku\HexaTrain-mode46-20261118`
ブランチ: `codex/hmx-d8-mode46-20261118`（`92c408757726d30efd03895c6c489ca14f268c7b` から作成）

---

## §0 開始時の確認

| 項目 | 実測 |
| --- | --- |
| 開始点の HEAD | `92c408757726d30efd03895c6c489ca14f268c7b`（指示書の期待値と**一致**） |
| 確認方法 | `git rev-parse origin/experiment/hmx-acc-residual-feedback` |
| ローカルの `experiment/` ブランチ | `3d4e9f96`（**古い**。remote を基準にした） |
| メイン repo | `PhoneLM`（**触れてはいけない旧 worktree**） |
| 新 worktree | `HexaTrain-mode46-20261118` を新規作成 |
| 新ブランチ | `codex/hmx-d8-mode46-20261118` |

**注意:** ローカルの `experiment/hmx-acc-residual-feedback` は `3d4e9f96` で、
指示書の `92c4087` ではなかった。`origin/` 側が `92c4087` だったので、
**remote の commit を直接指定して worktree を作成**した。

---

## §1 タスク1: revision 3 の正式登録（完了）

### §1.1 変更内容

| ファイル | 変更 |
| --- | --- |
| `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md` | Status を `D8_REV3_PROPOSED_CPU_ONLY` → **`D8_REV3_REGISTERED`**。タイトルの "(Revision 3 Proposal)" → "(Revision 3, Registered)"。ラベル表を登録後の値に更新（Mode 46 = **registered, NOT_RUN**、Device approval = **GRANTED for the revision 3 §11 budget only**）。Revision history の revision 3 を `registered 2026-11-18 (owner authorisation, CPU contract verified 2026-11-10)` に。**Change record 表を新設**し、所有者承認を追記。§11 の Authorisation を `GRANTED 2026-11-18 for this budget only` に。§14 の未解決項目 1 を「registered 2026-11-18」に更新 |
| `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md` | Status を登録済みに。タイトルを "(Revision 3, Registered)" に。Change record 表を新設 |
| `docs/hmx-v81-cand-b-d8-device-prereg.md`（revision 2） | **supersession note のみ追加**。本文と数値は無変更 |

### §1.2 登録数値が無変更であることの確認

`git diff` で確認した（**ヒット 0 件**）:

```
$ git diff -U0 | grep -E "^[+-].*(e5193d99|c6231266|3b2d22d2|8b88609f|a66af695|c9abf30e|f0b3b4aa|b6328e88|5285b463|f5ebd5c5)"
（出力なし）
```

**命令順序ブロック `<!-- d8-rev3-sequence:begin -->` 〜 `end` も無変更**（18 行そのまま）。
`d8-q-map` ブロックも無変更。

**唯一の「+/-」ヒットは revision history の行**（State 列を `proposed, CPU only` から
`registered ...` に変えたもの）で、**登録数値ではない**。

### §1.3 コミット

`e3de74f` — `docs(hmx): register D8 revision 3 and Mode 46`

---

## §2 重大な発見: CWD 依存の相対パスが旧 worktree に書き込んでいた

### §2.1 何が起きたか

`scripts/run_hmx_d8_sim.ps1` を**現在の作業ディレクトリ（`PhoneLM`）から**実行したところ、
シミュレータが `build/hmx_d8_reference_arrays.txt` を**相対パスで**書くため、
**`D:\ghq\github.com\yuubinnkyoku\PhoneLM\build\hmx_d8_reference_arrays.txt` に
ファイルが作られた。**

**これは指示書 §1「旧作業ツリー `PhoneLM` とそのバックアップには一切触れない」への違反である。**

### §2.2 影響と対処

| 項目 | 状態 |
| --- | --- |
| 書き込まれたファイル | `PhoneLM\build\hmx_d8_reference_arrays.txt`（1 件） |
| git 管理下か | **`build/` は `.gitignore` 対象**（`git check-ignore` で確認） |
| PhoneLM の追跡ファイルへの影響 | **なし**（`git status --porcelain` は 8 件で、作業前と同一） |
| 対処 | **削除した**（`Resolve-Path` で絶対パスを確認してから削除） |

**追跡ファイルは 1 件も変わっていない。** ただし**ファイルを作った事実は違反**であり、
記録する。

### §2.3 原因

`hmx_d8_min_sim.c` の 2120 行:

```c
const char *dir = "build/hmx_d8_reference_arrays.txt";
```

**相対パスである。** `run_hmx_d8_sim.ps1` は `& $gccExe` で起動するので、
**PowerShell の現在の作業ディレクトリ**がそのまま使われる。

### §2.4 正しい実行方法（以後これを使う）

```powershell
Push-Location D:\ghq\github.com\yuubinnkyoku\HexaTrain-mode46-20261118
try { .\scripts\run_hmx_d8_sim.ps1 } finally { Pop-Location }
```

**worktree を CWD にしてから実行する。** これで `build/` が正しい場所に作られる。

### §2.5 この落とし穴が引き起こした誤診断

CWD が違うと `build/hmx_d8_reference_arrays.txt` が期待された場所に無いため、
**Python の独立監査が `SKIP (no dump ...)` で exit 2 になり、mutation check が
5 件すべて FAIL と表示された。**

```
d8_mutation=trace_override_removed FAIL (exit=2 no mismatch)
...
d8_mutation_check=FAIL failed=5
d8_sim=FAIL
```

**これは「コードの欠陥」ではなく「実行場所の誤り」だった。** 私は最初これを
既存の失敗と判断し、`git stash` で base commit でも同じことが起きることを確認した
（実際そうだった）。**原因は CWD であって、コードではなかった。**

**正しい CWD で実行した結果:**

```
d8_mutation=trace_override_removed DETECTED exit=1
d8_mutation=numeric_unknown_is_safe DETECTED exit=1
d8_mutation=contract_checks_removed DETECTED exit=1
d8_mutation=mask_cd_swapped DETECTED exit=1
d8_mutation=mask_gate_removed DETECTED exit=1
d8_mutation_check=PASS mutants=5
d8_sim=PASS
```

**5/5 の mutant が検出され、`d8_sim=PASS` になった。**

---

## §3 タスク6: CPU 側回帰（完了）

正しい CWD で実行した結果:

| 項目 | 結果 |
| --- | --- |
| `run_hmx_d8_sim.ps1` | **`d8_sim=PASS`** |
| d8 checks | `d8_checks=79 fail=0` |
| rev3 selftest | `d8_rev3_checks=104 fail=0` |
| rev3 Python | `checks=140 fail=0`、`D8_REV3_INDEPENDENT_AUDIT_PASS` |
| replay | `d8_rev3_differential cases=244 mismatches=0` |
| mutation | `d8_mutation_check=PASS mutants=5` |
| disasm audit（合成） | `d8_disasm_audit_test=PASS clean=4 mutants=12` |
| feasibility | `d8_feasibility=PASS`（`D8_IMPLEMENTATION_BLOCKED` は**登録前の判定**） |
| `verify.ps1 -Profile Fast` | **`PASS=10 FAIL=0 SKIP=21`**、`total 1,681.5s` |

**4 参照の SHA-256 が登録値と一致:**

```
safe c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d
wlo  f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb
ahi  b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc
both 5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d
```

**オペランド FNV も登録値と一致:**

```
ahi c6231266  alo 3b2d22d2  whi 8b88609f  wlo a66af695
```

---

## §4 現状と次の作業

### §4.1 完了したもの

- **タスク1**: revision 3 の正式登録（数値無変更を `git diff` で確認）
- **タスク6**: CPU 側回帰（`d8_sim=PASS`、`verify.ps1 -Profile Fast` PASS）

### §4.2 未着手のもの

| タスク | 状態 | 理由 |
| --- | --- | --- |
| タスク2: DSP 側 Mode 46 登録（`probe_dsp.c`） | **未着手** | 実装量が大きい（D8 ハンドラ、VTCM マップ、18 命令の発行順序、メタデータ 23 ワード、guard の FNV、fill の条件分岐） |
| タスク3: ホスト側（`probe_host.c`） | **未着手** | 同上 |
| タスク4: ビルド | **未着手** | タスク2/3 の後 |
| タスク5: 逆アセンブル監査 | **未着手** | ビルドの後 |
| タスク7: 実機セッション | **未着手** | 上記すべての後 |

### §4.3 環境の確認結果

| 項目 | 状態 |
| --- | --- |
| Hexagon SDK | **`C:\Qualcomm\Hexagon_SDK\6.6.0.0` に存在**（文書の「SDK が無い」は古い） |
| QAIRT | `C:\Qualcomm\AIStack\QAIRT\2.47.0.260601` と `2.48.40.260702` |
| `HEXAGON_SDK_ROOT` 環境変数 | **未設定**（ビルド時に明示指定が必要） |
| `audit_hmx_d8_disasm.ps1` | 存在する（合成テストのみ、実 V81 バイナリ未適用） |
| `run_hmx_d8_sim.ps1` | 存在する |

**`d8_feas_finding=dsp_build_env_absent` は「このチェックアウトに SDK が無い」という
2026-11-10 時点の判定で、現在は SDK がある。** ただし `HEXAGON_SDK_ROOT` が
未設定なので、ビルドには明示的な設定が要る。

---

## §5 再現コマンド

```powershell
# CPU 回帰（必ず worktree を CWD にしてから）
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
Push-Location D:\ghq\github.com\yuubinnkyoku\HexaTrain-mode46-20261118
try {
    .\scripts\run_hmx_d8_sim.ps1
    .\build\hmx_d8_sim.exe --d8-sequence-rev3
    .\scripts\verify.ps1 -Profile Fast
} finally { Pop-Location }
```

**注意: `Push-Location` を省くと、`build/` が現在の CWD に作られ、
Python 監査が SKIP して mutation check が誤って FAIL する。**
