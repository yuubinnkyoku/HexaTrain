# G1 baseline promotion — systems / runtime cost（seed 2 / 3 / 4 paired）

一次 evidence: `docs/results/g1-1p5x-multiseed-3000-2026-09/seed{2,3,4}/{control,g1}/`。
**新しい training は行っていない。** seed 2 / 4 は preregistered、seed 3 は exploratory だが、
本書の runtime 比較は arm 間の paired ratio として 3 seed を平等に扱う（seed 3 の
registry role は品質判定にのみ効くので、runtime の paired 比較には影響しない）。

品質判定そのものは [g1-1p5x-multiseed-3000.md](g1-1p5x-multiseed-3000.md) の
combined interpretation を参照（本書はそれを繰り返さない）。

## 集計方法

- すべての数値は**一次 evidence の report フィールド**からのみ摘自。**推測禁止**。
- report に存在しない値は `NOT_MEASURED`。
- **host 事故は architecture / runtime cost ではない**（session interruption、eval recovery、
  detached launcher failure、stale lock、undrained pipe、PS5.1 `Double.IsFinite`、
  arm 4 annotation recovery）。該当 run の **run-total wall time** は
  `HOST_ORCHESTRATION_CONTAMINATED` として比較に使わない。
- 比較に使うのは **device 側 per-update カウンタ**（host orchestration の影響を受けない）と、
  step→秒変換のための **native `training_step_ms`**（per-update 値）のみ。

## 1. Architecture / resource cost（seed 3 除く全 seed で同一）

| metric | Control | G1 | delta | delta % |
|---|---:|---:|---:|---:|
| parameter_count | 758,528 | 760,960 | **+2,432** | **+0.321%** |
| Muon matrix count | 114 | 114 | 0 | 0% |
| Muon parameter count | 622,592 | 622,592 | 0 | 0% |
| Aux Adam parameter count | 135,936 | 138,368 | **+2,432** | **+1.789%** |

3 seed すべてで**完全に同一**。G1 の追加 parameter は gate に関する **Aux Adam 側 +2,432**
（embedding / output projection 周辺）だけで、Muon の対象行列数・Muon parameter 数は
一切増えていない。checkpoint format は V4（Control）→ V5（G1）。

- **checkpoint payload 差**: committed tree に checkpoint は含まれない（protocol で
  commit 禁止）ため `NOT_MEASURED`。ただし parameter 差が +0.321% であることから
  checkpoint payload 差も同オーダーと推測できるが、**測定していないので断定しない**。
- **optimizer state 差**: Aux Adam state のみ +2,432 element 相当。Muon momentum は
  同一 matrix 集合なので差なし。**ORT として `NOT_MEASURED`（resident set は未計測）。**
- **QNN node 差 / QNN tensor 差**: 本 report は graph 構造数を per-run で出力しない。
  **`NOT_MEASURED`。**

## 2. Training runtime cost（device-side per-update、paired）

3 seed の Δ%（G1 − Control、正 = G1 が遅い）:

| metric | seed2 | seed3 | seed4 | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| HTP forward+backward | +4.33% | +12.56% | +9.95% | **+9.95%** | +4.33% | +12.56% |
| HVX Muon | −1.71% | +10.48% | +2.10% | +2.10% | −1.71% | +10.48% |
| Aux Adam | −6.13% | +9.62% | −6.01% | −6.01% | −6.13% | +9.62% |
| parameter transfer | +10.14% | +35.20% | +20.07% | **+20.07%** | +10.14% | +35.20% |
| gradient accumulation | −10.18% | +9.55% | −9.24% | −9.24% | −10.18% | +9.55% |
| optimizer update wall | −1.75% | +10.46% | +2.02% | +2.02% | −1.75% | +10.46% |
| optimizer result move | −9.24% | +8.68% | −12.16% | −9.24% | −12.16% | +8.68% |
| mean QNN execute / call | +4.33% | +12.56% | +9.95% | **+9.95%** | +4.33% | +12.56% |

**3 seed なので統計的有意差は主張しない**（sign consistency のみ）。

### seed 3 の絶対値が外れ値であることの明示

seed 3 は**同一 device・同一 QAIRT・同一 APK** にもかかわらず、Control 単独で
per-update が seed 2 の **3.64 倍**、seed 4 の **3.22 倍**だった。device-side
カウンタで見ると:

| metric | seed2 C | seed4 C | seed3 C | seed3 / mean(seed2,4) |
|---|---:|---:|---:|---:|
| HTP forward+backward | 27.0 | 27.5 | 44.2 | **1.62x** |
| HVX Muon | 90.2 | 94.8 | 184.6 | **2.00x** |
| mean QNN execute | 3376 | 3434 | 5522 | **1.62x** |

これは seed 3 **全体の run-wide 条件差**（thermal / clock / background 負荷）であって
arm 固有のものではない。**G1/Control の ratio 自体は健全**であり（HTP execute は
seed 2 の 1.76 倍の範囲内で、seed 3 でも Control/G1 どちらも同程度に遅い）、したがって:

- **arm 間 paired ratio（Δ%）は 3 seed とも比較可能**として採用する
- **cross-seed の絶対値比較は行わない**。seed 3 の絶対秒数は文書の断定から外す

## 3. wall/update（host 汚染を除いた native per-update）

`training_step_ms` は device 側 per-update timer であり host orchestration の影響を受けない。
run-total wall（`training_total_seconds`）は build/install/pull/eval を含むため
`HOST_ORCHESTRATION_CONTAMINATED` であり比較に使わない。

| seed | Control ms/update | G1 ms/update | delta | delta % |
|---|---:|---:|---:|---:|
| 2 | 391.10 | 361.81 | −29.28 | **−7.49%** |
| 3 | 1421.94 | 1567.45 | +145.51 | **+10.23%** |
| 4 | 440.99 | 420.01 | −20.98 | **−4.76%** |

**median −4.76%**（G1 が速い）。ただし seed 3 のみ +10.23% で逆符号。
「≤5% 以内」という基準は seed 2 / 4 では満たすが、seed 3 では上回る。

run-total wall（参考、比較不可）: seed2 C 1173.3 s / G1 1085.4 s（ratio 0.925）、
seed4 C 1323.0 s / G1 1260.0 s（0.952）、seed3 C 4265.8 s / G1 4702.4 s（1.102）。

## 4. time-to-bpb（最重要）

Control の各 checkpoint の Balanced bpb を target とし、G1 がその bpb 以下へ
**初めて**到達した checkpoint を探索した。秒変換は **native `training_step_ms` の
step 比例換算**であり、checkpoint 実 timestamp は無いため `ESTIMATED_TIME_TO_BPB`
と明示する（実測 timestamp は `NOT_MEASURED`）。

各 seed の 7 target 全部（Control 500 / 1000 / 1500 / 1750 / 2000 / 2500 / 3000）:

| seed | median speedup % | G1 勝敗 |
|---|---:|---|
| 2 | **+7.49%** | G1 速い（7/7 target で G1 勝ち） |
| 3 | **−10.23%** | Control 速い（5/7 で負、2 で正） |
| 4 | **+4.76%** | G1 速い（7/7 target で G1 勝ち） |

**21 target 全体で median +4.76%、min −10.23%、max +25.99%、G1 勝ち 17/21。**
同点 step でも G1 が既に下回っている case を含む。target は結果を見て選ばず、
Control の全 checkpoint を機械的に使っている。

## 5. Baseline promotion 判定

判定基準（事前の定義）:

| verdict | 条件 |
|---|---|
| PROMOTE | quality gain 維持 / parameter ≲1% / stability 退行なし / wall ≤5% / time-to-bpb 改善 |
| PROMOTE_WITH_RUNTIME_FOLLOWUP | quality・time-to-bpb は勝つが wall +5〜10% または QNN execute overhead が明確 |
| HOLD | runtime cost が quality advantage をほぼ相殺 |
| DO_NOT_PROMOTE | time-to-bpb でも負け、または G1 固有の stability 問題 |

**判定 = `PROMOTE_WITH_RUNTIME_FOLLOWUP`**

根拠:

- **parameter overhead +0.321%**（≤1% を満たす）。memory / checkpoint overhead は
  実用上無視できる（+2,432 element）
- **quality gain は multi-seed で維持**（R1 が 4/4 seed で再現。late Dev reversal は
  1/4 seed のみ）
- **stability regression なし**（health 全 PASS、6031 再発 0、HVX / fallback / nonfinite 0）
- **time-to-bpb 改善**: median +4.76%、G1 勝ち 17/21 target
- **ただし HTP execute / mean QNN execute が +9.95%（median）で明確**.
  seed 2 / 4 の wall/update は逆に −7.5% / −4.8%（G1 が速い）だが、seed 3 は +10.2%。
  seed 間の符号が一致しないため、「wall/update +0〜3% かつ time-to-bpb 改善」という
  **PROMOTE の最も強い条件には到達しない**。

したがって **G1 を採用しつつ**、HTP execute の +9.95%（= forward+backward が
gate 分だけ増える）に対して **runtime optimization lane を残す**。
optimization 余地が大きいのは parameter transfer（+20.07% median）と
gradient accumulation の構成であり、HTP graph 自体ではない。

## 6. 最終回答（8 問）

1. **parameter overhead は何%か** → **+0.321%**（+2,432 / 758,528、3 seed 同一）
2. **memory / checkpoint overhead は実用上無視できるか** → **はい**。
   追加は Aux Adam state に +2,432 element のみ。Muon state と matrix 集合は不変。
   checkpoint payload 実測は `NOT_MEASURED` だが parameter 差が同オーダーなので
   実用上無視できる範囲。ORT / RSS は `NOT_MEASURED`。
3. **HVX Muon cost は増えているか** → **median +2.10%**。seed 2 は −1.71%（減）、
   seed 4 は +2.10%、seed 3 は +10.48%。方向は seed で揃わないが magnitude は小さい。
4. **HTP execute cost は何%増えているか** → **median +9.95%**
   （mean QNN execute / call: 3376→3522 / 5522→6216 / 3434→3776 us）。
   forward+backward が gate 分だけ増える影響。
5. **wall/update は何%増えているか** → **median −4.76%**（G1 が速い）。
   ただし seed 3 のみ **+10.23%** で符号が逆。seed 2 / 4 では G1 が速い。
6. **early quality gain が cost を上回るか** → **上回る**。
   time-to-bpb で G1 勝ち 17/21、median +4.76%。HTP execute +9.95% を quality gain が相殺し、
   むしろ wall で勝る seed もある。
7. **time-to-bpb では Control / G1 どちらが勝つか** → **G1**（median +4.76%、17/21 勝ち）。
   ただし seed 3 では median −10.23% で Control が勝つ。
8. **G1 を正式 baseline へ昇格するか** → **PROMOTE_WITH_RUNTIME_FOLLOWUP**。
   G1 を正式 baseline として採用し、HTP execute / parameter transfer の
   runtime optimization lane を別研究として残す。

## 7. この章が扱わないこと

- 品質判定そのもの（[g1-1p5x-multiseed-3000.md](g1-1p5x-multiseed-3000.md) が正本）
- 6031 incident（[g1-1p5x-multiseed-tier2-incident.md](g1-1p5x-multiseed-tier2-incident.md) が正本、
  `UNRESOLVED / DORMANT / WATCH`）
- seed 3 の registry role（`exploratory`。品質判定には入らないが、runtime の
  paired ratio 比較には平等に含む）
- QNN node / tensor 差、ORT / RSS、checkpoint payload 実測、checkpoint timestamp
  （いずれも `NOT_MEASURED`）