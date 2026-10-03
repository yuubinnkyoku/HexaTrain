# G1 1.5x multi-seed 3000-step（seed 2 / 4）

事前登録 protocol: [../../g1-1p5x-multiseed-3000.md](../../g1-1p5x-multiseed-3000.md)
（Protocol / R1–R5 / Decision rule は同文書が正本）。

## この tree の中身

一次 evidence（device 側 native run が所有する report）:

- `seed{2,4}/{control,g1}/seed<N>-l19-v1024-t32-d64-f128-steps3000-result.txt`
  — training 1 本ぶんの health / identity。QNN return code と tensor finite は別々に含む
- `seed{2,4}/{control,g1}/eval256-step<step>-htp.txt`
  — 登録済み EvalSteps（500 / 1000 / 1500 / 1750 / 2000 / 2500 / 3000）の HTP-native 評価
- `seed{2,4}/{control,g1}/arm-identity.json` — seed / role / arm identity

- `seed{2,4}/g1/gate-static-step<step>.txt`
  — G1 の gate 診断（host tool による静的集計。R5 の入力）

analyzer 生成物（**一次データではない**。上の report から再計算できる）:

- `quality-split-level.csv` — seed × step の split-level paired（bpb / NLL / top-1 / mean rank）
- `verdicts.csv` — R1–R5 の per-seed verdict
- `run-health.csv` / `run-identity.csv` — health と identity の集計
- `gate-trajectory.csv` — gate telemetry
- `analysis.md` — 上記の markdown 表
- `run-order.json` / `seed-registry.json` / `multiseed-outcomes.json` — 実行順と seed role

raw checkpoint、logcat、ADB endpoint、絶対 path はこの tree に含まれない。

## 結果サマリ

4 arm すべて training SUCCESS、eval 28/28、`problems=0`、analyzer exit 0、
6031 再発 0。`decision: ambiguous_tie_breaker`。

**seed 3 は未実施。** exploratory evidence であり事前登録判定には入らない。

詳細は protocol 文書の「実行結果」節が正本。