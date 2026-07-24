# QNN HTP full training-step result

## 概要

AndroidスマートフォンのQualcomm HTPを利用し、QAIRT 2.48のQNN graphとして2層ReLU MLPの学習処理を明示実装した実験結果です。QNNの自動微分は使用していません。学習stepの主要な数値演算を単一HTP graphで実行し、CPUはデータ供給、graph実行制御、weight buffer handoff、結果記録を担当します。

実験環境、correctness、5 seed比較、性能、weight trajectoryの公開用数値は、それぞれ [environment.json](environment.json)、[correctness.csv](correctness.csv)、[seeds.csv](seeds.csv)、[performance.csv](performance.csv)、[weight-trajectory-summary.csv](weight-trajectory-summary.csv) に収録しています。

## HTPで実行した演算

```text
Z1 = XW1
H = ReLU(Z1)
P = HW2

E = P - Y
loss = mean(E^2)
dP = E * 2/(B*O)

dW2 = H^T dP
dH = dP W2^T
dZ1 = Select(H > 0, dH, 0)
dW1 = X^T dZ1

W1_next = W1 - learning_rate * dW1
W2_next = W2 - learning_rate * dW2
```

## CPUに残る処理

```text
mini-batch供給
graph execute呼出し
step間のweight buffer切替
loss読取り
検証
ログ・レポート生成
```

## 成果

- 1 execute/step。
- graphはrunごとに1回だけcreate/finalize。
- app-owned weight bufferをping-pong bindingし、weight memcpyは0 byte/step。
- 5 seedすべてでCPU、既存fused backward、full-stepが収束。
- CPU fallback、NaN/Inf、HTP execute失敗なし。
- micro testのCPU/HTP誤差は、MSE `6.49690628e-6`、dP `1.95324421e-4`、SGD `2.14874744e-4`。
- small full-stepのCPU/HTP誤差は、prediction `4.27007675e-4`、loss `7.59959221e-5`、W1_next `6.09397888e-4`、W2_next `4.65631485e-4`。その他の公開値と実装由来の許容値は [correctness.csv](correctness.csv) を参照。
- 5 seedにおけるCPU/full-step final loss最大差は `1.09523535e-5`。

基本shape `B=8, I=128, H=128, O=64`では、既存の2 execute/step fused backward構成の3-run平均が`11060.6943 us/step`、新しいfull training-step構成が`4848.0033 us/step`で、`2.2815x`でした。

| B/I/H/O | fused backward (us/step) | full-step (us/step) | speedup |
|---|---:|---:|---:|
| 2/4/5/3 | 7413.5943 | 3917.8473 | 1.8923x |
| 8/128/128/64 | 11060.6943 | 4848.0033 | 2.2815x |
| 8/256/256/128 | 19779.1147 | 8036.1803 | 2.4613x |
| 32/256/256/128 | 19716.6147 | 7511.9617 | 2.6247x |

性能値は各runの`full_step_median_us`を3 run間で算術平均した値です。各shapeの集約値と未計測欄は [performance.csv](performance.csv) に明示しています。

## 注意事項

- CPU比較はscalar row-major C++、debug APKによるもので、端末CPUの最適上限性能を示しません。
- DVFS、温度、Androidのバックグラウンド負荷の影響を受けます。周波数固定やthermal throttling無効化は行っていません。
- 「学習stepの数値演算をHTPで実行した」結果であり、NPUだけで自律的に学習しているわけではありません。
- QAIRT runtime、Stub、Skel、APKはこのリポジトリに含みません。
- QAIRT配布物の再配布ライセンスについて、本成果は判断していません。
- raw tensor、raw weight、checksum、個別runログは公開対象外です。

再現条件と実行方法は [reproduction.md](reproduction.md)、QAIRT配布物に関する注意は [LICENSE-NOTICE.md](LICENSE-NOTICE.md) を参照してください。
## API trace evidence and A/B test

- `api-trace-evidence.txt` is an allow-list export of actual QNN API results and shared-object basenames from a successful counter-only HTP run.
- `api-trace-methodology.md` defines the evidence boundary and distinguishes QNN return codes, callback text, PhoneLM aggregate decisions, and hardware counters.
- `api-trace-ab-summary.csv` contains the public 10-run-per-condition aggregate. Raw callback text and individual runs are intentionally excluded.
