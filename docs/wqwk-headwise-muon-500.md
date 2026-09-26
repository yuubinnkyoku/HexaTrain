# Wq/Wk Head-wise Muon 500-step A/B

**Status:** matched A/B complete on CPU/reference  
**Branch:** `experiment/wqwk-headwise-muon-500`  
**Canonical parent:** `f68a3f20c37ff7cb5bf4515e1eb838fc012d076f`  
**Date:** 2026-09  
**Scope:** seed1 step2000→2500 continuation, Wq/Wk Muon partition only. No Wv split, no production optimizer change, no device use.

## parent checkpoint

```text
build/reports/hvx-promotion/quality-hvx-seed1-step8000/
  htp-seed1-l19-t32-d64-f128-step2000.ckpt
format: NPRTCKPTV4
identity: V1024 / T32 / D64 / FFN128 / L19 / H2, attention_gate=none
DataCursor: recordIndex=16000, orderSeed=20260806,
            dataset fnv1a64:0c7b2826f5f26fea
muon_lr=0.005, momentum=0.95, Nesterov, NS5
```

Control / Candidate は **同一 checkpoint bytes** から分岐。

## model identity

V1024 / T32 / D64 / FFN128 / L19 / H2  
Headwise G1: なし（parent と一致）

## backend fairness

```text
control:   CPU/reference full-matrix Original Muon
candidate: CPU/reference Wq/Wk head-wise Original Muon
```

Forward/Backward は両 arm とも `tiny_lm::forwardBackwardGeneralized`。  
Aux Adam は両 arm 同一。  
backend 差を混ぜていない。

## optimizer difference

| | Control | Candidate |
| --- | --- | --- |
| Wq | `[64,64]` × 1 Muon | `[64,32] × 2` head-wise |
| Wk | `[64,64]` × 1 Muon | `[64,32] × 2` head-wise |
| Wv | `[64,64]` × 1 Muon | 変更なし |
| Wo / FFN | full-matrix | full-matrix（同一） |
| Aux Adam | 変更なし | 変更なし |
| LR / NS5 / coeffs | 変更なし | 変更なし |

stored Wq/Wk の parameter identity は `[64,64]` のまま。  
semantic head split は optimizer update 時の column view。

## initial identity

| item | control | candidate |
| --- | --- | --- |
| parameter hash | `e8470da4bfde83ed` | `e8470da4bfde83ed` |
| optimizer state hash | `874b751d34c38b03` | `874b751d34c38b03` |
| dataset hash | `fnv1a64:0c7b2826f5f26fea` | 同一 |
| order seed | `20260806` | 同一 |
| record index | `16000` | 同一 |

## correctness

### 1-step exact-reference parity

step2000 で training path の candidate update を full と比較:

```text
cosine(full, headwise) = 0.8250446349
relative L2            = 0.5976642632
maxAbs                 = 0.001322908327
```

前回 exact reference (`docs/v41-optimizer-reference.md`) の step2000 値
`cosine 0.825 / relL2 0.598` と一致。

### non-finite / resume

- 500 step × 2 arm で non-finite = 0
- eval non-finite = 0
- 同一 checkpoint から両 arm が同一 initial hash で開始（matched）

## quality metrics

### metric identity（重要）

本 A/B の最初の評価で使った `val_bpb` 等の 6.x 台の値は、
**bits per BPE token**（token-level CE を ln(2) で割ったもの）であり、
formal baseline の **bits per original UTF-8 byte** とは別 metric だった。

| metric | 定義 | 典型値 |
| --- | --- | --- |
| bits per BPE token（旧） | `(Σ NLL_nats / tokens) / ln(2)` | 6.3–6.5 |
| **bits per UTF-8 byte（canonical）** | `(Σ NLL_nats / targetUtf8Bytes) / ln(2)` | 2.4–2.7 |

`targetUtf8Bytes = Σ bpeModel.tokenByteLength(truth_token)`。
canonical evaluator は `host_tests/htp_checkpoint_eval.cpp` および
`qnn_transformer_training.cpp` の `bits_per_utf8_byte` と同一。

変換は split ごとに**定数**（ground-truth token の byte 長のみを使うため
両 arm で同一）:

```text
Val: bytes/token = 2.603271484375  (targetBytes=21326 / tokens=8192)
Dev: bytes/token = 2.381958007485
canonical_bpb = bits_per_token / bytes_per_token
```

したがって Δ の符号は保存される（非線形変換ではない）。

### canonical quality（bits per UTF-8 byte, 256 chunks）

| step | Control Val | Candidate Val | ΔVal | Control Dev | Candidate Dev | ΔDev | Control Bal | Candidate Bal | ΔBalanced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2050 | 2.47116 | 2.47153 | +0.0004 | 2.73628 | 2.73613 | -0.0002 | 2.60372 | 2.60383 | +0.0001 |
| 2100 | 2.47265 | 2.47336 | +0.0007 | 2.73196 | 2.73245 | +0.0005 | 2.60230 | 2.60290 | +0.0006 |
| 2250 | 2.46129 | 2.45935 | -0.0019 | 2.71323 | 2.71029 | -0.0029 | 2.58726 | 2.58482 | **-0.0024** |
| 2500 | 2.44087 | 2.43617 | -0.0047 | 2.73018 | 2.72329 | -0.0069 | 2.58553 | 2.57973 | **-0.0058** |

### 旧 metric（bits per BPE token）との符号対応

| step | 旧 ΔBalanced | canonical ΔBalanced | sign |
| --- | --- | --- | --- |
| 2050 | +0.0003 | +0.0001 | MATCH |
| 2100 | +0.0015 | +0.0006 | MATCH |
| 2250 | -0.0060 | -0.0024 | MATCH |
| 2500 | -0.0143 | -0.0058 | MATCH |

- 早期 (2050/2100) はほぼ tie
- 2250 以降 Candidate が Val / Dev 両方で改善
- 2500 で ΔBalanced = **-5.8 milli-bpb**（canonical）
- final split は未使用

## training

| metric | control | candidate |
| --- | --- | --- |
| train loss @2500 | 4.347 | 4.357 |
| grad norm @2500 | 1.155 | 1.266 |
| update norm @2500 | 0.407 | 0.412 |
| update/weight @2500 | 0.00401 | 0.00406 |
| non-finite | 0 | 0 |

loss / update scale は両 arm 同水準。pathological な差はない。

## head geometry (candidate, layer_000.wq)

| step | weight norm ratio h1/h0 | update norm ratio | angular h0 | angular h1 |
| --- | --- | --- | --- | --- |
| 2001 | 0.976 | 0.883 | 1.54 | 1.56 |
| 2050 | 0.966 | 0.826 | 1.57 | 1.64 |
| 2100 | 0.969 | 0.970 | 1.61 | 1.50 |
| 2250 | 0.975 | 0.800 | 1.55 | 1.56 |
| 2500 | 0.965 | 0.773 | 1.61 | 1.56 |

- head 間 weight / update norm は均衡を保つ（ratio ≈ 0.96–0.98）
- angular step は両 head で 1.5–1.6 rad（同程度）
- exact reference で見えた update cosine 分岐は 500-step でも
  quality 悪化や norm explosion にはつながらなかった

## runtime caveat

同一 CPU/reference backend。

| arm | train_total | eval×4 |
| --- | --- | --- |
| control | 868.0 s | 約 47–52 s / 回 |
| candidate | 817.4 s | 約 47–48 s / 回 |

CPU reference であり formal HVX baseline との速度比較はしない。  
Control vs Candidate の約 6% 差は **Head-wise Muon の高速化とは解釈しない**
（実行順 / host cache / session 状態 / system load が混ざり得る）。
runtime 追試は本タスクでは行わない。

## health

- QNN: 使用なし（CPU/reference）
- non-finite: 0 / 0
- fallback: なし
- thermal: N/A（device 不使用）

## decision

```
PROMOTE_WQWK_HEADWISE_MUON_2000STEP
```

根拠（canonical bits-per-UTF-8-byte）:

1. ΔBalanced < 0 を 2250 と 2500 の複数 checkpoint で確認（-2.4 / -5.8 milli-bpb）
2. 旧 bits-per-token metric と **符号が 4 checkpoint すべて一致**
3. numerically stable（non-finite 0、update/weight 安定）
4. pathological geometry なし（head 均衡維持、angular step 同程度）
5. exact 1-step reference と training path の update が一致

次: **2000-step continuation A/B**（今回 500-step で即 8000 へは行かない）。
Wv split / seed2 / formal promotion はその後。

## artifacts

```text
docs/results/wqwk-headwise-muon-500-2026-09/
  quality.csv                  (canonical bits/UTF-8 byte)
  training-telemetry.csv
  head-geometry.csv
  runtime.csv
  run-manifest.json
  control/ candidate/          (canonical re-eval)
  token-bits-eval/             (旧 bits-per-BPE-token metric, 参考)
host_tests/wqwk_headwise_muon_500.cpp
```
