# Wq/Wk Head-wise Muon 2000-step A/B

**Status:** matched continuation complete on CPU/reference  
**Branch:** `experiment/wqwk-headwise-muon-500`  
**Canonical parent:** `46bbb616a949668ad8e7d7e39548ee606d887f8b`  
**Date:** 2026-09  
**Scope:** seed1 step2000→4000 continuation, Wq/Wk Muon partition only. No Wv split, no 8000, no seed2, no formal promotion.

## parent checkpoint

```text
build/reports/hvx-promotion/quality-hvx-seed1-step8000/
  htp-seed1-l19-t32-d64-f128-step2000.ckpt
NPRTCKPTV4
DataCursor recordIndex=16000, orderSeed=20260806
dataset fnv1a64:0c7b2826f5f26fea
muon_lr=0.005, momentum=0.95, Nesterov, NS5
```

Control / Candidate は **step2000 の同一 checkpoint bytes** から再分岐
（500-step run の step2500 からは継続していない）。

## A/B identity

| | Control | Candidate |
| --- | --- | --- |
| backend | CPU/reference | CPU/reference |
| Wq/Wk Muon | full `[64,64]` | head-wise `[64,32] × 2` |
| Wv / Wo / FFN | full-matrix | 変更なし |
| Aux Adam / LR / NS5 / batch | 同一 | 同一 |

model: V1024 / T32 / D64 / FFN128 / L19 / H2, attention_gate=none

## initial identity

| item | value |
| --- | --- |
| parameter hash | `e8470da4bfde83ed` (both) |
| optimizer state hash | `874b751d34c38b03` (both) |
| DataCursor | recordIndex=16000 / orderSeed=20260806 |
| 1-step exact parity | cosine=0.8250446349, relL2=0.5976642632 |

step2500 の Control 値は 500-step A/B と完全一致
（val 2.44087 / dev 2.73018 / bal 2.58553）。

## canonical metric

`bits_per_utf8_byte = Σ NLL_nats / targetUtf8Bytes / ln(2)`  
（`docs/wqwk-headwise-muon-500.md` の metric identity を正とする。bits/BPE token は不使用）

## quality trajectory (canonical bpb, 256 chunks)

| step | Control Val | Candidate Val | ΔVal | Control Dev | Candidate Dev | ΔDev | Control Bal | Candidate Bal | ΔBalanced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 2.44087 | 2.43617 | -0.0047 | 2.73018 | 2.72329 | -0.0069 | 2.58553 | 2.57973 | **-0.0058** |
| 3000 | 2.39461 | 2.41753 | +0.0229 | 2.73105 | 2.73352 | +0.0025 | 2.56283 | 2.57553 | **+0.0127** |
| 3500 | 2.38769 | 2.37919 | -0.0085 | 2.68113 | 2.66823 | -0.0129 | 2.53441 | 2.52371 | **-0.0107** |
| 4000 | 2.39355 | 2.39122 | -0.0023 | 2.63042 | 2.64021 | +0.0098 | 2.51198 | 2.51572 | **+0.0037** |

500-step 時点の canonical evidence（参考）:

```text
2050  +0.0001
2100  +0.0006
2250  -0.0024
2500  -0.0058   ← 今回と一致
```

### trajectory 読み方

- **early (2500)**: Candidate 優勢（-5.8 milli-bpb）
- **mid (3000)**: Control 偶優勢（+12.7 milli-bpb）— 最大の逆転
- **mid (3500)**: Candidate 優勢（-10.7 milli-bpb）— 最大の改善
- **final (4000)**: ほぼ tie、微小に Control 側（+3.7 milli-bpb）

改善が一貫して持続しないため、early effect の可能性が残る。
一方で 3500 でも明確な改善が出ており、単調消失でもない。

## training / health

| metric | control | candidate |
| --- | --- | --- |
| train loss @4000 | 4.096 | 4.085 |
| update/weight @4000 | 0.00320 | 0.00326 |
| non-finite | 0 | 0 |
| eval non-finite | 0 | 0 |

loss / update scale は同水準。pathological な差はない。

## head geometry (candidate, layer_000.wq)

| step | weight norm ratio h1/h0 | update norm ratio | angular h0 | angular h1 |
| --- | --- | --- | --- | --- |
| 2001 | 0.976 | 0.883 | 1.54 | 1.56 |
| 2500 | 0.965 | 0.773 | 1.61 | 1.56 |
| 3000 | 0.952 | 0.742 | 1.60 | 1.52 |
| 3500 | 0.954 | 0.711 | 1.57 | 1.58 |
| 4000 | 0.951 | 0.569 | 1.62 | 1.57 |

- head 間 weight norm は 0.95–0.98 で均衡を維持（わずかに head1 が縮む）
- update norm ratio は 0.57–1.15 で振動するが爆発なし
- angular step は両 head とも 1.45–1.7 rad
- 2000-step でも geometry は病的にならない

## runtime caveat

| arm | train_total |
| --- | --- |
| control | 2700.0 s |
| candidate | 2714.5 s |

同一 CPU/reference backend。runtime は secondary metric。
品質採否と混ぜない。前回の candidate 高速化観測も解釈しない。

## health

- non-finite: 0 / 0
- fallback: なし
- device: 不使用（CPU/reference）

## decision

```
HOLD_WQWK_HEADWISE_MUON
```

根拠:

1. 2500 / 3500 では Candidate 優勢（-5.8 / -10.7 milli-bpb）
2. しかし 3000 で逆転（+12.7）、4000 でも微小に Control 側（+3.7）
3. 「3000/3500/4000 で複数回継続した優位」を満たさない
4. 4000 で ΔBalanced < 0 も満たさない
5. ただし numerically stable、geometry 安定、REJECT に足る一貫悪化もない

500-step で見えた改善は **一時的な early / mid sample-efficiency の可能性**が残るが、
2000-step スパンでは持続が確認できない。seed2 / 8000 / formal には進まない。

## artifacts

```text
docs/results/wqwk-headwise-muon-2000-2026-09/
  quality.csv
  training-telemetry.csv
  head-geometry.csv
  runtime.csv
  run-manifest.json
  control/ candidate/
```
