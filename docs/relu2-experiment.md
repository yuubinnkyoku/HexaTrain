# ReLU² FFN Activation A/B (Rejected)

研究結果の記録。実装コードは main に含めない。

## Identity

| 項目 | 値 |
| --- | --- |
| canonical parent | `f6c8d9f0c78fc08c53567c467402bc18afd6a4e3` |
| architecture | V1024 / T32 / D64 / FFN128 / L19 / H2 |
| attention | Headwise G1 |
| parameters | 760,960（Muon 622,592 + Aux Adam 138,368） |
| optimizer | HVX Original Muon + Aux Adam |
| seed / batch | 1 / 8 |
| schedule | LR 0.0022 linear decay 4000→8000, target 0.0001 |
| evaluation | Val / Dev のみ（final split 不使用） |
| QAIRT | 2.48.40.260702151143 |
| device | NX741J / SM8850（物理端末のみ） |

## Change

Control: `FFN activation = ReLU`（現行）

Candidate: `FFN activation = ReLU²`

```text
forward:
  z = W1 x
  r = ReLU(z)
  h = r²
  y = W2 h

backward:
  dZ = dH * 2 * ReLU(z)
```

parameter delta: **0**

## Quality（matched seed1, ΔBalanced = ReLU² − ReLU）

| step | ΔBalanced | ΔVal | ΔDev |
| ---: | ---: | ---: | ---: |
| 500 | -0.0007 | -0.0070 | +0.0055 |
| 1000 | +0.0196 | +0.0047 | +0.0344 |
| 1500 | -0.0005 | +0.0167 | -0.0177 |
| 2000 | -0.0027 | -0.0083 | +0.0029 |

Val / Dev とも方向が安定せず、2000 step でも Balanced は実質 tie。
品質改善・sample-efficiency 改善は一貫して確認できなかった。

## Runtime（実測）

| 指标 | ReLU | ReLU² |
| --- | ---: | ---: |
| training total | 818 s | 2571 s |
| fwd/bwd per update | 30.1 ms | 45.1 ms |
| gradient accumulation | 82.0 ms | 299 ms |
| Muon update wall | 93.2 ms | 177 ms |
| eval | 102 ms/chunk | 352 ms/chunk |

> **注意:** The measured runtime penalty should not be interpreted as the
> intrinsic arithmetic cost of ReLU². gradient accumulation / Muon / eval
> まで遅くなっているため、runtime measurement には ReLU² の elementwise
> square 以外の影響（graph layout、host wrapper、device state、
> 二次的な実行経路の変化）が混在している可能性がある。
> 本実験ではこの runtime 差の原因切り分けは行わない。

## Health

- QNN return code success
- output tensors finite / all steps finite
- HVX fallback = 0 / non-finite = 0
- cpu fallback = false
- parameter count 両 arm 760,960 で一致
- initial parameter hash 両 arm 一致

## Resource delta（QNN graph）

- nodes: 2803 → 2822（+19 = L19 × forward square 1）
- tensors: 3423 → 3443（+20 = L19 × ff_act + scalar）

## Decision

```text
REJECT_RELU2
```

500/1000/1500/2000 の matched A/B で品質改善も sample-efficiency 改善も
一貫せず、2000 step でも実質 tie。現行 QNN path では大きな runtime
regression も観測されたため、4000/8000 へは昇格しない。

runtime penalty は ReLU² 固有の算術コストとは断定しない。

## Reproduction notes

- matched A/B は同一 initial weights / seed / data order / recipe。
- control は resume 不能な crash があったため 0→2000 を素通しで再取得。
- checkpoint identity は activation を含む fail-closed で検証済み
  （ReLU checkpoint を ReLU² として resume 不可）。
