# ニコニコ大百科実テキスト QNN HTP 学習

> 2026-08-10追記: canonical L19 seed1/batch8をNPRTCKPTV2 resumeで
> step 8,000まで継続した結果は
> [Nicopedia L19 HTP-native long training](qnn-nicopedia-htp-long-training.md)
> を参照。この節以降は初期320-step milestoneの記録である。

## 結論

CPU referenceを保持したまま、ニコニコ大百科実テキストの学習stepの数値演算を
Qualcomm QNN HTP graphとして実行することに成功した。UTF-8 byte tokenizer
(V=256)、T=32、D=16、FFN=32、H=2のCPU pilotと同一の入力・初期parameter・
学習条件に対して、forward / cross-entropy backward / Adam updateを明示的な
QNN HTP graphで実行し、L6はseed 1/2/4、L19はseed 1で320 stepの学習がすべて
finiteで成立した。CPU referenceとの1-step parityはlogits max abs error
4.1e-4〜1.9e-3、gradient max abs error 6.8e-5〜4.4e-3で、固定tolerance
(logits 2e-2 / probability 5e-3 / gradient 3e-2)を満たした。短いtrajectory
(2/4/8 step)のparameter driftは単調に増加したが、これはFP16中間による
数値蓄積差として説明され、320 stepで品質を損なうほどではなかった。

品質はCPUと同等の範囲に収まった。L6 seed1のvalidation NLLはHTP 2.9104 /
CPU 2.9113でほぼ一致、seed2/4ではHTPが0.036〜0.073高い。L19はHTP 2.9075 /
CPU 2.8192で0.088高いが、top-1一致率は完全一致(0.3005)だった。320 stepは
CPU pilotの1,000 stepより短く、学習は未収束である。この差分は学習step数を
増やした場合に再評価する必要がある。

## 環境

- 端末: nubia Z80 Ultra (NX741J / SM8850 / Android 16 / HTP V81)
- QAIRT: 2.48.40.260702151143 (SDK root: C:\Qualcomm\AIStack\QAIRT\2.48.40.260702)
- QNN Core API 2.37.0 / HTP Backend API 5.48.0
- matching runtime, V81 Stub, unsigned V81 Skel (app-private extraction/recovery)
- 実テキストfinal test: 未開封
- 人工データfinal holdout: 未開封

## データと前処理

元データはニコニコ大百科データ 2024-11-25版(NII IDR経由)。元データは
変更・移動・削除していない。記事本文・タイトル・ID・token列・checkpoint・
raw parameter/logitは公開成果物に含めない。実機へは必要最小限のprivate
tokenized pilot input (NPRTBYTEV1, train_pilot.bin 10.7MB)だけを転送し、
app-privateディレクトリで読み、結果はaggregateのみ公開した。

同一cache・同一training order (orderSeed 20260806)・同一初期parameter
(seed位相式LCG)をCPU referenceとQNN HTPで共有した。training order hashは
L6/L19ともCPU anchorと一致した。

## 実装

既存のQNN HTP Transformer学習実装 (qnn_runtime_transformer_training_generalized.inc
等)はV/T/D/FFN/L/Hを全てパラメータ化済みであり、V256/T32へは
`QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA`モードを追加して接続した:

- NPRTBYTEV1 cache読み込み (device: app-private files dir)
- CPU reference (`tiny_lm::forwardBackwardGeneralized`) と同一batchを供給
- 1-step parity: CPU/HTP同一初期parameter・同一batchでlogits/probability/
  dlogits/gradientを比較
- short trajectory: 同一batch orderで2/4/8 stepのCPU/HTPを並走比較
- full training: HTP graph (forward/backward) + HTP Adam graphでstep実行、
  CPU referenceはlockstepで最終状態を比較
- private checkpoint (NPRTCKPTV1) をapp-privateに保存し、ホストでCPU評価

既存の人工データ経路 (GENERIC mode等) は変更していない。

## 数値結果

1-step parity (tolerance: logits 2e-2, probability 5e-3, dlogits 5e-3,
gradient 3e-2):

| seed | layers | logits max abs | probability max abs | dlogits max abs | gradient max abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 4.14e-4 | 1.13e-5 | 7.16e-6 | 6.78e-5 | true |
| 2 | 6 | 5.78e-4 | 1.13e-5 | 7.63e-6 | 2.05e-3 | true |
| 4 | 6 | 4.12e-4 | 1.12e-5 | 7.63e-6 | 8.56e-4 | true |
| 1 | 19 | 1.86e-3 | 1.50e-5 | 7.65e-6 | 4.42e-3 | true |

short trajectory (batch 8, CPU/HTP同一batch):

| step | CPU loss | HTP loss | parameter max abs |
|---:|---:|---:|---:|
| 2 | 5.500 | 5.525 | 6.06e-3 |
| 4 | 5.389 | 5.493 | 1.21e-2 |
| 8 | 5.092 | 5.394 | 2.43e-2 |

## L6 学習 (320 step, batch 8)

| seed | first loss | last loss | completed | finite | train NLL (CPU anchor) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.548 | 2.549 | 320 | true | 2.541 |
| 2 | 5.553 | 2.622 | 320 | true | - |
| 4 | 5.548 | 2.495 | 320 | true | - |

品質比較 (validation 256 chunk / development 512 chunk, CPU評価):

| seed | CPU val NLL | HTP val NLL | CPU dev NLL | HTP dev NLL |
|---:|---:|---:|---:|---:|
| 1 | 2.9113 | 2.9104 | 2.9181 | 2.9458 |
| 2 | 2.8743 | 2.9472 | 2.9122 | 2.9630 |
| 4 | 2.8716 | 2.9073 | 2.8807 | 2.9216 |

## L19 学習 (320 step, batch 8)

| metric | CPU anchor | HTP |
|---:|---:|---:|
| train NLL (last) | 2.525 (HTP) | 2.525 |
| validation NLL | 2.8192 | 2.9075 |
| development NLL | 2.8784 | 2.9238 |
| validation top-1 | 0.3005 | 0.3005 |

## 性能 (実機 nubia Z80 Ultra, SM8850, HTP V81)

| metric | L6 | L19 |
|---:|---:|---:|
| HTP init | 627-729 ms | 2,938 ms |
| graph create | 59-159 ms | 51-55 ms |
| graph finalize | 465-469 ms | 2,763-2,827 ms |
| first execute | 6.9-7.0 ms | 19.3-19.7 ms |
| training step (batch 8) | 166-190 ms | 341-483 ms |
| execute/step | 2 | 3 |

CPU reference (host, batch 1): 3.36 ms/step。スマホHTPとPCのCPUは異なる
machineであり単純な比較はできない。この小規模モデルではHTPのgraph execute
オーバーヘッドが支配的で、step時間はCPUより遅い。

## thermal

- Android thermal status: 全runで0 (normal)
- battery temperature: 34-36 ℃ (記録のみ)
- 任意温度閾値によるcooldownは導入していない

## 既存回帰

- host tests (run_host_tests.ps1): PASS
- 人工データ経路: 変更なし、既存regression維持

## 制約

- 320 stepはCPU pilotの1,000 stepより短く未収束
- HTP内部精度をFP32と断定しない。CPU/HTP差はFP16中間による蓄積と解釈
- 実機は1台のみ (SM8850/HTP V81) で母集団推論ではない
- L19のvalidation NLL差 (+0.088) はstep増加で再評価が必要

## 再現

```powershell
# 1) 実機へprivate cache転送 + HTP学習
.\scripts\run_nicopedia_htp_training.ps1 `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Seed 1 -Layers 6 -Steps 320 -BatchSize 8

# 2) 公開bundle生成
.\scripts\export_public_qnn_nicopedia_htp_results.ps1 -SelfTest
```
