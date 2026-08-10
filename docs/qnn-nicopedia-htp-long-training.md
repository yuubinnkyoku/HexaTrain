# Nicopedia L19 HTP-native long training

## 結論

Nicopedia byte-level L19 modelをcanonical条件（seed 1、batch 8、
learning rate 0.003）のまま、NPRTCKPTV2のAdam stateを引き継いで
step 2,000から8,000まで継続した。学習stepの数値演算をQNN HTP graphで
実行し、全6 segment、60,000 graph execute、24 interval checkpointがfinite、
QNN failure 0、CPU fallbackなしで完了した。

full-cap held-out evaluationはstep 1,000から8,000まで全boundaryで改善した。
HTP-native validation NLLは2.419449から2.168421へ、development NLLは
2.403955から2.146852へ低下した。step 8,000がvalidation/developmentとも最良で、
hard ceilingに達したため停止した。plateauはまだ確定していないが、1,000 step
あたりの改善幅は逓減している。

legacy CPU-equivalence parity gateは変更していない。step 8,000でもlegacy gateは
FAILのままで、legacy generationはBLOCKEDである。別目的のexperimental
HTP-native modeはcheckpoint/evaluation/runtime/QNN/tensor healthを独立に
fail-closedで確認し、両anchorのGreedy/Sample generationを実行した。
生成本文とpromptはprivate evidenceおよび対話報告だけに残し、公開成果物には
aggregateのみ収録する。

公開aggregateは
[`docs/results/qnn-nicopedia-htp-long-training-2026-08/`](results/qnn-nicopedia-htp-long-training-2026-08/README.md)
を参照する。Nicopedia final testと人工データfinal holdoutは未開封である。

## 固定条件

- V=256、T=32、D=16、FFN=32、H=2、L=19
- seed=1、batch size=8、learning rate=0.003
- dataset、training subset/order、byte tokenizer、Adam設定は変更なし
- QAIRT 2.48.40.260702151143
- checkpoint interval=250、NPRTCKPTV2
- validationをprimary、developmentをsecondary、final testは使用しない

## hangの原因と修正

step 2,000 checkpointが生成済みなのに旧training runnerが終了しなかった主因は、
resume後もnative codeがstep 1から完了stepまでCPU forward/backwardを無表示で
再実行していたことだった。checkpointはHTP training loop内で先に書かれるため、
学習本体完了後にrunnerが停止したように見えた。QNN graph hangを示す証拠はない。

旧full-cap evalは24,576 executeの進捗をlogcatにしか出さず、ActivityはJNI return後に
単一result fileを書き、PowerShellの`PollLimit`は0.5秒単位だった。このため
指定値7,200は約1時間で、外側1時間timeoutと競合した。さらに旧Activity lifecycleが
別結果で同じfileを上書きできた。

対策は次の通り。

- Nicopedia training/eval/generationをheadless instrumentation suiteへ移行
- run-scoped input/status/resultとatomic heartbeatを追加
- 短いpollでprocess、progress、thermal、battery、focus、ADB transportを監視
- 各ADB commandを60秒、binary pullを120秒で打ち切り、transport timeoutをfail-closed化
- training checkpointが5分間増えない場合はsegmentをstalledとして安全停止
- resume時のlegacy CPU replayを実行せず、scratch diagnosticだけに限定
- checkpointをtemporary fileからatomic renameし、interval/final write failureをfatal化
- 未完了/非有限segmentではfinal checkpointを書かず、既存canonical resume sourceを保持
- binary pullを`cmd.exe`の`/c`とcommand stringの別引数で実行し、V2 identityをdecode
- QNN success diagnosticsをaggregate化し、full-cap reportを約5.8 MBから約7 KBへ削減

## full-cap trajectory

各boundaryでvalidation 8,192 chunks（262,144 tokens）、development 16,384 chunks
（524,288 tokens）を評価した。各HTP evaluationは24,576 graph execute、failure 0、
nonfinite chunk 0、CPU fallbackなしだった。

| step | HTP val NLL | HTP dev NLL | val top-1 | dev top-1 |
|---:|---:|---:|---:|---:|
| 1,000 | 2.419449 | 2.403955 | 0.367367 | 0.366882 |
| 2,000 | 2.323977 | 2.309789 | 0.387600 | 0.385679 |
| 3,000 | 2.281335 | 2.265121 | 0.396027 | 0.394527 |
| 4,000 | 2.252030 | 2.229089 | 0.402996 | 0.402020 |
| 5,000 | 2.228838 | 2.208112 | 0.407875 | 0.405617 |
| 6,000 | 2.208364 | 2.182856 | 0.406628 | 0.407257 |
| 7,000 | 2.189731 | 2.166958 | 0.414371 | 0.415295 |
| 8,000 | **2.168421** | **2.146852** | **0.417061** | **0.416859** |

step 8,000のCPU評価はvalidation 2.168439、development 2.146876で、HTPとの差は
それぞれ-1.85e-5、-2.39e-5だった。これはlegacy prefix parity gateの合否とは
別の、held-out qualityの一致を示す。

## adaptive判断

250-step medium capは局所的に反発し、特にstep 4,000/6,000/7,000 boundaryでは
full-capと改善方向が一致しなかった。このため単発のsubsample悪化では停止せず、
1,000-step boundaryのfull-capをprimaryにした。full-capはstep 8,000まで改善したため、
早期plateau停止ではなくhard ceilingで終了した。

## generationの解釈

step 8,000のHTP-native Sampleはstep 2,000より日本語byte sequenceと文節らしさが
増え、片方の128-byte sampleはinvalid UTF-8 byte 0だった。一方、Greedyは
short-period loop fraction 1.0で、同一文字の反復が強い。held-out NLL改善は明確だが、
自由生成はまだ低品質である。次の主投資は同じL19の無期限延長よりmodel capacityの
増加が妥当で、L19延長は新しいceilingを設定した対照実験としてのみ価値がある。

## 回帰

既存headless人工データscale-formalは5 seed、3,573 QNN executeでSUCCESS、全step
finite、QNN failure 0、CPU fallbackなしだった。FFN372/L3/H4のCOUNT_FROM_ONE
seed5回帰もSUCCESS、seed5は14,088 execute、nonzero return 0、training/evaluation
finiteだった。旧host commandはdevice run中に外側timeoutへ達したが、既存processを
再起動せずterminal reportを回収した。今回EXACT_SEED側は再実行していないため、
新しいdirect-seed equivalence主張は行わない。
