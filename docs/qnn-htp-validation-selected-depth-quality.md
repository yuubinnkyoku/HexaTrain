# QNN HTP validation-selected depth quality

## 結論

L19 の残課題は finite かつ seed-sensitive な training-quality shortfall である。
training objective が低下し続けても、独立 validation の品質は seed 2 で途中から
悪化した。ただし、validation loss 最小の checkpoint は未使用の held-out generation
を改善しなかった。このため `BEST_VALIDATION_V1` は明示的な実験 mode として実装したが、
今回の L19 formal 候補には採用しない。既定値は従来どおり `FINAL_STEP` である。

同一の公開 milestone では current canonical identity の direct-seed equivalence を
5/5 で完了した。`COUNT_FROM_ONE` と `EXACT_SEED` は L2/H2 seed 1/2/5、L19/H2
seed 2、FFN372/L3/H4 seed 5 の各caseで、初期parameter/Adam、全step loss/accuracy、
final parameter/logitsを含む harvested field が一致した。これは seed 選択経路の
equivalence 証拠であり、`BEST_VALIDATION_V1` 候補の採否とは独立である。

## Dataset partition

TRAIN は4つの周期規則を phase 0 から開始し、CURRENT_PHASE1_EVAL は同じ規則を
phase 1 から開始する。Oracle/Free の正式testは phase 0 prefixから8 tokenを生成し、
それぞれ expected token / predicted token でcontextを更新する。

Validation schema 2 (`ROTATED_LAST_POSITION_V2`) は周期4, 4, 3の3規則だけを
phase 2から開始し、last positionの次token loss、accuracy、margin、probabilityを測る。
周期2の規則は3番目の独立phaseを作れないため除外する。これによりTRAIN、Oracle、
Freeのstatic initial caseとの完全case重複と同一prefixは0になる。学習済み規則を
評価するため、token IDとtransitionは11個共有する。Freeの後続contextは予測依存だが、
checkpoint選択にはFree結果を一切使用しない。

Validation set hash `fnv1a64:8e1411f19126879c` はschema、case ID、input、targetを
固定する決定性・corruption検査用であり、暗号学的な改ざん証明ではない。

## 選択規則とcheckpoint

評価stepは `0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320`。
全320 stepを必ず学習し、次の優先順でbest stateを1つだけ保持する。

1. validation loss最小（host float referenceに合わせた同値幅 `1e-7`）
2. 同値ならvalidation accuracy最大
3. 完全同値なら早いstep

保持状態はparameters、Adam m/v、optimizer next step、validation metricsである。
codecはconfig hash、seed、selection mode、validation schema/hash、parameter registry
version/hash、state checksumを照合し、truncation、corruption、identity mismatchを拒否する。
正式runnerのtransport再開単位は従来どおりseed atomicであり、このbest-only fileを
mid-seed resume stateとは扱わない。formal中のmemoryはcurrent stateとbest stateだけで、
全checkpointは保持しない。

## CPU screening

L19 seed 1/2/4とL18 seed 1/2を320 stepで測定した。L19 seed 1と4、L18 seed 1は
step 320を選んだ。L19 seed 2はstep 128を選んだが、Oracle/Freeはselectedとfinalの
双方が2/4で改善しなかった。L18 seed 2もstep 128を選び、Oracle/Freeがfinalの3/4から
2/4へ悪化した。

全L19 checkpointを合わせると、validation loss rankとpost-hoc Oracle/Free exact rankの
Spearman相関は約0.807だった。ただし小標本でtieが多く、最小loss checkpointがformal
testを改善するという採用条件は満たさない。相関を統計的有意性として解釈しない。

したがって事前gateに従い、HTP smokeとL19 5-seed formalは
`NOT_RUN_CPU_GATE_REJECTED` と分類して開始しない。この棄却は
HTPのgeneralization能力、hardware depth limit、内部精度を意味しない。

## Legacyと公開範囲

`FINAL_STEP` はvalidation datasetを生成せず、乱数消費、parameter registry、optimizer、
正式reportのlegacy field順序、step 320 stateを変更しない。raw checkpoint、parameters、
Adam state、tensor/logits dump、APK情報、端末識別子、絶対path、logcat、QAIRT配布物は
公開しない。allow-list bundleは
[`docs/results/qnn-htp-validation-selected-depth-quality-2026-08`](results/qnn-htp-validation-selected-depth-quality-2026-08/README.md)
に置く。

取得済みdevice回帰では、L6/H8 seed 1とL19/H2 seed 1がそれぞれcanonical anchorの
`seed_1_*` 2248/2248 fieldに一致した。したがって数値回帰は双方ともPASSである。
品質はL6/H8がOracle/Free 4/4、L19がcanonicalどおりOracle/Free 2/4だった。
L19 reportのterminal `FAILED`は4/4の研究用品質基準を満たさない
`FINITE_QUALITY_SHORTFALL`を表し、numeric/device regressionではない。

Live Updateはprogressのforeground/background更新、foreground service継続、完了時の
ongoing true→false遷移がPASSした。一方、notification tapからActivityへ戻る動作と
auto-cancelはFAILのままであり、`FAIL_DEFERRED`の独立したAndroid UI不具合として公開する。
このdeferred UI issueはdirect-seed equivalenceやCPU gateによるvalidation候補棄却を
blockしない。

## 次の候補

取得済みtrajectoryからpatience 2/3/4をシミュレーションしたが、selection一致が
seed横断で安定しないため実early stoppingは実装していない。次のmilestoneでは、
新しいtestを追加で消費せずにvalidation objectiveとautoregressive rollout品質の関係を
再設計する必要がある。
