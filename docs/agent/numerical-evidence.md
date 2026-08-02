# 数値・Evidence方針

## 成功と有限性

QNN return code成功とapplication-visible tensorの有限性は別々に検査し、両方の結果を報告する。有限性は入力、主要中間tensor、出力、更新後parameterの必要範囲を含める。平方、exp、divideの前後でdynamic rangeを監査する。

HTP内部精度をFP32と断定しない。CPUとの差と、同一HTP条件を繰り返したときの非決定性を区別する。clamp、epsilon増加、learning rate低下だけを根本原因修正と扱わない。

数値主張にはconfiguration、seed数、seed値または選択規則、step数、repetition数、発生数/発生率を付ける。APP_READ/APP_WRITE方向と全面書き込み、poison/未書き込み検査を維持する。

## reproducibility

同じ入力、parameter、optimizer state、Build ID、APK、runner modeをanchorにし、raw hashとcanonical hashの意味を混同しない。同一process内、process間、CPU/HTP間の比較を区別する。

## canonical regressionとquality

回帰の期待値は理想品質ではなく、承認済みcanonical anchorである。canonical hash、trajectory、有限性、QNN statusがanchorと一致すればregressionはPASSとする。

承認済みanchor自体に既知の低品質がある場合、その結果は `quality shortfall` として別記する。品質閾値によるrunnerのterminal `FAILED`を自動的にコード回帰FAILへ変換しない。

判定は少なくとも次へ分離する。

- canonical mismatch、nonfinite、QNN nonzero、欠落/破損: regression failure候補
- canonical一致かつ既知の低品質: regression PASS + `FINITE_QUALITY_SHORTFALL`
- ADB切断、process loss、timeout、別端末reattach拒否: transport/infrastructure failure

terminal statusだけで分類せず、canonical identity、trajectory、finite fields、QNN return fields、quality metricsを確認する。

## privateとpublic evidence

private reportから公開結果を作るときは `scripts/export_public_*.ps1` のallow-list exporterを使う。raw checkpoint、raw tensor、logcat、ADB endpoint、端末識別子、run ID、APK hash、ローカル絶対pathをcommitしない。公開主張とsource evidenceのseed/step/rate、Build ID、canonical identityが一致することをself-testで確認する。
