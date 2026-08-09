# Taste: 実行ワークフローと自律性

- ユーザーへの途中確認は原則不要とし、安全境界・計算予算・明示scopeの範囲内で調査→実装→検証→commit→push→CI確認まで自律的に完了することを期待する（「ユーザーは就寝中なので確認待ちで停止しない」と明示されることも多い）。Confidence: 0.95
- サブエージェントは「人数を増やすこと自体を目的にしない」方針で、独立した観点が必要な場面（開始時のread-only監査、tokenizer/評価設計、文書作成Writer、最終の独立Reviewer）だけを適度に使う。tracked file編集・Git操作・正式実験・commit/pushは親エージェントだけが行う。Confidence: 0.95
- 主要な実験は実行直前にprivate/ignored領域へdecision（現在の証拠、残存仮説、選択理由、予測、判定条件、負の対照、予算）を事前登録し、結果を見た後に判定条件を変更しない。やむを得ず変更する場合も、元protocolを保存したまま理由を明記したamendmentとして行う。Confidence: 0.9
- 引き継ぎ・handoff作業では、報告された作業内容をローカル実ファイルとgit diffで実地確認してから進め、前エージェントの未commit実装は破棄・巻き戻しせず監査の上で完成させる。既存実装の一括書き換えは十分な理由がある場合のみ。Confidence: 0.9
- 高コストな実行（実機HTP、本番学習など）の前にCPU/smoke段階のgateを置き、CPU証拠で不適格な候補を高コスト段階へ進めない。Confidence: 0.95
- 既存のcanonical証拠・checkpoint・cache・trajectoryをidentity一致のまま再利用することを好み、同一identityの結果を理由なく再生成しない。Confidence: 0.95
- 実行予算（training run数、実験サイクル数、交換介入数など）を明示的に固定し、予算を使い切っても結論が出ない場合は「未解決」として停止する（無制限に探索を拡大しない）。Confidence: 0.9
- 時間を無駄にする実行を避ける：既に観測済みのequivalence再実行、高コストなseed再現モード、固定温度閾値での待機などは行わない。バッテリー温度は記録のみ。Confidence: 0.9
- 検証は -Wall -Wextra -Wpedantic でwarning 0を要求し、CIは実データでなく決定的な小さいfixture＋source hash/private aggregate hash/public bundle hashの組合せで検証する（固定CSVだけを無条件に信用しない）。PowerShell変更はparserで構文確認する。Confidence: 0.85
- scope外の作業（UI修正など）を理由にcommit・push・CI・数値研究を停止しない。Confidence: 0.9
- エラー時はreturn codeだけで原因を断定せず、層別（host setup / runtime / device / graph構築 / finalize / execute / tensor I-O / nonfinite / process lifecycle / thermal）に分類し、共有runtime・Skel・path条件を確認する。複数runnerで同じfailureが起きても、それだけを独立証拠としない。Confidence: 0.9
- 同じfailureを理由なく繰り返さない。失敗時はまずログと最小再現を確認してから次の試行へ進む。Confidence: 0.85
- 実機関連の検証gateは、必須項目（runtime root欠落・Build ID mismatch・APK audit failure等）はfail closed、optional tool/sampleの欠落だけでfatalにしない。実機を使うマイルストーンではQAIRT/device gateを省略しない。Confidence: 0.85
- 同じ物理端末が複数transport/interfaceで見えても別端末として数えない（stable identity resolverや既存runnerを可能な範囲で再利用する）。Confidence: 0.85
- 大規模タスクでは、着手前に既存のAGENTS.md・policy/docs・実在する名称をtreeで確認して読み、既存経路・既存結果を把握してから計画する。Confidence: 0.8
- 実機/高コストなdevice実行は最小規模から段階的に進める（shape検証→1-step parity→短期trajectory→full run→seed拡張→最大depth）。最初から最大config（例: L19 320 step）を実行しない。中間gateで説明不能な乖離があれば次段階へ進めず、段階の順序変更は理由を事前登録してから行う。Confidence: 0.9
- SDK/APIに存在しない機能・precision controlを想像で実装しない。ヘッダ・docs・サンプルで実在を確認してから使用する。Confidence: 0.85
- 固定SDK/QAIRT versionをsingle source of truthとして維持し、タスク中のversion変更をしない（version固定スクリプト等を尊重する）。Confidence: 0.75
