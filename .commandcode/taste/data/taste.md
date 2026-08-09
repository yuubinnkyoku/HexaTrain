# Taste: データ保護と公開成果物

- プライベートデータ（raw corpus、checkpoint、tokenizer vocab等）を外部API・Webサービスへ送らず、Gitへ登録せず、公開成果物へ本文・タイトル・ID・絶対path・raw log・raw tensor等を入れない。公開できるのは件数・byte数・分布・hash・設定・aggregate指標などの集計値だけ。Confidence: 0.95
- 公開bundleはallow-list exporterを使い、manifest（データ名・版・protocol hash・設定・seed・run数・結論）を付ける。記事本文の目視確認が必要でも最小限のprivate確認に留め、tracked logや公開成果物へ残さない。Confidence: 0.85
- データ処理はstreamingで行い、全データをRAMへ一括読み込みしない。parse errorは黙って無視せず件数と原因分類を記録する。Confidence: 0.85
- 決定性を重視する：split・subset・tokenizer・hashはstable identityと固定seed・固定順で決定的に決め、実行順やOSに依存させない。Confidence: 0.85
- 回帰の期待値は理想値ではなく承認済みcanonical anchor（hash・trajectory・quality）とし、既知の低品質は「quality shortfall」として別記する。過去の結論を修正する場合は訂正注記を追加し、過去CSVは履歴として残す。Confidence: 0.8
- 実施しなかった分析・工程の空CSV・空成果物を無理に作らない。Confidence: 0.85
- 実機転送が必要な場合も必要最小限のprivate artifact（private tokenized pilot等）だけを使い、ADB serial・端末固有識別子・runtime path・raw logは公開成果物へ記録しない。Confidence: 0.9
- proprietary SDK（QAIRT等）を再配布可能とは記述しない。ライセンス・再配布条件は既存policyに従い、根拠のない配布可否の主張をしない。Confidence: 0.85
- 元データ（raw corpus等）は読み取り専用の参照として扱い、変更・移動・削除しない。実機・派生処理へ渡すのは必要最小限の派生private artifactのみ。Confidence: 0.9
