# HexaTrain docs

このディレクトリには、現在の研究方針・開発規則・個別の研究記録・過去evidenceが同居している。最初から全体を検索するのではなく、目的に応じて次の入口から読む。

| 目的 | 入口 |
| --- | --- |
| 現在の研究課題・優先順位 | [`research-priorities.md`](research-priorities.md) |
| 開発基盤・リファクタリング候補 | [`engineering-backlog.md`](engineering-backlog.md) |
| AI作業規則・検証・QAIRT・実機安全条件 | [`agent/`](agent/) |
| 研究結果の生データ・公開evidence | [`results/`](results/) |
| 古い設計・履歴資料 | [`archive/`](archive/) |

## 読み方

通常の研究作業では `research-priorities.md` を起点にし、そこから必要な個別文書だけ読む。

`engineering-backlog.md` は研究課題の一覧ではない。研究中に開発基盤やコード構造が実際のボトルネックになった場合だけ参照する。

`results/` や個別の過去研究文書は再現性のため保持している。現在の実装を調べるときは、それらを最初の探索対象にせず、現在の研究方針やproduction codeから必要なevidenceへ辿る。
