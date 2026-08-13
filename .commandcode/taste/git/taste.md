# Taste: Git運用

- 禁止操作を厳格に守る：reset / rebase / stash / amend / clean / merge / force push / 勝手なbranch切替 / checkoutによる変更破棄は行わない。Confidence: 0.95
- 予期しないtracked変更・未commit変更・未追跡ファイルは破棄せず保全し、同じファイルで安全に分離できない場合だけ中断する。ユーザー由来の変更（例: .commandcode/taste/taste.md）は変更・stage・commit・cleanしてはならない。Confidence: 0.95
- commitは論理単位の通常commitで、conventional prefix（feat/test/docs/fix/build 等 + スコープ）を好む。自分が今回変更したpathだけを明示的にstageし、開始時から存在したユーザー変更を含めない。Confidence: 0.9
- pushはfast-forwardのみ。push前に fetch して behind=0、開始HEADが現在HEADの祖先、gate/Reviewer PASS、worktree clean を確認する。push後はGitHub Actionsが完了するまで確認する。Confidence: 0.9
- 現在branch・HEAD・upstreamの差分を確認し、HEAD==origin/mainを盲目的に仮定しない。不一致時はreset/rebase/mergeせずに報告する。Confidence: 0.85
- 一時的なCI失敗は同一HEADで1回だけ再実行する。Confidence: 0.8
- 実験タスクではproduction preset・UIを変更せず、実験記録（docs）と必要なrunner修正のみcommitする。検証・secret/binary監査後に通常push可。Confidence: 0.8
- 作業は既存worktreeを汚さず、並行・統合作業は新しいgit worktree（専用branch）で行う。既存の自動生成変更（.commandcode/taste等）はdiscard・commit・resetしない。Confidence: 0.85
- 複数branchの統合は専用の統合branch（例: codex/integrate-*）を作って通常mergeで行い、source branchはそのまま残す。conflictはours/theirs一括採用で済ませず、Gradle/Compose/Kotlin/AndroidX等の依存関係conflictは意味を確認して解消し、片方の機能を守るために他方を丸ごと戻さない。Confidence: 0.8
