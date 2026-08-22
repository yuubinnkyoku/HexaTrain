# Taste: Git運用

- 禁止操作を厳格に守る：reset / rebase / stash / amend / clean / merge / force push / 勝手なbranch切替 / checkoutによる変更破棄は行わない。Confidence: 0.95
- 予期しないtracked変更・未commit変更・未追跡ファイルは破棄せず保全し、同じファイルで安全に分離できない場合だけ中断する。ユーザー由来の変更（例: .commandcode/taste/taste.md）は変更・stage・commit・cleanしてはならない。Confidence: 0.95
- commitは論理単位の通常commitで、conventional prefix（feat/test/docs/fix/build 等 + スコープ）を好む。自分が今回変更したpathだけを明示的にstageし、開始時から存在したユーザー変更を含めない。Confidence: 0.9
- commit前にstaged状態を `git status --short` / `git diff --cached --check` / `git diff --cached --stat` で確認し、意図したpathだけがstagedされ、unrelated変更（.commandcode/taste等）が混ざっていないかを検証してからcommitする。Confidence: 0.85
- pushはfast-forwardのみ。push前に fetch して behind=0、開始HEADが現在HEADの祖先、gate/Reviewer PASS、worktree clean を確認する。push後はGitHub Actionsが完了するまで確認する。Confidence: 0.9
- 現在branch・HEAD・upstreamの差分を確認し、HEAD==origin/mainを盲目的に仮定しない。不一致時はreset/rebase/mergeせずに報告する。Confidence: 0.85
- 一時的なCI失敗は同一HEADで1回だけ再実行する。Confidence: 0.8
- タスク区切りの完了報告には git status --short / git diff --stat / git diff --check を含めて最終git状態を報告し、commit・pushはユーザーの明示指示があるまで実施しない。Confidence: 0.85
