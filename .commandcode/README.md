# .commandcode

## skills

`.commandcode/skills/` は **生成物** である。管理元は `.agents/skills/`。

- skill の編集・追加は **`.agents/skills/` だけ**に行う
- 同期は `.\scripts\sync_agent_skills.ps1`
- stale 確認は `.\scripts\sync_agent_skills.ps1 -Check`
- `.commandcode/skills/` を手編集しない（次回同期で上書きされる）

## settings.json

tracked な設定には **恒久的なプロジェクト設定だけ** を置く。

置いてよい例:

- 恒久的な tool permission（例: `powershell`）
- 安定した `defaultMode` / `deny`

置いてはいけない例:

- 特定 PID / 一時 process への参照
- その場限りの `Shell(...)` 許可
- 特定 log ファイル専用の wait / poll command
- 既に終了したセッションにしか意味のない状態

セッション固有の permission が必要な場合は、そのセッション内でのみ使い、**commit しない**。

## taste/

`.commandcode/taste/` は恒久的なスタイル指針であり、生成物ではない。ここでは手編集してよい。
