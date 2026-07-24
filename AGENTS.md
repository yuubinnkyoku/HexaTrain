必要があれば、このAGENTS.mdを自由に編集して良い。

## Windows でのファイル編集

Codex Desktop の Windows 環境では、組み込みの `apply_patch` が次の sandbox
初期化エラーで失敗することがある。

```text
windows unelevated restricted-token sandbox cannot enforce split writable root sets directly
```

このエラーはリポジトリや patch 内容の問題ではない。同じ呼び出しを何度も再試行しない。
また、`apply_patch.bat` を shell から実行すると WindowsApps 配下の実体が
`Access is denied` になることがあるため、この経路も繰り返さない。

組み込み `apply_patch` が上記理由で利用できない場合に限り、次の順で安全な代替を使う。

1. unified diff を `git apply --check -` で検証する。
2. 同じ diff を `git apply --whitespace=error -` で適用する。
3. diff 適用が構文上困難な場合だけ、PowerShell で一意な完全一致アンカーを検証して置換する。
4. 編集後に必ず `git diff --check` と対象差分を確認する。

代替編集では、対象が workspace 内にあることを確認し、既存のユーザー変更を保持する。
`Set-Content` によるファイル全体の無条件上書きや、曖昧な正規表現置換は行わない。
