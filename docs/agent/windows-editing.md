# Windowsでの安全な編集

Codex DesktopのWindows環境では、組み込み `apply_patch` が次のsandbox初期化エラーで失敗することがある。

```text
windows unelevated restricted-token sandbox cannot enforce split writable root sets directly
```

これはrepositoryやpatch内容の失敗ではない。同じ呼び出しを反復しない。`apply_patch.bat` がWindowsApps配下で `Access is denied` になる場合も反復しない。

その場合に限り、次の順で代替する。

1. unified diffの対象がworkspace内であり、既存ユーザー変更を上書きしないことを確認する。
2. diffを `git apply --check -` で検証する。
3. 同じdiffを `git apply --whitespace=error -` で適用する。
4. diff適用が構文上困難な場合だけ、PowerShellで一意な完全一致anchorを検証して置換する。
5. `git diff --check` と対象差分を確認する。

曖昧な正規表現置換、workspace外の編集、`Set-Content`によるファイル全体の無条件上書きは禁止する。失敗したpatchの一部だけが適用されていないか、既存の改行/encodingとユーザー差分が維持されたかも確認する。
