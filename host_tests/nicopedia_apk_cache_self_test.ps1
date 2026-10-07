# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Fault tests exercise orchestration with mocked ADB; no device is touched.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../scripts/nicopedia_runner_common.ps1')
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../build/reports'))
[IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
$fixture = Join-Path $fixtureRoot ('apk-cache-fixture-' + [guid]::NewGuid().ToString('N') + '.apk')
[IO.File]::WriteAllBytes($fixture, [byte[]](1..100))
$fixtureHash = (Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant()
function Invoke-PhoneLmAdb {
    param($Adb, $Device, [string[]]$Arguments, $TimeoutSeconds = 60, [switch]$AllowFailure)
    $command = $Arguments -join ' '
    $result = @{ ExitCode = 0; Classification = 'SUCCESS'; Text = ''; Output = @() }
    if ($command -match '^shell stat ') {
        if ($script:case -eq 'unknown') {
            $result.ExitCode = 1; $result.Classification = 'ADB_COMMAND_FAILURE'; $result.Text = 'Permission denied'
        } elseif ($script:case -eq 'miss' -and -not $script:uploaded) {
            $result.ExitCode = 1; $result.Classification = 'ADB_COMMAND_FAILURE'; $result.Text = 'No such file or directory'
        } else { $result.Text = if ($script:case -eq 'truncated') { '99' } else { '100' } }
    } elseif ($command -match '^shell sha256sum ') {
        $hashValue = if ($script:case -eq 'corrupt') { 'f' * 64 } else { $fixtureHash }
        $result.Text = $hashValue + '  fixture.apk'
    } elseif ($Arguments[0] -eq 'push') {
        if ($TimeoutSeconds -ne 300) { throw 'push deadline changed' }
        $script:uploaded = $true
    } elseif ($command -match '^shell mv ') {
        if (-not $script:uploaded) { throw 'unverified cache publication' }
    } elseif ($command -match '^shell pm install -r ') {
        if ($script:gates -ne 2 -or $TimeoutSeconds -ne 300) { throw 'install skipped active-run/deadline gate' }
        if (($Arguments -contains '-t') -ne ($script:case -eq 'test-apk')) { throw 'test APK install flag mismatch' }
        $script:installs++; $result.Text = if ($script:case -eq 'install-failed') { 'Failure [INSTALL_FAILED_TEST]' } else { 'Success' }
    } else { throw "unexpected mock command: $command" }
    return [pscustomobject]$result
}
function Assert-PhoneLmNoExistingRun { param($Adb,$Device,$Package); $script:gates++ }
function Assert-PhoneLmNoExistingHeadlessRun { param($Adb,$Device,$Package); $script:gates++ }
function Assert-PhoneLmInstalledApkMatches { param($Adb,$Device,$Package,$LocalApk); $script:verified++ }
try {
    foreach ($script:case in @('hit','miss','test-apk','install-failed','corrupt','truncated','unknown')) {
        $script:uploaded = $false; $script:installs = 0; $script:gates = 0; $script:verified = 0
        $failure = $null
        try {
            Install-PhoneLmVerifiedCachedApk -Adb 'mock' -Device 'mock' -ActivePackage 'fixture' `
                -TargetPackage 'fixture' -LocalApk $fixture -TestApk:($script:case -eq 'test-apk')
        } catch { $failure = $_ }
        if ($script:case -in @('hit','miss','test-apk')) {
            if ($failure) { throw $failure }
            if ($script:installs -ne 1 -or $script:verified -ne 1 -or
                $script:uploaded -ne ($script:case -eq 'miss')) { throw 'cache success contract failed' }
        } elseif ($script:case -eq 'install-failed') {
            if (-not $failure -or $script:installs -ne 1 -or $script:verified -ne 0) { throw 'failed package install accepted' }
        } elseif (-not $failure -or $script:installs -ne 0 -or $script:verified -ne 0) {
            throw "bad cache accepted: $script:case"
        }
    }
    Write-Host 'nicopedia_apk_cache_self_test=PASS (hit/miss/test-apk/install-failed/corrupt/truncated/unknown)'
} finally {
    if (-not $fixture.StartsWith($fixtureRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'fixture path escaped' }
    Remove-Item -LiteralPath $fixture -Force
}
