param(
    [Parameter(Mandatory = $true)][string]$ApkPath,
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [string]$ReportPath = "build/reports/qnn-apk-audit-2.48.txt"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "qairt_version.ps1")
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
    -ExpectedBuildId $ExpectedBuildId
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Sha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-EntryBytes($Entry) {
    $input = $Entry.Open()
    $memory = [IO.MemoryStream]::new()
    try { $input.CopyTo($memory); return $memory.ToArray() }
    finally { $memory.Dispose(); $input.Dispose() }
}

$apk = (Resolve-Path -LiteralPath $ApkPath).Path
$sdk = (Resolve-Path -LiteralPath $QairtSdkRoot).Path
$yaml = Get-Content -LiteralPath (Join-Path $sdk "sdk.yaml") -Raw
$header = Get-Content -LiteralPath (Join-Path $sdk "include\QNN\QnnSdkBuildId.h") -Raw
$version = [regex]::Match($yaml, '(?m)^version:\s*(\S+)').Groups[1].Value
$build = [regex]::Match($yaml, '(?m)^build_id:\s*(\S+)').Groups[1].Value
$headerBuild = [regex]::Match($header, 'QNN_SDK_BUILD_ID\s+"v([^"]+)"').Groups[1].Value
if ("$version.$build" -ne $headerBuild -or $headerBuild -ne $ExpectedBuildId) {
    throw "SDK build ID mismatch: yaml=$version.$build header=$headerBuild expected=$ExpectedBuildId"
}

$required = [ordered]@{
    "lib/arm64-v8a/libQnnSystem.so" = "lib\aarch64-android\libQnnSystem.so"
    "lib/arm64-v8a/libQnnCpu.so" = "lib\aarch64-android\libQnnCpu.so"
    "lib/arm64-v8a/libQnnHtp.so" = "lib\aarch64-android\libQnnHtp.so"
    "lib/arm64-v8a/libQnnHtpPrepare.so" = "lib\aarch64-android\libQnnHtpPrepare.so"
    "lib/arm64-v8a/libQnnHtpV81Stub.so" = "lib\aarch64-android\libQnnHtpV81Stub.so"
    "assets/qnn/libQnnHtpV81Skel.so" = "lib\hexagon-v81\unsigned\libQnnHtpV81Skel.so"
}
$failures = [Collections.Generic.List[string]]::new()
$report = [Collections.Generic.List[string]]::new()
$report.Add("audit=PHONELM_QNN_APK")
$report.Add("qairt_version=$version")
$report.Add("qairt_build_id=$headerBuild")
$report.Add("target_abi=arm64-v8a")
$report.Add("htp_architecture=V81")
$report.Add("apk_sha256=$((Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant())")

$zip = [IO.Compression.ZipFile]::OpenRead($apk)
try {
    $entries = @{}; foreach ($entry in $zip.Entries) { $entries[$entry.FullName] = $entry }
    foreach ($pair in $required.GetEnumerator()) {
        $entry = $entries[$pair.Key]
        if ($null -eq $entry) { $failures.Add("missing=$($pair.Key)"); continue }
        $apkHash = Get-Sha256 (Get-EntryBytes $entry)
        $sdkHash = (Get-FileHash -LiteralPath (Join-Path $sdk $pair.Value) -Algorithm SHA256).Hash.ToLowerInvariant()
        $report.Add("file=$($pair.Key) sha256=$apkHash sdk_sha256=$sdkHash match=$($apkHash -eq $sdkHash)")
        if ($apkHash -ne $sdkHash) { $failures.Add("hash_mismatch=$($pair.Key)") }
    }

    $stubs = @($zip.Entries | Where-Object { $_.FullName -match '^lib/arm64-v8a/libQnnHtpV\d+Stub\.so$' } | ForEach-Object FullName)
    $report.Add("htp_stubs=$($stubs -join ',')")
    if ($stubs.Count -ne 1 -or $stubs[0] -ne 'lib/arm64-v8a/libQnnHtpV81Stub.so') {
        $failures.Add("unexpected_htp_stub_set=$($stubs -join ',')")
    }
    $skels = @($zip.Entries | Where-Object { $_.FullName -match 'libQnnHtpV\d+Skel\.so$' } | ForEach-Object FullName)
    if ($skels.Count -ne 1 -or $skels[0] -ne 'assets/qnn/libQnnHtpV81Skel.so') {
        $failures.Add("unexpected_htp_skel_set=$($skels -join ',')")
    }

    $scanSet = @($required.Keys) + @(
        'assets/qnn/qairt.properties',
        'lib/arm64-v8a/libphonelm_native.so'
    )
    $scanNames = @($scanSet | Sort-Object -Unique | ForEach-Object { $entries[$_] } | Where-Object { $null -ne $_ })
    $foundExpected = $false
    foreach ($entry in $scanNames) {
        $text = [Text.Encoding]::Latin1.GetString((Get-EntryBytes $entry))
        if ($text.Contains($ExpectedBuildId)) { $foundExpected = $true }
        foreach ($forbidden in @('2.47.0', '260601114230')) {
            if ($text.Contains($forbidden)) { $failures.Add("forbidden_string=$forbidden entry=$($entry.FullName)") }
        }
        foreach ($hostPath in @($sdk, $sdk.Replace('\', '/'))) {
            if ($text.Contains($hostPath)) { $failures.Add("host_sdk_path_found=$($entry.FullName)") }
        }
    }
    if (-not $foundExpected) { $failures.Add("expected_build_id_not_found=$ExpectedBuildId") }
} finally { $zip.Dispose() }

$report.Add("forbidden_2_47_strings=false")
$report.Add("host_sdk_path_present=false")
$report.Add("status=$(if ($failures.Count -eq 0) { 'SUCCESS' } else { 'FAILED' })")
foreach ($failure in $failures) { $report.Add("error=$failure") }
$destination = if ([IO.Path]::IsPathFullyQualified($ReportPath)) {
    [IO.Path]::GetFullPath($ReportPath)
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location) $ReportPath))
}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
$report | Set-Content -LiteralPath $destination -Encoding utf8
$report
if ($failures.Count -ne 0) { throw "QNN APK audit failed; see $destination" }
