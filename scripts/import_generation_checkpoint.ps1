# SPDX-License-Identifier: Apache-2.0
# Imports a fully validated research checkpoint into the app-private Generation
# store.  This script deliberately has no direct Production-store copy path:
# the device test validates staging and performs the final atomic publish.
[CmdletBinding()]
param(
    [string]$Checkpoint = '',
    [string]$TokenizerModel = '',
    [string]$QairtSdkRoot = '',
    [string]$ExpectedBuildId = '',
    [string]$ImportId = '',
    [string]$Device = '',
    [Nullable[int]]$ExpectedStep,
    [Nullable[int]]$ExpectedSeed,
    [Nullable[int]]$ExpectedV,
    [Nullable[int]]$ExpectedT,
    [Nullable[int]]$ExpectedD,
    [Nullable[int]]$ExpectedFfn,
    [Nullable[int]]$ExpectedL,
    [Nullable[int]]$ExpectedH,
    [string]$ExpectedCheckpointSha256 = '',
    [ValidateRange(30, 1800)][int]$InstrumentationTimeoutSeconds = 600,
    [switch]$RunGenerationSmoke,
    [switch]$HostOnly,
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')

# Keep the host-side FNV-1a calculation byte-for-byte identical to the native
# checkpoint loader without paying a PowerShell BigInteger allocation per byte.
if (-not ('PhoneLmGenerationImportHash' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
public static class PhoneLmGenerationImportHash {
    public static ulong Update(ulong hash, byte[] bytes) {
        unchecked {
            const ulong prime = 1099511628211UL;
            for (int i = 0; i < bytes.Length; i++) hash = (hash ^ bytes[i]) * prime;
            return hash;
        }
    }
    public static bool AllFiniteFloat32(byte[] bytes) {
        if ((bytes.Length & 3) != 0) return false;
        for (int i = 0; i < bytes.Length; i += 4) {
            if (float.IsNaN(BitConverter.ToSingle(bytes, i)) ||
                float.IsInfinity(BitConverter.ToSingle(bytes, i))) return false;
        }
        return true;
    }
}
'@
}

function Read-GenerationImportU32([byte[]]$Bytes, [int]$Offset) {
    if ($Offset -lt 0 -or $Offset + 4 -gt $Bytes.Length) { throw 'CHECKPOINT_TRUNCATED_U32' }
    # Cast before shifting: PowerShell shifts a byte as a byte and would
    # truncate values such as 0x00000100 to zero.
    return [uint32](([uint32]$Bytes[$Offset] -shl 24) -bor
        ([uint32]$Bytes[$Offset + 1] -shl 16) -bor
        ([uint32]$Bytes[$Offset + 2] -shl 8) -bor
        [uint32]$Bytes[$Offset + 3])
}
function Read-GenerationImportU64([byte[]]$Bytes, [int]$Offset) {
    if ($Offset -lt 0 -or $Offset + 8 -gt $Bytes.Length) { throw 'CHECKPOINT_TRUNCATED_U64' }
    [uint64]$value = 0
    for ($i = 0; $i -lt 8; $i++) { $value = ($value -shl 8) -bor [uint64]$Bytes[$Offset + $i] }
    return $value
}
function Read-GenerationImportBytes([byte[]]$Bytes, [ref]$Offset, [int64]$Count, [string]$Failure) {
    if ($Count -lt 0 -or $Count -gt [int64]::MaxValue -or $Offset.Value -gt $Bytes.Length - $Count) { throw $Failure }
    $result = [byte[]]::new([int]$Count)
    [Buffer]::BlockCopy($Bytes, $Offset.Value, $result, 0, [int]$Count)
    $Offset.Value += [int]$Count
    return $result
}
function Get-GenerationImportRegistry([uint32]$V, [uint32]$D, [uint32]$Ffn, [uint32]$L) {
    if ($V -lt 2 -or $D -lt 1 -or $Ffn -lt 1 -or $L -lt 1) { throw 'CHECKPOINT_CONFIG_INVALID' }
    $entries = [Collections.Generic.List[object]]::new()
    $entries.Add([pscustomobject]@{ Name = 'token_embedding'; Count = [uint64]$V * $D })
    for ($layer = 0; $layer -lt $L; $layer++) {
        $prefix = 'layer_{0:D3}.' -f $layer
        $layerSpecs = @(
            [pscustomobject]@{ Name='norm1_gamma'; Count=[uint64]$D }, [pscustomobject]@{ Name='norm1_beta'; Count=[uint64]$D },
            [pscustomobject]@{ Name='wq'; Count=[uint64]$D * $D }, [pscustomobject]@{ Name='wk'; Count=[uint64]$D * $D }, [pscustomobject]@{ Name='wv'; Count=[uint64]$D * $D }, [pscustomobject]@{ Name='wo'; Count=[uint64]$D * $D },
            [pscustomobject]@{ Name='norm2_gamma'; Count=[uint64]$D }, [pscustomobject]@{ Name='norm2_beta'; Count=[uint64]$D },
            [pscustomobject]@{ Name='ffn_w1'; Count=[uint64]$D * $Ffn }, [pscustomobject]@{ Name='ffn_w2'; Count=[uint64]$D * $Ffn })
        foreach ($item in $layerSpecs) {
            $entries.Add([pscustomobject]@{ Name = $prefix + $item.Name; Count = [uint64]$item.Count })
        }
    }
    $entries.Add([pscustomobject]@{ Name = 'output_projection'; Count = [uint64]$D * $V })
    return @($entries)
}
function Update-GenerationImportFnv([uint64]$Hash, [byte[]]$Data) {
    return [PhoneLmGenerationImportHash]::Update($Hash, $Data)
}
function Get-GenerationImportParameterHash([Collections.IEnumerable]$RegistryValues) {
    [uint64]$hash = 14695981039346656037
    foreach ($entry in $RegistryValues) {
        $hash = Update-GenerationImportFnv $hash ([Text.Encoding]::ASCII.GetBytes([string]$entry.Name))
        # nprtParameterHash hashes the in-memory uint64 element count, which
        # is little endian on supported Android/Windows targets.
        $hash = Update-GenerationImportFnv $hash ([BitConverter]::GetBytes([uint64]$entry.Count))
        $hash = Update-GenerationImportFnv $hash ([byte[]]$entry.ValueBytes)
    }
    return ('fnv1a64:{0:x16}' -f $hash)
}
function Read-GenerationImportRegistryGroup([byte[]]$Bytes, [ref]$Offset, $Registry, [switch]$CollectParameterHash) {
    $count = Read-GenerationImportU32 $Bytes $Offset.Value; $Offset.Value += 4
    if ($count -ne $Registry.Count) { throw 'CHECKPOINT_REGISTRY_COUNT_MISMATCH' }
    $values = [Collections.Generic.List[object]]::new()
    for ($i = 0; $i -lt $Registry.Count; $i++) {
        $nameLength = Read-GenerationImportU32 $Bytes $Offset.Value; $Offset.Value += 4
        if ($nameLength -lt 1 -or $nameLength -gt 256) { throw 'CHECKPOINT_REGISTRY_NAME_LENGTH_INVALID' }
        $position = [int]$Offset.Value
        $name = [Text.Encoding]::ASCII.GetString((Read-GenerationImportBytes $Bytes ([ref]$position) $nameLength 'CHECKPOINT_REGISTRY_NAME_TRUNCATED'))
        $Offset.Value = $position
        $elementCount = Read-GenerationImportU64 $Bytes $Offset.Value; $Offset.Value += 8
        if ($name -cne $Registry[$i].Name -or $elementCount -ne [uint64]$Registry[$i].Count -or $elementCount -gt 100000000) { throw "CHECKPOINT_REGISTRY_ORDER_OR_COUNT_MISMATCH:index=$i actual=$name/$elementCount expected=$($Registry[$i].Name)/$($Registry[$i].Count)" }
        $byteCount = [int64]$elementCount * 4
        $position = [int]$Offset.Value
        $raw = Read-GenerationImportBytes $Bytes ([ref]$position) $byteCount 'CHECKPOINT_REGISTRY_VALUES_TRUNCATED'
        $Offset.Value = $position
        if (-not [PhoneLmGenerationImportHash]::AllFiniteFloat32($raw)) { throw 'CHECKPOINT_NONFINITE_VALUE' }
        if ($CollectParameterHash) { $values.Add([pscustomobject]@{ Name = $name; Count = $elementCount; ValueBytes = $raw }) }
    }
    return @($values)
}
function Read-GenerationImportCheckpoint([Parameter(Mandatory)][string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'CHECKPOINT_MISSING' }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -lt 64 -or $item.Length -gt 64MB) { throw 'CHECKPOINT_SIZE_INVALID' }
    $bytes = [IO.File]::ReadAllBytes($item.FullName)
    $offset = 0
    $magic = [Text.Encoding]::ASCII.GetString((Read-GenerationImportBytes $bytes ([ref]$offset) 11 'CHECKPOINT_MAGIC_TRUNCATED'))
    if ($magic -notin @("NPRTCKPTV2`n", "NPRTCKPTV3`n")) { throw 'CHECKPOINT_FORMAT_UNSUPPORTED' }
    $v = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $t = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $d = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $ffn = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $l = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $h = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $seed = Read-GenerationImportU32 $bytes $offset; $offset += 4
    $step = Read-GenerationImportU32 $bytes $offset; $offset += 4
    # Apply the same bounded-header contract as the Android inspector before
    # constructing the registry or entering any dimension-sized loops.
    if ($v -notin @(256, 1024) -or $t -lt 1 -or $t -gt 4096 -or
        $d -lt 1 -or $d -gt 4096 -or $ffn -lt 1 -or $ffn -gt 16384 -or
        $l -lt 1 -or $l -gt 128 -or $h -lt 1 -or $h -gt 128 -or
        $d % $h -ne 0 -or $seed -lt 1 -or $seed -gt 99999 -or
        $step -lt 1 -or $step -ge 1000000) { throw 'CHECKPOINT_HEADER_INVALID' }
    $kind = ''; $tokenizerHash = ''
    if ($magic -eq "NPRTCKPTV3`n") {
        $kindLength = Read-GenerationImportU32 $bytes $offset; $offset += 4
        if ($kindLength -lt 1 -or $kindLength -gt 32) { throw 'CHECKPOINT_TOKENIZER_KIND_INVALID' }
        $kind = [Text.Encoding]::ASCII.GetString((Read-GenerationImportBytes $bytes ([ref]$offset) $kindLength 'CHECKPOINT_TOKENIZER_KIND_TRUNCATED'))
        $hashLength = Read-GenerationImportU32 $bytes $offset; $offset += 4
        if ($hashLength -ne 71) { throw 'CHECKPOINT_TOKENIZER_HASH_INVALID' }
        $tokenizerHash = [Text.Encoding]::ASCII.GetString((Read-GenerationImportBytes $bytes ([ref]$offset) $hashLength 'CHECKPOINT_TOKENIZER_HASH_TRUNCATED'))
        if ($kind -cne 'byte_bpe' -or $tokenizerHash -notmatch '^sha256:[0-9a-f]{64}$') { throw 'CHECKPOINT_TOKENIZER_IDENTITY_INVALID' }
    }
    if (($v -eq 1024) -ne ($magic -eq "NPRTCKPTV3`n")) { throw 'CHECKPOINT_FORMAT_VOCABULARY_MISMATCH' }
    $registry = Get-GenerationImportRegistry $v $d $ffn $l
    $parameters = Read-GenerationImportRegistryGroup $bytes ([ref]$offset) $registry -CollectParameterHash
    [void](Read-GenerationImportRegistryGroup $bytes ([ref]$offset) $registry)
    [void](Read-GenerationImportRegistryGroup $bytes ([ref]$offset) $registry)
    if ($offset -ne $bytes.Length) { throw 'CHECKPOINT_TRAILING_BYTES' }
    $sha = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    return [pscustomobject][ordered]@{ Path = $item.FullName; Format = $magic.Trim(); V = $v; T = $t; D = $d; Ffn = $ffn; L = $l; H = $h; Seed = $seed; Step = $step; TokenizerKind = $kind; TokenizerHash = $tokenizerHash; ParameterHash = Get-GenerationImportParameterHash $parameters; Sha256 = $sha; Size = [int64]$item.Length }
}
function Assert-GenerationImportExpectedIdentity($Identity) {
    $checks = @{ ExpectedStep = 'Step'; ExpectedSeed = 'Seed'; ExpectedV = 'V'; ExpectedT = 'T'; ExpectedD = 'D'; ExpectedFfn = 'Ffn'; ExpectedL = 'L'; ExpectedH = 'H' }
    foreach ($key in $checks.Keys) {
        $expected = Get-Variable -Name $key -ValueOnly
        if ($null -ne $expected -and [int]$Identity.($checks[$key]) -ne [int]$expected) { throw "CHECKPOINT_EXPECTED_$($checks[$key].ToUpperInvariant())_MISMATCH" }
    }
    $expectedSha = $ExpectedCheckpointSha256.ToLowerInvariant().Replace('sha256:', '')
    if ($ExpectedCheckpointSha256 -and $expectedSha -notmatch '^[0-9a-f]{64}$') { throw 'EXPECTED_CHECKPOINT_SHA256_INVALID' }
    if ($ExpectedCheckpointSha256 -and $Identity.Sha256 -cne $expectedSha) { throw 'CHECKPOINT_EXPECTED_SHA256_MISMATCH' }
}
function Get-GenerationImportId($Identity) {
    $base = 'v{0}-t{1}-d{2}-f{3}-l{4}-h{5}-seed{6}-step{7}' -f $Identity.V, $Identity.T, $Identity.D, $Identity.Ffn, $Identity.L, $Identity.H, $Identity.Seed, $Identity.Step
    $candidate = if ($ImportId) { $ImportId.ToLowerInvariant() } else { "$base-$($Identity.Sha256.Substring(0, 12))" }
    if ($candidate -notmatch '^[a-z0-9][a-z0-9-]{0,119}$') { throw 'IMPORT_ID_INVALID' }
    return $candidate
}
function Assert-GenerationImportTokenizer($Identity, [string]$ModelPath) {
    if ($Identity.V -ne 1024) {
        if ($Identity.Format -eq 'NPRTCKPTV3') { throw 'CHECKPOINT_V3_REQUIRES_V1024' }
        return ''
    }
    if ($Identity.TokenizerKind -ne 'byte_bpe') { throw 'CHECKPOINT_TOKENIZER_KIND_REQUIRED' }
    if ([string]::IsNullOrWhiteSpace($ModelPath)) { throw 'TOKENIZER_MODEL_REQUIRED_FOR_V1024' }
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw 'TOKENIZER_MODEL_MISSING' }
    $sha = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Identity.TokenizerHash -cne "sha256:$sha") { throw 'CHECKPOINT_TOKENIZER_SHA256_MISMATCH' }
    return $sha
}
function Invoke-GenerationImportSelfTest {
    # Protocol tests are device-free.  A real fixture is intentionally not
    # committed: raw checkpoints are private artifacts.
    $id = Get-GenerationImportId ([pscustomobject]@{ V=1024;T=32;D=64;Ffn=128;L=19;H=2;Seed=1;Step=8000;Sha256=('a' * 64) })
    if ($id -ne 'v1024-t32-d64-f128-l19-h2-seed1-step8000-aaaaaaaaaaaa') { throw 'SELFTEST_DETERMINISTIC_ID_FAILED' }
    $script:ImportId = 'bad/id'
    try { [void](Get-GenerationImportId ([pscustomobject]@{ V=2;T=1;D=1;Ffn=1;L=1;H=1;Seed=1;Step=1;Sha256=('b' * 64) })); throw 'SELFTEST_BAD_ID_ACCEPTED' } catch { if ($_.Exception.Message -ne 'IMPORT_ID_INVALID') { throw } }
    $script:ImportId = ''
    if ((Get-GenerationImportRegistry 2 1 1 1).Count -ne 12) { throw 'SELFTEST_REGISTRY_FAILED' }
    try { Read-GenerationImportCheckpoint -Path (Join-Path $env:TEMP 'does-not-exist.ckpt') | Out-Null; throw 'SELFTEST_MISSING_CHECKPOINT_ACCEPTED' } catch { if ($_.Exception.Message -ne 'CHECKPOINT_MISSING') { throw } }
    function Add-Be32($list, [uint32]$value) { foreach ($shift in @(24, 16, 8, 0)) { $list.Add([byte](($value -shr $shift) -band 255)) } }
    function Add-Be64($list, [uint64]$value) { foreach ($shift in @(56, 48, 40, 32, 24, 16, 8, 0)) { $list.Add([byte](($value -shr $shift) -band 255)) } }
    $fixture = Join-Path ([IO.Path]::GetTempPath()) ('phonelm-import-selftest-' + [guid]::NewGuid().ToString('N') + '.ckpt')
    try {
        # A small V2 checkpoint verifies the complete three-registry parser;
        # the real importer accepts arbitrary valid dimensions, not this shape.
        $wire = [Collections.Generic.List[byte]]::new()
        $wire.AddRange([Text.Encoding]::ASCII.GetBytes("NPRTCKPTV2`n"))
        foreach ($number in @(256, 1, 2, 2, 1, 1, 7, 9)) { Add-Be32 $wire ([uint32]$number) }
        $registry = Get-GenerationImportRegistry 256 2 2 1
        foreach ($group in 1..3) {
            Add-Be32 $wire ([uint32]$registry.Count)
            foreach ($entry in $registry) {
                $nameBytes = [Text.Encoding]::ASCII.GetBytes($entry.Name); Add-Be32 $wire ([uint32]$nameBytes.Length); $wire.AddRange($nameBytes)
                Add-Be64 $wire ([uint64]$entry.Count)
                for ($index = 0; $index -lt [int]$entry.Count * 4; $index++) { $wire.Add(0) }
            }
        }
        [IO.File]::WriteAllBytes($fixture, $wire.ToArray())
        $parsed = Read-GenerationImportCheckpoint -Path $fixture
        if ($parsed.Format -ne 'NPRTCKPTV2' -or $parsed.ParameterHash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw 'SELFTEST_VALID_CHECKPOINT_REJECTED' }
        # A header-bound tokenizer is never inferred from a filename.
        $tokenizerFixture = "$fixture.tokenizer"
        [IO.File]::WriteAllBytes($tokenizerFixture, [byte[]](1, 2, 3))
        $v3Identity = [pscustomobject]@{ V=1024; Format='NPRTCKPTV3'; TokenizerKind='byte_bpe'; TokenizerHash=('sha256:' + ('0' * 64)) }
        try { Assert-GenerationImportTokenizer $v3Identity $tokenizerFixture | Out-Null; throw 'SELFTEST_TOKENIZER_MISMATCH_ACCEPTED' } catch { if ($_.Exception.Message -ne 'CHECKPOINT_TOKENIZER_SHA256_MISMATCH') { throw } }
        $truncated = [IO.File]::ReadAllBytes($fixture)[0..([IO.File]::ReadAllBytes($fixture).Length - 2)]
        [IO.File]::WriteAllBytes($fixture, $truncated)
        try { Read-GenerationImportCheckpoint -Path $fixture | Out-Null; throw 'SELFTEST_TRUNCATED_CHECKPOINT_ACCEPTED' } catch { if ($_.Exception.Message -notmatch '^CHECKPOINT_') { throw } }
    } finally { Remove-Item -LiteralPath $fixture -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath "$fixture.tokenizer" -Force -ErrorAction SilentlyContinue }
    Write-Output 'generation_checkpoint_import_self_test=PASS'
}

if ($SelfTest) { Invoke-GenerationImportSelfTest; exit 0 }
if ($HostOnly) {
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) { throw 'Checkpoint_REQUIRED' }
    $hostIdentity = Read-GenerationImportCheckpoint -Path $Checkpoint
    Assert-GenerationImportExpectedIdentity $hostIdentity
    $hostTokenizerSha = Assert-GenerationImportTokenizer $hostIdentity $TokenizerModel
    Write-Output "status=HOST_VALIDATED`nformat=$($hostIdentity.Format)`ncheckpoint_sha256=$($hostIdentity.Sha256)`ncheckpoint_size=$($hostIdentity.Size)`nparameter_hash=$($hostIdentity.ParameterHash)`nidentity=V$($hostIdentity.V)/T$($hostIdentity.T)/D$($hostIdentity.D)/FFN$($hostIdentity.Ffn)/L$($hostIdentity.L)/H$($hostIdentity.H)/seed$($hostIdentity.Seed)/step$($hostIdentity.Step)`ntokenizer_sha256=$hostTokenizerSha"
    exit 0
}
foreach ($required in @(@{Name='Checkpoint';Value=$Checkpoint}, @{Name='QairtSdkRoot';Value=$QairtSdkRoot}, @{Name='ExpectedBuildId';Value=$ExpectedBuildId})) { if ([string]::IsNullOrWhiteSpace($required.Value)) { throw "$($required.Name)_REQUIRED" } }
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$qairtOutput = & (Join-Path $PSScriptRoot 'check_qairt.ps1') -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
if ($LASTEXITCODE -notin @(0, 3)) { throw "QAIRT_CHECK_FAILED: exit=$LASTEXITCODE" }
$identity = Read-GenerationImportCheckpoint -Path $Checkpoint
Assert-GenerationImportExpectedIdentity $identity
$tokenizerSha = Assert-GenerationImportTokenizer $identity $TokenizerModel
$resolvedId = Get-GenerationImportId $identity
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }
$package = 'com.yuubinnkyoku.phonelm'
$root = Split-Path -Parent $PSScriptRoot
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$testApk = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
if (-not (Test-Path -LiteralPath $apk -PathType Leaf) -or
    -not (Test-Path -LiteralPath $testApk -PathType Leaf)) {
    throw 'APK_OR_TEST_APK_MISSING'
}
$auditOutput = & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk `
    -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
# audit_qnn_apk.ps1 throws on any mismatch; unlike an external executable it
# does not own a meaningful LASTEXITCODE, so never reuse the QAIRT inventory's
# advisory exit code here.
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
if ($Device -and $Device -ne $device) { throw 'DEVICE_ENDPOINT_NOT_CANONICAL' }
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package $package -LocalApk $apk
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package "$package.test" -LocalApk $testApk
function Invoke-ImportAdb([string[]]$Arguments) { return Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments $Arguments }
function Invoke-ImportPush([string]$LocalPath, [string]$RemotePath) {
    # Invoke the executable with an argv array so local paths containing spaces
    # remain one argument (Start-Process joins unquoted arrays).
    $output = @(& $adb -s $device push $LocalPath $RemotePath 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "ADB_PUSH_FAILED: $($output -join ' ')" }
}
$staging = "files/generation-import-staging/$resolvedId.tmp"
$lockRoot = 'files/generation-import-locks'
$lockPath = "$lockRoot/$resolvedId"
$transactionNonce = [guid]::NewGuid().ToString('N')
$tmpCheckpoint = "/data/local/tmp/phonelm-import-$transactionNonce.ckpt"
$tmpTokenizer = "/data/local/tmp/phonelm-import-$transactionNonce.tokenizer"
$lockAcquired = $false
$published = $false
$script:retainImportLock = $false
try {
    Invoke-ImportAdb @('shell', 'run-as', $package, 'mkdir', '-p', $lockRoot) | Out-Null
    try {
        # mkdir without -p is the per-import compare-and-swap: an existing
        # lock belongs to another or an unresolved transaction and is never
        # removed by this invocation.
        Invoke-ImportAdb @('shell', 'run-as', $package, 'mkdir', $lockPath) | Out-Null
        $lockAcquired = $true
    } catch {
        $script:retainImportLock = $true
        throw
    }
    # A stale path is removed only after the exact per-import lock is held;
    # mkdir without -p then gives this transaction an exclusive directory.
    Invoke-ImportAdb @('shell', 'run-as', $package, 'rm', '-rf', '--', $staging) | Out-Null
    Invoke-ImportAdb @('shell', 'run-as', $package, 'mkdir', '-p', $staging) | Out-Null
    try {
        Invoke-ImportPush -LocalPath $identity.Path -RemotePath $tmpCheckpoint
        Invoke-ImportAdb @('shell', 'run-as', $package, 'cp', $tmpCheckpoint, "$staging/model.ckpt") | Out-Null
        if ($identity.V -eq 1024) {
            Invoke-ImportPush -LocalPath (Get-Item -LiteralPath $TokenizerModel).FullName -RemotePath $tmpTokenizer
            Invoke-ImportAdb @('shell', 'run-as', $package, 'cp', $tmpTokenizer, "$staging/byte-bpe-v1024.model") | Out-Null
        }
    } catch {
        # This exact import staging directory is disposable; Production and
        # other staging entries are never touched by transfer failure cleanup.
        if ($_.Exception.Message -match 'ADB_|TRANSPORT|transport') {
            $script:retainImportLock = $true
        } else {
            Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @(
                'shell', 'run-as', $package, 'rm', '-rf', '--', $staging
            ) -AllowFailure | Out-Null
        }
        throw
    } finally {
        Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @(
            'shell', 'rm', '-f', '--', $tmpCheckpoint, $tmpTokenizer
        ) -AllowFailure | Out-Null
    }
$runner = "$package.test/androidx.test.runner.AndroidJUnitRunner"
function Invoke-GenerationImportInstrumentation([ValidateSet('publish', 'smoke')][string]$Operation) {
    Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
    Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
    $runId = [guid]::NewGuid().ToString('N')
    $operationResultPath = "files/generation-import-results/$resolvedId-$runId.txt"
    Invoke-ImportAdb @('shell', 'run-as', $package, 'rm', '-f', '--', $operationResultPath) | Out-Null
    $instrumentArgs = @(
        '-s', $device, 'shell', 'am', 'instrument', '-w', '-r',
        '-e', 'class', "$package.GenerationCheckpointImportDeviceTest",
        '-e', 'importId', $resolvedId,
        '-e', 'runId', $runId,
        '-e', 'expectedCheckpointSha256', $identity.Sha256,
        '-e', 'expectedCheckpointSizeBytes', [string]$identity.Size,
        '-e', 'expectedV', [string]$identity.V,
        '-e', 'expectedT', [string]$identity.T,
        '-e', 'expectedD', [string]$identity.D,
        '-e', 'expectedFfn', [string]$identity.Ffn,
        '-e', 'expectedL', [string]$identity.L,
        '-e', 'expectedH', [string]$identity.H,
        '-e', 'expectedStep', [string]$identity.Step,
        '-e', 'expectedSeed', [string]$identity.Seed,
        '-e', 'expectedTokenizerKind', $(if ($identity.V -eq 1024) { $identity.TokenizerKind } else { 'byte' }),
        '-e', 'expectedParameterHash', $identity.ParameterHash,
        '-e', 'operation', $Operation
    )
    if ($identity.V -eq 1024) {
        $instrumentArgs += @('-e', 'expectedTokenizerHash', $identity.TokenizerHash)
    }
    $instrumentArgs += $runner

    $evidenceDirectory = Join-Path $root "build\generation-import\$resolvedId\$Operation"
    [IO.Directory]::CreateDirectory($evidenceDirectory) | Out-Null
    $process = $null
    $instrumentationExited = $false
    try {
        $process = Start-Process -FilePath $adb -ArgumentList $instrumentArgs `
            -RedirectStandardOutput (Join-Path $evidenceDirectory 'instrumentation-stdout.txt') `
            -RedirectStandardError (Join-Path $evidenceDirectory 'instrumentation-stderr.txt') `
            -PassThru -WindowStyle Hidden
        $focusTakeovers = 0
        $deadline = [DateTime]::UtcNow.AddSeconds($InstrumentationTimeoutSeconds)
        do {
            $top = Get-PhoneLmTopPackage -Adb $adb -Device $device
            if ($top -eq 'UNKNOWN') {
                Stop-PhoneLmProcessTree -RootProcessId $process.Id
                throw 'FOCUS_STATE_UNKNOWN'
            }
            if ($top -eq $package) { $focusTakeovers++ }
            if ([DateTime]::UtcNow -ge $deadline) {
                Stop-PhoneLmProcessTree -RootProcessId $process.Id
                throw "IMPORT_INSTRUMENTATION_TIMEOUT: operation=$Operation"
            }
        } while (-not $process.WaitForExit(2000))
        $process.WaitForExit()
        $instrumentationExited = $true
        if ($focusTakeovers -ne 0) { throw "FOCUS_TAKEOVER_DETECTED: count=$focusTakeovers" }
        if ($process.ExitCode -ne 0) { throw "IMPORT_INSTRUMENTATION_FAILED: operation=$Operation exit=$($process.ExitCode)" }

        $text = (Invoke-ImportAdb @('shell', 'run-as', $package, 'cat', $operationResultPath)).Text
        $resultMap = Get-PhoneLmKeyValueMap -Text $text
        foreach ($pair in @{ status='SUCCESS'; operation=$Operation; run_id=$runId; activity_launched='false'; atomic_publish=$(if ($Operation -eq 'publish') { 'true' } else { 'false' }); atomic=$(if ($Operation -eq 'publish') { 'true' } else { 'false' }); operation_result_fresh='true'; import_id=$resolvedId; checkpoint_sha256=$identity.Sha256; checkpoint_size=[string]$identity.Size; checkpoint_format=$identity.Format; header_vocabulary=[string]$identity.V; header_tokens=[string]$identity.T; header_dimension=[string]$identity.D; header_feedforward=[string]$identity.Ffn; header_layers=[string]$identity.L; header_heads=[string]$identity.H; header_step=[string]$identity.Step; header_seed=[string]$identity.Seed; finite='true'; parameter_hash=$identity.ParameterHash; training_resume='false'; compatibility='compatible'; visible='true' }.GetEnumerator()) {
            if (-not $resultMap.Contains($pair.Key) -or $resultMap[$pair.Key] -cne $pair.Value) {
                throw "IMPORT_DEVICE_RESULT_MISMATCH:$Operation/$($pair.Key)"
            }
        }
        if ($identity.V -eq 1024 -and (-not $resultMap.Contains('tokenizer_sha256') -or $resultMap.tokenizer_sha256 -cne $tokenizerSha)) {
            throw "IMPORT_DEVICE_RESULT_MISMATCH:$Operation/tokenizer_sha256"
        }
        if ($Operation -eq 'smoke') {
            foreach ($pair in @{ smoke_mode='greedy'; smoke_max_new_bytes='8'; htp='true'; qnn_return_code_success='true'; output_tensors_finite='true'; cpu_fallback='false'; qnn_execute_failures='0' }.GetEnumerator()) {
                if (-not $resultMap.Contains($pair.Key) -or $resultMap[$pair.Key] -cne $pair.Value) {
                    throw "IMPORT_SMOKE_RESULT_MISMATCH:$($pair.Key)"
                }
            }
        }
        return $resultMap
    } catch {
        # If the host-side adb client did not observe a clean instrumentation
        # exit, the device may still own the test process.  Keep the lock and
        # staged bytes so a later invocation cannot race that unknown state.
        if (-not $instrumentationExited) { $script:retainImportLock = $true }
        throw
    }
}
$publishResult = Invoke-GenerationImportInstrumentation -Operation publish
$published = $true
$smokeResult = if ($RunGenerationSmoke) { Invoke-GenerationImportInstrumentation -Operation smoke } else { $null }
Write-Output "status=SUCCESS`nimport_id=$resolvedId`ncheckpoint_sha256=$($identity.Sha256)`nparameter_hash=$($identity.ParameterHash)`natomic_publish=$($publishResult.atomic_publish)`nidempotent=$($publishResult.idempotent)`ngeneration_smoke=$([bool]$smokeResult)"
} catch {
    if ($_.Exception.Message -match 'ADB_TRANSPORT_(FAILURE|TIMEOUT)|ADB_PUSH_FAILED') {
        $script:retainImportLock = $true
    }
    throw
} finally {
    if ($lockAcquired -and -not $published -and -not $script:retainImportLock) {
        Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @(
            'shell', 'run-as', $package, 'rm', '-rf', '--', $staging
        ) -AllowFailure | Out-Null
    }
    if ($lockAcquired -and -not $script:retainImportLock) {
        Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @(
            'shell', 'run-as', $package, 'rm', '-rf', '--', $lockPath
        ) -AllowFailure | Out-Null
    } elseif ($lockAcquired) {
        Write-Warning 'Import lock retained because device state was not proven quiescent; reconcile before retrying.'
    }
}
