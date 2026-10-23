# test_backward_formal_transport.ps1
# Host-only test harness for scripts/backward_formal_transport.ps1.
# No device calls, no adb, no network. Scratch is written only under
# build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/host-tests/
# with a GUID suffix. Total runtime target: well under 90 s.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$modulePath = Join-Path $PSScriptRoot 'backward_formal_transport.ps1'
if (-not [System.IO.File]::Exists($modulePath)) {
    throw ('module not found: {0}' -f $modulePath)
}
. $modulePath

$repoRoot = Split-Path -Parent $PSScriptRoot
$scratchRoot = Join-Path $repoRoot 'build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/host-tests'
$caseRoot = Join-Path $scratchRoot ('host-' + [guid]::NewGuid().ToString('N'))
[void][System.IO.Directory]::CreateDirectory($caseRoot)

$script:caseFailures = [System.Collections.Generic.List[string]]::new()
$script:binaryRoundtripPass = $false
$deviceCalls = 0
$totalClock = New-PhoneLmTransferClock

# ---------------------------------------------------------------------------
# Host-only fake chunk invoker with configurable fault injection.
# Contract: & $invoker $SkipBytes $CountBytes $OutputPath $TimeoutSeconds
# writes the raw bytes of the range to $OutputPath and returns
# @{ExitCode; Classification; BytesWritten}. It never returns bytes in memory.
# ---------------------------------------------------------------------------
function New-HostChunkInvoker {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Source,

        [int]$ShortWriteAttempts = 0,
        [int]$ShortBy = 1,
        [string]$ForcedClassification = '',
        [int]$ForcedClassificationFromAttempt = 0,
        [switch]$CorruptBytes,
        [double]$DelayMilliseconds = 0,
        [switch]$RemoteMissing,

        [Parameter(Mandatory = $true)]
        [hashtable]$State
    )

    $invoker = {
        param(
            [long]$SkipBytes,
            [long]$CountBytes,
            [string]$OutputPath,
            [double]$TimeoutSeconds
        )

        $State.Attempts = [int]$State.Attempts + 1
        $attemptNumber = [int]$State.Attempts

        if ($DelayMilliseconds -gt 0) {
            Start-Sleep -Milliseconds ([int][Math]::Round($DelayMilliseconds))
        }

        if ($RemoteMissing) {
            return [ordered]@{ ExitCode = 1; Classification = 'REMOTE_RAW_MISSING'; BytesWritten = [long]0 }
        }

        $forcedNow = $false
        if (-not [string]::IsNullOrEmpty($ForcedClassification)) {
            if ($ForcedClassificationFromAttempt -le 0) { $forcedNow = $true }
            elseif ($attemptNumber -ge $ForcedClassificationFromAttempt) { $forcedNow = $true }
        }
        if ($forcedNow) {
            return [ordered]@{ ExitCode = 124; Classification = $ForcedClassification; BytesWritten = [long]0 }
        }

        $length = [int]$CountBytes
        if (($SkipBytes + $CountBytes) -gt ([long]$Source.Length)) {
            throw 'HOST_INVOKER_RANGE_OUT_OF_BOUNDS'
        }
        $bytes = [byte[]]::new($length)
        if ($length -gt 0) {
            [System.Array]::Copy($Source, $SkipBytes, $bytes, 0L, $CountBytes)
        }
        if ($CorruptBytes -and $length -gt 0) {
            $bytes[0] = [byte]($bytes[0] -bxor 0xFF)
        }

        $writeCount = $length
        if ($ShortWriteAttempts -gt 0 -and $attemptNumber -le $ShortWriteAttempts) {
            $writeCount = $length - [Math]::Max(1, $ShortBy)
            if ($writeCount -lt 0) { $writeCount = 0 }
        }

        $payload = $bytes
        if ($writeCount -ne $length) {
            $payload = [byte[]]::new($writeCount)
            if ($writeCount -gt 0) {
                [System.Array]::Copy($bytes, 0, $payload, 0, $writeCount)
            }
        }
        [System.IO.File]::WriteAllBytes($OutputPath, $payload)

        return [ordered]@{ ExitCode = 0; Classification = 'OK'; BytesWritten = [long]$writeCount }
    }.GetNewClosure()

    return $invoker
}

function New-HostBinarySource {
    param(
        [int]$Length,
        [int]$Seed = 7
    )
    $bytes = [byte[]]::new($Length)
    for ($i = 0; $i -lt $Length; $i++) {
        $bytes[$i] = [byte](($i * 31 + $Seed) -band 0xFF)
    }
    return $bytes
}

function Invoke-HostCase {
    param(
        [int]$Number,
        [string]$Name,
        [scriptblock]$Body
    )
    $passed = $false
    $detail = ''
    try {
        $outcome = & $Body
        if ($outcome -is [System.Collections.IDictionary]) {
            $passed = [bool]($outcome['Ok'])
            $detail = [string]($outcome['Detail'])
        }
        else {
            $detail = 'case returned unsupported output type'
        }
    }
    catch {
        $passed = $false
        $detail = $_.Exception.Message
    }
    if ($passed) {
        Write-Host ('case={0} {1} result=PASS' -f $Number, $Name)
    }
    else {
        Write-Host ('case={0} {1} result=FAIL' -f $Number, $Name)
        Write-Host ('  detail: {0}' -f $detail)
        $script:caseFailures.Add(('case{0}_{1}' -f $Number, $Name))
    }
}

# ---------------------------------------------------------------------------
# Case 1: happy path, single chunk
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 1 -Name 'happy_path_single_chunk' -Body {
    $source = New-HostBinarySource -Length 32 -Seed 3
    $dest = Join-Path $caseRoot 'case1.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case1.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $actual = [System.IO.File]::ReadAllBytes($dest)
    $mismatch = 0
    for ($i = 0; $i -lt $source.Length; $i++) {
        if ($actual[$i] -ne $source[$i]) { $mismatch = $mismatch + 1 }
    }
    $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and $result.Complete -and `
        ($result.Chunks -eq 1) -and (@($result.Attempts).Count -eq 1) -and ($mismatch -eq 0) -and ($result.Error -eq $null)
    return @{ Ok = $ok; Detail = ('status={0} mismatch={1}' -f $result.Status, $mismatch) }
}

# ---------------------------------------------------------------------------
# Case 2: happy path, four chunks
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 2 -Name 'happy_path_four_chunks' -Body {
    $source = New-HostBinarySource -Length 16384 -Seed 11
    $dest = Join-Path $caseRoot 'case2.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case2.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $hashes = @($result.ChunkSha256)
    $hashOk = $true
    for ($c = 0; $c -lt 4; $c++) {
        $slice = [byte[]]::new(4096)
        [System.Array]::Copy($source, ($c * 4096), $slice, 0, 4096)
        $want = Get-PhoneLmTransferByteSha256 -Bytes $slice
        if ($want -ne $hashes[$c]) { $hashOk = $false }
    }
    $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($result.Chunks -eq 4) -and `
        ($hashes.Count -eq 4) -and $hashOk -and ($result.Bytes -eq 16384)
    return @{ Ok = $ok; Detail = ('status={0} chunks={1} hash_ok={2}' -f $result.Status, $result.Chunks, $hashOk) }
}

# ---------------------------------------------------------------------------
# Case 3: last chunk partial size
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 3 -Name 'last_chunk_partial_size' -Body {
    $source = New-HostBinarySource -Length 10000 -Seed 13
    $dest = Join-Path $caseRoot 'case3.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case3.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $hashes = @($result.ChunkSha256)
    $tail = [byte[]]::new(1808)
    [System.Array]::Copy($source, 8192, $tail, 0, 1808)
    $tailHash = Get-PhoneLmTransferByteSha256 -Bytes $tail
    $attempts = @($result.Attempts)
    $lastCount = $attempts[$attempts.Count - 1].CountBytes
    $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($result.Chunks -eq 3) -and `
        ($hashes.Count -eq 3) -and ($hashes[2] -eq $tailHash) -and ($result.Bytes -eq 10000) -and ($lastCount -eq 1808)
    return @{ Ok = $ok; Detail = ('status={0} chunks={1} last_count={2}' -f $result.Status, $result.Chunks, $lastCount) }
}

# ---------------------------------------------------------------------------
# Case 4: short write on attempt 1, success on attempt 2
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 4 -Name 'short_write_retry_then_success' -Body {
    $source = New-HostBinarySource -Length 8192 -Seed 17
    $dest = Join-Path $caseRoot 'case4.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -ShortWriteAttempts 1
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case4.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $attempts = @($result.Attempts)
    $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($attempts.Count -eq 3) -and `
        ($attempts[0].Reason -eq 'SHORT_WRITE') -and (-not $attempts[0].Success) -and `
        $attempts[1].Success -and $attempts[2].Success -and ($result.Bytes -eq 8192)
    return @{ Ok = $ok; Detail = ('status={0} attempts={1} first_reason={2}' -f $result.Status, $attempts.Count, $attempts[0].Reason) }
}

# ---------------------------------------------------------------------------
# Case 5: short write on every attempt -> incomplete, partial preserved
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 5 -Name 'short_write_always_incomplete' -Body {
    $source = New-HostBinarySource -Length 8192 -Seed 19
    $dest = Join-Path $caseRoot 'case5.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -ShortWriteAttempts 1000000
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case5.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $partPath = '{0}.chunk-0.part' -f $dest
    $partOk = [System.IO.File]::Exists($partPath)
    $partLength = [long]-1
    if ($partOk) { $partLength = ([System.IO.FileInfo]::new($partPath)).Length }
    $ok = (-not $result.Complete) -and ($result.Status -eq 'CHUNK_BYTE_COUNT_MISMATCH') -and `
        ($result.FailedChunkIndex -eq 0) -and $partOk -and ($partLength -eq 4095) -and `
        (-not [System.IO.File]::Exists($dest))
    return @{ Ok = $ok; Detail = ('status={0} part_len={1}' -f $result.Status, $partLength) }
}

# ---------------------------------------------------------------------------
# Case 6: timeout classification on every attempt -> transport timeout status
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 6 -Name 'timeout_classification_every_attempt' -Body {
    $source = New-HostBinarySource -Length 8192 -Seed 23
    $dest = Join-Path $caseRoot 'case6.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -ForcedClassification 'ADB_WRAPPER_TIMEOUT'
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case6.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $timeoutSet = @('ADB_WRAPPER_TIMEOUT', 'ADB_TRANSPORT_TIMEOUT')
    $attempts = @($result.Attempts)
    $ok = (-not $result.Complete) -and ($timeoutSet -contains $result.Status) -and `
        ($result.Status -eq 'ADB_WRAPPER_TIMEOUT') -and ($attempts.Count -eq 2) -and `
        ($result.LastClassification -eq 'ADB_WRAPPER_TIMEOUT')
    return @{ Ok = $ok; Detail = ('status={0} attempts={1}' -f $result.Status, $attempts.Count) }
}

# ---------------------------------------------------------------------------
# Case 7: wrong bytes but correct length -> TRANSFER_HASH_MISMATCH
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 7 -Name 'wrong_bytes_hash_mismatch' -Body {
    $source = New-HostBinarySource -Length 8192 -Seed 29
    $dest = Join-Path $caseRoot 'case7.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -CorruptBytes
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case7.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $destLength = [long]-1
    if ([System.IO.File]::Exists($dest)) { $destLength = ([System.IO.FileInfo]::new($dest)).Length }
    $ok = (-not $result.Complete) -and ($result.Status -eq 'TRANSFER_HASH_MISMATCH') -and `
        ($destLength -eq $source.Length) -and ($result.Bytes -eq $source.Length) -and `
        ($result.FailedChunkIndex -eq $null)
    return @{ Ok = $ok; Detail = ('status={0} dest_len={1}' -f $result.Status, $destLength) }
}

# ---------------------------------------------------------------------------
# Case 8: remote raw missing -> REMOTE_RAW_MISSING
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 8 -Name 'remote_raw_missing' -Body {
    $source = New-HostBinarySource -Length 4096 -Seed 31
    $dest = Join-Path $caseRoot 'case8.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -RemoteMissing
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/absent.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $ok = (-not $result.Complete) -and ($result.Status -eq 'REMOTE_RAW_MISSING') -and `
        ($result.LastClassification -eq 'REMOTE_RAW_MISSING')
    return @{ Ok = $ok; Detail = ('status={0}' -f $result.Status) }
}

# ---------------------------------------------------------------------------
# Case 9: destination exists -> DESTINATION_EXISTS (and is never overwritten)
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 9 -Name 'destination_exists_refused' -Body {
    $source = New-HostBinarySource -Length 64 -Seed 37
    $dest = Join-Path $caseRoot 'case9.bin'
    [System.IO.File]::WriteAllBytes($dest, [byte[]]::new(16))
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state
    $threw = $false
    $msg = ''
    try {
        Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
            -RemotePath '/data/local/tmp/case9.bin' -DestinationPath $dest `
            -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
            -ChunkBytes 4096 -ChunkAttempts 2 | Out-Null
    }
    catch {
        $msg = $_.Exception.Message
        $threw = ($msg -like '*DESTINATION_EXISTS*')
    }
    $kept = ([System.IO.FileInfo]::new($dest)).Length -eq 16
    $ok = $threw -and $kept -and ([int]$state.Attempts -eq 0)
    return @{ Ok = $ok; Detail = ('msg={0} kept_len_unchanged={1} invoker_calls={2}' -f $msg, $kept, [int]$state.Attempts) }
}

# ---------------------------------------------------------------------------
# Case 10: global FileTimeoutSeconds exceeded -> TRANSFER_TIMEOUT
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 10 -Name 'global_file_timeout_exceeded' -Body {
    $source = New-HostBinarySource -Length 16384 -Seed 41
    $dest = Join-Path $caseRoot 'case10.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -DelayMilliseconds 400
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case10.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2 -FileTimeoutSeconds 0.1
    $ok = (-not $result.Complete) -and ($result.Status -eq 'TRANSFER_TIMEOUT') -and `
        ($result.Bytes -lt $source.Length)
    return @{ Ok = $ok; Detail = ('status={0} bytes={1} elapsed_ms={2}' -f $result.Status, $result.Bytes, [Math]::Round($result.ElapsedMilliseconds, 1)) }
}

# ---------------------------------------------------------------------------
# Case 11: remote sha changed between before/after -> TRANSFER_REMOTE_CHANGED
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 11 -Name 'remote_changed_before_after' -Body {
    $before = [ordered]@{ Exists = $true; SizeBytes = [long]12345; Sha256 = ('a' * 64) }
    $after = [ordered]@{ Exists = $true; SizeBytes = [long]12345; Sha256 = ('f' * 64) }
    $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]12345 }
    $r = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after `
        -Received $received -ReceivedSha256 ('a' * 64)
    $ok = (-not $r.Verified) -and ($r.Classification -eq 'TRANSFER_REMOTE_CHANGED') -and (@($r.Reasons).Count -ge 1)
    return @{ Ok = $ok; Detail = ('classification={0} reasons={1}' -f $r.Classification, @($r.Reasons).Count) }
}

# ---------------------------------------------------------------------------
# Case 12: integrity verified only when all three identities agree
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 12 -Name 'integrity_three_identities' -Body {
    $before = [ordered]@{ Exists = $true; SizeBytes = [long]2048; Sha256 = ('b' * 64) }
    $after = [ordered]@{ Exists = $true; SizeBytes = [long]2048; Sha256 = ('b' * 64) }
    $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]2048 }

    $allAgree = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after `
        -Received $received -ReceivedSha256 ('b' * 64)

    $wrongHash = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after `
        -Received $received -ReceivedSha256 ('e' * 64)

    $notVerified = [ordered]@{ Status = 'TRANSFER_HASH_MISMATCH'; Bytes = [long]2048 }
    $badStatus = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after `
        -Received $notVerified -ReceivedSha256 ('b' * 64)

    $missingHash = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after `
        -Received $received -ReceivedSha256 ''

    $ok = $allAgree.Verified -and ($allAgree.Classification -eq 'TRANSFER_VERIFIED') -and `
        (-not $wrongHash.Verified) -and ($wrongHash.Classification -eq 'POST_EXIT_EVIDENCE_INCOMPLETE') -and `
        (-not $badStatus.Verified) -and ($badStatus.Classification -eq 'POST_EXIT_EVIDENCE_INCOMPLETE') -and `
        (-not $missingHash.Verified) -and ($missingHash.Classification -eq 'POST_EXIT_EVIDENCE_INCOMPLETE')
    return @{ Ok = $ok; Detail = ('agree={0} wrong_hash={1} bad_status={2} missing_hash={3}' -f `
        $allAgree.Classification, $wrongHash.Classification, $badStatus.Classification, $missingHash.Classification) }
}

# ---------------------------------------------------------------------------
# Case 13: zero-byte expected size refused
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 13 -Name 'zero_byte_expected_size_refused' -Body {
    $source = New-HostBinarySource -Length 64 -Seed 43
    $dest = Join-Path $caseRoot 'case13.bin'
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state
    $threw = $false
    $msg = ''
    try {
        Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
            -RemotePath '/data/local/tmp/case13.bin' -DestinationPath $dest `
            -ExpectedSizeBytes ([long]0) -ExpectedSha256 (Get-PhoneLmTransferByteSha256 -Bytes $source) `
            -ChunkBytes 4096 -ChunkAttempts 2 | Out-Null
    }
    catch {
        $msg = $_.Exception.Message
        $threw = ($msg -like '*EXPECTED_SIZE_REQUIRED*')
    }
    $ok = $threw -and ([int]$state.Attempts -eq 0)
    return @{ Ok = $ok; Detail = ('msg={0} invoker_calls={1}' -f $msg, [int]$state.Attempts) }
}

# ---------------------------------------------------------------------------
# Case 14: unsafe remote path refused (and the invoker is never called)
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 14 -Name 'unsafe_remote_path_refused' -Body {
    $adbProbe = @{ Calls = 0 }
    $fakeAdb = {
        param([object]$Arguments, [double]$TimeoutSeconds)
        $adbProbe.Calls = [int]$adbProbe.Calls + 1
        return [ordered]@{ ExitCode = 0; Text = ''; Classification = 'OK' }
    }.GetNewClosure()
    $blocked = 0
    $lastMsg = ''
    foreach ($bad in @('', 'data/../local/tmp/x.bin', 'has"quote.bin')) {
        $threwUnsafe = $false
        try {
            Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath $bad | Out-Null
        }
        catch {
            $lastMsg = $_.Exception.Message
            $threwUnsafe = ($lastMsg -like '*UNSAFE_REMOTE_PATH*')
        }
        if ($threwUnsafe) { $blocked = $blocked + 1 }
    }
    $adbCalls = [int]$adbProbe.Calls
    $ok = ($blocked -eq 3) -and ($adbCalls -eq 0)
    return @{ Ok = $ok; Detail = ('blocked={0} adb_calls={1} last_msg={2}' -f $blocked, $adbCalls, $lastMsg) }
}

# ---------------------------------------------------------------------------
# Case 15: binary fixture round-trips byte-identically (CR/LF/NUL/0xFF)
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 15 -Name 'binary_fixture_roundtrip' -Body {
    $fixture = New-HostBinarySource -Length 20000 -Seed 47
    # Sentinels a text pipeline would corrupt: at the head and straddling a chunk boundary.
    $fixture[0] = [byte]0x00
    $fixture[1] = [byte]0x0A
    $fixture[2] = [byte]0x0D
    $fixture[3] = [byte]0xFF
    $fixture[4094] = [byte]0x0D
    $fixture[4095] = [byte]0x0A
    $fixture[4096] = [byte]0x00
    $fixture[4097] = [byte]0xFF

    $dest = Join-Path $caseRoot 'case15.bin'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $fixture
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $fixture -State $state
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case15.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$fixture.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2
    $actual = [System.IO.File]::ReadAllBytes($dest)
    $mismatch = 0
    if ($actual.Length -ne $fixture.Length) {
        $mismatch = -1
    }
    else {
        for ($i = 0; $i -lt $fixture.Length; $i++) {
            if ($actual[$i] -ne $fixture[$i]) { $mismatch = $mismatch + 1 }
        }
    }
    $sentinels = ($actual[0] -eq 0x00) -and ($actual[1] -eq 0x0A) -and ($actual[2] -eq 0x0D) -and `
        ($actual[3] -eq 0xFF) -and ($actual[4094] -eq 0x0D) -and ($actual[4095] -eq 0x0A) -and `
        ($actual[4096] -eq 0x00) -and ($actual[4097] -eq 0xFF)
    $verified = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($mismatch -eq 0) -and $sentinels
    $script:binaryRoundtripPass = $verified
    return @{ Ok = $verified; Detail = ('status={0} mismatch={1} sentinels={2}' -f $result.Status, $mismatch, $sentinels) }
}

# ---------------------------------------------------------------------------
# Case 16: evidence jsonl has one line per attempt plus one final line
# ---------------------------------------------------------------------------
Invoke-HostCase -Number 16 -Name 'evidence_jsonl_per_attempt' -Body {
    $source = New-HostBinarySource -Length 12288 -Seed 53
    $dest = Join-Path $caseRoot 'case16.bin'
    $evPath = Join-Path $caseRoot 'case16.jsonl'
    $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
    $state = @{ Attempts = 0 }
    $invoker = New-HostChunkInvoker -Source $source -State $state -ShortWriteAttempts 1
    $result = Receive-PhoneLmEvidenceFile -ChunkInvoker $invoker `
        -RemotePath '/data/local/tmp/case16.bin' -DestinationPath $dest `
        -ExpectedSizeBytes ([long]$source.Length) -ExpectedSha256 $expected `
        -ChunkBytes 4096 -ChunkAttempts 2 -EvidencePath $evPath
    $lines = [System.IO.File]::ReadAllLines($evPath)
    $parsed = 0
    foreach ($line in $lines) {
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            try { $null = ConvertFrom-Json -InputObject $line; $parsed = $parsed + 1 } catch { }
        }
    }
    $attemptLines = 0
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $obj = ConvertFrom-Json -InputObject $line
        if ([string]($obj.Kind) -eq 'attempt') { $attemptLines = $attemptLines + 1 }
    }
    $expectedAttempts = @($result.Attempts).Count
    $expectedLines = $expectedAttempts + 1
    $lastObj = ConvertFrom-Json -InputObject $lines[$lines.Count - 1]
    $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($lines.Count -eq $expectedLines) -and `
        ($parsed -eq $lines.Count) -and ($attemptLines -eq $expectedAttempts) -and `
        ([string]($lastObj.Kind) -eq 'final') -and ([string]($lastObj.Status) -eq 'TRANSFER_VERIFIED') -and `
        ([int]($lastObj.AttemptCount) -eq $expectedAttempts)
    return @{ Ok = $ok; Detail = ('lines={0} expected={1} parsed={2} attempt_lines={3} final_status={4}' -f `
        $lines.Count, $expectedLines, $parsed, $attemptLines, [string]($lastObj.Status)) }
}

# ---------------------------------------------------------------------------
# Module self-test (pure, host-only, no device calls)
# ---------------------------------------------------------------------------
$selfTest = Test-PhoneLmTransferSelfTest -ScratchRoot $caseRoot
$selfCases = @($selfTest.Cases)
$selfFailures = @($selfTest.Failures)
if (-not $selfTest.Passed) {
    foreach ($case in $selfCases) {
        if (-not $case.Passed) {
            Write-Host ('selftest_case={0} result=FAIL detail={1}' -f $case.Name, $case.Detail)
            $script:caseFailures.Add(('selftest_{0}' -f $case.Name))
        }
    }
}
$selfPassed = $selfTest.Passed

$elapsedMs = [Math]::Round((Get-PhoneLmTransferMilliseconds -Clock $totalClock), 1)

if ($deviceCalls -ne 0) {
    $script:caseFailures.Add('device_calls_nonzero')
}

$binaryLabel = 'FAIL'
if ($script:binaryRoundtripPass) { $binaryLabel = 'PASS' }

# Run-time hash of the module under test; printed exactly once, before the final line.
$moduleSha256 = Get-PhoneLmTransferFileSha256 -Path $modulePath

if ($script:caseFailures.Count -eq 0 -and $selfPassed) {
    Write-Host ('host_tests_scratch={0}' -f $caseRoot)
    Write-Host ('host_tests_elapsed_ms={0}' -f $elapsedMs)
    Write-Host ('transport_module_sha256={0}' -f $moduleSha256)
    Write-Host ('backward_formal_transport_host=PASS cases=16 selftest_cases={0} binary_roundtrip={1} device_calls={2}' -f `
        $selfCases.Count, $binaryLabel, $deviceCalls)
    exit 0
}

foreach ($failure in $script:caseFailures) {
    Write-Host ('failure={0}' -f $failure)
}
Write-Host ('host_tests_elapsed_ms={0}' -f $elapsedMs)
Write-Host ('transport_module_sha256={0}' -f $moduleSha256)
Write-Host ('backward_formal_transport_host=FAIL cases=16 selftest_cases={0} binary_roundtrip={1} device_calls={2} failures={3} elapsed_ms={4}' -f `
    $selfCases.Count, $binaryLabel, $deviceCalls, $script:caseFailures.Count, $elapsedMs)
exit 1
