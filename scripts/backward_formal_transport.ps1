# backward_formal_transport.ps1
# Dot-sourceable module: bounded, resumable, per-chunk-verified, binary-safe evidence
# readback for post-exit formal transport. Never routes raw tensor/evidence bytes
# through a PowerShell text pipeline.
#
# Rules honoured by this file:
#   - No top-level side effects: only function definitions. No device calls, no exit.
#   - Safe under `Set-StrictMode -Version Latest`: every property is initialised before read.
#   - All decisions use a monotonic [System.Diagnostics.Stopwatch]; wall clock is annotation only.

function Get-PhoneLmTransferMember {
    <#
        StrictMode-safe member read. Hashtables/ordered dictionaries and PSCustomObjects are
        both supported; returns $Default when the member is absent or the input is $null.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [AllowNull()]
        [object]$InputObject,

        [Parameter(Mandatory = $true, Position = 1)]
        [string]$Name,

        [Parameter(Position = 2)]
        [AllowNull()]
        [object]$Default = $null
    )

    if ($null -eq $InputObject) {
        return $Default
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if ($InputObject.Contains($Name)) {
            return $InputObject[$Name]
        }
        return $Default
    }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -ne $property) {
        return $property.Value
    }
    return $Default
}

function New-PhoneLmTransferClock {
    <#
        Monotonic decision clock. Wall clock is never used for control flow.
    #>
    [CmdletBinding()]
    param()

    return [ordered]@{
        Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    }
}

function Get-PhoneLmTransferMilliseconds {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object]$Clock
    )

    $stopwatch = Get-PhoneLmTransferMember -InputObject $Clock -Name 'Stopwatch'
    if ($null -eq $stopwatch -or $stopwatch -isnot [System.Diagnostics.Stopwatch]) {
        throw 'INVALID_TRANSFER_CLOCK'
    }
    return [double]$stopwatch.Elapsed.TotalMilliseconds
}

function Get-PhoneLmTransferFileSha256 {
    <#
        Streaming SHA-256 over a file. Reads through a FileStream only; no text pipeline.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $hash = $algorithm.ComputeHash($stream)
        }
        finally {
            $stream.Dispose()
        }
    }
    finally {
        $algorithm.Dispose()
    }
    return [System.BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()
}

function Test-PhoneLmTransferHex64 {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [AllowNull()]
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    return [bool]($Value -match '^[0-9a-fA-F]{64}$')
}

function Add-PhoneLmTransferEvidenceLine {
    <#
        Appends one JSON line (UTF-8, no BOM). Any failure throws TRANSFER_EVIDENCE_WRITE_FAILED.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [object]$Payload
    )

    try {
        $directory = [System.IO.Path]::GetDirectoryName($Path)
        if (-not [string]::IsNullOrWhiteSpace($directory) -and -not [System.IO.Directory]::Exists($directory)) {
            [void][System.IO.Directory]::CreateDirectory($directory)
        }
        $json = ConvertTo-Json -InputObject $Payload -Compress -Depth 6
        [System.IO.File]::AppendAllText(
            $Path,
            $json + "`n",
            [System.Text.UTF8Encoding]::new($false)
        )
    }
    catch {
        throw [System.Exception]::new('TRANSFER_EVIDENCE_WRITE_FAILED: ' + $_.Exception.Message, $_.Exception)
    }
}

function Copy-PhoneLmTransferFileBytes {
    <#
        Binary-safe append of $SourcePath onto $DestinationPath using FileStreams (append mode).
        Never loads the whole file into memory as a string.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    $readStream = [System.IO.File]::OpenRead($SourcePath)
    try {
        $appendStream = [System.IO.File]::Open(
            $DestinationPath,
            [System.IO.FileMode]::Append,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::Read
        )
        try {
            $readStream.CopyTo($appendStream)
            $appendStream.Flush()
        }
        finally {
            $appendStream.Dispose()
        }
    }
    finally {
        $readStream.Dispose()
    }
}

function Get-PhoneLmRemoteEvidenceIdentity {
    <#
        Reads the remote evidence container identity (existence, size, sha256) through an
        injected adb wrapper. Every command is recorded with arguments, exit code,
        classification and monotonic elapsed ms.

        $AdbInvoker is invoked as: & $AdbInvoker $Arguments $TimeoutSeconds
        and must return @{ExitCode; Text; Classification}.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$AdbInvoker,

        [Parameter()]
        [AllowEmptyString()]
        [string]$RemotePath = '',

        [Parameter()]
        [double]$TimeoutSeconds = 120
    )

    if ([string]::IsNullOrWhiteSpace($RemotePath) -or $RemotePath.Contains('..') -or $RemotePath.Contains('"')) {
        throw 'UNSAFE_REMOTE_PATH'
    }
    if ($null -eq $AdbInvoker) {
        throw 'ADB_INVOKER_REQUIRED'
    }

    $clock = New-PhoneLmTransferClock
    $commands = [System.Collections.Generic.List[object]]::new()
    $exists = $false
    $sizeBytes = $null
    $sha256 = $null
    $errorText = $null

    # Step 1: existence probe.
    $arguments = @('shell', 'test', '-f', $RemotePath)
    $startedMs = Get-PhoneLmTransferMilliseconds -Clock $clock
    $probe = $null
    $probeThrew = $false
    $probeThrewMessage = ''
    try {
        $probe = & $AdbInvoker $arguments $TimeoutSeconds
    }
    catch {
        $probeThrew = $true
        $probeThrewMessage = $_.Exception.Message
    }
    $finishedMs = Get-PhoneLmTransferMilliseconds -Clock $clock

    $probeExitCode = 1
    $probeText = ''
    $probeClassification = ''
    if ($probeThrew) {
        $probeClassification = 'INVOKER_EXCEPTION'
        $probeText = $probeThrewMessage
    }
    else {
        $probeExitCode = [int](Get-PhoneLmTransferMember -InputObject $probe -Name 'ExitCode' -Default 1)
        $probeText = [string](Get-PhoneLmTransferMember -InputObject $probe -Name 'Text' -Default '')
        $probeClassification = [string](Get-PhoneLmTransferMember -InputObject $probe -Name 'Classification' -Default '')
    }
    $commands.Add([ordered]@{
        Step              = 'EXISTS'
        Arguments         = @($arguments)
        ExitCode          = $probeExitCode
        Classification    = $probeClassification
        Text              = $probeText
        ElapsedMilliseconds = [double]($finishedMs - $startedMs)
        TimestampUtc      = [DateTime]::UtcNow.ToString('o')
    })

    if ($probeExitCode -ne 0) {
        return [ordered]@{
            Exists             = $false
            SizeBytes          = $null
            Sha256             = $null
            Error              = $null
            RemotePath         = $RemotePath
            Commands           = @($commands)
            ElapsedMilliseconds = [double](Get-PhoneLmTransferMilliseconds -Clock $clock)
            TimestampUtc       = [DateTime]::UtcNow.ToString('o')
        }
    }
    $exists = $true

    # Step 2: byte size.
    $arguments = @('shell', 'stat', '-c', '%s', $RemotePath)
    $startedMs = Get-PhoneLmTransferMilliseconds -Clock $clock
    $sizeProbe = $null
    $sizeThrew = $false
    $sizeThrewMessage = ''
    try {
        $sizeProbe = & $AdbInvoker $arguments $TimeoutSeconds
    }
    catch {
        $sizeThrew = $true
        $sizeThrewMessage = $_.Exception.Message
    }
    $finishedMs = Get-PhoneLmTransferMilliseconds -Clock $clock

    $sizeExitCode = 1
    $sizeText = ''
    $sizeClassification = ''
    if ($sizeThrew) {
        $sizeClassification = 'INVOKER_EXCEPTION'
        $sizeText = $sizeThrewMessage
    }
    else {
        $sizeExitCode = [int](Get-PhoneLmTransferMember -InputObject $sizeProbe -Name 'ExitCode' -Default 1)
        $sizeText = [string](Get-PhoneLmTransferMember -InputObject $sizeProbe -Name 'Text' -Default '')
        $sizeClassification = [string](Get-PhoneLmTransferMember -InputObject $sizeProbe -Name 'Classification' -Default '')
    }
    $commands.Add([ordered]@{
        Step              = 'SIZE'
        Arguments         = @($arguments)
        ExitCode          = $sizeExitCode
        Classification    = $sizeClassification
        Text              = $sizeText
        ElapsedMilliseconds = [double]($finishedMs - $startedMs)
        TimestampUtc      = [DateTime]::UtcNow.ToString('o')
    })

    if ($sizeExitCode -ne 0) {
        $errorText = 'REMOTE_SIZE_UNREADABLE'
    }
    else {
        $sizeMatch = [regex]::Match($sizeText, '\d+')
        if ($sizeMatch.Success) {
            $parsedSize = [long]0
            if ([long]::TryParse($sizeMatch.Value, [ref]$parsedSize)) {
                $sizeBytes = $parsedSize
            }
            else {
                $errorText = 'REMOTE_SIZE_UNREADABLE'
            }
        }
        else {
            $errorText = 'REMOTE_SIZE_UNREADABLE'
        }
    }

    # Step 3: sha256 (only meaningful when the size was readable; still attempted either way).
    $arguments = @('shell', 'sha256sum', $RemotePath)
    $startedMs = Get-PhoneLmTransferMilliseconds -Clock $clock
    $hashProbe = $null
    $hashThrew = $false
    $hashThrewMessage = ''
    try {
        $hashProbe = & $AdbInvoker $arguments $TimeoutSeconds
    }
    catch {
        $hashThrew = $true
        $hashThrewMessage = $_.Exception.Message
    }
    $finishedMs = Get-PhoneLmTransferMilliseconds -Clock $clock

    $hashExitCode = 1
    $hashText = ''
    $hashClassification = ''
    if ($hashThrew) {
        $hashClassification = 'INVOKER_EXCEPTION'
        $hashText = $hashThrewMessage
    }
    else {
        $hashExitCode = [int](Get-PhoneLmTransferMember -InputObject $hashProbe -Name 'ExitCode' -Default 1)
        $hashText = [string](Get-PhoneLmTransferMember -InputObject $hashProbe -Name 'Text' -Default '')
        $hashClassification = [string](Get-PhoneLmTransferMember -InputObject $hashProbe -Name 'Classification' -Default '')
    }
    $commands.Add([ordered]@{
        Step              = 'HASH'
        Arguments         = @($arguments)
        ExitCode          = $hashExitCode
        Classification    = $hashClassification
        Text              = $hashText
        ElapsedMilliseconds = [double]($finishedMs - $startedMs)
        TimestampUtc      = [DateTime]::UtcNow.ToString('o')
    })

    if ($hashExitCode -ne 0) {
        if ([string]::IsNullOrEmpty($errorText)) {
            $errorText = 'REMOTE_HASH_UNREADABLE'
        }
    }
    else {
        $hashMatch = [regex]::Match($hashText, '(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])')
        if ($hashMatch.Success) {
            $sha256 = $hashMatch.Value.ToLowerInvariant()
        }
        else {
            if ([string]::IsNullOrEmpty($errorText)) {
                $errorText = 'REMOTE_HASH_UNREADABLE'
            }
        }
    }

    return [ordered]@{
        Exists             = $exists
        SizeBytes          = $sizeBytes
        Sha256             = $sha256
        Error              = $errorText
        RemotePath         = $RemotePath
        Commands           = @($commands)
        ElapsedMilliseconds = [double](Get-PhoneLmTransferMilliseconds -Clock $clock)
        TimestampUtc       = [DateTime]::UtcNow.ToString('o')
    }
}

function Receive-PhoneLmEvidenceFile {
    <#
        Bounded, resumable, per-chunk-verified, binary-safe readback.

        $ChunkInvoker is invoked as: & $ChunkInvoker $SkipBytes $CountBytes $OutputPath $TimeoutSeconds
        and must write the raw bytes of that range to $OutputPath (never return them in memory).
        It returns @{ExitCode; Classification; BytesWritten}.

        Preconditions throw DESTINATION_EXISTS / EXPECTED_SIZE_REQUIRED / CHUNK_BYTES_OUT_OF_RANGE.
        On any failure the partial destination and every leftover .part file are preserved and
        Complete=$false is returned. Status is one of the closed set documented below.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$ChunkInvoker,

        [Parameter()]
        [string]$RemotePath = '',

        [Parameter()]
        [string]$DestinationPath = '',

        [Parameter()]
        [long]$ExpectedSizeBytes = 0,

        [Parameter()]
        [string]$ExpectedSha256 = '',

        [Parameter()]
        [int]$ChunkBytes = 8388608,

        [Parameter()]
        [double]$ChunkTimeoutSeconds = 120,

        [Parameter()]
        [int]$ChunkAttempts = 2,

        [Parameter()]
        [double]$FileTimeoutSeconds = 1800,

        [Parameter()]
        [string]$EvidencePath = ''
    )

    # ---- preconditions (throw, never partial-write) ----
    if ([string]::IsNullOrWhiteSpace($DestinationPath)) {
        throw 'DESTINATION_PATH_REQUIRED'
    }
    if ([System.IO.File]::Exists($DestinationPath)) {
        throw 'DESTINATION_EXISTS'
    }
    if ($ExpectedSizeBytes -le 0) {
        throw 'EXPECTED_SIZE_REQUIRED'
    }
    if ($ChunkBytes -lt 4096 -or $ChunkBytes -gt 67108864) {
        throw 'CHUNK_BYTES_OUT_OF_RANGE'
    }
    if ($ChunkAttempts -lt 1) {
        throw 'CHUNK_ATTEMPTS_REQUIRED'
    }
    if ([string]::IsNullOrWhiteSpace($RemotePath)) {
        throw 'REMOTE_PATH_REQUIRED'
    }
    if ($null -eq $ChunkInvoker) {
        throw 'CHUNK_INVOKER_REQUIRED'
    }

    $clock = New-PhoneLmTransferClock
    $writeEvidence = -not [string]::IsNullOrWhiteSpace($EvidencePath)
    $attemptLog = [System.Collections.Generic.List[object]]::new()
    $chunkHashes = [System.Collections.Generic.List[string]]::new()
    $knownStatuses = @(
        'REMOTE_RAW_MISSING',
        'ADB_WRAPPER_TIMEOUT',
        'ADB_TRANSPORT_TIMEOUT',
        'ADB_TRANSPORT_DISCONNECTED'
    )
    $byteCountReasons = @('SHORT_WRITE', 'PART_MISSING', 'PART_LENGTH_MISMATCH', 'PART_HASH_FAILED')

    $sequence = 0
    $bytes = [long]0
    $chunkCount = [long][Math]::Ceiling($ExpectedSizeBytes / [double]$ChunkBytes)
    $status = 'TRANSFER_VERIFIED'
    $complete = $true
    $errorText = $null
    $failedChunkIndex = $null
    $lastClassification = $null
    $stopTransfer = $false

    for ([long]$k = 0; $k -lt $chunkCount; $k++) {
        if ($stopTransfer) {
            break
        }
        $skip = [long]$k * [long]$ChunkBytes
        $count = [Math]::Min([long]$ChunkBytes, $ExpectedSizeBytes - $skip)
        $partPath = '{0}.chunk-{1}.part' -f $DestinationPath, $k

        $attemptNumber = 0
        $chunkVerified = $false
        $lastReason = ''

        while ($attemptNumber -lt $ChunkAttempts) {
            # Global budget check happens before each attempt (monotonic clock only).
            $elapsedBefore = Get-PhoneLmTransferMilliseconds -Clock $clock
            if ($elapsedBefore -gt $FileTimeoutSeconds) {
                $status = 'TRANSFER_TIMEOUT'
                $complete = $false
                $failedChunkIndex = $k
                $errorText = 'file timeout budget of {0} ms exceeded before chunk {1} attempt {2} (elapsed {3} ms)' -f `
                    $FileTimeoutSeconds, $k, ($attemptNumber + 1), [Math]::Round($elapsedBefore, 3)
                $stopTransfer = $true
                break
            }

            $attemptNumber++
            $sequence++

            # Never reuse bytes from a previous attempt.
            [System.IO.File]::Delete($partPath)

            $attemptStartMs = Get-PhoneLmTransferMilliseconds -Clock $clock
            $invokerResult = $null
            $invokerThrew = $false
            $invokerMessage = ''
            try {
                $invokerResult = & $ChunkInvoker $skip $count $partPath $ChunkTimeoutSeconds
            }
            catch {
                $invokerThrew = $true
                $invokerMessage = $_.Exception.Message
            }
            $attemptEndMs = Get-PhoneLmTransferMilliseconds -Clock $clock

            $exitCode = -1
            $classification = ''
            $bytesWritten = [long]-1
            if ($invokerThrew) {
                $classification = 'INVOKER_EXCEPTION'
            }
            else {
                $exitCode = [int](Get-PhoneLmTransferMember -InputObject $invokerResult -Name 'ExitCode' -Default -1)
                $classification = [string](Get-PhoneLmTransferMember -InputObject $invokerResult -Name 'Classification' -Default '')
                $rawWritten = Get-PhoneLmTransferMember -InputObject $invokerResult -Name 'BytesWritten' -Default $null
                $parsedWritten = [long]0
                if ($null -ne $rawWritten -and [long]::TryParse([string]$rawWritten, [ref]$parsedWritten)) {
                    $bytesWritten = $parsedWritten
                }
            }

            $partExists = [System.IO.File]::Exists($partPath)
            $partLength = [long]-1
            if ($partExists) {
                $partLength = ([System.IO.FileInfo]::new($partPath)).Length
            }

            $reason = 'OK'
            $attemptSucceeded = $true
            $classificationStatus = ''
            if (-not [string]::IsNullOrEmpty($classification) -and ($knownStatuses -contains $classification)) {
                $classificationStatus = $classification
            }
            if ($classificationStatus -ne '') {
                $attemptSucceeded = $false
                $reason = $classificationStatus
            }
            elseif ($invokerThrew) {
                $attemptSucceeded = $false
                $reason = 'INVOKER_EXCEPTION'
            }
            elseif ($exitCode -ne 0) {
                $attemptSucceeded = $false
                $reason = 'NONZERO_EXIT'
            }
            elseif ($bytesWritten -ne $count) {
                $attemptSucceeded = $false
                $reason = 'SHORT_WRITE'
            }
            elseif (-not $partExists) {
                $attemptSucceeded = $false
                $reason = 'PART_MISSING'
            }
            elseif ($partLength -ne $count) {
                $attemptSucceeded = $false
                $reason = 'PART_LENGTH_MISMATCH'
            }

            $partSha256 = $null
            if ($attemptSucceeded) {
                try {
                    $partSha256 = Get-PhoneLmTransferFileSha256 -Path $partPath
                }
                catch {
                    $attemptSucceeded = $false
                    $reason = 'PART_HASH_FAILED'
                    $partSha256 = $null
                }
            }

            $attemptRecord = [ordered]@{
                Kind               = 'attempt'
                Sequence           = $sequence
                ChunkIndex         = $k
                Attempt            = $attemptNumber
                SkipBytes          = $skip
                CountBytes         = $count
                PartPath           = $partPath
                ExitCode           = $exitCode
                Classification     = $classification
                BytesWritten       = $bytesWritten
                PartLength         = $partLength
                Sha256             = $partSha256
                Reason             = $reason
                Success            = $attemptSucceeded
                InvokerError       = $invokerMessage
                ElapsedMilliseconds = [double]($attemptEndMs - $attemptStartMs)
                TimestampUtc       = [DateTime]::UtcNow.ToString('o')
            }
            $attemptLog.Add($attemptRecord)
            $lastClassification = $classification
            $lastReason = $reason
            if ($writeEvidence) {
                Add-PhoneLmTransferEvidenceLine -Path $EvidencePath -Payload $attemptRecord
            }

            if ($attemptSucceeded) {
                try {
                    Copy-PhoneLmTransferFileBytes -SourcePath $partPath -DestinationPath $DestinationPath
                }
                catch {
                    # Destination may be partially written; keep it as-is and stop.
                    $actualLength = [long]-1
                    if ([System.IO.File]::Exists($DestinationPath)) {
                        $actualLength = ([System.IO.FileInfo]::new($DestinationPath)).Length
                    }
                    $bytes = $actualLength
                    $status = 'TRANSFER_BYTE_COUNT_MISMATCH'
                    $complete = $false
                    $failedChunkIndex = $k
                    $errorText = 'DESTINATION_APPEND_FAILED: ' + $_.Exception.Message
                    $stopTransfer = $true
                    break
                }
                $bytes += $count
                $chunkHashes.Add($partSha256)
                $chunkVerified = $true
                break
            }
            # failed attempt -> loop for a retry while attempts remain
        }

        if ($stopTransfer) {
            break
        }
        if (-not $chunkVerified) {
            $mapped = 'ADB_TRANSPORT_DISCONNECTED'
            if (-not [string]::IsNullOrEmpty($lastClassification) -and ($knownStatuses -contains $lastClassification)) {
                $mapped = $lastClassification
            }
            elseif ($byteCountReasons -contains $lastReason) {
                $mapped = 'CHUNK_BYTE_COUNT_MISMATCH'
            }
            $status = $mapped
            $complete = $false
            $failedChunkIndex = $k
            $errorText = 'chunk {0} failed after {1} attempt(s): {2}' -f $k, $attemptNumber, $lastReason
            break
        }
    }

    # ---- final whole-file verification ----
    if ($complete) {
        $actualLength = [long]-1
        if ([System.IO.File]::Exists($DestinationPath)) {
            $actualLength = ([System.IO.FileInfo]::new($DestinationPath)).Length
        }
        if ($actualLength -ne $ExpectedSizeBytes) {
            $status = 'TRANSFER_BYTE_COUNT_MISMATCH'
            $complete = $false
            $errorText = 'expected {0} bytes, destination has {1}' -f $ExpectedSizeBytes, $actualLength
        }
        else {
            $actualSha256 = Get-PhoneLmTransferFileSha256 -Path $DestinationPath
            if (-not [string]::Equals($actualSha256, $ExpectedSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
                $status = 'TRANSFER_HASH_MISMATCH'
                $complete = $false
                $errorText = 'expected sha256 {0}, destination has {1}' -f $ExpectedSha256, $actualSha256
            }
            else {
                $status = 'TRANSFER_VERIFIED'
                $complete = $true
                $errorText = $null
            }
        }
    }

    $elapsedTotal = [double](Get-PhoneLmTransferMilliseconds -Clock $clock)
    $result = [ordered]@{
        Status             = $status
        Complete           = $complete
        RemotePath         = $RemotePath
        Bytes              = $bytes
        ExpectedBytes      = $ExpectedSizeBytes
        Chunks             = [int]$chunkCount
        ChunkSha256        = @($chunkHashes)
        Attempts           = @($attemptLog)
        ElapsedMilliseconds = $elapsedTotal
        FailedChunkIndex   = $failedChunkIndex
        LastClassification = $lastClassification
        Error              = $errorText
    }

    if ($writeEvidence) {
        Add-PhoneLmTransferEvidenceLine -Path $EvidencePath -Payload ([ordered]@{
            Kind               = 'final'
            Status             = $status
            Complete           = $complete
            RemotePath         = $RemotePath
            Bytes              = $bytes
            ExpectedBytes      = $ExpectedSizeBytes
            Chunks             = [int]$chunkCount
            AttemptCount       = $sequence
            ElapsedMilliseconds = $elapsedTotal
            FailedChunkIndex   = $failedChunkIndex
            LastClassification = $lastClassification
            Error              = $errorText
            TimestampUtc       = [DateTime]::UtcNow.ToString('o')
        })
    }

    return $result
}

function Test-PhoneLmTransferIntegrity {
    <#
        Cross-checks the three identities of a post-exit evidence transfer:
        the remote identity before the transfer, the remote identity after it,
        and the received file identity.

        Verified is true only when:
          - Before.Sha256 is 64 hex chars,
          - After.Sha256 equals Before.Sha256,
          - Received.Status is TRANSFER_VERIFIED,
          - Received.Bytes equals Before.SizeBytes,
          - the recomputed destination sha256 (-ReceivedSha256) equals Before.Sha256.
        Otherwise Classification is TRANSFER_REMOTE_CHANGED, TRANSFER_IDENTITY_UNVERIFIED
        or POST_EXIT_EVIDENCE_INCOMPLETE with Reasons listing every failing condition.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [object]$BeforeIdentity,

        [Parameter(Mandatory = $true)]
        [object]$AfterIdentity,

        [Parameter(Mandatory = $true)]
        [object]$Received,

        [Parameter()]
        [AllowNull()]
        [AllowEmptyString()]
        [string]$ReceivedSha256 = ''
    )

    $beforeHash = Get-PhoneLmTransferMember -InputObject $BeforeIdentity -Name 'Sha256' -Default ''
    $afterHash = Get-PhoneLmTransferMember -InputObject $AfterIdentity -Name 'Sha256' -Default ''
    $beforeSize = Get-PhoneLmTransferMember -InputObject $BeforeIdentity -Name 'SizeBytes' -Default $null
    $receivedStatus = Get-PhoneLmTransferMember -InputObject $Received -Name 'Status' -Default ''
    $receivedBytes = Get-PhoneLmTransferMember -InputObject $Received -Name 'Bytes' -Default $null

    $beforeHash = [string]$beforeHash
    $afterHash = [string]$afterHash
    $receivedStatus = [string]$receivedStatus

    $reasons = [System.Collections.Generic.List[string]]::new()

    $beforeHashOk = Test-PhoneLmTransferHex64 -Value $beforeHash
    $afterHashOk = Test-PhoneLmTransferHex64 -Value $afterHash
    $receivedHashOk = Test-PhoneLmTransferHex64 -Value $ReceivedSha256

    $remoteChanged = $false
    if ($beforeHashOk -and $afterHashOk) {
        if (-not [string]::Equals($beforeHash, $afterHash, [System.StringComparison]::OrdinalIgnoreCase)) {
            $remoteChanged = $true
            $reasons.Add(('REMOTE_HASH_CHANGED: after={0} before={1}' -f $afterHash, $beforeHash))
        }
    }

    if (-not $beforeHashOk) {
        $reasons.Add(('BEFORE_HASH_UNREADABLE: {0}' -f $beforeHash))
    }
    if (-not $afterHashOk) {
        $reasons.Add(('AFTER_HASH_UNREADABLE: {0}' -f $afterHash))
    }
    if ($null -eq $beforeSize -or ($beforeSize -isnot [long] -and $beforeSize -isnot [int] -and $beforeSize -isnot [int64])) {
        # Numeric size present? (string digit forms are accepted via conversion below.)
        $parsedSize = [long]0
        if ($null -eq $beforeSize -or -not [long]::TryParse([string]$beforeSize, [ref]$parsedSize)) {
            $reasons.Add('BEFORE_SIZE_UNREADABLE')
        }
    }

    $identityUnverified = $false
    if ((-not $beforeHashOk) -or (-not $afterHashOk) -or ($reasons -contains 'BEFORE_SIZE_UNREADABLE')) {
        $identityUnverified = $true
    }

    $receivedIncomplete = $false
    if ($receivedStatus -ne 'TRANSFER_VERIFIED') {
        $receivedIncomplete = $true
        $reasons.Add(('RECEIVED_STATUS_NOT_VERIFIED: {0}' -f $receivedStatus))
    }
    $parsedBeforeSize = [long]0
    $beforeSizeOk = [long]::TryParse([string]$beforeSize, [ref]$parsedBeforeSize)
    if ($beforeSizeOk) {
        $parsedReceivedBytes = [long]0
        if (-not [long]::TryParse([string]$receivedBytes, [ref]$parsedReceivedBytes) -or $parsedReceivedBytes -ne $parsedBeforeSize) {
            $receivedIncomplete = $true
            $reasons.Add(('RECEIVED_SIZE_MISMATCH: before={0} received={1}' -f $parsedBeforeSize, $receivedBytes))
        }
    }
    if (-not $receivedHashOk) {
        $receivedIncomplete = $true
        $reasons.Add(('RECEIVED_HASH_UNREADABLE: {0}' -f $ReceivedSha256))
    }
    elseif ($beforeHashOk) {
        if (-not [string]::Equals($ReceivedSha256, $beforeHash, [System.StringComparison]::OrdinalIgnoreCase)) {
            $receivedIncomplete = $true
            $reasons.Add(('RECEIVED_HASH_MISMATCH: received={0} before={1}' -f $ReceivedSha256, $beforeHash))
        }
    }

    $verified = $false
    $classification = ''
    if ($remoteChanged) {
        $classification = 'TRANSFER_REMOTE_CHANGED'
    }
    elseif ($identityUnverified) {
        $classification = 'TRANSFER_IDENTITY_UNVERIFIED'
    }
    elseif ($receivedIncomplete) {
        $classification = 'POST_EXIT_EVIDENCE_INCOMPLETE'
    }
    else {
        $verified = $true
        $classification = 'TRANSFER_VERIFIED'
    }

    return [ordered]@{
        Classification = $classification
        Verified       = $verified
        Reasons        = @($reasons)
    }
}

function Get-PhoneLmTransferByteSha256 {
    <#
        SHA-256 of an in-memory byte[] (test/self-test helper). Binary in, hex out.
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [AllowNull()]
        [AllowEmptyCollection()]
        [byte[]]$Bytes = @()
    )

    $payload = $Bytes
    if ($null -eq $payload) {
        $payload = [byte[]]::new(0)
    }
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $algorithm.ComputeHash($payload)
    }
    finally {
        $algorithm.Dispose()
    }
    return [System.BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()
}

function New-PhoneLmTransferMemoryChunkInvoker {
    <#
        Pure (host-only) chunk invoker over an in-memory byte[]. It writes the raw slice to
        $OutputPath and returns only metadata - never bytes in memory as a return value.
        Fault injection: short write, forced classification, wrong bytes, delay, missing remote.
        No device calls of any kind.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Source,

        [Parameter()]
        [int]$ShortWriteAttempts = 0,

        [Parameter()]
        [int]$ShortBy = 1,

        [Parameter()]
        [AllowEmptyString()]
        [string]$ForcedClassification = '',

        [Parameter()]
        [int]$ForcedClassificationFromAttempt = 0,

        [Parameter()]
        [switch]$CorruptBytes,

        [Parameter()]
        [double]$DelayMilliseconds = 0,

        [Parameter()]
        [switch]$RemoteMissing
    )

    $state = @{ Attempts = 0 }

    $invoker = {
        param(
            [long]$SkipBytes,
            [long]$CountBytes,
            [string]$OutputPath,
            [double]$TimeoutSeconds
        )

        $state.Attempts = [int]$state.Attempts + 1
        $attemptNumber = [int]$state.Attempts

        if ($DelayMilliseconds -gt 0) {
            Start-Sleep -Milliseconds ([int][Math]::Round($DelayMilliseconds))
        }

        if ($RemoteMissing) {
            return [ordered]@{
                ExitCode     = 1
                Classification = 'REMOTE_RAW_MISSING'
                BytesWritten = [long]0
            }
        }

        $forcedNow = $false
        if (-not [string]::IsNullOrEmpty($ForcedClassification)) {
            if ($ForcedClassificationFromAttempt -le 0) {
                $forcedNow = $true
            }
            elseif ($attemptNumber -ge $ForcedClassificationFromAttempt) {
                $forcedNow = $true
            }
        }
        if ($forcedNow) {
            return [ordered]@{
                ExitCode     = 124
                Classification = $ForcedClassification
                BytesWritten = [long]0
            }
        }

        $length = [int]$CountBytes
        if (($SkipBytes + $CountBytes) -gt ([long]$Source.Length)) {
            throw 'FAKE_INVOKER_RANGE_OUT_OF_BOUNDS'
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
            if ($writeCount -lt 0) {
                $writeCount = 0
            }
        }

        $payload = $bytes
        if ($writeCount -ne $length) {
            $payload = [byte[]]::new($writeCount)
            if ($writeCount -gt 0) {
                [System.Array]::Copy($bytes, 0, $payload, 0, $writeCount)
            }
        }
        [System.IO.File]::WriteAllBytes($OutputPath, $payload)

        return [ordered]@{
            ExitCode     = 0
            Classification = 'OK'
            BytesWritten = [long]$writeCount
        }
    }.GetNewClosure()

    return [ordered]@{
        Invoker = $invoker
        State   = $state
    }
}

function Test-PhoneLmTransferSelfTest {
    <#
        Pure host self-test for backward_formal_transport. No device calls, no adb, no exit.
        Uses in-memory byte[] fixtures (including 0x00/0x0A/0x0D/0xFF) to prove that raw bytes
        survive the transfer without any text transformation.

        Returns @{Passed; Cases; Failures} where Cases is one record per assertion case.
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [AllowEmptyString()]
        [string]$ScratchRoot = ''
    )

    $caseList = [System.Collections.Generic.List[object]]::new()
    $failureList = [System.Collections.Generic.List[string]]::new()

    function Add-SelfTestResult {
        param(
            [string]$CaseName,
            [bool]$CasePassed,
            [string]$CaseDetail
        )
        $detailText = $CaseDetail
        if ($null -eq $detailText) {
            $detailText = ''
        }
        $caseList.Add([ordered]@{
            Name   = $CaseName
            Passed = $CasePassed
            Detail = $detailText
        })
        if (-not $CasePassed) {
            $failureList.Add($CaseName)
        }
    }

    function Invoke-SelfTestStep {
        param(
            [string]$CaseName,
            [scriptblock]$Step
        )
        $passed = $false
        $detail = ''
        try {
            $outcome = & $Step
            if ($outcome -is [System.Collections.IDictionary]) {
                $passed = [bool]($outcome['Ok'])
                $detail = [string]($outcome['Detail'])
            }
            elseif ($outcome -is [bool]) {
                $passed = [bool]$outcome
            }
            else {
                $detail = 'step returned unsupported output type'
            }
        }
        catch {
            $passed = $false
            $detail = $_.Exception.Message
        }
        Add-SelfTestResult -CaseName $CaseName -CasePassed $passed -CaseDetail $detail
    }

    # ---- scratch (GUID-suffixed, never touches existing files) ----
    $root = ''
    if ([string]::IsNullOrWhiteSpace($ScratchRoot)) {
        $root = Join-Path ([System.IO.Path]::GetTempPath()) ('phonelm-transfer-selftest-' + [guid]::NewGuid().ToString('N'))
    }
    else {
        $root = Join-Path $ScratchRoot ('selftest-' + [guid]::NewGuid().ToString('N'))
    }
    [void][System.IO.Directory]::CreateDirectory($root)

    # Binary dataset with the four sentinel bytes every text pipeline would corrupt.
    $binarySource = [byte[]]::new(20000)
    for ($i = 0; $i -lt $binarySource.Length; $i++) {
        $binarySource[$i] = [byte](($i * 31 + 7) -band 0xFF)
    }
    $binarySource[0] = [byte]0x00
    $binarySource[1] = [byte]0x0A
    $binarySource[2] = [byte]0x0D
    $binarySource[3] = [byte]0xFF

    # 1. monotonic clock
    $null = Invoke-SelfTestStep 'clock_monotonic' {
        $clock = New-PhoneLmTransferClock
        $m1 = Get-PhoneLmTransferMilliseconds -Clock $clock
        Start-Sleep -Milliseconds 25
        $m2 = Get-PhoneLmTransferMilliseconds -Clock $clock
        $ok = ($m1 -ge 0) -and ($m2 -ge $m1) -and ($m2 -ge 20)
        return @{ Ok = $ok; Detail = ('m1={0} m2={1}' -f $m1, $m2) }
    }

    # 2. unsafe remote paths rejected before any invoker call
    $null = Invoke-SelfTestStep 'unsafe_remote_path_refused' {
        $probe = @{ Calls = 0 }
        $fakeAdb = {
            param([object]$Arguments, [double]$TimeoutSeconds)
            $probe.Calls = [int]$probe.Calls + 1
            return [ordered]@{ ExitCode = 0; Text = ''; Classification = 'OK' }
        }.GetNewClosure()
        $blocked = 0
        foreach ($bad in @('', 'ok/../x', 'say"hi')) {
            $threwUnsafe = $false
            try {
                Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath $bad | Out-Null
            }
            catch {
                $threwUnsafe = ($_.Exception.Message -like '*UNSAFE_REMOTE_PATH*')
            }
            if ($threwUnsafe) {
                $blocked = $blocked + 1
            }
        }
        $adbCalls = [int]$probe.Calls
        return @{ Ok = (($blocked -eq 3) -and ($adbCalls -eq 0)); Detail = ('blocked={0} adb_calls={1}' -f $blocked, $adbCalls) }
    }

    # 3. identity happy path: all three commands recorded with elapsed ms
    $null = Invoke-SelfTestStep 'identity_full_read' {
        $fakeAdb = {
            param([object]$Arguments, [double]$TimeoutSeconds)
            $sub = [string]$Arguments[1]
            switch ($sub) {
                'test' { return [ordered]@{ ExitCode = 0; Text = ''; Classification = 'OK' } }
                'stat' { return [ordered]@{ ExitCode = 0; Text = "4242`n"; Classification = 'OK' } }
                'sha256sum' { return [ordered]@{ ExitCode = 0; Text = (('a' * 64) + "  ./evidence.gz`n"); Classification = 'OK' } }
                default { return [ordered]@{ ExitCode = 1; Text = 'unexpected'; Classification = 'UNEXPECTED_COMMAND' } }
            }
        }
        $id = Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath '/data/local/tmp/evidence.bin'
        $cmds = @($id.Commands)
        $allRecorded = $true
        foreach ($c in $cmds) {
            if ($c.ElapsedMilliseconds -lt 0) { $allRecorded = $false }
            if ($c.ExitCode -eq $null) { $allRecorded = $false }
            if ($c.Arguments.Count -lt 3) { $allRecorded = $false }
        }
        $ok = $id.Exists -and ($id.SizeBytes -eq 4242) -and ($id.Sha256 -eq ('a' * 64)) -and `
            ($null -eq $id.Error) -and ($cmds.Count -eq 3) -and $allRecorded
        return @{ Ok = $ok; Detail = ('size={0} commands={1} recorded={2}' -f $id.SizeBytes, $cmds.Count, $allRecorded) }
    }

    # 4. stat failure -> REMOTE_SIZE_UNREADABLE (hash step still recorded)
    $null = Invoke-SelfTestStep 'identity_size_unreadable' {
        $fakeAdb = {
            param([object]$Arguments, [double]$TimeoutSeconds)
            $sub = [string]$Arguments[1]
            switch ($sub) {
                'test' { return [ordered]@{ ExitCode = 0; Text = ''; Classification = 'OK' } }
                'stat' { return [ordered]@{ ExitCode = 1; Text = 'stat: cannot statx'; Classification = 'REMOTE_SIZE_UNREADABLE' } }
                default { return [ordered]@{ ExitCode = 0; Text = (('a' * 64) + "  ./evidence.gz`n"); Classification = 'OK' } }
            }
        }
        $id = Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath '/data/local/tmp/evidence.bin'
        $ok = $id.Exists -and ($null -eq $id.SizeBytes) -and ($id.Error -eq 'REMOTE_SIZE_UNREADABLE') -and ($null -ne $id.Sha256)
        return @{ Ok = $ok; Detail = ('error={0} size={1}' -f [string]$id.Error, [string]$id.SizeBytes) }
    }

    # 5. sha256sum failure -> REMOTE_HASH_UNREADABLE
    $null = Invoke-SelfTestStep 'identity_hash_unreadable' {
        $fakeAdb = {
            param([object]$Arguments, [double]$TimeoutSeconds)
            $sub = [string]$Arguments[1]
            switch ($sub) {
                'test' { return [ordered]@{ ExitCode = 0; Text = ''; Classification = 'OK' } }
                'sha256sum' { return [ordered]@{ ExitCode = 1; Text = 'sha256sum: READ FAILED'; Classification = 'REMOTE_HASH_UNREADABLE' } }
                default { return [ordered]@{ ExitCode = 0; Text = "4242`n"; Classification = 'OK' } }
            }
        }
        $id = Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath '/data/local/tmp/evidence.bin'
        $ok = $id.Exists -and ($id.SizeBytes -eq 4242) -and ($null -eq $id.Sha256) -and ($id.Error -eq 'REMOTE_HASH_UNREADABLE')
        return @{ Ok = $ok; Detail = ('error={0} sha={1}' -f [string]$id.Error, [string]$id.Sha256) }
    }

    # 6. missing remote file: Exists=false, only the probe command recorded
    $null = Invoke-SelfTestStep 'identity_missing_file' {
        $fakeAdb = {
            param([object]$Arguments, [double]$TimeoutSeconds)
            return [ordered]@{ ExitCode = 1; Text = ''; Classification = 'REMOTE_RAW_MISSING' }
        }
        $id = Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $fakeAdb -RemotePath '/data/local/tmp/absent.bin'
        $cmds = @($id.Commands)
        $ok = (-not $id.Exists) -and ($null -eq $id.SizeBytes) -and ($null -eq $id.Sha256) -and ($cmds.Count -eq 1)
        return @{ Ok = $ok; Detail = ('exists={0} commands={1}' -f $id.Exists, $cmds.Count) }
    }

    # 7. single-chunk verified transfer
    $null = Invoke-SelfTestStep 'transfer_single_chunk_verified' {
        $source = [byte[]]::new(64)
        for ($i = 0; $i -lt 64; $i++) { $source[$i] = [byte](0xA0 -bxor $i) }
        $dest = Join-Path $root 'single.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/single.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $actual = [System.IO.File]::ReadAllBytes($dest)
        $mismatch = 0
        for ($i = 0; $i -lt $source.Length; $i++) {
            if ($actual[$i] -ne $source[$i]) { $mismatch = $mismatch + 1 }
        }
        $hashes = @($result.ChunkSha256)
        $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and $result.Complete -and ($result.Bytes -eq 64) -and `
            ($result.Chunks -eq 1) -and ($hashes.Count -eq 1) -and (@($result.Attempts).Count -eq 1) -and `
            ($mismatch -eq 0) -and ($hashes[0] -eq $expected)
        return @{ Ok = $ok; Detail = ('status={0} mismatch={1} hashes={2}' -f $result.Status, $mismatch, $hashes.Count) }
    }

    # 8. multi-chunk transfer with per-chunk sha256 verification
    $null = Invoke-SelfTestStep 'transfer_multi_chunk_hashes' {
        $dest = Join-Path $root 'multi.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $binarySource
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $binarySource
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/multi.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$binarySource.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $hashes = @($result.ChunkSha256)
        $chunkHashOk = $true
        $idx = 0
        for ($off = 0; $off -lt $binarySource.Length; $off = $off + 4096) {
            $len = [Math]::Min(4096, $binarySource.Length - $off)
            $slice = [byte[]]::new($len)
            [System.Array]::Copy($binarySource, $off, $slice, 0, $len)
            $want = Get-PhoneLmTransferByteSha256 -Bytes $slice
            if ($idx -ge $hashes.Count) { $chunkHashOk = $false }
            elseif ($want -ne $hashes[$idx]) { $chunkHashOk = $false }
            $idx = $idx + 1
        }
        $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and $result.Complete -and ($result.Chunks -eq 5) -and `
            ($hashes.Count -eq 5) -and $chunkHashOk
        return @{ Ok = $ok; Detail = ('status={0} chunks={1} hashes_ok={2}' -f $result.Status, $result.Chunks, $chunkHashOk) }
    }

    # 9. binary dataset (0x00/0x0A/0x0D/0xFF) round-trips element-wise
    $null = Invoke-SelfTestStep 'binary_roundtrip_elementwise' {
        $dest = Join-Path $root 'binary.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $binarySource
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $binarySource
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/binary.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$binarySource.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $actual = [System.IO.File]::ReadAllBytes($dest)
        $mismatch = 0
        if ($actual.Length -ne $binarySource.Length) {
            $mismatch = -1
        }
        else {
            for ($i = 0; $i -lt $binarySource.Length; $i++) {
                if ($actual[$i] -ne $binarySource[$i]) { $mismatch = $mismatch + 1 }
            }
        }
        $sentinels = ($actual[0] -eq 0x00) -and ($actual[1] -eq 0x0A) -and ($actual[2] -eq 0x0D) -and ($actual[3] -eq 0xFF)
        $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($mismatch -eq 0) -and $sentinels
        return @{ Ok = $ok; Detail = ('mismatch={0} sentinels={1}' -f $mismatch, $sentinels) }
    }

    # 10. short write on attempt 1, success on attempt 2
    $null = Invoke-SelfTestStep 'short_write_retry_then_success' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 17 + 3) -band 0xFF) }
        $dest = Join-Path $root 'retry.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -ShortWriteAttempts 1
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/retry.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $attempts = @($result.Attempts)
        $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($attempts.Count -eq 3) -and `
            ($attempts[0].Reason -eq 'SHORT_WRITE') -and (-not $attempts[0].Success) -and `
            $attempts[1].Success -and $attempts[2].Success
        return @{ Ok = $ok; Detail = ('status={0} attempts={1} first_reason={2}' -f $result.Status, $attempts.Count, $attempts[0].Reason) }
    }

    # 11. short write on every attempt -> incomplete, partial destination + .part preserved
    $null = Invoke-SelfTestStep 'short_write_always_incomplete' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 13 + 5) -band 0xFF) }
        $dest = Join-Path $root 'short-always.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -ShortWriteAttempts 1000000
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/short.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $partPath = '{0}.chunk-0.part' -f $dest
        $partOk = [System.IO.File]::Exists($partPath)
        $partLength = [long]-1
        if ($partOk) { $partLength = ([System.IO.FileInfo]::new($partPath)).Length }
        $ok = (-not $result.Complete) -and ($result.Status -eq 'CHUNK_BYTE_COUNT_MISMATCH') -and `
            ($result.FailedChunkIndex -eq 0) -and $partOk -and ($partLength -lt 4096) -and `
            (-not [System.IO.File]::Exists($dest))
        return @{ Ok = $ok; Detail = ('status={0} part_len={1} dest_exists={2}' -f $result.Status, $partLength, [System.IO.File]::Exists($dest)) }
    }

    # 12. timeout classification on every attempt -> matching transport status
    $null = Invoke-SelfTestStep 'timeout_classification_status' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 19 + 11) -band 0xFF) }
        $dest = Join-Path $root 'timeout.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -ForcedClassification 'ADB_TRANSPORT_TIMEOUT'
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/timeout.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $attempts = @($result.Attempts)
        $ok = (-not $result.Complete) -and ($result.Status -eq 'ADB_TRANSPORT_TIMEOUT') -and `
            ($attempts.Count -eq 2) -and ($result.LastClassification -eq 'ADB_TRANSPORT_TIMEOUT')
        return @{ Ok = $ok; Detail = ('status={0} attempts={1}' -f $result.Status, $attempts.Count) }
    }

    # 13. wrong bytes, correct length -> TRANSFER_HASH_MISMATCH at final verification
    $null = Invoke-SelfTestStep 'wrong_bytes_hash_mismatch' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 23 + 17) -band 0xFF) }
        $dest = Join-Path $root 'wrong-bytes.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -CorruptBytes
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/wrong.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $destLength = [long]-1
        if ([System.IO.File]::Exists($dest)) { $destLength = ([System.IO.FileInfo]::new($dest)).Length }
        $ok = (-not $result.Complete) -and ($result.Status -eq 'TRANSFER_HASH_MISMATCH') -and `
            ($destLength -eq $source.Length) -and ($result.Bytes -eq $source.Length)
        return @{ Ok = $ok; Detail = ('status={0} dest_len={1}' -f $result.Status, $destLength) }
    }

    # 14. existing destination refused
    $null = Invoke-SelfTestStep 'destination_exists_precondition' {
        $source = [byte[]]::new(64)
        $dest = Join-Path $root 'already-there.bin'
        [System.IO.File]::WriteAllBytes($dest, [byte[]]::new(16))
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/x.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = (Get-PhoneLmTransferByteSha256 -Bytes $source)
        }
        $threw = $false
        $msg = ''
        try { Receive-PhoneLmEvidenceFile @p | Out-Null }
        catch { $msg = $_.Exception.Message; $threw = ($msg -like '*DESTINATION_EXISTS*') }
        return @{ Ok = $threw; Detail = $msg }
    }

    # 15. zero-byte expected size refused
    $null = Invoke-SelfTestStep 'expected_size_required_precondition' {
        $source = [byte[]]::new(64)
        $dest = Join-Path $root 'zero-size.bin'
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/x.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]0
            ExpectedSha256   = (Get-PhoneLmTransferByteSha256 -Bytes $source)
        }
        $threw = $false
        $msg = ''
        try { Receive-PhoneLmEvidenceFile @p | Out-Null }
        catch { $msg = $_.Exception.Message; $threw = ($msg -like '*EXPECTED_SIZE_REQUIRED*') }
        return @{ Ok = $threw; Detail = $msg }
    }

    # 16. out-of-range ChunkBytes refused
    $null = Invoke-SelfTestStep 'chunk_bytes_range_precondition' {
        $source = [byte[]]::new(64)
        $dest = Join-Path $root 'bad-chunk.bin'
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/x.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = (Get-PhoneLmTransferByteSha256 -Bytes $source)
            ChunkBytes       = 100
        }
        $threw = $false
        $msg = ''
        try { Receive-PhoneLmEvidenceFile @p | Out-Null }
        catch { $msg = $_.Exception.Message; $threw = ($msg -like '*CHUNK_BYTES_OUT_OF_RANGE*') }
        return @{ Ok = $threw; Detail = $msg }
    }

    # 17. global FileTimeoutSeconds budget enforced via the monotonic clock
    $null = Invoke-SelfTestStep 'global_timeout_budget' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 29 + 3) -band 0xFF) }
        $dest = Join-Path $root 'budget.bin'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -DelayMilliseconds 400
        $p = @{
            ChunkInvoker      = $fake.Invoker
            RemotePath        = '/data/local/tmp/budget.bin'
            DestinationPath   = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256    = $expected
            ChunkBytes        = 4096
            ChunkAttempts     = 2
            FileTimeoutSeconds = 0.1
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $ok = (-not $result.Complete) -and ($result.Status -eq 'TRANSFER_TIMEOUT') -and ($result.Bytes -lt $source.Length)
        return @{ Ok = $ok; Detail = ('status={0} bytes={1} elapsed_ms={2}' -f $result.Status, $result.Bytes, [Math]::Round($result.ElapsedMilliseconds, 1)) }
    }

    # 18. integrity: all three identities agree -> verified
    $null = Invoke-SelfTestStep 'integrity_all_agree' {
        $before = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('b' * 64) }
        $after = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('b' * 64) }
        $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]1000 }
        $r = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after -Received $received -ReceivedSha256 ('b' * 64)
        $ok = $r.Verified -and ($r.Classification -eq 'TRANSFER_VERIFIED') -and (@($r.Reasons).Count -eq 0)
        return @{ Ok = $ok; Detail = $r.Classification }
    }

    # 19. integrity: remote sha changed between before/after
    $null = Invoke-SelfTestStep 'integrity_remote_changed' {
        $before = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('b' * 64) }
        $after = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('c' * 64) }
        $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]1000 }
        $r = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after -Received $received -ReceivedSha256 ('b' * 64)
        $ok = (-not $r.Verified) -and ($r.Classification -eq 'TRANSFER_REMOTE_CHANGED') -and (@($r.Reasons).Count -ge 1)
        return @{ Ok = $ok; Detail = $r.Classification }
    }

    # 20. integrity: received file identity disagrees -> POST_EXIT_EVIDENCE_INCOMPLETE
    $null = Invoke-SelfTestStep 'integrity_received_incomplete' {
        $before = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('b' * 64) }
        $after = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = ('b' * 64) }
        $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]1000 }
        $r = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after -Received $received -ReceivedSha256 ('d' * 64)
        $ok = (-not $r.Verified) -and ($r.Classification -eq 'POST_EXIT_EVIDENCE_INCOMPLETE') -and (@($r.Reasons).Count -ge 1)
        return @{ Ok = $ok; Detail = $r.Classification }
    }

    # 21. integrity: unreadable before/after identity -> TRANSFER_IDENTITY_UNVERIFIED
    $null = Invoke-SelfTestStep 'integrity_identity_unverified' {
        $before = [ordered]@{ Exists = $true; SizeBytes = [long]1000; Sha256 = 'not-a-hash' }
        $after = [ordered]@{ Exists = $true; SizeBytes = $null; Sha256 = $null }
        $received = [ordered]@{ Status = 'TRANSFER_VERIFIED'; Bytes = [long]1000 }
        $r = Test-PhoneLmTransferIntegrity -BeforeIdentity $before -AfterIdentity $after -Received $received -ReceivedSha256 ('b' * 64)
        $ok = (-not $r.Verified) -and ($r.Classification -eq 'TRANSFER_IDENTITY_UNVERIFIED') -and (@($r.Reasons).Count -ge 1)
        return @{ Ok = $ok; Detail = $r.Classification }
    }

    # 22. failed evidence write throws TRANSFER_EVIDENCE_WRITE_FAILED
    $null = Invoke-SelfTestStep 'evidence_write_failure_throws' {
        $source = [byte[]]::new(64)
        $dest = Join-Path $root 'ev-fail.bin'
        $dirPath = Join-Path $root 'ev-fail-dir'
        [void][System.IO.Directory]::CreateDirectory($dirPath)
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/x.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = (Get-PhoneLmTransferByteSha256 -Bytes $source)
            EvidencePath     = $dirPath
        }
        $threw = $false
        $msg = ''
        try { Receive-PhoneLmEvidenceFile @p | Out-Null }
        catch { $msg = $_.Exception.Message; $threw = ($msg -like '*TRANSFER_EVIDENCE_WRITE_FAILED*') }
        return @{ Ok = $threw; Detail = $msg }
    }

    # 23. evidence jsonl: one line per attempt plus one final line, all valid JSON
    $null = Invoke-SelfTestStep 'evidence_jsonl_lines' {
        $source = [byte[]]::new(8192)
        for ($i = 0; $i -lt 8192; $i++) { $source[$i] = [byte](($i * 37 + 9) -band 0xFF) }
        $dest = Join-Path $root 'ev-ok.bin'
        $evPath = Join-Path $root 'ev-ok.jsonl'
        $expected = Get-PhoneLmTransferByteSha256 -Bytes $source
        $fake = New-PhoneLmTransferMemoryChunkInvoker -Source $source -ShortWriteAttempts 1
        $p = @{
            ChunkInvoker     = $fake.Invoker
            RemotePath       = '/data/local/tmp/ev.bin'
            DestinationPath  = $dest
            ExpectedSizeBytes = [long]$source.Length
            ExpectedSha256   = $expected
            ChunkBytes       = 4096
            ChunkAttempts    = 2
            EvidencePath     = $evPath
        }
        $result = Receive-PhoneLmEvidenceFile @p
        $lines = [System.IO.File]::ReadAllLines($evPath)
        $parsed = 0
        foreach ($line in $lines) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                try { $null = ConvertFrom-Json -InputObject $line; $parsed = $parsed + 1 } catch { }
            }
        }
        $firstKind = ''
        $lastKind = ''
        if ($lines.Count -gt 0) {
            $firstObj = ConvertFrom-Json -InputObject $lines[0]
            $firstKind = [string]($firstObj.Kind)
            $lastObj = ConvertFrom-Json -InputObject $lines[$lines.Count - 1]
            $lastKind = [string]($lastObj.Kind)
        }
        $expectedLines = @($result.Attempts).Count + 1
        $ok = ($result.Status -eq 'TRANSFER_VERIFIED') -and ($lines.Count -eq $expectedLines) -and `
            ($parsed -eq $lines.Count) -and ($firstKind -eq 'attempt') -and ($lastKind -eq 'final')
        return @{ Ok = $ok; Detail = ('lines={0} expected={1} parsed={2} first={3} last={4}' -f $lines.Count, $expectedLines, $parsed, $firstKind, $lastKind) }
    }

    return [ordered]@{
        Passed   = ($failureList.Count -eq 0)
        Cases    = @($caseList)
        Failures = @($failureList)
    }
}
