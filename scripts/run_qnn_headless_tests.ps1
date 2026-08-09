param(
    [Parameter(Mandatory = $true)][Alias("SdkRoot")][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [ValidateSet(
        "device-probe", "qnn-forward", "linear", "mlp-split", "mlp-fused", "mlp-full-step",
        "transformer-forward", "softmax-backward", "attention-backward", "layernorm-backward",
        "transformer-mse", "tiny-lm-ce", "sgd-one-step", "momentum-one-step", "adam-one-step",
        "tiny-lm-stability", "phase01-adam", "generation-diagnostics", "api-trace", "callback-bound",
        "qnn-reproducibility", "qnn-graph-bisection",
        "qnn-graph-bisection-prelude", "qnn-graph-full-isolated",
        "qnn-graph-dinput-isolated", "qnn-graph-dembedding-isolated",
        "qnn-graph-order-full-dinput-dembedding", "qnn-graph-order-full-dembedding-dinput",
        "qnn-graph-order-dinput-full-dembedding", "qnn-graph-order-dinput-dembedding-full",
        "qnn-graph-order-dembedding-full-dinput", "qnn-graph-order-dembedding-dinput-full",
        "qnn-graph-order-full-full-full", "qnn-graph-order-dinput-dinput-dinput",
        "qnn-graph-order-dembedding-dembedding-dembedding",
        "qnn-tap-backward-regions", "qnn-tap-layernorm1", "qnn-tap-dscores-only",
        "qnn-tap-dprob-dscores",
        "qnn-adam-diagnostic", "qnn-adam-late-baseline", "qnn-adam-late-diagnostic",
        "post-fix-end-to-end",
        "scale-sequence-16-smoke", "scale-sequence-32-smoke",
        "scale-dimension-32-smoke", "scale-layers-2-smoke",
        "scale-heads-2-smoke", "scale-formal",
        "scale-l2h1-t16d16-smoke", "scale-l2h1-t32d32-smoke",
        "scale-l1h2-t16d16-smoke", "scale-l1h2-t32d32-smoke",
        "scale-l2h2-t16d16-smoke", "scale-l2h2-t32d32-smoke",
        "scale-l2h1-formal", "scale-l1h2-formal",
        "scale-l2h2-t32d32-formal", "scale-l2h2-t32d32-diagnostic", "nicopedia-parity"
    )][string]$Suite = "device-probe",
    [ValidateSet("BACKGROUND_CORRECTNESS", "EXCLUSIVE_BENCHMARK")]
    [string]$TestMode = "BACKGROUND_CORRECTNESS",
    [ValidateRange(60, 86400)][int]$TimeoutSeconds = 14400,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss-fff"),
    [string]$CheckpointHostPath,
    [string]$NicopediaAnchorHostPath,
    [string]$PromptHostPath,
    [string]$ExpectedPromptSha256,
    [ValidateSet(0, 1, 2)][int]$HtpContextGraphSplitting = 0,
    [switch]$SkipBuild,
    [switch]$SkipInstall,
    [switch]$SkipAudit,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "qairt_version.ps1")
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
    -ExpectedBuildId $ExpectedBuildId
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ($RunId.Length -notin 1..64 -or $RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId must match [A-Za-z0-9._-]+ and be at most 64 characters."
}
function Resolve-StablePhysicalIdentity([string]$Serial, [string]$BootSerial) {
    $values = @(@($Serial, $BootSerial) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -ne "unknown" } |
        Select-Object -Unique)
    if ($values.Count -ne 1) {
        throw "Unable to establish one stable physical-device identity (identifier redacted)."
    }
    return [string]($values[0])
}
function Get-AndroidIntegerField([string]$Text, [string]$Name, [string]$Source) {
    $match = [regex]::Match(
        $Text, "(?m)^\s*$([regex]::Escape($Name)):\s*(-?\d+)")
    if (-not $match.Success) { throw "Android $Source $Name is unavailable." }
    return [int]$match.Groups[1].Value
}
if ($SelfTest) {
    if ((Resolve-StablePhysicalIdentity "stable-a" "stable-a") -ne "stable-a") {
        throw "HEADLESS_SELFTEST_IDENTITY_EQUAL"
    }
    if ((Resolve-StablePhysicalIdentity "stable-b" "") -ne "stable-b") {
        throw "HEADLESS_SELFTEST_IDENTITY_SINGLE"
    }
    foreach ($invalid in @(
        @{ serial = ""; boot = "" },
        @{ serial = "unknown"; boot = "" },
        @{ serial = "stable-c"; boot = "stable-d" }
    )) {
        $failedClosed = $false
        try { Resolve-StablePhysicalIdentity $invalid.serial $invalid.boot | Out-Null }
        catch { $failedClosed = $true }
        if (-not $failedClosed) { throw "HEADLESS_SELFTEST_IDENTITY_FAIL_CLOSED" }
    }
    $batteryFixture = "Current Battery Service state:`n  level: 87`n  temperature: 315`n  health: 2`n"
    if ((Get-AndroidIntegerField $batteryFixture "level" "battery") -ne 87 -or
        (Get-AndroidIntegerField $batteryFixture "temperature" "battery") -ne 315 -or
        (Get-AndroidIntegerField $batteryFixture "health" "battery") -ne 2) {
        throw "HEADLESS_SELFTEST_DEVICE_CONDITION_PARSE"
    }
    Write-Host "run_qnn_headless_tests_self_test=PASS"
    exit 0
}
if ($Suite -eq "nicopedia-parity") {
    if ($TestMode -ne "BACKGROUND_CORRECTNESS") {
        throw "nicopedia-parity is restricted to BACKGROUND_CORRECTNESS."
    }
    if ([string]::IsNullOrWhiteSpace($CheckpointHostPath) -or -not (Test-Path -LiteralPath $CheckpointHostPath -PathType Leaf)) {
        throw "nicopedia-parity requires an existing checkpoint host file."
    }
    if ([string]::IsNullOrWhiteSpace($NicopediaAnchorHostPath) -or -not (Test-Path -LiteralPath $NicopediaAnchorHostPath -PathType Leaf)) {
        throw "nicopedia-parity requires an existing checkpoint anchor report."
    }
    if ([string]::IsNullOrWhiteSpace($PromptHostPath) -or -not (Test-Path -LiteralPath $PromptHostPath -PathType Leaf)) {
        throw "nicopedia-parity requires an existing prompt host file."
    }
    if ($ExpectedPromptSha256 -notmatch '^[0-9a-f]{64}$') {
        throw "nicopedia-parity requires a private lowercase SHA-256 prompt identity."
    }
}
if (-not (Test-Path -LiteralPath $QairtSdkRoot -PathType Container)) {
    throw "QAIRT SDK root does not exist."
}
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA "Android\Sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = "com.yuubinnkyoku.phonelm"
$runner = "$package.test/androidx.test.runner.AndroidJUnitRunner"
$class = "$package.HeadlessDeviceTestRunner"
$reportBase = [IO.Path]::GetFullPath((Join-Path $root "build\reports\qnn-headless"))
$reportRoot = [IO.Path]::GetFullPath((Join-Path $reportBase $RunId))
if (-not $reportRoot.StartsWith($reportBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Resolved report path escaped build/reports."
}
if (Test-Path -LiteralPath $reportRoot) { throw "Report run already exists: $RunId" }
$apk = Join-Path $root "app\build\outputs\apk\debug\app-debug.apk"
$testApk = Join-Path $root "app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

function Get-PrivateAdbOutput([string]$Endpoint, [string[]]$Arguments) {
    $output = & $adb -s $Endpoint @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB device identity query failed (endpoint redacted)." }
    return ($output -join "`n").Trim()
}
function Get-PhysicalDeviceIdentity([string]$Endpoint) {
    $emulator = Get-PrivateAdbOutput $Endpoint @("shell", "getprop", "ro.kernel.qemu")
    if ($emulator -eq "1") { throw "Emulator endpoints are not eligible for headless QNN testing." }
    $serial = Get-PrivateAdbOutput $Endpoint @("shell", "getprop", "ro.serialno")
    $bootSerial = Get-PrivateAdbOutput $Endpoint @("shell", "getprop", "ro.boot.serialno")
    return Resolve-StablePhysicalIdentity $serial $bootSerial
}
$online = @((& $adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } | ForEach-Object { $Matches[1] })
if ($online.Count -eq 0) { throw "No online ADB endpoint is available." }
$physicalDevices = @{}
foreach ($endpoint in $online) {
    $stableId = Get-PhysicalDeviceIdentity $endpoint
    if (-not $physicalDevices.ContainsKey($stableId)) { $physicalDevices[$stableId] = [Collections.Generic.List[string]]::new() }
    $physicalDevices[$stableId].Add($endpoint)
}
if ($physicalDevices.Count -ne 1) { throw "Expected exactly one physical ADB device after alias deduplication; found $($physicalDevices.Count)." }
$deviceAliases = [Collections.Generic.List[string]]::new()
foreach ($aliases in $physicalDevices.Values) {
    foreach ($alias in $aliases) { $deviceAliases.Add($alias) }
}
$device = @($deviceAliases | Sort-Object @{ Expression = { if ($_ -match ':') { 1 } else { 0 } } }, @{ Expression = { $_ } })[0]
[ordered]@{
    schema_version = 1
    stable_device_id = [string]($physicalDevices.Keys | Select-Object -First 1)
    selected_endpoint = $device
    endpoint_aliases = @($deviceAliases)
    physical_device_count = $physicalDevices.Count
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $reportRoot "device-identity-private.json") -Encoding utf8

$nicopediaAnchorHash = ""
if ($Suite -eq "nicopedia-parity") {
    $anchorText = Get-Content -Raw -LiteralPath $NicopediaAnchorHostPath
    $anchorMatch = [regex]::Match($anchorText, '(?m)^final_parameter_hash=(fnv1a64:[0-9a-f]{16})$')
    if (-not $anchorMatch.Success) { throw "Nicopedia checkpoint anchor hash is unavailable." }
    $nicopediaAnchorHash = $anchorMatch.Groups[1].Value
    $promptSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PromptHostPath).Hash.ToLowerInvariant()
    if ($promptSha256 -ne $ExpectedPromptSha256) {
        throw "Nicopedia parity requires the fixed Greedy generation prompt."
    }
    [ordered]@{
        schema_version = 1
        checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $CheckpointHostPath).Hash.ToLowerInvariant()
        anchor_parameter_hash = $nicopediaAnchorHash
        prompt_sha256 = $promptSha256
        checkpoint_step = 1000
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot "nicopedia-input-identity-private.json") -Encoding utf8
}
function Adb([string[]]$Arguments) {
    $output = & $adb -s $device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output" }
    return $output
}
function Private-Cat([string]$Path) { return (Adb @("shell", "run-as", $package, "cat", $Path)) -join "`n" }
function Get-DeviceCondition {
    $thermal = (Adb @("shell", "dumpsys", "thermalservice")) -join "`n"
    $thermalMatch = [regex]::Match($thermal, '(?m)^Thermal Status:\s*(\d+)')
    if (-not $thermalMatch.Success) { throw "Android thermal status is unavailable." }
    $battery = (Adb @("shell", "dumpsys", "battery")) -join "`n"
    [ordered]@{
        thermal_status = [int]$thermalMatch.Groups[1].Value
        battery_level = Get-AndroidIntegerField $battery "level" "battery"
        battery_temperature_c = (Get-AndroidIntegerField $battery "temperature" "battery") / 10.0
        battery_health = Get-AndroidIntegerField $battery "health" "battery"
    }
}
function Assert-SafeDeviceCondition([Collections.IDictionary]$Condition) {
    if ([int]$Condition.thermal_status -ge 5) {
        throw "THERMAL_ABORT: Android thermal EMERGENCY or SHUTDOWN."
    }
    if ([int]$Condition.battery_health -ne 2) {
        throw "BATTERY_ABORT: Android reported an explicit battery health fault."
    }
}
function Stage-NicopediaParityInputs {
    $relativeDirectory = "files/headless-input/$RunId"
    $checkpointTarget = "$relativeDirectory/htp-seed1-l19-step1000.ckpt"
    $promptTarget = "$relativeDirectory/prompt.bin"
    $temporaryCheckpoint = "/data/local/tmp/phonelm-headless-$RunId-checkpoint"
    $temporaryPrompt = "/data/local/tmp/phonelm-headless-$RunId-prompt"
    try {
        & $adb -s $device push $CheckpointHostPath $temporaryCheckpoint *> $null
        if ($LASTEXITCODE -ne 0) { throw "Unable to stage the nicopedia parity checkpoint (host path redacted)." }
        & $adb -s $device push $PromptHostPath $temporaryPrompt *> $null
        if ($LASTEXITCODE -ne 0) { throw "Unable to stage the nicopedia parity prompt (host path redacted)." }
        # Keep each command as an argv operation. `adb shell ... sh -c` joins
        # Windows arguments before the remote shell parses them and can strip
        # the command-string boundary, as demonstrated by the fail-closed
        # staging control.
        Adb @("shell", "run-as", $package, "mkdir", "-p", $relativeDirectory) | Out-Null
        Adb @("shell", "run-as", $package, "cp", $temporaryCheckpoint, $checkpointTarget) | Out-Null
        Adb @("shell", "run-as", $package, "cp", $temporaryPrompt, $promptTarget) | Out-Null
        Adb @("shell", "run-as", $package, "chmod", "600", $checkpointTarget, $promptTarget) | Out-Null
    } finally {
        Adb @("shell", "rm", "-f", "--", $temporaryCheckpoint, $temporaryPrompt) | Out-Null
    }
}
function Top-Package {
    $dump = (Adb @("shell", "dumpsys", "activity", "activities")) -join "`n"
    $match = [regex]::Match($dump, '(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
    if ($match.Success) { return $match.Groups[1].Value }
    return "UNKNOWN"
}

Push-Location $root
try {
    if (-not $SkipBuild) {
        $gradleArguments = @(
            ":app:assembleDebug", ":app:assembleDebugAndroidTest",
            "-Pphonelm.enableQnn=true", "-Pqairt.sdkRoot=$QairtSdkRoot",
            "-Pqairt.expectedBuildId=$ExpectedBuildId", "--no-daemon"
        )
        & .\gradlew.bat $gradleArguments
        if ($LASTEXITCODE -ne 0) { throw "QNN instrumentation build failed" }
    }
    if (-not $SkipAudit) {
        & (Join-Path $PSScriptRoot "audit_qnn_apk.ps1") -ApkPath $apk `
            -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
            -ReportPath (Join-Path $reportRoot "apk-audit.txt")
    }
    if (-not $SkipInstall) {
        Adb @("install", "-r", "-t", $apk) | Out-Null
        Adb @("install", "-r", "-t", $testApk) | Out-Null
    }
    if ($Suite -eq "nicopedia-parity") { Stage-NicopediaParityInputs }
    $conditionBefore = Get-DeviceCondition
    Assert-SafeDeviceCondition $conditionBefore
    Adb @("shell", "am", "force-stop", $package) | Out-Null
    $beforeTop = Top-Package
    $instrumentStdout = Join-Path $reportRoot "instrumentation.txt"
    $instrumentStderr = Join-Path $reportRoot "instrumentation-stderr.txt"
    $instrumentArguments = @(
        "-s", $device, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", $class, "-e", "suite", $Suite, "-e", "testMode", $TestMode,
        "-e", "runId", $RunId, $runner
    )
    if ($Suite -eq "nicopedia-parity") {
        $instrumentArguments = @(
            "-s", $device, "shell", "am", "instrument", "-w", "-r",
            "-e", "class", $class, "-e", "suite", $Suite, "-e", "testMode", $TestMode,
            "-e", "runId", $RunId, "-e", "htpContextGraphSplitting", "$HtpContextGraphSplitting", $runner
        )
    }
    $instrumentProcess = Start-Process -FilePath $adb -ArgumentList $instrumentArguments `
        -RedirectStandardOutput $instrumentStdout -RedirectStandardError $instrumentStderr `
        -PassThru -WindowStyle Hidden
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $samples = [Collections.Generic.List[object]]::new()
    $thermalSamples = [Collections.Generic.List[int]]::new()
    $thermalSamples.Add([int]$conditionBefore.thermal_status)
    $phoneLmTopCount = 0
    $lastThermalSample = [DateTime]::UtcNow
    while (-not $instrumentProcess.HasExited) {
        if ([DateTime]::UtcNow -ge $deadline) {
            $instrumentProcess.Kill()
            Adb @("shell", "am", "force-stop", $package) | Out-Null
            throw "Instrumentation timed out after $TimeoutSeconds seconds."
        }
        $top = Top-Package
        if ($top -eq $package) { $phoneLmTopCount++ }
        $samples.Add([ordered]@{ elapsed_seconds = [int](([DateTime]::UtcNow - $deadline.AddSeconds(-$TimeoutSeconds)).TotalSeconds); phonelm_is_top = ($top -eq $package) })
        if (([DateTime]::UtcNow - $lastThermalSample).TotalSeconds -ge 30) {
            $conditionDuring = Get-DeviceCondition
            $thermalStatus = [int]$conditionDuring.thermal_status
            $thermalSamples.Add($thermalStatus)
            $lastThermalSample = [DateTime]::UtcNow
            try {
                Assert-SafeDeviceCondition $conditionDuring
            } catch {
                $instrumentProcess.Kill()
                Adb @("shell", "am", "force-stop", $package) | Out-Null
                throw
            }
        }
        Start-Sleep -Seconds 2
        $instrumentProcess.Refresh()
    }
    $instrumentProcess.WaitForExit()
    $conditionAfter = Get-DeviceCondition
    $thermalSamples.Add([int]$conditionAfter.thermal_status)
    Assert-SafeDeviceCondition $conditionAfter
    [ordered]@{
        schema_version = 1
        before = $conditionBefore
        after = $conditionAfter
        maximum_thermal_status = ($thermalSamples | Measure-Object -Maximum).Maximum
        cooldown_requested = $false
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $reportRoot "device-condition-private.json") -Encoding utf8
    $instrumentOutput = Get-Content -Raw -LiteralPath $instrumentStdout
    $afterTop = Top-Package
    [ordered]@{
        schema_version = 1
        sample_count = $samples.Count
        phonelm_became_top_activity_count = $phoneLmTopCount
        focus_takeover_count = $phoneLmTopCount
        foreground_preserved = ($phoneLmTopCount -eq 0 -and $beforeTop -ne $package -and $afterTop -ne $package)
        samples = $samples
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $reportRoot "activity-sampling.json") -Encoding utf8
    $status = Private-Cat "files/headless/status.json"
    $statusSize = ((Adb @("shell", "run-as", $package, "stat", "-c", "%s", "files/headless/status.json")) -join "").Trim()
    if ($statusSize -ne "4096") { throw "Headless status record is not exactly 4096 bytes." }
    $status | Set-Content -LiteralPath (Join-Path $reportRoot "status.json") -Encoding utf8
    $deviceReport = ""
    $reportMatch = [regex]::Match($status, '"report_relative_path":"([^"]*)"')
    if ($reportMatch.Success -and $reportMatch.Groups[1].Value) {
        $relative = $reportMatch.Groups[1].Value
        $privatePath = "files/$relative"
        $deviceReport = Private-Cat $privatePath
        $deviceReport | Set-Content -LiteralPath (Join-Path $reportRoot "device-report.txt") -Encoding utf8
    }
    if ($instrumentProcess.ExitCode -ne 0) { throw "Instrumentation process returned exit code $($instrumentProcess.ExitCode)." }
    if ($instrumentOutput -notmatch '(?m)^OK \(') { throw "Instrumentation did not report success" }
    if ($status -notmatch '"status":"PASSED"') { throw "Headless suite did not pass" }
    if ($phoneLmTopCount -ne 0) { throw "PhoneLM became the top activity during headless execution." }
    foreach ($required in @(
        "activity_create_count=0", "activity_resume_count=0",
        "phonelm_became_top_activity_count=0", "focus_takeover_count=0",
        "single_flight_result=ALREADY_RUNNING", "compile_time_qairt_build_id=$ExpectedBuildId",
        "headless_test_mode=$TestMode", "backend_requested=HTP", "cpu_fallback=false"
    )) {
        if ($deviceReport -notmatch "(?m)^$([regex]::Escape($required))$") {
            throw "Device report is missing required evidence: $required"
        }
    }
    if ($Suite -eq "nicopedia-parity") {
        $contextConfigRejected = $HtpContextGraphSplitting -ne 0 -and
            $deviceReport -match '(?m)^status=FAILED$' -and
            $deviceReport -match '(?m)^error=context_create: contextCreate=5010$' -and
            $deviceReport -match '(?m)^failed_api=context_create$' -and
            $deviceReport -match '(?m)^context_create_result=5010$' -and
            $deviceReport -match '(?m)^api_trace_graph_execute_attempt_count=0$'
        if (-not $contextConfigRejected) {
            $deviceHash = [regex]::Match(
                $deviceReport, '(?m)^checkpoint_parameter_hash=(fnv1a64:[0-9a-f]{16})$')
            if (-not $deviceHash.Success -or $deviceHash.Groups[1].Value -ne $nicopediaAnchorHash) {
                throw "Nicopedia checkpoint identity does not match the approved anchor."
            }
        }
    }
    Get-Content -LiteralPath (Join-Path $reportRoot "status.json")
} finally { Pop-Location }
