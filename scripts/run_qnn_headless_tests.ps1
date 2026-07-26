param(
    [Parameter(Mandatory = $true)][Alias("SdkRoot")][string]$QairtSdkRoot,
    [string]$ExpectedBuildId = "2.48.40.260702151143",
    [ValidateSet(
        "device-probe", "qnn-forward", "linear", "mlp-split", "mlp-fused", "mlp-full-step",
        "transformer-forward", "softmax-backward", "attention-backward", "layernorm-backward",
        "transformer-mse", "tiny-lm-ce", "sgd-one-step", "momentum-one-step", "adam-one-step",
        "tiny-lm-stability", "phase01-adam", "generation-diagnostics", "api-trace", "callback-bound",
        "qnn-reproducibility", "qnn-graph-bisection",
        "qnn-graph-bisection-prelude", "qnn-graph-full-isolated",
        "qnn-graph-dinput-isolated", "qnn-graph-dembedding-isolated",
        "qnn-adam-diagnostic"
    )][string]$Suite = "device-probe",
    [ValidateSet("BACKGROUND_CORRECTNESS", "EXCLUSIVE_BENCHMARK")]
    [string]$TestMode = "BACKGROUND_CORRECTNESS",
    [ValidateRange(60, 86400)][int]$TimeoutSeconds = 14400,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss-fff"),
    [switch]$SkipBuild,
    [switch]$SkipInstall,
    [switch]$SkipAudit
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ($RunId.Length -notin 1..64 -or $RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId must match [A-Za-z0-9._-]+ and be at most 64 characters."
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

$online = @((& $adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } | ForEach-Object { $Matches[1] })
if ($online.Count -ne 1) { throw "Expected exactly one online ADB device; found $($online.Count)." }
$device = $online[0]
function Adb([string[]]$Arguments) {
    $output = & $adb -s $device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output" }
    return $output
}
function Private-Cat([string]$Path) { return (Adb @("shell", "run-as", $package, "cat", $Path)) -join "`n" }
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
    Adb @("shell", "am", "force-stop", $package) | Out-Null
    $beforeTop = Top-Package
    $instrumentStdout = Join-Path $reportRoot "instrumentation.txt"
    $instrumentStderr = Join-Path $reportRoot "instrumentation-stderr.txt"
    $instrumentArguments = @(
        "-s", $device, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", $class, "-e", "suite", $Suite, "-e", "testMode", $TestMode,
        "-e", "runId", $RunId, $runner
    )
    $instrumentProcess = Start-Process -FilePath $adb -ArgumentList $instrumentArguments `
        -RedirectStandardOutput $instrumentStdout -RedirectStandardError $instrumentStderr `
        -PassThru -WindowStyle Hidden
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $samples = [Collections.Generic.List[object]]::new()
    $phoneLmTopCount = 0
    while (-not $instrumentProcess.HasExited) {
        if ([DateTime]::UtcNow -ge $deadline) {
            $instrumentProcess.Kill()
            Adb @("shell", "am", "force-stop", $package) | Out-Null
            throw "Instrumentation timed out after $TimeoutSeconds seconds."
        }
        $top = Top-Package
        if ($top -eq $package) { $phoneLmTopCount++ }
        $samples.Add([ordered]@{ elapsed_seconds = [int](([DateTime]::UtcNow - $deadline.AddSeconds(-$TimeoutSeconds)).TotalSeconds); phonelm_is_top = ($top -eq $package) })
        Start-Sleep -Seconds 2
        $instrumentProcess.Refresh()
    }
    $instrumentProcess.WaitForExit()
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
    Get-Content -LiteralPath (Join-Path $reportRoot "status.json")
} finally { Pop-Location }
