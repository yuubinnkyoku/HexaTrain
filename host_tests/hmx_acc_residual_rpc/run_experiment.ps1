# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
<#
.SYNOPSIS
  Build / deploy / provenance orchestrator for the HMX residual FastRPC probe.

.DESCRIPTION
  Steps (always in this order):
    1. Record repo provenance: git HEAD SHA, branch, dirty flag.
    2. Build the HLOS stub + host exe and the V81 DSP skel with the Hexagon
       SDK build_cmake wrapper (README "Building" section), passing the same
       -DHMX_PROBE_BUILD_ID=0x<32-bit> to both builds through a response
       file (the SDK's bash arg parser corrupts values containing '=', so
       CMAKE_C_FLAGS=@<file> is used instead of CMAKE_C_FLAGS=-D...=...).
    3. Compute SHA-256 of each artifact (stub .so, skel .so, host exe) and
       print them as a table.
    4. Optional -DeployAndRun (DEFAULT OFF, no device contact otherwise):
       adb push to the recorded device paths, set DSP_LIBRARY_PATH, run the
       test binary with retry. Records the CDSP / unsigned-PD settings.
    5. Write build\hmx_evidence\b_provenance_<timestamp>.txt with git SHA,
       artifact hashes, device paths, DSP_LIBRARY_PATH, PD config and a
       tombstoned (truncated) build/deploy stdout.

  Default with no switch == -Provenance: steps 1-3 + the recorded (but NOT
  executed) device plan + step 5. This is the cheap validation mode.

.PARAMETER Provenance
  Explicit no-device mode (identical to the default). Used for cheap
  validation runs.

.PARAMETER DeployAndRun
  Also deploy to a connected device and run the probe. Never implied.

.PARAMETER SkipBuild
  Reuse existing build outputs instead of running build_cmake.

.EXAMPLE
  .\run_experiment.ps1 -Provenance
  .\run_experiment.ps1 -DeployAndRun -Q 10 -Mode 0
#>
[CmdletBinding()]
param(
  [switch]$Provenance,
  [switch]$DeployAndRun,
  [switch]$SkipBuild,
  [string]$SdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [string]$BuildId = '',
  [string]$Q = '10',
  [string]$Mode = '0',
  [int]$DeployRetries = 3,
  # Deterministic device endpoint (docs/agent/device-test-tiers.md: never rely
  # on an implicit "the only device"; a physical phone can present several
  # online adb endpoints, so the runner must address one explicitly).
  [string]$DeviceSerial = '',
  [string]$DeviceDir = '/data/local/tmp/hmx_probe',
  [string]$DspLibraryPath = '/data/local/tmp/hmx_probe:/system/lib/rfsa/adsp:/vendor/lib/rfsa/adsp:/dsp'
)

$ErrorActionPreference = 'Stop'

# Pure build-id provenance gate (fail-closed orchestrator-level identity
# check; device-harness semantics in hmx_identity.h stay UNAVAILABLE-tolerant).
. (Join-Path $PSScriptRoot 'hmx_provenance_gate.ps1')

$script:ProjectDir = $PSScriptRoot
$script:RepoRoot = (git -C $PSScriptRoot rev-parse --show-toplevel 2>$null)
if (-not $script:RepoRoot) { throw 'GIT_NOT_A_REPO: cannot resolve repository root' }
$script:EvidenceDir = Join-Path $script:RepoRoot 'build\hmx_evidence'
$script:LogLines = New-Object System.Collections.Generic.List[string]

function Write-Log([string]$Text) {
  $script:LogLines.Add($Text)
  Write-Host $Text
}

function Get-GitValue([string[]]$GitArgs) {
  $out = & git -C $script:RepoRoot @GitArgs 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  return (@($out) -join "`n").Trim()
}

# ---------------------------------------------------------------------------
# Step 1: git provenance.
# ---------------------------------------------------------------------------
function Get-GitProvenance {
  $sha = Get-GitValue @('rev-parse', 'HEAD')
  if (-not $sha) { throw 'GIT_REV_PARSE_FAILED: HEAD SHA unavailable' }
  $branch = Get-GitValue @('rev-parse', '--abbrev-ref', 'HEAD')
  $statusRaw = & git -C $script:RepoRoot status --porcelain 2>$null
  $statusCount = @($statusRaw | Where-Object { $_ }).Count
  [pscustomobject]@{
    Sha        = $sha
    Short      = $sha.Substring(0, 12)
    Branch     = $branch
    Dirty      = ($statusCount -gt 0)
    StatusLine = $statusCount
  }
}

# ---------------------------------------------------------------------------
# Step 2: SDK environment + build_cmake builds.
# ---------------------------------------------------------------------------
function Initialize-SdkEnvironment {
  # Reproduces what setup_sdk_env.cmd/setup_sdk_env.ps1 would set, but
  # without rebuilding the qaic compiler or opening an SDK shell.
  # Needed env for build_cmake.exe:
  #   SDK_SETUP_ENV, HEXAGON_SDK_ROOT, DEFAULT_HEXAGON_TOOLS_ROOT,
  #   DEFAULT_DSP_ARCH, DEFAULT_BUILD, DEFAULT_HLOS_ARCH, DEFAULT_NO_QURT_INC,
  #   DEFAULT_TREE, CMAKE_ROOT_PATH, DEFAULT_QURT_PATH, ANDROID_ROOT_DIR,
  #   ANDROID_NDK, DEFAULT_TOOLS_VARIANT, HEXAGONSDK_TELEMATICS_ROOT, PATH
  # PATH must additionally contain: gow bin (uname/dirname/basename),
  # build\cmake\WinNT (build_cmake.exe) and Python.
  if (-not (Test-Path -LiteralPath $SdkRoot)) {
    throw "SDK_NOT_FOUND: $SdkRoot"
  }
  $hexTools = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'tools\HEXAGON_Tools') -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
  $cmakeTools = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'tools') -Directory -Filter 'cmake-*' |
    Sort-Object Name -Descending | Select-Object -First 1
  $androidNdk = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'tools') -Directory -Filter 'android-ndk-*' |
    Sort-Object Name -Descending | Select-Object -First 1
  $gow = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'tools\utils') -Directory -Filter 'gow-*' |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $hexTools) { throw "SDK_HEXAGON_TOOLS_NOT_FOUND: under $SdkRoot\tools\HEXAGON_Tools" }
  if (-not $cmakeTools) { throw "SDK_CMAKE_NOT_FOUND: under $SdkRoot\tools\cmake-*" }
  if (-not $androidNdk) { throw "SDK_NDK_NOT_FOUND: under $SdkRoot\tools\android-ndk-*" }
  if (-not $gow) { throw "SDK_GOW_NOT_FOUND: under $SdkRoot\tools\utils\gow-*" }
  $toolsVariant = 'toolv' + ($hexTools.Name -replace '\..*$', '')

  $env:SDK_SETUP_ENV = 'Done'
  $env:HEXAGON_SDK_ROOT = $SdkRoot
  $env:DEFAULT_HEXAGON_TOOLS_ROOT = Join-Path $SdkRoot ("tools\HEXAGON_Tools\" + $hexTools.Name)
  $env:DEFAULT_DSP_ARCH = 'v68'
  $env:DEFAULT_BUILD = 'ReleaseG'
  $env:DEFAULT_HLOS_ARCH = '64'
  $env:DEFAULT_NO_QURT_INC = '0'
  $env:DEFAULT_TREE = '1'
  $env:CMAKE_ROOT_PATH = $cmakeTools.FullName
  $env:DEFAULT_QURT_PATH = Join-Path $SdkRoot 'rtos\qurt'
  $env:ANDROID_ROOT_DIR = $androidNdk.FullName
  $env:ANDROID_NDK = $androidNdk.FullName
  $env:DEFAULT_TOOLS_VARIANT = $toolsVariant
  $env:HEXAGONSDK_TELEMATICS_ROOT = Join-Path $SdkRoot 'utils\telematics'
  $env:Path = (@(
      (Join-Path $gow.FullName 'bin'),
      (Join-Path $SdkRoot 'build\cmake\WinNT'),
      $env:Path
    ) -join ';')

  [pscustomobject]@{
    HexagonTools = $env:DEFAULT_HEXAGON_TOOLS_ROOT
    CmakeRoot    = $env:CMAKE_ROOT_PATH
    AndroidNdk   = $env:ANDROID_ROOT_DIR
    ToolsVariant = $toolsVariant
    BuildCmake   = (Join-Path $SdkRoot 'build\cmake\WinNT\build_cmake.exe')
  }
}

function Invoke-ProjectBuild([string]$BuildIdDefine) {
  # CMAKE_C_FLAGS=@<response file>: the SDK's cmake_configure.bash parses
  # unknown KEY=VALUE args with ${arg%=*} (last '='), which corrupts any
  # value that itself contains '=' (e.g. -DHMX_PROBE_BUILD_ID=0x...). A
  # response file carries the '=' safely and both NDK clang and
  # hexagon-clang expand @file at compile time. The same file is passed to
  # BOTH builds so the host and the DSP agree on the build id.
  # The FILE NAME itself carries the build id: a constant path would leave
  # the cmake cache unaware that the flags changed, and stale objects (from
  # a previous HEAD) would keep the old-id stamp while other sources got
  # rebuilt with the new id — a mismatch the device-side identity check
  # then correctly rejects (observed 2026-10-20, battery2 first pass).
  $idHex = ($BuildIdDefine -replace '^-DHMX_PROBE_BUILD_ID=', '')
  $rspPath = Join-Path $script:EvidenceDir "b_build_id_$idHex.rsp"
  Set-Content -LiteralPath $rspPath -Value $BuildIdDefine -NoNewline
  $rspUri = ($rspPath -replace '\\', '/')

  $results = @()
  $builds = @(
    @{ Target = 'hexagon'; Args = @('hexagon', 'DSP_ARCH=v81', "CMAKE_C_FLAGS=@$rspUri") },
    @{ Target = 'android'; Args = @('android', "CMAKE_C_FLAGS=@$rspUri") }
  )
  Push-Location $script:ProjectDir
  try {
    foreach ($build in $builds) {
      $buildArgs = $build.Args
      $stdout = & build_cmake.exe @buildArgs 2>&1 | ForEach-Object { "$_" }
      $code = $LASTEXITCODE
      $tail = @($stdout | Select-Object -Last 12)
      foreach ($line in $tail) { $script:LogLines.Add("  [build $($build.Target)] $line") }
      $results += [pscustomobject]@{
        Target = $build.Target
        Exit   = $code
        Tail   = $tail
      }
    }
  } finally {
    Pop-Location
  }
  $results
}

# ---------------------------------------------------------------------------
# Step 3: artifact hashing.
# ---------------------------------------------------------------------------
function Get-Artifacts {
  $paths = @(
    @{ Name = 'stub_so';  Path = (Join-Path $script:ProjectDir 'android_ReleaseG_aarch64\ship\libhexatrain_hmx_acc_residual.so') },
    @{ Name = 'skel_so';  Path = (Join-Path $script:ProjectDir 'hexagon_ReleaseG_toolv19_v81\ship\libhexatrain_hmx_acc_residual_skel.so') },
    @{ Name = 'host_exe'; Path = (Join-Path $script:ProjectDir 'android_ReleaseG_aarch64\ship\hexatrain_hmx_acc_residual_test') }
  )
  $rows = @()
  foreach ($entry in $paths) {
    $exists = Test-Path -LiteralPath $entry.Path
    $hash = $null
    $size = -1
    if ($exists) {
      $size = (Get-Item -LiteralPath $entry.Path).Length
      $hash = (Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $rows += [pscustomobject]@{
      Name   = $entry.Name
      Path   = $entry.Path
      Exists = $exists
      Size   = $size
      Sha256 = $hash
    }
  }
  $rows
}

# ---------------------------------------------------------------------------
# Step 4: optional deploy (only with -DeployAndRun; default OFF).
# ---------------------------------------------------------------------------
function Invoke-DeployAndRun($artifacts) {
  # Deploy is only reachable with -DeployAndRun (explicit opt-in). The CDSP
  # / unsigned-PD settings below are recorded in the provenance file and
  # mirror probe_host.c: CDSP_DOMAIN_ID=3, unsigned PD control enabled at
  # runtime (DSPRPC_CONTROL_UNSIGNED_MODULE, module=1), capability
  # expectation raw_arch=0x8081 (V81 CDSP), identity mode 30 runs first
  # on-device.
  foreach ($row in $artifacts) {
    if (-not $row.Exists) { throw "ARTIFACT_MISSING: $($row.Path) (build first)" }
  }
  $adbArgs = @()
  if ($DeviceSerial) { $adbArgs += @('-s', $DeviceSerial) }
  $serial = (& adb @adbArgs get-serialno 2>&1 | ForEach-Object { "$_" })
  if ($LASTEXITCODE -ne 0) { throw "ADB_SERIAL_FAILED: $serial" }
  Write-Log "device_serial=$serial"
  Write-Log "device_endpoint=explicit ($DeviceSerial) via -DeviceSerial -s flag"

  & adb @adbArgs shell mkdir -p $DeviceDir 2>&1 | Out-Null
  $map = @{
    'libhexatrain_hmx_acc_residual.so'       = ($artifacts | Where-Object Name -eq 'stub_so').Path
    'libhexatrain_hmx_acc_residual_skel.so'  = ($artifacts | Where-Object Name -eq 'skel_so').Path
    'hexatrain_hmx_acc_residual_test'        = ($artifacts | Where-Object Name -eq 'host_exe').Path
  }
  foreach ($name in $map.Keys) {
    Write-Log "adb push $($map[$name]) -> ${DeviceDir}/$name"
    & adb @adbArgs push $map[$name] "$DeviceDir/$name" 2>&1 | ForEach-Object { $script:LogLines.Add("  [adb] $_") }
    if ($LASTEXITCODE -ne 0) { throw "ADB_PUSH_FAILED: $name" }
  }

  # adb push does not carry the execute bit; /data/local/tmp allows exec for
  # shell user, and chmod is a plain non-root, non-sticky operation.
  & adb @adbArgs shell chmod 755 $DeviceDir/hexatrain_hmx_acc_residual_test 2>&1 | ForEach-Object { $script:LogLines.Add("  [adb] $_") }
  if ($LASTEXITCODE -ne 0) { throw 'ADB_CHMOD_FAILED: host test binary' }

  # Orchestrator-driven runs always append -strict-build-id: a nonzero
  # matching build id is required on the same handle BEFORE any numerical
  # RPC (probe_host fails closed with strict_identity_fail + exit 9).
  $remoteCmd = "cd $DeviceDir && export LD_LIBRARY_PATH=$DeviceDir && export DSP_LIBRARY_PATH=$DspLibraryPath && ./hexatrain_hmx_acc_residual_test $Q $Mode -strict-build-id"
  $attempt = 0
  $runExit = -1
  $runOutput = @()
  while ($attempt -lt $DeployRetries) {
    $attempt++
    Write-Log "device_run_attempt=$attempt/$DeployRetries cmd=$remoteCmd"
    $runOutput = & adb @adbArgs shell $remoteCmd 2>&1 | ForEach-Object { "$_" }
    $runExit = $LASTEXITCODE
    foreach ($line in $runOutput) { Write-Log "  [device] $line" }
    if ($runExit -eq 0) { break }
  }
  # Fail-closed build-id identity gate: a probe exit 0 with BUILD_ID=UNAVAILABLE
  # (dsp build id 0 does NOT fail identity inside the device harness) or any
  # unparsed/mismatched identity must not count as a successful provenance run.
  # Runs regardless of $runExit and AFTER the [device] lines above were logged,
  # so the evidence stays on the console. A throw here aborts before
  # Write-ProvenanceFile, exactly like the ADB_* throws above.
  $gate = Test-HmxBuildIdGate -Lines $runOutput
  if (-not $gate.Good) {
    $observed = if ($gate.MatchedLine) {
      "line=[$($gate.MatchedLine)] host=0x$($gate.HostBuildId.ToString('x8')) dsp=0x$($gate.DspBuildId.ToString('x8'))"
    } else {
      'line=<none> (no identity line in device output)'
    }
    throw "BUILD_ID_GATE_FAILED: reason=$($gate.Reason) expected=build_id_status=MATCH with nonzero equal ids; observed $observed"
  }
  Write-Log ("build_id_gate=PASS host=0x{0:x8} dsp=0x{1:x8}" -f $gate.HostBuildId, $gate.DspBuildId)
  [pscustomobject]@{ Exit = $runExit; Attempts = $attempt; Output = $runOutput }
}

# ---------------------------------------------------------------------------
# Step 5: provenance file.
# ---------------------------------------------------------------------------
function Write-ProvenanceFile {
  param(
    $Git,
    [string]$BuildIdDefine,
    $SdkInfo,
    $BuildResults,
    $Artifacts,
    $DeploySettings,
    $DeployResult
  )
  if (-not (Test-Path -LiteralPath $script:EvidenceDir)) {
    New-Item -ItemType Directory -Path $script:EvidenceDir -Force | Out-Null
  }
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $file = Join-Path $script:EvidenceDir "b_provenance_$stamp.txt"
  $out = New-Object System.Collections.Generic.List[string]
  $out.Add('# HMX residual probe provenance (b_provenance)')
  $out.Add("timestamp_local=$stamp")
  $out.Add("timestamp_utc=$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))")
  $out.Add("repo_root=$($script:RepoRoot)")
  $out.Add("git_head_sha=$($Git.Sha)")
  $out.Add("git_head_short=$($Git.Short)")
  $out.Add("git_branch=$($Git.Branch)")
  $out.Add("git_dirty=$($Git.Dirty)")
  $out.Add("git_dirty_file_count=$($Git.StatusLine)")
  $out.Add("git_note=tree state as of this run; rebuilds may regenerate untracked SDK build dirs")
  $out.Add("build_id_define=$BuildIdDefine")
  $out.Add("sdk_root=$SdkRoot")
  $out.Add("sdk_hexagon_tools=$($SdkInfo.HexagonTools)")
  $out.Add("sdk_cmake_root=$($SdkInfo.CmakeRoot)")
  $out.Add("sdk_android_ndk=$($SdkInfo.AndroidNdk)")
  $out.Add("sdk_tools_variant=$($SdkInfo.ToolsVariant)")
  $out.Add("build_cmake=$($SdkInfo.BuildCmake)")
  $out.Add('')
  $out.Add('[build]')
  if (@($BuildResults).Count -eq 0) {
    $out.Add('build=NOT_RUN (SkipBuild)')
  }
  foreach ($b in $BuildResults) {
    $out.Add("build_target=$($b.Target) exit=$($b.Exit)")
    foreach ($line in $b.Tail) { $out.Add("  $line") }
  }
  $out.Add('stdout_note=build stdout tombstoned to the last 12 lines per target')
  $out.Add('')
  $out.Add('[artifacts_sha256]')
  foreach ($a in $Artifacts) {
    $out.Add("artifact=$($a.Name) exists=$($a.Exists) size=$($a.Size) sha256=$($a.Sha256)")
    $out.Add("  path=$($a.Path)")
  }
  $out.Add('')
  $out.Add('[device_plan]')
  if ($DeploySettings) {
    $out.Add("domain=$($DeploySettings.Domain)")
    $out.Add("unsigned_pd=$($DeploySettings.UnsignedPd)")
    $out.Add("capability_expect=$($DeploySettings.CapabilityExpect)")
    $out.Add("device_dir=$($DeploySettings.DeviceDir)")
    $out.Add("dsp_library_path=$($DeploySettings.DspLibraryPath)")
    $out.Add("q=$($DeploySettings.Q)")
    $out.Add("mode=$($DeploySettings.Mode)")
    $out.Add("retries=$($DeploySettings.Retries)")
  } else {
    $out.Add('domain=CDSP_DOMAIN_ID=3')
    $out.Add('unsigned_pd=DSPRPC_CONTROL_UNSIGNED_MODULE enabled (control=1)')
    $out.Add('capability_expect=raw_arch=0x8081 (Hexagon V81 CDSP)')
    $out.Add("device_dir=$DeviceDir")
    $out.Add("dsp_library_path=$DspLibraryPath")
    $out.Add("q=$Q")
    $out.Add("mode=$Mode")
    $out.Add("retries=$DeployRetries")
  }
  $out.Add("deploy_executed=$(if ($DeployResult) { 'yes' } else { 'no' })")
  if ($DeployResult) {
    $out.Add("deploy_exit=$($DeployResult.Exit)")
    $out.Add("deploy_attempts=$($DeployResult.Attempts)")
    foreach ($line in $DeployResult.Output) { $out.Add("  [device] $line") }
    $out.Add('stdout_note=device stdout tombstoned per attempt above')
  } else {
    $out.Add('deploy_note=NOT EXECUTED (default no-device provenance mode)')
  }
  $out.Add('')
  $out.Add('[local_run_log]')
  foreach ($line in $script:LogLines) { $out.Add($line) }
  Set-Content -LiteralPath $file -Value $out
  $file
}

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
$gitInfo = Get-GitProvenance
Write-Log "git_head_sha=$($gitInfo.Sha)"
Write-Log "git_branch=$($gitInfo.Branch)"
Write-Log "git_dirty=$($gitInfo.Dirty) (changed_paths=$($gitInfo.StatusLine))"

if (-not $BuildId) {
  # Deterministic per-commit build id: first 8 hex digits of the HEAD SHA.
  $BuildId = '0x' + $gitInfo.Sha.Substring(0, 8)
}
$BuildIdDefine = "-DHMX_PROBE_BUILD_ID=$BuildId"
Write-Log "build_id=$BuildId"
Write-Log "build_id_define=$BuildIdDefine"

$sdkInfo = $null
$buildResults = @()
if ($SkipBuild) {
  Write-Log 'build=SKIPPED (-SkipBuild)'
} else {
  $sdkInfo = Initialize-SdkEnvironment
  Write-Log "sdk_root=$SdkRoot"
  Write-Log "hexagon_tools=$($sdkInfo.HexagonTools)"
  $buildResults = Invoke-ProjectBuild $BuildIdDefine
  foreach ($b in $buildResults) {
    Write-Log "build_target=$($b.Target) exit=$($b.Exit)"
    if ($b.Exit -ne 0) {
      Write-Log "build_status=FAILED target=$($b.Target) (see tombstoned stdout below)"
    } else {
      Write-Log "build_status=OK target=$($b.Target)"
    }
  }
}

$artifacts = Get-Artifacts
Write-Host ''
Write-Host ('{0,-10} {1,-8} {2,-10} {3}' -f 'artifact', 'size', 'exists', 'sha256')
foreach ($a in $artifacts) {
  Write-Host ('{0,-10} {1,-8} {2,-10} {3}' -f $a.Name, $a.Size, $a.Exists, $a.Sha256)
  if (-not $a.Exists) { Write-Log "artifact_missing=$($a.Path)" }
}
Write-Host ''

$deploySettings = $null
$deployResult = $null
$missingCount = @($artifacts | Where-Object { -not $_.Exists }).Count
if ($DeployAndRun) {
  if ($missingCount -eq 0) {
    $deploySettings = [pscustomobject]@{
      Domain           = 'CDSP_DOMAIN_ID=3'
      UnsignedPd       = 'DSPRPC_CONTROL_UNSIGNED_MODULE enabled (control=1)'
      CapabilityExpect = 'raw_arch=0x8081 (Hexagon V81 CDSP)'
      DeviceDir        = $DeviceDir
      DspLibraryPath   = $DspLibraryPath
      Q                = $Q
      Mode             = $Mode
      Retries          = $DeployRetries
    }
    $deployResult = Invoke-DeployAndRun $artifacts
    Write-Log "deploy_exit=$($deployResult.Exit) attempts=$($deployResult.Attempts)"
  } else {
    throw 'ARTIFACTS_INCOMPLETE: refusing to deploy (run without -SkipBuild so artifacts exist)'
  }
} else {
  Write-Log 'deploy=NOT_REQUESTED (no device contact; pass -DeployAndRun to deploy)'
}

$file = Write-ProvenanceFile -Git $gitInfo -BuildIdDefine $BuildIdDefine `
  -SdkInfo $sdkInfo -BuildResults $buildResults -Artifacts $artifacts `
  -DeploySettings $deploySettings -DeployResult $deployResult
Write-Log "provenance_file=$file"

$failed = (@($buildResults | Where-Object { $_.Exit -ne 0 }).Count -gt 0)
$failed = $failed -or (@($artifacts | Where-Object { -not $_.Exists }).Count -gt 0)
if ($DeployAndRun -and $deployResult -and $deployResult.Exit -ne 0) { $failed = $true }
if ($failed) { exit 1 }
exit 0
