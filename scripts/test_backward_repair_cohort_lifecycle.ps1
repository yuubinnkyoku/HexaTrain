# Host-only lifecycle audit for the G1 backward-repair formal cohort exit139.
#
# Diagnostic only. No device, no DSP arithmetic, no formal evidence, and no
# change to the frozen probe/DSP/Host binaries or to the formal cohort plan.
# The audit exercises the frozen Host graph-construction code against an
# instrumented mock QNN API and reports the QNN lifetime structure that the
# certified cohort plan creates inside one OS process.
param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out="$root/build/reports/g1-backward-repair-exit139-diagnosis"
[IO.Directory]::CreateDirectory($out)|Out-Null
$inc="$QairtSdkRoot/include/QNN"
$hostcc=(Get-Command g++).Source
if(!$hostcc){throw 'HOST_COMPILER_REQUIRED'}
$common=@('-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',"-I$inc")
$src="$root/host_tests/fp32_custom/backward_repair_cohort_lifecycle_test.cpp"
$exe="$out/cohort_lifecycle_audit.exe"
& $hostcc @common $src "$root/app/src/main/cpp/tiny_language_model_cpu.cpp" "$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp" -o $exe *> "$out/cohort_lifecycle_audit-build.txt"
if($LASTEXITCODE){throw 'HOST_BUILD cohort_lifecycle_audit'}
& $exe *> "$out/cohort_lifecycle_audit.txt"
if($LASTEXITCODE){throw 'HOST_GATE cohort_lifecycle_audit'}
Get-Content "$out/cohort_lifecycle_audit.txt" | Where-Object { $_ -match '^(cohort_lifecycle_audit|roles=|role=|graphs_per_role|per_role_lifetimes|htp_provider|log_handles|descriptor_faults|single_session_variant|backend_shared_variant|fault=)' } | ForEach-Object { Write-Host $_ }
Write-Host 'backward_repair_cohort_lifecycle_audit=PASS HOST_ONLY=1 NOT_FORMAL_EVIDENCE=1'
