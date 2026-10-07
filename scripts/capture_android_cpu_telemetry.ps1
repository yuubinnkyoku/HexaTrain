# SPDX-License-Identifier: Apache-2.0
# One-shot, best-effort Android CPU/thermal snapshot for private device audits.
param(
  [Parameter(Mandatory=$true)][string]$AdbPath,
  [Parameter(Mandatory=$true)][string]$Device,
  [Parameter(Mandatory=$true)][string]$Package,
  [Parameter(Mandatory=$true)][ValidateSet('pre','mid','post')][string]$Phase,
  [Parameter(Mandatory=$true)][string]$OutputPath,
  [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9._-]{1,64}$')][string]$RunId
)
$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1'
. $common
$root = Split-Path -Parent $PSScriptRoot
$output = [IO.Path]::GetFullPath($OutputPath)
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $output.StartsWith($buildRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'TELEMETRY_OUTPUT_MUST_BE_UNDER_BUILD' }
if (-not (Test-Path -LiteralPath $AdbPath -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }

# Keep this to one shell round-trip per timing point. Individual sysfs or
# dumpsys permission failures become NOT_AVAILABLE and do not abort collection.
$snapshotScript = @'
na=NOT_AVAILABLE
emit_file() { key="$1"; path="$2"; value=$(cat "$path" 2>/dev/null); [ -n "$value" ] || value=$na; value=$(printf '%s' "$value" | tr '\r\n=' '   '); printf '%s=%s\n' "$key" "$value"; }
emit_command() { key="$1"; shift; value=$("$@" 2>/dev/null); [ -n "$value" ] || value=$na; value=$(printf '%s' "$value" | tr '\r\n=' '   '); printf '%s=%s\n' "$key" "$value"; }
emit_file online_cores /sys/devices/system/cpu/online
for d in /sys/devices/system/cpu/cpu[0-9]*; do
  [ -d "$d" ] || continue
  c=${d##*/}
  emit_file "${c}_online" "$d/online"
  emit_file "${c}_cur_freq_khz" "$d/cpufreq/scaling_cur_freq"
  emit_file "${c}_min_freq_khz" "$d/cpufreq/scaling_min_freq"
  emit_file "${c}_max_freq_khz" "$d/cpufreq/scaling_max_freq"
  emit_file "${c}_governor" "$d/cpufreq/scaling_governor"
done
emit_file cpuset_top_app_cpus /dev/cpuset/top-app/cpus
emit_file cpuset_foreground_cpus /dev/cpuset/foreground/cpus
emit_command uptime_seconds awk '{print $1}' /proc/uptime
emit_file loadavg /proc/loadavg
thermal=$(dumpsys thermalservice 2>/dev/null | grep -i -m 1 -E 'thermal status|current.*status' | tr '\r\n=' '   ')
[ -n "$thermal" ] || thermal=$na
printf 'thermal_status=%s\n' "$thermal"
battery=$(dumpsys battery 2>/dev/null)
bt=$(printf '%s\n' "$battery" | sed -n 's/^[[:space:]]*temperature:[[:space:]]*//p' | head -1)
[ -n "$bt" ] || bt=$na
printf 'battery_temperature_deci_c=%s\n' "$bt"
emit_command battery_saver settings get global low_power
power=$(dumpsys power 2>/dev/null | grep -i -m 1 -E 'mInteractive=|mWakefulness=|Display Power: state=' | tr '\r\n=' '   ')
[ -n "$power" ] || power=$na
printf 'screen_state=%s\n' "$power"
foreground=$(dumpsys activity activities 2>/dev/null | grep -i -m 1 -E 'topResumedActivity|ResumedActivity' | tr '\r\n=' '   ')
[ -n "$foreground" ] || foreground=$na
printf 'foreground_activity=%s\n' "$foreground"
if [ "$foreground" = "$na" ]; then app_foreground=$na
elif printf '%s' "$foreground" | grep -F "$PACKAGE" >/dev/null 2>&1; then app_foreground=FOREGROUND
else app_foreground=BACKGROUND
fi
printf 'app_foreground_state=%s\n' "$app_foreground"
pid=$(pidof "$PACKAGE" 2>/dev/null | awk '{print $1}')
[ -n "$pid" ] || pid=$na
printf 'process_pid=%s\n' "$pid"
if [ "$pid" != "$na" ]; then
  emit_file process_status "/proc/$pid/status"
  emit_file process_sched "/proc/$pid/sched"
  emit_file process_cgroup "/proc/$pid/cgroup"
else
  printf 'process_status=%s\nprocess_sched=%s\nprocess_cgroup=%s\n' "$na" "$na" "$na"
fi
'@
$snapshotScript = $snapshotScript.Replace('$PACKAGE', $Package)
$adbResult = Get-PhoneLmAdbResult -Adb $AdbPath -Device $Device -Arguments @('shell','sh','-c',$snapshotScript) -TimeoutSeconds 60
if ($adbResult.ExitCode -ne 0) { throw 'ADB_TELEMETRY_TRANSPORT_FAILURE' }
$result = $adbResult.Output
$fields = [ordered]@{}
foreach ($line in $result) {
  $text = [string]$line
  $separator = $text.IndexOf('=')
  if ($separator -gt 0) { $fields[$text.Substring(0, $separator)] = $text.Substring($separator + 1).Trim() }
}
$fields = [ordered]@{ schema_version = 1; run_id = $RunId; phase = $Phase; captured_utc = [DateTimeOffset]::UtcNow.ToString('o'); fields = $fields }
[IO.Directory]::CreateDirectory((Split-Path -Parent $output)) | Out-Null
[IO.File]::WriteAllText($output, ($fields | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Write-Output $output
