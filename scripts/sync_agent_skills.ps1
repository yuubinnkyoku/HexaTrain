# Sync AI skill trees: .agents/skills is SSOT; .commandcode/skills is generated.
# Edit skills only under .agents/skills. Do not hand-edit .commandcode/skills.
#
#   .\scripts\sync_agent_skills.ps1          # mirror source -> target
#   .\scripts\sync_agent_skills.ps1 -Check   # fail if target is stale/extra/missing
param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root ".agents\skills"
$Target = Join-Path $Root ".commandcode\skills"

if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "SOURCE_MISSING: $Source"
}

function Get-RelativeSkillFiles([string]$Base) {
    if (-not (Test-Path -LiteralPath $Base -PathType Container)) {
        return @{}
    }
    $map = [System.Collections.Generic.Dictionary[string, string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    $baseFull = [IO.Path]::GetFullPath($Base)
    foreach ($file in Get-ChildItem -LiteralPath $Base -Recurse -File -Force) {
        $full = $file.FullName
        # Do not walk through reparse points as extra copies; replace them on generate.
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        $rel = $full.Substring($baseFull.Length).TrimStart('\', '/') -replace '\\', '/'
        $map[$rel] = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
    }
    return $map
}

function Remove-TargetReparsePoint([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        # Directory junction / symlink: remove the link only, never the source.
        [IO.Directory]::Delete($Path)
    }
}

$sourceFiles = Get-RelativeSkillFiles $Source
$targetFiles = Get-RelativeSkillFiles $Target

$missing = @($sourceFiles.Keys | Where-Object { -not $targetFiles.ContainsKey($_) } | Sort-Object)
$stale = @($sourceFiles.Keys | Where-Object {
        $targetFiles.ContainsKey($_) -and $targetFiles[$_] -ne $sourceFiles[$_]
    } | Sort-Object)
$extra = @($targetFiles.Keys | Where-Object { -not $sourceFiles.ContainsKey($_) } | Sort-Object)

# Also treat a junction at the skill root as out-of-sync even if hashes match
# through the link, so generate materializes real files.
$reparseRoots = @()
foreach ($dir in Get-ChildItem -LiteralPath $Target -Force -ErrorAction SilentlyContinue) {
    if ($dir.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        $reparseRoots += $dir.Name
    }
}

if ($Check) {
    $problems = @()
    if ($missing.Count -gt 0) { $problems += "missing=$($missing.Count)" }
    if ($stale.Count -gt 0) { $problems += "stale=$($stale.Count)" }
    if ($extra.Count -gt 0) { $problems += "extra=$($extra.Count)" }
    if ($reparseRoots.Count -gt 0) { $problems += "reparse=$($reparseRoots -join ',')" }
    if ($problems.Count -gt 0) {
        Write-Host "agent-skills-sync=STALE ($($problems -join ' '))"
        foreach ($rel in ($missing | Select-Object -First 10)) { Write-Host "  missing $rel" }
        foreach ($rel in ($stale | Select-Object -First 10)) { Write-Host "  stale $rel" }
        foreach ($rel in ($extra | Select-Object -First 10)) { Write-Host "  extra $rel" }
        foreach ($name in $reparseRoots) { Write-Host "  reparse $name" }
        throw "AGENT_SKILLS_STALE: run scripts/sync_agent_skills.ps1"
    }
    Write-Host "agent-skills-sync=PASS files=$($sourceFiles.Count)"
    exit 0
}

# Generate: materialize a real directory tree (replace reparse-point skill roots).
foreach ($name in $reparseRoots) {
    Remove-TargetReparsePoint (Join-Path $Target $name)
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null

foreach ($rel in $sourceFiles.Keys) {
    $src = Join-Path $Source ($rel -replace '/', '\')
    $dst = Join-Path $Target ($rel -replace '/', '\')
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dst -Force
}

foreach ($rel in $extra) {
    $path = Join-Path $Target ($rel -replace '/', '\')
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

# Drop empty directories left after extra removal.
foreach ($dir in Get-ChildItem -LiteralPath $Target -Recurse -Directory -Force | Sort-Object FullName -Descending) {
    if (-not (Get-ChildItem -LiteralPath $dir.FullName -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $dir.FullName -Force
    }
}

Write-Host "agent-skills-sync=UPDATED source=$($sourceFiles.Count) missing_fixed=$($missing.Count) stale_fixed=$($stale.Count) extra_removed=$($extra.Count) reparse_replaced=$($reparseRoots.Count)"
exit 0
