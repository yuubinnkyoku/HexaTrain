param(
    [Parameter(Position = 0)]
    [string]$SdkRoot = "",
    # Optional QAIRT build ID (e.g. 2.48.40.260702151143). Verified against
    # sdk.yaml and include/QNN/QnnSdkBuildId.h inside the selected root only.
    [string]$ExpectedBuildId = "",
    # Run self-contained regression checks for the root-selection rules.
    [switch]$SelfTest,
    # Internal injection point used only by -SelfTest. It replaces every real
    # environment/property/common-path candidate so the test cannot inventory
    # an installed SDK.
    [Parameter(DontShow = $true)]
    [string]$SelfTestDiscoveryRoot = ""
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot

if (-not [string]::IsNullOrWhiteSpace($SelfTestDiscoveryRoot)) {
    if ([string]::IsNullOrWhiteSpace($env:PHONELM_QAIRT_SELFTEST_ROOT)) {
        throw "-SelfTestDiscoveryRoot is restricted to check_qairt.ps1 -SelfTest child processes"
    }
    $allowedRoot = [IO.Path]::GetFullPath($env:PHONELM_QAIRT_SELFTEST_ROOT).
        TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $injectedRoot = [IO.Path]::GetFullPath($SelfTestDiscoveryRoot)
    if (-not $injectedRoot.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "-SelfTestDiscoveryRoot must remain under the active self-test temp directory"
    }
}

# Exit codes: 2 = SDK not installed / explicit root missing,
# 3 = inventory incomplete, 4 = build ID mismatch, 1 = self-test failure.

function Write-Result([string]$Key, [object]$Value) {
    $text = if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        "NONE"
    } else {
        ([string]$Value -replace "[\r\n]+", " " -replace "\s+", " ").Trim()
    }
    Write-Output "$Key=$text"
}

function Join-Paths($Items) {
    $paths = @($Items | ForEach-Object {
        if ($_ -is [System.IO.FileSystemInfo]) { $_.FullName } else { [string]$_ }
    } | Where-Object { $_ } | Sort-Object -Unique)
    if ($paths.Count -eq 0) { return "NONE" }
    return $paths -join ";"
}

function Convert-GradlePath([string]$Value) {
    $result = $Value.Trim()
    $result = $result -replace '\\:', ':'
    $result = $result -replace '\\\\', '\'
    return $result
}

if ($SelfTest) {
    $pwshExe = Join-Path $PSHOME "pwsh.exe"
    $selfTestDir = Join-Path $env:TEMP ("phonelm-qairt-selftest-" + [Guid]::NewGuid().ToString("N"))
    $fakeRoot = Join-Path $selfTestDir "explicit\9.99.0.999999"
    $fakeDiscoveryRoot = Join-Path $selfTestDir "discovery\8.88.0.888888"
    $fakeVersion = "9.99.0"
    $fakeBuildNumber = "999999123456"
    $fakeBuildId = "$fakeVersion.$fakeBuildNumber"
    $selfFailures = [System.Collections.Generic.List[string]]::new()
    $previousSelfTestRoot = $env:PHONELM_QAIRT_SELFTEST_ROOT
    try {
        $env:PHONELM_QAIRT_SELFTEST_ROOT = $selfTestDir

        function New-FakeSdk([string]$Root, [string]$Version, [string]$BuildNumber) {
            $buildId = "$Version.$BuildNumber"
            New-Item -ItemType Directory -Force -Path (Join-Path $Root "include\QNN") | Out-Null
            Set-Content -LiteralPath (Join-Path $Root "include\QNN\QnnInterface.h") -Value "// selftest stub"
            Set-Content -LiteralPath (Join-Path $Root "include\QNN\QnnTypes.h") -Value "// selftest stub"
            Set-Content -LiteralPath (Join-Path $Root "include\QNN\QnnSdkBuildId.h") `
                -Value "#define QNN_SDK_BUILD_ID `"v$buildId`""
            Set-Content -LiteralPath (Join-Path $Root "sdk.yaml") `
                -Value "version: $Version`nbuild_id: $BuildNumber"
        }

        New-FakeSdk $fakeRoot $fakeVersion $fakeBuildNumber
        New-FakeSdk $fakeDiscoveryRoot "8.88.0" "888888123456"

        function Test-Case([string]$Name, [string[]]$Arguments, [scriptblock]$Assert) {
            $output = & $pwshExe -NoProfile -File $PSCommandPath @Arguments 2>&1 | Out-String
            $code = $LASTEXITCODE
            if (& $Assert $output $code) {
                Write-Result "selftest_$Name" "PASS"
            } else {
                Write-Result "selftest_$Name" "FAIL (exit=$code)"
                $selfFailures.Add($Name)
            }
        }

        Test-Case "explicit_root_beats_all_candidates" @(
            "-SdkRoot", $fakeRoot, "-SelfTestDiscoveryRoot", $fakeDiscoveryRoot) {
            param($out, $code)
            ($code -eq 3) -and
                ($out -match "sdk_root_source=argument") -and
                ($out -match [regex]::Escape("sdk_root=$fakeRoot")) -and
                ($out -notmatch [regex]::Escape($fakeDiscoveryRoot))
        }

        Test-Case "missing_explicit_root_fails_without_fallback" @(
            "-SdkRoot", (Join-Path $selfTestDir "does-not-exist"),
            "-SelfTestDiscoveryRoot", $fakeDiscoveryRoot) {
            param($out, $code)
            ($code -eq 2) -and ($out -match "BLOCKED_BY_QAIRT_SDK_NOT_INSTALLED") -and
                ($out -notmatch "sdk_root_source=") -and
                ($out -notmatch [regex]::Escape($fakeDiscoveryRoot))
        }

        Test-Case "no_argument_allows_discovery" @(
            "-SelfTestDiscoveryRoot", $fakeDiscoveryRoot) {
            param($out, $code)
            ($code -eq 3) -and
                ($out -match "sdk_root_source=common_path_scan") -and
                ($out -match [regex]::Escape("sdk_root=$fakeDiscoveryRoot"))
        }

        Test-Case "expected_build_id_match_passes" @(
            "-SdkRoot", $fakeRoot, "-ExpectedBuildId", $fakeBuildId,
            "-SelfTestDiscoveryRoot", $fakeDiscoveryRoot) {
            param($out, $code)
            ($code -eq 3) -and ($out -match "expected_build_id_match=true")
        }

        Test-Case "expected_build_id_mismatch_fails" @(
            "-SdkRoot", $fakeRoot, "-ExpectedBuildId", "0.0.0.000000000000",
            "-SelfTestDiscoveryRoot", $fakeDiscoveryRoot) {
            param($out, $code)
            ($code -eq 4) -and ($out -match "expected_build_id_match=false") -and
                ($out -match "QAIRT_BUILD_ID_MISMATCH")
        }
    } finally {
        $env:PHONELM_QAIRT_SELFTEST_ROOT = $previousSelfTestRoot
        Remove-Item -LiteralPath $selfTestDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($selfFailures.Count -gt 0) {
        Write-Result "selftest_failed_cases" ($selfFailures -join ";")
        exit 1
    }
    Write-Result "selftest_passed_cases" "5/5"
    exit 0
}

function Get-PropertyCandidates {
    $propertyFiles = @(
        (Join-Path $RepositoryRoot "local.properties"),
        (Join-Path $RepositoryRoot "gradle.properties"),
        (Join-Path $HOME ".gradle\gradle.properties")
    )
    foreach ($file in $propertyFiles) {
        if (-not (Test-Path -LiteralPath $file)) { continue }
        foreach ($line in Get-Content -LiteralPath $file -ErrorAction SilentlyContinue) {
            if ($line -match '^\s*qairt\.sdkRoot\s*=\s*(.+?)\s*$') {
                [pscustomobject]@{ Path = (Convert-GradlePath $Matches[1]); Source = $file }
            }
        }
    }
}

function Test-Aarch64Elf([System.IO.FileInfo]$File) {
    try {
        $stream = [System.IO.File]::OpenRead($File.FullName)
        try {
            $bytes = New-Object byte[] 20
            if ($stream.Read($bytes, 0, $bytes.Length) -ne $bytes.Length) { return $false }
            if ($bytes[0] -ne 0x7f -or $bytes[1] -ne 0x45 -or
                $bytes[2] -ne 0x4c -or $bytes[3] -ne 0x46) { return $false }
            $machine = if ($bytes[5] -eq 2) {
                ($bytes[18] -shl 8) -bor $bytes[19]
            } else {
                $bytes[18] -bor ($bytes[19] -shl 8)
            }
            return $machine -eq 183
        } finally {
            $stream.Dispose()
        }
    } catch {
        return $false
    }
}

function Get-VersionTriplets($HeaderFiles, [string]$PrefixPattern) {
    $macros = @{}
    foreach ($file in $HeaderFiles) {
        foreach ($line in Get-Content -LiteralPath $file.FullName -ErrorAction SilentlyContinue) {
            if ($line -match '^\s*#\s*define\s+([A-Za-z0-9_]*VERSION)_(MAJOR|MINOR|PATCH)\s+\(?([0-9]+)') {
                $prefix = $Matches[1]
                $part = $Matches[2]
                $value = $Matches[3]
                if ($prefix -notmatch $PrefixPattern) { continue }
                if (-not $macros.ContainsKey($prefix)) { $macros[$prefix] = @{} }
                $macros[$prefix][$part] = $value
            }
        }
    }
    $versions = foreach ($prefix in $macros.Keys) {
        $parts = $macros[$prefix]
        if ($parts.ContainsKey("MAJOR") -and $parts.ContainsKey("MINOR")) {
            $version = "$($parts['MAJOR']).$($parts['MINOR'])"
            if ($parts.ContainsKey("PATCH")) { $version += ".$($parts['PATCH'])" }
            "${prefix}:$version"
        }
    }
    return @($versions | Sort-Object -Unique)
}

if (-not [string]::IsNullOrWhiteSpace($SdkRoot)) {
    # An explicit -SdkRoot wins exclusively: no env/properties/common-path
    # fallback, so a different installed version is never silently selected.
    $candidates = @([pscustomobject]@{ Path = $SdkRoot; Source = "argument" })
} elseif (-not [string]::IsNullOrWhiteSpace($SelfTestDiscoveryRoot)) {
    $candidates = @(
        [pscustomobject]@{ Path = $SelfTestDiscoveryRoot; Source = "common_path_scan" }
    )
} else {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:QAIRT_SDK_ROOT)) {
        $candidates += [pscustomobject]@{ Path = $env:QAIRT_SDK_ROOT; Source = "QAIRT_SDK_ROOT" }
    }
    $candidates += @(Get-PropertyCandidates)

    $commonRoots = @(
        "C:\Qualcomm",
        "C:\Program Files\Qualcomm",
        "C:\Program Files (x86)\Qualcomm",
        (Join-Path $HOME "Qualcomm"),
        (Join-Path $HOME "qairt"),
        (Join-Path $HOME ".qairt"),
        (Join-Path $env:LOCALAPPDATA "Qualcomm")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    foreach ($commonRoot in $commonRoots) {
        $header = Get-ChildItem -LiteralPath $commonRoot -Recurse -File -Filter "QnnInterface.h" `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($header) {
            $cursor = $header.Directory
            while ($cursor.Parent -and $cursor.Parent.FullName.StartsWith($commonRoot,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                if (Test-Path -LiteralPath (Join-Path $cursor.FullName "include")) { break }
                $cursor = $cursor.Parent
            }
            $candidates += [pscustomobject]@{ Path = $cursor.FullName; Source = "common_path_scan" }
        }
    }

    $candidates = @($candidates | Where-Object { $_.Path } |
        Group-Object { [System.IO.Path]::GetFullPath($_.Path) } | ForEach-Object { $_.Group[0] })
}

$selected = $candidates | Where-Object { Test-Path -LiteralPath $_.Path -PathType Container } |
    Where-Object {
        Get-ChildItem -LiteralPath $_.Path -Recurse -File -Filter "QnnInterface.h" `
            -ErrorAction SilentlyContinue | Select-Object -First 1
    } | Select-Object -First 1
if (-not $selected) { $selected = $candidates | Select-Object -First 1 }

Write-Result "check" "QAIRT_SDK_INVENTORY"
Write-Result "requested_sdk_root" $SdkRoot
Write-Result "candidate_sources" (($candidates | ForEach-Object { "$($_.Source):$($_.Path)" }) -join ";")

if (-not $selected -or -not (Test-Path -LiteralPath $selected.Path -PathType Container)) {
    Write-Result "sdk_root_exists" "false"
    Write-Result "qnn_interface_header_exists" "false"
    Write-Result "qnn_types_header_exists" "false"
    Write-Result "qnn_implementation_ready" "false"
    Write-Result "status" "BLOCKED_BY_QAIRT_SDK_NOT_INSTALLED"
    exit 2
}

$resolvedRoot = (Resolve-Path -LiteralPath $selected.Path).Path

if (-not [string]::IsNullOrWhiteSpace($SdkRoot)) {
    $requestedRoot = [System.IO.Path]::GetFullPath($SdkRoot)
    $resolvedSelected = [System.IO.Path]::GetFullPath($resolvedRoot)
    if ($requestedRoot -ne $resolvedSelected) {
        throw "Explicit QAIRT SDK root was not honored: requested=$requestedRoot resolved=$resolvedSelected"
    }
}

if (-not [string]::IsNullOrWhiteSpace($ExpectedBuildId)) {
    $buildIdMatch = $false
    $sdkYamlPath = Join-Path $resolvedRoot "sdk.yaml"
    $buildIdHeaderPath = Join-Path $resolvedRoot "include\QNN\QnnSdkBuildId.h"
    $resolvedBuildId = "N/A"
    if ((Test-Path -LiteralPath $sdkYamlPath) -and (Test-Path -LiteralPath $buildIdHeaderPath)) {
        $sdkYaml = Get-Content -LiteralPath $sdkYamlPath -Raw
        $buildIdHeader = Get-Content -LiteralPath $buildIdHeaderPath -Raw
        $yamlVersion = [regex]::Match($sdkYaml, '(?m)^version:\s*(\S+)').Groups[1].Value
        $yamlBuild = [regex]::Match($sdkYaml, '(?m)^build_id:\s*(\S+)').Groups[1].Value
        $resolvedBuildId = [regex]::Match($buildIdHeader, 'QNN_SDK_BUILD_ID\s+"v([^"]+)"').Groups[1].Value
        $buildIdMatch = ($resolvedBuildId -eq "$yamlVersion.$yamlBuild") -and
            ($resolvedBuildId -eq $ExpectedBuildId)
    }
    Write-Result "expected_build_id" $ExpectedBuildId
    Write-Result "resolved_build_id" $resolvedBuildId
    Write-Result "expected_build_id_match" $buildIdMatch.ToString().ToLowerInvariant()
    if (-not $buildIdMatch) {
        Write-Result "status" "QAIRT_BUILD_ID_MISMATCH"
        exit 4
    }
}

$allFiles = @(Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Force `
    -ErrorAction SilentlyContinue)
$interfaceHeaders = @($allFiles | Where-Object { $_.Name -ceq "QnnInterface.h" })
$typesHeaders = @($allFiles | Where-Object { $_.Name -ceq "QnnTypes.h" })
$qnnHeaders = @($allFiles | Where-Object {
    $_.Extension -eq ".h" -and $_.Name -match '^(Qnn|Qairt)'
})
$apiVersions = @(Get-VersionTriplets $qnnHeaders '(?i)(QNN|QAIRT).*(API|INTERFACE)|QNN')
$sdkVersions = @(Get-VersionTriplets $qnnHeaders '(?i)(QAIRT|QNN).*SDK')
if ($sdkVersions.Count -eq 0 -and (Split-Path -Leaf $resolvedRoot) -match '(\d+\.\d+(?:\.\d+)?)') {
    $sdkVersions = @("root_directory:$($Matches[1])")
}

$sharedObjects = @($allFiles | Where-Object { $_.Extension -eq ".so" })
$aarch64Objects = @($sharedObjects | Where-Object { Test-Aarch64Elf $_ })
$androidArm64Objects = @($aarch64Objects | Where-Object {
    $_.FullName -match '(?i)(android.*(?:aarch64|arm64)|(?:aarch64|arm64).*android)'
})
$cpuBackendCandidates = @($androidArm64Objects | Where-Object {
    $_.FullName -match '(?i)(qnn|qairt).*cpu|cpu.*(qnn|qairt)'
})
$htpBackendCandidates = @($androidArm64Objects | Where-Object {
    $_.FullName -match '(?i)(qnn|qairt).*htp|htp.*(qnn|qairt)'
})
$htpPrepareCandidates = @($sharedObjects | Where-Object {
    $_.FullName -match '(?i)(htp.*(prepare|prep)|(prepare|prep).*htp)'
})
$dspSkelCandidates = @($sharedObjects | Where-Object {
    $_.FullName -match '(?i)(dsp|hexagon|htp)' -and $_.Name -match '(?i)skel'
})
$dspStubCandidates = @($sharedObjects | Where-Object {
    $_.FullName -match '(?i)(dsp|hexagon|htp)' -and $_.Name -match '(?i)stub'
})
$netRunTools = @($allFiles | Where-Object { $_.Name -match '^qnn-net-run(?:\..*)?$' })
$validatorTools = @($allFiles | Where-Object {
    $_.Name -match '^qnn-platform-validator(?:\..*)?$'
})

$sampleFiles = @($allFiles | Where-Object {
    $_.FullName -match '(?i)[\\/](sample|samples|example|examples)[\\/]' -and
    $_.Extension -match '^\.(c|cc|cpp|cxx|h|hpp|cmake|txt|md|json|xml)$'
})
$sampleEvidence = @()
$cpuSampleEvidence = @()
$htpSampleEvidence = @()
foreach ($file in $sampleFiles) {
    if ($file.Length -gt 5MB) { continue }
    if (Select-String -LiteralPath $file.FullName -Pattern 'QnnInterface\.h' -Quiet `
            -ErrorAction SilentlyContinue) {
        $sampleEvidence += $file
        if ($file.FullName -match '(?i)cpu' -or
            (Select-String -LiteralPath $file.FullName -Pattern '(?i)\bCPU\b|QnnCpu' `
                -Quiet -ErrorAction SilentlyContinue)) {
            $cpuSampleEvidence += $file
        }
        if ($file.FullName -match '(?i)htp' -or
            (Select-String -LiteralPath $file.FullName -Pattern '(?i)\bHTP\b|QnnHtp' `
                -Quiet -ErrorAction SilentlyContinue)) {
            $htpSampleEvidence += $file
        }
    }
}

Write-Result "sdk_root_exists" "true"
Write-Result "sdk_root" $resolvedRoot
Write-Result "sdk_root_source" $selected.Source
Write-Result "qnn_interface_header_exists" ($interfaceHeaders.Count -gt 0).ToString().ToLowerInvariant()
Write-Result "qnn_interface_headers" (Join-Paths $interfaceHeaders)
Write-Result "qnn_types_header_exists" ($typesHeaders.Count -gt 0).ToString().ToLowerInvariant()
Write-Result "qnn_types_headers" (Join-Paths $typesHeaders)
Write-Result "include_directories" (Join-Paths (($interfaceHeaders + $typesHeaders) | ForEach-Object { $_.Directory }))
Write-Result "qairt_sdk_version" $(if ($sdkVersions.Count) { $sdkVersions -join ";" } else { "UNDETERMINED" })
Write-Result "qnn_api_version" $(if ($apiVersions.Count) { $apiVersions -join ";" } else { "UNDETERMINED" })
Write-Result "android_arm64_library_directories" (Join-Paths ($androidArm64Objects | ForEach-Object { $_.Directory }))
Write-Result "android_arm64_libraries" (Join-Paths $androidArm64Objects)
Write-Result "cpu_backend_library_candidates" (Join-Paths $cpuBackendCandidates)
Write-Result "htp_backend_library_candidates" (Join-Paths $htpBackendCandidates)
Write-Result "htp_runtime_library_directories" (Join-Paths ($htpBackendCandidates | ForEach-Object { $_.Directory }))
Write-Result "htp_prepare_or_equivalent_candidates" (Join-Paths $htpPrepareCandidates)
Write-Result "dsp_skel_candidates" (Join-Paths $dspSkelCandidates)
Write-Result "dsp_stub_candidates" (Join-Paths $dspStubCandidates)
Write-Result "dsp_library_directories" (Join-Paths (($dspSkelCandidates + $dspStubCandidates) | ForEach-Object { $_.Directory }))
Write-Result "qnn_net_run" (Join-Paths $netRunTools)
Write-Result "qnn_platform_validator" (Join-Paths $validatorTools)
Write-Result "official_sample_candidates" (Join-Paths $sampleEvidence)
Write-Result "official_sample_directories" (Join-Paths ($sampleEvidence | ForEach-Object { $_.Directory }))
Write-Result "cpu_sample_candidates" (Join-Paths $cpuSampleEvidence)
Write-Result "htp_sample_candidates" (Join-Paths $htpSampleEvidence)
Write-Result "classification_note" "Candidate roles are inferred from installed paths/names and must be confirmed against that SDK's official build files; no library basename is hard-coded."
Write-Result "qnn_implementation_ready" "false"

$inventoryComplete = $interfaceHeaders.Count -gt 0 -and $typesHeaders.Count -gt 0 -and
    $sdkVersions.Count -gt 0 -and $apiVersions.Count -gt 0 -and
    $androidArm64Objects.Count -gt 0 -and
    $cpuBackendCandidates.Count -gt 0 -and $htpBackendCandidates.Count -gt 0 -and
    $htpPrepareCandidates.Count -gt 0 -and
    $dspSkelCandidates.Count -gt 0 -and $dspStubCandidates.Count -gt 0 -and
    $netRunTools.Count -gt 0 -and $validatorTools.Count -gt 0 -and
    $sampleEvidence.Count -gt 0 -and $cpuSampleEvidence.Count -gt 0 -and
    $htpSampleEvidence.Count -gt 0
if ($inventoryComplete) {
    Write-Result "status" "QAIRT_SDK_INVENTORY_COMPLETE_ADAPTER_NOT_IMPLEMENTED"
    exit 0
}

Write-Result "status" "QAIRT_SDK_FOUND_INVENTORY_INCOMPLETE"
exit 3
