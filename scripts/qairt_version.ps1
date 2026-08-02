# PhoneLM's pinned QAIRT configuration. Operational scripts must accept the
# values explicitly, then call Assert-PhoneLmQairtPinnedArguments before use.
$PhoneLmQairtSdkRoot = 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702'
$PhoneLmQairtBuildId = '2.48.40.260702151143'

function Assert-PhoneLmQairtPinnedArguments {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$SdkRoot,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$ExpectedBuildId
    )

    if ([string]::IsNullOrWhiteSpace($SdkRoot)) {
        throw 'QAIRT_SDK_ROOT_UNAVAILABLE: an explicit SDK root is required'
    }
    if ([string]::IsNullOrWhiteSpace($ExpectedBuildId)) {
        throw 'QAIRT_BUILD_ID_MISMATCH: an explicit expected Build ID is required'
    }

    $requestedRoot = [IO.Path]::GetFullPath($SdkRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    $pinnedRoot = [IO.Path]::GetFullPath($PhoneLmQairtSdkRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar)
    if (-not $requestedRoot.Equals($pinnedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "QAIRT_SDK_ROOT_MISMATCH: requested=$requestedRoot pinned=$pinnedRoot; fallback is forbidden"
    }
    if ($ExpectedBuildId -cne $PhoneLmQairtBuildId) {
        throw "QAIRT_BUILD_ID_MISMATCH: expected=$ExpectedBuildId pinned=$PhoneLmQairtBuildId"
    }
}
