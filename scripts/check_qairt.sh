#!/usr/bin/env sh

# Compatibility entry point. QAIRT selection and inventory policy lives in
# check_qairt.ps1 and qairt_version.ps1; this wrapper must not rediscover or
# reinterpret SDK roots independently.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v pwsh >/dev/null 2>&1; then
    echo "check=QAIRT_SDK_INVENTORY"
    echo "status=QAIRT_POWERSHELL_REQUIRED"
    echo "error=pwsh is required to execute scripts/check_qairt.ps1"
    exit 2
fi

if [ "${1:-}" = "--sdk-root" ]; then
    if [ "$#" -lt 2 ]; then
        echo "error=--sdk-root requires a path"
        exit 64
    fi
    SDK_ROOT=$2
    shift 2
    set -- -SdkRoot "$SDK_ROOT" "$@"
fi

exec pwsh -NoLogo -NoProfile -File "$SCRIPT_DIR/check_qairt.ps1" "$@"
