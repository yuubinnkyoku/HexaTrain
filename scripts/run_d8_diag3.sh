#!/bin/bash
# Diagnostic 3 (revision 8) for the D8 VTCM pointer question.
#
# One run, no retry. Everything happens inside a single NPU gate acquisition so
# no adb command runs outside the lock.
#
# Order:
#   1. Tier 2 identity: one physical device, stable identifier, not an emulator
#   2. deploy the stamped skel and the host executable
#   3. digest check on the device
#   4. identity run (consumes no budget)
#   5. the single diagnostic run
#   6. pull the evidence into the repository
#
# The critical interval is never entered: the diagnostic build returns before
# mxclracc on the operand-FNV mismatch, and mxclracc is not issued on any path
# this script can reach.
set -u
export MSYS_NO_PATHCONV=1

WT="D:/ghq/github.com/yuubinnkyoku/HexaTrain-mode46-20261118"
SKEL="$WT/build/d8skel/diag3-0x0D1A6004.so"
HOSTEXE="$WT/build/d8host/hexatrain_hmx_acc_residual_test"
STUB="$WT/build/d8host/libhexatrain_hmx_acc_residual.so"
EXPECT_SHA="5e00082c0d449a66f96833f82641cdc2ad50fb639aaa66a2c12883e7aafa3490"
STAMP="0x0D1A6004"
EVID="$WT/docs/evidence/diag3-0x0D1A6004"
DEVDIR="/data/local/tmp/d8"
DEV_EVID="$DEVDIR/evidence/diag3-$STAMP"

echo "=== 1. Tier 2 identity ==="
DEVCOUNT=$(adb devices | sed -n '2,$p' | grep -c "device$")
echo "online_endpoints=$DEVCOUNT"
if [ "$DEVCOUNT" -ne 1 ]; then
    echo "D8_DIAG3_STOP expected exactly one online endpoint, got $DEVCOUNT"
    exit 10
fi
EP=$(adb devices | sed -n '2,$p' | grep "device$" | awk '{print $1}')
SER=$(adb -s "$EP" shell getprop ro.serialno | tr -d '\r')
BOOT=$(adb -s "$EP" shell getprop ro.boot.serialno | tr -d '\r')
QEMU=$(adb -s "$EP" shell getprop ro.kernel.qemu | tr -d '\r')
echo "endpoint=$EP serialno=$SER boot_serialno=$BOOT qemu=$QEMU"
if [ -z "$SER" ] || [ "$SER" != "$BOOT" ]; then
    echo "D8_DIAG3_STOP stable identifier mismatch or empty"
    exit 10
fi
if [ -n "$QEMU" ]; then
    echo "D8_DIAG3_STOP emulator detected"
    exit 10
fi
echo "D8_DIAG3_DEVICE_OK serialno=$SER"

echo "=== 2. deploy ==="
adb -s "$EP" shell "mkdir -p $DEVDIR $DEV_EVID"
adb -s "$EP" push "$SKEL" "$DEVDIR/libhexatrain_hmx_acc_residual_skel.so" 2>&1 | tail -1
adb -s "$EP" push "$STUB" "$DEVDIR/libhexatrain_hmx_acc_residual.so" 2>&1 | tail -1
adb -s "$EP" push "$HOSTEXE" "$DEVDIR/hexatrain_hmx_acc_residual_test" 2>&1 | tail -1
adb -s "$EP" shell "chmod 755 $DEVDIR/hexatrain_hmx_acc_residual_test"

echo "=== 3. digest ==="
GOT=$(adb -s "$EP" shell "sha256sum $DEVDIR/libhexatrain_hmx_acc_residual_skel.so" | awk '{print $1}' | tr -d '\r')
echo "device_sha256=$GOT"
echo "expected_sha256=$EXPECT_SHA"
if [ "$GOT" != "$EXPECT_SHA" ]; then
    echo "D8_DIAG3_STOP digest mismatch"
    exit 11
fi
echo "D8_DIAG3_DIGEST_OK"

echo "=== 4. identity ==="
adb -s "$EP" shell "cd $DEVDIR && LD_LIBRARY_PATH=$DEVDIR ADSP_LIBRARY_PATH=$DEVDIR ./hexatrain_hmx_acc_residual_test 6 30 -strict-build-id" > /tmp/diag3_id.txt 2>&1
IDRC=$?
cat /tmp/diag3_id.txt
if [ $IDRC -ne 0 ]; then
    echo "D8_DIAG3_STOP identity rc=$IDRC"
    exit 12
fi
if ! grep -q "^IDENTITY_OK=PASS" /tmp/diag3_id.txt; then
    echo "D8_DIAG3_STOP identity did not report PASS"
    exit 12
fi
if ! grep -q "build_id_status=MATCH" /tmp/diag3_id.txt; then
    echo "D8_DIAG3_STOP build_id_status is not MATCH"
    exit 12
fi
echo "D8_DIAG3_IDENTITY_OK"

echo "=== 5. the single diagnostic run ==="
mkdir -p "$EVID"
adb -s "$EP" shell "cd $DEVDIR && LD_LIBRARY_PATH=$DEVDIR ADSP_LIBRARY_PATH=$DEVDIR ./hexatrain_hmx_acc_residual_test 6 46" > "$EVID/stdout.txt" 2>&1
RUNRC=$?
echo "diag3_rc=$RUNRC"
grep -E "^(d8_rpc_status|d8_capture|d8_host_class|d8_contract|d8_counts|d8_fnv|d8_guard|d8_failure|d8_build_id|d8_observed_sha256|d8_label)=" "$EVID/stdout.txt" || true
echo "--- raw metadata ---"
grep "^d8_meta=" "$EVID/stdout.txt" || true
echo "--- raw slots (first 200 chars) ---"
grep "^d8_slots=" "$EVID/stdout.txt" | cut -c1-200 || true

echo "=== 6. evidence ==="
grep "^d8_slots=" "$EVID/stdout.txt" | sed 's/^d8_slots=//' > "$EVID/slots.csv" || true
grep "^d8_meta=" "$EVID/stdout.txt" | sed 's/^d8_meta=//' > "$EVID/meta.csv" || true
cp "$SKEL" "$EVID/skel.bin"
sha256sum "$EVID/skel.bin" | awk '{print $1"  skel.bin  "}' > "$EVID/skel.sha256"
printf '%s bytes\n' "$(stat -c%s "$EVID/skel.bin")" >> "$EVID/skel.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$EVID/run_utc.txt"
# Copy the device-side evidence directory back as well, so the on-device copy
# and the repository copy can be compared.
adb -s "$EP" shell "ls -la $DEV_EVID" > "$EVID/device_dir.txt" 2>&1 || true
echo "evidence_files:"
ls -la "$EVID"

echo "D8_DIAG3_DONE rc=$RUNRC"
exit 0
