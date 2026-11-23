#!/bin/bash
# Diagnostic 2 for the D8 operand-FNV gate.
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
set -u
export MSYS_NO_PATHCONV=1

WT="D:/ghq/github.com/yuubinnkyoku/HexaTrain-mode46-20261118"
SKEL="$WT/build/d8skel/diag2.so"
HOSTEXE="$WT/build/d8host/hexatrain_hmx_acc_residual_test"
EXPECT_SHA="e0a4b4d9d5bc230c038f20813160df664c5c2dab590246de6ded7849c7146d8d"
STAMP="0x0D1A6002"
EVID="$WT/docs/evidence/diag2-0x0D1A6002"

echo "=== 1. Tier 2 identity ==="
DEVCOUNT=$(adb devices | sed -n '2,$p' | grep -c "device$")
echo "online_endpoints=$DEVCOUNT"
if [ "$DEVCOUNT" -ne 1 ]; then
    echo "D8_DIAG2_STOP expected exactly one online endpoint, got $DEVCOUNT"
    exit 10
fi
EP=$(adb devices | sed -n '2,$p' | grep "device$" | awk '{print $1}')
SER=$(adb -s "$EP" shell getprop ro.serialno | tr -d '\r')
BOOT=$(adb -s "$EP" shell getprop ro.boot.serialno | tr -d '\r')
QEMU=$(adb -s "$EP" shell getprop ro.kernel.qemu | tr -d '\r')
echo "endpoint=$EP serialno=$SER boot_serialno=$BOOT qemu=$QEMU"
if [ -z "$SER" ] || [ "$SER" != "$BOOT" ]; then
    echo "D8_DIAG2_STOP stable identifier mismatch or empty"
    exit 10
fi
if [ -n "$QEMU" ]; then
    echo "D8_DIAG2_STOP emulator detected"
    exit 10
fi
echo "D8_DIAG2_DEVICE_OK serialno=$SER"

echo "=== 2. deploy ==="
adb -s "$EP" push "$SKEL" /data/local/tmp/d8/libhexatrain_hmx_acc_residual_skel.so 2>&1 | tail -1
adb -s "$EP" push "$HOSTEXE" /data/local/tmp/d8/hexatrain_hmx_acc_residual_test 2>&1 | tail -1
adb -s "$EP" shell "chmod 755 /data/local/tmp/d8/hexatrain_hmx_acc_residual_test"

echo "=== 3. digest ==="
GOT=$(adb -s "$EP" shell "sha256sum /data/local/tmp/d8/libhexatrain_hmx_acc_residual_skel.so" | awk '{print $1}' | tr -d '\r')
echo "device_sha256=$GOT"
echo "expected_sha256=$EXPECT_SHA"
if [ "$GOT" != "$EXPECT_SHA" ]; then
    echo "D8_DIAG2_STOP digest mismatch"
    exit 11
fi
echo "D8_DIAG2_DIGEST_OK"

echo "=== 4. identity ==="
adb -s "$EP" shell "cd /data/local/tmp/d8 && LD_LIBRARY_PATH=/data/local/tmp/d8 ADSP_LIBRARY_PATH=/data/local/tmp/d8 ./hexatrain_hmx_acc_residual_test 6 30 -strict-build-id" > /tmp/diag2_id.txt 2>&1
IDRC=$?
cat /tmp/diag2_id.txt
if [ $IDRC -ne 0 ]; then
    echo "D8_DIAG2_STOP identity rc=$IDRC"
    exit 12
fi
if ! grep -q "^IDENTITY_OK=PASS" /tmp/diag2_id.txt; then
    echo "D8_DIAG2_STOP identity did not report PASS"
    exit 12
fi
echo "D8_DIAG2_IDENTITY_OK"

echo "=== 5. the single diagnostic run ==="
mkdir -p "$EVID"
adb -s "$EP" shell "mkdir -p /data/local/tmp/d8/evidence/diag2-$STAMP"
adb -s "$EP" shell "cd /data/local/tmp/d8 && LD_LIBRARY_PATH=/data/local/tmp/d8 ADSP_LIBRARY_PATH=/data/local/tmp/d8 ./hexatrain_hmx_acc_residual_test 6 46" > "$EVID/stdout.txt" 2>&1
RUNRC=$?
echo "diag2_rc=$RUNRC"
grep -E "^(d8_rpc_status|d8_capture|d8_host_class|d8_contract|d8_counts|d8_fnv|d8_guard|d8_failure|d8_build_id|d8_observed_sha256|d8_label)=" "$EVID/stdout.txt" || true
echo "--- raw metadata ---"
grep "^d8_meta=" "$EVID/stdout.txt" || true
echo "--- raw slots (first 200 chars) ---"
grep "^d8_slots=" "$EVID/stdout.txt" | cut -c1-200 || true

echo "=== 6. evidence ==="
grep "^d8_slots=" "$EVID/stdout.txt" | sed 's/^d8_slots=//' > "$EVID/slots.csv" || true
grep "^d8_meta=" "$EVID/stdout.txt" | sed 's/^d8_meta=//' > "$EVID/meta.csv" || true
cp "$SKEL" "$EVID/skel.so"
sha256sum "$EVID/skel.so" > "$EVID/skel.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$EVID/run_utc.txt"
echo "evidence_files:"
ls -la "$EVID"

echo "D8_DIAG2_DONE rc=$RUNRC"
exit 0
