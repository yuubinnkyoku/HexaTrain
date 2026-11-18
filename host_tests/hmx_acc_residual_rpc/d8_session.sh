#!/system/bin/sh
# D8 Mode 46 device session harness.
#
#   sh d8_session.sh <session-id>
#
# Runs the registered budget exactly once: one identity call, then the four
# condition runs A(q=6) -> B(q=7) -> C(q=8) -> D(q=9), each once, no retry.
#
# Evidence preservation (prereg section 8, fastrpc failure contract):
#   - every run's full stdout is written to its own file before anything else;
#   - the raw 1,024-slot array and the 32-word metadata are captured from the
#     d8_slots= / d8_meta= lines the host prints, verbatim;
#   - the observed SHA-256, the build id, the identity echo and the timestamps
#     are recorded next to them;
#   - a run whose exit code is non-zero is recorded as such and the session
#     stops, because the budget has no retry.
#
# The stop conditions of prereg section 10 are enforced by the host's own
# contract gate; this script only refuses to continue when the host says the
# run did not pass, and it never re-runs anything.
set -u

BASE=/data/local/tmp/d8
SID=${1:-session}
OUT=$BASE/evidence/$SID
mkdir -p "$OUT"

export LD_LIBRARY_PATH=$BASE
export ADSP_LIBRARY_PATH=$BASE

# The vendor libcdsprpc is the one that matches the ABI. A copy in $BASE makes
# the DSP unreachable (capability_status=-1), so refuse to start if one exists.
if [ -f "$BASE/libcdsprpc.so" ]; then
    echo "D8_REFUSING libcdsprpc.so present in $BASE; remove it (vendor copy is required)"
    exit 2
fi

BIN=$BASE/hexatrain_hmx_acc_residual_test
[ -x "$BIN" ] || { echo "D8_REFUSING missing $BIN"; exit 2; }

SKEL_SHA=$(sha256sum "$BASE/libhexatrain_hmx_acc_residual_skel.so" | cut -d' ' -f1)
HOST_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
echo "D8_SESSION id=$SID skel_sha256=$SKEL_SHA host_sha256=$HOST_SHA"
echo "D8_SESSION started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) device=$(getprop ro.product.model) build=$(getprop ro.build.display.id)"

# ---- identity gate (mode 30, q=6; never the experiment q) -------------------
ID_OUT=$OUT/00-identity.txt
"$BIN" 6 30 -strict-build-id > "$ID_OUT" 2>&1
ID_RC=$?
echo "D8_IDENTITY rc=$ID_RC file=$ID_OUT"
grep -E "^(build_id|dsp_identity|IDENTITY_OK|RPC_OK|RESOURCE_CLEANUP_OK|exit_code)=" "$ID_OUT" | sed 's/^/D8_IDENTITY_/' || true
if [ $ID_RC -ne 0 ]; then
    echo "D8_STOP identity gate failed rc=$ID_RC; no experiment run"
    exit 3
fi
if ! grep -q "^IDENTITY_OK=PASS" "$ID_OUT"; then
    echo "D8_STOP identity gate did not report PASS; no experiment run"
    exit 3
fi

# ---- the four condition runs ----------------------------------------------
run_condition() {
    COND=$1
    Q=$2
    F=$OUT/$(printf '%02d' "$3")-condition-$COND-q$Q.txt
    T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    "$BIN" "$Q" 46 > "$F" 2>&1
    RC=$?
    T1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "D8_RUN condition=$COND q=$Q rc=$RC started=$T0 finished=$T1 file=$F"
    # The host prints the whole evidence block; surface the keys that decide
    # the classification so the session log is readable without the file.
    grep -E "^(d8_rpc_status|d8_capture|d8_host_class|d8_contract|d8_selector|d8_counts|d8_fill|d8_address|d8_fnv|d8_guard|d8_failure|d8_schema|d8_readback|d8_cleanup|d8_build_id|d8_results_ok|d8_observed_sha256|d8_matched|d8_label)=" "$F" | sed "s/^/D8_${COND}_/" || true
    # Preserve the raw arrays verbatim, one file each.
    grep "^d8_slots=" "$F" | sed 's/^d8_slots=//' > "$OUT/$(printf '%02d' "$3")-condition-$COND-slots.csv" || true
    grep "^d8_meta=" "$F" | sed 's/^d8_meta=//' > "$OUT/$(printf '%02d' "$3")-condition-$COND-meta.csv" || true
    if [ $RC -ne 0 ]; then
        echo "D8_STOP condition $COND returned rc=$RC; the budget has no retry"
        return 1
    fi
    if ! grep -q "^d8_contract=PASS" "$F"; then
        echo "D8_STOP condition $COND contract gate did not pass; no numeric reading"
        return 1
    fi
    return 0
}

run_condition A 6 1 || exit 4
# Condition A is the numeric control. If it is not D8_CONTROL_SAFE the session
# stops here and B/C/D are not run (prereg section 9).
if ! grep -q "^d8_label=D8_CONTROL_SAFE" "$OUT/01-condition-A-q6.txt"; then
    echo "D8_STOP condition A is not D8_CONTROL_SAFE; B/C/D not run"
    exit 5
fi

run_condition B 7 2 || exit 4
run_condition C 8 3 || exit 4
run_condition D 9 4 || exit 4

echo "D8_SESSION complete id=$SID finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "D8_SESSION labels:"
for f in "$OUT"/0[1-4]-condition-*.txt; do
    echo "  $(basename "$f") $(grep -m1 '^d8_label=' "$f" || true)"
done
exit 0


