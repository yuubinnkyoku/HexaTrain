#!/system/bin/sh
# CPU-only test of the D8 session harness decision logic.
#
#   sh d8_session_logic_test.sh
#
# The harness spends the registered budget (one session, four runs, no retry),
# so a defect in its stop logic would either waste the budget or, worse, let the
# session continue past a condition that should have stopped it. This test
# drives the same decision points with synthetic host output and checks that
# each one fires.
#
# It does not touch the DSP. It proves that:
#   - a missing IDENTITY_OK stops before any experiment run;
#   - a non-zero run stops the session;
#   - a missing d8_contract=PASS stops before any numeric reading;
#   - condition A that is not D8_CONTROL_SAFE stops before B/C/D;
#   - a clean session runs A, B, C, D in that order and exits 0.
set -u

WORK=${TMPDIR:-/tmp}/d8-logic-$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

CHECKS=0
FAILS=0
check() {
    CHECKS=$((CHECKS + 1))
    if [ "$1" = "1" ]; then
        :
    else
        FAILS=$((FAILS + 1))
        echo "FAIL $2"
    fi
}

# The decision points, copied from d8_session.sh so the test exercises the same
# predicates rather than a paraphrase of them.
identity_ok() {
    grep -q "^IDENTITY_OK=PASS" "$1"
}
run_ok() {
    [ "$2" -eq 0 ] && grep -q "^d8_contract=PASS" "$1"
}
control_safe() {
    grep -q "^d8_label=D8_CONTROL_SAFE" "$1"
}

# ---- 1. identity gate ------------------------------------------------------
printf 'IDENTITY_OK=PASS\n' > "$WORK/id_pass.txt"
printf 'IDENTITY_OK=FAIL\n' > "$WORK/id_fail.txt"
printf 'RPC_OK=PASS\n' > "$WORK/id_none.txt"
if identity_ok "$WORK/id_pass.txt"; then check 1 identity_pass; else check 0 identity_pass; fi
if identity_ok "$WORK/id_fail.txt"; then check 0 identity_fail_detected; else check 1 identity_fail_detected; fi
if identity_ok "$WORK/id_none.txt"; then check 0 identity_missing_detected; else check 1 identity_missing_detected; fi

# ---- 2. per-run gate -------------------------------------------------------
printf 'd8_contract=PASS first_bad=ok\n' > "$WORK/run_pass.txt"
printf 'd8_contract=FAIL first_bad=guard_digest\n' > "$WORK/run_fail.txt"
printf 'd8_rpc_status=0\n' > "$WORK/run_none.txt"
if run_ok "$WORK/run_pass.txt" 0; then check 1 run_pass; else check 0 run_pass; fi
if run_ok "$WORK/run_fail.txt" 0; then check 0 run_contract_fail_detected; else check 1 run_contract_fail_detected; fi
if run_ok "$WORK/run_none.txt" 0; then check 0 run_missing_contract_detected; else check 1 run_missing_contract_detected; fi
if run_ok "$WORK/run_pass.txt" 7; then check 0 run_nonzero_rc_detected; else check 1 run_nonzero_rc_detected; fi

# ---- 3. condition A control gate ------------------------------------------
printf 'd8_label=D8_CONTROL_SAFE\n' > "$WORK/a_safe.txt"
printf 'd8_label=D8_CONTROL_ANOMALY\n' > "$WORK/a_anom.txt"
printf 'd8_label=D8_NO_VIOLATION_OBSERVED\n' > "$WORK/a_wrong.txt"
if control_safe "$WORK/a_safe.txt"; then check 1 control_safe; else check 0 control_safe; fi
if control_safe "$WORK/a_anom.txt"; then check 0 control_anomaly_detected; else check 1 control_anomaly_detected; fi
if control_safe "$WORK/a_wrong.txt"; then check 0 control_wrong_label_detected; else check 1 control_wrong_label_detected; fi

# ---- 4. the full sequence, driven end to end -------------------------------
# A miniature session: the same order and the same gates, with synthetic files.
simulate() {
    SID=$1
    A_LABEL=$2
    B_RC=$3
    D=$WORK/$SID
    mkdir -p "$D"
    printf 'IDENTITY_OK=PASS\n' > "$D/00-identity.txt"
    identity_ok "$D/00-identity.txt" || { echo "stop_identity"; return 3; }
    printf 'd8_contract=PASS first_bad=ok\nd8_label=%s\n' "$A_LABEL" > "$D/01-A.txt"
    run_ok "$D/01-A.txt" 0 || { echo "stop_A"; return 4; }
    control_safe "$D/01-A.txt" || { echo "stop_control"; return 5; }
    printf 'd8_contract=PASS first_bad=ok\nd8_label=D8_NO_VIOLATION_OBSERVED\n' > "$D/02-B.txt"
    run_ok "$D/02-B.txt" "$B_RC" || { echo "stop_B"; return 4; }
    printf 'd8_contract=PASS first_bad=ok\nd8_label=D8_NO_VIOLATION_OBSERVED\n' > "$D/03-C.txt"
    run_ok "$D/03-C.txt" 0 || { echo "stop_C"; return 4; }
    printf 'd8_contract=PASS first_bad=ok\nd8_label=D8_NO_VIOLATION_OBSERVED\n' > "$D/04-D.txt"
    run_ok "$D/04-D.txt" 0 || { echo "stop_D"; return 4; }
    echo "complete"
    return 0
}

OUT=$(simulate clean D8_CONTROL_SAFE 0); RC=$?
check "$([ "$RC" -eq 0 ] && [ "$OUT" = "complete" ] && echo 1 || echo 0)" "session_clean_completes"

OUT=$(simulate anom D8_CONTROL_ANOMALY 0); RC=$?
check "$([ "$RC" -eq 5 ] && [ "$OUT" = "stop_control" ] && echo 1 || echo 0)" "session_stops_on_control_anomaly"

OUT=$(simulate bfail D8_CONTROL_SAFE 7); RC=$?
check "$([ "$RC" -eq 4 ] && [ "$OUT" = "stop_B" ] && echo 1 || echo 0)" "session_stops_on_run_failure"

# A session that stopped at A must not have produced B/C/D files.
check "$([ ! -f "$WORK/anom/02-B.txt" ] && echo 1 || echo 0)" "stopped_session_has_no_B"

echo "d8_session_logic_checks=$CHECKS fail=$FAILS"
if [ "$FAILS" -ne 0 ]; then
    echo "d8_session_logic=FAIL"
    exit 1
fi
echo "d8_session_logic=PASS"
exit 0
