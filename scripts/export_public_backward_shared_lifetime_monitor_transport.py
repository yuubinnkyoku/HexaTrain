"""Closed allow-list projection of the monitor/transport recovery mission.

Updates docs/results/g1-backward-shared-lifetime-formal-summary.json by adding
exactly one new key, monitor_transport_recovery, on top of the previously
published 55-key document. The historical document is re-validated unchanged;
the new key has its own closed template. Writes atomically via a temporary file.
"""
import argparse
import contextlib
import copy
import io
import json
import os
import re
from pathlib import Path

from assess_backward_repair import ROOT, read, require
from certify_backward_native_cohort import ROLES
from prepare_saved_p_backward import sha
from export_public_recovered_backward_shared_lifetime import validate as validate_base
from export_public_recovered_backward_shared_lifetime import selftest as historical_selftest
from assess_monitor_transport_recovery import (FORMAL_BLOCKERS, hex64, reject_private_strings,
    validate as validate_assessment)

PUBLIC = ROOT/'docs/results/g1-backward-shared-lifetime-formal-summary.json'
RECOVERY = ROOT/'build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery'
FORMAL_DIR = ROOT/'build/reports/g1-backward-shared-lifetime-formal/formal-1'
NEW_PLAN_RECORD = RECOVERY/'new-plan-sha256.json'
MONITOR_ATTRIBUTION = RECOVERY/'monitor-cadence-attribution-private.json'
TRANSPORT_ATTRIBUTION = RECOVERY/'transport-attribution-private.json'
GATE = RECOVERY/'device-entry-gate-private.json'
ASSESSMENT = RECOVERY/'formal-eligibility-private.json'
MONITOR_LOG = RECOVERY/'host-tests/monitor-host-test.log'
TRANSPORT_LOG = RECOVERY/'host-tests/transport-host-test.log'
DRYRUN = RECOVERY/'monitor-dryrun-private.json'

MONITOR_PASS = re.compile(r'backward_formal_monitor_host=PASS cases=16 selftest_cases=(\d+) realistic_period=2000 max_start_to_start_ms=([0-9]+(?:\.[0-9]+)?) device_calls=0')
TRANSPORT_PASS = re.compile(r'backward_formal_transport_host=PASS cases=16 selftest_cases=(\d+) binary_roundtrip=PASS device_calls=0')
MONITOR_MODULE = re.compile(r'^monitor_module_sha256=([0-9a-f]{64})$')
TRANSPORT_MODULE = re.compile(r'^transport_module_sha256=([0-9a-f]{64})$')

MONITOR_HISTORY_EXPECTED = dict(runtime_loop_records=34, runtime_intervals=33,
    minimum_interval_seconds=3.250952, median_interval_seconds=4.029617,
    maximum_interval_seconds=5.244854, period_required_seconds=2.0,
    root_cause='ADDITIVE_WAIT_PLUS_SERIAL_QUERY_BLOCK', classification='MONITOR_CONTRACT_FAILED')
MONITOR_CONTRACT_EXPECTED = dict(registered_requirement='foreground sample start-to-start period <= 2.000 s',
    period_milliseconds=2000, lead_milliseconds=100, command_timeout_seconds=8, monotonic_clock=True,
    explicit_scheduled_deadline=True, missed_deadline_detected=True, overlapping_poll_possible=False)
TRANSPORT_HISTORY_EXPECTED = dict(failure_operation='adb pull of the device-gzipped 5-role container',
    configured_timeout_seconds=180, partial_bytes=332333056, partial_complete=False,
    classification='ADB_WRAPPER_TIMEOUT', remote_raw_preserved=True,
    container_reconstructed_losslessly=True, five_role_files_verified=True)
TRANSPORT_CONTRACT_EXPECTED = dict(chunk_bytes=8388608, chunk_timeout_seconds=120, chunk_attempts=2,
    file_timeout_seconds=2700, per_chunk_sha256=True, remote_before_after_sha256=True,
    binary_safe_readback=True, distinct_from_monitor_timeout=True)
STATUS_EXPECTED = 'HOST_REPAIR_QUALIFIED_DEVICE_UNVERIFIED'


def monitor_context(mon):
    require(mon.get('kind') == 'G1_SHARED_LIFETIME_MONITOR_CADENCE_ATTRIBUTION', 'monitor attribution kind')
    interval = mon['runtime_interval_seconds_completion_to_completion']
    require(mon['violation']['classification'] == 'MONITOR_CONTRACT_FAILED'
        and interval['max'] > mon['violation']['required_max_seconds'] == 2.0,
        'historical monitor did not violate the two-second contract')
    return dict(runtime_loop_records=mon['record_counting']['runtime_loop_records'],
        runtime_intervals=interval['count'], minimum_interval_seconds=interval['min'],
        median_interval_seconds=interval['median'], maximum_interval_seconds=interval['max'],
        root_cause=mon['root_cause_decomposition']['dominant_hypothesis'], period_required_seconds=2.0)


def transport_context(tra):
    require(tra.get('kind') == 'G1_SHARED_LIFETIME_POST_EXIT_TRANSPORT_ATTRIBUTION', 'transport attribution kind')
    fc = tra['failure_classification']
    require(fc['post_exit_evidence_incomplete'] is True
        and fc['transfer_verified_for_container_gzip'] is False,
        'historical transport was complete; recovery mission not applicable')
    require(fc['adb_wrapper_timeout'] is True and fc['adb_command_timeout'] is False
        and fc['adb_transport_disconnected'] is False, 'transport failure family')
    roles = tra['role_identity']
    require(list(roles) == ROLES, 'missing role')
    for name, a in roles.items():
        require(a['remote_sha256'] == a['local_sha256'] and a['remote_gzip_sha256'] == a['local_gzip_sha256']
            and a['raw_identity_verified'] is True and a['gzip_identity_verified'] is True,
            'remote/local hash mismatch ' + name)
    require(tra['recovery']['container_reconstructed_losslessly'] is True
        and tra['recovery']['five_role_raw_and_gzip_recovered'] is True
        and tra['recovery']['remote_deleted_or_rewritten'] is False, 'lossless recovery')
    return dict(failure_operation=tra['failure_operation'],
        configured_timeout_seconds=tra['configured_timeout_seconds'],
        partial_bytes=tra['local_partial_bytes'], partial_complete=tra['local_partial_complete'],
        remote_raw_preserved=tra['device_side_file_preserved'])


def host_log_context(kind, text):
    require(isinstance(text, str), 'host log text')
    lines = [x for x in text.splitlines() if x.strip()]
    require(lines, 'missing ' + kind + ' host log')
    pattern = MONITOR_PASS if kind == 'monitor' else TRANSPORT_PASS
    module_pattern = MONITOR_MODULE if kind == 'monitor' else TRANSPORT_MODULE
    m = pattern.fullmatch(lines[-1])
    require(m is not None, 'missing ' + kind + ' host PASS line')
    modules = [x for x in lines if module_pattern.match(x)]
    require(len(modules) == 1, kind + ' module sha recorded exactly once')
    result = dict(cases=16, selftest_cases=int(m.group(1)), device_calls=0,
        module_sha256=module_pattern.match(modules[0]).group(1))
    if kind == 'monitor':
        value = float(m.group(2))
        require(value <= 2000, 'monitor start-to-start exceeds contract')
        result['max_start_to_start_milliseconds'] = value
    else:
        result['binary_roundtrip'] = 'PASS'
    return result


def new_template():
    return dict(status=STATUS_EXPECTED, monitor_contract=None, monitor_history=None,
        monitor_host_qualification=None, transport_history=None, transport_contract=None,
        transport_host_qualification=None, device_entry_gate=None, device_cadence_established=False,
        device_transfer_qualified=False, formal_eligible=False, formal_eligible_reason=None,
        plan_sha256=None, assessment_sha256=None, previous_public_summary_sha256=None)


def validate_new(v, ctx):
    t = new_template()
    require(set(v) == set(t), 'monitor_transport_recovery private/missing field')
    require(v['status'] == STATUS_EXPECTED, 'numerical PASS relabelled as formal PASS')
    require(v['monitor_contract'] == MONITOR_CONTRACT_EXPECTED, 'monitor contract drift')
    require(v['monitor_contract']['missed_deadline_detected'] is True, 'ignored missed deadline')
    mh = v['monitor_history']
    require(isinstance(mh, dict) and set(mh) == set(MONITOR_HISTORY_EXPECTED), 'missing monitoring evidence')
    for key, value in MONITOR_HISTORY_EXPECTED.items():
        require(mh[key] == value, 'monitor history drift ' + key)
    require(mh['maximum_interval_seconds'] > mh['period_required_seconds']
        and mh['classification'] == 'MONITOR_CONTRACT_FAILED', '4 s monitor treated as 2 s PASS')
    mhost = v['monitor_host_qualification']
    require(isinstance(mhost, dict)
        and set(mhost) == {'cases', 'result', 'selftest_cases', 'device_calls',
            'realistic_period_milliseconds', 'max_start_to_start_milliseconds', 'monitor_source_sha256'},
        'monitor host qualification fields')
    require(mhost['cases'] == 16 and mhost['result'] == 'PASS' and mhost['device_calls'] == 0
        and mhost['realistic_period_milliseconds'] == 2000, 'monitor host qualification gate')
    require(isinstance(mhost['selftest_cases'], int) and not isinstance(mhost['selftest_cases'], bool)
        and mhost['selftest_cases'] >= 0, 'monitor selftest count')
    require(isinstance(mhost['max_start_to_start_milliseconds'], (int, float))
        and 0 <= mhost['max_start_to_start_milliseconds'] <= 2000, 'monitor max start-to-start')
    require(hex64(mhost['monitor_source_sha256']), 'monitor source sha')
    th = v['transport_history']
    require(isinstance(th, dict) and set(th) == set(TRANSPORT_HISTORY_EXPECTED), 'transport history fields')
    for key, value in TRANSPORT_HISTORY_EXPECTED.items():
        require(th[key] == value, 'transport history drift ' + key)
    require(th['partial_complete'] is False and th['classification'] == 'ADB_WRAPPER_TIMEOUT',
        'ADB timeout treated as success')
    require(v['transport_contract'] == TRANSPORT_CONTRACT_EXPECTED, 'transport contract drift')
    require(v['transport_contract']['distinct_from_monitor_timeout'] is True
        and v['transport_contract']['chunk_timeout_seconds'] != MONITOR_CONTRACT_EXPECTED['command_timeout_seconds'],
        'transport/monitor timeout distinction')
    thead = v['transport_host_qualification']
    require(isinstance(thead, dict)
        and set(thead) == {'cases', 'result', 'selftest_cases', 'device_calls', 'binary_roundtrip',
            'transport_source_sha256'}, 'transport host qualification fields')
    require(thead == dict(cases=16, result='PASS', selftest_cases=thead['selftest_cases'], device_calls=0,
        binary_roundtrip='PASS', transport_source_sha256=thead['transport_source_sha256']),
        'transport host qualification gate')
    require(isinstance(thead['selftest_cases'], int) and not isinstance(thead['selftest_cases'], bool)
        and thead['selftest_cases'] >= 0, 'transport selftest count')
    require(hex64(thead['transport_source_sha256']), 'transport source sha')
    g = v['device_entry_gate']
    require(isinstance(g, dict)
        and set(g) == {'result', 'online_endpoint', 'same_physical_device_verified', 'dry_run_executed',
            'new_formal_processes'}, 'device entry gate fields')
    require(g['result'] in ('PASS', 'BLOCKED_DEVICE_OFFLINE'), 'device entry gate result')
    require(isinstance(g['online_endpoint'], bool) and isinstance(g['same_physical_device_verified'], bool)
        and isinstance(g['dry_run_executed'], bool) and isinstance(g['new_formal_processes'], int)
        and not isinstance(g['new_formal_processes'], bool) and g['new_formal_processes'] >= 0,
        'device entry gate value types')
    if g['result'] == 'BLOCKED_DEVICE_OFFLINE':
        require(g['online_endpoint'] is False and g['same_physical_device_verified'] is False
            and g['new_formal_processes'] == 0, 'device marked online while gate blocked')
    require(isinstance(v['device_cadence_established'], bool)
        and isinstance(v['device_transfer_qualified'], bool), 'device qualification types')
    if not (ctx['gate_pass'] and ctx['formal_present']):
        require(v['device_cadence_established'] is False and v['device_transfer_qualified'] is False,
            'device qualification without gate PASS and formal process')
    require(v['formal_eligible'] is False, 'formal eligibility promotion')
    require(v['formal_eligible_reason'] in FORMAL_BLOCKERS, 'formal eligible reason')
    if ctx['gate_pass']:
        require(v['formal_eligible_reason'] not in ('DEVICE_OFFLINE', 'DEVICE_ENTRY_GATE_MISSING',
            'DEVICE_ENTRY_GATE_NOT_PASS'), 'gate PASS bound to a gate blocker')
    for key in ('plan_sha256', 'assessment_sha256', 'previous_public_summary_sha256'):
        require(hex64(v[key]), key + ' sha')
    require(mhost['monitor_source_sha256'] == ctx['monitor_source_sha256'], 'monitor module identity')
    require(mhost['selftest_cases'] == ctx['monitor_selftest_cases']
        and mhost['max_start_to_start_milliseconds'] == ctx['monitor_max_start_to_start_milliseconds'],
        'monitor host evidence binding')
    require(thead['transport_source_sha256'] == ctx['transport_source_sha256'], 'transport module identity')
    require(thead['selftest_cases'] == ctx['transport_selftest_cases'], 'transport host evidence binding')
    require(v['plan_sha256'] == ctx['plan_sha256'], 'plan identity')
    require(v['assessment_sha256'] == ctx['assessment_sha256'], 'assessment identity')
    require(v['previous_public_summary_sha256'] == ctx['previous_public_summary_sha256'], 'previous public summary identity')
    require(v['formal_eligible_reason'] == ctx['formal_blocker'], 'formal eligible reason binding')
    require(g['result'] == ctx['gate_result'] and g['online_endpoint'] == ctx['gate_online']
        and g['same_physical_device_verified'] == ctx['gate_same']
        and g['new_formal_processes'] == ctx['gate_new_formal_processes'], 'device entry gate binding')
    reject_private_strings(v)


def project():
    freeze = read(RECOVERY/'evidence-freeze-private.json')
    require(freeze.get('kind') == 'G1_SHARED_LIFETIME_MONITOR_TRANSPORT_EVIDENCE_FREEZE', 'evidence freeze kind')
    frozen = freeze['frozen_identities']
    identity = dict(compiled_source_commit=frozen['compiled_source_commit'], probe_sha256=frozen['probe_sha256'],
        dsp_sha256=frozen['dsp_sha256'], host_sha256=frozen['host_sha256'])
    base = read(PUBLIC)
    validate_base(base, identity)
    require('monitor_transport_recovery' not in base, 'recovery key already published')
    require(base['probe_sha256'] not in (frozen['original_arm_sha256'], frozen['two_role_arm_sha256']),
        'old binary promoted')
    mon = monitor_context(read(MONITOR_ATTRIBUTION))
    require(mon == {k: MONITOR_HISTORY_EXPECTED[k] for k in mon}, 'published monitor history is not the attribution')
    tra = transport_context(read(TRANSPORT_ATTRIBUTION))
    require(tra['configured_timeout_seconds'] == TRANSPORT_HISTORY_EXPECTED['configured_timeout_seconds']
        and tra['partial_bytes'] == TRANSPORT_HISTORY_EXPECTED['partial_bytes']
        and tra['partial_complete'] is False and tra['remote_raw_preserved'] is True
        and tra['failure_operation'] == TRANSPORT_HISTORY_EXPECTED['failure_operation'],
        'published transport history is not the attribution')
    monitor_host = host_log_context('monitor', MONITOR_LOG.read_text(encoding='utf-8-sig'))
    transport_host = host_log_context('transport', TRANSPORT_LOG.read_text(encoding='utf-8-sig'))
    assessment = read(ASSESSMENT)
    validate_assessment(assessment)
    require(assessment['formal_eligible'] is False, 'formal process already eligible; later stage exporter required')
    require(assessment['host_repair_qualified'] is True, 'host repair not qualified')
    require(assessment['monitor_host']['module_sha256'] == monitor_host['module_sha256']
        and assessment['monitor_host']['selftest_cases'] == monitor_host['selftest_cases']
        and assessment['monitor_host']['max_start_to_start_milliseconds'] == monitor_host['max_start_to_start_milliseconds'],
        'assessment/monitor log mismatch')
    require(assessment['transport_host']['module_sha256'] == transport_host['module_sha256']
        and assessment['transport_host']['selftest_cases'] == transport_host['selftest_cases'],
        'assessment/transport log mismatch')
    require(assessment['historical_monitor']['attribution_sha256'] == sha(MONITOR_ATTRIBUTION)
        and assessment['historical_monitor']['maximum_interval_seconds'] == mon['maximum_interval_seconds'],
        'assessment/monitor attribution mismatch')
    require(assessment['historical_transport']['attribution_sha256'] == sha(TRANSPORT_ATTRIBUTION)
        and assessment['historical_transport']['local_partial_bytes'] == tra['partial_bytes'],
        'assessment/transport attribution mismatch')
    gate = read(GATE)
    require(gate.get('result') in ('PASS', 'BLOCKED_DEVICE_OFFLINE'), 'unknown device entry gate result')
    require(assessment['device_entry_gate']['result'] == gate['result'], 'assessment/gate mismatch')
    gate_pass = gate['result'] == 'PASS' and gate['online_endpoint'] is True and gate['same_physical_device_verified'] is True
    if gate_pass:
        require(assessment['formal_blocker'] not in ('DEVICE_OFFLINE', 'DEVICE_ENTRY_GATE_MISSING',
            'DEVICE_ENTRY_GATE_NOT_PASS'), 'gate PASS but assessment gate blocker')
    plan_record = read(NEW_PLAN_RECORD)
    plan_path = ROOT/plan_record['plan']
    require(plan_path.resolve() == (ROOT/'host_tests/fp32_custom/backward_shared_lifetime_monitor_transport_plan.json').resolve(),
        'unknown plan path')
    require(plan_record['sha256'] == sha(plan_path), 'plan sha record differs from plan file')
    formal_present = FORMAL_DIR.is_dir()
    dry_run_executed = DRYRUN.is_file()
    mt = new_template()
    mt.update(monitor_contract=dict(MONITOR_CONTRACT_EXPECTED),
        monitor_history=dict(runtime_loop_records=mon['runtime_loop_records'],
            runtime_intervals=mon['runtime_intervals'],
            minimum_interval_seconds=mon['minimum_interval_seconds'],
            median_interval_seconds=mon['median_interval_seconds'],
            maximum_interval_seconds=mon['maximum_interval_seconds'],
            period_required_seconds=mon['period_required_seconds'], root_cause=mon['root_cause'],
            classification='MONITOR_CONTRACT_FAILED'),
        monitor_host_qualification=dict(cases=16, result='PASS', selftest_cases=monitor_host['selftest_cases'],
            device_calls=0, realistic_period_milliseconds=2000,
            max_start_to_start_milliseconds=monitor_host['max_start_to_start_milliseconds'],
            monitor_source_sha256=monitor_host['module_sha256']),
        transport_history=dict(TRANSPORT_HISTORY_EXPECTED),
        transport_contract=dict(TRANSPORT_CONTRACT_EXPECTED),
        transport_host_qualification=dict(cases=16, result='PASS', selftest_cases=transport_host['selftest_cases'],
            device_calls=0, binary_roundtrip='PASS', transport_source_sha256=transport_host['module_sha256']),
        device_entry_gate=dict(result=gate['result'], online_endpoint=gate['online_endpoint'] is True,
            same_physical_device_verified=gate['same_physical_device_verified'] is True,
            dry_run_executed=dry_run_executed, new_formal_processes=gate['new_formal_processes']),
        device_cadence_established=bool(gate_pass and formal_present and assessment['formal_monitor_contract_met']),
        device_transfer_qualified=bool(gate_pass and formal_present and assessment['formal_transport_verified']),
        formal_eligible=False, formal_eligible_reason=assessment['formal_blocker'],
        plan_sha256=plan_record['sha256'], assessment_sha256=sha(ASSESSMENT),
        previous_public_summary_sha256=sha(PUBLIC))
    ctx = dict(monitor_source_sha256=monitor_host['module_sha256'],
        monitor_selftest_cases=monitor_host['selftest_cases'],
        monitor_max_start_to_start_milliseconds=monitor_host['max_start_to_start_milliseconds'],
        transport_source_sha256=transport_host['module_sha256'],
        transport_selftest_cases=transport_host['selftest_cases'],
        plan_sha256=plan_record['sha256'], assessment_sha256=sha(ASSESSMENT),
        previous_public_summary_sha256=sha(PUBLIC), gate_result=gate['result'],
        gate_online=gate['online_endpoint'] is True, gate_same=gate['same_physical_device_verified'] is True,
        gate_new_formal_processes=gate['new_formal_processes'], gate_pass=gate_pass,
        formal_present=formal_present, formal_blocker=assessment['formal_blocker'])
    validate_new(mt, ctx)
    new_doc = copy.deepcopy(base)
    new_doc['monitor_transport_recovery'] = mt
    require(set(new_doc) == set(base) | {'monitor_transport_recovery'}, 'public schema change')
    validate_base({k: new_doc[k] for k in base}, identity)
    tmp = PUBLIC.with_name(PUBLIC.name + '.tmp')
    require(not tmp.exists(), 'stale temporary export file')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(new_doc, f, indent=2, allow_nan=False)
    os.replace(tmp, PUBLIC)
    return new_doc


def synthetic_new(ctx_overrides=None):
    ctx = dict(monitor_source_sha256='a'*64, monitor_selftest_cases=16,
        monitor_max_start_to_start_milliseconds=19.5, transport_source_sha256='b'*64,
        transport_selftest_cases=16, plan_sha256='c'*64, assessment_sha256='d'*64,
        previous_public_summary_sha256='e'*64, gate_result='BLOCKED_DEVICE_OFFLINE',
        gate_online=False, gate_same=False, gate_new_formal_processes=0, gate_pass=False,
        formal_present=False, formal_blocker='DEVICE_OFFLINE')
    if ctx_overrides:
        ctx.update(ctx_overrides)
    v = dict(status=STATUS_EXPECTED, monitor_contract=dict(MONITOR_CONTRACT_EXPECTED),
        monitor_history=dict(MONITOR_HISTORY_EXPECTED),
        monitor_host_qualification=dict(cases=16, result='PASS', selftest_cases=16, device_calls=0,
            realistic_period_milliseconds=2000, max_start_to_start_milliseconds=19.5,
            monitor_source_sha256=ctx['monitor_source_sha256']),
        transport_history=dict(TRANSPORT_HISTORY_EXPECTED),
        transport_contract=dict(TRANSPORT_CONTRACT_EXPECTED),
        transport_host_qualification=dict(cases=16, result='PASS', selftest_cases=16, device_calls=0,
            binary_roundtrip='PASS', transport_source_sha256=ctx['transport_source_sha256']),
        device_entry_gate=dict(result=ctx['gate_result'], online_endpoint=ctx['gate_online'],
            same_physical_device_verified=ctx['gate_same'], dry_run_executed=False,
            new_formal_processes=ctx['gate_new_formal_processes']),
        device_cadence_established=False, device_transfer_qualified=False, formal_eligible=False,
        formal_eligible_reason=ctx['formal_blocker'], plan_sha256=ctx['plan_sha256'],
        assessment_sha256=ctx['assessment_sha256'],
        previous_public_summary_sha256=ctx['previous_public_summary_sha256'])
    return v, ctx


def selftest():
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        historical_selftest()
    line = captured.getvalue().strip().splitlines()[-1]
    require(line.startswith('recovered_public_exporter_selftest=PASS rejection_cases=45'), 'historical suite')
    cases = 0
    v, ctx = synthetic_new()
    validate_new(v, ctx)
    cases += 1

    def reject(mutate=None, ctx_overrides=None, ctx_mutation=None):
        nonlocal cases
        b, c = synthetic_new(ctx_overrides)
        if mutate:
            mutate(b)
        if ctx_mutation:
            ctx_mutation(c)
        try:
            validate_new(b, c)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid monitor_transport_recovery accepted')
        cases += 1

    reject(lambda b: b.update(status='FIVE_ROLE_NUMERICS_AND_LIFETIME_PASS'))
    reject(lambda b: b['monitor_history'].update(maximum_interval_seconds=1.9, classification='PASS'))
    reject(lambda b: b.pop('monitor_history'))
    reject(lambda b: b['monitor_contract'].update(missed_deadline_detected=False))
    reject(lambda b: b['transport_history'].update(classification='PASS'))
    reject(lambda b: b['transport_history'].update(partial_complete=True))
    reject(lambda b: b['transport_history'].update(five_role_files_verified=False))
    reject(lambda b: b.update(plan_sha256='g'*64))
    reject(lambda b: b['monitor_host_qualification'].update(monitor_source_sha256='h'*64))
    reject(lambda b: b['monitor_host_qualification'].pop('max_start_to_start_milliseconds'))
    reject(lambda b: b['transport_history'].update(remote_raw_preserved=False))
    reject(lambda b: b['device_entry_gate'].pop('same_physical_device_verified'))
    reject(lambda b: b['transport_contract'].update(per_chunk_sha256=False))
    reject(lambda b: b['device_entry_gate'].update(new_formal_processes=1))
    reject(lambda b: b.update(formal_eligible=True))
    reject(lambda b: b.update(formal_eligible_reason='NONE'))
    reject(lambda b: b.update(private_path='secret'))
    reject(lambda b: b['device_entry_gate'].update(online_endpoint=True))
    reject(lambda b: b['device_entry_gate'].update(new_formal_processes=2))
    reject(lambda b: b['monitor_host_qualification'].update(device_calls=1))
    reject(lambda b: b['transport_host_qualification'].update(binary_roundtrip='FAIL'))
    reject(lambda b: b.update(device_cadence_established=True))
    reject(lambda b: b.update(assessment_sha256='x'*32))
    reject(ctx_mutation=lambda c: c.update(monitor_source_sha256='9'*64))
    reject(ctx_mutation=lambda c: c.update(transport_selftest_cases=99))
    reject(ctx_mutation=lambda c: c.update(plan_sha256='9'*64))
    reject(ctx_mutation=lambda c: c.update(assessment_sha256='9'*64))
    reject(ctx_mutation=lambda c: c.update(previous_public_summary_sha256='9'*64))
    reject(ctx_mutation=lambda c: c.update(formal_blocker='PILOT_NOT_NUMERICAL'))
    reject(ctx_mutation=lambda c: c.update(gate_result='PASS', gate_online=True, gate_same=True,
        gate_new_formal_processes=1, gate_pass=True))

    try:
        monitor_context(dict(kind='G1_SHARED_LIFETIME_MONITOR_CADENCE_ATTRIBUTION',
            violation=dict(classification='MONITOR_CONTRACT_FAILED', required_max_seconds=2.0),
            record_counting=dict(runtime_loop_records=34),
            runtime_interval_seconds_completion_to_completion=dict(count=33, min=1., median=1., max=1.9),
            root_cause_decomposition=dict(dominant_hypothesis='X')))
    except ValueError:
        pass
    else:
        raise AssertionError('sub-2 s attribution accepted as contract failure')
    cases += 1

    good_roles = {name: dict(remote_sha256='1'*64, local_sha256='1'*64, remote_gzip_sha256='2'*64,
        local_gzip_sha256='2'*64, raw_identity_verified=True, gzip_identity_verified=True) for name in ROLES}
    base_tra = dict(kind='G1_SHARED_LIFETIME_POST_EXIT_TRANSPORT_ATTRIBUTION',
        failure_operation='adb pull of the device-gzipped 5-role container', configured_timeout_seconds=180,
        local_partial_bytes=332333056, local_partial_complete=False, device_side_file_preserved=True,
        failure_classification=dict(adb_command_timeout=False, adb_wrapper_timeout=True,
            adb_transport_disconnected=False, post_exit_evidence_incomplete=True,
            transfer_verified_for_container_gzip=False, remote_raw_missing=False),
        recovery=dict(container_reconstructed_losslessly=True, five_role_raw_and_gzip_recovered=True,
            remote_deleted_or_rewritten=False), role_identity=dict(good_roles))
    transport_context(copy.deepcopy(base_tra))
    cases += 1
    broken = copy.deepcopy(base_tra)
    del broken['role_identity']['dual']
    try:
        transport_context(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('missing role accepted')
    cases += 1
    broken = copy.deepcopy(base_tra)
    broken['role_identity']['dual']['local_sha256'] = '9'*64
    try:
        transport_context(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('remote/local hash mismatch accepted')
    cases += 1

    monitor_lines = 'monitor_module_sha256=' + 'a'*64 + '\nbackward_formal_monitor_host=PASS cases=16 selftest_cases=17 realistic_period=2000 max_start_to_start_ms=12.0 device_calls=0\n'
    host_log_context('monitor', monitor_lines)
    cases += 1
    try:
        host_log_context('monitor', monitor_lines.replace('PASS', 'FAIL', 1).replace('device_calls=0', 'device_calls=1'))
    except ValueError:
        pass
    else:
        raise AssertionError('missing monitor PASS line accepted')
    cases += 1
    transport_lines = 'transport_module_sha256=' + 'b'*64 + '\nbackward_formal_transport_host=PASS cases=16 selftest_cases=18 binary_roundtrip=PASS device_calls=0\n'
    host_log_context('transport', transport_lines)
    cases += 1
    try:
        host_log_context('transport', transport_lines.rsplit('\n', 2)[0] + '\n')
    except ValueError:
        pass
    else:
        raise AssertionError('missing transport PASS line accepted')
    cases += 1
    try:
        host_log_context('monitor', monitor_lines.replace('monitor_module_sha256=' + 'a'*64, 'monitor_module_sha256=zz'))
    except ValueError:
        pass
    else:
        raise AssertionError('malformed module sha accepted')
    cases += 1
    print('monitor_transport_public_exporter_selftest=PASS historical_cases=45 new_cases=' + str(cases)
        + ' device_calls=0')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--out', type=Path, default=PUBLIC)
    a = p.parse_args()
    if a.self_test:
        selftest()
    else:
        require(a.out is not None, 'output required')
        require(a.out.resolve() == PUBLIC.resolve(), 'closed target path only')
        new_doc = project()
        print('monitor_transport_public_exporter=PASS status=' + new_doc['monitor_transport_recovery']['status']
            + ' formal_eligible=false device_calls=0')
