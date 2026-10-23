"""Deterministic read-only monitor/transport recovery assessor; never a device runner.

Derives five independent verdicts (numerical pilot, historical monitor contract,
historical transport completeness, host repair qualification, formal eligibility)
and never merges them. The --entry mode is a hard, non-throwing entry gate for the
new device runner; it performs no device calls of its own.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from assess_backward_repair import ROOT, read, require
from certify_backward_native_cohort import ROLES
from prepare_saved_p_backward import sha

RECOVERY_REL = Path('build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery')
FORMAL_REL = Path('build/reports/g1-backward-shared-lifetime-formal')
PLAN_REL = Path('host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json')
PILOT_ASSESSMENT_REL = Path('build/reports/g1-backward-shared-lifetime-formal/pilot/recovered-pilot-assessment-private.json')
RUNNER_V3_REL = Path('scripts/run_backward_shared_lifetime_formal_v3.ps1')
NEW_PLAN_REL = Path('host_tests/fp32_custom/backward_shared_lifetime_monitor_transport_plan.json')
MONITOR_LOG_REL = RECOVERY_REL/'host-tests/monitor-host-test.log'
TRANSPORT_LOG_REL = RECOVERY_REL/'host-tests/transport-host-test.log'
DRYRUN_REL = RECOVERY_REL/'monitor-dryrun-private.json'
GATE_REL = RECOVERY_REL/'device-entry-gate-private.json'
NEW_PLAN_RECORD_REL = RECOVERY_REL/'new-plan-sha256.json'

PERIOD_REQUIRED_SECONDS = 2.0
MONITOR_PASS_LINE = re.compile(r'backward_formal_monitor_host=PASS cases=16 selftest_cases=(\d+) realistic_period=2000 max_start_to_start_ms=([0-9]+(?:\.[0-9]+)?) device_calls=0')
TRANSPORT_PASS_LINE = re.compile(r'backward_formal_transport_host=PASS cases=16 selftest_cases=(\d+) binary_roundtrip=PASS device_calls=0')
MONITOR_MODULE_LINE = re.compile(r'monitor_module_sha256=([0-9a-f]{64})')
TRANSPORT_MODULE_LINE = re.compile(r'transport_module_sha256=([0-9a-f]{64})')
HEX64 = re.compile(r'[0-9a-f]{64}')

FORMAL_BLOCKERS = ('DEVICE_OFFLINE', 'DEVICE_ENTRY_GATE_MISSING', 'DEVICE_ENTRY_GATE_NOT_PASS',
    'HOST_REPAIR_NOT_QUALIFIED', 'PILOT_NOT_NUMERICAL', 'FORMAL_PROCESS_NOT_RUN',
    'MONITOR_SUMMARY_MISSING', 'MONITOR_CONTRACT_NOT_MET', 'TRANSPORT_RECORD_MISSING',
    'TRANSPORT_TRANSFER_NOT_VERIFIED', 'CANDIDATE_IDENTITY_MISMATCH')
ENTRY_REASONS = ('HOST_REPAIR_NOT_QUALIFIED', 'MONITOR_DRYRUN_MISSING', 'MONITOR_DRYRUN_CONTRACT_FAILED',
    'NUMERICAL_PILOT_NOT_VERIFIED', 'DEVICE_ENTRY_GATE_NOT_PASS', 'PRIOR_FORMAL_ATTEMPT_EXISTS',
    'SOURCE_NOT_COMMITTED', 'RUNNER_IDENTITY_MISMATCH', 'PLAN_IDENTITY_MISMATCH')
TRANSPORT_CLASSIFICATIONS = ('ADB_WRAPPER_TIMEOUT', 'ADB_COMMAND_TIMEOUT', 'UNATTRIBUTED')

PRIVATE_STRING_PATTERNS = (
    re.compile(r'[A-Za-z]:\\'), re.compile(r'/data/'), re.compile(r'/sdcard/'),
    re.compile(r'://'), re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'),
    re.compile(r'(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])'), re.compile(r'\bcom\.[A-Za-z]'))


def hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def reject_private_strings(value):
    if isinstance(value, str):
        for pattern in PRIVATE_STRING_PATTERNS:
            require(pattern.search(value) is None, 'private identifier in public projection')
    elif isinstance(value, dict):
        for k, v in value.items():
            reject_private_strings(k)
            reject_private_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            reject_private_strings(v)


def template():
    return dict(schema=1, kind='G1_SHARED_LIFETIME_MONITOR_TRANSPORT_FORMAL_ELIGIBILITY',
        numerical_pilot_verified=False,
        historical_monitor=dict(classification=None, period_required_seconds=None,
            runtime_loop_records=None, runtime_intervals=None, minimum_interval_seconds=None,
            median_interval_seconds=None, maximum_interval_seconds=None, samples_over_contract=None,
            root_cause=None, attribution_sha256=None),
        monitor_contract_failed=False,
        historical_transport=dict(classification=None, failure_operation=None,
            configured_timeout_seconds=None, local_partial_bytes=None, local_partial_complete=None,
            transfer_verified_for_container_gzip=None, post_exit_evidence_incomplete=None,
            five_role_files_recovered=None, attribution_sha256=None),
        post_exit_transport_incomplete=False,
        host_repair_qualified=False,
        monitor_host=dict(qualified=False, cases=None, selftest_cases=None,
            max_start_to_start_milliseconds=None, device_calls=None, module_sha256=None, log_sha256=None),
        transport_host=dict(qualified=False, cases=None, selftest_cases=None, binary_roundtrip=None,
            device_calls=None, module_sha256=None, log_sha256=None),
        device_entry_gate=dict(result=None, online_endpoint=False, same_physical_device_verified=False,
            new_formal_processes=0),
        candidate_identity_match=False,
        formal_process_present=False,
        formal_monitor_contract_met=False,
        formal_monitor_missed_deadlines=None,
        formal_transport_verified=False,
        formal_eligible=False,
        formal_blocker=None,
        new_device_processes=0)


def validate(v):
    t = template()
    require(set(v) == set(t), 'assessor private/missing field')
    require(v['schema'] == t['schema'] and v['kind'] == t['kind'], 'assessor kind')
    require(v['new_device_processes'] == 0, 'assessor launched a device process')
    for k in ('numerical_pilot_verified', 'monitor_contract_failed', 'post_exit_transport_incomplete',
              'host_repair_qualified', 'candidate_identity_match', 'formal_process_present',
              'formal_monitor_contract_met', 'formal_transport_verified', 'formal_eligible'):
        require(isinstance(v[k], bool), 'assessor verdict type ' + k)
    hm = v['historical_monitor']
    require(set(hm) == set(t['historical_monitor']), 'historical monitor fields')
    require(isinstance(hm['classification'], str) and hm['classification'], 'monitor classification')
    require(hm['period_required_seconds'] == PERIOD_REQUIRED_SECONDS, 'monitor period projection')
    require(hm['runtime_loop_records'] == 34 and hm['runtime_intervals'] == 33, 'monitor record counts')
    require(isinstance(hm['samples_over_contract'], int) and hm['samples_over_contract'] >= 0, 'monitor samples over contract')
    require(isinstance(hm['root_cause'], str) and hm['root_cause'], 'monitor root cause')
    for k in ('minimum_interval_seconds', 'median_interval_seconds', 'maximum_interval_seconds'):
        require(isinstance(hm[k], (int, float)) and math.isfinite(hm[k]) and hm[k] > 0, 'monitor ' + k)
    require(hm['minimum_interval_seconds'] <= hm['median_interval_seconds'] <= hm['maximum_interval_seconds'], 'monitor interval order')
    require(hex64(hm['attribution_sha256']), 'monitor attribution sha')
    require(v['monitor_contract_failed'] == (hm['maximum_interval_seconds'] > PERIOD_REQUIRED_SECONDS), 'monitor contract verdict')
    require((hm['classification'] == 'MONITOR_CONTRACT_FAILED') == v['monitor_contract_failed'], 'monitor classification verdict')
    ht = v['historical_transport']
    require(set(ht) == set(t['historical_transport']), 'historical transport fields')
    require(ht['classification'] in TRANSPORT_CLASSIFICATIONS, 'transport classification')
    require(isinstance(ht['failure_operation'], str) and ht['failure_operation'], 'transport failure operation')
    require(isinstance(ht['configured_timeout_seconds'], int) and ht['configured_timeout_seconds'] > 0, 'transport timeout seconds')
    require(isinstance(ht['local_partial_bytes'], int) and ht['local_partial_bytes'] >= 0, 'transport partial bytes')
    for k in ('local_partial_complete', 'transfer_verified_for_container_gzip',
              'post_exit_evidence_incomplete', 'five_role_files_recovered'):
        require(isinstance(ht[k], bool), 'transport flag ' + k)
    require(ht['five_role_files_recovered'] is True, 'historical five-role recovery missing')
    require(v['post_exit_transport_incomplete'] == (ht['post_exit_evidence_incomplete'] and not ht['transfer_verified_for_container_gzip']), 'transport verdict')
    require(hex64(ht['attribution_sha256']), 'transport attribution sha')
    for name in ('monitor_host', 'transport_host'):
        h = v[name]
        require(set(h) == set(t[name]), name + ' fields')
        require(isinstance(h['qualified'], bool), name + ' qualified')
        for k in ('cases', 'selftest_cases', 'device_calls'):
            require(h[k] is None or isinstance(h[k], int), name + ' ' + k)
        require(h['module_sha256'] is None or hex64(h['module_sha256']), name + ' module sha')
        require(h['log_sha256'] is None or hex64(h['log_sha256']), name + ' log sha')
        require(h['qualified'] == (h['cases'] == 16 and h['device_calls'] == 0 and h['module_sha256'] is not None and h['log_sha256'] is not None), name + ' qualification')
    require(v['monitor_host']['max_start_to_start_milliseconds'] is None or
        isinstance(v['monitor_host']['max_start_to_start_milliseconds'], (int, float)), 'monitor max start-to-start')
    require(v['transport_host']['binary_roundtrip'] is None or v['transport_host']['binary_roundtrip'] == 'PASS', 'transport binary roundtrip')
    require(v['host_repair_qualified'] == (v['monitor_host']['qualified'] and v['transport_host']['qualified']), 'host repair verdict')
    g = v['device_entry_gate']
    require(set(g) == set(t['device_entry_gate']), 'entry gate fields')
    require(g['result'] is None or g['result'] in ('PASS', 'BLOCKED_DEVICE_OFFLINE'), 'entry gate result')
    require(isinstance(g['online_endpoint'], bool) and isinstance(g['same_physical_device_verified'], bool), 'entry gate flags')
    require(isinstance(g['new_formal_processes'], int) and g['new_formal_processes'] >= 0, 'entry gate process count')
    require(v['formal_monitor_missed_deadlines'] is None or
        (isinstance(v['formal_monitor_missed_deadlines'], int) and v['formal_monitor_missed_deadlines'] >= 0), 'missed deadlines')
    gate_pass = g['result'] == 'PASS' and g['online_endpoint'] and g['same_physical_device_verified']
    conjuncts = (v['numerical_pilot_verified'], v['host_repair_qualified'], gate_pass,
        v['formal_process_present'], v['formal_monitor_contract_met'], v['formal_monitor_missed_deadlines'] == 0,
        v['formal_transport_verified'], v['candidate_identity_match'])
    require(v['formal_eligible'] == all(conjuncts), 'formal eligibility consistency')
    if v['formal_eligible']:
        require(v['formal_blocker'] is None, 'blocker on eligible formal')
    else:
        require(v['formal_blocker'] in FORMAL_BLOCKERS, 'unknown formal blocker')
    reject_private_strings(v)


def _numerical_pilot(base):
    plan = read(base/PLAN_REL)
    anchors = plan['anchor_legacy_digests']
    p = read(base/PILOT_ASSESSMENT_REL)
    roles = p['roles']
    if list(roles) != ROLES:
        return False
    if p.get('actual_exit') != 0 or p.get('teardown') != 'PASS':
        return False
    if sum(r['pairs'] for r in roles.values()) != 58944:
        return False
    for role, a in roles.items():
        if a.get('classification') != 'PASS':
            return False
        if a.get('numerical_digest') != anchors[role]:
            return False
        if a.get('arithmetic_word_mismatches') != 0 or a.get('observer_mismatch_words') != 0:
            return False
    return True


def _parse_host_log(base, rel, pass_pattern, module_pattern, max_group=None):
    path = base/rel
    result = dict(qualified=False, cases=None, selftest_cases=None, binary_roundtrip=None,
        max_start_to_start_milliseconds=None, device_calls=None, module_sha256=None, log_sha256=None)
    if not path.is_file():
        return result
    text = path.read_text(encoding='utf-8-sig')
    lines = [x for x in text.splitlines() if x.strip()]
    matches = [pass_pattern.fullmatch(x) for x in lines]
    matched = [m for m in matches if m]
    if not lines or len(matched) != 1 or matched[0] is not matches[-1]:
        return result
    modules = module_pattern.findall(text)
    if len(modules) != 1:
        return result
    m = matched[0]
    result.update(cases=16, selftest_cases=int(m.group(1)), device_calls=0,
        module_sha256=modules[0], log_sha256=sha(path))
    if max_group is not None:
        value = float(m.group(max_group))
        result['max_start_to_start_milliseconds'] = value
        if value > 2000:
            return template()['monitor_host']
    else:
        result['binary_roundtrip'] = 'PASS'
    result['qualified'] = True
    return result


def parse_monitor_log(base_dir=None):
    base = ROOT if base_dir is None else Path(base_dir)
    return _parse_host_log(base, MONITOR_LOG_REL, MONITOR_PASS_LINE, MONITOR_MODULE_LINE, max_group=2)


def parse_transport_log(base_dir=None):
    base = ROOT if base_dir is None else Path(base_dir)
    return _parse_host_log(base, TRANSPORT_LOG_REL, TRANSPORT_PASS_LINE, TRANSPORT_MODULE_LINE)


def assess(base_dir=None):
    base = ROOT if base_dir is None else Path(base_dir)
    recovery = base/RECOVERY_REL
    freeze = read(recovery/'evidence-freeze-private.json')
    require(freeze.get('kind') == 'G1_SHARED_LIFETIME_MONITOR_TRANSPORT_EVIDENCE_FREEZE', 'evidence freeze kind')
    mon = read(recovery/'monitor-cadence-attribution-private.json')
    require(mon.get('kind') == 'G1_SHARED_LIFETIME_MONITOR_CADENCE_ATTRIBUTION', 'monitor attribution kind')
    tra = read(recovery/'transport-attribution-private.json')
    require(tra.get('kind') == 'G1_SHARED_LIFETIME_POST_EXIT_TRANSPORT_ATTRIBUTION', 'transport attribution kind')
    interval = mon['runtime_interval_seconds_completion_to_completion']
    require(isinstance(interval['max'], (int, float)) and interval['min'] <= interval['median'] <= interval['max'], 'monitor interval ordering')
    require(mon['violation']['required_max_seconds'] == PERIOD_REQUIRED_SECONDS, 'monitor required period')
    monitor_failed = interval['max'] > PERIOD_REQUIRED_SECONDS
    historical_monitor = dict(classification=mon['violation']['classification'],
        period_required_seconds=mon['violation']['required_max_seconds'],
        runtime_loop_records=mon['record_counting']['runtime_loop_records'],
        runtime_intervals=interval['count'], minimum_interval_seconds=interval['min'],
        median_interval_seconds=interval['median'], maximum_interval_seconds=interval['max'],
        samples_over_contract=mon['violation']['samples_over_contract'],
        root_cause=mon['root_cause_decomposition']['dominant_hypothesis'],
        attribution_sha256=sha(recovery/'monitor-cadence-attribution-private.json'))
    fc = tra['failure_classification']
    classification = 'ADB_WRAPPER_TIMEOUT' if fc['adb_wrapper_timeout'] else (
        'ADB_COMMAND_TIMEOUT' if fc['adb_command_timeout'] else 'UNATTRIBUTED')
    historical_transport = dict(classification=classification, failure_operation=tra['failure_operation'],
        configured_timeout_seconds=tra['configured_timeout_seconds'],
        local_partial_bytes=tra['local_partial_bytes'], local_partial_complete=tra['local_partial_complete'],
        transfer_verified_for_container_gzip=fc['transfer_verified_for_container_gzip'],
        post_exit_evidence_incomplete=fc['post_exit_evidence_incomplete'],
        five_role_files_recovered=tra['recovery']['five_role_raw_and_gzip_recovered'],
        attribution_sha256=sha(recovery/'transport-attribution-private.json'))
    post_incomplete = bool(fc['post_exit_evidence_incomplete']) and not bool(fc['transfer_verified_for_container_gzip'])
    pilot_ok = _numerical_pilot(base)
    mh = parse_monitor_log(base)
    th = parse_transport_log(base)
    host_ok = mh['qualified'] and th['qualified']
    gate_path = recovery/GATE_REL.name
    if gate_path.is_file():
        g = read(gate_path)
        require(g.get('result') in ('PASS', 'BLOCKED_DEVICE_OFFLINE'), 'unknown device entry gate result')
        gate = dict(result=g['result'], online_endpoint=g['online_endpoint'] is True,
            same_physical_device_verified=g['same_physical_device_verified'] is True,
            new_formal_processes=g['new_formal_processes'])
        require(isinstance(gate['new_formal_processes'], int) and gate['new_formal_processes'] >= 0, 'gate process count')
    else:
        gate = dict(result=None, online_endpoint=False, same_physical_device_verified=False, new_formal_processes=0)
    gate_pass = gate['result'] == 'PASS' and gate['online_endpoint'] and gate['same_physical_device_verified']
    frozen = freeze['frozen_identities']
    old_arms = {frozen['original_arm_sha256'], frozen['two_role_arm_sha256']}
    probe_ok = hex64(frozen['probe_sha256']) and frozen['probe_sha256'] not in old_arms
    formal_dir = base/FORMAL_REL/'formal-1'
    formal_present = formal_dir.is_dir()
    monitor_contract_met = False
    missed_deadlines = None
    transport_verified = False
    identity_match = False
    if gate['result'] is None:
        blocker = 'DEVICE_ENTRY_GATE_MISSING'
    elif not gate_pass:
        blocker = 'DEVICE_OFFLINE' if gate['result'] == 'BLOCKED_DEVICE_OFFLINE' else 'DEVICE_ENTRY_GATE_NOT_PASS'
    elif not host_ok:
        blocker = 'HOST_REPAIR_NOT_QUALIFIED'
    elif not pilot_ok:
        blocker = 'PILOT_NOT_NUMERICAL'
    elif not formal_present:
        blocker = 'FORMAL_PROCESS_NOT_RUN'
    else:
        blocker = None
        summary_path = formal_dir/'monitor-summary-private.json'
        if summary_path.is_file():
            s = read(summary_path)
            missed_deadlines = s.get('missed_deadlines')
            monitor_contract_met = s.get('contract_met') is True and missed_deadlines == 0
            if not monitor_contract_met:
                blocker = 'MONITOR_CONTRACT_NOT_MET'
        else:
            blocker = 'MONITOR_SUMMARY_MISSING'
        record_path = formal_dir/'transport-record-private.json'
        if record_path.is_file():
            roles = read(record_path).get('roles', {})
            transport_verified = list(roles) == ROLES and all(
                a.get('status') == 'TRANSFER_VERIFIED' and hex64(a.get('host_sha256'))
                and a.get('remote_before_sha256') == a.get('host_sha256') == a.get('remote_after_sha256')
                for a in roles.values())
            if not transport_verified and blocker is None:
                blocker = 'TRANSPORT_TRANSFER_NOT_VERIFIED'
        elif blocker is None:
            blocker = 'TRANSPORT_RECORD_MISSING'
        identity_path = formal_dir/'identity-private.json'
        if identity_path.is_file() and probe_ok:
            b = read(identity_path).get('build', {})
            identity_match = (b.get('probe_sha256') == frozen['probe_sha256']
                and b.get('binary_hashes', {}).get('libHexaBackwardRepairDsp.so') == frozen['dsp_sha256']
                and b.get('binary_hashes', {}).get('libHexaBackwardRepairHost.so') == frozen['host_sha256'])
            if not identity_match and blocker is None:
                blocker = 'CANDIDATE_IDENTITY_MISMATCH'
        elif blocker is None:
            blocker = 'CANDIDATE_IDENTITY_MISMATCH'
    formal_eligible = blocker is None
    result = template()
    result.update(numerical_pilot_verified=pilot_ok, historical_monitor=historical_monitor,
        monitor_contract_failed=monitor_failed, historical_transport=historical_transport,
        post_exit_transport_incomplete=post_incomplete, host_repair_qualified=host_ok,
        monitor_host={k: mh[k] for k in template()['monitor_host']},
        transport_host={k: th[k] for k in template()['transport_host']},
        device_entry_gate=gate, candidate_identity_match=identity_match if formal_present else False,
        formal_process_present=formal_present, formal_monitor_contract_met=monitor_contract_met,
        formal_monitor_missed_deadlines=missed_deadlines, formal_transport_verified=transport_verified,
        formal_eligible=formal_eligible, formal_blocker=blocker, new_device_processes=0)
    return result


def entry_gate(stage='formal-1', base_dir=None, git_porcelain=None):
    require(stage in ('formal-1',), 'unregistered entry stage')
    base = ROOT if base_dir is None else Path(base_dir)
    recovery = base/RECOVERY_REL
    try:
        host_ok = parse_monitor_log(base)['qualified'] and parse_transport_log(base)['qualified']
    except Exception:
        host_ok = False
    if not host_ok:
        return False, 'HOST_REPAIR_NOT_QUALIFIED'
    dry_path = base/DRYRUN_REL
    if not dry_path.is_file():
        return False, 'MONITOR_DRYRUN_MISSING'
    try:
        d = read(dry_path)
        ok = (d.get('contract_met') is True and d.get('monitoring_status') == 'PASS'
            and d.get('missed_deadlines') == 0 and d.get('command_timeouts') == 0
            and d.get('invalid_samples') == 0 and d.get('unknown_states') == 0
            and isinstance(d.get('maximum_start_to_start_milliseconds'), (int, float))
            and d['maximum_start_to_start_milliseconds'] <= 2000
            and isinstance(d.get('sample_count'), int) and d['sample_count'] >= 8
            and d.get('host_timeout_tests_pass') is True)
    except Exception:
        ok = False
    if not ok:
        return False, 'MONITOR_DRYRUN_CONTRACT_FAILED'
    try:
        pilot_ok = _numerical_pilot(base)
    except Exception:
        pilot_ok = False
    if not pilot_ok:
        return False, 'NUMERICAL_PILOT_NOT_VERIFIED'
    try:
        g = read(base/GATE_REL)
        gate_ok = g.get('result') == 'PASS' and g.get('online_endpoint') is True and g.get('same_physical_device_verified') is True
    except Exception:
        gate_ok = False
    if not gate_ok:
        return False, 'DEVICE_ENTRY_GATE_NOT_PASS'
    if (base/FORMAL_REL/'formal-1-attempt-private.json').exists() or (base/FORMAL_REL/'formal-1').exists():
        return False, 'PRIOR_FORMAL_ATTEMPT_EXISTS'
    if git_porcelain is None:
        try:
            git_porcelain = subprocess.run(['git', '-C', str(base), 'status', '--porcelain'],
                capture_output=True, text=True, timeout=60).stdout
        except Exception:
            git_porcelain = 'unavailable'
    if git_porcelain.strip():
        return False, 'SOURCE_NOT_COMMITTED'
    plan_record_path = base/NEW_PLAN_RECORD_REL
    try:
        record = read(plan_record_path)
        runner = base/RUNNER_V3_REL
        ok = runner.is_file() and record.get('runner_sha256') == sha(runner)
    except Exception:
        ok = False
    if not ok:
        return False, 'RUNNER_IDENTITY_MISMATCH'
    try:
        record = read(plan_record_path)
        ok = (record.get('plan') == NEW_PLAN_REL.as_posix() and (base/NEW_PLAN_REL).is_file()
            and record.get('sha256') == sha(base/NEW_PLAN_REL))
    except Exception:
        ok = False
    if not ok:
        return False, 'PLAN_IDENTITY_MISMATCH'
    return True, None


def _hex(seed):
    return hashlib.sha256(seed.encode()).hexdigest()


def _tmp_root(prefix):
    preferred = Path(os.environ.get('TEMP', tempfile.gettempdir()))/'opencode'
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(preferred) if preferred.is_dir() else None))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _fixture(root, formal=True, gate_result='PASS', dryrun=False, plan_and_runner=False):
    anchors = {r: _hex('anchor-'+r) for r in ROLES}
    probe, dsp, host = _hex('probe'), _hex('dsp'), _hex('host')
    _write_json(root/PLAN_REL, dict(anchor_legacy_digests=anchors, frozen=dict(
        original_binary_sha256=dict(probe=_hex('old-probe'), dsp=dsp, host=host),
        two_role_arm_sha256=_hex('two-role'))))
    _write_json(root/PILOT_ASSESSMENT_REL, dict(status='PASS', actual_exit=0, teardown='PASS',
        roles={r: dict(pairs=19648 if r == 'dual' else 9824, classification='PASS',
            numerical_digest=anchors[r], arithmetic_word_mismatches=0, observer_mismatch_words=0)
            for r in ROLES}))
    rec = root/RECOVERY_REL
    _write_json(rec/'evidence-freeze-private.json', dict(
        kind='G1_SHARED_LIFETIME_MONITOR_TRANSPORT_EVIDENCE_FREEZE',
        frozen_identities=dict(probe_sha256=probe, dsp_sha256=dsp, host_sha256=host,
            original_arm_sha256=_hex('orig-arm'), two_role_arm_sha256=_hex('two-role'))))
    _write_json(rec/'monitor-cadence-attribution-private.json', dict(
        kind='G1_SHARED_LIFETIME_MONITOR_CADENCE_ATTRIBUTION',
        record_counting=dict(runtime_loop_records=34),
        runtime_interval_seconds_completion_to_completion=dict(count=33, min=3.250952, median=4.029617, max=5.244854),
        violation=dict(required_max_seconds=2.0, samples_over_contract=33, classification='MONITOR_CONTRACT_FAILED'),
        root_cause_decomposition=dict(dominant_hypothesis='ADDITIVE_WAIT_PLUS_SERIAL_QUERY_BLOCK')))
    _write_json(rec/'transport-attribution-private.json', dict(
        kind='G1_SHARED_LIFETIME_POST_EXIT_TRANSPORT_ATTRIBUTION',
        failure_operation='adb pull of the device-gzipped 5-role container',
        configured_timeout_seconds=180, local_partial_bytes=332333056, local_partial_complete=False,
        failure_classification=dict(adb_command_timeout=False, adb_wrapper_timeout=True,
            post_exit_evidence_incomplete=True, transfer_verified_for_container_gzip=False, remote_raw_missing=False),
        recovery=dict(five_role_raw_and_gzip_recovered=True, container_reconstructed_losslessly=True),
        role_identity={r: dict(remote_sha256=_hex('raw-'+r), local_sha256=_hex('raw-'+r),
            remote_gzip_sha256=_hex('gz-'+r), local_gzip_sha256=_hex('gz-'+r),
            raw_identity_verified=True, gzip_identity_verified=True) for r in ROLES}))
    monitor_module, transport_module = _hex('monitor-module'), _hex('transport-module')
    _write_text(rec/'host-tests/monitor-host-test.log',
        'monitor_module_sha256=%s\nbackward_formal_monitor_host=PASS cases=16 selftest_cases=42 realistic_period=2000 max_start_to_start_ms=12.5 device_calls=0\n' % monitor_module)
    _write_text(rec/'host-tests/transport-host-test.log',
        'transport_module_sha256=%s\nbackward_formal_transport_host=PASS cases=16 selftest_cases=38 binary_roundtrip=PASS device_calls=0\n' % transport_module)
    if gate_result is not None:
        _write_json(rec/'device-entry-gate-private.json', dict(result=gate_result,
            online_endpoint=gate_result == 'PASS', same_physical_device_verified=gate_result == 'PASS',
            new_formal_processes=1 if gate_result == 'PASS' else 0, reason='synthetic'))
    if formal:
        d = root/FORMAL_REL/'formal-1'
        _write_json(d/'identity-private.json', dict(build=dict(probe_sha256=probe, binary_hashes={
            'libHexaBackwardRepairDsp.so': dsp, 'libHexaBackwardRepairHost.so': host})))
        _write_json(d/'monitor-summary-private.json', dict(contract_met=True, missed_deadlines=0))
        _write_json(d/'transport-record-private.json', dict(roles={r: dict(status='TRANSFER_VERIFIED',
            remote_before_sha256=_hex('raw-'+r), host_sha256=_hex('raw-'+r), remote_after_sha256=_hex('raw-'+r))
            for r in ROLES}))
    if dryrun:
        _write_json(rec/'monitor-dryrun-private.json', dict(contract_met=True, monitoring_status='PASS',
            missed_deadlines=0, command_timeouts=0, invalid_samples=0, unknown_states=0,
            maximum_start_to_start_milliseconds=12.5, sample_count=16, host_timeout_tests_pass=True))
    if plan_and_runner:
        _write_text(root/RUNNER_V3_REL, 'Write-Host synthetic runner\n')
        _write_text(root/NEW_PLAN_REL, '{"schema":1,"kind":"synthetic"}\n')
        _write_json(rec/'new-plan-sha256.json', dict(plan=NEW_PLAN_REL.as_posix(),
            sha256=sha(root/NEW_PLAN_REL), runner_sha256=sha(root/RUNNER_V3_REL)))
    return dict(probe=probe, dsp=dsp, host=host, monitor_module=monitor_module,
        transport_module=transport_module, anchors=anchors, two_role=_hex('two-role'))


def selftest():
    roots = []
    cases = 0

    def fresh(**kwargs):
        r = _tmp_root('amtr-')
        roots.append(r)
        _fixture(r, **kwargs)
        return r

    try:
        good = fresh()
        v = assess(good)
        validate(v)
        require(v['formal_eligible'] is True and v['formal_blocker'] is None
            and v['monitor_contract_failed'] is True and v['post_exit_transport_incomplete'] is True
            and v['host_repair_qualified'] is True and v['numerical_pilot_verified'] is True,
            'synthetic baseline must qualify')
        cases += 1

        def expect_reject(root, blocker=None, **flips):
            nonlocal cases
            v = assess(root)
            validate(v)
            require(v['formal_eligible'] is False, 'formal eligibility must stay false')
            if blocker is not None:
                require(v['formal_blocker'] == blocker, 'expected blocker ' + blocker)
            for key, value in flips.items():
                path = key.split('.')
                node = v
                for part in path[:-1]:
                    node = node[part]
                require(node[path[-1]] == value, 'expected verdict ' + key)
            cases += 1

        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        del p['roles']['dual']
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        p['roles']['dual']['numerical_digest'] = 'd'*64
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        p['roles']['dual']['arithmetic_word_mismatches'] = 1
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        p['roles']['dual']['observer_mismatch_words'] = 1
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        p['actual_exit'] = 4
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        p = read(r/PILOT_ASSESSMENT_REL)
        p['teardown'] = None
        _write_json(r/PILOT_ASSESSMENT_REL, p)
        expect_reject(r, 'PILOT_NOT_NUMERICAL', numerical_pilot_verified=False)
        r = fresh()
        (r/FORMAL_REL/'formal-1/monitor-summary-private.json').unlink()
        expect_reject(r, 'MONITOR_SUMMARY_MISSING', formal_monitor_contract_met=False)
        r = fresh()
        _write_json(r/FORMAL_REL/'formal-1/monitor-summary-private.json', dict(contract_met=False, missed_deadlines=0))
        expect_reject(r, 'MONITOR_CONTRACT_NOT_MET')
        r = fresh()
        _write_json(r/FORMAL_REL/'formal-1/monitor-summary-private.json', dict(contract_met=True, missed_deadlines=1))
        expect_reject(r, 'MONITOR_CONTRACT_NOT_MET')
        r = fresh()
        _write_json(r/FORMAL_REL/'formal-1/transport-record-private.json', dict(roles={x: dict(
            status='PENDING', remote_before_sha256=_hex('raw-'+x), host_sha256=_hex('raw-'+x),
            remote_after_sha256=_hex('raw-'+x)) for x in ROLES}))
        expect_reject(r, 'TRANSPORT_TRANSFER_NOT_VERIFIED', formal_transport_verified=False)
        r = fresh()
        roles = {x: dict(status='TRANSFER_VERIFIED', remote_before_sha256=_hex('raw-'+x),
            host_sha256=_hex('raw-'+x), remote_after_sha256=_hex('wrong-'+x)) for x in ROLES}
        _write_json(r/FORMAL_REL/'formal-1/transport-record-private.json', dict(roles=roles))
        expect_reject(r, 'TRANSPORT_TRANSFER_NOT_VERIFIED', formal_transport_verified=False)
        r = fresh()
        (r/FORMAL_REL/'formal-1/transport-record-private.json').unlink()
        expect_reject(r, 'TRANSPORT_RECORD_MISSING', formal_transport_verified=False)
        r = fresh()
        (r/FORMAL_REL/'formal-1/identity-private.json').unlink()
        expect_reject(r, 'CANDIDATE_IDENTITY_MISMATCH', candidate_identity_match=False)
        r = fresh()
        _write_json(r/FORMAL_REL/'formal-1/identity-private.json', dict(build=dict(
            probe_sha256=_hex('two-role'), binary_hashes={'libHexaBackwardRepairDsp.so': _hex('dsp'),
            'libHexaBackwardRepairHost.so': _hex('host')})))
        expect_reject(r, 'CANDIDATE_IDENTITY_MISMATCH', candidate_identity_match=False)
        r = fresh()
        log = r/MONITOR_LOG_REL
        lines = [x for x in log.read_text(encoding='utf-8').splitlines() if x.strip()]
        _write_text(log, '\n'.join(lines[:-1])+'\n')
        expect_reject(r, 'HOST_REPAIR_NOT_QUALIFIED', host_repair_qualified=False)
        r = fresh()
        log = r/TRANSPORT_LOG_REL
        text = log.read_text(encoding='utf-8')
        _write_text(log, text.replace('device_calls=0', 'device_calls=1'))
        expect_reject(r, 'HOST_REPAIR_NOT_QUALIFIED', host_repair_qualified=False)
        r = fresh(gate_result=None)
        expect_reject(r, 'DEVICE_ENTRY_GATE_MISSING')
        r = fresh(gate_result='BLOCKED_DEVICE_OFFLINE')
        expect_reject(r, 'DEVICE_OFFLINE', numerical_pilot_verified=True)
        r = fresh(formal=False)
        expect_reject(r, 'FORMAL_PROCESS_NOT_RUN', formal_process_present=False)
        r = fresh()
        v = assess(r)
        v['private_path'] = 'secret'
        try:
            validate(v)
        except ValueError:
            pass
        else:
            raise AssertionError('private field accepted')
        cases += 1
        r = fresh()
        mon = read(r/RECOVERY_REL/'monitor-cadence-attribution-private.json')
        mon['runtime_interval_seconds_completion_to_completion'] = dict(count=33, min=1.5, median=1.7, max=1.9)
        mon['violation']['classification'] = 'MONITOR_CONTRACT_MET'
        _write_json(r/RECOVERY_REL/'monitor-cadence-attribution-private.json', mon)
        v = assess(r)
        validate(v)
        require(v['monitor_contract_failed'] is False, 'monitor verdict must follow attribution')
        cases += 1
        r = fresh()
        tra = read(r/RECOVERY_REL/'transport-attribution-private.json')
        tra['failure_classification'] = dict(adb_command_timeout=False, adb_wrapper_timeout=True,
            post_exit_evidence_incomplete=False, transfer_verified_for_container_gzip=True, remote_raw_missing=False)
        _write_json(r/RECOVERY_REL/'transport-attribution-private.json', tra)
        v = assess(r)
        validate(v)
        require(v['post_exit_transport_incomplete'] is False, 'transport verdict must follow attribution')
        cases += 1
        r = fresh(gate_result='BOGUS_GATE_STATE')
        try:
            assess(r)
        except ValueError:
            pass
        else:
            raise AssertionError('unknown gate result accepted')
        cases += 1

        entry_roots = []

        def entry_fixture(**kwargs):
            r = _tmp_root('amtr-entry-')
            roots.append(r)
            entry_roots.append(r)
            _fixture(r, formal=False, dryrun=True, plan_and_runner=True, **kwargs)
            return r

        e = entry_fixture()
        ok, reason = entry_gate('formal-1', base_dir=e, git_porcelain='')
        require(ok is True and reason is None, 'synthetic entry must pass')
        cases += 1

        def expect_entry(root, expected):
            nonlocal cases
            ok, reason = entry_gate('formal-1', base_dir=root, git_porcelain='')
            require(ok is False and reason == expected, 'expected entry reason ' + expected)
            require(reason in ENTRY_REASONS, 'entry reason outside fixed set')
            cases += 1

        e = entry_fixture()
        (e/MONITOR_LOG_REL).unlink()
        expect_entry(e, 'HOST_REPAIR_NOT_QUALIFIED')
        e = entry_fixture()
        (e/DRYRUN_REL).unlink()
        expect_entry(e, 'MONITOR_DRYRUN_MISSING')
        e = entry_fixture()
        _write_json(e/DRYRUN_REL, dict(contract_met=False, monitoring_status='PASS', missed_deadlines=0,
            command_timeouts=0, invalid_samples=0, unknown_states=0,
            maximum_start_to_start_milliseconds=12.5, sample_count=16, host_timeout_tests_pass=True))
        expect_entry(e, 'MONITOR_DRYRUN_CONTRACT_FAILED')
        e = entry_fixture()
        p = read(e/PILOT_ASSESSMENT_REL)
        p['roles']['dual']['numerical_digest'] = 'd'*64
        _write_json(e/PILOT_ASSESSMENT_REL, p)
        expect_entry(e, 'NUMERICAL_PILOT_NOT_VERIFIED')
        e = entry_fixture(gate_result=None)
        expect_entry(e, 'DEVICE_ENTRY_GATE_NOT_PASS')
        e = entry_fixture()
        (e/FORMAL_REL/'formal-1').mkdir(parents=True)
        expect_entry(e, 'PRIOR_FORMAL_ATTEMPT_EXISTS')
        e = entry_fixture()
        require(entry_gate('formal-1', base_dir=e, git_porcelain=' M scripts/x.py') == (False, 'SOURCE_NOT_COMMITTED'),
            'dirty source entry')
        cases += 1
        e = entry_fixture()
        _write_text(e/RUNNER_V3_REL, 'Write-Host changed\n')
        expect_entry(e, 'RUNNER_IDENTITY_MISMATCH')
        e = entry_fixture()
        _write_text(e/NEW_PLAN_REL, '{"schema":2}\n')
        expect_entry(e, 'PLAN_IDENTITY_MISMATCH')
        print('monitor_transport_assessor_selftest=PASS rejection_cases=' + str(cases) + ' device_calls=0')
    finally:
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--entry', choices=('formal-1',))
    a = p.parse_args()
    if a.self_test:
        selftest()
    elif a.entry is not None:
        ok, reason = entry_gate(a.entry)
        if ok:
            print('entry_gate=PASS stage=' + a.entry + ' device_calls=0')
            sys.exit(0)
        print('ENTRY_BLOCKED: ' + reason)
        sys.exit(3)
    else:
        out = a.out if a.out is not None else ROOT/RECOVERY_REL/'formal-eligibility-private.json'
        require(not out.exists(), 'exclusive create')
        v = assess()
        validate(v)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as f:
            json.dump(v, f, indent=2, allow_nan=False)
        print('monitor_transport_assessor=PASS formal_eligible=' + str(v['formal_eligible']).lower()
            + ' blocker=' + str(v['formal_blocker']) + ' device_calls=0')
