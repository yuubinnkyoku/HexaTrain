"""Strict public projection of this mission's measured evidence, with no permits."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import struct
from assess_backward_repair import ROOT, assess, assess_cohort, read, require
from diagnose_backward_top_state import classify
from export_public_backward_precision import compact, exact_keys
from prepare_saved_p_backward import sha

BASE = 'c6c30c18713084e002ce5485b8f28d67d26faad2'
BRANCH = 'investigation/g1-backward-repair-native-completion'
REPORT = ROOT / 'build/reports/g1-backward-repair-native-completion'
OLD = ROOT / 'build/reports/g1-backward-primitive-precision-repair'
MODES = ['standalone_softmax', 'standalone_dvalue', 'native_softmax', 'native_dvalue', 'dual']
HASHES = dict(DSP='658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5', Host='0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843', probe='23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05')
CLOSEOUT_BASE = 'dffbc50efaec04f59d16bf126f0631d355f99ae7'


def validate_blocked_closeout(value):
    exact_keys(value, 'starting_commit status preflight_reason wakefulness top_resumed preflight_actual_exit started_processes complete_processes graph_executes actual_probe_exit manual_wake_required pilot_reassessment previous_regression_is_historical fresh_regression protected_state performance Formal_profile')
    require(value['starting_commit'] == CLOSEOUT_BASE, 'closeout starting commit')
    expected = dict(status='BLOCKED_FORMAL_COHORT', preflight_reason='DEVICE_NOT_AWAKE_OR_UNKNOWN',
        wakefulness='Dozing', top_resumed='NOT_REACHED_AT_AWAKE_GATE', preflight_actual_exit=2,
        started_processes=0, complete_processes=0, graph_executes=0, actual_probe_exit=None,
        manual_wake_required=True, pilot_reassessment='FIVE_ROLES_PILOT_PASS',
        previous_regression_is_historical=True, protected_state='PASS', performance=None,
        Formal_profile='BLOCKED_SCOPE_NOT_EXECUTED')
    require(all(value[k] == v for k, v in expected.items()), 'unsafe closeout promoted')
    exact_keys(value['fresh_regression'], 'Fast Host Qnn Device_smoke Muon_W8 remaining_device_regressions')
    require(value['fresh_regression']['Fast'] in ('PASS', 'BLOCKED'), 'fresh Fast state')
    require(all(v == 'NOT_EXECUTED' for k, v in value['fresh_regression'].items() if k != 'Fast'), 'unmeasured fresh regression')


def blocked_closeout(directory):
    directory = directory.resolve()
    require(directory.parent == REPORT.resolve(), 'closeout must be a new private mission directory')
    audit = read(directory / 'starting-audit-private.json')
    attempt = read(directory / 'attempt-private.json')
    preflight_path = directory / 'preflight/device-preflight-private.json'
    preflight = read(preflight_path)
    require(audit['status'] == 'PASS' and audit['starting_commit'] == CLOSEOUT_BASE and audit['clean_tree'], 'closeout audit')
    require(audit['prior_evidence_files_unchanged'] == 333, 'prior evidence coverage')
    require(sha(preflight_path) == attempt['preflight_sha256'], 'preflight evidence changed')
    require(preflight['status'] == 'BLOCKED' and preflight['read_only'] is True, 'unsafe preflight')
    require(preflight['device_executes'] == preflight['device_operations_after_stop'] == 0, 'device work after stop')
    require(re.search(r'^\s*mWakefulness=Dozing\s*$', preflight['power_private'], re.M), 'actual Dozing evidence')
    require('awake' not in preflight['gates'] and 'top_resumed' not in preflight['gates'], 'unsafe preflight passed')
    require(attempt['preflight_only'] and attempt['started_probe_processes'] == attempt['completed_cohort_processes'] == attempt['graph_executes'] == 0, 'preflight counted as process')
    require(not (directory / 'identity-private.json').exists() and not (directory / 'stdout.txt').exists(), 'unexpected probe evidence')
    pilots = read(directory / 'pilot-audit-private.json')
    require(set(pilots) == set(MODES), 'missing reassessed pilot')
    entry = read(REPORT / 'cohort-entry-private.json')['measured_gates']
    require({p['digest'] for p in pilots.values()} == {p['digest'] for p in entry.values()}, 'pilot digests changed')
    require(all(p['status'] == 'PILOT_PASS' for p in pilots.values()), 'incomplete pilot audit')
    for mode, pilot in pilots.items():
        measured_role = read(directory / (mode + '-reassessment-private.json'))
        require(measured_role['mode'] == mode and measured_role['classification'] == 'PASS' and
            measured_role['numerical_digest'] == pilot['digest'], 'reassessed pilot role binding')
    for p, h in audit['protected'].items(): require(sha(ROOT / p) == h, 'protected mismatch')
    for p, h in audit['existing_mission_evidence'].items(): require(sha(ROOT / p) == h, 'existing mission evidence changed')
    log = directory / 'fast-closeout.txt'
    fast = 'PASS' if log.exists() and re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$', log.read_text(encoding='utf-8-sig')) else 'BLOCKED'
    result = dict(starting_commit=CLOSEOUT_BASE, status=attempt['status'], preflight_reason=preflight['reason'],
        wakefulness='Dozing', top_resumed=attempt['top_resumed'], preflight_actual_exit=attempt['preflight_actual_exit'],
        started_processes=attempt['started_probe_processes'], complete_processes=attempt['completed_cohort_processes'],
        graph_executes=attempt['graph_executes'], actual_probe_exit=attempt['actual_probe_exit'],
        manual_wake_required=attempt['manual_wake_required'], pilot_reassessment='FIVE_ROLES_PILOT_PASS',
        previous_regression_is_historical=True, fresh_regression=dict(Fast=fast, Host='NOT_EXECUTED', Qnn='NOT_EXECUTED',
            Device_smoke='NOT_EXECUTED', Muon_W8='NOT_EXECUTED', remaining_device_regressions='NOT_EXECUTED'),
        protected_state='PASS', performance=None, Formal_profile=attempt['formal_profile'])
    validate_blocked_closeout(result)
    return result


def empty(status='NOT_EXECUTED'):
    return dict(status=status, actual_exit=None, consumers=[], core_pairs_per_variant=0,
                holdout_pairs_per_variant=0, observer_variants=0, observer_word_mismatches=None,
                arithmetic_word_mismatches=None, numerical_digest=None, processes=0, native_contract=None)


def measured(directory, mode):
    if not (directory / 'identity-private.json').exists():
        return empty('INFRASTRUCTURE_FAILURE' if directory.exists() else 'NOT_EXECUTED')
    identity = read(directory / 'identity-private.json')
    require(identity['mode'] == mode, 'wrong measured role')
    require(read(directory / 'preflight/device-preflight-private.json')['status'] == 'PASS', 'preflight')
    result = assess(directory)
    entry = compact(result, 1)
    entry['actual_exit'] = identity['exit_code']
    entry['native_contract'] = None
    if mode.startswith('native') or mode == 'dual':
        metrics = result['metrics']
        entry['native_contract'] = dict(p_maxAbs=metrics['P_vs_ideal']['maxAbs'], row_sum_error=metrics['P_row_sum']['maxAbs'],
            induced_bounds={str(v): metrics['p_induced_bound_' + str(v)]['maxAbs'] for v in ([0, 1] if mode == 'dual' else [int(mode.endswith('dvalue'))])}, joint_pass=result['forward_joint_pass'])
    return entry


def validate_cohort(value):
    if value.get('status') == 'NOT_EXECUTED':
        exact_keys(value, 'status complete_processes digests')
        require(value == dict(status='NOT_EXECUTED', complete_processes=0, digests={}), 'unverified formal claim')
        return
    exact_keys(value, 'status complete_processes digests attempted_processes failed_processes')
    require(value['status'] == 'FAILED_PARTIAL' and value['complete_processes'] == 0 and value['digests'] == {}, 'failed cohort promoted')
    require(value['attempted_processes'] == 1 and len(value['failed_processes']) == 1, 'failed process accounting')
    failed = value['failed_processes'][0]
    exact_keys(failed, 'source_commit actual_exit preflight focus_takeover_count completed_roles completed_gradient_pairs incomplete_role incomplete_gradient_pairs graph_execute_successes process_teardown partial_raw_bytes completed_role_raw_preserved api_6031_occurrences formal_eligible')
    require(re.fullmatch('[0-9a-f]{40}', failed['source_commit']) is not None, 'failed source commit')
    expected = dict(actual_exit=139, preflight='PASS', focus_takeover_count=0,
        completed_roles=['standalone_softmax'], completed_gradient_pairs=9824,
        incomplete_role='standalone_dvalue', incomplete_gradient_pairs=0, graph_execute_successes=4912,
        process_teardown='INCOMPLETE', partial_raw_bytes=0, completed_role_raw_preserved=True,
        api_6031_occurrences=0, formal_eligible=False)
    require(all(failed[k] == v for k, v in expected.items()), 'unverified failed process claim')


def failed_cohort(attempt_root):
    """Project this measured exit139 episode; never grant numerical certification."""
    attempt_root = attempt_root.resolve()
    require(attempt_root.parent == REPORT.resolve(), 'failed attempt must be private')
    entry = read(attempt_root / 'cohort-execution-entry-private.json')
    directory = REPORT / entry['planned_cohort_directories'][0]
    require(directory.resolve().parent == REPORT.resolve(), 'failed cohort path containment')
    identity = read(directory / 'identity-private.json')
    completion = read(directory / 'execution-before-readback-private.json')
    runner = read(attempt_root / 'cohort-1-runner-exit-private.json')
    require(identity['mode'] == 'cohort' and identity['forward_limit'] == 60, 'failed cohort mode/coverage plan')
    require(identity['exit_code'] == completion['exit_code'] == runner['runner_actual_exit'] == 139, 'actual failed exit missing')
    require(identity['source_sha'] == entry['source_commit'] == runner['source_commit'], 'failed source identity')
    require(identity['build'] == read(ROOT / 'build/fp32-saved-p-backward-repair-final/build-identity.json'), 'failed binary identity')
    require(identity['orchestration_sha256'] == completion['orchestration_sha256'] ==
        'ddaec3470adea03c287ef3138f64a0d04ac9a0ca408d096c24303385d0c9f811', 'failed trace identity')
    require(read(directory / 'preflight/device-preflight-private.json')['status'] == 'PASS' and identity['focus_takeover_count'] == 0, 'failed preflight/monitor')
    require(sha(directory / 'saved-p-words.bin') == identity['raw_evidence_sha256'] and
        sha(directory / 'saved-p-words.bin.gz') == identity['compressed_evidence_sha256'] and
        (directory / 'saved-p-words.bin').stat().st_size == 0, 'partial raw identity')
    text = (directory / 'stdout.txt').read_text(encoding='utf-8-sig')
    stderr = (directory / 'stderr.txt').read_text(encoding='utf-8-sig')
    require(len(re.findall(r'^owned_probe_pid=(\d+)\s*$', text, re.M)) == 1, 'actual failed process missing')
    require(re.findall(r'^cohort_begin,mode=([^\r\n]+)', text, re.M) == MODES[:2], 'failed role order')
    require(re.findall(r'^cohort_end,mode=([^\r\n]+)', text, re.M) == ['standalone_softmax,exit=0'], 'failed completion markers')
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{') and json.loads(line).get('kind') == 'repair_pair']
    keys = [tuple(row[k] for k in ('observed', 'tap', 'forward', 'gradient', 'head', 'consumer')) for row in rows]
    expected = {(o, t, fi, gi, h, 0) for o in (0, 1) for t in (0, 1) for fi in range(60)
        for gi in range(25 if fi < 44 else 8) for h in (0, 1)}
    require(len(keys) == len(set(keys)) == 9824 and set(keys) == expected and
        all(row['mode'] == 'standalone_softmax' for row in rows), 'partial role coverage')
    require('mode=standalone_softmax,pairs=9824,required_failures=0,input_mutation=0,finite=1,complete_writes=1,poison_remaining=0,canary=1,teardown=1,cpu_arithmetic_fallback=0' in text, 'completed role terminal')
    api_lines = [line for line in text.splitlines() if line.startswith('api,')]
    require(all(line.endswith(',0') for line in api_lines) and api_lines.count('api,graphExecute,0') == 4912, 'failed execute accounting')
    unfinished = text.split('cohort_begin,mode=standalone_dvalue\n', 1)[1]
    require('api,graphExecute,' not in unfinished and 'api,graphFinalize,' not in unfinished and
        api_lines[-1] == 'api,graphAddNode,0' and 'Segmentation fault' in stderr, 'failed graph-construction stage')
    require(not re.search(r'(?<!\d)6031(?!\d)', text + stderr), 'exit139 must not hide API6031')
    receipt = read(attempt_root / 'completed-role-readback-private.json')
    raw = attempt_root / 'preserved-completed-roles/standalone_softmax.bin'
    require(receipt['status'] == 'PASS' and receipt['source_role'] == 'standalone_softmax' and
        receipt['actual_process_exit'] == 139 and receipt['physical_identity_verified'] and
        receipt['read_only_evidence_preservation'] and receipt['additional_dsp_executes'] == receipt['pull_actual_exit'] == 0,
        'completed role preservation missing')
    raw_hash = sha(raw)
    require(raw_hash == receipt['remote_before_sha256'] == receipt['local_sha256'] == receipt['remote_after_private'].split()[0] and
        raw.stat().st_size == receipt['local_bytes'] > 16, 'completed role raw hash')
    with raw.open('rb') as source: require(struct.unpack('<4I', source.read(16)) == (0x42525001, 1, 0, 60), 'completed role raw header')
    failed = dict(source_commit=identity['source_sha'], actual_exit=139, preflight='PASS', focus_takeover_count=0,
        completed_roles=['standalone_softmax'], completed_gradient_pairs=9824,
        incomplete_role='standalone_dvalue', incomplete_gradient_pairs=0, graph_execute_successes=4912,
        process_teardown='INCOMPLETE', partial_raw_bytes=0, completed_role_raw_preserved=True,
        api_6031_occurrences=0, formal_eligible=False)
    result = dict(status='FAILED_PARTIAL', complete_processes=0, digests={}, attempted_processes=1, failed_processes=[failed])
    validate_cohort(result)
    return directory.resolve(), result


def validate(value):
    exact_keys(value, 'schema base source_commit branch binary_sha256 fixture_sha256 previous_pilots repair monitor runtime_6031 forward cohort performance regression protected_state classifications training' + (' formal_closeout' if 'formal_closeout' in value else '') + (' idle_npu_reproduction' if 'idle_npu_reproduction' in value else ''))
    if 'formal_closeout' in value: validate_blocked_closeout(value['formal_closeout'])
    if 'idle_npu_reproduction' in value:
        from export_public_backward_idle_npu import validate_idle
        validate_idle(value['idle_npu_reproduction'])
    require(value['schema'] == 1 and value['base'] == BASE and value['branch'] == BRANCH, 'mission identity')
    require(re.fullmatch('[0-9a-f]{40}', value['source_commit']) is not None, 'source commit')
    require(value['binary_sha256'] == HASHES and value['training'] == 'NO-GO' and value['protected_state'] == 'PASS', 'binary/protected/training')
    require(value['fixture_sha256'] == dict(forward='f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2', gradient='c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b'), 'fixture changed')
    for entries in (value['previous_pilots'], value['repair']):
        for mode, entry in entries.items():
            require(mode in MODES, 'unknown role')
            exact_keys(entry, 'status actual_exit consumers core_pairs_per_variant holdout_pairs_per_variant observer_variants observer_word_mismatches arithmetic_word_mismatches numerical_digest processes native_contract')
            require(entry['status'] in ('NOT_EXECUTED', 'INFRASTRUCTURE_FAILURE', 'PILOT_PASS', 'NUMERICAL_OR_WORD_FAIL', 'PASS'), 'role status')
            if entry['status'] in ('NOT_EXECUTED', 'INFRASTRUCTURE_FAILURE'):
                require(entry == empty(entry['status']), 'partial process promoted')
                continue
            require(entry['actual_exit'] in (0, 4), 'missing actual exit')
            require((entry['core_pairs_per_variant'], entry['holdout_pairs_per_variant'], entry['observer_variants']) == (2200, 256, 4), 'coverage/observer downgrade')
            require(entry['processes'] == (3 if entry['status'] == 'PASS' else 1), 'process count')
            require(re.fullmatch('[0-9a-f]{64}', entry['numerical_digest']) is not None, 'digest missing')
            require(len(entry['consumers']) == (2 if mode == 'dual' else 1), 'missing gradient')
            require({c['consumer'] for c in entry['consumers']} == ({0, 1} if mode == 'dual' else {int(mode.endswith('dvalue'))}), 'wrong gradient')
            for c in entry['consumers']:
                exact_keys(c, 'consumer gate core_maxAbs holdout_maxAbs local_maxAbs total_maxAbs p_induced_actual_maxAbs worst')
                require(c['gate'] == (.01 if c['consumer'] else .002), 'gate change')
                require(all(isinstance(c[k], (int, float)) and math.isfinite(c[k]) for k in ('core_maxAbs', 'holdout_maxAbs', 'local_maxAbs', 'total_maxAbs', 'p_induced_actual_maxAbs')), 'missing/nonfinite metric')
                if entry['status'] in ('PASS', 'PILOT_PASS'):
                    require(entry['actual_exit'] == 0 and max(c['total_maxAbs'], c['core_maxAbs'], c['holdout_maxAbs']) <= c['gate'], 'false accuracy PASS')
                    require(entry['observer_word_mismatches'] == entry['arithmetic_word_mismatches'] == 0, 'word mismatch')
            native = mode.startswith('native') or mode == 'dual'
            if native:
                n = entry['native_contract']; exact_keys(n, 'p_maxAbs row_sum_error induced_bounds joint_pass')
                require(n['joint_pass'] and n['p_maxAbs'] <= 1e-6 and n['row_sum_error'] <= 2e-6, 'Forward native gate')
                require(set(n['induced_bounds']) == {str(c['consumer']) for c in entry['consumers']}, 'missing induced bound')
                for consumer, bound in n['induced_bounds'].items():
                    require(math.isfinite(bound) and 0 <= bound <= (.001 if consumer == '1' else .00025), 'P-induced bound')
            else:
                require(entry['native_contract'] is None, 'false native claim')
    require(set(value['repair']) == {'native_dvalue', 'dual'}, 'repair roles')
    require(set(value['previous_pilots']) == set(MODES[:3]), 'missing previous pilot role')
    require(all(e['status'] == 'PILOT_PASS' and e['processes'] == 1 for e in value['previous_pilots'].values()), 'previous pilot promoted to formal')
    require(all(e['status'] != 'PASS' for e in value['repair'].values()), 'formal role without complete cohort')
    validate_cohort(value['cohort'])
    require(value['performance'] is None, 'unmeasured performance')
    exact_keys(value['monitor'], 'diagnostic_commands diagnostic_rounds control_verified full_activity_agreement activity_task_empty alternative_verified historical_response_classification new_unknown_responses safety_waived')
    require(value['monitor']['alternative_verified'] is False and value['monitor']['safety_waived'] is False, 'unverified monitor/waiver')
    for item in value['monitor']['new_unknown_responses']:
        exact_keys(item, 'classification response_characters top_field_count')
        require(item['classification'] != 'TOP_RESUMED_VERIFIED', 'unknown promoted')
    exact_keys(value['runtime_6031'], 'state new_api_6031_occurrences root_cause')
    require(value['runtime_6031']['state'] == 'UNRESOLVED_DORMANT_WATCH' and value['runtime_6031']['root_cause'] == 'UNRESOLVED', 'runtime certainty')
    exact_keys(value['forward'], 'status p_maxAbs ds_bound dv_bound observer_word_mismatches')
    exact_keys(value['regression'], 'Fast Host Qnn Device_smoke Muon_W8 remaining_device_regressions Formal_profile')
    for state in value['regression'].values(): require(state in ('PASS', 'NOT_EXECUTED', 'BLOCKED'), 'gate status')
    allowed = {'TOP_RESUMED_MONITOR_CHARACTERIZED', 'BLOCKED_TOP_RESUMED_STATE', 'NATIVE_DVALUE_SINGLE_VERIFIED', 'SAVED_P_DUAL_NATIVE_VERIFIED', 'SAVED_P_LIFETIME_VERIFIED', 'SAVED_P_DUAL_OBSERVER_FREE_VERIFIED', 'BLOCKED_NATIVE_DVALUE', 'BLOCKED_NATIVE_DUAL', 'BLOCKED_FORMAL_COHORT', 'UNRESOLVED'}
    require(set(value['classifications']) <= allowed, 'unsupported classification')
    require(('NATIVE_DVALUE_SINGLE_VERIFIED' in value['classifications']) == (value['repair']['native_dvalue']['status'] == 'PILOT_PASS'), 'Native classification')
    require(('SAVED_P_DUAL_NATIVE_VERIFIED' in value['classifications']) == (value['repair']['dual']['status'] == 'PILOT_PASS'), 'dual classification')
    if value['monitor']['new_unknown_responses']: require('BLOCKED_TOP_RESUMED_STATE' in value['classifications'], 'unknown hidden')
    def safe(x):
        if isinstance(x, dict):
            for k, v in x.items(): safe(k); safe(v)
        elif isinstance(x, list):
            for v in x: safe(v)
        elif isinstance(x, str):
            require(not re.search(r'[A-Za-z]:[/\\]|(?:^|\s)/(?:data|Users|home|tmp)/|\b\d{1,3}(?:\.\d{1,3}){3}:\d+|com\.[a-z]+\.', x), 'private identity/path')
        elif isinstance(x, float): require(math.isfinite(x), 'nonfinite public')
    safe(value)


def make(failed_attempt_root=None):
    initial = read(REPORT / 'starting-audit-private.json')
    for line in initial['refs'].splitlines():
        value, ref = line.split()
        if ref != 'refs/heads/' + BRANCH:
            require(subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip() == value, 'protected ref mismatch')
    for p, h in initial['protected'].items(): require(sha(ROOT / p) == h, 'protected mismatch')
    for name, bundle in initial['bundles'].items():
        current = read(ROOT / 'build' / name / 'build-identity.json')
        for p, h in current['source_hashes'].items(): require(sha(ROOT / p) == h, 'frozen source')
        for p, h in bundle['binary_hashes'].items(): require(sha(ROOT / 'build' / name / p) == h, 'frozen binary')
    prior = {mode: measured(OLD / folder, mode) for mode, folder in [('standalone_softmax', 'pilot-softmax-recovery'), ('standalone_dvalue', 'pilot-dvalue-recovery4'), ('native_softmax', 'pilot-native-softmax-trace1')]}
    repair = {mode: measured(REPORT / folder, mode) for mode, folder in [('native_dvalue', 'pilot-native-dvalue-1'), ('dual', 'pilot-dual-1')]}
    diagnostic = read(REPORT / 'monitor-diagnosis-1/summary-private.json')
    unknowns = []
    for f in REPORT.glob('*/top-unknown-*-private.json'):
        r = read(f); text = r['actual_response_private']
        unknowns.append(dict(classification=classify(0, text)['classification'], response_characters=len(text), top_field_count=text.count('topResumedActivity')))
    rounds = diagnostic['rounds']
    monitor = dict(diagnostic_commands=diagnostic['commands'], diagnostic_rounds=len(rounds), control_verified=sum(r['safety_verified'] for r in rounds),
        full_activity_agreement=sum(c['parser_private']['identity'] == r['parser_private']['identity'] and c['parser_private']['classification'] == 'TOP_RESUMED_VERIFIED' for r in rounds for c in r['comparisons'] if c['service'] == 'activity'),
        activity_task_empty=sum(c['parser_private']['classification'] == 'EMPTY_RESPONSE' for r in rounds for c in r['comparisons'] if c['service'] == 'activity_task'),
        alternative_verified=False, historical_response_classification='VALID_RESPONSE_NO_FIELD', new_unknown_responses=unknowns, safety_waived=False)
    joint = read(REPORT / 'forward-control-2/joint-assessment-private.json')
    require(joint['status'] == 'PASS' and joint['joint']['joint_pass'], 'Forward gate')
    api6031 = sum(len(re.findall(r'^api,[^,\r\n]+,6031\s*$', f.read_text(encoding='utf-8-sig'), re.M)) for f in REPORT.glob('*/stdout.txt'))
    gates = {}
    for profile in ('Fast', 'Host', 'Qnn'):
        log = REPORT / {'Fast':'fast-forward-assessor.txt', 'Host':'host-profile.txt', 'Qnn':'qnn-profile.txt'}[profile]
        gates[profile] = 'PASS' if log.exists() and re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$', log.read_text(encoding='utf-8-sig')) else 'BLOCKED'
    result = dict(schema=1, base=BASE, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), branch=BRANCH,
        binary_sha256=HASHES, fixture_sha256=dict(forward=sha(ROOT/'build/fp32-double-qk/double-inputs.bin'), gradient=sha(ROOT/'build/reports/g1-native-saved-p-backward-dsp/gradients.bin')), previous_pilots=prior, repair=repair, monitor=monitor,
        runtime_6031=dict(state='UNRESOLVED_DORMANT_WATCH', new_api_6031_occurrences=api6031, root_cause='UNRESOLVED'),
        forward=dict(status='PASS', p_maxAbs=joint['joint']['p_maxAbs'], ds_bound=joint['joint']['ds_bound'], dv_bound=joint['joint']['dv_bound'], observer_word_mismatches=joint['joint']['variant_word_mismatches']),
        cohort=dict(status='NOT_EXECUTED', complete_processes=0, digests={}), performance=None,
        regression=dict(**gates, Device_smoke='NOT_EXECUTED', Muon_W8='NOT_EXECUTED', remaining_device_regressions='NOT_EXECUTED', Formal_profile='NOT_EXECUTED'),
        protected_state='PASS', classifications=['TOP_RESUMED_MONITOR_CHARACTERIZED', 'BLOCKED_FORMAL_COHORT', 'UNRESOLVED'], training='NO-GO')
    result['classifications'].append('NATIVE_DVALUE_SINGLE_VERIFIED' if repair['native_dvalue']['status'] == 'PILOT_PASS' else 'BLOCKED_NATIVE_DVALUE')
    if repair['dual']['status'] == 'PILOT_PASS':
        result['classifications'] += ['SAVED_P_DUAL_NATIVE_VERIFIED', 'SAVED_P_LIFETIME_VERIFIED', 'SAVED_P_DUAL_OBSERVER_FREE_VERIFIED']
    else:
        result['classifications'].append('BLOCKED_NATIVE_DUAL')
    actual = {p.parent.resolve() for pattern in ('cohort-*/identity-private.json',
        'cohort-*/execution-before-readback-private.json', 'cohort-*/stdout.txt') for p in REPORT.glob(pattern)}
    if actual:
        require(failed_attempt_root is not None, 'actual or partial cohort evidence requires explicit projection')
        directory, result['cohort'] = failed_cohort(failed_attempt_root)
        require(actual == {directory}, 'unaccounted actual cohort process')
    else:
        require(failed_attempt_root is None, 'failed attempt without actual process')
    require(not (REPORT / 'performance-1/identity-private.json').exists(), 'performance requires measured projection')
    if unknowns: result['classifications'].insert(1, 'BLOCKED_TOP_RESUMED_STATE')
    validate(result)
    return result


def selftest():
    # No live device or live evidence consumed by the rejection fixtures.
    def pilot(mode):
        consumer = int(mode.endswith('dvalue'))
        r = empty('PILOT_PASS')
        r.update(actual_exit=0, core_pairs_per_variant=2200, holdout_pairs_per_variant=256, observer_variants=4,
                 observer_word_mismatches=0, arithmetic_word_mismatches=0, numerical_digest='a'*64, processes=1,
                 consumers=[dict(consumer=consumer, gate=.01 if consumer else .002, core_maxAbs=0., holdout_maxAbs=0., local_maxAbs=0., total_maxAbs=0., p_induced_actual_maxAbs=0., worst=[0]*7)])
        if mode.startswith('native'):
            r['native_contract'] = dict(p_maxAbs=0., row_sum_error=0., induced_bounds={str(consumer):0.}, joint_pass=True)
        return r
    value = dict(schema=1, base=BASE, source_commit='a'*40, branch=BRANCH, binary_sha256=HASHES.copy(), fixture_sha256=dict(forward='f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2', gradient='c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b'),
        previous_pilots={m:pilot(m) for m in MODES[:3]}, repair={'native_dvalue':empty(), 'dual':empty()},
        monitor=dict(diagnostic_commands=25, diagnostic_rounds=3, control_verified=3, full_activity_agreement=3, activity_task_empty=3, alternative_verified=False, historical_response_classification='VALID_RESPONSE_NO_FIELD', new_unknown_responses=[], safety_waived=False),
        runtime_6031=dict(state='UNRESOLVED_DORMANT_WATCH', new_api_6031_occurrences=0, root_cause='UNRESOLVED'),
        forward=dict(status='PASS', p_maxAbs=0., ds_bound=0., dv_bound=0., observer_word_mismatches=0),
        cohort=dict(status='NOT_EXECUTED', complete_processes=0, digests={}), performance=None,
        regression={k:'NOT_EXECUTED' for k in ('Fast','Host','Qnn','Device_smoke','Muon_W8','remaining_device_regressions','Formal_profile')}, protected_state='PASS',
        classifications=['TOP_RESUMED_MONITOR_CHARACTERIZED','BLOCKED_NATIVE_DVALUE','BLOCKED_NATIVE_DUAL','BLOCKED_FORMAL_COHORT','UNRESOLVED'], training='NO-GO')
    validate(value)
    rejected = 0
    def reject(change):
        nonlocal rejected
        v = copy.deepcopy(value); change(v)
        try: validate(v)
        except (ValueError, TypeError): rejected += 1
        else: raise AssertionError('false public evidence accepted')
    reject(lambda v:v.update(raw_tensor=[]))
    reject(lambda v:v['binary_sha256'].update(probe='b'*64))
    reject(lambda v:v['previous_pilots']['native_softmax'].update(actual_exit=None))
    reject(lambda v:v['previous_pilots']['native_softmax'].update(core_pairs_per_variant=2199))
    reject(lambda v:v['previous_pilots']['native_softmax'].update(observer_variants=3))
    reject(lambda v:v['previous_pilots']['native_softmax'].update(observer_word_mismatches=1))
    reject(lambda v:v['previous_pilots']['native_softmax']['consumers'][0].update(total_maxAbs=1.))
    reject(lambda v:v['previous_pilots']['native_softmax']['consumers'][0].update(local_maxAbs=None))
    reject(lambda v:v['previous_pilots']['native_softmax']['consumers'][0].update(gate=1.))
    reject(lambda v:v['previous_pilots']['native_softmax'].update(status='PASS', processes=3))
    reject(lambda v:v['repair']['native_dvalue'].update(core_pairs_per_variant=2200))
    reject(lambda v:v['monitor'].update(alternative_verified=True))
    reject(lambda v:v['monitor'].update(safety_waived=True))
    reject(lambda v:v['cohort'].update(status='PASS', complete_processes=3))
    reject(lambda v:v.update(performance=[dict(median_us=1.)]))
    reject(lambda v:v['classifications'].append('NATIVE_DVALUE_SINGLE_VERIFIED'))
    reject(lambda v:v.update(old_negative_controls='OVERWRITTEN'))
    reject(lambda v:v['monitor']['new_unknown_responses'].append(dict(classification='TOP_RESUMED_VERIFIED',response_characters=100,top_field_count=0)))
    print(f'native_completion_exporter_selftest=PASS rejection_cases={rejected} device_calls=0')
    closeout = dict(starting_commit=CLOSEOUT_BASE, status='BLOCKED_FORMAL_COHORT',
        preflight_reason='DEVICE_NOT_AWAKE_OR_UNKNOWN', wakefulness='Dozing', top_resumed='NOT_REACHED_AT_AWAKE_GATE',
        preflight_actual_exit=2, started_processes=0, complete_processes=0, graph_executes=0, actual_probe_exit=None,
        manual_wake_required=True, pilot_reassessment='FIVE_ROLES_PILOT_PASS', previous_regression_is_historical=True,
        fresh_regression=dict(Fast='PASS', Host='NOT_EXECUTED', Qnn='NOT_EXECUTED', Device_smoke='NOT_EXECUTED',
            Muon_W8='NOT_EXECUTED', remaining_device_regressions='NOT_EXECUTED'), protected_state='PASS',
        performance=None, Formal_profile='BLOCKED_SCOPE_NOT_EXECUTED')
    validate_blocked_closeout(closeout)
    corruptions = [dict(status='PASS'), dict(wakefulness='Awake'), dict(top_resumed='PASS'),
        dict(started_processes=1), dict(complete_processes=3), dict(graph_executes=1), dict(actual_probe_exit=0),
        dict(manual_wake_required=False), dict(protected_state='FAIL'), dict(performance={'median':1}),
        dict(fresh_regression=dict(closeout['fresh_regression'], Muon_W8='PASS')),
        dict(previous_regression_is_historical=False), dict(Formal_profile='PASS'), dict(raw_activity='private')]
    for change in corruptions:
        try: validate_blocked_closeout(dict(closeout, **change))
        except (ValueError, TypeError): pass
        else: raise AssertionError('false blocked closeout accepted')
    print(f'blocked_closeout_exporter_selftest=PASS rejection_cases={len(corruptions)} device_calls=0')
    partial = dict(status='FAILED_PARTIAL', complete_processes=0, digests={}, attempted_processes=1,
        failed_processes=[dict(source_commit='a'*40, actual_exit=139, preflight='PASS', focus_takeover_count=0,
            completed_roles=['standalone_softmax'], completed_gradient_pairs=9824, incomplete_role='standalone_dvalue',
            incomplete_gradient_pairs=0, graph_execute_successes=4912, process_teardown='INCOMPLETE', partial_raw_bytes=0,
            completed_role_raw_preserved=True, api_6031_occurrences=0, formal_eligible=False)])
    validate_cohort(partial)
    mutations = [dict(status='PASS'), dict(complete_processes=1), dict(attempted_processes=0), dict(digests={'dual':'a'*64})]
    role_mutations = [dict(actual_exit=None), dict(actual_exit=0), dict(preflight='BLOCKED'), dict(formal_eligible=True),
        dict(process_teardown='PASS'), dict(completed_roles=['dual']), dict(completed_gradient_pairs=9823),
        dict(incomplete_gradient_pairs=1), dict(completed_role_raw_preserved=False), dict(api_6031_occurrences=1)]
    for change in mutations:
        try: validate_cohort(dict(partial, **change))
        except (ValueError, TypeError): pass
        else: raise AssertionError('failed cohort promoted')
    for change in role_mutations:
        candidate = copy.deepcopy(partial); candidate['failed_processes'][0].update(change)
        try: validate_cohort(candidate)
        except (ValueError, TypeError): pass
        else: raise AssertionError('failed process promoted')
    # These role entries would otherwise claim three formal replicas alongside
    # the failed cohort's zero complete processes.
    for mode, granted in (('native_dvalue', 'NATIVE_DVALUE_SINGLE_VERIFIED'), ('dual', 'SAVED_P_DUAL_NATIVE_VERIFIED')):
        candidate = copy.deepcopy(value); candidate['cohort'] = copy.deepcopy(partial)
        role = pilot('native_dvalue' if mode == 'native_dvalue' else 'native_softmax')
        if mode == 'dual':
            role['consumers'] += pilot('native_dvalue')['consumers']
            role['native_contract']['induced_bounds'] = {'0':0., '1':0.}
        candidate['repair'][mode] = role; candidate['classifications'].append(granted)
        validate(candidate)
        role.update(status='PASS', processes=3); candidate['classifications'].remove(granted)
        try: validate(candidate)
        except (ValueError, TypeError): pass
        else: raise AssertionError('formal role without cohort accepted')
    candidate = copy.deepcopy(value); candidate['cohort'] = copy.deepcopy(partial)
    del candidate['previous_pilots']['native_softmax']
    try: validate(candidate)
    except (ValueError, TypeError): pass
    else: raise AssertionError('missing prerequisite pilot accepted')
    print(f'failed_cohort_exporter_selftest=PASS rejection_cases={len(mutations)+len(role_mutations)+3} device_calls=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true'); parser.add_argument('--blocked-closeout-root', type=Path); parser.add_argument('--failed-attempt-root', type=Path); args = parser.parse_args()
    if args.self_test:
        selftest(); raise SystemExit(0)
    require(args.output is not None, 'output required')
    result = make(args.failed_attempt_root)
    if args.blocked_closeout_root:
        result['formal_closeout'] = blocked_closeout(args.blocked_closeout_root)
        validate(result)
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print('native_completion_public=PASS formal_processes=0 training=NO-GO')
