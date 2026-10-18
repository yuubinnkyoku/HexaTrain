"""Allow-list export for the standalone stopping boundary (never native DSP PASS)."""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import re
import subprocess
import tempfile
from assess_saved_p_backward import assess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def keys(value, allowed):
    if not isinstance(value, dict) or set(value) != set(allowed.split()):
        raise ValueError('missing or forbidden field')


def validate(value, expected_probe):
    keys(value, 'schema source_commit candidate_source_commit branch base forward_base plan_sha256 probe_sha256 gradient_sha256 classifications standalone native coverage_complete cohort_processes performance regression training limitations')
    if value['schema'] != 1 or value['base'] != 'be4271efd1b8fb71f0c66f3ae7c35f04571aa78e' or value['training'] != 'NO-GO':
        raise ValueError('provenance/training contract')
    if value['probe_sha256'] != expected_probe or not re.fullmatch('[0-9a-f]{64}', expected_probe):
        raise ValueError('wrong binary identity')
    if value['cohort_processes'] != 0 or value['performance'] is not None:
        raise ValueError('standalone pilot exporter cannot certify cohort/performance')
    keys(value['native'], 'softmax dvalue dual lifetime observer_free sensitivity')
    if any(v != 'NOT_EXECUTED' for v in value['native'].values()):
        raise ValueError('false native DSP PASS or observer downgrade')
    allowed_classes = {'UNRESOLVED', 'BLOCKED_SOFTMAX_BACKWARD_ACCURACY', 'BLOCKED_DVALUE_ACCURACY', 'BLOCKED_OBSERVER_INDEPENDENCE', 'BLOCKED_DEVICE_SAFETY'}
    if not value['classifications'] or not set(value['classifications']) <= allowed_classes:
        raise ValueError('unsupported DSP classification')
    keys(value['standalone'], 'softmax dvalue')
    all_complete = True
    for name, entry in value['standalone'].items():
        keys(entry, 'status maxAbs worst core_pairs_per_variant holdout_pairs_per_variant observer_variants failed_pairs accuracy_failed_pairs observer_mismatch_pairs numerical_digest actual_gradient_gate ordered_fp32_is_word_gate independent_python_double_recheck process_exit_evidence')
        if entry['status'] not in ('PASS', 'NUMERICAL_FAIL', 'OBSERVER_FAIL', 'UNRESOLVED'):
            raise ValueError('invalid status')
        if entry['process_exit_evidence'] not in ('CAPTURED', 'UNKNOWN_AFTER_COMPLETE_FAILURE_SUMMARY') or (entry['status']=='PASS' and entry['process_exit_evidence']!='CAPTURED'):
            raise ValueError('missing process completion evidence for PASS')
        complete = entry['core_pairs_per_variant'] == 2200 and entry['holdout_pairs_per_variant'] == 256 and entry['observer_variants'] == 4
        all_complete &= complete
        if entry['status'] != 'UNRESOLVED' and not complete:
            raise ValueError('incomplete coverage cannot be terminal arithmetic result')
        if entry['ordered_fp32_is_word_gate'] is not False or entry['actual_gradient_gate'] != (.002 if name == 'softmax' else .01):
            raise ValueError('gate mutation')
        if entry['status'] != 'UNRESOLVED':
            if not isinstance(entry['maxAbs'], (int, float)) or not math.isfinite(entry['maxAbs']) or entry['independent_python_double_recheck'] != 'PASS':
                raise ValueError('missing required numerical evidence')
            if entry['status'] == 'PASS' and (entry['maxAbs'] > entry['actual_gradient_gate'] or entry['failed_pairs']):
                raise ValueError('unexpected numerical FAIL masking')
            if entry['status'] == 'NUMERICAL_FAIL' and not entry['accuracy_failed_pairs']:
                raise ValueError('missing failure evidence')
            if entry['status'] == 'OBSERVER_FAIL' and (entry['accuracy_failed_pairs'] or not entry['observer_mismatch_pairs']):
                raise ValueError('observer attribution')
            keys(entry['worst'], 'forward head gradient coordinate observed tap')
    if value['coverage_complete'] != all_complete:
        raise ValueError('coverage claim inconsistent')
    for field, status in value['regression'].items():
        if not re.fullmatch('[A-Za-z0-9_]+', field) or status not in ('PASS', 'FAIL', 'BLOCKED', 'NOT_EXECUTED', 'HISTORICAL_ONLY'):
            raise ValueError('unsafe regression field')

    def safe(item):
        if isinstance(item, dict):
            for key, child in item.items():
                safe(key); safe(child)
        elif isinstance(item, list):
            for child in item:
                safe(child)
        elif isinstance(item, str):
            if re.search(r'[A-Za-z]:[/\\]|(?:^|\s)/(?:Users|data|home|tmp)/|\b\d{1,3}(?:\.\d{1,3}){3}:\d+', item):
                raise ValueError('forbidden local path/endpoint')
    safe(value)


def make(report):
    initial = read(report / 'initial-private.json')
    build = read(ROOT / 'build/fp32-saved-p-backward-compressed-final/build-identity.json')
    if sha(ROOT / 'build/fp32-saved-p-backward-compressed-final/saved_p_backward_probe') != build['probe_sha256']:
        raise ValueError('wrong current binary')
    for path, digest in build['source_hashes'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('candidate source changed')
    if initial['base'] != 'be4271efd1b8fb71f0c66f3ae7c35f04571aa78e':
        raise ValueError('wrong research base')
    output = {'schema': 1, 'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'candidate_source_commit': build['source_sha'], 'branch': 'investigation/g1-native-saved-p-backward-dsp',
              'base': initial['base'], 'forward_base': 'e05f3539e2e0c0e461157a3abf1d9833d74b392b',
              'plan_sha256': build['plan_sha256'], 'probe_sha256': build['probe_sha256'], 'gradient_sha256': build['gradient_sha256'],
              'classifications': ['UNRESOLVED'], 'standalone': {},
              'native': {name: 'NOT_EXECUTED' for name in ('softmax', 'dvalue', 'dual', 'lifetime', 'observer_free', 'sensitivity')},
              'coverage_complete': True, 'cohort_processes': 0, 'performance': None,
              'regression': read(report / 'regression-status-private.json'), 'training': 'NO-GO',
              'limitations': ['Fixed G1 diagnostic geometry and reference-derived frozen gradients.', 'No native Saved-P or full Attention/layer/training certification.']}
    for name, mode in (('softmax', 'standalone_softmax'), ('dvalue', 'standalone_dvalue')):
        process = report / ('pilot-' + name)
        identity = read(process / 'identity-private.json')
        if identity['build']['probe_sha256'] != build['probe_sha256'] or identity['forward_limit'] != 60 or identity['focus_takeover_count']:
            raise ValueError('wrong binary, coverage, or unsafe process')
        preflight = read(process / 'preflight/device-preflight-private.json')
        if preflight['status'] != 'PASS' or preflight['device_executes'] != 0:
            raise ValueError('missing safe preflight')
        if sha(process / 'saved-p-words.bin') != identity['raw_evidence_sha256']:
            raise ValueError('raw evidence identity mismatch')
        result = assess(process / 'stdout.txt', mode, 60)
        expected_exit = 4 if result['required_failed_pairs'] else 0
        recovered_failure = (result['classification']=='NUMERICAL_FAIL' and identity['exit_code'] is None and
                             identity.get('declared_probe_exit')==4 and identity.get('readback_recovered') is True and
                             identity.get('exit_status')=='UNKNOWN_OUTER_ADB_COMMAND_FAILURE_AFTER_COMPLETE_PROBE_SUMMARY')
        if identity['exit_code'] != expected_exit and not recovered_failure:
            raise ValueError('unexpected exit cannot be expected negative control')
        if any(c != {'core': 2200, 'holdout': 256} for c in result['coverage'].values()) or len(result['coverage']) != 4:
            raise ValueError('coverage incomplete')
        worst = result['worst']['actual_vs_received_input_double']
        o, t, fi, gi, h = worst['case']
        output['standalone'][name] = {
            'status': result['classification'], 'maxAbs': worst['maxAbs'],
            'worst': {'forward': fi, 'head': h, 'gradient': gi, 'coordinate': worst['coordinate'], 'observed': o, 'tap': t},
            'core_pairs_per_variant': 2200, 'holdout_pairs_per_variant': 256, 'observer_variants': 4,
            'failed_pairs': result['required_failed_pairs'], 'numerical_digest': result['numerical_digest'],
            'accuracy_failed_pairs': result['accuracy_failed_pairs'], 'observer_mismatch_pairs': result['observer_mismatch_pairs'],
            'actual_gradient_gate': .002 if name == 'softmax' else .01, 'ordered_fp32_is_word_gate': False,
            'process_exit_evidence': 'UNKNOWN_AFTER_COMPLETE_FAILURE_SUMMARY' if recovered_failure else 'CAPTURED',
            'independent_python_double_recheck': result['independent_python_double_recheck']}
        if result['classification'] == 'NUMERICAL_FAIL':
            output['classifications'].append('BLOCKED_SOFTMAX_BACKWARD_ACCURACY' if name == 'softmax' else 'BLOCKED_DVALUE_ACCURACY')
        if result['observer_mismatch_pairs'] and 'BLOCKED_OBSERVER_INDEPENDENCE' not in output['classifications']:
            output['classifications'].append('BLOCKED_OBSERVER_INDEPENDENCE')
    validate(output, build['probe_sha256'])
    return output


def selftest():
    # Synthetic validator fixtures only, never device evidence.
    entry = {'status': 'NUMERICAL_FAIL', 'maxAbs': .1, 'worst': dict(forward=0, head=0, gradient=0, coordinate=0, observed=1, tap=1),
             'core_pairs_per_variant': 2200, 'holdout_pairs_per_variant': 256, 'observer_variants': 4, 'failed_pairs': 1,
             'accuracy_failed_pairs': 1, 'observer_mismatch_pairs': 0,
             'process_exit_evidence': 'CAPTURED',
             'numerical_digest': 'a'*64, 'actual_gradient_gate': .002, 'ordered_fp32_is_word_gate': False, 'independent_python_double_recheck': 'PASS'}
    value = dict(schema=1, source_commit='a'*40, candidate_source_commit='a'*40, branch='investigation/test',
                 base='be4271efd1b8fb71f0c66f3ae7c35f04571aa78e', forward_base='e05f3539e2e0c0e461157a3abf1d9833d74b392b',
                 plan_sha256='a'*64, probe_sha256='a'*64, gradient_sha256='a'*64, classifications=['UNRESOLVED'],
                 standalone={'softmax': entry, 'dvalue': dict(entry, actual_gradient_gate=.01)},
                 native={k: 'NOT_EXECUTED' for k in ('softmax', 'dvalue', 'dual', 'lifetime', 'observer_free', 'sensitivity')},
                 coverage_complete=True, cohort_processes=0, performance=None, regression={'Fast': 'PASS'}, training='NO-GO', limitations=[])
    validate(value, 'a'*64)
    mutations = [lambda x: x['standalone']['softmax'].pop('maxAbs'),
                 lambda x: x['native'].update(dual='PASS'),
                 lambda x: x.update(probe_sha256='b'*64),
                 lambda x: x['standalone']['softmax'].update(core_pairs_per_variant=2199),
                 lambda x: x['standalone']['softmax'].update(status='PASS'),
                 lambda x: x.update(raw_tensor=[1, 2]),
                 lambda x: x['native'].update(observer_free='ObservedOnly'),
                 lambda x: x['standalone']['softmax'].update(maxAbs=None),
                 lambda x: x['limitations'].append('C:/private/tensor.bin')]
    for mutate in mutations:
        invalid = copy.deepcopy(value); mutate(invalid)
        try:
            validate(invalid, 'a'*64)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe export accepted')
    print('saved_p_backward_exporter_selftest=PASS rejections=9 device_calls=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--report-root', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        result = make(args.report_root)
        with args.output.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        print('saved_p_backward_public_export=PASS native_certification=0')
