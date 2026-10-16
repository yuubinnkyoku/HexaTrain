"""Allow-list the three-process ordered FFN2 cohort; no raw tensors or IDs."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import statistics


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fields(line, start=2):
    return dict(v.split('=', 1) for v in line.split(',')[start:])


def distribution(values):
    v = sorted(values)
    def q(p):
        at = (len(v) - 1) * p
        lo = int(at)
        return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (at - lo)
    return dict(samples=len(v), median=statistics.median(v), p10=q(.1), p90=q(.9), minimum=v[0], maximum=v[-1])


def conditions(text):
    def get(pattern):
        m = re.search(pattern, text, re.M | re.I)
        need(m, 'CONDITION_MISSING')
        return m[1]
    return dict(thermal_status=int(get(r'^\s*(?:Current\s+)?Thermal Status:\s*(\d+)\s*$')))


def process(ledger, text):
    need(ledger['exit_code'] == 0 and ledger['status'] == 'COMPLETE' and ledger['mode'] == 'BACKGROUND_CORRECTNESS', 'PROCESS_FAILED')
    need(ledger['variant'] == 'oneHvx' and ledger['focus_takeover_count'] == 0 and ledger['protected_identity'] == 'PASS', 'SAFETY_FAILED')
    need('summary,oneHvx,accuracy_failures=0,word_mismatches=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0' in text, 'SUMMARY_FAILED')
    api = re.findall(r'^api,([^,]+),(\d+)$', text, re.M)
    need(api and all(code == '0' for _, code in api), 'API_FAILURE')
    need(sum(stage == 'graphExecute' for stage, _ in api) == 596, 'EXECUTE_COUNT')
    need(all(any(stage == name for stage, _ in api) for name in ('contextFree', 'deviceFree', 'backendFree')), 'TEARDOWN_MISSING')
    lines = text.splitlines()
    metrics = [r for r in lines if r.startswith('onemetric,')]
    comparisons = [r for r in lines if r.startswith('hvxwords,')]
    need(len(metrics) == 1266 and len(comparisons) == 2158, 'NUMERICAL_COVERAGE')
    expected = set()
    for mode in range(4):
        expected.update(('orderedFFN' + str(mode), str(ci), 'ffn2') for ci in range(28))
    for stage in (1, 2):
        for mode in range(8):
            visible = mode % 4 < 2 or mode % 2 == 1
            keys = (['x'] if stage == 1 else ['y', 'dx', 'dg', 'db']) + (['ffn2'] if visible else [])
            if stage == 2 and visible:
                keys += ['x', 'hat', 'inverse']
            expected.update(('orderedSegment' + str(stage) + str(mode), str(ci), key) for ci in range(16) for key in keys)
    for stage in range(3):
        for obs in (0, 1):
            keys = ['ffn2'] if stage == 0 else (['x'] + (['ffn2'] if obs else [])) if stage == 1 else ['y', 'dx', 'dg', 'db'] + (['ffn2', 'x', 'hat', 'inverse'] if obs else [])
            expected.update(('orderedPerf' + str(stage) + str(obs) + kind, '0', key) for kind in ('S', 'H') for key in keys)
    actual = [tuple(row.split(',')[1:4]) for row in metrics]
    need(len(actual) == len(set(actual)) and set(actual) == expected, 'METRIC_KEY_COVERAGE')
    need(len(set(tuple(row.split(',')[1:4]) for row in comparisons)) == len(comparisons), 'DUPLICATE_WORD_COMPARISON')
    worst = defaultdict(float)
    for row in metrics:
        parts = row.split(','); v = fields(row, 4)
        need(v['gate'] == '1' and v['nonfinite'] == '0', 'ACCURACY_FAILED')
        need(float(v['abs']) <= float(v['limit']), 'ACCURACY_LIMIT')
        if parts[3] in ('ffn2', 'x'):
            need(v['fp_word_mismatch'] == '0', 'HOST_ORDERED_MISMATCH')
        worst[parts[3]] = max(worst[parts[3]], float(v['abs']))
    need(all(fields(row, 4)['count'] == '0' for row in comparisons), 'WORD_IDENTITY_FAILED')
    receipts = [fields(row, 3) for row in lines if row.startswith('hvxreceipt,')]
    need(len(receipts) == 298 and all(r['all_words'] == '1' for r in receipts), 'RECEIPT_MISSING')
    need(set(r['marker'] for r in receipts) == {'48383112', '48383114'}, 'EXECUTION_MARKER')
    need(sum(r.startswith('hvxnative,') for r in lines) == 132, 'NATIVE_RECEIPT_COVERAGE')
    for stage in ('standalone', 'native1', 'native2'):
        need(re.search(r'^hvxgate,' + stage + r',accuracy_failures=0,word_mismatches=0', text, re.M), 'GATE_MISSING')
    groups = defaultdict(list); blocks = defaultdict(list); allocation = None; api_us = None
    for row in lines:
        if row.startswith('allocation_copy_us,'):
            allocation = float(row.split(',')[1])
        if row.startswith('api_us,graphExecute,'):
            api_us = float(row.split(',')[2])
        if not row.startswith('hvxperf,'):
            continue
        v = fields(row); key = tuple(int(v[k]) for k in ('stage', 'observed', 'hvx'))
        need(allocation is not None and api_us is not None, 'HOST_TIMING_MISSING'); v['allocation_copy_us'] = allocation; v['qnn_api_us'] = api_us
        need(v['final_words'] == '1', 'PERF_NUMERICAL_FAILURE')
        need(key[1] or ('math_us' not in v and 'window_us' not in v), 'UNOBSERVED_TIMER')
        groups[key].append(v); blocks[key[:2] + (int(v['cycle']),)].append(v)
    need(len(groups) == 12 and all(len(v) == 16 for v in groups.values()), 'STEADY_COVERAGE')
    ratios = {}
    for (stage, obs, cycle), rows in blocks.items():
        need([int(r['order']) for r in rows] == [0, 1, 2, 3], 'PAIR_ORDER')
        need([int(r['hvx']) for r in rows] == ([0, 1, 1, 0] if cycle % 2 == 0 else [1, 0, 0, 1]), 'PAIR_SEQUENCE')
        for timer in ('wall_us', 'math_us') if obs else ('wall_us',):
            med = [statistics.median(float(r[timer]) for r in rows if int(r['hvx']) == h) for h in (0, 1)]
            need(med[1] > 0, 'ZERO_TIMER')
            ratios[(stage, obs, cycle, timer)] = med[0] / med[1]
    perf = []
    first = {row.split(',')[1]: float(fields(row)['wall_us']) for row in lines if row.startswith('hvxfirst,')}
    graph = {row.split(',')[1]: fields(row) for row in lines if row.startswith('hvxgraph,')}
    memory = {row.split(',')[1]: {k: int(v) for k, v in fields(row).items()} for row in lines if row.startswith('closurememory,')}
    for stage in range(3):
        for obs in (0, 1):
            record = dict(stage=stage, observed=bool(obs))
            for h, label in ((0, 'scalar'), (1, 'hvx')):
                rows = groups[(stage, obs, h)]; name = 'orderedPerf' + str(stage) + str(obs) + ('H' if h else 'S')
                data = dict(wall_us=distribution([float(r['wall_us']) for r in rows]), qnn_api_us=distribution([r['qnn_api_us'] for r in rows]), host_allocation_copy_us=distribution([r['allocation_copy_us'] for r in rows]), first_execute_wall_us=first[name], finalize_wrapper_us=float(graph[name]['finalize_wrapper_us']), memory=memory[name])
                if obs:
                    data.update(math_us=distribution([int(r['math_us']) for r in rows]), receipt_window_us=distribution([int(r['window_us']) for r in rows]))
                else:
                    data.update(math_us=None, receipt_window_us=None)
                record[label] = data
            for timer in ('wall_us', 'math_us') if obs else ('wall_us',):
                record[timer + '_speedup'] = record['scalar'][timer]['median'] / record['hvx'][timer]['median']
                record[timer + '_paired_speedup'] = distribution([ratios[(stage, obs, cycle, timer)] for cycle in range(8)])
                record[timer + '_abba_speedup'] = statistics.median(ratios[(stage, obs, cycle, timer)] for cycle in (0, 2, 4, 6))
                record[timer + '_baab_speedup'] = statistics.median(ratios[(stage, obs, cycle, timer)] for cycle in (1, 3, 5, 7))
                record[timer + '_absolute_median_difference_us'] = record['scalar'][timer]['median'] - record['hvx'][timer]['median']
            perf.append(record)
    numerical = '\n'.join(r for r in lines if r.startswith(('onemetric,', 'hvxwords,', 'hvxgate,', 'summary,'))) + '\n'
    thermal = [conditions(ledger[k])['thermal_status'] for k in ('thermal_before', 'thermal_after')]
    battery = []
    powered = []
    power = []
    for k in ('battery_before', 'battery_after'):
        m = re.search(r'^\s*temperature:\s*(\d+)', ledger[k], re.M); need(m, 'BATTERY_MISSING'); battery.append(int(m[1]) / 10)
        m = re.search(r'^\s*USB powered:\s*(\w+)', ledger[k], re.M); need(m, 'POWER_MISSING'); powered.append(m[1])
        state = {}
        for label in ('AC', 'USB', 'Wireless'):
            m = re.search(r'^\s*' + label + r' powered:\s*(\w+)', ledger[k], re.M); need(m, 'POWER_SOURCE_MISSING'); state[label.lower()] = m[1]
        power.append(state)
    return dict(numerical_digest=hashlib.sha256(numerical.encode()).hexdigest(), metric_checks=len(metrics), word_comparisons=len(comparisons), qnn_executes=596, api_failures=0, required_word_mismatches=0, accuracy_failures=0, finite=True, input_mutation=0, poison_remaining=0, canary_failures=0, cpu_fallback=0, focus_takeovers=0, teardown='PASS', max_abs=dict(worst), performance=perf, thermal_status=thermal, battery_temperature_c=battery, usb_powered=powered, power=power, peak_rss_kib=int(re.search(r'^hvx_peak_rss_kib,(\d+)$', text, re.M)[1]), alignment_patterns=sorted(set(re.findall(r'a_mod128=\d+,b_mod128=\d+,y_mod128=\d+', text))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True); p.add_argument('--self-test', action='store_true'); a = p.parse_args()
    root = Path(a.root).resolve(); private = root / 'build/reports/ordered-fp32-hvx-ffn2'
    cohort = json.loads((private / 'final-cohort-private.json').read_text(encoding='utf-8-sig'))
    need(cohort['phase'] == 'Final' and len(cohort['processes']) == len(set(cohort['processes'])) == 3, 'THREE_PROCESSES_REQUIRED')
    need(not set(cohort['processes']) & set(cohort['pilot']), 'PILOT_IN_FINAL_COHORT')
    records = []; ledgers = []; texts = []
    for ordinal, name in enumerate(cohort['processes'], 1):
        need(re.fullmatch(r'customfp32-[0-9a-f]{32}', name), 'PRIVATE_RECORD_NAME')
        d = private / name; ledger = json.loads((d / 'identity-private.json').read_text(encoding='utf-8-sig')); text = (d / 'stdout.txt').read_text()
        need(ledger['source_sha'] == cohort['source_sha'] == ledger['build']['source_sha'], 'SOURCE_COMMIT_MISMATCH')
        need(ledger['branch'] == 'investigation/ordered-fp32-hvx-ffn2', 'BRANCH_MISMATCH')
        record = process(ledger, text); record['process'] = ordinal; records.append(record); ledgers.append(ledger); texts.append(text)
    need(len(set(r['numerical_digest'] for r in records)) == 1, 'INDEPENDENT_NUMERICAL_DIGEST_MISMATCH')
    need(len(set(l['started_utc'] for l in ledgers)) == 3, 'DUPLICATE_PROCESS_RECORD')
    build = ledgers[0]['build']; hashes = ledgers[0]['hashes']
    need(all(l['build'] == build and l['hashes'] == hashes for l in ledgers), 'BINARY_OR_FIXTURE_MISMATCH')
    need(build['qairt_build_id'] == '2.48.40.260702151143' and build['architecture'] == 'v81' and build['fp_contract'] == 'off' and build['hvx_fp_mode'] == 'strict-ieee' and build['hvx_vector_bytes'] == 128, 'FIXED_BUILD_CONTRACT')
    for name, expected in build['source_hashes'].items():
        file = (root / name).resolve(); need(file.is_relative_to(root), 'SOURCE_PATH_OUTSIDE_ROOT'); need(sha(file) == expected, 'SOURCE_HASH_MISMATCH')
    for name, expected in build['binary_hashes'].items():
        need(sha(root / 'build/fp32-custom' / name) == expected, 'BINARY_CHANGED')
    fixed = {'one-layer-inputs.bin': '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f', 'ln1-parameters.bin': '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f', 'ln1-producers.bin': 'ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580', 'libQnnHtp.so': '6e6da5284060ca4369bb6fd3c6f2b661a9cebabb2cf4cf695cd77f558082265f', 'libQnnHtpV81Skel.so': 'a0b9750d900a0afbba240e3506ba63ba91ec0e7f1b3a44de9afbb6818a665a32'}
    need(all(hashes.get(k) == v for k, v in fixed.items()), 'FIXTURE_OR_SDK_MISMATCH')
    audit = json.loads((private / 'codegen-audit.json').read_text()); need(audit['package_sha256'] == build['binary_hashes']['libHexaFp32Dsp.so'] and audit['fma_sites'] == audit['calls_in_target'] == audit['stack_spill_sites'] == 0, 'CODEGEN_MISMATCH')
    need(sha(private / 'target-disassembly-private.txt') == audit['target_disassembly_sha256'], 'DISASSEMBLY_CHANGED')
    def eligible(r, stage, obs, timer, gate):
        v = next(x for x in r['performance'] if x['stage'] == stage and x['observed'] == obs)
        return v[timer + '_speedup'] >= gate and v[timer + '_paired_speedup']['minimum'] > 1 and v[timer + '_abba_speedup'] > 1 and v[timer + '_baab_speedup'] > 1
    thermal_ok = all(r['thermal_status'] == [0, 0] for r in records)
    power_ok = len({json.dumps(state, sort_keys=True) for r in records for state in r['power']}) == 1
    kernel = thermal_ok and power_ok and all(eligible(r, 0, True, 'math_us', 1.20) for r in records)
    segment = thermal_ok and power_ok and all(eligible(r, 2, False, 'wall_us', 1.10) for r in records)
    classifications = ['HVX_FFN2_CODEGEN_VERIFIED', 'HVX_FFN2_ORDERED_WORD_IDENTITY_VERIFIED', 'HVX_FFN2_NUMERICAL_VERIFIED', 'HVX_FFN2_NATIVE_SEGMENT_VERIFIED', 'HVX_FFN2_OBSERVER_FREE_VERIFIED']
    classifications += ['HVX_KERNEL_SPEEDUP_VERIFIED'] if kernel else ['PERFORMANCE_NOT_PROMOTED']
    classifications += ['HVX_SEGMENT_SPEEDUP_VERIFIED'] if segment else ['SEGMENT_PERFORMANCE_UNRESOLVED']
    sdk = json.loads((private / 'sdk-hashes.json').read_text(encoding='utf-8-sig'))
    host_log = (private / 'host-suite-cohort-final.txt').read_text(encoding='utf-8-sig')
    tests = ('ordered_ffn2_host_test', 'closure_kernel_test', 'segment_kernel_test', 'custom_matmul_kernel_test', 'head_split_kernel_test', 'observer_kernel_test', 'softmax_kernel_test', 'attention_kernel_test', 'broadcast_kernel_test', 'one_layer_reference_test', 'upstream_reference_test', 'one_layer_graph_test', 'ordered_hvx_graph_test')
    need(all(re.search(r'^' + name + r'=PASS\r?$', host_log, re.M) for name in tests), 'HOST_REGRESSION_INCOMPLETE')
    need('PASS=1 FAIL=0 SKIP=0' in (private / 'verify-completed-fast.txt').read_text(encoding='utf-8-sig'), 'FAST_INCOMPLETE')
    need('PASS=4 FAIL=0 SKIP=0' in (private / 'verify-qnn.txt').read_text(encoding='utf-8-sig'), 'QNN_INCOMPLETE')
    baseline = json.loads((private / 'baseline-private.json').read_text(encoding='utf-8-sig'))
    protected = baseline['protected']['protected_sha256']; need(len(protected) == 5, 'PROTECTED_SET')
    for name, expected in protected.items():
        file = (root / name).resolve(); need(file.is_relative_to(root / 'build'), 'PROTECTED_PATH'); need(sha(file) == expected, 'PROTECTED_ARTIFACT_CHANGED')
    validation = json.loads((private / 'regression-closeout-private.json').read_text(encoding='utf-8-sig'))
    need(validation['device_smoke'] == 'PASS', 'DEVICE_SMOKE_INCOMPLETE')
    need('PASS=1 FAIL=0 SKIP=0' in (private / 'device-smoke-console-private.txt').read_text(encoding='utf-8-sig'), 'DEVICE_SMOKE_EVIDENCE')
    muon = validation['muon_w8']
    need(muon['status'] in ('PASS', 'BLOCKED_ADB_TRANSPORT'), 'MUON_STATUS_UNKNOWN')
    muon_public = dict(status=muon['status'])
    if muon['status'] == 'PASS':
        file = (root / muon['report_relative']).resolve(); need(file.is_relative_to(root / 'build'), 'MUON_REPORT_PATH')
        report = file.read_text(encoding='utf-8-sig')
        for k, v in {'status': 'SUCCESS', 'rpc_status_success': 'true', 'output_tensors_finite': 'true', 'fallback': 'false', 'full_parity_matrices': '114', 'full_parity_total_matrices': '114'}.items():
            need(re.search(r'^' + k + '=' + v + r'$', report, re.M), 'MUON_REQUIRED_GATE')
        muon_public.update(matrices=114, total_matrices=114)
        for k in ('full_parity_max_abs', 'full_parity_worst_relative_l2', 'full_parity_minimum_cosine'):
            m = re.search(r'^' + k + r'=([^\r\n]+)$', report, re.M); need(m, 'MUON_METRIC_MISSING'); muon_public[k] = float(m[1])
    result = dict(schema='ORDERED_HVX_FFN2_FINAL_V1', branch='investigation/ordered-fp32-hvx-ffn2', source_commit=cohort['source_sha'], device_tested_source_commit=cohort['source_sha'], qairt_build_id=build['qairt_build_id'], qnn_api='2.37.0', architecture='V81', hexagon_sdk='6.6.0.0', hexagon_tools='19.0.07', ndk='26.2.11394342', vector_bytes=128, fp32_lanes=32, fp_contract='off', hvx_fp_mode='strict-ieee', binary_sha256={k: build['binary_hashes'][k] for k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe')}, hexagon_compiler_sha256=build['hexagon_compiler_sha256'], ndk_compiler_sha256=build['ndk_compiler_sha256'], sdk_sha256={k: sdk[k] for k in ('sdk_manifest', 'hvx_protos', 'hexagon_types', 'v81_hvx_prm', 'compiler_guide')}, fixture_sha256={k: fixed[k] for k in ('one-layer-inputs.bin', 'ln1-parameters.bin', 'ln1-producers.bin')}, target_disassembly_sha256=audit['target_disassembly_sha256'], independent_processes=3, qnn_executes=sum(r['qnn_executes'] for r in records), numerical_digest=records[0]['numerical_digest'], numerical_digest_excludes_timing=True, required_word_mismatches=0, accuracy_failures=0, process_records=records, explicit_kernel_scratch_bytes=0, prepack_bytes=0, observed_receipt_extra_bytes=32, classification=classifications, training_updates=0, training='NO-GO')
    result['regression'] = dict(fast='PASS', host_tests_passed=13, shape_graphs=20, qnn='PASS', device_smoke='PASS', muon_w8=muon_public)
    result['protected_hashes_verified'] = 5
    result['verification_complete'] = muon['status'] == 'PASS'
    if a.self_test:
        tests = [lambda l, t: (dict(l, exit_code=4), t), lambda l, t: (dict(l, focus_takeover_count=1), t), lambda l, t: (l, t.replace('accuracy_failures=0', 'accuracy_failures=1', 1)), lambda l, t: (l, t.replace('count=0,first=', 'count=1,first=', 1)), lambda l, t: (l, t.replace('api,contextFree,0', 'api,contextFree,1')), lambda l, t: (l, t.replace('api,contextFree,0', 'removed')), lambda l, t: (l, t.replace('final_words=1', 'final_words=0', 1)), lambda l, t: (l, t.replace('marker=48383114', 'marker=43505514'))]
        for test in tests:
            try:
                process(*test(deepcopy(ledgers[0]), texts[0]))
            except (ValueError, KeyError):
                continue
            raise ValueError('SELF_TEST_DID_NOT_REJECT')
        print('cohort_export_self_test=PASS rejected=' + str(len(tests)))
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('final_export=PASS processes=3 executes=1788 kernel=' + str(kernel) + ' segment=' + str(segment))


if __name__ == '__main__':
    main()
