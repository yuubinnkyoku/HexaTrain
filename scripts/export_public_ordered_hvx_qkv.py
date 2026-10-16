"""Allow-list the fixed three-process QKV cohort; raw evidence stays private."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
from export_public_ordered_hvx_ffn2 import need, sha, fields, distribution, conditions


def keys(stage, visible):
    k = ['qkv0', 'qkv1', 'qkv2'] if stage < 2 or visible else []
    if stage:
        k += ['dx', 'dg', 'db'] + (['y', 'hat', 'inverse'] if visible else [])
    if stage == 2:
        k += ['head' + str(i) for i in range(6)]
    return k


def final_keys(stage):
    return (['qkv0', 'qkv1', 'qkv2'] if stage < 2 else ['head' + str(i) for i in range(6)]) + (['dx', 'dg', 'db'] if stage else [])


def expected_coverage():
    metrics, words = Counter(), Counter()
    def add(name, ci, stage, visible, repeats=1):
        for k in keys(stage, visible):
            metrics[name, str(ci), k] += repeats
        if visible:
            for k in ('qkv0', 'qkv1', 'qkv2'):
                words[name, str(ci), 'host_ordered_' + k] += repeats
            if stage == 2:
                for i in range(6):
                    words[name, str(ci), 'lossless_observed_head' + str(i)] += repeats
    for s in range(3):
        for m in range(8):
            name = 'qkvAccuracy' + str(s) + str(m)
            visible = s == 0 or m % 4 < 2 or m % 2 == 1
            for ci in range(57 if s == 0 else 44):
                add(name, ci, s, visible)
                for prefix in (['scalar_anchor_'] if m else []) + (['same_variant_scalar_'] if m >= 4 else []):
                    for k in keys(s, visible):
                        words[name, str(ci), prefix + k] += 1
    for s in range(3):
        for obs in (0, 1):
            for kind in ('S', 'H'):
                name = 'qkvPerf' + str(s) + str(obs) + kind
                add(name, 0, s, s == 0 or bool(obs), 19 if obs else 1)
                if kind == 'H':
                    for k in final_keys(s):
                        words[name, '0', 'scalar_pair_' + k] += 1
    return metrics, words


def visible_graph(name):
    if name.startswith('qkvAccuracy'):
        s, m = map(int, name[-2:])
        return s == 0 or m % 4 < 2 or m % 2 == 1
    s, obs = map(int, name[-3:-1])
    return s == 0 or bool(obs)


def process(ledger, text):
    need(ledger['exit_code'] == 0 and ledger['focus_takeover_count'] == 0 and ledger['protected_identity'] == 'PASS', 'PROCESS_OR_SAFETY_FAILED')
    need(ledger['variant'] == 'oneQkvHvx' and ledger['mode'] == 'BACKGROUND_CORRECTNESS', 'WRONG_VARIANT')
    lines = text.splitlines()
    need('summary,oneQkvHvx,accuracy_failures=0,word_mismatches=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0' in lines, 'SUMMARY_FAILED')
    api = [r for r in lines if r.startswith('api,')]
    need(all(r.rsplit(',', 1)[1] == '0' for r in api), 'API_FAILURE')
    need(api.count('api,graphExecute,0') == 1388, 'EXECUTE_COVERAGE')
    for op in ('contextFree', 'backendFree', 'deviceFree'):
        need(api.count('api,' + op + ',0') == 1, 'TEARDOWN_MISSING')
    for s in range(3):
        need('qkvgate,stage' + str(s) + ',accuracy_failures=0,word_mismatches=0' in lines, 'NUMERICAL_GATE_MISSING')
    metrics = [r for r in lines if r.startswith('onemetric,')]
    comparisons = [r for r in lines if r.startswith('hvxwords,')]
    em, ew = expected_coverage()
    need(Counter(tuple(r.split(',')[1:4]) for r in metrics) == em, 'METRIC_COVERAGE')
    need(Counter(tuple(r.split(',')[1:4]) for r in comparisons) == ew, 'WORD_COVERAGE')
    worst = defaultdict(lambda: dict(max_abs=0.0, worst_case_rms=0.0, worst_relative_l2=0.0, worst_max_relative_floor_0_001=0.0))
    stage_worst = defaultdict(dict)
    for row in metrics:
        p = row.split(','); v = fields(row, 4); k = p[3]
        need(v['gate'] == '1' and v['nonfinite'] == '0', 'ACCURACY_OR_FINITE_FAILED')
        limit = .02 if k in ('dx', 'dg', 'db') else .002
        need(float(v['limit']) == limit and float(v['abs']) <= limit, 'ACCURACY_LIMIT_CHANGED')
        if (k.startswith('qkv') or k.startswith('head')) and visible_graph(p[1]):
            need(v['fp_word_mismatch'] == '0', 'HOST_ORDERED_WORD_MISMATCH')
        for out, src in (('max_abs', 'abs'), ('worst_case_rms', 'rms'), ('worst_relative_l2', 'relative_l2'), ('worst_max_relative_floor_0_001', 'max_relative')):
            value = float(v[src]); need(value >= 0 and value < float('inf'), 'NONFINITE_METRIC'); worst[k][out] = max(worst[k][out], value)
        s = int(p[1][-2]) if p[1].startswith('qkvAccuracy') else int(p[1][-3])
        stage_worst[str(s)][k] = max(stage_worst[str(s)].get(k, 0), float(v['abs']))
    need(all(fields(r, 4)['count'] == '0' for r in comparisons), 'OUTPUT_WORD_MISMATCH')
    receipts = [fields(r, 3) for r in lines if r.startswith('qkvreceipt,')]
    need(len(receipts) == 694 and all(r['all_words'] == '1' for r in receipts), 'RECEIPT_COVERAGE')
    need(Counter(r['marker'] for r in receipts) == {'4838310d': 347, '48383115': 347}, 'DSP_MARKERS')
    native = [fields(r, 3) for r in lines if r.startswith('qkvnative,')]
    need(len(native) == 428 and all(all(r[k] == '1' for k in ('ln_input_words', 'consumer_y_words', 'saved_affine_y_words', 'saved_state_words', 'reference_fed_dy')) for r in native), 'LN_NATIVE_COVERAGE')
    need(Counter(r['producer_y_tap_words'] for r in native) == {'0': 252, '1': 176}, 'PRODUCER_TAP_VISIBILITY')
    heads = [fields(r, 3) for r in lines if r.startswith('qkvheads,')]
    need(len(heads) == 214 and all(r['producer_consumer_words'] == r['lossless_words'] == '1' for r in heads), 'HEAD_NATIVE_COVERAGE')
    fixtures = [fields(r, 2) for r in lines if r.startswith('qkvfixture,')]
    need(Counter(r['collection'] for r in fixtures) == {'ln_stage_accuracy': 44, 'independent_arithmetic_control': 13}, 'FIXTURE_COVERAGE')
    need(sum(r.startswith('qkvoracle,') for r in lines) == 180, 'HIDDEN_ORACLE_SCOPE')
    groups, blocks = defaultdict(list), defaultdict(list)
    allocation = api_us = None
    for row in lines:
        if row.startswith('allocation_copy_us,'):
            allocation = float(row.split(',')[1])
        if row.startswith('api_us,graphExecute,'):
            api_us = float(row.split(',')[2])
        if not row.startswith('qkvperf,'):
            continue
        v = fields(row); key = tuple(int(v[k]) for k in ('stage', 'observed', 'hvx'))
        need(v['final_words'] == '1' and allocation is not None and api_us is not None, 'PERF_INVALID')
        need(key[1] or ('math_us' not in v and 'window_us' not in v), 'UNOBSERVED_TIMER')
        v['allocation_copy_us'] = allocation; v['qnn_api_us'] = api_us
        groups[key].append(v); blocks[key[:2] + (int(v['cycle']),)].append(v)
    need(len(groups) == 12 and all(len(v) == 16 for v in groups.values()), 'STEADY_COVERAGE')
    ratios = {}
    for (s, obs, cycle), rows in blocks.items():
        need([int(r['order']) for r in rows] == [0, 1, 2, 3], 'PAIR_ORDER')
        need([int(r['hvx']) for r in rows] == ([0, 1, 1, 0] if cycle % 2 == 0 else [1, 0, 0, 1]), 'PAIR_SEQUENCE')
        for timer in ('wall_us', 'math_us') if obs else ('wall_us',):
            med = [statistics.median(float(r[timer]) for r in rows if int(r['hvx']) == h) for h in (0, 1)]
            need(med[0] > 0 and med[1] > 0, 'INVALID_TIMER'); ratios[s, obs, cycle, timer] = med[0] / med[1]
    expected_graphs = {'qkvAccuracy' + str(s) + str(m) for s in range(3) for m in range(8)} | {'qkvPerf' + str(s) + str(o) + k for s in range(3) for o in range(2) for k in ('S', 'H')}
    allowed = {'qkvfirst,': {'wall_us'}, 'qkvgraph,': {'stage', 'hvx', 'observed', 'tap', 'finalize_wrapper_us'}, 'closurememory,': {'input_bytes', 'output_bytes', 'native_bytes', 'host_buffer_lower_bound', 'dynamic_kernel_scratch'}, 'segmentgraph,': {'nodes', 'tensors', 'logical_bytes', 'native'}}
    for prefix, names in allowed.items():
        rows = [r for r in lines if r.startswith(prefix)]
        need(len(rows) == 36 and {r.split(',')[1] for r in rows} == expected_graphs and all(set(fields(r)) == names for r in rows), 'GRAPH_METADATA_ALLOW_LIST')
    alignment = [fields(r, 3) for r in lines if r.startswith('qkvalignment,')]
    alignment_keys = ('y', 'wq', 'wk', 'wv', 'q', 'k', 'v', 'vector_bytes', 'scratch_bytes')
    need(len(alignment) == 347 and all(set(r) == set(alignment_keys) and r['vector_bytes'] == '128' and r['scratch_bytes'] == '0' and all(0 <= int(r[k]) < 128 for k in alignment_keys[:7]) for r in alignment), 'ALIGNMENT_ALLOW_LIST')
    first = {r.split(',')[1]: float(fields(r)['wall_us']) for r in lines if r.startswith('qkvfirst,')}
    graph = {r.split(',')[1]: fields(r) for r in lines if r.startswith('qkvgraph,')}
    memory = {r.split(',')[1]: {k: int(v) for k, v in fields(r).items()} for r in lines if r.startswith('closurememory,')}
    geometry = {r.split(',')[1]: {k: int(v) for k, v in fields(r).items()} for r in lines if r.startswith('segmentgraph,')}
    perf = []
    for s in range(3):
        for obs in (0, 1):
            record = dict(stage=s, observed=bool(obs))
            for h, label in ((0, 'scalar'), (1, 'hvx')):
                rows = groups[s, obs, h]; name = 'qkvPerf' + str(s) + str(obs) + ('H' if h else 'S')
                data = dict(wall_us=distribution([float(r['wall_us']) for r in rows]), qnn_api_us=distribution([r['qnn_api_us'] for r in rows]), host_allocation_copy_us=distribution([r['allocation_copy_us'] for r in rows]), first_execute_wall_us=first[name], finalize_wrapper_us=float(graph[name]['finalize_wrapper_us']), memory=memory[name], graph=geometry[name])
                data.update(math_us=distribution([int(r['math_us']) for r in rows]) if obs else None, receipt_window_us=distribution([int(r['window_us']) for r in rows]) if obs else None)
                record[label] = data
            for timer in ('wall_us', 'math_us') if obs else ('wall_us',):
                record[timer + '_speedup'] = record['scalar'][timer]['median'] / record['hvx'][timer]['median']
                record[timer + '_paired_speedup'] = distribution([ratios[s, obs, cycle, timer] for cycle in range(8)])
                record[timer + '_abba_speedup'] = statistics.median(ratios[s, obs, cycle, timer] for cycle in (0, 2, 4, 6))
                record[timer + '_baab_speedup'] = statistics.median(ratios[s, obs, cycle, timer] for cycle in (1, 3, 5, 7))
                record[timer + '_absolute_median_difference_us'] = record['scalar'][timer]['median'] - record['hvx'][timer]['median']
            perf.append(record)
    numerical = '\n'.join(r for r in lines if r.startswith(('onemetric,', 'hvxwords,', 'qkvgate,', 'summary,'))) + '\n'
    battery, power = [], []
    for phase in ('before', 'after'):
        text = ledger['battery_' + phase]
        m = re.search(r'^\s*temperature:\s*(\d+)', text, re.M); need(m, 'BATTERY_MISSING'); battery.append(int(m[1]) / 10)
        state = {}
        for label in ('AC', 'USB', 'Wireless'):
            m = re.search(r'^\s*' + label + r' powered:\s*(true|false)', text, re.M); need(m, 'POWER_MISSING'); state[label.lower()] = m[1] == 'true'
        power.append(state)
    need(all(0 < v <= 40 for v in battery), 'THERMAL_BATTERY_GUARD')
    return dict(numerical_digest=hashlib.sha256(numerical.encode()).hexdigest(), metric_checks=len(metrics), word_comparisons=len(comparisons), qnn_executes=1388, api_failures=0, required_word_mismatches=0, accuracy_failures=0, finite=True, input_mutation=0, poison_remaining=0, canary_failures=0, cpu_fallback=0, focus_takeovers=0, teardown='PASS', errors=dict(worst), stage_max_abs=dict(stage_worst), performance=perf, thermal_status=[conditions(ledger['thermal_' + p])['thermal_status'] for p in ('before', 'after')], battery_temperature_c=battery, power=power, peak_rss_kib=int(re.search(r'^qkv_peak_rss_kib,(\d+)$', '\n'.join(lines), re.M)[1]), observed_alignment_patterns=sorted({','.join(k + '=' + str(int(r[k])) for k in alignment_keys) for r in alignment}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--output', required=True); parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    root = Path(args.root).resolve(); private = root / 'build/reports/ordered-fp32-hvx-qkv'
    cohort = json.loads((private / 'final-cohort-private.json').read_text(encoding='utf-8-sig'))
    need(cohort['phase'] == 'Final' and cohort['complete'] and len(cohort['processes']) == len(set(cohort['processes'])) == 3, 'THREE_FINAL_PROCESSES')
    need(not set(cohort['processes']) & set(cohort['pilot']), 'PILOT_IN_FINAL')
    ledgers, texts, records = [], [], []
    for ordinal, name in enumerate(cohort['processes'], 1):
        need(re.fullmatch(r'customfp32-[0-9a-f]{32}', name), 'RECORD_NAME'); d = private / name
        ledger = json.loads((d / 'identity-private.json').read_text(encoding='utf-8-sig')); text = (d / 'stdout.txt').read_text()
        need(ledger['source_sha'] == cohort['source_sha'] == ledger['build']['source_sha'] and ledger['branch'] == 'investigation/ordered-fp32-hvx-qkv', 'SOURCE_IDENTITY')
        record = process(ledger, text); record['process'] = ordinal; records.append(record); ledgers.append(ledger); texts.append(text)
    need(len(set(r['numerical_digest'] for r in records)) == 1 and len(set(l['started_utc'] for l in ledgers)) == 3, 'INDEPENDENT_NUMERICAL_DIGEST')
    build = ledgers[0]['build']; hashes = ledgers[0]['hashes']
    need(all(l['build'] == build and l['hashes'] == hashes for l in ledgers), 'BINARY_FIXTURE_IDENTITY')
    initial = json.loads((private / 'initial-preflight-private.json').read_text(encoding='utf-8-sig'))
    need(initial['status'] == 'PASS' and all(l['device_private']['Serial'] == initial['device']['Serial'] for l in ledgers), 'PHYSICAL_DEVICE_IDENTITY')
    need(build['qairt_build_id'] == '2.48.40.260702151143' and build['architecture'] == 'v81' and build['hvx_vector_bytes'] == 128 and build['hvx_fp_mode'] == 'strict-ieee' and build['fp_contract'] == 'off', 'BUILD_CONTRACT')
    for name, expected in build['source_hashes'].items():
        file = (root / name).resolve(); need(file.is_relative_to(root), 'SOURCE_PATH'); need(sha(file) == expected, 'SOURCE_CHANGED')
    for name, expected in build['binary_hashes'].items():
        need(sha(root / 'build/fp32-custom' / name) == expected, 'BINARY_CHANGED')
    fixed = {'one-layer-inputs.bin': '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f', 'ln1-parameters.bin': '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f', 'ln1-producers.bin': 'ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580', 'libQnnHtp.so': '6e6da5284060ca4369bb6fd3c6f2b661a9cebabb2cf4cf695cd77f558082265f', 'libQnnHtpV81Skel.so': 'a0b9750d900a0afbba240e3506ba63ba91ec0e7f1b3a44de9afbb6818a665a32'}
    need(all(hashes.get(k) == v for k, v in fixed.items()), 'FIXED_FIXTURE_SDK')
    audit = json.loads((private / 'codegen-audit.json').read_text()); need(audit['package_sha256'] == build['binary_hashes']['libHexaFp32Dsp.so'] and audit['k'] == 64 and audit['q_loop_trips'] == 61 and audit['fma_sites'] == audit['calls_in_target'] == audit['stack_spill_sites'] == 0 and audit['multiply_sites'] > 0 and audit['add_sites'] > 0, 'CODEGEN_FAILED')
    need(sha(private / 'target-disassembly-private.txt') == audit['target_disassembly_sha256'], 'DISASSEMBLY_CHANGED')
    wrapper_audit = json.loads((private / 'wrapper-codegen-audit.json').read_text())
    need(wrapper_audit['package_sha256'] == build['binary_hashes']['libHexaFp32Dsp.so'] and sha(private / 'linked-disassembly-private.txt') == wrapper_audit['linked_disassembly_sha256'], 'WRAPPER_CODEGEN_IDENTITY')
    expected_wrappers = {'scalar_observed': (120, 0), 'scalar_no_receipt': (96, 0), 'hvx_observed': (152, 3), 'hvx_no_receipt': (96, 3)}
    need(set(wrapper_audit['wrappers']) == set(expected_wrappers), 'WRAPPER_COVERAGE')
    for label, (frame, calls) in expected_wrappers.items():
        need(wrapper_audit['wrappers'][label] == dict(allocframe_operand_bytes=frame, vector_stack_accesses=0, arithmetic_helper_static_call_sites=calls, unexpected_calls=0), 'WRAPPER_CONTRACT')
    baseline = json.loads((private / 'baseline-private.json').read_text(encoding='utf-8-sig')); protected = baseline['protected']['protected_sha256']; need(len(protected) == 5, 'PROTECTED_SET')
    for row in (private / 'branch-refs-before.txt').read_text(encoding='utf-8-sig').splitlines():
        ref, expected = row.split(); need(subprocess.check_output(['git', '-C', str(root), 'rev-parse', ref], text=True).strip() == expected, 'PREVIOUS_BRANCH_CHANGED')
    for file in ('segment_ops.h', 'custom_matmul_ops.h', 'layernorm_ops.h', 'closure_ops.h', 'head_split_op.h', 'hvx_ffn2_op.h'):
        rel = 'host_tests/fp32_custom/' + file
        old_source = subprocess.check_output(['git', '-C', str(root), 'show', baseline['base'] + ':' + rel]).decode().replace('\r\n', '\n')
        need((root / rel).read_text() == old_source, 'SCALAR_OR_FFN2_ABI_CHANGED')
    old_package = subprocess.check_output(['git', '-C', str(root), 'show', baseline['base'] + ':host_tests/fp32_custom/package.cpp']).decode()
    def names(t):
        return re.findall(r'"([^"]+)"', re.search(r'const char\* names\[\] = \{(.*?)\};', t, re.S)[1])
    legacy_names, current_names = names(old_package), names((root / 'host_tests/fp32_custom/package.cpp').read_text())
    need(len(legacy_names) == 39 and current_names[:39] == legacy_names and current_names[39:] == ['ProjectionQKVHvx', 'ProjectionQKVHvxNoReceipt'], 'LEGACY_REGISTRY_CHANGED')
    for name, expected in protected.items():
        file = (root / name).resolve(); need(file.is_relative_to(root / 'build') and sha(file) == expected, 'PROTECTED_CHANGED')
    old_log = (private / 'host-suite-final-console.txt').read_text(encoding='utf-8-sig')
    tests = ('ordered_ffn2_host_test', 'closure_kernel_test', 'segment_kernel_test', 'custom_matmul_kernel_test', 'head_split_kernel_test', 'observer_kernel_test', 'softmax_kernel_test', 'attention_kernel_test', 'broadcast_kernel_test', 'one_layer_reference_test', 'upstream_reference_test', 'one_layer_graph_test', 'ordered_hvx_graph_test')
    need(all(re.search(r'^' + name + r'=PASS\r?$', old_log, re.M) for name in tests), 'HOST_EXISTING_REGRESSION')
    need('ordered_hvx_qkv_host_suite=PASS tests=15' in old_log, 'HOST_HARNESS_INCOMPLETE')
    need('ordered_qkv_host=PASS controls=13 calls=26' in (private / 'ordered_qkv_host_test.txt').read_text() and 'ordered_qkv_graph_host=PASS graphs=24' in (private / 'ordered_qkv_graph_test.txt').read_text(), 'NEW_HOST_REGRESSION')
    for file, marker in (('verify-closeout-fast.txt', 'PASS=1 FAIL=0 SKIP=0'), ('verify-qnn.txt', 'PASS=4 FAIL=0 SKIP=0'), ('device-smoke-console-private.txt', 'PASS=1 FAIL=0 SKIP=0')):
        need(marker in (private / file).read_text(encoding='utf-8-sig'), 'PROFILE_INCOMPLETE')
    ffn = json.loads((private / 'ffn2-regression-private.json').read_text(encoding='utf-8-sig')); need(ffn['status'] == 'PASS' and ffn['source_commit'] == cohort['source_sha'], 'FFN2_REGRESSION_MISSING')
    path = (root / ffn['report_relative']).resolve(); need(path.is_relative_to(root / 'build'), 'FFN2_REPORT_PATH')
    old = json.loads((path / 'identity-private.json').read_text()); ft = (path / 'stdout.txt').read_text()
    need(old['exit_code'] == old['focus_takeover_count'] == 0 and old['build'] == build, 'FFN2_REGRESSION_IDENTITY')
    need('summary,oneHvx,accuracy_failures=0,word_mismatches=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0' in ft and len(re.findall(r'^api,graphExecute,0$', ft, re.M)) == 596, 'FFN2_REGRESSION_FAILED')
    need(all(r.rsplit(',', 1)[1] == '0' for r in ft.splitlines() if r.startswith('api,')), 'FFN2_API_FAILED')
    fw = [r for r in ft.splitlines() if r.startswith('hvxwords,')]
    need(len(fw) == 2158 and all(fields(r, 4)['count'] == '0' for r in fw), 'FFN2_WORD_FAILED')
    old_target = root / 'build/reports/ordered-fp32-hvx-ffn2/target-disassembly-private.txt'
    need(sha(old_target) == '7c88e9ea2d49667ac4c52d5eb6602f84e22863466d98fde6390e79427290a168', 'FFN2_OLD_CODEGEN_CHANGED')
    ff_audit = json.loads((private / 'ffn2-codegen-regression/codegen-audit.json').read_text())
    new_target = private / 'ffn2-codegen-regression/target-disassembly-private.txt'
    need(ff_audit['package_sha256'] == build['binary_hashes']['libHexaFp32Dsp.so'] and sha(new_target) == ff_audit['target_disassembly_sha256'], 'FFN2_NEW_CODEGEN_IDENTITY')
    def instructions(t):
        return re.findall(r'^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} ){3}[0-9a-f]{2})', t, re.M)
    old_words, new_words = instructions(old_target.read_text()), instructions(new_target.read_text())
    need(len(old_words) == 65 and new_words[:65] == old_words and new_words[65:] == ['00 40 00 7f', '00 40 00 7f', '00 c0 00 7f'], 'FFN2_INSTRUCTIONS_CHANGED')
    validation = json.loads((private / 'regression-closeout-private.json').read_text(encoding='utf-8-sig')); muon = validation['muon_w8']; need(muon['status'] == 'PASS' and muon['source_commit'] == cohort['source_sha'] and validation['device_smoke'] == 'PASS', 'REGRESSION_INCOMPLETE')
    path = (root / muon['report_relative']).resolve(); need(path.is_relative_to(root / 'build'), 'MUON_REPORT_PATH'); mt = path.read_text(encoding='utf-8-sig')
    mi = json.loads((path.parent / 'device-identity-private.json').read_text(encoding='utf-8-sig')); need(mi['stable_device_id'] == initial['device']['Serial'], 'MUON_PHYSICAL_IDENTITY')
    for k, v in {'status': 'SUCCESS', 'rpc_status_success': 'true', 'output_tensors_finite': 'true', 'fallback': 'false', 'full_parity_matrices': '114', 'full_parity_total_matrices': '114', 'focus_takeover_count': '0'}.items():
        need(re.search(r'^' + k + '=' + v + r'$', mt, re.M), 'MUON_GATE_FAILED')
    muon_public = dict(status='PASS', matrices=114, total_matrices=114, source_commit=muon['source_commit'])
    for k in ('full_parity_max_abs', 'full_parity_worst_relative_l2', 'full_parity_minimum_cosine'):
        m = re.search(r'^' + k + r'=([^\r\n]+)$', mt, re.M); need(m, 'MUON_METRIC_MISSING'); muon_public[k] = float(m[1]); need(abs(muon_public[k]) < float('inf'), 'MUON_NONFINITE')
    def eligible(record, s, obs, timer, minimum):
        v = next(x for x in record['performance'] if x['stage'] == s and x['observed'] == obs)
        return v[timer + '_speedup'] >= minimum and v[timer + '_paired_speedup']['minimum'] > 1 and v[timer + '_abba_speedup'] > 1 and v[timer + '_baab_speedup'] > 1
    thermal = all(r['thermal_status'] == [0, 0] for r in records)
    power = len({json.dumps(p, sort_keys=True) for r in records for p in r['power']}) == 1
    kernel = thermal and power and all(eligible(r, 0, True, 'math_us', 1.20) for r in records)
    segment = thermal and power and all(eligible(r, 2, False, 'wall_us', 1.10) for r in records)
    classes = ['HVX_QKV_CODEGEN_VERIFIED', 'HVX_QKV_ORDERED_WORD_IDENTITY_VERIFIED', 'HVX_QKV_NUMERICAL_VERIFIED', 'HVX_QKV_NATIVE_SEGMENT_VERIFIED', 'HVX_QKV_OBSERVER_FREE_VERIFIED']
    classes += ['HVX_QKV_KERNEL_SPEEDUP_VERIFIED'] if kernel else ['PERFORMANCE_NOT_PROMOTED']
    classes += ['HVX_QKV_SEGMENT_SPEEDUP_VERIFIED'] if segment else ['UNRESOLVED']
    sdk = json.loads((private / 'sdk-hashes.json').read_text(encoding='utf-8-sig'))
    result = dict(schema='ORDERED_HVX_QKV_FINAL_V1', branch='investigation/ordered-fp32-hvx-qkv', base_commit=baseline['base'], device_tested_source_commit=cohort['source_sha'], qairt_build_id=build['qairt_build_id'], qnn_api='2.37.0', architecture='V81', hexagon_sdk='6.6.0.0', hexagon_tools='19.0.07', ndk='26.2.11394342', geometry=dict(m=32, k=64, n=64, projections=3, mac_per_layer_microbatch=393216), vector_bytes=128, fp32_lanes=32, fp_contract='off', hvx_fp_mode='strict-ieee', binary_sha256={k: build['binary_hashes'][k] for k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe')}, hexagon_compiler_sha256=build['hexagon_compiler_sha256'], ndk_compiler_sha256=build['ndk_compiler_sha256'], sdk_sha256={k: sdk[k] for k in ('sdk_manifest', 'hvx_protos', 'hexagon_types', 'v81_hvx_prm', 'compiler_guide')}, fixture_sha256={k: fixed[k] for k in ('one-layer-inputs.bin', 'ln1-parameters.bin')}, target_disassembly_sha256=audit['target_disassembly_sha256'], independent_processes=3, qnn_executes=4164, metric_checks=sum(r['metric_checks'] for r in records), word_comparisons=sum(r['word_comparisons'] for r in records), numerical_digest=records[0]['numerical_digest'], numerical_digest_excludes_timing=True, required_word_mismatches=0, accuracy_failures=0, process_records=records, explicit_kernel_scratch_bytes=0, prepack_bytes=0, observed_receipt_extra_bytes=64, classification=classes, performance_gates=dict(kernel=1.20, observer_free_segment=1.10), regression=dict(fast='PASS', host_tests_passed=15, new_shape_graphs=24, qnn='PASS', device_smoke='PASS', new_package_ffn2=dict(status='PASS', executes=596, word_mismatches=0, accuracy_failures=0), muon_w8=muon_public), protected_hashes_verified=5, verification_complete=True, training_updates=0, training='NO-GO')
    result['regression'].update(legacy_op_names_preserved=39, scalar_arithmetic_sources_unchanged=True, ffn2_arithmetic_instruction_words_preserved=65)
    result['qkv_wrappers'] = {name: {k: wrapper_audit['wrappers'][name][k] for k in ('allocframe_operand_bytes', 'vector_stack_accesses', 'arithmetic_helper_static_call_sites', 'unexpected_calls')} for name in expected_wrappers}
    result['observer_contract'] = dict(no_receipt_internal_operands_observed=False, no_receipt_dsp_timers=None, hidden_host_ln_oracle_word_differences_are_not_matmul_word_failures=True, required_actual_final_word_mismatches=0)
    for record in records:
        record['app_visible_complete_write'] = True
    if args.self_test:
        corruptions = [lambda l,t:(dict(l,exit_code=4),t), lambda l,t:(dict(l,focus_takeover_count=1),t), lambda l,t:(l,t.replace('accuracy_failures=0','accuracy_failures=1',1)), lambda l,t:(l,t.replace('count=0,first=','count=1,first=',1)), lambda l,t:(l,t.replace('api,contextFree,0','api,contextFree,1')), lambda l,t:(l,t.replace('api,contextFree,0','removed')), lambda l,t:(l,t.replace('marker=48383115','marker=43505515')), lambda l,t:(l,t.replace('final_words=1','final_words=0',1)), lambda l,t:(l,re.sub(r'^qkvheads,.*\n','',t,count=1,flags=re.M)), lambda l,t:(l,re.sub(r'^(onemetric,.*)$',r'\1\n\1',t,count=1,flags=re.M))]
        corruptions += [lambda l,t:(l,re.sub(r'^(qkvperf,[^,\n]+,stage=\d,observed=0,[^\n]+)',r'\1,math_us=0',t,count=1,flags=re.M)), lambda l,t:(l,t.replace('producer_consumer_words=1,lossless_words=1','producer_consumer_words=0,lossless_words=1',1))]
        corruptions += [lambda l,t:(l,re.sub(r'^(closurememory,[^\n]+)',r'\1,unexpected_private_field=123',t,count=1,flags=re.M)), lambda l,t:(l,re.sub(r'^(qkvalignment,[^\n]+)',r'\1,unexpected_private_field=123',t,count=1,flags=re.M))]
        for corrupt in corruptions:
            try:
                process(*corrupt(deepcopy(ledgers[0]), texts[0]))
            except (ValueError, KeyError):
                continue
            raise ValueError('SELF_TEST_ACCEPTED_CORRUPTION')
        print('qkv_export_self_test=PASS rejected=' + str(len(corruptions)))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('qkv_final_export=PASS processes=3 executes=4164 kernel=' + str(kernel) + ' segment=' + str(segment))


if __name__ == '__main__':
    main()
