"""Fail-closed allow-list export of a frozen six-process frontier cohort."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from export_public_ordered_hvx_ffn2 import need, fields, distribution, conditions, sha

BRANCH = 'investigation/g1-fp32-qk-score-scale-frontier'
GRADS = ['h0_dq', 'h0_dk', 'h1_dq', 'h1_dk']


def coverage(lane):
    metrics, words = Counter(), Counter()
    def add(counter, name, ci, keys):
        counter.update((name, str(ci), k) for k in keys)
    for mode in range(2):
        name = ('qk' if lane == 'A' else 'scale') + 'Standalone' + str(mode)
        for ci in range(55 if lane == 'A' else 35):
            add(metrics, name, ci, ['out' + str(j) for j in range(2 if lane == 'A' else 4)])
            add(words, name, ci, ['received_ordered'] * (2 if lane == 'A' else 4))
            if mode:
                add(words, name, ci, ['observed_final'] * (2 if lane == 'A' else 4))
    for ci in range(44 if lane == 'A' else 25):
        add(metrics, ('qk' if lane == 'A' else 'scale') + 'StandardStandalone', ci,
            ['out' + str(j) for j in range(2 if lane == 'A' else 4)])
    if lane == 'A':
        upstream = ['dx', 'dg', 'db', 'y', 'hat', 'inverse'] + ['qkv' + str(j) for j in range(3)] + ['head' + str(j) for j in range(6)]
        for next_ in range(3):
            for mode in range(4):
                name = 'qkNative' + str(next_) + str(mode)
                visible = mode != 2
                for ci in range(44):
                    add(metrics, name, ci, upstream if visible else ['dx', 'dg', 'db'])
                    add(metrics, name, ci, ['out0', 'out1'])
                    if visible:
                        add(metrics, name + 'Local', ci, ['out0', 'out1'])
                        if next_ == 2:
                            add(metrics, name, ci, ['raw0', 'raw1'])
                        add(words, name, ci, ['host_ordered_qkv' + str(j) for j in range(3)] + ['lossless_observed_head' + str(j) for j in range(6)])
                        if next_ != 1:
                            add(words, name, ci, ['received_ordered' if next_ == 0 else 'raw_received_ordered'] * 2)
                    if next_ == 0 and mode:
                        add(words, name, ci, ['observer_final'] * 2)
    else:
        for next_ in range(2):
            for domain in (2, 1):
                for mode in range(4):
                    name = 'scaleNative' + str(next_) + str(domain) + str(mode)
                    obs, tap = mode < 2, mode % 2 == 1
                    visible = obs or tap
                    keys = ['db', 'dg', 'dw']
                    if visible:
                        keys += ['dc', 'dx', 'inverse', 'residual', 'xhat']
                        keys += ['h' + str(h) + '_' + k for h in range(2) for k in ('selected', 'dc', 'dp', 'ds')]
                    if obs or (domain == 1 and tap):
                        keys += ['h' + str(h) + '_' + k for h in range(2) for k in ('centered', 'dot', 'product')]
                    if tap:
                        keys += ['y']
                    if visible:
                        keys += [k + 'raw' for k in GRADS] + GRADS
                    keys += ['scatter' + str(j) for j in range(4)] if next_ else GRADS
                    for ci in range(25):
                        add(metrics, name, ci, keys)
                        if next_ and visible:
                            add(metrics, name + 'Local', ci, ['scatter' + str(j) for j in range(4)])
                        if obs:
                            add(words, name, ci, ['raw_received_ordered'] * 4)
                        if visible:
                            add(words, name, ci, ['scale_received_ordered'] * 4)
                        if not next_ and (domain != 2 or mode != 0):
                            add(words, name, ci, ['observer_and_softmax_final'] * 4)
        for mode in range(4):
            for ci in range(25):
                add(metrics, 'scaleNativeNegative' + str(mode), ci, GRADS)
    return metrics, words


def required(lane, graph, ci, key):
    if 'StandardStandalone' in graph or 'Negative' in graph:
        return False
    if 'Standalone' in graph:
        return int(ci) < (44 if lane == 'A' else 25)
    if lane == 'A' and re.fullmatch(r'qkNative[12][0-3](Local)?', graph) and key.startswith('out'):
        return False
    return not key.startswith('scatter')


def process(ledger, text, lane, legacy_pilot=False):
    variant = 'oneQkScale' + lane
    need(ledger['variant'] == variant and ledger['branch'] == BRANCH, 'WRONG_VARIANT_BRANCH')
    need(ledger['exit_code'] == 0 and ledger['status'] == 'COMPLETE' and ledger['mode'] == 'BACKGROUND_CORRECTNESS', 'PROCESS_FAILED')
    need(ledger['focus_takeover_count'] == 0 and ledger['protected_identity'] == 'PASS', 'SAFETY_FAILED')
    summary = 'summary,' + variant + ',accuracy_failures=0,word_mismatches=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0'
    lines = text.splitlines()
    need(lines.count(summary) == 1 and 'SEGMENT_CONTRACT_FAIL' not in text, 'SUMMARY_FAILED')
    api = re.findall(r'^api,([^,]+),(\d+)$', text, re.M)
    need(api and all(v == '0' for _, v in api), 'API_FAILED')
    need(sum(k == 'graphExecute' for k, _ in api) == (754 if lane == 'A' else 715), 'EXECUTE_COVERAGE')
    for k in ('contextFree', 'deviceFree', 'backendFree'):
        need(sum(a == k for a, _ in api) == 1, 'TEARDOWN_MISSING')
    expected_m, expected_w = coverage(lane)
    observed_m, observed_w = Counter(), Counter()
    aggregate = defaultdict(lambda: dict(checks=0, max_abs=0., failures=0, nonfinite=0, max_fp_word_mismatch=0))
    canonical = []
    for line in lines:
        if line.startswith('onemetric,'):
            a = line.split(','); graph, ci, key = a[1:4]; f = fields(line, 4)
            observed_m[graph, ci, key] += 1
            need(all(math.isfinite(float(f[k])) for k in ('abs', 'rms', 'relative_l2', 'max_relative', 'fp_abs', 'cosine')), 'METRIC_NONFINITE')
            need(int(f['nonfinite']) == 0, 'TENSOR_NONFINITE')
            if required(lane, graph, ci, key):
                gate = .01
                if key in ('dx', 'dg', 'db', 'residual'):
                    gate = .02
                elif key in ('dc', 'dw'):
                    gate = .03
                elif key in ('y', 'hat', 'xhat', 'inverse') or key.startswith(('qkv', 'head')) or key.endswith(('_ds', '_selected')):
                    gate = .002
                elif key.endswith(('_dot', '_centered', '_product')):
                    gate = .005
                need(float(f['limit']) == gate and float(f['abs']) <= gate and f['gate'] == '1', 'REQUIRED_ACCURACY_FAILED')
            # Stage controls and large arithmetic controls remain separate groups.
            group = graph + ('Stress' if 'Standalone' in graph and int(ci) >= (44 if lane == 'A' else 25) else '') + '/' + key
            dst = aggregate[group]; dst['checks'] += 1; dst['max_abs'] = max(dst['max_abs'], float(f['abs'])); dst['failures'] += f['gate'] != '1'
            dst['max_fp_word_mismatch'] = max(dst['max_fp_word_mismatch'], int(f['fp_word_mismatch']))
            canonical.append(line)
        elif line.startswith('hvxwords,'):
            a = line.split(','); observed_w[tuple(a[1:4])] += 1
            need(fields(line, 4)['count'] == '0', 'WORD_MISMATCH')
            canonical.append(line)
        elif line.startswith(('frontiercontract,', 'frontierscale,', 'summary,')):
            canonical.append(line)
    need(observed_m == expected_m and observed_w == expected_w, 'EXACT_NUMERICAL_COVERAGE')
    need(sum(l.startswith('frontiercontract,scale_bits=3e3504f3,') for l in lines) == 1, 'SCALE_CONTRACT')
    marker = '48383116' if lane == 'A' else '48383117'
    receipts = [fields(l, 1) for l in lines if l.startswith('segmentreceipt,marker=' + marker + ',')]
    need(len(receipts) == (231 if lane == 'A' else 235) and all(r['raw_inputs'] == r['complete_output'] == '1' for r in receipts), 'DSP_RECEIPT_COVERAGE')
    if lane == 'B' and not legacy_pilot:
        need(sum(l == 'frontierscale,received_bits=3e3504f3,head_gradient_order=H0Q_H0K_H1Q_H1K' for l in lines) == 235, 'RECEIVED_SCALE_BITS')
    if legacy_pilot:
        need(lane == 'B' and ledger['source_sha'] == 'd640e86064cff113aa963640c2a7468fe43d1463' and ledger['build']['binary_hashes']['one_layer_probe'] == '67ceaedbabd1fcdf66211ae426edfbeb8175204247cf59dda327fbd9770d6959', 'WRONG_PILOT_SOURCE')
        need(not any(l.startswith('frontierscale,') for l in lines), 'PILOT_LOG_FORMAT_CHANGED')
    perf = defaultdict(list); first = {}; finalize = {}; graph_info = {}; memory = {}
    allowed_graphs = {k[0] for k in expected_m} - {k[0] for k in expected_m if k[0].endswith('Local')}
    for line in lines:
        if line.startswith('frontierperf,'):
            name = line.split(',')[1]; need(name in allowed_graphs, 'UNKNOWN_GRAPH'); perf[name].append(fields(line))
        elif line.startswith(('frontierfirst,', 'frontierfinalize,')):
            name = line.split(',')[1]; need(name in allowed_graphs, 'UNKNOWN_GRAPH'); f = fields(line)
            (first if line.startswith('frontierfirst,') else finalize)[name] = float(next(iter(f.values())))
        elif line.startswith(('segmentgraph,', 'closurememory,')):
            name = line.split(',')[1]; need(name in allowed_graphs, 'UNKNOWN_GRAPH'); f = {k: int(v) for k, v in fields(line).items()}
            (graph_info if line.startswith('segmentgraph,') else memory)[name] = f
    perf_names = ['qkStandalone0', 'qkStandalone1'] + ['qkNative0' + str(m) for m in range(4)] if lane == 'A' else ['scaleStandalone0', 'scaleStandalone1'] + ['scaleNative0' + str(d) + str(m) for d in (2, 1) for m in range(4)]
    need(set(perf) == set(perf_names), 'PERFORMANCE_COVERAGE')
    performance = {}
    for name, values in perf.items():
        need(len(values) == 9 and {v['rep'] for v in values} == set(map(str, range(9))) and all(v['final_words'] == '1' for v in values), 'PERFORMANCE_SAMPLES')
        performance[name] = {k: distribution([float(v[k]) for v in values]) if k in values[0] else None for k in ('wall_us', 'math_us', 'window_us')}
    thermal = [conditions(ledger['thermal_' + p])['thermal_status'] for p in ('before', 'after')]
    battery = []; power = []
    for phase in ('before', 'after'):
        s = ledger['battery_' + phase]
        battery.append(int(re.search(r'^\s*temperature:\s*(\d+)', s, re.M)[1]) / 10.)
        power.append({k: re.search(r'^\s*' + k + r' powered:\s*(true|false)', s, re.M)[1] == 'true' for k in ('AC', 'USB', 'Wireless')})
    return dict(numerical_digest=hashlib.sha256(('\n'.join(canonical) + '\n').encode()).hexdigest(), metric_checks=sum(observed_m.values()), word_comparisons=sum(observed_w.values()), qnn_executes=sum(k == 'graphExecute' for k, _ in api), stage_metrics=dict(aggregate), performance=performance, first_execute_us=first, finalize_us=finalize, graphs=graph_info, memory=memory, thermal_status=thermal, battery_temperature_c=battery, power=power, api_failures=0, word_mismatches=0, required_accuracy_failures=0, input_mutation=0, poison_remaining=0, canary_failures=0, cpu_fallback=0, teardown='PASS')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--private-root', required=True); parser.add_argument('--output', required=True); parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    root = Path(args.private_root); cohort = json.loads((root / 'final-cohort-private.json').read_text(encoding='utf-8-sig'))
    need(cohort['phase'] == 'Final' and cohort['complete'], 'COHORT_NOT_FINAL')
    all_ledgers = []; result = {}; tested = 0
    for lane in ('A', 'B'):
        ids = cohort['lanes'][lane]; need(len(ids) == len(set(ids)) == 3 and not set(ids) & set(cohort['pilot']), 'THREE_INDEPENDENT_PROCESSES')
        records = []
        for ordinal, name in enumerate(ids, 1):
            path = root / name; ledger = json.loads((path / 'identity-private.json').read_text(encoding='utf-8-sig')); text = (path / 'stdout.txt').read_text()
            need(ledger['source_sha'] == ledger['build']['source_sha'] == cohort['source_sha'], 'SOURCE_IDENTITY')
            record = process(ledger, text, lane); record['process'] = ordinal; records.append(record); all_ledgers.append(ledger)
            if args.self_test:
                changes = [lambda s: s.replace('accuracy_failures=0', 'accuracy_failures=1', 1), lambda s: s.replace('word_mismatches=0', 'word_mismatches=1', 1), lambda s: s.replace('api,graphExecute,0', 'api,graphExecute,1', 1), lambda s: s.replace('api,contextFree,0', '', 1), lambda s: s.replace('onemetric,', 'removedmetric,', 1), lambda s: s.replace('hvxwords,', 'removedwords,', 1), lambda s: s.replace('frontiercontract,scale_bits=3e3504f3', 'frontiercontract,scale_bits=00000000', 1), lambda s: s.replace('segmentreceipt,marker=' + ('48383116' if lane == 'A' else '48383117'), 'removedreceipt,marker=00000000', 1)]
                for change in changes:
                    try: process(ledger, change(text), lane)
                    except (ValueError, KeyError): tested += 1
                    else: raise ValueError('SELF_TEST_ACCEPTED_CORRUPTION')
        need(len({r['numerical_digest'] for r in records}) == 1, 'NUMERICAL_DIGEST_MISMATCH')
        result[lane] = records
    need(len({l['started_utc'] for l in all_ledgers}) == 6, 'PROCESSES_NOT_INDEPENDENT')
    build = all_ledgers[0]['build']; staged = all_ledgers[0]['hashes']
    need(all(l['build'] == build and l['hashes'] == staged for l in all_ledgers), 'FROZEN_BINARY_FIXTURE_MISMATCH')
    need(build['qairt_build_id'] == '2.48.40.260702151143' and build['hexagon_compiler_sha256'] == '039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047' and build['ndk_compiler_sha256'] == 'ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3', 'COMPILER_SDK_MISMATCH')
    initial = json.loads((root / 'initial-safety-private.json').read_text(encoding='utf-8-sig'))
    need(all(l['device_private']['Serial'] == initial['device_private']['Serial'] for l in all_ledgers), 'PHYSICAL_IDENTITY')
    for filename, expected in [('one-layer-inputs.bin', '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f'), ('ln1-parameters.bin', '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f')]:
        need(staged[filename] == expected, 'FIXTURE_IDENTITY')
    validation = json.loads((root / 'regression-private.json').read_text(encoding='utf-8-sig'))
    need(validation['source_sha'] == cohort['source_sha'] and all(validation[k] == 'PASS' for k in ('fast', 'host', 'qnn', 'device_smoke', 'muon_w8', 'protected_artifacts')), 'REQUIRED_VALIDATION_MISSING')
    from export_public_qk_score_scale_blocked import profile, muon
    for key, name in [('fast', 'Fast'), ('host', 'Host'), ('qnn', 'Qnn'), ('device', 'Device')]:
        profile(root, validation[key + '_log'], name)
    muon_record = muon(root, validation)
    mi = json.loads((Path(validation['muon_report_private']).parent / 'device-identity-private.json').read_text(encoding='utf-8-sig'))
    need(mi['stable_device_id'] == initial['device_private']['Serial'], 'MUON_PHYSICAL_IDENTITY')
    need(validation['previous_branch_refs'] == 'PASS' and validation['protected_hash_count'] == 5, 'PROTECTION_INCOMPLETE')
    sdk = json.loads((root / 'sdk-hashes.json').read_text(encoding='utf-8-sig'))
    # Existing allow-list parsers independently validate same-package numerical regressions.
    from export_public_ordered_hvx_ffn2 import process as ffn_process
    from export_public_ordered_hvx_qkv import process as qkv_process
    legacy = {}
    for key, check in [('ffn2', ffn_process), ('qkv', qkv_process)]:
        path = Path(validation[key + '_private']); ledger = json.loads((path / 'identity-private.json').read_text(encoding='utf-8-sig'))
        need(ledger['build']['binary_hashes'] == build['binary_hashes'] and ledger['source_sha'] == cohort['source_sha'], 'REGRESSION_BINARY_SOURCE')
        record = check(ledger, (path / 'stdout.txt').read_text())
        legacy[key] = {k: record[k] for k in ('qnn_executes', 'metric_checks', 'word_comparisons', 'numerical_digest')}
    public = dict(schema='G1_FP32_QK_SCORE_SCALE_FRONTIER_V1', branch=BRANCH, base_commit='8e13b1ed1e2464b4aef790233682a11c43c838c5', device_tested_source_commit=cohort['source_sha'], binary_sha256={k: build['binary_hashes'][k] for k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe')}, compiler_sha256={k: build[k] for k in ('hexagon_compiler_sha256', 'ndk_compiler_sha256')}, fixture_sha256={k: staged[k] for k in ('one-layer-inputs.bin', 'ln1-parameters.bin')}, qairt_build_id=build['qairt_build_id'], qnn_api='2.37.0', architecture='V81', scale_bits='3e3504f3', gates=dict(raw=.01, scaled_gradient=.01, forward_scale=.01, scatter=.01), independent_processes_per_lane=3, process_records=result, numerical_digest_excludes_timing=True, regression={k: validation[k] for k in ('fast', 'host', 'qnn', 'device_smoke', 'muon_w8', 'protected_artifacts')}, new_package_hvx_ffn2_qkv_regression='PASS', observer_incident='REPRODUCED / CAUSE UNRESOLVED', full_one_layer='UNVERIFIED', full_19_layer='UNVERIFIED', training='NO-GO', training_updates=0, self_test_corruptions_rejected=tested)
    public.update(status='VERIFIED_SEGMENTS', verification_complete=True, sdk_sha256=sdk, muon_w8=muon_record, legacy_regression=legacy, protected_hash_count=5, previous_branch_refs='PASS', classification=['DSP_FORWARD_QK_RAW_VERIFIED', 'FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIED', 'DSP_BACKWARD_SCORE_SCALE_VERIFIED', 'FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIED', 'OBSERVER_FREE_PRECISION_VERIFIED', 'FP32_ARCHITECTURE_REASSESSMENT_REQUIRED'])
    forward_blocked = all(r['stage_metrics']['qkNative20/out1']['failures'] > 0 for r in result['A'])
    backward_blocked = all(r['stage_metrics']['scaleNative120/scatter1']['failures'] > 0 for r in result['B'])
    need(forward_blocked and backward_blocked, 'NEXT_FRONTIER_REPRODUCTION_MISSING')
    public['classification'] += ['BLOCKED_FORWARD_CONSUMER', 'BLOCKED_BACKWARD_CONSUMER']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print('qk_score_scale_public_export=PASS corruption_rejections=' + str(tested))


if __name__ == '__main__':
    main()
