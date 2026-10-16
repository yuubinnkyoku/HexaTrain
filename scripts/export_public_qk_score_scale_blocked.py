"""Export incomplete evidence without promoting pilot/partial device results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from export_public_qk_score_scale import BRANCH, need, process, sha


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def profile(root, filename, name):
    text = (root / filename).read_text(encoding='utf-8-sig')
    need('===== verify.ps1 -Profile ' + name + ' =====' in text, 'PROFILE_NOT_FINISHED')
    need(re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$', text), 'PROFILE_FAILED')
    return 'PASS'


def muon(root, closeout):
    p = Path(closeout['muon_report_private']); text = p.read_text(encoding='utf-8-sig')
    for key, value in dict(status='SUCCESS', rpc_status_success='true', output_tensors_finite='true', fallback='false', full_parity_matrices='114', full_parity_total_matrices='114', focus_takeover_count='0').items():
        need(re.search('^' + key + '=' + value + '$', text, re.M), 'MUON_GATE_FAILED')
    result = dict(status='PASS', matrices=114, total_matrices=114, source_commit=closeout['source_sha'])
    for key in ('full_parity_max_abs', 'full_parity_worst_relative_l2', 'full_parity_minimum_cosine'):
        result[key] = float(re.search('^' + key + r'=([^\r\n]+)$', text, re.M)[1])
    return result


def export(root):
    c = load(root / 'final-cohort-private.json'); stop = load(root / 'blocked-closeout-private.json')
    need(c['phase'] == 'Final' and c['complete'] is False and stop['reason'] == 'DEVICE_NOT_AWAKE_OR_UNKNOWN', 'NOT_A_SAFETY_STOP')
    need(stop['source_sha'] == c['source_sha'] and stop['device_operations_after_stop'] == 0, 'SAFETY_OR_SOURCE')
    need(len(c['lanes']['A']) == len(set(c['lanes']['A'])) == 2 and c['lanes']['B'] == [], 'PARTIAL_COHORT_CHANGED')
    need(not set(c['lanes']['A']) & set(c['pilot']), 'PILOT_PROMOTION')
    need('DEVICE_NOT_AWAKE_OR_UNKNOWN' in (root / 'final-A-3-console.txt').read_text(encoding='utf-8-sig'), 'STOP_EVIDENCE_MISSING')
    initial = load(root / 'initial-safety-private.json'); a = []; ledgers = []
    for ordinal, name in enumerate(c['lanes']['A'], 1):
        p = root / name; ledger = load(p / 'identity-private.json')
        need(ledger['source_sha'] == ledger['build']['source_sha'] == c['source_sha'], 'SOURCE_IDENTITY')
        record = process(ledger, (p / 'stdout.txt').read_text(), 'A'); record.update(process=ordinal, cohort='INCOMPLETE_FINAL', certification='UNRESOLVED'); a.append(record); ledgers.append(ledger)
    need(a[0]['numerical_digest'] == a[1]['numerical_digest'] and ledgers[0]['started_utc'] != ledgers[1]['started_utc'], 'TWO_PROCESS_DIGEST_OR_INDEPENDENCE')
    need(stop['pilot_b_private'] in c['pilot'], 'PILOT_IDENTITY')
    p = root / stop['pilot_b_private']; pilot = load(p / 'identity-private.json')
    need(pilot['source_sha'] == pilot['build']['source_sha'], 'PILOT_BUILD_SOURCE')
    b = process(pilot, (p / 'stdout.txt').read_text(), 'B', legacy_pilot=True)
    b.update(cohort='PILOT', certification='UNRESOLVED', source_commit=pilot['source_sha'], probe_sha256=pilot['build']['binary_hashes']['one_layer_probe'], direct_received_scale_bits_log=False, scalar_words_checked_in_probe_receipt=True)
    build = ledgers[0]['build']; need(ledgers[1]['build'] == build and ledgers[1]['hashes'] == ledgers[0]['hashes'], 'FINAL_BINARY_CHANGED')
    for key, expected in [('one-layer-inputs.bin', '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f'), ('ln1-parameters.bin', '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f')]:
        need(ledgers[0]['hashes'][key] == pilot['hashes'][key] == expected, 'FIXTURE_IDENTITY')
    need(all(l['device_private']['Serial'] == initial['device_private']['Serial'] for l in ledgers + [pilot]), 'PHYSICAL_IDENTITY')
    need(all(pilot['build']['binary_hashes'][k] == build['binary_hashes'][k] for k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so')), 'PILOT_PACKAGE_DIFFERENT')
    need(build['qairt_build_id'] == '2.48.40.260702151143' and build['hexagon_compiler_sha256'] == '039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047' and build['ndk_compiler_sha256'] == 'ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3', 'PINNED_COMPILER_SDK')
    validation = {name: profile(root, file, name) for file, name in [('fast-candidate.txt', 'Fast'), ('host-profile.txt', 'Host'), ('qnn.txt', 'Qnn'), ('device-smoke.txt', 'Device')]}
    host = (root / 'host-console.txt').read_text(encoding='utf-8-sig')
    need('ordered_hvx_qkv_host_suite=PASS tests=15' in host and 'qk_scale_host_test=PASS' in host and 'qk_scale_graph_test=PASS' in host, 'TARGETED_HOST_MISSING')
    need('qk_scale_graph_host=PASS graphs=28 new_nodes=24' in (root / 'host/qk_scale_graph_test-final.txt').read_text(), 'FINAL_GRAPH_HOST_MISSING')
    validation.update(muon_w8=muon(root, stop), new_package_hvx_ffn2='BLOCKED', new_package_hvx_qkv='BLOCKED', final_three_process_cohort='BLOCKED')
    need(stop['protected_artifacts'] == 'PASS' and stop['protected_hash_count'] == 5 and stop['previous_branch_refs'] == 'PASS', 'PROTECTION_FAILED')
    sdk = load(root / 'sdk-hashes.json')
    need(set(sdk) == {'sdk_manifest', 'compiler_guide', 'v81_hvx_prm', 'hvx_protos', 'hexagon_types'} and all(re.fullmatch('[0-9a-f]{64}', v) for v in sdk.values()), 'SDK_IDENTITY')
    public = dict(schema='G1_FP32_QK_SCORE_SCALE_FRONTIER_BLOCKED_V1', status='BLOCKED', reason=stop['reason'], branch=BRANCH, base_commit='8e13b1ed1e2464b4aef790233682a11c43c838c5', device_tested_source_commit=c['source_sha'], binary_sha256={k: build['binary_hashes'][k] for k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe')}, compiler_sha256={k: build[k] for k in ('hexagon_compiler_sha256', 'ndk_compiler_sha256')}, sdk_sha256=sdk, qairt_build_id=build['qairt_build_id'], qnn_api='2.37.0', architecture='V81', fixture_sha256={k: ledgers[0]['hashes'][k] for k in ('one-layer-inputs.bin', 'ln1-parameters.bin')}, scale_bits='3e3504f3', final_processes=dict(A=a, B=[]), pilot_b=b, pilot_b_is_final_candidate_evidence=False, final_verified_classifications=[], classification={k: 'UNRESOLVED' for k in ('DSP_FORWARD_QK_RAW_VERIFIED', 'FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIED', 'DSP_BACKWARD_SCORE_SCALE_VERIFIED', 'FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIED', 'OBSERVER_FREE_PRECISION_VERIFIED')}, next_consumers=dict(forward='OBSERVED_FAIL_STANDARD_SCALE_FINAL_2_PROCESSES', backward='OBSERVED_FAIL_STANDARD_SCATTER_PILOT'), architecture_reassessment_required=True, validation=validation, protected_artifacts='PASS', protected_hash_count=5, previous_branch_refs='PASS', numerical_digest_excludes_timing=True, final_candidate_rebuild_after_freeze=0, device_operations_after_stop=0, full_one_layer='UNVERIFIED', full_19_layer='UNVERIFIED', observer_incident='REPRODUCED / CAUSE UNRESOLVED', incident_6031='UNRESOLVED / DORMANT / WATCH', training='NO-GO', training_updates=0)
    need(stop.get('resume_reason') in (None, 'ADB_NO_ONLINE_DEVICE'), 'UNKNOWN_RESUME_STOP')
    if stop.get('resume_reason'):
        need('ADB_NO_ONLINE_DEVICE' in (root / 'resume-connection-stop-private.txt').read_text(), 'RESUME_STOP_MISSING')
    public.update(latest_reason=stop.get('resume_reason', stop['reason']), authorized_resume_attempts=stop.get('authorized_resume_attempts', 0), device_operations_after_authorized_resume=0)
    return public


def main():
    p = argparse.ArgumentParser(); p.add_argument('--private-root', required=True); p.add_argument('--output', required=True); p.add_argument('--self-test', action='store_true'); a = p.parse_args()
    root = Path(a.private_root); public = export(root)
    # Same full process checker rejects numerical, coverage and teardown damage.
    rejected = 0
    if a.self_test:
        c = load(root / 'final-cohort-private.json'); path = root / c['lanes']['A'][0]; ledger = load(path / 'identity-private.json'); text = (path / 'stdout.txt').read_text()
        for old, new in [('accuracy_failures=0', 'accuracy_failures=1'), ('word_mismatches=0', 'word_mismatches=1'), ('api,graphExecute,0', 'api,graphExecute,1'), ('api,contextFree,0', ''), ('onemetric,', 'removedmetric,'), ('hvxwords,', 'removedwords,'), ('frontiercontract,scale_bits=3e3504f3', 'frontiercontract,scale_bits=00000000'), ('segmentreceipt,marker=48383116', 'removedreceipt,marker=00000000')]:
            try: process(ledger, text.replace(old, new, 1), 'A')
            except ValueError: rejected += 1
            else: raise ValueError('CORRUPTION_ACCEPTED')
        need(public['status'] == 'BLOCKED' and public['final_verified_classifications'] == [] and public['final_processes']['B'] == [], 'INCOMPLETE_PROMOTION')
    public['self_test_corruptions_rejected'] = rejected
    target = Path(a.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print('qk_score_scale_blocked_export=PASS corruption_rejections=' + str(rejected) + ' certification=BLOCKED')


if __name__ == '__main__':
    main()
