"""Allow-list export for this study's safety-stopped, Host-only evidence.

It deliberately cannot emit a device VERIFIED classification. A future device
cohort needs a separate complete-process validator, not a status override here.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

BASE = '6a4ac12f454b39737918c12c7c93c978127b900f'
BRANCH = 'investigation/g1-fp32-forward-probability-closure'
FIXTURES = {
    'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin': '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f',
    'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin': '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f',
    'build/reports/g1-fp32-attention-ln1-frontier/ln1-producers.bin': 'ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def profile(text, name):
    need('===== verify.ps1 -Profile ' + name + ' =====' in text, 'PROFILE_INCOMPLETE')
    need(re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$', text), 'PROFILE_FAILED')
    return 'PASS'


def fields(line):
    return dict(item.split('=', 1) for item in line.strip().split(',') if '=' in item)


def host_record(text):
    lines = [s for s in text.splitlines() if s.startswith('probability_reference_host,')]
    need(len(lines) == 1, 'REFERENCE_RECORD_MISSING_OR_DUPLICATED')
    f = fields(lines[0])
    need(f['cases'] == '44' and f['heads'] == '88' and f['gradient_cases'] == '25'
         and f['cartesian_boundary'] == '1' and f['dsp_claim'] == '0', 'REFERENCE_COVERAGE')
    result = {k: float(f[k]) for k in ('p_maxAbs', 'row_error', 'ds_bound', 'dv_bound', 'ds_actual', 'dv_actual')}
    need(all(0 <= v < float('inf') for v in result.values()), 'REFERENCE_NONFINITE')
    result['joint_failures'] = int(f['joint_failures'])
    failed = result['p_maxAbs'] > 1e-6 or result['row_error'] > 2e-6 or result['ds_bound'] > .00025 or result['dv_bound'] > .001
    need((result['joint_failures'] > 0) == failed, 'REFERENCE_GATE_CONTRADICTION')
    result['joint_gate'] = 'FAIL' if failed else 'PASS'
    return result


def safety(stop, text):
    need(stop['reason'] == 'DEVICE_NOT_AWAKE_OR_UNKNOWN' and stop['device_operations_after_stop'] == 0, 'SAFETY_STOP_CHANGED')
    need(stop['device_executes'] == 0 and stop['final_processes'] == 0, 'DEVICE_PROMOTION')
    need(stop['device_tested_source_commit'] is None, 'FALSE_DEVICE_SOURCE')
    need('DEVICE_NOT_AWAKE_OR_UNKNOWN' in text and 'variant=oneProbabilityScale exit=0' not in text, 'STOP_EVIDENCE')


def export(repo, root):
    stop = load(root / 'blocked-closeout-private.json')
    safety(stop, (root / 'pilot-scale-runner.txt').read_text(encoding='utf-8-sig'))
    need(not list(root.glob('customfp32-*/identity-private.json')), 'UNVALIDATED_DEVICE_RECORDS')
    build = load(repo / 'build/fp32-custom/build-identity.json')
    need(build['qairt_build_id'] == '2.48.40.260702151143', 'SDK_CHANGED')
    need(build['hexagon_compiler_sha256'] == '039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047'
         and build['ndk_compiler_sha256'] == 'ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3', 'COMPILER_CHANGED')
    for file, expected in build['source_hashes'].items():
        need(sha(repo / file) == expected, 'COMPILED_SOURCE_CHANGED')
    for name, expected in build['binary_hashes'].items():
        need(sha(repo / 'build/fp32-custom' / name) == expected, 'BINARY_CHANGED')
    for file, expected in FIXTURES.items():
        need(sha(repo / file) == expected, 'FIXTURE_CHANGED')
    old = subprocess.check_output(['git', 'show', BASE + ':host_tests/fp32_custom/package.cpp'], cwd=repo, text=True)
    new = (repo / 'host_tests/fp32_custom/package.cpp').read_text()
    def names(s):
        return re.findall(r'"([^"]+)"', re.search(r'const char\* names\[\] = \{([^;]+)\};', s)[1])
    need(len(names(old)) == 45 and names(new)[:45] == names(old) and len(names(new)) == 47, 'OLD_ABI_PREFIX_CHANGED')
    old_body = old[old.index('inline float elementProduct'):old.index('QHPI_Kernel_v1 kernels[4]')]
    new_body = new[new.index('inline float elementProduct'):new.index('QHPI_Kernel_v1 kernels[4]')]
    need(old_body == new_body, 'EXISTING_MULTIPLY_ADD_CHANGED')
    old_build = load(root / 'previous-build-private.json')
    for name in ('hvx_ffn2.cpp', 'hvx_ffn2_op.h', 'hvx_ordered_nn.h', 'hvx_qkv_op.h', 'layernorm_ops.h', 'custom_matmul_ops.h', 'softmax_ops.h', 'qk_scale_ops.h', 'head_split_op.h'):
        path = 'host_tests/fp32_custom/' + name
        need(build['source_hashes'][path] == old_build['source_hashes'][path], 'LEGACY_NUMERICAL_SOURCE_CHANGED')
    validation = {name: profile((root / file).read_text(encoding='utf-8-sig'), name) for name, file in [('Fast', 'fast-final.txt'), ('Host', 'host-profile.txt'), ('Qnn', 'qnn.txt')]}
    legacy = (root / 'host-suite.txt').read_text(encoding='utf-8-sig')
    for marker in ('ordered_hvx_qkv_host_suite=PASS tests=15', 'qk_scale_host_test=PASS', 'qk_scale_graph_test=PASS', 'forward_probability_host_test=PASS', 'probability_graph_test=PASS', 'probability_gate_test=PASS'):
        need(marker in legacy, 'HOST_SUITE_INCOMPLETE')
    host_identity = load(root / 'host-identity-private.json')
    for name, item in host_identity['tests'].items():
        need(sha(repo / ('host_tests/fp32_custom/' + name + '.cpp')) == item['source_sha256'], 'HOST_SOURCE_CHANGED')
        need(sha(root / 'host' / (name + '.exe')) == item['executable_sha256'], 'HOST_EXECUTABLE_CHANGED')
        need(sha(root / 'host' / (name + '.txt')) == item['log_sha256'], 'HOST_LOG_CHANGED')
        need(item['exit_code'] == (4 if name == 'probability_reference_test' else 0), 'HOST_EXIT_CHANGED')
    ref = host_record((root / 'host/probability_reference_test.txt').read_text())
    latest_host = (root / 'host-new-final.txt').read_text(encoding='utf-8-sig')
    need('PROBABILITY_HOST_TEST_FAILED: probability_reference_test exit=4' in latest_host and ref['joint_gate'] == 'FAIL', 'HOST_FAILURE_NOT_RECORDED')
    protected = load(repo / 'build/reports/g1-long-linear-forensics/final-protected-identities.json')['protected_sha256']
    need(len(protected) == 5, 'PROTECTED_COUNT_CHANGED')
    for file, expected in protected.items():
        need(sha(repo / file) == expected, 'PROTECTED_CHANGED')
    initial = load(root / 'start-private.json')
    need(initial['protected'] == initial['status'] == 'PASS', 'INITIAL_SAFETY_INCOMPLETE')
    for entry in initial['refs']:
        name, expected = entry.split()
        if name == 'refs/heads/' + BRANCH:
            continue
        actual = subprocess.check_output(['git', 'rev-parse', name], cwd=repo, text=True).strip()
        need(actual == expected, 'PREVIOUS_RESEARCH_REF_CHANGED')
    mask = b''.join(struct.pack('<f', -10000. if c > r else 0.) for r in range(32) for c in range(32))
    validation.update(targeted_host_suite='FAIL', targeted_host_kernel_graph_contract='PASS', host_probability_joint_contract=ref['joint_gate'], Device='BLOCKED', Muon_W8='BLOCKED', new_package_hvx_ffn2='BLOCKED', new_package_hvx_qkv='BLOCKED', new_package_qk_raw='BLOCKED', new_package_backward_scale='BLOCKED', final_three_process_cohort='BLOCKED')
    return dict(schema=1, status='BLOCKED', reason=stop['reason'], branch=BRANCH, research_base=BASE,
                compiled_candidate_source_commit=build['source_sha'], device_tested_source_commit=None,
                binary_sha256=build['binary_hashes'], qairt_build_id=build['qairt_build_id'], qnn_api='2.37.0', architecture='V81',
                hexagon_compiler_sha256=build['hexagon_compiler_sha256'], ndk_compiler_sha256=build['ndk_compiler_sha256'],
                fixtures={Path(k).name: v for k, v in FIXTURES.items()}, mask_sha256=hashlib.sha256(mask).hexdigest(),
                mask_bits='0xc61c4000', allowed_mask_bits='0x00000000', scale_bits='0x3e3504f3',
                gates=dict(P=1e-6, row_sum=2e-6, P_induced_dScores=.00025, P_induced_dValue=.001, downstream_dScores=.002, downstream_dValue=.01, scale=.01, local_mask=.001),
                probability_gate_source_commit='198f3504deb9e779fd6bb030fd72e0635dc57d68', gate_relaxation=False, host_probability=ref,
                device_executes=0, final_processes=0, numerical_digests=[], final_verified_classifications=[],
                forward_scale='HOST_ONLY_UNRESOLVED', standard_mask='UNMEASURED', standard_softmax='UNMEASURED',
                custom_softmax='NOT_IMPLEMENTED_CONDITION_NOT_MET', native_P_backward='GRAPH_SHAPE_ONLY_UNRESOLVED',
                host_reference_source_commit=host_identity['source_commit'], host_compiler_sha256=host_identity['compiler_sha256'],
                validation=validation, old_registry_prefix=45, registered_operations=47, existing_arithmetic_unchanged=True,
                saved_P_logical_bytes=dict(per_layer_microbatch=8192, layers19=155648, layers19_microbatches8=1245184),
                scheduling_and_physical_peak_allocation='UNMEASURED', device_performance='UNMEASURED',
                protected_artifacts='PASS', protected_hash_count=5, previous_research_refs='PASS',
                device_operations_after_stop=0, observer_incident='REPRODUCED / CAUSE UNRESOLVED',
                backward_head_scatter='PREVIOUSLY_BLOCKED_NOT_RETESTED', incident_6031='UNRESOLVED / DORMANT / WATCH',
                full_attention='UNVERIFIED', full_one_layer='UNVERIFIED', full_19_layer='UNVERIFIED', training='NO-GO', training_updates=0)


def self_test():
    base = dict(reason='DEVICE_NOT_AWAKE_OR_UNKNOWN', device_operations_after_stop=0, device_executes=0, final_processes=0, device_tested_source_commit=None)
    safety(base, 'DEVICE_NOT_AWAKE_OR_UNKNOWN')
    n = 0
    for key, value in [('reason', 'PASS'), ('device_operations_after_stop', 1), ('device_executes', 1), ('final_processes', 3), ('device_tested_source_commit', 'fake')]:
        altered = deepcopy(base); altered[key] = value
        try:
            safety(altered, 'DEVICE_NOT_AWAKE_OR_UNKNOWN')
        except ValueError:
            n += 1
        else:
            raise ValueError('SAFETY_CORRUPTION_ACCEPTED')
    for text in ('total 1s PASS=0 FAIL=1 SKIP=0', '===== verify.ps1 -Profile Fast =====\ntotal 1s PASS=1 FAIL=0 SKIP=1'):
        try:
            profile(text, 'Fast')
        except ValueError:
            n += 1
        else:
            raise ValueError('PROFILE_CORRUPTION_ACCEPTED')
    good = 'probability_reference_host,cases=44,heads=88,gradient_cases=25,cartesian_boundary=1,p_maxAbs=0,row_error=0,ds_bound=0,dv_bound=0,ds_actual=0,dv_actual=0,joint_failures=0,dsp_claim=0'
    host_record(good)
    for altered in (good.replace('cases=44', 'cases=43'), good.replace('p_maxAbs=0', 'p_maxAbs=nan'), good.replace('p_maxAbs=0', 'p_maxAbs=.01')):
        try:
            host_record(altered)
        except ValueError:
            n += 1
        else:
            raise ValueError('HOST_NUMERICAL_CORRUPTION_ACCEPTED')
    return n


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo'); p.add_argument('--private-root'); p.add_argument('--output'); p.add_argument('--self-test', action='store_true'); a = p.parse_args()
    if a.self_test:
        print('forward_probability_exporter_self_test=PASS rejected=' + str(self_test()))
    if a.output:
        result = export(Path(a.repo), Path(a.private_root)); text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
        need(not re.search(r'[A-Z]:[\\/]|customfp32-|device_private|device_serial|endpoint', text), 'PUBLIC_PRIVATE_DATA')
        Path(a.output).write_text(text, encoding='utf-8')
        print('forward_probability_public_export=PASS certification=BLOCKED')
