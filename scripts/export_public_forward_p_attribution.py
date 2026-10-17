"""Allow-list Host attribution export. Safety-stop mode cannot certify a DSP op.

All scalar records come from the hashed analysis, never raw tensor/row fields.
Device certification requires a separate, complete-process validator.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from export_public_forward_probability import FIXTURES, host_record, need, profile, sha, load

BASE = '52f5f04edf70a69fdc48251baddb2348472f5c1d'
BRANCH = 'investigation/g1-forward-p-attribution-safe-resume'
FAILURE_FIELDS = ('forward', 'head', 'gradient', 'predicate', 'value', 'threshold', 'excess',
                  'p_coordinate', 'row', 'column', 'mask_changes_critical', 'p_reference', 'p_actual',
                  'logit_min', 'logit_max', 'top2_gap', 'dp_max', 'dc_max', 'ds_signed', 'dv_signed',
                  'ds_fp_actual', 'dv_fp_actual', 'ds_bound', 'dv_bound', 'bound_coordinate',
                  'signed_t', 'absolute_t', 'bound_term1', 'bound_term2', 'e_at_bound',
                  'p_at_bound', 'dp_at_bound', 'signed_at_bound', 'dp_error_cosine')
IMPACT_FIELDS = ('ds_signed', 'dv_signed', 'ds_fp_actual', 'dv_fp_actual', 'ds_bound', 'dv_bound')
STAGE_FIELDS = ('ln_y_error', 'q_error', 'k_error', 'qk_cumulative_error', 'qk_local_error',
                'scale_cumulative_error', 'scale_local_error', 'mask_cumulative_error', 'mask_local_error',
                'ln_only_p_error', 'qkv_only_p_error', 'pre_qk_rounding_p_error',
                'input_quantization_p_error', 'local_softmax_p_error', 'r4_r5_error')


def analyze(records):
    groups = {kind: [r for r in records if r['kind'] == kind]
              for kind in ('summary', 'path', 'impact', 'failure', 'stage', 'row', 'softmath')}
    need(len(groups['summary']) == 1, 'SUMMARY_COUNT')
    s = groups['summary'][0]
    need((s['cases'], s['heads'], s['gradient_cases'], s['cartesian_pairs'], s['same_fixture_pairs'], s['failures'])
         == (44, 88, 25, 2200, 48, 9), 'COVERAGE_CHANGED')
    need(s['bound_checks'] == 'PASS' and s['old_contract'] == 'FAIL' and s['dsp_claim'] is False, 'FALSE_PROMOTION')
    need(len(groups['path']) == 528 and len(groups['impact']) == 13200 and len(groups['row']) == 16896
         and len(groups['stage']) == 88 and len(groups['softmath']) == 88, 'RECORD_COVERAGE')
    need(len({(r['forward'], r['head'], r['path']) for r in groups['path']}) == 528, 'DUPLICATE_PATH')
    need(len({(r['forward'], r['head'], r['path'], r['gradient']) for r in groups['impact']}) == 13200, 'DUPLICATE_IMPACT')
    f = groups['failure']
    need(len(f) == 9 and sum(r['predicate'] == 'p_maxAbs' for r in f) == 1
         and sum(r['predicate'] == 'ds_bound' for r in f) == 8, 'FAILURE_BREAKDOWN')
    expected = {(16,1,9), (17,1,9), (20,1,9), (21,1,9), (21,1,12), (22,0,9), (22,0,12), (22,1,9)}
    need({(r['forward'], r['head'], r['gradient']) for r in f if r['predicate'] == 'ds_bound'} == expected, 'FAILURE_IDS')
    need(all(r['value'] > r['threshold'] and r['excess'] == r['value']-r['threshold'] for r in f), 'FAILURE_VALUE')
    need(all(r['threshold'] == (1e-6 if r['predicate'] == 'p_maxAbs' else .00025) for r in f), 'GATE_CHANGED')
    for r in groups['impact']:
        need(r['ds_signed'] <= r['ds_bound']+1e-10 and r['dv_signed'] <= r['dv_bound']+1e-10, 'BOUND_VIOLATION')
        need(r['provenance_same_fixture'] == (r['forward'] < 24 and r['forward'] == r['gradient']), 'MATCHED_INDEX')
    paths = {}
    for path in range(6):
        p = [r for r in groups['path'] if r['path'] == path]
        b = [r for r in groups['impact'] if r['path'] == path]
        worst = max(p, key=lambda r: r['p_maxAbs'])
        paths['R'+str(path)] = {k: max(r[k] for r in p) for k in
                                ('p_maxAbs', 'p_rms', 'p_relative_floor_1e12', 'row_error')}
        paths['R'+str(path)].update({k: max(r[k] for r in b) for k in IMPACT_FIELDS})
        paths['R'+str(path)]['worst_P'] = {k: worst[k] for k in ('forward', 'head', 'coordinate')}
        paths['R'+str(path)]['allowed_underflow_count'] = sum(r['allowed_underflow'] for r in groups['row'] if r['path'] == path)
    matched = [r for r in groups['impact'] if r['path'] == 5 and r['provenance_same_fixture']]
    worst = max((r for r in f if r['predicate'] == 'ds_bound'), key=lambda r: r['value'])
    failures = [{k: r[k] for k in FAILURE_FIELDS if r['gradient'] >= 0 or k in FAILURE_FIELDS[:16]} for r in f]
    return dict(original={k:s[k] for k in ('failures','p_maxAbs','row_error','ds_bound','dv_bound','ds_actual','dv_actual')},
                failures=failures, unique_forward_cases=sorted({r['forward'] for r in f}),
                unique_heads=sorted({r['head'] for r in f}), unique_gradient_cases=sorted({r['gradient'] for r in f if r['gradient']>=0}),
                unique_forward_head_count=len({(r['forward'],r['head']) for r in f}),
                paths=paths, relative_denominator_floor=1e-12,
                stage_maxima={k:max(r[k] for r in groups['stage']) for k in STAGE_FIELDS},
                worst_P_case_stages={k:next(r for r in groups['stage'] if r['forward']==21 and r['head']==1)[k] for k in STAGE_FIELDS},
                host_softmax_components={k:max(r[k] for r in groups['softmath']) for k in
                                         ('exp_rounding_normalized_error','sum_rounding_increment','output_rounding_increment','total_local_error')},
                matched=dict(status='PROVENANCE_SAME_FIXTURE_DIAGNOSTIC',forward_cases=24,head_pairs=48,
                             actual_training_trajectory=False, certified_device_producers=False,
                             maxima={k:max(r[k] for r in matched) for k in IMPACT_FIELDS}),
                worst_bound_to_global_signed_error=worst['ds_bound']/worst['ds_signed'],
                worst_bound_to_same_element_signed_error=worst['ds_bound']/abs(worst['signed_at_bound']),
                worst_bound_cancellation_ratio=abs(worst['signed_t'])/worst['absolute_t'],
                cartesian_pairs=2200, bound_formula_checks=13200, old_contract='FAIL', analysis_suite='PASS')


def safety(record):
    need(record['status'] == 'BLOCKED' and record['reason'], 'STOP_NOT_ESTABLISHED')
    need(record['reason'] in ('DEVICE_NOT_AWAKE_OR_UNKNOWN', 'EXTERNAL_POWER_UNKNOWN_OR_ABSENT',
                             'BOUNDED_DIAGNOSTIC_POWER_THERMAL_UNSAFE', 'ADB_NO_ONLINE_DEVICE',
                             'INSTALLED_APK_IDENTITY_MISMATCH_OR_UNKNOWN', 'DEVICE_STORAGE_HEADROOM_LOW_OR_UNKNOWN'), 'UNALLOWLISTED_STOP_REASON')
    need(record['read_only'] is True and record['device_executes'] == 0
         and record['device_operations_after_stop'] == 0, 'DEVICE_PROMOTION')
    # Export this mode only after a real read-only device gate was reached.
    need(record['gates']['package_probe'] == record['gates']['protected'] == record['gates']['host_analysis'] == 'PASS', 'HOST_PREFLIGHT_INCOMPLETE')


def export(repo, root):
    identity = load(root/'analysis-identity-private.json')
    for name, item in identity['tests'].items():
        need(sha(repo/'host_tests/fp32_custom'/f'{name}.cpp') == item['source_sha256'], 'ANALYSIS_SOURCE_CHANGED')
        need(sha(root/f'{name}.exe') == item['executable_sha256'], 'ANALYSIS_BINARY_CHANGED')
        log = root/('attribution.jsonl' if name=='probability_attribution_test' else f'{name}.txt')
        need(sha(log) == item['log_sha256'], 'ANALYSIS_LOG_CHANGED')
        need(item['exit_code'] == (4 if name=='probability_reference_test' else 0), 'ANALYSIS_EXIT')
    need('attribution_analysis_suite=PASS old_joint_contract=FAIL' in (root/'host-analysis.txt').read_text(encoding='utf-8-sig'), 'SUITE_INCOMPLETE')
    analysis = analyze([json.loads(s) for s in (root/'attribution.jsonl').read_text(encoding='utf-8-sig').splitlines()])
    repeat = load(root/'host-reproduction-private.json')
    need(repeat['kind'] == 'HOST_ONLY_REPRODUCTION' and repeat['device_claim'] is False and repeat['processes'] == 3, 'HOST_REPEAT_CLASSIFICATION')
    need(repeat['sha256'] == sha(root/'attribution.jsonl') and len(repeat['runs']) == 2, 'HOST_REPEAT_COUNT')
    need({r['host_process'] for r in repeat['runs']} == {2,3}, 'HOST_REPEAT_IDS')
    for r in repeat['runs']:
        need(sha(root/f"attribution-repeat-{r['host_process']}.jsonl") == r['sha256'] == repeat['sha256'], 'HOST_REPEAT_DIGEST')
    old = host_record((root/'probability_reference_test.txt').read_text())
    need(old['joint_failures'] == analysis['original']['failures'] == 9, 'ORIGINAL_FAILURE_COUNT')
    for k in ('p_maxAbs','row_error','ds_bound','dv_bound','ds_actual','dv_actual'):
        need(abs(old[k]-analysis['original'][k]) < 1e-15, 'OLD_RESULT_MISMATCH')
    initial = load(root/'start-private.json')
    need(initial['base'] == BASE, 'BASE_CHANGED')
    for ref in initial['refs']:
        name, expected = ref.split()
        if name != 'refs/heads/'+BRANCH:
            need(subprocess.check_output(['git','rev-parse',name],cwd=repo,text=True).strip() == expected, 'PROTECTED_REF_CHANGED')
    for name, expected in initial['protected']['protected_sha256'].items():
        need(sha(repo/name) == expected, 'PROTECTED_ARTIFACT_CHANGED')
    for name, expected in FIXTURES.items():
        need(sha(repo/name) == expected, 'FROZEN_FIXTURE_CHANGED')
    stop = load(root/'device-preflight-private.json');safety(stop)
    build = load(repo/'build/fp32-custom/build-identity.json')
    need(build == initial['build'], 'UNVALIDATED_NEW_CANDIDATE')
    for name, expected in build['binary_hashes'].items():
        need(sha(repo/'build/fp32-custom'/name) == expected, 'CANDIDATE_CHANGED')
    for name, expected in build['source_hashes'].items():
        need(sha(repo/name) == expected, 'DEVICE_SOURCE_CHANGED')
    validation={n:profile((root/f).read_text(encoding='utf-8-sig'),n) for n,f in
                (('Fast','fast.txt'),('Host','host.txt'),('Qnn','qnn.txt'))}
    need(len(identity['tests']) == 9, 'LEGACY_HOST_TESTS_MISSING')
    legacy = (root/'host-analysis.txt').read_text(encoding='utf-8-sig')
    need('ordered_hvx_ffn2_host_suite=PASS' in legacy, 'LEGACY_FFN2_INCOMPLETE')
    validation.update(attribution_analysis='PASS', original_P_joint_contract='FAIL',
                      Device='DEVICE_REGRESSION_NOT_RUN', HVX_FFN2_device='DEVICE_REGRESSION_NOT_RUN',
                      HVX_QKV_device='DEVICE_REGRESSION_NOT_RUN', Muon_W8='DEVICE_REGRESSION_NOT_RUN',
                      three_process_cohort='NOT_RUN')
    return dict(schema=1,status='BLOCKED',branch=BRANCH,research_base=BASE,analysis=analysis,
                analysis_source_commit=identity['source_commit'],analysis_executable_sha256=identity['tests']['probability_attribution_test']['executable_sha256'],
                host_reproduction=dict(independent_processes=3,ordered_numerical_digest=repeat['sha256'],device_claim=False),
                analysis_log_sha256=identity['tests']['probability_attribution_test']['log_sha256'],host_compiler_sha256=identity['compiler_sha256'],
                compiled_device_candidate_source_commit=build['source_sha'],device_tested_source_commit=None,
                binary_sha256=build['binary_hashes'],qairt_build_id=build['qairt_build_id'],qnn_api='2.37.0',architecture='V81',
                hexagon_compiler_sha256=build['hexagon_compiler_sha256'],ndk_compiler_sha256=build['ndk_compiler_sha256'],
                fixtures={Path(k).name:v for k,v in FIXTURES.items()},
                scale_bits='0x3e3504f3',mask_allowed_bits='0x00000000',mask_future_bits='0xc61c4000',axis=3,beta=1,
                original_gates=dict(P=1e-6,row=2e-6,P_dScores=.00025,P_dValue=.001,dScores=.002,dValue=.01),gate_relaxation=False,
                classifications=['HOST_P_ERROR_ATTRIBUTION_VERIFIED','HOST_P_JOINT_CONTRACT_FAIL','BLOCKED_HOST_P_BUDGET','BLOCKED_DEVICE_SAFETY'],
                safety_reason=stop['reason'],preflight_gates={k:stop['gates'][k] for k in
                    ('host_analysis','sdk_apk_host','host_storage','compiler','package_probe','protected',
                     'same_physical_device','active_run_absence','awake','top_resumed','power_thermal',
                     'device_storage','installed_apk') if k in stop['gates']},device_executes=0,final_processes=0,device_numerical_digests=[],
                forward_scale='NOT_RUN',standard_mask='NOT_RUN',standard_softmax='NOT_RUN',custom_softmax='NOT_IMPLEMENTED_CONDITION_NOT_MET',
                native_P='UNVERIFIED',P_backward='SHAPE_OWNERSHIP_ONLY',validation=validation,
                protected_artifacts='PASS',protected_hash_count=5,previous_branches='PASS',training='NO-GO',training_updates=0,
                full_one_layer='UNVERIFIED',full_19_layer='UNVERIFIED',observer_incident='REPRODUCED / CAUSE UNRESOLVED',incident_6031='UNRESOLVED / DORMANT / WATCH')


def self_test(records):
    analyze(records)
    for kind, key, value in [('summary','failures',0),('summary','dsp_claim',True),('failure','threshold',.01),
                             ('impact','ds_signed',1),('impact','provenance_same_fixture',False)]:
        changed=deepcopy(records);r=next(r for r in changed if r['kind']==kind);r[key]=value
        try: analyze(changed)
        except ValueError: pass
        else: raise ValueError('CORRUPTION_ACCEPTED '+key)
    good=dict(status='BLOCKED',reason='DEVICE_NOT_AWAKE_OR_UNKNOWN',read_only=True,device_executes=0,device_operations_after_stop=0,
              gates=dict(package_probe='PASS',protected='PASS',host_analysis='PASS'))
    safety(good)
    for k,v in [('status','PASS'),('device_executes',1),('device_operations_after_stop',1),('read_only',False)]:
        changed=deepcopy(good);changed[k]=v
        try: safety(changed)
        except ValueError: pass
        else: raise ValueError('SAFETY_CORRUPTION_ACCEPTED')
    print('attribution_exporter_selftest=PASS corruptions=9')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path('.'));parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test([json.loads(s) for s in (args.private_root/'attribution.jsonl').read_text(encoding='utf-8-sig').splitlines()])
    public=export(args.repo,args.private_root)
    args.output.write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('attribution_export=PASS certification=BLOCKED')
