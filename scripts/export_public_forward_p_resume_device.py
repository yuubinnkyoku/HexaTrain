"""Allow-list exporter for complete Forward P resume processes.

The historical safety-stop exporter stays immutable. This validator independently
binds the current Host assessment, binaries, per-process safety and raw results.
It never promotes local Softmax success to cumulative P or backward success.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from export_public_forward_probability import FIXTURES, fields, host_record, load, need, profile, sha
from export_public_forward_p_attribution import analyze, BASE, BRANCH

VARIANTS = ('oneProbabilityScale', 'oneProbabilityMask', 'oneProbabilityP')
SAFETY_GATES = ('host_analysis', 'sdk_apk_host', 'host_storage', 'compiler',
                'package_probe', 'protected', 'same_physical_device',
                'active_run_absence', 'awake', 'top_resumed', 'power_thermal',
                'device_storage', 'installed_apk')


def records(text, prefix):
    return [s for s in text.splitlines() if s.startswith(prefix+',')]


def numeric_digest(text):
    """Keep ordered numerical words/errors, exclude timings, counters and IDs."""
    selected = []
    for line in text.splitlines():
        tag = line.split(',')[0]
        if tag in ('onemetric','hvxwords') and 'Standard' in line.split(',')[1]:
            continue  # Negative-control behavior is assessed separately.
        if tag in ('onemetric', 'hvxwords', 'mask_native_final', 'softmax_local',
                   'p_joint', 'p_cumulative', 'summary', 'probdecision'):
            selected.append(line)
        elif tag in ('segmentreceipt', 'qkvreceipt', 'qkvheads', 'forward_softmax_receipt'):
            selected.append(','.join(s for s in line.split(',')
                                     if not s.startswith(('math_us=', 'window_us='))))
    need(selected, 'EMPTY_NUMERICAL_DIGEST')
    return hashlib.sha256(('\n'.join(selected)+'\n').encode()).hexdigest()


def safety_pass(record):
    need(record['status'] == 'PASS' and record['read_only'] is True
         and record['device_executes'] == 0 and record['device_operations_after_stop'] == 0,
         'PREFLIGHT_NOT_PASSED')
    need(all(record['gates'].get(k) == 'PASS' for k in SAFETY_GATES), 'SAFETY_GATE_MISSING')


def complete(text, ledger, variant, build):
    need(ledger['status'] == 'COMPLETE' and ledger['variant'] == variant
         and ledger['mode'] == 'BACKGROUND_CORRECTNESS'
         and ledger['focus_takeover_count'] == 0 and ledger['protected_identity'] == 'PASS',
         'PROCESS_NOT_COMPLETE')
    need(ledger['build'] == build, 'COHORT_BINARY_CHANGED')
    temperatures=[];thermal=[]
    for phase in ('before','after'):
        battery=ledger['battery_'+phase];status=ledger['thermal_'+phase]
        t=re.search(r'(?m)^\s*temperature:\s*(\d+)\s*$',battery)
        level=re.search(r'(?m)^\s*level:\s*(\d+)\s*$',battery)
        th=re.search(r'(?m)^\s*Thermal Status:\s*(\d+)\s*$',status)
        need(t and level and th and int(t[1])<=400 and int(level[1])>=30 and int(th[1])<=1
             and re.search(r'(?m)^\s*(?:AC|USB|Wireless) powered:\s*true\s*$',battery), 'PROCESS_POWER_THERMAL')
        temperatures.append(int(t[1])/10);thermal.append(int(th[1]))
    for k, v in build['binary_hashes'].items():
        if k in ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe'):
            need(ledger['hashes'].get(k) == v, 'STAGED_BINARY_MISMATCH')
    for p, expected in FIXTURES.items():
        need(ledger['hashes'].get(Path(p).name) == expected, 'STAGED_FIXTURE_MISMATCH')
    api = records(text, 'api')
    need(api and all(s.rsplit(',', 1)[1] == '0' for s in api), 'QNN_API_FAILED')
    for name in ('registerHTP', 'graphExecute', 'contextFree', 'deviceFree', 'backendFree'):
        need(f'api,{name},0' in api, 'API_OR_TEARDOWN_MISSING')
    summaries = records(text, 'summary')
    need(len(summaries) == 1 and summaries[0].split(',')[1] == variant, 'SUMMARY_COUNT')
    s = fields(summaries[0])
    for key in ('cpu_fallback', 'input_mutation', 'poison_remaining', 'canary_failures', 'training_updates'):
        need(s.get(key) == '0', 'REQUIRED_GUARD_FAILED '+key)
    need(s.get('finite') == '1' and s.get('word_mismatches') == '0', 'FINITE_OR_WORD_FAILED')
    for line in records(text, 'hvxwords'):
        if ',maskStandard' not in line:
            need(fields(line)['count'] == '0', 'WORD_COMPARISON_FAILED')
    need(len([x for x in records(text,'hvxwords') if x.split(',')[3]=='observer_final'])==376,
         'CROSS_OBSERVER_WORD_COVERAGE')
    for line in records(text, 'segmentreceipt'):
        f = fields(line)
        need(f.get('raw_inputs') == f.get('complete_output') == '1'
             and f.get('marker', '').startswith('483831'), 'DSP_RECEIPT_FAILED')
    need(len(records(text,'segmentreceipt'))=={'oneProbabilityScale':232,'oneProbabilityMask':320,'oneProbabilityP':264}[variant],
         'DSP_RECEIPT_COVERAGE')
    need(len(records(text,'qkvreceipt'))==88 and len(records(text,'qkvheads'))==88, 'QKV_RECEIPT_COVERAGE')
    for line in records(text, 'qkvreceipt'):
        need(fields(line).get('all_words') == '1', 'QKV_RECEIPT_FAILED')
    for line in records(text, 'qkvheads'):
        need(fields(line).get('producer_consumer_words') == fields(line).get('lossless_words') == '1',
             'HEAD_EDGE_FAILED')
    metrics = [line.split(',') for line in records(text, 'onemetric')]
    need(all(math.isfinite(float(fields(','.join(m))['abs'])) for m in metrics
             if 'Standard' not in m[1]), 'CUSTOM_NONFINITE_METRIC')
    required = [m for m in metrics if 'Standard' not in m[1] and
                (int(m[2]) < 44 or m[1].startswith('softmaxCustom')) and
                not (m[1].startswith('probabilityNative') and m[3] in ('P0','P1'))]
    # Cumulative P has its own unchanged joint contract; its metrics must not
    # suppress a valid upstream/local result, nor be silently treated as PASS.
    for m in required:
        f = fields(','.join(m));need(f['gate'] == '1' and f['nonfinite'] == '0'
                                  and float(f['abs'])<=float(f['limit']), 'REQUIRED_STAGE_FAILED')
    result = dict(ordered_numerical_digest=numeric_digest(text),
                  execution_source_commit=ledger['source_sha'],
                  api_calls=len(api), api_success=True, teardown=True, dsp_marker=True,
                  input_mutation=0, poison_remaining=0, canary_failures=0,
                  word_mismatches=0, finite=True, cpu_fallback=0, focus_takeover_count=0,
                  maximum_battery_temperature_c=max(temperatures),maximum_thermal_status=max(thermal),external_power=True)
    if variant == 'oneProbabilityScale':
        need(s['accuracy_failures'] == '0' and ledger['exit_code'] == 0, 'SCALE_FAILED')
        for mode in range(4):
            subset=[m for m in metrics if m[1] == f'forwardScaleNative{mode}' and m[3] in ('final0','final1')]
            need(len(subset) == 88 and len({(m[2],m[3]) for m in subset}) == 88, 'SCALE_NATIVE_COVERAGE')
        standalone=[m for m in metrics if m[1].startswith('forwardScaleStandalone') and int(m[2])<44]
        need(len(standalone)==176, 'SCALE_STANDALONE_COVERAGE')
        need(len([m for m in metrics if m[1].startswith('forwardScaleStandalone')])==224, 'SCALE_CONTROL_COVERAGE')
        result.update(standalone_core_maxAbs=max(float(fields(','.join(m))['abs']) for m in standalone),
                      native_maxAbs=max(float(fields(','.join(m))['abs']) for m in metrics
                                        if m[1].startswith('forwardScaleNative') and m[3] in ('final0','final1')))
    elif variant == 'oneProbabilityMask':
        need(s['accuracy_failures'] == '0' and s['custom_mask'] == '1'
             and ledger['exit_code'] == 0, 'MASK_FAILED')
        finals=records(text,'mask_native_final')
        need(len(finals)==352 and len({tuple(x.split(',')[1:4]) for x in finals})==352, 'MASK_NATIVE_COVERAGE')
        need(len([m for m in metrics if m[1].startswith('maskCustomStandalone')])==224, 'MASK_STANDALONE_COVERAGE')
        standard=[m for m in metrics if m[1].startswith('maskStandardH') and int(m[2])<44]
        need(len(standard)==88 and all(fields(','.join(m))['gate']=='0'
             and fields(','.join(m))['nonfinite']=='0' for m in standard),'STANDARD_MASK_FAILURE_COVERAGE')
        first=[int(fields(x)['first_word_mismatch']) for x in records(text,'probmask')
               if x.split(',')[1].startswith('maskStandardH') and int(x.split(',')[2])<44]
        need(len(first)==88 and all(q<1024 and q%32<=q//32 for q in first),'STANDARD_MASK_ALLOWED_WORD_FAILURE')
        result.update(standard_failed_predicates=int(s['standard_required_predicate_failures']),
                      standard_core_maxAbs=max(float(fields(','.join(m))['abs']) for m in standard),
                      standard_core_first_word_failure_allowed_heads=len(first),
                      standard_future_only_maximum_separately_measured=False,
                      custom_mask=True,
                      standalone_core_maxAbs=max(float(fields(','.join(m))['abs']) for m in metrics
                        if m[1].startswith('maskCustomStandalone') and int(m[2])<44),
                      native_maxAbs=max(float(fields(','.join(m))['abs']) for m in metrics if m[1].startswith('maskNative')))
    else:
        need(s['local_failures'] == s['upstream_failures'] == '0'
             and s['native_backward_executed'] == '0', 'LOCAL_P_OR_UPSTREAM_FAILED')
        local=records(text,'softmax_local')
        need(len(local)==488 and len({tuple(x.split(',')[1:4]) for x in local})==488
             and all(fields(x)['gate']=='PASS' and float(fields(x)['row_error'])<=2e-6
                     and fields(x)['negative']==fields(x)['causal_leak']=='0' for x in local), 'LOCAL_P_COVERAGE')
        exps=records(text,'forward_softmax_receipt')
        need(len(exps)==144 and all(fields(x)['arithmetic_word_mismatches']=='0' for x in exps), 'SOFTMAX_RECEIPT_COVERAGE')
        cumulative=records(text,'p_cumulative');joint=records(text,'p_joint')
        need(len(cumulative)==352 and len({tuple(x.split(',')[1:4]) for x in cumulative})==352 and len(joint)==8800 and
             len({tuple(x.split(',')[1:5]) for x in joint})==8800, 'CARTESIAN_P_COVERAGE')
        failed=sum(int(fields(x)['joint_failures']) for x in cumulative)
        need(failed==int(s['cumulative_failures']) and ledger['exit_code']==(4 if failed else 0), 'JOINT_STATUS_CONTRADICTION')
        need(all(fields(x)['host_gradient_gate_fail']=='0' for x in joint), 'HOST_CONDITIONED_GRADIENT_FAILED')
        pm=[m for m in metrics if m[1].startswith('probabilityNative') and m[3] in ('P0','P1')]
        need(len(pm)==352, 'CUMULATIVE_P_METRICS_MISSING')
        groups={}
        for line in joint:
            parts=line.split(',');f=fields(line);values={k:float(f[k]) for k in
                ('ds_bound','dv_bound','host_ds_actual','host_dv_actual')}
            need(all(math.isfinite(x) and x>=0 for x in values.values()),'P_SENSITIVITY_NONFINITE')
            allocation=int(values['ds_bound']>.00025)+int(values['dv_bound']>.001)
            consumer=int(values['host_ds_actual']>.002)+int(values['host_dv_actual']>.01)
            need(allocation==int(f['allocation_fail']) and consumer==int(f['host_gradient_gate_fail']),
                 'P_BOUND_PREDICATE_MISMATCH')
            groups.setdefault(tuple(parts[1:4]),[]).append((allocation+consumer,values))
        pmap={(m[1],m[2],'head='+m[3][1:]):fields(','.join(m)) for m in pm}
        for line in cumulative:
            parts=line.split(',');f=fields(line);key=tuple(parts[1:4]);pmetric=pmap[key]
            need(float(pmetric['limit'])==1e-6 and pmetric['nonfinite']=='0','P_ELEMENT_GATE_CHANGED')
            error_failed=float(pmetric['abs'])>1e-6
            need((pmetric['gate']=='0')==error_failed,'P_ELEMENT_PREDICATE_MISMATCH')
            group=groups[(key[0],key[1],key[2][5:])]
            expected=int(error_failed)+int(float(f['row_error'])>2e-6 or int(f['invariant_failures'])!=0)+sum(x[0] for x in group)
            need(expected==int(f['joint_failures']),'CUMULATIVE_P_PREDICATE_MISMATCH')
            for k in ('ds_bound','dv_bound','host_ds_actual','host_dv_actual'):
                need(float(f[k])==max(x[1][k] for x in group),'CUMULATIVE_P_MAXIMUM_MISMATCH')
        lm=[m for m in metrics if m[1].startswith(('softmaxCustom','probabilityLocal'))]
        matched=[x for x in joint if int(x.split(',')[2])<24 and x.split(',')[2]==x.split(',')[4]]
        need(len(matched)==192,'SAME_FIXTURE_DIAGNOSTIC_COVERAGE')
        result.update(local_maxAbs=max(float(fields(','.join(m))['abs']) for m in lm),
                      standalone_local_maxAbs=max(float(fields(','.join(m))['abs']) for m in lm if m[1].startswith('softmaxCustom')),
                      native_received_input_local_maxAbs=max(float(fields(','.join(m))['abs']) for m in lm if m[1].startswith('probabilityLocal')),
                      cumulative_maxAbs=max(float(fields(','.join(m))['abs']) for m in pm),
                      cumulative_failed_predicates=failed,
                      cumulative_joint_contract='FAIL' if failed else 'PASS',
                      exp_double_maxAbs=max(float(fields(x)['exp_double_maxAbs']) for x in exps),
                      exp_host_word_mismatches=sum(int(fields(x)['exp_host_word_mismatches']) for x in exps),
                      row_error=max(float(fields(x)['row_error']) for x in cumulative),
                      ds_bound=max(float(fields(x)['ds_bound']) for x in joint),
                      dv_bound=max(float(fields(x)['dv_bound']) for x in joint),
                      host_conditioned_ds_actual=max(float(fields(x)['host_ds_actual']) for x in joint),
                      host_conditioned_dv_actual=max(float(fields(x)['host_dv_actual']) for x in joint),
                      same_fixture_diagnostic={k:max(float(fields(x)[k]) for x in matched) for k in
                                               ('ds_bound','dv_bound','host_ds_actual','host_dv_actual')},
                      native_backward_executed=False, hidden_no_receipt_no_tap_input_directly_measured=False)
    return result


def host(repo, root, candidate):
    identity=load(candidate/'analysis-identity-private.json')
    need(len(identity['tests'])==11, 'HOST_TEST_COVERAGE')
    for name,item in identity['tests'].items():
        need(sha(repo/'host_tests/fp32_custom'/f'{name}.cpp')==item['source_sha256'], 'HOST_SOURCE_CHANGED')
        need(sha(candidate/f'{name}.exe')==item['executable_sha256'], 'HOST_EXECUTABLE_CHANGED')
        file=candidate/('attribution.jsonl' if name=='probability_attribution_test' else f'{name}.txt')
        need(sha(file)==item['log_sha256'] and item['exit_code']==(4 if name=='probability_reference_test' else 0), 'HOST_LOG_OR_EXIT')
    need('attribution_analysis_suite=PASS old_joint_contract=FAIL' in (candidate/'host-analysis.txt').read_text(encoding='utf-8-sig'), 'HOST_SUITE_INCOMPLETE')
    need('ordered_hvx_ffn2_host_suite=PASS' in (candidate/'host-analysis.txt').read_text(encoding='utf-8-sig'), 'LEGACY_HOST_MISSING')
    analysis=analyze([json.loads(s) for s in (candidate/'attribution.jsonl').read_text(encoding='utf-8-sig').splitlines()])
    old=host_record((candidate/'probability_reference_test.txt').read_text())
    need(old['joint_failures']==9 and sha(candidate/'attribution.jsonl')==sha(root/'attribution.jsonl'), 'ORIGINAL_NINE_NOT_REPRODUCED')
    repeat=load(root/'host-reproduction-private.json')
    need(repeat['processes']==3 and repeat['sha256']==sha(root/'attribution.jsonl'), 'HOST_REPRODUCTION_MISSING')
    for i in (2,3):need(sha(root/f'attribution-repeat-{i}.jsonl')==repeat['sha256'], 'HOST_REPEAT_CHANGED')
    return analysis, identity


def anchors(repo, root, candidate):
    build=load(candidate/'build-identity-private.json')
    need(build==load(repo/'build/fp32-custom/build-identity.json'), 'FINAL_BUILD_CHANGED')
    for k,v in build['binary_hashes'].items():need(sha(repo/'build/fp32-custom'/k)==v,'FINAL_BINARY_CHANGED')
    for k,v in build['source_hashes'].items():need(sha(repo/k)==v,'FINAL_SOURCE_CHANGED')
    need(build['qairt_build_id']=='2.48.40.260702151143' and build['fp_contract']=='off'
         and build['hvx_fp_mode']=='strict-ieee', 'BUILD_CONFIGURATION_CHANGED')
    initial=load(root/'start-private.json')
    need(initial['base']==BASE, 'BASE_CHANGED')
    for ref in initial['refs']:
        name,expected=ref.split()
        if name!='refs/heads/'+BRANCH:
            need(subprocess.check_output(['git','rev-parse',name],cwd=repo,text=True).strip()==expected,'PROTECTED_REF_CHANGED')
    for k,v in initial['protected']['protected_sha256'].items():need(sha(repo/k)==v,'PROTECTED_ARTIFACT_CHANGED')
    for k,v in FIXTURES.items():need(sha(repo/k)==v,'FROZEN_FIXTURE_CHANGED')
    return build


def export_partial(repo, root, candidate, stop_directory):
    analysis,identity=host(repo,root,candidate);build=anchors(repo,root,candidate)
    stop=load(candidate/stop_directory/'device-preflight-private.json')
    need(stop['status']=='BLOCKED' and stop['reason']=='ADB_TRANSPORT_FAILURE: adb operation failed'
         and stop['read_only'] is True and stop['device_executes']==0
         and stop['device_operations_after_stop']==0, 'TRANSPORT_STOP_NOT_ESTABLISHED')
    need(all(stop['gates'].get(k)=='PASS' for k in SAFETY_GATES[:6]), 'HOST_PREFLIGHT_INCOMPLETE')
    pilots={}
    for item in load(candidate/'pilots-private.json'):
        need(item['phase']=='PILOT' and item['variant'] in VARIANTS[:2], 'UNVALIDATED_PILOT')
        variant=item['variant'];folder=repo/item['relative_report_path']
        safety_pass(load(candidate/f'preflight-{variant}-pilot/device-preflight-private.json'))
        pilots[variant]=dict(certified=False,phase='PILOT',**complete(
            (folder/'stdout.txt').read_text(),load(folder/'identity-private.json'),variant,build))
    need(set(pilots)==set(VARIANTS[:2]), 'COMPLETED_PILOT_COVERAGE')
    validation={n:profile((candidate/f'{n.lower()}.txt').read_text(encoding='utf-8-sig'),n) for n in ('Fast','Host','Qnn')}
    validation.update(Device='DEVICE_REGRESSION_NOT_RUN',HVX_FFN2_device='DEVICE_REGRESSION_NOT_RUN',
                      HVX_QKV_device='DEVICE_REGRESSION_NOT_RUN',Muon_W8='DEVICE_REGRESSION_NOT_RUN',
                      final_three_process_cohort='NOT_RUN')
    return dict(schema=2,status='BLOCKED',branch=BRANCH,research_base=BASE,analysis=analysis,
                analysis_source_commit=identity['source_commit'],host_compiler_sha256=identity['compiler_sha256'],
                host_reproduction=dict(independent_processes=3,ordered_numerical_digest=sha(root/'attribution.jsonl')),
                compiled_device_candidate_source_commit=build['source_sha'],device_pilot_source_commit=build['source_sha'],
                device_certified_source_commit=None,binary_sha256=build['binary_hashes'],
                qairt_build_id=build['qairt_build_id'],qnn_api='2.37.0',architecture='V81',
                hexagon_compiler_sha256=build['hexagon_compiler_sha256'],ndk_compiler_sha256=build['ndk_compiler_sha256'],
                fixtures={Path(k).name:v for k,v in FIXTURES.items()},
                classifications=['HOST_P_ERROR_ATTRIBUTION_VERIFIED','HOST_P_JOINT_CONTRACT_FAIL',
                                 'BLOCKED_HOST_P_BUDGET','BLOCKED_DEVICE_SAFETY','UNRESOLVED'],
                safety_reason='ADB_TRANSPORT_FAILURE',safety_failure_is_numerical_failure=False,
                read_only_failed_preflight=True,device_operations_after_stop=0,
                completed_current_candidate_pilots=pilots,final_processes=0,
                current_candidate_standard_softmax='NOT_RUN',current_candidate_custom_softmax='HOST_ONLY',
                native_P='NOT_RUN',P_backward='SHAPE_OWNERSHIP_ONLY_NOT_EXECUTED',
                original_gates=dict(P=1e-6,row=2e-6,P_dScores=.00025,P_dValue=.001,dScores=.002,dValue=.01),
                gate_relaxation=False,scale_bits='0x3e3504f3',mask_allowed_bits='0x00000000',mask_future_bits='0xc61c4000',axis=3,beta=1,
                validation=validation,cumulative_certified=False,observer_free_cumulative_certified=False,backward_certified=False,
                protected_artifacts='PASS',protected_hash_count=5,previous_branches='PASS',training='NO-GO',training_updates=0,
                full_one_layer='UNVERIFIED',full_19_layer='UNVERIFIED',observer_incident='REPRODUCED / CAUSE UNRESOLVED',
                incident_6031='UNRESOLVED / DORMANT / WATCH')


def regressions(repo, candidate, build):
    manifest=load(candidate/'regression-summary-private.json')
    need(manifest['candidate_binary_hashes']==build['binary_hashes'], 'REGRESSION_CANDIDATE_CHANGED')
    expected={'oneHvx':'HVX_FFN2_device','oneQkvHvx':'HVX_QKV_device',
              'oneQkScaleA':'Scalar_QK_device','oneQkScaleB':'Backward_Scale_device',
              'oneSoftmax':'LN2_Softmax_backward_device'}
    need({x['variant'] for x in manifest['custom']}==set(expected), 'CUSTOM_REGRESSION_COVERAGE')
    result={}
    for item in manifest['custom']:
        safety_pass(load(candidate/item['preflight_directory']/'device-preflight-private.json'))
        folder=repo/item['relative_report_path'];ledger=load(folder/'identity-private.json')
        need(ledger['build']==build and ledger['exit_code']==0 and ledger['focus_takeover_count']==0
             and ledger['variant']==item['variant'] and ledger['status']=='COMPLETE', 'CUSTOM_REGRESSION_PROCESS')
        for k in ('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe'):
            need(ledger['hashes'].get(k)==build['binary_hashes'][k],'REGRESSION_STAGED_BINARY')
        text=(folder/'stdout.txt').read_text();api=records(text,'api');summaries=records(text,'summary')
        need(api and all(x.rsplit(',',1)[1]=='0' for x in api) and len(summaries)==1, 'REGRESSION_API_OR_SUMMARY')
        for n in ('contextFree','deviceFree','backendFree'):need(f'api,{n},0' in api,'REGRESSION_TEARDOWN')
        f=fields(summaries[0])
        for k in ('accuracy_failures','word_mismatches','failures','cpu_fallback','input_mutation',
                  'poison_remaining','canary_failures','training_updates'):
            if k in f:need(f[k]=='0','REGRESSION_GUARD '+k)
        need(f.get('finite')=='1','REGRESSION_NONFINITE')
        result[expected[item['variant']]]='PASS'
    for key, label in (('device','Device'),('w8','Muon_W8')):
        item=manifest[key];safety_pass(load(candidate/item['preflight_directory']/'device-preflight-private.json'))
        folder=repo/item['relative_report_path'];text=(folder/'device-report.txt').read_text(encoding='utf-8-sig')
        physical=load(folder/'device-identity-private.json')
        plan=load(repo/'build/reports/g1-long-linear-200k-primary/run-plan.json')
        need(physical['stable_device_id']==plan['device_serial_private'] and physical['physical_device_count']==1,
             'HEADLESS_PHYSICAL_IDENTITY')
        need(load(folder/'status.json')['status']=='PASSED','HEADLESS_STATUS')
        activity=load(folder/'activity-sampling.json')
        need(activity['focus_takeover_count']==0 and activity['foreground_preserved'] is True,'HEADLESS_FOCUS')
        for line in ('activity_create_count=0','activity_resume_count=0','focus_takeover_count=0',
                     'headless_test_mode=BACKGROUND_CORRECTNESS'):
            need(line in text.splitlines(),'HEADLESS_GUARD')
        need(item['runner_exit']==0 and item['candidate_binary_hashes']==build['binary_hashes'], 'HEADLESS_BINDING')
        need(sha(folder/'device-report.txt')==item['report_sha256'],'HEADLESS_REPORT_CHANGED')
        guard=(candidate/f'{key}.txt').read_text(encoding='utf-8-sig')
        need('headless_awake_safety_guard=PASS phase=before read_only=1' in guard and
             'headless_awake_safety_guard=PASS phase=after read_only=1' in guard,'HEADLESS_IMMEDIATE_SAFETY_GUARD')
        if key=='w8':
            for line in ('status=SUCCESS','full_parity_matrices=114','full_parity_total_matrices=114','rpc_status_success=true',
                         'output_tensors_finite=true','fallback=false','backend_requested=HVX'):
                need(line in text.splitlines(),'W8_REQUIRED_EVIDENCE')
            result[label]='PASS_114_OF_114'
        else:
            for line in ('compile_time_sdk_build_id=2.48.40.260702151143',
                         'runtime_backend_build_id=v2.48.40.260702151143','backend_build_id_match=true',
                         'backend_requested=HTP','cpu_fallback=false'):
                need(line in text.splitlines(),'DEVICE_REQUIRED_EVIDENCE')
            result[label]='PASS'
    return result


def negative_controls(repo, candidate, build):
    pilots=load(candidate/'pilots-private.json')
    matches=[x for x in pilots if x['variant']=='oneProbabilitySoftmax']
    need(len(matches)==1,'STANDARD_SOFTMAX_PILOT_COUNT')
    item=matches[0]
    # The original pilot runner uses the deterministic directory convention;
    # resumed pilots also record that directory explicitly. Validate the actual
    # full preflight in either case, never infer PASS from the pilot summary.
    preflight=item.get('preflight_directory',f"preflight-{item['variant']}-pilot")
    safety_pass(load(candidate/preflight/'device-preflight-private.json'))
    folder=repo/item['relative_report_path'];ledger=load(folder/'identity-private.json');text=(folder/'stdout.txt').read_text()
    need(ledger['build']==build and ledger['exit_code']==4 and ledger['status']=='COMPLETE'
         and ledger['focus_takeover_count']==0,'STANDARD_SOFTMAX_PROCESS')
    api=records(text,'api');need(api and all(x.rsplit(',',1)[1]=='0' for x in api),'STANDARD_SOFTMAX_API')
    for n in ('contextFree','deviceFree','backendFree'):need(f'api,{n},0' in api,'STANDARD_SOFTMAX_TEARDOWN')
    summary=records(text,'summary');need(len(summary)==1,'STANDARD_SOFTMAX_SUMMARY')
    f=fields(summary[0]);need(f['local_failures']=='224' and f['nonfinite_outputs']=='0','STANDARD_SOFTMAX_FAILURE_NOT_REPRODUCED')
    allmetrics=[x.split(',') for x in records(text,'onemetric') if x.split(',')[1]=='softmaxStandardLocal']
    core=[m for m in allmetrics if int(m[2])<44]
    need(len(core)==88 and all(fields(','.join(m))['gate']=='0' for m in core),'STANDARD_SOFTMAX_CORE_FAILURE')
    local=records(text,'softmax_local');need(len(local)==112,'STANDARD_SOFTMAX_LOCAL_COVERAGE')
    worst=max(core,key=lambda m:float(fields(','.join(m))['abs']))
    return dict(standard_softmax=dict(status='FAIL',phase='PILOT_NEGATIVE_CONTROL',core_failed_heads=88,core_heads=88,
                local_failed_predicates=224,core_maxAbs=float(fields(','.join(worst))['abs']),
                worst_forward=int(worst[2]),worst_head=int(worst[3][1:]),
                core_row_error=max(float(fields(x)['row_error']) for x in local if int(x.split(',')[2])<44),
                all_control_maxAbs=max(float(fields(','.join(m))['abs']) for m in allmetrics),
                all_control_row_error=max(float(fields(x)['row_error']) for x in local),
                nonfinite_outputs=0,api_success=True,teardown=True,formal_three_process_claim=False))


def receipt_source_fix(repo, candidate, build):
    file=candidate/'receipt-fix-private.json'
    if not file.exists():return None
    audit=load(file);header='host_tests/fp32_custom/forward_softmax_op.h'
    need(audit['new_source_commit']==build['source_sha'] and audit['changed_source_paths']==[header]
         and audit['math_expressions_changed'] is False and audit['receipt_layout_changed'] is False
         and audit['old_cohort_reused'] is False,'RECEIPT_FIX_SCOPE')
    before=subprocess.check_output(['git','show',audit['old_source_commit']+':'+header],cwd=repo).decode()
    old='std::memcpy(r+2120+h*1024+base+c,const_cast<const float*>(&shifted),4);'
    new='float observedShift=shifted;std::memcpy(r+2120+h*1024+base+c,&observedShift,4);'
    need(before.count(old)==1 and before.replace(old,new)==(repo/header).read_text(),'RECEIPT_FIX_CHANGED_MATH')
    previous=load(candidate.parent/'candidate-02/build-identity-private.json')
    need(previous['source_sha']==audit['old_source_commit'] and
         [k for k,v in previous['source_hashes'].items() if build['source_hashes'].get(k)!=v]==[header],
         'OTHER_PACKAGE_SOURCE_CHANGED')
    return dict(old_source_commit=audit['old_source_commit'],new_source_commit=build['source_sha'],
                changed_source_paths=[header],math_expressions_changed=False,receipt_layout_changed=False,
                previous_cohort_reused=False,source_rule='https://eel.is/c++draft/dcl.type.cv#5')


def export(repo, root, candidate):
    analysis,identity=host(repo,root,candidate)
    build=anchors(repo,root,candidate)
    cohort=load(candidate/'final-cohort-private.json')
    need(len(cohort)==9 and all(r['phase']=='FINAL' for r in cohort), 'FINAL_COHORT_COUNT')
    output={}
    for variant in VARIANTS:
        group=[r for r in cohort if r['variant']==variant]
        need(len(group)==3 and len({r['relative_report_path'] for r in group})==3, 'INDEPENDENT_PROCESS_COUNT')
        results=[]
        for item in group:
            folder=repo/item['relative_report_path'];ledger=load(folder/'identity-private.json')
            safety_pass(load(candidate/item['preflight_directory']/'device-preflight-private.json'))
            results.append(complete((folder/'stdout.txt').read_text(),ledger,variant,build))
        need(len({r['ordered_numerical_digest'] for r in results})==1, 'PROCESS_NUMERICAL_NONDETERMINISM')
        need(len({r['execution_source_commit'] for r in results})==1,'PROCESS_EXECUTION_SOURCE_CHANGED')
        output[variant]=dict(independent_processes=3,**results[0])
        output[variant]['maximum_battery_temperature_c']=max(r['maximum_battery_temperature_c'] for r in results)
        output[variant]['maximum_thermal_status']=max(r['maximum_thermal_status'] for r in results)
    validation={n:profile((candidate/('fast-final.txt' if n=='Fast' else f'{n.lower()}.txt')).read_text(encoding='utf-8-sig'),n) for n in ('Fast','Host','Qnn')}
    validation.update(regressions(repo,candidate,build))
    classifications=['HOST_P_ERROR_ATTRIBUTION_VERIFIED','HOST_P_JOINT_CONTRACT_FAIL',
                     'BLOCKED_HOST_P_BUDGET','FORWARD_SCALE_DSP_VERIFIED',
                     'CAUSAL_MASK_DEVICE_VERIFIED','SOFTMAX_P_LOCAL_VERIFIED','BLOCKED_STANDARD_SOFTMAX']
    return dict(schema=2,status='BLOCKED',branch=BRANCH,research_base=BASE,analysis=analysis,
                analysis_source_commit=identity['source_commit'],host_compiler_sha256=identity['compiler_sha256'],
                host_reproduction=dict(kind='ORIGINAL_HOST_ONLY_REPRODUCTION',independent_processes=3,
                    ordered_numerical_digest=sha(root/'attribution.jsonl'),
                    executable_sha256=load(root/'analysis-identity-private.json')['tests']['probability_attribution_test']['executable_sha256'],
                    current_candidate_attribution_log_matches=True),
                compiled_device_candidate_source_commit=build['source_sha'],device_tested_source_commit=build['source_sha'],
                binary_sha256=build['binary_hashes'],qairt_build_id=build['qairt_build_id'],qnn_api='2.37.0',architecture='V81',
                hexagon_compiler_sha256=build['hexagon_compiler_sha256'],ndk_compiler_sha256=build['ndk_compiler_sha256'],
                fixtures={Path(k).name:v for k,v in FIXTURES.items()},classifications=classifications,
                original_gates=dict(P=1e-6,row=2e-6,P_dScores=.00025,P_dValue=.001,dScores=.002,dValue=.01),
                gate_relaxation=False,scale_bits='0x3e3504f3',mask_allowed_bits='0x00000000',mask_future_bits='0xc61c4000',axis=3,beta=1,
                cohorts=output,negative_controls=negative_controls(repo,candidate,build),
                receipt_source_fix=receipt_source_fix(repo,candidate,build),
                validation=validation,device_safety='PASS_AT_EACH_EXECUTED_PROCESS',
                historical_transport_stop_preserved=True,current_device_safety_blocked=False,
                causal_mask_implementation='REUSED_DSP_FP32_ADD',custom_softmax='LOCAL_VERIFIED',
                native_P='DIAGNOSTIC_ONLY_JOINT_CONTRACT_FAIL',P_backward='SHAPE_OWNERSHIP_ONLY_NOT_EXECUTED',
                cumulative_certified=False,observer_free_cumulative_certified=False,backward_certified=False,
                protected_artifacts='PASS',protected_hash_count=5,previous_branches='PASS',training='NO-GO',training_updates=0,
                full_one_layer='UNVERIFIED',full_19_layer='UNVERIFIED',observer_incident='REPRODUCED / CAUSE UNRESOLVED',
                incident_6031='UNRESOLVED / DORMANT / WATCH')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--private-root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--stop-preflight',type=Path);a=p.parse_args()
    if a.self_test:
        good=dict(status='PASS',read_only=True,device_executes=0,device_operations_after_stop=0,
                  gates={k:'PASS' for k in SAFETY_GATES})
        safety_pass(good)
        for key,value in [('status','BLOCKED'),('device_executes',1),('read_only',False),('device_operations_after_stop',1)]:
            changed=deepcopy(good);changed[key]=value
            try:safety_pass(changed)
            except ValueError:pass
            else:raise ValueError('SAFETY_CORRUPTION_ACCEPTED')
        need(numeric_digest('onemetric,a,0,x,abs=0\napi_us,x,1\n')==
             numeric_digest('onemetric,a,0,x,abs=0\napi_us,x,99\n'),'TIMING_IN_DIGEST')
        print('forward_p_resume_exporter_selftest=PASS safety_corruptions=4 timing_excluded=1')
        build=load(a.candidate/'build-identity-private.json');count=0
        for item in load(a.candidate/'pilots-private.json'):
            if item['variant'] not in VARIANTS[:2]:continue
            folder=a.repo/item['relative_report_path'];text=(folder/'stdout.txt').read_text();ledger=load(folder/'identity-private.json')
            complete(text,ledger,item['variant'],build)
            for key,value in [('exit_code',139),('focus_takeover_count',1),('status','PARTIAL')]:
                changed=deepcopy(ledger);changed[key]=value
                try:complete(text,changed,item['variant'],build)
                except ValueError:count+=1
                else:raise ValueError('PROCESS_CORRUPTION_ACCEPTED '+key)
            for old,new in [('api,backendFree,0','api,backendFree,1'),('poison_remaining=0','poison_remaining=1'),
                            ('word_mismatches=0','word_mismatches=1'),('finite=1','finite=0')]:
                need(old in text,'SELFTEST_ANCHOR_MISSING')
                try:complete(text.replace(old,new),ledger,item['variant'],build)
                except ValueError:count+=1
                else:raise ValueError('RESULT_CORRUPTION_ACCEPTED '+old)
        need(count==14,'SELFTEST_PROCESS_COVERAGE')
        print('forward_p_resume_process_corruption_checks=PASS corruptions=14 pilot_promotion=0')
        if not a.stop_preflight:
            item=next(x for x in load(a.candidate/'pilots-private.json') if x['variant']=='oneProbabilityP')
            folder=a.repo/item['relative_report_path'];text=(folder/'stdout.txt').read_text();ledger=load(folder/'identity-private.json')
            complete(text,ledger,'oneProbabilityP',build)
            changes=[('cumulative_failures=36','cumulative_failures=0'),
                     ('allocation_fail=1','allocation_fail=0'),
                     ('joint_failures=3','joint_failures=0'),
                     ('arithmetic_word_mismatches=0','arithmetic_word_mismatches=1'),
                     ('native_backward_executed=0','native_backward_executed=1'),
                     ('limit=1e-06','limit=0.01')]
            for old,new in changes:
                need(old in text,'P_SELFTEST_ANCHOR')
                try:complete(text.replace(old,new),ledger,'oneProbabilityP',build)
                except ValueError:pass
                else:raise ValueError('P_FALSE_PROMOTION_ACCEPTED '+old)
            # Removing the only cumulative P element failure must also fail,
            # even when all actual Host gradient gates stay within budget.
            changed=text.replace('abs=1.00750624249e-06,','abs=0,')
            need(changed!=text,'P_ERROR_ANCHOR')
            try:complete(changed,ledger,'oneProbabilityP',build)
            except ValueError:pass
            else:raise ValueError('CUMULATIVE_ERROR_CORRUPTION_ACCEPTED')
            print('forward_p_joint_contract_corruption_checks=PASS corruptions=7 old_fail_not_promoted=1')
    result=(export_partial(a.repo,a.private_root,a.candidate,a.stop_preflight) if a.stop_preflight
            else export(a.repo,a.private_root,a.candidate))
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('forward_p_device_export=PASS cumulative_certification=BLOCKED')
