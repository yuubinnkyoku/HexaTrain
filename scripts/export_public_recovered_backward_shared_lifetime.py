"""Closed public pilot projection. A recovered pilot cannot reopen Device work."""
import argparse
import copy
import json
import math
import re
from pathlib import Path
from assess_backward_repair import ROOT,read,require
from assess_recovered_backward_shared_lifetime_pilot import assess
from certify_backward_shared_lifetime_formal_v2 import PRIVATE,PLAN,COUNTS,ROLES,ADDENDUM,manifest,write_once
from export_public_backward_shared_lifetime_formal_v2 import protected
from prepare_saved_p_backward import sha


def role_projection(a,digest):
    return dict(pairs=a['pairs'],coverage=a['coverage'],accuracy_metrics={name:v['maxAbs'] for name,v in a['metrics'].items()},
        numerical_digest=a['numerical_digest'],supplemental_role_contract_digest=digest,
        arithmetic_word_mismatches=a['arithmetic_word_mismatches'],observer_word_mismatches=a['observer_mismatch_words'],
        forward_joint_pass=a['forward_joint_pass'],execution_invariants=a['execution_invariants'],classification=a['classification'])


def template():
    return dict(schema=1,candidate='shared-lifetime-five-role-final',base_commit=read(PLAN)['base_commit'],compiled_source_commit=None,probe_sha256=None,dsp_sha256=None,host_sha256=None,
        plan_sha256=sha(PLAN),analysis_addendum_sha256=sha(ADDENDUM),classification='BLOCKED_FORMAL_COHORT',pilot_classification='SHARED_LIFETIME_FIVE_ROLE_PILOT_VERIFIED',
        blocker='POST_EXIT_ADB_TRANSFER_TIMEOUT',pilot_processes=1,formal_processes=0,formal_completed='0/3',pilot_formal_eligible=False,further_device_arithmetic='STOPPED_NO_RETRY',
        actual_exit=0,required_teardown='PASS',logger_leak='RETAINED_NOT_REPAIRED',duration_seconds=None,resource_counts=COUNTS.copy(),prepare_loads=1,prepare_unloads=1,inter_role_prepare_reload=0,
        first_dvalue_finalize_return=0,api6031_occurrences=0,dma6033_error_occurrences=0,sigsegv_observed=False,timing_false_matches={'6031':4,'6033':3},
        roles={},legacy_anchor_agreement='PASS_FIVE_PILOT_ROLES',three_process_legacy_agreement='NOT_EXECUTED',three_process_supplemental_agreement='NOT_EXECUTED',
        native_same_P_fanout='PASS_PILOT_ONLY',observer_free_accuracy='PASS_PILOT_ONLY',
        safety_during_process=dict(preflight='PASS',awake='PASS',known_top_resumed='PASS',focus_takeovers=0,recognizable_competing_workloads=0,critical_protection_events=0),
        transfer=dict(status='TIMEOUT_AFTER_EXIT0',five_role_files_verified=True,reconstructed_container_matches_original=True,partial_gzip_preserved=True,original_identity_fabricated=False,new_device_arithmetic=0),
        crash_capture=dict(logcat_snapshots='AVAILABLE',runtime_stderr='AVAILABLE',tombstones='AVAILABLE',dropbox='AVAILABLE',continuous_stream='LIMITED_NO_OWNED_RECORDS',faulting_library=None,relative_offset=None,main_thread_stack=None),
        host_gates={},performance=dict(status='NOT_MEASURED',figures=None),fresh_device_regressions='NOT_EXECUTED',negative_controls='PRESERVED',protected_state='PASS',
        hypothesis='BACKEND_REINITIALIZATION_HYPOTHESIS_STRENGTHENED',root_cause='NOT_PROVEN',training='NO-GO',full_attention_backward='NOT_CERTIFIED',full_attention='NOT_CERTIFIED',full1_layer='NOT_CERTIFIED',full19_layer='NOT_CERTIFIED',training_ready=False)


def validate(v,expected_identity=None):
    t=template();require(set(v)==set(t),'missing/private public field')
    variable={'compiled_source_commit','probe_sha256','dsp_sha256','host_sha256','duration_seconds','roles','host_gates'}
    require(all(v[k]==t[k] for k in set(t)-variable),'pilot/transport/formal/scope promotion')
    if expected_identity is None:
        m=manifest();expected_identity=dict(compiled_source_commit=m['source_sha'],probe_sha256=m['probe_sha256'],dsp_sha256=m['binary_hashes']['libHexaBackwardRepairDsp.so'],host_sha256=m['binary_hashes']['libHexaBackwardRepairHost.so'])
    require({k:v[k] for k in expected_identity}==expected_identity,'mixed/wrong compiled candidate')
    frozen=read(PLAN)['frozen'];require(v['dsp_sha256']==frozen['original_binary_sha256']['dsp'] and v['host_sha256']==frozen['original_binary_sha256']['host'],'DSP/Host replacement')
    require(v['probe_sha256'] not in (frozen['original_binary_sha256']['probe'],frozen['two_role_arm_sha256']),'old binary promoted')
    for name,n in [('compiled_source_commit',40),('probe_sha256',64),('dsp_sha256',64),('host_sha256',64)]:require(isinstance(v[name],str) and re.fullmatch('[0-9a-f]{'+str(n)+'}',v[name]) is not None,'public source/binary hash')
    require(isinstance(v['duration_seconds'],(int,float)) and math.isfinite(v['duration_seconds']) and 0<v['duration_seconds']<=1200,'actual process duration')
    require(list(v['roles'])==ROLES,'five roles/order')
    for role,a in v['roles'].items():
        require(set(a)=={'pairs','coverage','accuracy_metrics','numerical_digest','supplemental_role_contract_digest','arithmetic_word_mismatches','observer_word_mismatches','forward_joint_pass','execution_invariants','classification'},'role private/missing field')
        kinds=[0,1] if role=='dual' else [int(role.endswith('dvalue'))]
        require(a['pairs']==(19648 if role=='dual' else 9824) and a['classification']=='PASS' and a['forward_joint_pass'] is True and a['execution_invariants'] is True,'full role/P contract')
        require(a['coverage']=={f'O{o}T{t}_{s}_consumer{x}':n for o in (1,0) for t in (1,0) for s,n in [('core',2200),('holdout',256)] for x in kinds},'observer Core/Holdout coverage')
        require(a['numerical_digest']==read(PLAN)['anchor_legacy_digests'][role] and a['arithmetic_word_mismatches']==a['observer_word_mismatches']==0,'digest or words mismatch')
        require(re.fullmatch('[0-9a-f]{64}',a['supplemental_role_contract_digest']) is not None,'supplemental digest')
        allowed={name+'_'+str(x) for x in kinds for name in ('local','total','core','holdout','p_induced_actual')}
        if role.startswith('native_') or role=='dual':allowed|={'P_vs_ideal','P_row_sum'}|{'p_induced_bound_'+str(x) for x in kinds}
        require(set(a['accuracy_metrics'])==allowed and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in a['accuracy_metrics'].values()),'private/invalid accuracy metric')
        for x in kinds:require(a['accuracy_metrics']['total_'+str(x)]<=(.01 if x else .002),'accuracy gate')
        if 'P_vs_ideal' in allowed:
            require(a['accuracy_metrics']['P_vs_ideal']<=1e-6 and a['accuracy_metrics']['P_row_sum']<=2e-6,'Forward P joint gate')
            for x in kinds:require(a['accuracy_metrics']['p_induced_bound_'+str(x)]<=(.001 if x else .00025),'P-induced bound')
    require(set(v['host_gates'])=={'Fast','Host','Qnn','lifecycle','source_audit','exporter'} and all(x=='PASS' for x in v['host_gates'].values()),'fresh Host gates')


def project():
    protected();m=manifest();r=assess();v=template();completion=read(PRIVATE/'pilot/execution-before-readback-private.json')
    v.update(compiled_source_commit=m['source_sha'],probe_sha256=m['probe_sha256'],dsp_sha256=m['binary_hashes']['libHexaBackwardRepairDsp.so'],host_sha256=m['binary_hashes']['libHexaBackwardRepairHost.so'],duration_seconds=completion['duration_seconds'],
        roles={name:role_projection(a,r['role_contract_digests'][name]) for name,a in r['roles'].items()})
    require(r['analysis_source_sha256']==sha(ROOT/'scripts/assess_recovered_backward_shared_lifetime_pilot.py'),'assessment source binding')
    require(r['logcat_stream_owned_records']==0 and r['timer_6031_false_matches']==4 and r['timer_6033_false_matches']==3,'actual capture/timing classification')
    gates=read(PRIVATE/'host-gates-private.json')
    for name,a in gates.items():
        require(a['status']=='PASS' and a['actual_exit']==0 and a['sha256']==sha(ROOT/a['log']),'fresh Host evidence binding');v['host_gates'][name]=a['status']
    validate(v);return v


def selftest():
    v=template();f=read(PLAN)['frozen']['original_binary_sha256']
    v.update(compiled_source_commit='a'*40,probe_sha256='b'*64,dsp_sha256=f['dsp'],host_sha256=f['host'],duration_seconds=1,host_gates={k:'PASS' for k in ('Fast','Host','Qnn','lifecycle','source_audit','exporter')})
    for role in ROLES:
        kinds=[0,1] if role=='dual' else [int(role.endswith('dvalue'))]
        metrics={name+'_'+str(x):0. for x in kinds for name in ('local','total','core','holdout','p_induced_actual')}
        if role.startswith('native_') or role=='dual':metrics.update(P_vs_ideal=0.,P_row_sum=0.);metrics.update({'p_induced_bound_'+str(x):0. for x in kinds})
        v['roles'][role]=dict(pairs=19648 if role=='dual' else 9824,coverage={f'O{o}T{t}_{s}_consumer{x}':n for o in (1,0) for t in (1,0) for s,n in [('core',2200),('holdout',256)] for x in kinds},accuracy_metrics=metrics,numerical_digest=read(PLAN)['anchor_legacy_digests'][role],supplemental_role_contract_digest='c'*64,arithmetic_word_mismatches=0,observer_word_mismatches=0,forward_joint_pass=True,execution_invariants=True,classification='PASS')
    expected={k:v[k] for k in ('compiled_source_commit','probe_sha256','dsp_sha256','host_sha256')}
    validate(v,expected);tests=[]
    for key,value in [('formal_processes',1),('formal_completed','3/3'),('pilot_processes',2),('pilot_formal_eligible',True),('further_device_arithmetic','RESUMED'),('actual_exit',None),('required_teardown',None),('first_dvalue_finalize_return',None),('resource_counts',None),('prepare_loads',2),('inter_role_prepare_reload',1),('sigsegv_observed',True),('api6031_occurrences',1),('dma6033_error_occurrences',1),('crash_capture',{}),('performance',{'status':'PASS','figures':1}),('fresh_device_regressions','PASS'),('protected_state','FAIL'),('private_path','secret'),('root_cause','PROVEN'),('safety_during_process',{}),('host_gates',{}),('three_process_legacy_agreement','PASS'),('three_process_supplemental_agreement','PASS'),('probe_sha256',read(PLAN)['frozen']['two_role_arm_sha256']),('dsp_sha256','d'*64),('host_sha256','d'*64)]:
        b=copy.deepcopy(v);b[key]=value;tests.append(b)
    for field,value in [('pairs',200),('coverage',{}),('numerical_digest','d'*64),('forward_joint_pass',False),('arithmetic_word_mismatches',1),('observer_word_mismatches',1),('supplemental_role_contract_digest',None),('private_path','secret')]:
        b=copy.deepcopy(v);b['roles']['dual'][field]=value;tests.append(b)
    b=copy.deepcopy(v);b['roles']['dual']['accuracy_metrics']['total_1']=.02;tests.append(b)
    b=copy.deepcopy(v);del b['roles']['native_softmax'];tests.append(b)
    b=copy.deepcopy(v);b['probe_sha256']='e'*64;tests.append(b)
    for b in tests:
        try:validate(b,expected)
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid public evidence accepted')
    print('recovered_public_exporter_selftest=PASS rejection_cases='+str(len(tests))+' device_calls=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.out is not None,'output required');v=project();write_once(a.out,v);print('recovered_pilot_public=PASS formal=0/3 further_device_arithmetic=STOPPED')
