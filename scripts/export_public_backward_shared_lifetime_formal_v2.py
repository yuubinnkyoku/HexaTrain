"""Closed allow-list projection; private objects are never serialized wholesale."""
import argparse
import copy
import json
import re
import subprocess
from pathlib import Path
from assess_backward_repair import ROOT, read, require
from prepare_saved_p_backward import sha
from certify_backward_shared_lifetime_formal_v2 import (PRIVATE,PLAN,BUNDLE,ROLES,COUNTS,ADDENDUM,RUNNER,manifest,certify,performance,validate_success,write_once)
from export_public_backward_shared_lifetime import integrity as historical_integrity


def protected():
    historical_integrity()
    a=read(PRIVATE/'starting-audit-private.json')
    require(a['status']=='PASS' and a['base']==read(PLAN)['base_commit'],'research provenance')
    for key,path in [('previous_diagnostic_public_sha256','docs/results/g1-backward-shared-lifetime-diagnostic-summary.json'),('previous_report_sha256','docs/g1-backward-shared-lifetime-diagnostic.md'),('diagnostic_manifest_sha256','build/fp32-saved-p-backward-shared-lifetime-v1/build-identity.json')]:
        require(a[key]==sha(ROOT/path),'prior report/manifest changed')
    # The prior diagnostic's complete private evidence set is frozen independently.
    for field in ('prior_diagnostic_evidence','diagnostic_evidence','previous_diagnostic_evidence'):
        if field in a:
            for p,h in a[field].items():require(sha(ROOT/p)==h,'old diagnostic evidence changed')
    refs=dict(line.split()[::-1] for line in subprocess.check_output(['git','show-ref'],cwd=ROOT,text=True).splitlines())
    lines=a['refs'] if isinstance(a['refs'],str) else None
    if lines:
        for line in lines.splitlines():
            h,ref=line.split();require(refs.get(ref)==h,'protected refs changed')
    return 'PASS'


def safe_role(r):
    return dict(pairs=r['pairs'],coverage=r['coverage'],classification=r['classification'],
        numerical_digest=r['numerical_digest'],arithmetic_word_mismatches=r['arithmetic_word_mismatches'],observer_mismatch_words=r['observer_mismatch_words'],
        total_maxAbs={str(v):r['metrics']['total_'+str(v)]['maxAbs'] for v in (0,1) if 'total_'+str(v) in r['metrics']},
        forward_joint_pass=r['forward_joint_pass'])


def safe_process(r):
    return dict(stage=r['stage'],actual_exit=r['actual_exit'],teardown=r['teardown'],safety=r['safety'],counts=r['counts'],
        inter_role_prepare_reload=r['inter_role_prepare_reload'],api6031=r['api6031'],token6033=r['token6033'],sigsegv=r['sigsegv'],
        roles={k:safe_role(v) for k,v in r['roles'].items()},role_contract_digests=r['role_contract_digests'])


def skeleton(m):
    frozen=read(PLAN)['frozen']['original_binary_sha256']
    return dict(schema=1,base_commit=read(PLAN)['base_commit'],candidate=read(PLAN)['candidate'],
        analysis_addendum_sha256=sha(ADDENDUM),formal_runner_sha256=sha(ROOT/RUNNER),analysis_certifier_sha256=sha(ROOT/'scripts/certify_backward_shared_lifetime_formal_v2.py'),
        source_commit=m['source_sha'] if m else None,probe_sha256=m['probe_sha256'] if m else None,
        dsp_sha256=frozen['dsp'],host_sha256=frozen['host'],protected_state='PASS',source_delta='LIFETIME_ORCHESTRATION_ONLY',
        classification='BLOCKED_SHARED_LIFETIME_PILOT',pilot=None,formal=[],formal_completed='0/3',
        performance=None,regression='NOT_EXECUTED',host_gates={},root_cause='NOT_PROVEN',
        hypothesis='SUPPORTED_BY_PREVIOUS_TWO_ROLE_INTERVENTION',training='NO-GO',full_attention='NOT_CERTIFIED',full1_layer='NOT_CERTIFIED',full19_layer='NOT_CERTIFIED',training_ready=False)


def validate_public(v):
    template=skeleton(None);require(set(v)==set(template),'private/unknown public field')
    require(all(v[k]==template[k] for k in ('analysis_addendum_sha256','formal_runner_sha256','analysis_certifier_sha256')),'analysis identity')
    require(v['candidate']==template['candidate'] and v['protected_state']=='PASS' and v['source_delta']==template['source_delta'],'protected/candidate scope')
    for k in ('dsp_sha256','host_sha256'):require(v[k]==template[k],'DSP/Host replacement')
    p=read(PLAN)
    require(v['probe_sha256'] not in (p['frozen']['original_binary_sha256']['probe'],p['frozen']['two_role_arm_sha256']),'old candidate promotion')
    for key,n in [('base_commit',40),('source_commit',40),('probe_sha256',64),('dsp_sha256',64),('host_sha256',64)]:
        require(v[key] is None or re.fullmatch('[0-9a-f]{'+str(n)+'}',v[key]) is not None,'public hash identity')
    require(v['training']=='NO-GO' and v['root_cause']=='NOT_PROVEN' and v['training_ready'] is False,'scientific promotion')
    require(all(v[k]=='NOT_CERTIFIED' for k in ('full_attention','full1_layer','full19_layer')),'layer promotion')
    require(v['regression']=='NOT_EXECUTED','unvalidated regression promotion')
    if v['performance'] is not None:
        perf=v['performance'];require(len(v['formal'])==3 and set(perf)=={'classification','records','setup_teardown_us'},'unmeasured/private performance promotion')
        require(perf['classification']=='BACKWARD_REPAIR_PERFORMANCE_MEASURED' and len(perf['records'])==16,'performance completion')
        expected_fields={'kind','mode','observed','tap','samples','first_us','median_us','p10_us','p90_us','finalize_us','nodes','tensors','native_logical_bytes','host_logical_bytes','host_output_guard_bytes','explicit_scratch_bytes','static_stack_frame_bytes','arithmetic_us','receipt_inclusive_us'}
        for r in perf['records']:
            require(set(r)==expected_fields and r['kind']=='shared_performance' and r['mode'] in ('forward','native_softmax','native_dvalue','dual'),'private performance field')
            require(r['observed'] in (0,1) and r['tap'] in (0,1) and r['samples']==31,'performance condition')
            for key in expected_fields-{'kind','mode'}:require(r[key] is None or isinstance(r[key],(int,float)),'private/non-numerical performance value')
        require(set(perf['setup_teardown_us'])=={'logCreate','backendCreate','deviceCreate','contextCreate','contextFree','deviceFree','backendFree','backendRegisterOpPackage'},'private setup field')
        require(all(isinstance(xs,list) and all(isinstance(x,(int,float)) and x>=0 for x in xs) for xs in perf['setup_teardown_us'].values()),'setup values')
    require(v['formal_completed']==str(len(v['formal']))+'/3' and len(v['formal'])<=3,'false formal count')
    expected_process={'stage','actual_exit','teardown','safety','counts','inter_role_prepare_reload','api6031','token6033','sigsegv','roles','role_contract_digests'}
    for r,stage in [(v['pilot'],'pilot')]+[(r,'formal-'+str(i+1)) for i,r in enumerate(v['formal'])]:
        if r is None:continue
        require(set(r)==expected_process and r['stage']==stage,'pilot promotion/private process field')
        require(r['actual_exit']==0 and r['teardown']==r['safety']=='PASS' and r['counts']==COUNTS and r['inter_role_prepare_reload']==0,'actual exit/teardown/lifetime')
        require(r['api6031']==r['token6033']==0 and r['sigsegv'] is False,'runtime failure')
        require(list(r['roles'])==ROLES and set(r['role_contract_digests'])==set(ROLES),'missing role/digest')
        for role,a in r['roles'].items():
            require(set(a)=={'pairs','coverage','classification','numerical_digest','arithmetic_word_mismatches','observer_mismatch_words','total_maxAbs','forward_joint_pass'},'private role field')
            kinds=[0,1] if role=='dual' else [int(role.endswith('dvalue'))]
            require(a['pairs']==(19648 if role=='dual' else 9824) and a['classification']=='PASS' and a['forward_joint_pass'] is True,'full role/P contract')
            require(a['coverage']=={f'O{o}T{t}_{s}_consumer{x}':n for o in (1,0) for t in (1,0) for s,n in [('core',2200),('holdout',256)] for x in kinds},'missing observer/Core/Holdout')
            require(a['numerical_digest']==p['anchor_legacy_digests'][role] and a['arithmetic_word_mismatches']==a['observer_mismatch_words']==0,'digest/words')
            require(set(a['total_maxAbs'])==set(map(str,kinds)) and all(0<=a['total_maxAbs'][str(x)]<=(.01 if x else .002) for x in kinds),'numerical accuracy')
            require(re.fullmatch('[0-9a-f]{64}',r['role_contract_digests'][role]) is not None,'supplemental digest')
    require(not v['formal'] or v['pilot'] is not None,'formal without pilot')
    for r in v['formal']:require(r['role_contract_digests']==v['pilot']['role_contract_digests'],'false numerical agreement')
    require(v['classification'] in ('BLOCKED_DEVICE_SAFETY','BLOCKED_SHARED_LIFETIME_PILOT','BLOCKED_RUNTIME_139','BLOCKED_RUNTIME_6031','BLOCKED_NUMERICAL_ACCURACY','BLOCKED_OBSERVER_IDENTITY','BLOCKED_FORMAL_COHORT','SHARED_LIFETIME_FORMAL_COHORT_VERIFIED','UNRESOLVED'),'classification')
    if v['classification']=='SHARED_LIFETIME_FORMAL_COHORT_VERIFIED':require(len(v['formal'])==3,'false formal certification')
    for key,value in v['host_gates'].items():require(key in ('Fast','Host','Qnn','lifecycle','source_audit','exporter') and value in ('PASS','FAIL','NOT_EXECUTED'),'host gate field')


def project():
    protected();m=manifest() if (BUNDLE/'build-identity.json').exists() else None
    v=skeleton(m)
    host=PRIVATE/'host-gates-private.json'
    if host.exists():
        for k,record in read(host).items():
            require(record['sha256']==sha(ROOT/record['log']),'host gate freshness/binding')
            v['host_gates'][k]=record['status']
    for name in ('pilot','formal-1','formal-2','formal-3'):
        directory=PRIVATE/name
        if not (directory/'qualification-private.json').exists():break
        r=certify(directory,check_prerequisites=False);safe=safe_process(r)
        if name=='pilot':v['pilot']=safe;v['classification']='BLOCKED_FORMAL_COHORT'
        else:v['formal'].append(safe)
    v['formal_completed']=str(len(v['formal']))+'/3'
    if len(v['formal'])==3:v['classification']='SHARED_LIFETIME_FORMAL_COHORT_VERIFIED'
    if (PRIVATE/'performance/performance-private.json').exists():
        perf=performance(PRIVATE/'performance');v['performance']={k:perf[k] for k in ('classification','records','setup_teardown_us')}
    stop=PRIVATE/'mission-stop-private.json'
    if stop.exists():
        reason=read(stop)['reason']
        if any(x in reason for x in ('SAFETY','AWAKE','TOP_','FOREGROUND','COMPET','CRITICAL','PREFLIGHT','ADB_')):v['classification']='BLOCKED_DEVICE_SAFETY'
        elif '139' in reason:v['classification']='BLOCKED_RUNTIME_139'
        elif '6031' in reason or '6033' in reason:v['classification']='BLOCKED_RUNTIME_6031'
    validate_public(v);return v


def selftest():
    m=dict(source_sha='a'*40,probe_sha256='b'*64)
    private=dict(status='PASS',stage='pilot',build=m,actual_exit=0,safety='PASS',teardown='PASS',counts=COUNTS.copy(),prepare_loads=1,prepare_unloads=1,inter_role_prepare_reload=0,api6031=0,token6033=0,sigsegv=False,crash_capture='AVAILABLE',roles={},role_contract_digests={r:'c'*64 for r in ROLES})
    for role in ROLES:
        kinds=[0,1] if role=='dual' else [int(role.endswith('dvalue'))]
        private['roles'][role]=dict(mode=role,pairs=19648 if role=='dual' else 9824,classification='PASS',coverage={f'O{o}T{t}_{s}_consumer{x}':n for o in (1,0) for t in (1,0) for s,n in [('core',2200),('holdout',256)] for x in kinds},api_success=True,execution_invariants=True,head_mapping=True,receipt_words_verified=True,forward_joint_pass=True,required_failed_pairs=0,observer_mismatch_words=0,arithmetic_word_mismatches=0,numerical_digest=read(PLAN)['anchor_legacy_digests'][role],metrics={'total_'+str(x):{'maxAbs':0.} for x in kinds})
    validate_success(private,'pilot',m)
    v=skeleton(m);v['pilot']=safe_process(private);validate_public(v)
    tests=[]
    for key,value in [('actual_exit',None),('teardown','INCOMPLETE'),('counts',None),('inter_role_prepare_reload',1),('prepare_loads',2),('prepare_unloads',None),('crash_capture',None),('safety','BLOCKED'),('build',{}),('stage','formal-1')]:
        b=copy.deepcopy(private);b[key]=value;tests.append(lambda b=b:validate_success(b,'pilot',m))
    for name in ('backend_create','device_create','context_create','graphs'):
        b=copy.deepcopy(private);b['counts'][name]+=1;tests.append(lambda b=b:validate_success(b,'pilot',m))
    b=copy.deepcopy(private);del b['roles']['native_softmax'];tests.append(lambda b=b:validate_success(b,'pilot',m))
    for rolekey,value in [('pairs',200),('coverage',{}),('numerical_digest','d'*64),('forward_joint_pass',False),('arithmetic_word_mismatches',1),('observer_mismatch_words',1),('classification','FAIL')]:
        b=copy.deepcopy(private);b['roles']['dual'][rolekey]=value;tests.append(lambda b=b:validate_success(b,'pilot',m))
    for key,value in [('performance',{'median_us':1}),('regression','PASS'),('private_path','secret'),('protected_state','FAIL'),('formal_completed','3/3'),('classification','SHARED_LIFETIME_FORMAL_COHORT_VERIFIED'),('probe_sha256',read(PLAN)['frozen']['two_role_arm_sha256']),('dsp_sha256','e'*64)]:
        b=copy.deepcopy(v);b[key]=value;tests.append(lambda b=b:validate_public(b))
    b=copy.deepcopy(v);b['formal']=[copy.deepcopy(v['pilot'])];b['formal_completed']='1/3';tests.append(lambda:validate_public(b))
    b2=copy.deepcopy(v);r=copy.deepcopy(v['pilot']);r['stage']='formal-1';r['role_contract_digests']['dual']='d'*64;b2['formal']=[r];b2['formal_completed']='1/3';tests.append(lambda:validate_public(b2))
    b3=copy.deepcopy(v);b3['pilot']['roles']['dual']['private_path']='secret';tests.append(lambda:validate_public(b3))
    for check in tests:
        try:check()
        except (ValueError,KeyError,TypeError):pass
        else:raise AssertionError('invalid evidence accepted')
    print('formal_exporter_selftest=PASS rejection_cases='+str(len(tests))+' device_calls=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.out is not None,'output required');v=project();write_once(a.out,v);print('formal_public_projection='+v['classification'])
