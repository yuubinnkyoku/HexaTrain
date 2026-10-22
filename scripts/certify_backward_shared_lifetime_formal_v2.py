"""Append-only analysis revision: typed microsecond records are not API errors."""
import argparse
import hashlib
import json
import math
import numpy as np
import re
from pathlib import Path
from assess_backward_repair import ROOT, read, require, words, exact, fp32
from assess_backward_shared_lifetime_formal import assess_role
from certify_backward_native_cohort import ROLES, role_digest
from prepare_saved_p_backward import sha
from audit_backward_shared_lifetime_formal import audit
from backward_shared_lifetime_runtime_scope import runtime_error_text

PRIVATE=ROOT/'build/reports/g1-backward-shared-lifetime-formal'
PLAN=ROOT/'host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json'
BUNDLE=ROOT/'build/fp32-saved-p-backward-shared-lifetime-five-role-final'
RUNNER='scripts/run_backward_shared_lifetime_formal_v2.ps1'
ORIGINAL_RUNNER='scripts/run_backward_shared_lifetime_formal.ps1'
ADDENDUM=ROOT/'host_tests/fp32_custom/backward_shared_lifetime_formal_analysis_addendum.json'
STAGES=['pilot','formal-1','formal-2','formal-3','performance']
COUNTS=dict(backend_create=1,backend_free=1,device_create=1,device_free=1,context_create=5,context_free=5,graphs=20,executes=24560)


def write_once(path,value):
    if path.exists():require(read(path)==value,'existing evidence differs')
    else:
        with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def manifest():
    audit()
    m=read(BUNDLE/'build-identity.json');p=read(PLAN)
    require(m['candidate']==p['candidate'] and m['classification']=='FIVE_ROLE_REQUIRES_FRESH_QUALIFICATION','candidate impersonation')
    require(m['probe_sha256'] not in (p['frozen']['original_binary_sha256']['probe'],p['frozen']['two_role_arm_sha256']),'old ARM promotion')
    for name,h in m['source_hashes'].items():require(sha(ROOT/name)==h,'frozen source mismatch')
    for name,h in m['binary_hashes'].items():require(sha(BUNDLE/name)==h,'frozen binary mismatch')
    for name,kind in [('libHexaBackwardRepairDsp.so','dsp'),('libHexaBackwardRepairHost.so','host')]:
        require(m['binary_hashes'][name]==p['frozen']['original_binary_sha256'][kind],'original DSP/Host changed')
    require(m['host_gates']=='PASS','build Host gates')
    amendment=read(ADDENDUM)
    require(amendment['frozen_manifest_sha256']==sha(BUNDLE/'build-identity.json') and amendment['frozen_arm_sha256']==m['probe_sha256'],'analysis amendment frozen candidate binding')
    for name,h in amendment['analysis_source_hashes'].items():require(sha(ROOT/name)==h,'analysis source changed')
    require(amendment['arithmetic_changed'] is False and amendment['gates_changed'] is False and amendment['new_device_retry_authorized'] is False,'analysis scope')
    return m


def validate_success(r,stage,m):
    require(r['status']=='PASS' and r['stage']==stage and stage in STAGES[:4],'pilot/formal stage identity')
    require(r['build']==m and r['actual_exit']==0 and r['safety']=='PASS' and r['teardown']=='PASS','completion/candidate')
    require(r['counts']==COUNTS and r['prepare_loads']==r['prepare_unloads']==1 and r['inter_role_prepare_reload']==0,'resource counts')
    require(r['api6031']==r['token6033']==0 and r['sigsegv'] is False and r['crash_capture']=='AVAILABLE','runtime/crash evidence')
    require(list(r['roles'])==ROLES,'five roles/order')
    require(set(r['role_contract_digests'])==set(ROLES),'missing supplemental digest')
    anchors=read(PLAN)['anchor_legacy_digests']
    for role,a in r['roles'].items():
        kinds=[0,1] if role=='dual' else [int(role.endswith('dvalue'))]
        expected={f'O{o}T{t}_{split}_consumer{v}':n for o in (1,0) for t in (1,0) for split,n in [('core',2200),('holdout',256)] for v in kinds}
        require(a['mode']==role and a['pairs']==(19648 if role=='dual' else 9824) and a['coverage']==expected,'observer Core/Holdout coverage')
        require(a['classification']=='PASS' and all(a[k] is True for k in ('api_success','execution_invariants','head_mapping','receipt_words_verified','forward_joint_pass')),'P ownership/accuracy contract')
        require(all(a[k]==0 for k in ('required_failed_pairs','observer_mismatch_words','arithmetic_word_mismatches')),'word mismatch')
        require(a['numerical_digest']==anchors[role],'legacy digest mismatch')
        require(re.fullmatch('[0-9a-f]{64}',r['role_contract_digests'][role]) is not None,'supplemental digest format')
        for v in kinds:require(0<=a['metrics']['total_'+str(v)]['maxAbs']<=(.01 if v else .002),'numerical accuracy')


def prerequisites(stage):
    require(stage in STAGES,'unregistered process')
    require(not (PRIVATE/'mission-stop-private.json').exists(),'mission Device stop active')
    m=manifest();prior=[]
    for name in STAGES[:STAGES.index(stage)]:
        r=certify(PRIVATE/name,check_prerequisites=False)
        validate_success(r,name,m);prior.append(r)
    require(len({r['owned_pid_private'] for r in prior})==len(prior),'previous OS processes reused')
    if prior:require(all(r['role_contract_digests']==prior[0]['role_contract_digests'] for r in prior),'prior supplemental agreement')
    return m,prior


def monitoring(directory,owned_pid):
    tops=[json.loads(x) for x in (directory/'top-private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    awake=[json.loads(x) for x in (directory/'awake-private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    require(tops and all(x['known'] is True and x['package_private'] and x['package_private']!='com.yuubinnkyoku.phonelm' for x in tops),'known top-resumed throughout')
    require(awake and all(x['exit']==0 and re.search(r'^\s*mWakefulness=Awake\s*$',x['text_private'],re.M) for x in awake),'Awake throughout')
    verified=set();infrastructure=directory/'infrastructure-private.jsonl'
    if infrastructure.exists():
        for line in infrastructure.read_text(encoding='utf-8-sig').splitlines():
            x=json.loads(line);parts=x['identity_private'].split('|')
            require(len(parts)==5 and x['verified'] is True,'infrastructure attestation')
            pid,ppid,user,name,cmd=parts;s=x['status_private'];c=x['cmdline_private']
            kernel=(ppid=='2' and user=='root' and name==cmd=='[kworker/R-policy_probe_wq]' and not c and re.search(r'^Kthread:\s*1\s*$',s,re.M) and re.search(r'^PPid:\s*2\s*$',s,re.M) and re.search(r'^Uid:\s*0\s+0\s+0\s+0\s*$',s,re.M))
            trace=(ppid=='1' and user=='nobody' and name==cmd=='traced_probes' and c=='/system/bin/traced_probes\0' and re.search(r'^Kthread:\s*0\s*$',s,re.M) and re.search(r'^PPid:\s*1\s*$',s,re.M) and re.search(r'^Uid:\s*9999\s+9999\s+9999\s+9999\s*$',s,re.M))
            require(kernel or trace,'unknown infrastructure exception');verified.add(tuple(parts))
    snapshots=[json.loads(x) for x in (directory/'competition-private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    require(snapshots and all(x['exit']==0 for x in snapshots),'competition snapshots')
    for x in snapshots:
        for line in x['text_private'].splitlines():
            m=re.match(r'^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+?)\s*$',line)
            if m and re.search(r'(probe|saved_p_backwa|qnn-net-run|simple_app|hmx|hexatrain|training)',m[4],re.I):
                require(int(m[1])==owned_pid or m.groups() in verified,'recognizable competing workload')


def certify(directory,check_prerequisites=True):
    directory=directory.resolve()
    require(directory.parent==PRIVATE.resolve() and directory.name in STAGES[:4],'actual exclusive pilot/formal directory')
    stage=directory.name
    runner=ORIGINAL_RUNNER if stage=='pilot' else RUNNER
    m,previous=prerequisites(stage) if check_prerequisites else (manifest(),[])
    identity=read(directory/'identity-private.json');completion=read(directory/'execution-before-readback-private.json')
    require(identity['exit_code']==completion['exit_code']==0,'missing actual successful OS exit')
    require(identity['stage']==completion['stage']==stage and identity['mode']==completion['mode']=='shared-five-role','actual five-role scope')
    require(identity['build']==completion['build']==m and identity['forward_limit']==60,'same immutable candidate')
    require(identity['orchestration_file']==completion['orchestration_file']==runner and identity['orchestration_sha256']==completion['orchestration_sha256']==sha(ROOT/runner),'runner binding')
    require(read(directory/'preflight/device-preflight-private.json')['status']=='PASS','fresh preflight')
    require(identity['focus_takeover_count']==0 and identity['protected_identity']=='PASS' and identity['safety_complete'] is True and not (directory/'stop-private.json').exists(),'safety incomplete')
    claim=read(PRIVATE/(stage+'-attempt-private.json'))
    permit=read(directory/'launch-permit-private.json')
    require(claim['stage']==stage and claim['maximum_os_processes']==1 and claim['retry']==0 and claim['candidate_sha256']==m['probe_sha256'],'one-shot claim')
    require(permit['candidate_sha256']==m['probe_sha256'] and permit['runner_sha256']==sha(ROOT/runner) and permit['plan_sha256']==sha(PLAN) and permit['host_inventory_sha256']==sha(directory/'host-processes-private.json'),'entry permit binding')
    require(all(permit[k] is True for k in ('host_workload_review_pass','host_gates_pass','protected_pass','source_delta_pass')),'entry review')
    text=(directory/'stdout.txt').read_text(encoding='utf-8-sig');err=(directory/'stderr.txt').read_text(encoding='utf-8-sig')
    pids=re.findall(r'^owned_probe_pid=(\d+)\s*$',text,re.M);require(len(pids)==1,'one owned process')
    monitoring(directory,int(pids[0]))
    apis=[x.split(',') for x in text.splitlines() if x.startswith('api,')]
    require(apis and all(len(x)==3 and x[2]=='0' for x in apis),'QNN API failure')
    mapping=dict(backend_create='backendCreate',backend_free='backendFree',device_create='deviceCreate',device_free='deviceFree',context_create='contextCreate',context_free='contextFree',graphs='graphCreate',executes='graphExecute')
    counts={key:sum(x[1]==name for x in apis) for key,name in mapping.items()};require(counts==COUNTS,'resource counts')
    require(sum(x[1]=='registerForwardPackage' for x in apis)==8 and sum(x[1]=='registerRepairHost' for x in apis)==sum(x[1]=='registerRepairDSP' for x in apis)==1,'package registration lifetime')
    require(text.count('lifetime,graphFinalize_entry\n')==text.count('lifetime,graphFinalize_return,0\n')==20,'finalize entry/actual return')
    require(text.count('summary,repair_cohort,roles=5,all_role_exits=0,independent_os_processes=1')==1 and text.count('shared_process,mode=shared-five-role,exit=0')==1,'whole-process terminal')
    order=[]
    for role in ROLES:
        start='cohort_begin,mode='+role+'\n';end='cohort_end,mode='+role+',exit=0\n'
        require(text.count(start)==text.count(end)==1,'role boundary')
        a=text.index(start);b=text.index(end);require(a<b,'boundary order');order.extend([a,b])
        segment=text[a:b]
        require(segment.count('api,contextCreate,0\n')==segment.count('api,contextFree,0\n')==1,'role context lifetime')
        require(all('api,'+n+',' not in segment for n in ('backendCreate','backendFree','deviceCreate','deviceFree')),'inter-role backend/device lifetime')
    require(order==sorted(order),'role order')
    require(err.count('PrepareLibLoader Loading libQnnHtpPrepare.so')==err.count('PrepareLibLoader Unloading libQnnHtpPrepare.so')==1,'Prepare reload/unknown')
    require('QnnBackend_free started' in err and err.index('PrepareLibLoader Unloading')>err.index('QnnBackend_free started'),'Prepare unload before final backend teardown')
    for n in ('logcat-before','logcat-after','tombstones-before','tombstones-after','dropbox-before','dropbox-after'):
        require(read(directory/(n+'-private.json'))['exit']==0,'required read-only crash collection')
    require((directory/'logcat-stream-private.txt').stat().st_size>0,'continuous capture absent')
    competition=[json.loads(x) for x in (directory/'competition-private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    require(competition and all(x['exit']==0 for x in competition),'competition monitoring absent')
    for n,h in identity['role_raw'].items():
        require(sha(directory/n)==h['sha256'] and sha(directory/(n+'.gz'))==h['gzip_sha256'],'raw/gzip evidence changed')
    # Container and original role files must agree byte for byte; exclusive readback.
    with (directory/'saved-p-words.bin').open('rb') as raw:
        require(tuple(words(raw,4))==(0x42524301,1,5,60),'full cohort container')
        for code,role in zip((0,1,3,4,5),ROLES):
            actual,lo,hi=map(int,words(raw,3));size=lo+(hi<<32)
            require(actual==code and 16<size<1<<31,'role code/size')
            with (directory/('cohort-'+role+'.bin')).open('rb') as part:
                while size:
                    b=raw.read(min(size,1<<20));require(b and part.read(len(b))==b,'role/container mismatch');size-=len(b)
                require(part.read(1)==b'','trailing role bytes')
        require(raw.read(1)==b'','trailing cohort bytes')
    roles={role:assess_role(directory,role,m,identity['exit_code']) for role in ROLES}
    result=dict(status='PASS',stage=stage,build=m,actual_exit=0,safety='PASS',teardown='PASS',counts=counts,
        prepare_loads=1,prepare_unloads=1,inter_role_prepare_reload=0,
        api6031=len(re.findall(r'(?<!\d)6031(?!\d)',runtime_error_text(text,err))),token6033=len(re.findall(r'(?<!\d)6033(?!\d)',runtime_error_text(text,err))),sigsegv=False,crash_capture='AVAILABLE',
        analysis_addendum_sha256=sha(ADDENDUM),roles=roles,role_contract_digests={role:role_digest(directory/('cohort-'+role+'.bin'),role,r) for role,r in roles.items()},
        owned_pid_private=int(pids[0]),identity_sha256=sha(directory/'identity-private.json'),raw_sha256=sha(directory/'saved-p-words.bin'))
    validate_success(result,stage,m)
    for prior in previous:
        require(prior['owned_pid_private']!=result['owned_pid_private'],'OS process reused')
        require(prior['role_contract_digests']==result['role_contract_digests'],'supplemental digest mismatch')
    write_once(directory/'qualification-private.json',result)
    return result


def performance(directory):
    directory=directory.resolve();require(directory==PRIVATE.resolve()/'performance','performance directory')
    m,prior=prerequisites('performance');require(len(prior)==4,'three formal processes plus separate pilot')
    identity=read(directory/'identity-private.json');completion=read(directory/'execution-before-readback-private.json')
    require(identity['stage']==completion['stage']=='performance' and identity['mode']==completion['mode']=='shared-performance','performance scope')
    require(identity['build']==completion['build']==m and identity['exit_code']==completion['exit_code']==0,'actual same candidate completion')
    require(identity['orchestration_sha256']==sha(ROOT/RUNNER) and completion['orchestration_sha256']==sha(ROOT/RUNNER),'performance runner identity')
    require(identity['focus_takeover_count']==0 and identity['safety_complete'] is True and not (directory/'stop-private.json').exists(),'performance safety')
    require(read(directory/'preflight/device-preflight-private.json')['status']=='PASS','fresh performance preflight')
    claim=read(PRIVATE/'performance-attempt-private.json');require(claim['candidate_sha256']==m['probe_sha256'] and claim['retry']==0,'one-shot performance claim')
    text=(directory/'stdout.txt').read_text(encoding='utf-8-sig');err=(directory/'stderr.txt').read_text(encoding='utf-8-sig')
    pids=re.findall(r'^owned_probe_pid=(\d+)\s*$',text,re.M);require(len(pids)==1,'performance process');monitoring(directory,int(pids[0]))
    require(not re.search(r'(?<!\d)(6031|6033)(?!\d)',runtime_error_text(text,err)),'performance API6031/6033')
    apis=[x.split(',') for x in text.splitlines() if x.startswith('api,')]
    require(all(len(x)==3 and x[2]=='0' for x in apis),'performance QNN return')
    mapping=dict(backendCreate=1,backendFree=1,deviceCreate=1,deviceFree=1,contextCreate=4,contextFree=4,graphCreate=16,graphFinalize=16,graphExecute=512)
    require(all(sum(x[1]==k for x in apis)==n for k,n in mapping.items()),'performance resources/samples')
    require(text.count('shared_process,mode=shared-performance,exit=0')==1,'performance teardown')
    require(text.count('summary,shared_performance_validation,words=PASS,accuracy=PASS,executes=512')==1,'performance validation terminal')
    require(err.count('PrepareLibLoader Loading libQnnHtpPrepare.so')==err.count('PrepareLibLoader Unloading libQnnHtpPrepare.so')==1,'performance Prepare lifetime')
    records=[json.loads(x) for x in text.splitlines() if x.startswith('{') and json.loads(x).get('kind')=='shared_performance']
    expected=[(mode,o,t) for mode in ('forward','native_softmax','native_dvalue','dual') for o in (1,0) for t in (1,0)]
    require([(r['mode'],r['observed'],r['tap']) for r in records]==expected,'performance mode/observer order')
    fields={'kind','mode','observed','tap','samples','first_us','median_us','p10_us','p90_us','finalize_us','nodes','tensors','native_logical_bytes','host_logical_bytes','host_output_guard_bytes','explicit_scratch_bytes','static_stack_frame_bytes','arithmetic_us','receipt_inclusive_us'}
    for r in records:
        require(set(r)==fields and r['samples']==31,'performance public fields/sample count')
        require(all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) and r[k]>=0 for k in ('first_us','median_us','p10_us','p90_us','finalize_us','nodes','tensors','native_logical_bytes','host_logical_bytes','host_output_guard_bytes')),'invalid timing/memory')
        require(r['p10_us']<=r['median_us']<=r['p90_us'],'timing quantiles')
        require(r['explicit_scratch_bytes'] is None and r['static_stack_frame_bytes'] is None,'unobserved runtime memory')
        if not r['observed'] or r['mode']=='forward':require(r['arithmetic_us'] is None and r['receipt_inclusive_us'] is None,'NoReceipt timing invented')
    setup={}
    for x in text.splitlines():
        if x.startswith('api_us,'):
            _,name,value=x.split(',');v=float(value);require(math.isfinite(v) and v>=0,'setup timer');setup.setdefault(name,[]).append(v)
    require(all(len(setup.get(k,[]))==n for k,n in [('backendCreate',1),('deviceCreate',1),('contextCreate',4),('contextFree',4),('deviceFree',1),('backendFree',1),('backendRegisterOpPackage',10)]),'separate process/role setup')
    for name in ('logcat-before','logcat-after','dropbox-before','dropbox-after','tombstones-before','tombstones-after'):require(read(directory/(name+'-private.json'))['exit']==0,'performance crash capture')
    require((directory/'logcat-stream-private.txt').stat().st_size>0,'performance continuous capture')
    path=directory/'performance-words.bin';raw_identity=identity['role_raw'][path.name]
    require(sha(path)==raw_identity['sha256'] and sha(directory/(path.name+'.gz'))==raw_identity['gzip_sha256'],'performance raw binding')
    fixture=(ROOT/'build/fp32-double-qk/double-inputs.bin').read_bytes()
    gradients=(ROOT/'build/reports/g1-native-saved-p-backward-dsp/gradients.bin').read_bytes()
    frozen=read(ROOT/'host_tests/fp32_custom/saved_p_backward_device_plan.json')
    require(hashlib.sha256(fixture).hexdigest()==frozen['protected_forward_identities']['double_fixture'] and hashlib.sha256(gradients).hexdigest()==frozen['gradient_sha256'],'performance frozen operands')
    anchors={};maximum={};records_checked=0
    with path.open('rb') as raw:
        require(tuple(words(raw,4))==(0x50465201,1,4,32),'performance raw header')
        for mode in (2,3,4,5):
            for o in (1,0):
                for t in (1,0):
                    for sample in range(32):
                        for h in range(2):
                            for v in ([2] if mode==2 else [0] if mode==3 else [1] if mode==4 else [0,1]):
                                require(tuple(words(raw,6))==(mode,o,t,sample,h,v),'performance case order')
                                pw,gw,yw=[words(raw,1024) for _ in range(3)];p,g,y=[x.view('<f4').reshape(32,32) for x in (pw,gw,yw)]
                                require(all(np.isfinite(x).all() for x in (p,g,y)),'performance finite')
                                if h not in anchors:anchors[h]=pw
                                require(np.array_equal(anchors[h],pw),'performance observer/P word identity')
                                if v==2:require(np.array_equal(pw,gw) and np.array_equal(pw,yw),'Forward P output words')
                                else:
                                    expected=np.frombuffer(gradients,dtype='<u4',count=1024,offset=16+((9*2+h)*2+v)*4096)
                                    require(np.array_equal(gw,expected),'performance gradient/head')
                                    require(np.array_equal(yw,fp32(p,g,v).view('<u4').reshape(-1)),'performance ordered FP32 words')
                                for ri in (0,1):
                                    ideal=np.frombuffer(fixture,dtype='<f8',count=1024,offset=12+19*122888+90120+(ri*2+h)*8192).reshape(32,32)
                                    require(np.max(np.abs(p.astype(np.float64)-ideal))<=1e-6 and np.max(np.abs(p.astype(np.float64).sum(-1)-1))<=2e-6,'performance Forward P joint accuracy')
                                    if v!=2:
                                        error=float(np.max(np.abs(y.astype(np.float64)-exact(ideal,g,v))))
                                        require(error<=(.01 if v else .002),'performance independent double oracle')
                                        maximum[str(v)]=max(maximum.get(str(v),0.),error)
                                records_checked+=1
        require(raw.read(1)==b'','performance trailing raw')
    require(records_checked==1280,'performance validation coverage')
    result=dict(status='PASS',actual_exit=0,classification='BACKWARD_REPAIR_PERFORMANCE_MEASURED',records=records,setup_teardown_us=setup,validation_records=records_checked,total_maxAbs=maximum,identity_sha256=sha(directory/'identity-private.json'),stdout_sha256=sha(directory/'stdout.txt'),raw_sha256=sha(path))
    write_once(directory/'performance-private.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=Path);p.add_argument('--entry',choices=STAGES);a=p.parse_args()
    if a.entry:
        m,prior=prerequisites(a.entry);print(json.dumps(dict(entry='PASS',stage=a.entry,prior_passes=len(prior),probe_sha256=m['probe_sha256'])))
    else:
        require(a.report_root is not None,'report root required')
        if a.report_root.name=='performance':
            r=performance(a.report_root);print(json.dumps(dict(status=r['status'],classification=r['classification'])))
        else:
            r=certify(a.report_root)
            print(json.dumps(dict(status=r['status'],stage=r['stage'],numerical_digests={k:v['numerical_digest'] for k,v in r['roles'].items()},role_contract_digests=r['role_contract_digests'])))
