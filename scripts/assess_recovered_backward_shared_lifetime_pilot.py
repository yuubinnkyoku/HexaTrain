"""Host-only assessment after exit0 and a later interrupted container transfer.

No ADB calls. Original evidence, original runner identity and stopped mission
remain untouched. The oracle view cannot qualify as any formal process.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from assess_backward_repair import ROOT,read,require
from assess_backward_shared_lifetime_formal import assess_role
from certify_backward_native_cohort import role_digest
from certify_backward_shared_lifetime_formal_v2 import (PRIVATE,PLAN,COUNTS,ROLES,ORIGINAL_RUNNER,manifest,monitoring,validate_success,write_once)
from backward_shared_lifetime_runtime_scope import runtime_error_text
from prepare_saved_p_backward import sha


def validate_receipt(r,completion):
    require(r['kind']=='HOST_ONLY_LOSSLESS_RECOVERY_AFTER_POST_EXIT_TRANSPORT_TIMEOUT' and r['stage']=='pilot','recovery provenance')
    require(r['actual_exit']==completion['exit_code']==0,'actual OS exit missing')
    require(r['new_device_arithmetic']==0 and r['formal_eligible'] is False and r['mission_stop_retained'] is True,'recovery/formal scope')
    require(r['original_raw_overwritten'] is False and r['original_identity_fabricated'] is False,'raw/identity preservation')
    require(r['recovered_container_sha256']==r['original_remote_container_sha256'],'lossless original container identity')
    require(set(r['role_raw'])=={'cohort-'+role+'.bin' for role in ROLES},'five role files')
    for a in r['role_raw'].values():require(a['remote_match'] is True and a['decompressed_match'] is True,'verified remote role evidence')


def assess():
    run=PRIVATE/'pilot';m=manifest();receipt=read(run/'recovery-receipt-private.json');completion=read(run/'execution-before-readback-private.json')
    validate_receipt(receipt,completion)
    require(read(PRIVATE/'mission-stop-private.json')['reason']=='ADB_TRANSPORT_TIMEOUT: adb operation failed','stopped transport history')
    require(not any((PRIVATE/(s+'-attempt-private.json')).exists() for s in ('formal-1','formal-2','formal-3','performance')),'additional arithmetic after stop')
    require(not (run/'identity-private.json').exists(),'do not substitute an original runner identity')
    require(receipt['completion_sha256']==sha(run/'execution-before-readback-private.json') and receipt['remote_hash_capture_sha256']==sha(run/'recovery-remote-hashes-private.json'),'recovery source binding')
    remote=read(run/'recovery-remote-hashes-private.json')
    require(remote['exit']==0 and remote['same_physical_device'] is True and remote['new_device_arithmetic']==0 and remote['read_only'] is True and remote['reconnect_performed'] is False,'read-only metadata provenance')
    require(sha(run/receipt['recovered_container_name'])==receipt['original_remote_container_sha256'],'reconstructed container changed')
    require(sha(run/'saved-p-words.bin.gz')==receipt['partial_container_gzip_sha256'],'original partial transfer changed')
    require(completion['build']==m and completion['stage']=='pilot' and completion['mode']=='shared-five-role' and completion['forward_limit']==60,'actual compiled candidate/scope')
    require(completion['orchestration_file']==ORIGINAL_RUNNER and completion['orchestration_sha256']==sha(ROOT/ORIGINAL_RUNNER),'preserved original runner identity')
    require(completion['focus_takeover_count']==0 and read(run/'preflight/device-preflight-private.json')['status']=='PASS' and not (run/'stop-private.json').exists(),'safety during the actual OS process')
    claim=read(PRIVATE/'pilot-attempt-private.json');permit=read(run/'launch-permit-private.json')
    require(claim['stage']=='pilot' and claim['maximum_os_processes']==1 and claim['retry']==0 and claim['candidate_sha256']==m['probe_sha256'],'single actual pilot')
    require(permit['candidate_sha256']==m['probe_sha256'] and permit['runner_sha256']==sha(ROOT/ORIGINAL_RUNNER) and permit['host_inventory_sha256']==sha(run/'host-processes-private.json') and permit['plan_sha256']==sha(PLAN),'entry permit binding')
    require(all(permit[k] is True for k in ('host_workload_review_pass','host_gates_pass','protected_pass','source_delta_pass')),'original entry gates')
    text=(run/'stdout.txt').read_text(encoding='utf-8-sig');err=(run/'stderr.txt').read_text(encoding='utf-8-sig')
    pids=re.findall(r'^owned_probe_pid=(\d+)\s*$',text,re.M);require(len(pids)==1,'one actual announced process');monitoring(run,int(pids[0]))
    apis=[x.split(',') for x in text.splitlines() if x.startswith('api,')]
    require(apis and all(len(x)==3 and x[2]=='0' for x in apis),'actual QNN return failure')
    names=dict(backend_create='backendCreate',backend_free='backendFree',device_create='deviceCreate',device_free='deviceFree',context_create='contextCreate',context_free='contextFree',graphs='graphCreate',executes='graphExecute')
    counts={k:sum(x[1]==n for x in apis) for k,n in names.items()};require(counts==COUNTS,'resource evidence')
    require(text.count('lifetime,graphFinalize_entry\n')==text.count('lifetime,graphFinalize_return,0\n')==20,'all actual finalize returns')
    positions=[]
    for role in ROLES:
        start='cohort_begin,mode='+role+'\n';end='cohort_end,mode='+role+',exit=0\n'
        require(text.count(start)==text.count(end)==1,'complete role boundary')
        a=text.index(start);b=text.index(end);require(a<b,'role ordering');positions.extend([a,b]);segment=text[a:b]
        require(segment.count('api,contextCreate,0\n')==segment.count('api,contextFree,0\n')==1,'independent role context')
        require(all('api,'+name+',' not in segment for name in ('backendCreate','backendFree','deviceCreate','deviceFree')),'backend/device retained between roles')
    require(positions==sorted(positions),'five-role sequence')
    require(text.count('shared_process,mode=shared-five-role,exit=0')==1,'whole-process teardown terminal')
    require(err.count('PrepareLibLoader Loading libQnnHtpPrepare.so')==err.count('PrepareLibLoader Unloading libQnnHtpPrepare.so')==1 and 'QnnBackend_free started' in err and err.index('PrepareLibLoader Unloading')>err.index('QnnBackend_free started'),'actual Prepare lifetime')
    for name in ('logcat-before','logcat-after','dropbox-before','dropbox-after','tombstones-before','tombstones-after'):
        require(read(run/(name+'-private.json'))['exit']==0,'read-only crash evidence unavailable')
    require(err and read(run/'logcat-after-private.json')['text_private'],'runtime log capture missing')
    # Replay the unchanged frozen oracle in an explicitly Host-only namespace.
    view=PRIVATE/'pilot-host-oracle-view';view.mkdir(exist_ok=True)
    metadata=dict(kind='HOST_ONLY_ORACLE_VIEW_NOT_A_DEVICE_EXECUTION',mode='HOST_REPLAY_ONLY',role_raw=receipt['role_raw'],recovery_receipt_sha256=sha(run/'recovery-receipt-private.json'),new_os_processes=0,formal_eligible=False)
    write_once(view/'identity-private.json',metadata)
    for name,record in receipt['role_raw'].items():
        require(sha(run/name)==record['sha256'] and sha(run/(name+'.gz'))==record['gzip_sha256'],'role/gzip changed')
        if not (view/name).exists():os.link(run/name,view/name)
        require(sha(view/name)==record['sha256'],'Host oracle view differs')
    if not (view/'stdout.txt').exists():shutil.copyfile(run/'stdout.txt',view/'stdout.txt')
    require(sha(view/'stdout.txt')==sha(run/'stdout.txt'),'Host replay log differs')
    roles={role:assess_role(view,role,m,completion['exit_code']) for role in ROLES}
    runtime=runtime_error_text(text,err)
    result=dict(status='PASS',stage='pilot',build=m,actual_exit=0,safety='PASS',teardown='PASS',counts=counts,prepare_loads=1,prepare_unloads=1,inter_role_prepare_reload=0,
        api6031=len(re.findall(r'(?<!\d)6031(?!\d)',runtime)),token6033=len(re.findall(r'(?<!\d)6033(?!\d)',runtime)),sigsegv=False,crash_capture='AVAILABLE',
        roles=roles,role_contract_digests={role:role_digest(run/('cohort-'+role+'.bin'),role,r) for role,r in roles.items()},
        classification='SHARED_LIFETIME_FIVE_ROLE_PILOT_VERIFIED',completion_transport='TIMEOUT_AFTER_EXIT0_HOST_LOSSLESS_RECOVERY',formal_completed=0,formal_entry_authorized=False,
        recovery_receipt_sha256=sha(run/'recovery-receipt-private.json'),recovered_container_sha256=receipt['recovered_container_sha256'],original_identity_fabricated=False,
        analysis_source_sha256=sha(Path(__file__)),oracle_source_sha256=sha(ROOT/'scripts/assess_backward_shared_lifetime_formal.py'),
        logcat_stream_bytes=(run/'logcat-stream-private.txt').stat().st_size,logcat_stream_owned_records=sum(re.search(r'\b'+pids[0]+r'\b',x) is not None for x in (run/'logcat-stream-private.txt').read_text(encoding='utf-8-sig').splitlines()),
        timer_6031_false_matches=len(re.findall(r'(?m)^api_us,[^\n]*?(?<!\d)6031(?!\d)',text)),timer_6033_false_matches=len(re.findall(r'(?m)^api_us,[^\n]*?(?<!\d)6033(?!\d)',text)))
    validate_success(result,'pilot',m)
    write_once(run/'recovered-pilot-assessment-private.json',result)
    return result


def selftest():
    r=dict(kind='HOST_ONLY_LOSSLESS_RECOVERY_AFTER_POST_EXIT_TRANSPORT_TIMEOUT',stage='pilot',actual_exit=0,new_device_arithmetic=0,formal_eligible=False,mission_stop_retained=True,original_raw_overwritten=False,original_identity_fabricated=False,recovered_container_sha256='a'*64,original_remote_container_sha256='a'*64,role_raw={'cohort-'+role+'.bin':dict(remote_match=True,decompressed_match=True) for role in ROLES})
    validate_receipt(r,dict(exit_code=0));n=0
    for key,value in [('actual_exit',None),('formal_eligible',True),('new_device_arithmetic',1),('mission_stop_retained',False),('original_raw_overwritten',True),('original_identity_fabricated',True),('recovered_container_sha256','b'*64),('role_raw',{}),('stage','formal-1')]:
        bad=copy.deepcopy(r);bad[key]=value
        try:validate_receipt(bad,dict(exit_code=0))
        except ValueError:n+=1
        else:raise AssertionError('invalid recovered evidence accepted')
    print('recovered_pilot_selftest=PASS rejection_cases='+str(n)+' device_calls=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selftest()
    else:
        r=assess();print(json.dumps(dict(classification=r['classification'],formal_completed='0/3',further_device_arithmetic='STOPPED',legacy_digests={k:v['numerical_digest'] for k,v in r['roles'].items()},total_maxAbs={k:{metric:data['maxAbs'] for metric,data in v['metrics'].items() if metric.startswith('total_')} for k,v in r['roles'].items()})))
