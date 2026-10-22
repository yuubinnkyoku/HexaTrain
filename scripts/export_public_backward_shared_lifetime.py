"""Closed public projection for this independent, one-process diagnostic only."""
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from assess_backward_repair import ROOT, read, require
from prepare_saved_p_backward import sha
from assess_backward_shared_lifetime import assess_role

PRIVATE=ROOT/'build/reports/g1-backward-shared-lifetime-diagnostic'
PLAN=ROOT/'host_tests/fp32_custom/backward_shared_lifetime_diagnostic_plan.json'
BUNDLE=ROOT/'build/fp32-saved-p-backward-shared-lifetime-v1'
ANCHOR='fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267'
ORIGINAL='23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05'
COUNTS=dict(backend_create=1,backend_free=1,device_create=1,device_free=1,context_create=2,context_free=2)

def source_delta():
    old=(ROOT/'host_tests/fp32_custom/backward_repair_probe.cpp').read_text()
    new=(ROOT/'host_tests/fp32_custom/backward_shared_lifetime_probe.cpp').read_text()
    # Verbatim worker, oracle, record layout, thresholds, observers, graph use.
    old=old[:old.index('int main(int argc,char** argv){')]
    new=new[new.index('#include "backward_repair_graph.h"'):new.index('int main(int argc,char** argv){')]
    a='void initialize(){initializeDoubleQk();check("registerRepairHost",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairHost.so","HexaBackwardRepairInterfaceProvider","CPU"));check("registerRepairDSP",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairDsp.so","HexaBackwardRepairInterfaceProvider","HTP"));}'
    require(old.count(a)==1,'frozen initialization anchor')
    require(old.replace(a,'void initialize(){sharedlife::owner->beginRole();}').strip()==new.strip(),'non-lifetime worker delta')

def integrity():
    source_delta()
    a=read(PRIVATE/'starting-audit-private.json')
    require(a['status']=='PASS' and a['clean_tree'] and a['head']==read(PLAN)['base_commit'],'starting provenance')
    for field in ('prior_333','protected','existing_evidence'):
        for p,h in a[field].items():require(sha(ROOT/p)==h,'protected/old evidence mismatch')
    require(len(a['prior_333'])==333,'old ledger coverage')
    refs=dict(line.split()[::-1] for line in subprocess.check_output(['git','show-ref'],cwd=ROOT,text=True).splitlines())
    for line in a['refs'].splitlines():
        h,ref=line.split();require(refs.get(ref)==h,'protected ref changed')
    for name,entry in a['bundles'].items():
        p=ROOT/'build'/name/'build-identity.json';require(sha(p)==entry['manifest_sha256'],'old manifest changed')
        m=read(p)
        for f,h in m['source_hashes'].items():require(sha(ROOT/f)==h,'old source changed')
        for f,h in m['binary_hashes'].items():require(sha(p.parent/f)==h,'old binary changed')
    m=read(BUNDLE/'build-identity.json')
    require(m['candidate']=='shared-lifetime-boundary-v1' and m['classification']=='DIAGNOSTIC_ONLY_NOT_FORMAL','candidate impersonation')
    require(m['probe_sha256']!=ORIGINAL and m['source_sha']!='bb6aaceb42de505b29689a362ce8293687e7ab89','candidate independent identity')
    for f,h in m['source_hashes'].items():require(sha(ROOT/f)==h,'diagnostic source changed')
    for f,h in m['binary_hashes'].items():require(sha(BUNDLE/f)==h,'diagnostic binary changed')
    for n,k in [('libHexaBackwardRepairDsp.so','dsp'),('libHexaBackwardRepairHost.so','host')]:require(m['binary_hashes'][n]==read(PLAN)['original_binary_sha256'][k],'DSP/Host not bit-identical')
    require(m['host_gates']=='PASS','build host gates')
    return m

def skeleton(m):
    return dict(schema=1,base_commit=read(PLAN)['base_commit'],candidate='shared-lifetime-boundary-v1',
      source_commit=m['source_sha'],probe_sha256=m['probe_sha256'],dsp_sha256=m['binary_hashes']['libHexaBackwardRepairDsp.so'],
      host_sha256=m['binary_hashes']['libHexaBackwardRepairHost.so'],classification='INCONCLUSIVE',
      diagnostic_processes=0,retries=0,formal_processes_added=0,formal_completed='0/3',formal_status='BLOCKED_FORMAL_COHORT',
      training='NO-GO',performance='NOT_MEASURED',full_attention='NOT_CERTIFIED',full1_layer='NOT_CERTIFIED',full19_layer='NOT_CERTIFIED',
      root_cause='NOT_PROVEN',hypothesis='SUPPORTED_BY_ORIGINAL_FAILURES_INTERVENTION_UNMEASURED',protected_state='PASS',source_delta='LIFETIME_ONLY',
      host_lifecycle='PASS',preflight=None,resource_counts=None,inter_role_prepare_reload=None,
      softmax=None,dvalue=None,dvalue_first_finalize_return=None,actual_exit=None,process_teardown=None,
      sigsegv=None,api6031_occurrences=None,token6033_occurrences=None,crash_collection=None,
      faulting_library=None,faulting_library_offset=None,matching_main_thread_frames=None,safety=None,recognizable_competition=None)

def validate(v):
    # Exact top-level and nested schemas; no private/runtime fields can pass.
    frozen=read(PLAN)['original_binary_sha256']
    fake=dict(source_sha='a'*40,probe_sha256='b'*64,binary_hashes={'libHexaBackwardRepairDsp.so':frozen['dsp'],'libHexaBackwardRepairHost.so':frozen['host']})
    require(set(v)==set(skeleton(fake)),'unknown/missing public field')
    require(v['candidate']=='shared-lifetime-boundary-v1' and v['probe_sha256']!=ORIGINAL,'candidate impersonation')
    require(v['dsp_sha256']==frozen['dsp'] and v['host_sha256']==frozen['host'],'old binary overwrite / package replacement')
    require(v['diagnostic_processes'] in (0,1) and v['retries']==0,'unregistered second process')
    require(v['formal_processes_added']==0 and v['formal_completed']=='0/3' and v['formal_status']=='BLOCKED_FORMAL_COHORT','formal promotion')
    require(v['training']=='NO-GO' and v['root_cause']=='NOT_PROVEN' and v['protected_state']=='PASS' and v['source_delta']=='LIFETIME_ONLY','protected/scientific promotion')
    require(v['performance']=='NOT_MEASURED' and all(v[k]=='NOT_CERTIFIED' for k in ('full_attention','full1_layer','full19_layer')),'scope promotion')
    for key in ('source_commit','probe_sha256','dsp_sha256','host_sha256'):
        value=v[key];require(isinstance(value,str) and len(value)==(40 if key=='source_commit' else 64) and all(c in '0123456789abcdef' for c in value),'safe hash identity')
    if v['resource_counts'] is not None:require(set(v['resource_counts'])==set(COUNTS),'resource fields')
    for role in ('softmax','dvalue'):
        if v[role] is not None:
            require(set(v[role])=={'pairs','classification','maxAbs','numerical_digest','arithmetic_word_mismatches','observer_mismatch_words'},'numerical private/unknown field')
            digest=v[role]['numerical_digest'];require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'private numerical digest')
    require(v['faulting_library'] in (None,'libQnnHtpPrepare.so'),'private crash library')
    require(v['faulting_library_offset'] is None or (v['faulting_library_offset'].startswith('0x') and len(v['faulting_library_offset'])<=12 and all(c in '0123456789abcdef' for c in v['faulting_library_offset'][2:])),'absolute/private PC')
    require(v['classification'] in read(PLAN)['outcomes'],'outcome classification')
    if v['diagnostic_processes']==0:
        require(all(v[k] is None for k in ('resource_counts','softmax','dvalue','dvalue_first_finalize_return','actual_exit','process_teardown','sigsegv','api6031_occurrences','token6033_occurrences','crash_collection')),'unexecuted evidence promoted')
    if v['classification']=='SHARED_LIFETIME_BOUNDARY_PILOT_PASS':
        require(v['diagnostic_processes']==1 and v['actual_exit']==0 and v['process_teardown']=='PASS','missing process exit/teardown')
        require(v['resource_counts']==COUNTS and v['inter_role_prepare_reload']==0,'missing resource-count evidence')
        require(v['dvalue_first_finalize_return']==0,'missing finalize return')
        require(v['preflight']=='PASS' and v['safety']=='PASS' and v['recognizable_competition']==0,'safety/competition')
        require(v['sigsegv'] is False and v['api6031_occurrences']==0 and v['token6033_occurrences']==0 and v['crash_collection']=='AVAILABLE','crash evidence absent or failure hidden')
        for key,n,gate in [('softmax',9824,.002),('dvalue',200,.01)]:
            r=v[key];require(r is not None and r['pairs']==n and r['classification']=='PASS' and 0<=r['maxAbs']<=gate and r['arithmetic_word_mismatches']==0 and r['observer_mismatch_words']==0,'arithmetic mismatch hidden')
        require(v['softmax']['numerical_digest']==ANCHOR,'Softmax anchor mismatch')

def project(run):
    m=integrity();v=skeleton(m)
    require(run.resolve().parent==PRIVATE.resolve(),'independent private root required')
    if not (run/'execution-before-readback-private.json').exists():
        pre=read(run/'preflight/device-preflight-private.json')
        require(pre['status']!='PASS' and not (PRIVATE/'probe-attempt-private.json').exists(),'missing actual process exit')
        v.update(classification='BLOCKED_DEVICE_SAFETY',preflight='BLOCKED',safety='BLOCKED')
        validate(v);return v
    identity=read(run/'identity-private.json');completion=read(run/'execution-before-readback-private.json')
    require(identity['exit_code']==completion['exit_code'],'actual completion')
    claim=read(PRIVATE/'probe-attempt-private.json');require(claim['maximum_os_processes']==1 and claim['candidate_sha256']==m['probe_sha256'],'one-shot identity')
    out=(run/'stdout.txt').read_text(encoding='utf-8-sig');err=(run/'stderr.txt').read_text(encoding='utf-8-sig')
    require(out.count('owned_probe_pid=')==1,'unregistered process')
    apis=[line.split(',') for line in out.splitlines() if line.startswith('api,')]
    counts={key:sum(x[1]==name and x[2]=='0' for x in apis) for key,name in [('backend_create','backendCreate'),('backend_free','backendFree'),('device_create','deviceCreate'),('device_free','deviceFree'),('context_create','contextCreate'),('context_free','contextFree')]}
    require('diagnostic_begin,mode=standalone_softmax' in out and 'native_softmax' not in out and 'diagnostic_begin,mode=dual' not in out,'role scope')
    pre=read(run/'preflight/device-preflight-private.json');require(pre['status']=='PASS','preflight')
    import re
    api6031=sum(x[2]=='6031' for x in apis);token6033=len(re.findall(r'(?<!\d)6033(?!\d)',out+'\n'+err))
    v.update(diagnostic_processes=1,resource_counts=counts,actual_exit=identity['exit_code'],preflight='PASS',api6031_occurrences=api6031,token6033_occurrences=token6033,
       sigsegv=identity['exit_code']==139,process_teardown='PASS' if counts==COUNTS and identity['exit_code']==0 else 'INCOMPLETE')
    between=out.split('diagnostic_begin,mode=standalone_dvalue',1)
    if len(between)==2:
        f=re.search(r'lifetime,graphFinalize_return,(\d+)',between[1]);v['dvalue_first_finalize_return']=int(f.group(1)) if f else None
    roles={}
    for mode,key in [('standalone_softmax','softmax'),('standalone_dvalue','dvalue')]:
        if 'diagnostic_end,mode='+mode+',exit=0' in out:
            a=assess_role(run,mode,m,identity['exit_code']);roles[key]=a
            v[key]={k:a[k] for k in ('pairs','classification','numerical_digest','arithmetic_word_mismatches','observer_mismatch_words')}
            v[key]['maxAbs']=a['metrics']['total_'+str(int(key=='dvalue'))]['maxAbs']
    with (PRIVATE/'independent-assessment-private.json').open('x') as f:json.dump(roles,f,indent=2)
    capture=all((run/n).exists() for n in ('logcat-before-private.json','logcat-after-private.json','logcat-stream-private.txt','dropbox-after-private.json','tombstones-after-private.json'))
    capture=capture and all(read(run/n)['exit']==0 for n in ('logcat-before-private.json','logcat-after-private.json'))
    v['crash_collection']='AVAILABLE' if capture else None
    # Parent ownership trace must show no backend/device free between roles.
    if len(between)==2:
        boundary=between[0].split('diagnostic_begin,mode=standalone_softmax',1)[1]
        v['inter_role_prepare_reload']=0 if 'api,backendFree,' not in boundary and 'api,deviceFree,' not in boundary and counts['backend_create']==counts['device_create']==1 else None
    competition=[json.loads(x) for x in (run/'competition-private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    require(competition and all(x['exit']==0 for x in competition),'competition snapshots')
    # A separate reviewed receipt binds the actual snapshots/logs and safety.
    review=read(PRIVATE/'device-review-private.json')
    require(review['identity_sha256']==sha(run/'identity-private.json') and review['competition_sha256']==sha(run/'competition-private.jsonl'),'review binding')
    v['recognizable_competition']=review['recognizable_competition'];v['safety']=review['safety']
    v.update(review.get('crash_projection',{}))
    if identity['exit_code']==0 and all(k in roles and roles[k]['classification']=='PASS' for k in ('softmax','dvalue')):
        v['classification']='SHARED_LIFETIME_BOUNDARY_PILOT_PASS';v['hypothesis']='BACKEND_REINITIALIZATION_HYPOTHESIS_STRENGTHENED'
    elif v['sigsegv'] and v['faulting_library']=='libQnnHtpPrepare.so' and v['faulting_library_offset']=='0x3007bac' and v['matching_main_thread_frames']==17:
        v['classification']='SHARED_LIFETIME_DIAGNOSTIC_FAILED';v['hypothesis']='BACKEND_REINITIALIZATION_HYPOTHESIS_NOT_SUFFICIENT'
    validate(v);return v

def selftest():
    frozen=read(PLAN)['original_binary_sha256']
    m=dict(source_sha='a'*40,probe_sha256='b'*64,binary_hashes={'libHexaBackwardRepairDsp.so':frozen['dsp'],'libHexaBackwardRepairHost.so':frozen['host']})
    v=skeleton(m);v.update(classification='SHARED_LIFETIME_BOUNDARY_PILOT_PASS',diagnostic_processes=1,actual_exit=0,process_teardown='PASS',resource_counts=COUNTS.copy(),inter_role_prepare_reload=0,dvalue_first_finalize_return=0,preflight='PASS',safety='PASS',recognizable_competition=0,sigsegv=False,api6031_occurrences=0,token6033_occurrences=0,crash_collection='AVAILABLE')
    for key,n in [('softmax',9824),('dvalue',200)]:v[key]=dict(pairs=n,classification='PASS',maxAbs=0.,numerical_digest=ANCHOR,arithmetic_word_mismatches=0,observer_mismatch_words=0)
    validate(v)
    mutations=[('probe_sha256',ORIGINAL),('candidate','ordered-fp32-v1'),('dsp_sha256','e'*64),('host_sha256','e'*64),('formal_processes_added',1),('formal_status','PASS'),('classification','FORMAL_PASS'),('dvalue_first_finalize_return',None),('actual_exit',None),('process_teardown',None),('resource_counts',None),('crash_collection',None),('diagnostic_processes',2),('protected_state','FAIL'),('private_path','secret'),('root_cause','VERIFIED'),('api6031_occurrences',1),('token6033_occurrences',1),('sigsegv',True),('safety','BLOCKED'),('inter_role_prepare_reload',1),('faulting_library','private'),('faulting_library_offset','0x123456789abcdef0')]
    count=0
    for key,value in mutations:
        bad=copy.deepcopy(v);bad[key]=value
        try:validate(bad)
        except (ValueError,AssertionError):count+=1
        else:raise ValueError('exporter accepted '+key)
    for field in ('maxAbs','arithmetic_word_mismatches','pairs'):
        bad=copy.deepcopy(v);bad['dvalue'][field]=99
        try:validate(bad)
        except (ValueError,AssertionError):count+=1
        else:raise ValueError('exporter accepted numerical mismatch')
    print(f'shared_lifetime_exporter_selftest=PASS rejection_cases={count} device_calls=0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    if args.self_test:selftest()
    else:
        value=project(args.run)
        with args.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        print('shared_lifetime_public_projection='+value['classification'])
