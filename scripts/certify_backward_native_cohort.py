"""Independent full cohort audit with additional role/output/contract binding."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from assess_backward_repair import ROOT, assess_cohort, read, require, words
from prepare_saved_p_backward import sha

ROLES = ['standalone_softmax','standalone_dvalue','native_softmax','native_dvalue','dual']
REPORT = ROOT/'build/reports/g1-backward-repair-native-completion'


def role_digest(path, role, result):
    digest=hashlib.sha256(b'G1_NATIVE_COMPLETION_ROLE_V1\0'+role.encode('ascii')+b'\0')
    with path.open('rb') as raw:
        require(tuple(words(raw,4))[:2] == (0x42525001,1),'role header')
        while raw.peek(1):
            key=tuple(map(int,words(raw,7)));output=None
            for _ in range(key[6]):
                code,head,count=map(int,words(raw,3));value=words(raw,count)
                if code==3:output=value
            require(output is not None,'actual output missing')
            digest.update(struct.pack('<7I',*key[:6],len(output)));digest.update(output.astype('<u4').tobytes())
    contracts={k:result[k] for k in ('mode','coverage','api_success','execution_invariants','head_mapping','receipt_words_verified','forward_joint_pass','required_failed_pairs','observer_mismatch_words','arithmetic_word_mismatches')}
    digest.update(b'\0contracts\0'+json.dumps(contracts,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii'))
    return digest.hexdigest()


def certify(directory, previous=()):
    require(directory.resolve().parent == REPORT.resolve() and re.fullmatch('cohort-(?:1(?:-awake)?|[23])',directory.name),'actual new cohort directory required')
    identity=read(directory/'identity-private.json');completion=read(directory/'execution-before-readback-private.json')
    require(identity['mode']=='cohort' and identity['exit_code']==completion['exit_code']==0,'actual cohort completion')
    require(read(directory/'preflight/device-preflight-private.json')['status']=='PASS','fresh preflight')
    text=(directory/'stdout.txt').read_text(encoding='utf-8-sig');pids=re.findall(r'^owned_probe_pid=(\d+)\s*$',text,re.M)
    require(len(pids)==1,'owned process identity')
    full=assess_cohort(directory)
    require(full['classification']=='PASS' and list(full['roles'])==ROLES,'five required roles')
    for mode,result in full['roles'].items():
        require(result['classification']=='PASS' and result['pairs']==(19648 if mode=='dual' else 9824),'full role coverage')
    result=dict(status='PASS',independent_os_processes=1,source_commit=identity['source_sha'],
        assessor_source_hashes={p:sha(ROOT/p) for p in ('scripts/certify_backward_native_cohort.py','scripts/assess_backward_repair.py','scripts/analyze_backward_precision.py')},
        orchestration_sha256=identity['orchestration_sha256'],build=identity['build'],actual_exit=identity['exit_code'],
        owned_pid_private=int(pids[0]),raw_sha256=identity['raw_evidence_sha256'],roles=full['roles'],
        numerical_digests={m:r['numerical_digest'] for m,r in full['roles'].items()},
        role_contract_digests={m:role_digest(directory/'decoded-cohort'/(m+'.bin'),m,r) for m,r in full['roles'].items()})
    for prior in previous:
        old=read(prior/'cohort-assessment-private.json')
        require(old['status']=='PASS' and old['independent_os_processes']==1 and old['actual_exit']==0,'previous actual process')
        require(old['owned_pid_private']!=result['owned_pid_private'] and prior.resolve()!=directory.resolve(),'process reused')
        require(old['raw_sha256']==sha(prior/'saved-p-words.bin'),'previous raw changed')
        for key in ('source_commit','orchestration_sha256','build','numerical_digests','role_contract_digests'):
            require(old[key]==result[key],'process reproducibility '+key)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=Path,required=True);p.add_argument('--previous',type=Path,action='append',default=[]);a=p.parse_args()
    result=certify(a.report_root,a.previous)
    destination=a.report_root/'cohort-assessment-private.json'
    if destination.exists():require(read(destination)==result,'existing cohort assessment differs')
    else:
        with destination.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],roles=list(result['roles']),independent_os_processes=1,numerical_digests=result['numerical_digests'],role_contract_digests=result['role_contract_digests'])))
