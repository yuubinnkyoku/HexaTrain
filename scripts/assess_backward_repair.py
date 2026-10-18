"""Independent binary/coverage/receipt/FP32/double audit, never a device runner."""
import argparse
import hashlib
import json
import pathlib
import struct
import numpy as np
from analyze_backward_precision import ROOT, ordered
from prepare_saved_p_backward import sha


def require(ok,reason):
    if not ok:raise ValueError(reason)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def words(f,n):
    b=f.read(n*4)
    require(len(b)==n*4,'truncated evidence')
    return np.frombuffer(b,dtype='<u4').copy()


def exact(p,g,value):
    p,g=p.astype(np.float64),g.astype(np.float64)
    if value:
        terms=p.T[:,None,:]*g.T[None,:,:]
        # Ordered binary64 to agree with recovered source reference.
        s=np.zeros((32,32),dtype=np.float64)
        for k in range(32):s=s+terms[...,k]
        return s
    dot=np.zeros(32,dtype=np.float64)
    for c in range(32):dot=dot+p[:,c]*g[:,c]
    return p*(g-dot[:,None])


def fp32(p,g,value):
    if value:return ordered(p.T[:,None,:]*g.T[None,:,:])
    pr=np.float32(p*g);dt=ordered(pr);ce=np.float32(g-dt[:,None])
    return np.float32(p*ce)


def fnv(words):
    h=14695981039346656037
    for b in words.astype('<u4').tobytes():h=((h^b)*1099511628211)&((1<<64)-1)
    return f'{h:016x}'


def assess(directory):
    identity=read(directory/'identity-private.json')
    completion=read(directory/'execution-before-readback-private.json')
    require(identity['exit_code']==completion['exit_code'],'missing/mismatched captured completion')
    require(identity['focus_takeover_count']==0 and identity['protected_identity']=='PASS','safety/protected')
    require(sha(directory/'saved-p-words.bin')==identity['raw_evidence_sha256'],'raw SHA')
    require(sha(directory/'saved-p-words.bin.gz')==identity['compressed_evidence_sha256'],'compressed SHA')
    build=identity['build']
    require(build['bundle_leaf'].startswith('fp32-saved-p-backward-') and '/' not in build['bundle_leaf'] and '\\' not in build['bundle_leaf'],'bundle leaf')
    candidate=ROOT/'build'/build['bundle_leaf']
    require(build['candidate']=='ordered-fp32-v1','candidate identity')
    for p,h in build['binary_hashes'].items():require(sha(candidate/p)==h,'binary identity')
    for p,h in build['source_hashes'].items():require(sha(ROOT/p)==h,'tested source changed')
    mode=identity['mode'];modes=['standalone_softmax','standalone_dvalue','forward','native_softmax','native_dvalue','dual']
    require(mode in modes and mode!='forward','unsupported certification mode')
    native=mode.startswith('native') or mode=='dual'
    kinds=[0,1] if mode=='dual' else [int(mode.endswith('dvalue'))]
    limit=identity['forward_limit'];require(limit==60,'full coverage required')
    expected={(o,t,fi,gi,h,v) for o in (1,0) for t in (1,0) for fi in range(60)
              for gi in range(25 if fi<44 else 8) for h in range(2) for v in kinds}
    rows={};apis=[];terminal=[];graphs=[]
    for line in (directory/'stdout.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('{'):
            r=json.loads(line)
            if r.get('kind')=='repair_pair':
                key=tuple(r[n] for n in ('observed','tap','forward','gradient','head','consumer'))
                require(key not in rows and r['mode']==mode,'duplicate/mode')
                rows[key]=r
        elif line.startswith('api,'):
            fields=line.split(',');require(len(fields)==3 and fields[2]=='0','API failure');apis.append(fields[1])
        elif line.startswith('summary,backward_repair,'):terminal.append(dict(x.split('=',1) for x in line.split(',')[2:]))
        elif line.startswith('graph,'):graphs.append(line)
    require(set(rows)==expected and len(terminal)==1,'incomplete numerical coverage')
    require(apis.count('graphFinalize')==4 and apis.count('graphExecute')==4912,'QNN graph coverage')
    require({'registerRepairHost','registerRepairDSP','registerOldDSP','registerS2DSP','registerMixedDSP','registerDoubleDSP','contextFree','deviceFree','backendFree'}<=set(apis),'HTP/teardown missing')
    for field,value in dict(input_mutation='0',finite='1',complete_writes='1',poison_remaining='0',canary='1',teardown='1',cpu_arithmetic_fallback='0').items():require(terminal[0][field]==value,'runtime invariant '+field)
    fixture=(ROOT/'build/fp32-double-qk/double-inputs.bin').read_bytes()
    gradients=(ROOT/'build/reports/g1-native-saved-p-backward-dsp/gradients.bin').read_bytes()
    plan=read(ROOT/'host_tests/fp32_custom/saved_p_backward_device_plan.json')
    require(hashlib.sha256(fixture).hexdigest()==plan['protected_forward_identities']['double_fixture'],'P fixture SHA')
    require(hashlib.sha256(gradients).hexdigest()==plan['gradient_sha256'],'gradient SHA')
    def frozen(fi,h):return np.frombuffer(fixture,dtype='<f4',count=1024,offset=12+fi*122888+65544+h*4096).reshape(32,32)
    def ideal(fi,h,ri):return np.frombuffer(fixture,dtype='<f8',count=1024,offset=12+fi*122888+90120+(ri*2+h)*8192).reshape(32,32)
    def gradient(fi,gi,h,v):return np.frombuffer(gradients,dtype='<f4',count=1024,offset=16+(((gi if fi<44 else 25+gi)*2+h)*2+v)*4096).reshape(32,32)
    result=dict(mode=mode,candidate='ordered-fp32-v1',probe_sha256=build['probe_sha256'],dsp_sha256=build['binary_hashes']['libHexaBackwardRepairDsp.so'],
                source_commit=build['source_sha'],pairs=len(expected),coverage={},metrics={},classification='UNRESOLVED',
                p_observation_limit='NoReceipt/NoTap P words are the observed anchor, not a direct invisible read; final accuracy uses ideal P independently.',
                api_success=True,execution_invariants=True,head_mapping=True,receipt_words_verified=True)
    seen=set();outwords={};pwords={};digest=hashlib.sha256();failures=0;observer_words=0;word_failures=0
    def worst(name,value,key,index=0):
        if name not in result['metrics'] or value>result['metrics'][name]['maxAbs']:result['metrics'][name]=dict(maxAbs=float(value),case=list(key),coordinate=int(index))
    with (directory/'saved-p-words.bin').open('rb') as f:
        require(tuple(words(f,4))==(0x42525001,1,modes.index(mode),60),'raw header')
        while f.peek(1):
            o,t,fi,gi,h,v,n=map(int,words(f,7));key=(o,t,fi,gi,h,v);require(key in expected and key not in seen,'raw identity/coverage');seen.add(key)
            fields={}
            require(n==3+3*int(v==0 and (o or t))+o,'raw field count')
            for _ in range(n):
                code,head,count=map(int,words(f,3));require(head==h and code not in fields,'raw field head/duplicate')
                require(code in range(1,8) and count==(32 if code==5 else (3104 if v else 5184) if code==7 else 1024),'raw field shape')
                fields[code]=words(f,count)
            p,g,y=[fields[i].view('<f4').reshape(32,32) for i in (1,2,3)]
            require(all(np.isfinite(x).all() for x in (p,g,y)),'nonfinite')
            require(np.array_equal(fields[2],gradient(fi,gi,h,v).view(np.uint32).reshape(-1)),'gradient/head/coordinate')
            if not native:require(np.array_equal(fields[1],frozen(fi,h).view(np.uint32).reshape(-1)),'frozen P words')
            require(fnv(fields[3])==rows[key]['output_hash'],'output FNV')
            require(rows[key]['p_visible']==bool(not native or o or t),'P visibility mislabeled')
            pk=(fi,h);anchor=(fi,gi,h,v)
            if o and t:
                if pk in pwords:require(np.array_equal(pwords[pk],fields[1]),'P lifetime/head/dual input mutation')
                pwords[pk]=fields[1];outwords[anchor]=fields[3]
            else:require(np.array_equal(fields[1],pwords[pk]),'P observer mismatch')
            local=exact(p,g,v);contract=fp32(p,g,v)
            mismatches=int(np.count_nonzero(fields[3]!=contract.view(np.uint32).reshape(-1)))
            variants=int(np.count_nonzero(fields[3]!=outwords[anchor]));observer_words+=variants;word_failures+=mismatches
            require(mismatches==rows[key]['word_mismatches'] and variants==rows[key]['observer_mismatches'],'C++ vs independent word gate')
            local_error=np.abs(y.astype(np.float64)-local);total_error=local_error.copy();pinduced=0.
            if native:
                total_error=np.zeros((32,32))
                for ri in (0,1):
                    ip=ideal(fi,h,ri);ideal_y=exact(ip,g,v)
                    total_error=np.maximum(total_error,np.abs(y.astype(np.float64)-ideal_y))
                    pinduced=max(pinduced,float(np.max(np.abs(local-ideal_y))))
                    pd=p.astype(np.float64);delta=pd-ip
                    worst('P_vs_ideal',np.max(np.abs(delta)),key)
                    worst('P_row_sum',np.max(np.abs(pd.sum(axis=-1)-1)),key)
                    if v:
                        bound=np.zeros((32,32))
                        for k in range(32):bound+=np.abs(delta[k,:,None]*g[k,None,:].astype(np.float64))
                    else:
                        dots=np.zeros(32);deltas=np.zeros(32)
                        for c in range(32):dots+=ip[:,c]*g[:,c].astype(np.float64);deltas+=np.abs(delta[:,c]*g[:,c].astype(np.float64))
                        bound=np.abs(delta)*np.abs(g.astype(np.float64)-dots[:,None])+np.abs(pd)*deltas[:,None]
                    worst('p_induced_bound_'+str(v),np.max(bound),key)
            for code,model in [(4,np.float32(p*g)),(5,ordered(np.float32(p*g))),(6,np.float32(g-ordered(np.float32(p*g))[:,None]))] if v==0 else []:
                if code in fields:require(np.array_equal(fields[code],model.view(np.uint32).reshape(-1)),'stage arithmetic words')
            if o:
                rec=fields[7];require(tuple(rec[:4])==(0x48385000+v,2048,1024,32) and rec[5]==81 and rec[6]==v and rec[8]==(t if not v else 0),'receipt identity')
                require(np.array_equal(rec[32:1056],fields[1]) and np.array_equal(rec[1056:2080],fields[2]) and np.array_equal(rec[2080:3104],fields[3]),'received/output words')
                if not v:require(np.array_equal(rec[3104:4128],fields[4]) and np.array_equal(rec[4128:4160],fields[5]) and np.array_equal(rec[4160:],fields[6]),'receipt boundary words')
            max_local=float(local_error.max());max_total=float(total_error.max());gate=.01 if v else .002
            require(abs(max_local-rows[key]['local_maxAbs'])<1e-10 and abs(max_total-rows[key]['total_maxAbs'])<1e-10,'independent double oracle')
            require(abs(pinduced-rows[key]['p_induced_actual_maxAbs'])<1e-10,'P-induced oracle')
            passed=max_total<=gate and not mismatches and not variants
            require(passed==rows[key]['pass'],'false PASS/FAIL')
            failures+=not passed
            label=f'O{o}T{t}_'+('core' if fi<44 else 'holdout')+f'_consumer{v}'
            result['coverage'][label]=result['coverage'].get(label,0)+1
            worst('local_'+str(v),max_local,key,np.argmax(local_error));worst('total_'+str(v),max_total,key,np.argmax(total_error))
            worst(('core_' if fi<44 else 'holdout_')+str(v),max_total,key,np.argmax(total_error))
            worst('p_induced_actual_'+str(v),pinduced,key)
            digest.update(struct.pack('<6I',*key));digest.update(fields[3].astype('<u4').tobytes())
    require(seen==expected and int(terminal[0]['pairs'])==len(expected) and int(terminal[0]['required_failures'])==failures,'complete coverage/terminal')
    p_gates=True
    if native:
        for name,gate in [('P_vs_ideal',1e-6),('P_row_sum',2e-6)]+[('p_induced_bound_'+str(v),.001 if v else .00025) for v in kinds]:p_gates &= result['metrics'][name]['maxAbs']<=gate
    result.update(required_failed_pairs=int(failures),observer_mismatch_words=observer_words,arithmetic_word_mismatches=word_failures,
                  numerical_digest=digest.hexdigest(),forward_joint_pass=bool(p_gates),independent_double_recheck='PASS',
                  classification='PASS' if not failures and p_gates else 'NUMERICAL_OR_WORD_FAIL')
    require(identity['exit_code']==(0 if failures==0 else 4),'process completion inconsistent with measured gates')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    result=assess(a.report_root)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('mode','pairs','classification','required_failed_pairs','numerical_digest')}))
