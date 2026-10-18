"""Independent received-input controls; ranks/lowerings kept separate."""
import argparse
import json
import pathlib
import struct
import numpy as np
from analyze_backward_precision import half,bf,ordered,pairwise,metric,multiply_models
from assess_backward_repair import require,read,words,exact,fp32
from prepare_saved_p_backward import sha,ROOT


def assess(directory):
    ident=read(directory/'identity-private.json')
    require(ident['exit_code']==0 and ident['focus_takeover_count']==0,'process/safety')
    require(sha(directory/'saved-p-words.bin')==ident['raw_evidence_sha256'],'raw SHA')
    expected={(r,s,c,h) for r in (2,4) for s in range(6) for c in range(24) for h in range(2)}
    rows={}
    with (directory/'saved-p-words.bin').open('rb') as f:
        require(tuple(words(f,4))==(0x42504301,1,2,6),'primitive header')
        while f.peek(1):
            r,s,c,h,n=map(int,words(f,5));key=(r,s,c,h)
            require(key in expected and key not in rows and n==(2 if s==1 else 3),'primitive identity')
            fields={}
            for _ in range(n):
                code,count=map(int,words(f,2));require(code in (1,2,3) and code not in fields,'field identity')
                require(count==(32 if (code==3 and s in (1,2)) or (code==2 and s==3) else 1024),'field shape')
                fields[code]=words(f,count).view('<f4')
            require(all(np.isfinite(fields[i]).all() for i in fields if i!=3),'nonfinite control input')
            rows[key]=fields
    require(set(rows)==expected,'primitive incomplete coverage')
    text=(directory/'stdout.txt').read_text(encoding='utf-8-sig')
    apis=[l.split(',')[1:] for l in text.splitlines() if l.startswith('api,')]
    require(all(a[1]=='0' for a in apis),'primitive API')
    names=[a[0] for a in apis]
    require(names.count('graphExecute')==576 and names.count('graphFinalize')==12 and 'contextFree' in names and 'backendFree' in names,'primitive execute/finalize/teardown')
    terminal=[dict(x.split('=',1) for x in line.split(',')[2:]) for line in text.splitlines() if line.startswith('summary,primitive_controls,')]
    require(len(terminal)==1 and terminal[0]['cases']=='576','primitive terminal')
    nonfinite=sum(int(np.count_nonzero(~np.isfinite(v[3]))) for v in rows.values())
    require(int(terminal[0]['nonfinite_words'])==nonfinite and int(terminal[0]['finite'])==int(nonfinite==0),'nonfinite FAIL masking')
    for name,value in dict(input_mutation='0',poison_remaining='0',canary='1',teardown='1',cpu_arithmetic_fallback='0').items():require(terminal[0][name]==value,'primitive invariant')
    result=dict(classification='NUMERICAL_FAIL_CONTROLS_COMPLETE' if nonfinite else 'RECEIVED_INPUT_CONTROLS_COMPLETE',nonfinite_output_words=nonfinite,controls=576,ranks=[2,4],stages={},structural_dvalue={})
    labels=['product','reduce_correct_product','product_reduce_native','subtract_known_dot','final_multiply','dvalue_TN32']
    for r in (2,4):
        for s in range(6):
            selected=[rows[(r,s,c,h)] for c in range(24) for h in range(2)]
            a=np.stack([v[1] for v in selected]).reshape(-1,32,32)
            y=np.stack([v[3] for v in selected]);b=None if s==1 else np.stack([v[2] for v in selected])
            if s in (0,4):
                b=b.reshape(-1,32,32);models={n:v.reshape(-1,1024) for n,v in multiply_models(a,b).items()};oracle=a.astype(np.float64)*b.astype(np.float64)
            elif s==3:
                b=b.reshape(-1,32,1);models=dict(separate_fp32=(a-b).reshape(-1,1024),fp16_operands_fp16_result=half(half(a)-half(b)).reshape(-1,1024),fp16_operands_fp32_result=(half(a)-half(b)).reshape(-1,1024),bf16_operands_bf16_result=bf(bf(a)-bf(b)).reshape(-1,1024));oracle=a.astype(np.float64)-b.astype(np.float64)
            elif s in (1,2):
                pr=a if s==1 else np.float32(a*b.reshape(-1,32,32))
                models=dict(ordered_fp32=ordered(pr),reverse_fp32=ordered(pr,reverse=True),pairwise_fp32=pairwise(pr),fp16_accumulation=ordered(half(pr),'half'),fp16_inputs_fp32_sum=ordered(half(pr)),bf16_accumulation=ordered(bf(pr),'bf'),double_final_fp32=pr.astype(np.float64).sum(axis=-1).astype(np.float32),fp16_output_double_sum=half(pr.astype(np.float64).sum(axis=-1)),fp16_output_pairwise=half(pairwise(pr)))
                oracle=pr.astype(np.float64).sum(axis=-1)
            else:
                b=b.reshape(-1,32,32);oracle=np.stack([exact(x,z,True) for x,z in zip(a,b)])
                models=dict(ordered_fp32=np.stack([fp32(x,z,True) for x,z in zip(a,b)]).reshape(-1,1024))
                for name,q in [('fp16',half),('bf16',bf)]:
                    terms=q(a).transpose(0,2,1)[:,:,None,:]*q(b).transpose(0,2,1)[:,None,:,:]
                    models[name+'_input_fp32_sum']=ordered(terms).reshape(-1,1024)
                    models[name+'_product_fp32_sum']=ordered(q(terms)).reshape(-1,1024)
                    models[name+'_fp32_sum_half_output']=half(ordered(terms)).reshape(-1,1024)
            errors=np.abs(y.astype(np.float64)-oracle.reshape(y.shape));finite=np.isfinite(errors);idx=np.unravel_index(np.argmax(errors),errors.shape)
            bad=int(np.count_nonzero(~np.isfinite(y)))
            result['stages'][f'rank{r}_{labels[s]}']=dict(cases=48,status='NONFINITE_NUMERICAL_FAIL' if bad else 'FINITE',nonfinite_output_words=bad,
                received_double_maxAbs=None if bad else float(errors.max()),finite_component_maxAbs=float(errors[finite].max()),rms=None if bad else float(np.sqrt(np.mean(errors**2))),worst=dict(case=int(idx[0]//2),head=int(idx[0]%2),coordinate=int(idx[1])),models={name:metric(y,m) for name,m in models.items()})
            if s==5:
                # Structural layout proof is restricted to onehot/identity/
                # permutation/head-asymmetric inputs with exact selection sums.
                for c in range(16,20):
                    for h in range(2):
                        v=rows[(r,s,c,h)];expected_y=exact(v[1].reshape(32,32),v[2].reshape(32,32),True).astype(np.float32)
                        result['structural_dvalue'][f'rank{r}_control{c}_head{h}']=metric(v[3][None,:],expected_y.reshape(1,-1))
    result['layout_controls_pass']=all(v['bitwise_mismatches']==0 for v in result['structural_dvalue'].values())
    result['internal_precision']='UNRESOLVED: standard kernel implementation/order not uniquely identified by output-model agreement'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();result=assess(a.report_root)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('classification','controls','layout_controls_pass','internal_precision')}))
