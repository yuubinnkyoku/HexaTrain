"""Independent NumPy precision hypotheses on actual saved standalone operands.

No hypothesis is an assertion about proprietary backend implementation.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import numpy as np
from prepare_saved_p_backward import ROOT, sha

OUT = ROOT / 'build/reports/g1-backward-primitive-precision-repair'


def read_raw(path):
    with path.open('rb') as f:
        header = struct.unpack('<4I', f.read(16))
        assert header[:2] == (0x53425002, 1)
        while b := f.read(24):
            o,t,fi,gi,h,n = struct.unpack('<6I', b)
            fields = {}
            for _ in range(n):
                field,head,count = struct.unpack('<3I', f.read(12))
                assert head == h and field not in fields
                fields[field] = np.frombuffer(f.read(count*4), dtype='<f4').copy()
            yield (o,t,fi,gi,h), fields


def bf(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def half(x):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return np.asarray(x, dtype=np.float16).astype(np.float32)


def ordered(x, kind='fp32', reverse=False):
    x = np.asarray(x)
    q = half if kind == 'half' else bf if kind == 'bf' else lambda a: np.asarray(a, dtype=np.float32)
    total = np.zeros(x.shape[:-1], dtype=np.float32)
    for j in (range(x.shape[-1]-1,-1,-1) if reverse else range(x.shape[-1])):
        total = q(total + x[...,j])
    return total


def pairwise(x):
    v = np.asarray(x,dtype=np.float32).copy()
    while v.shape[-1] > 1:
        v = (v[...,::2] + v[...,1::2]).astype(np.float32)
    return v[...,0]


def metric(actual, model):
    a,b = np.asarray(actual,dtype=np.float32),np.asarray(model,dtype=np.float32)
    assert a.shape == b.shape
    au,bu = a.view(np.uint32),b.view(np.uint32)
    finite = np.isfinite(a)&np.isfinite(b)
    d = np.abs(a.astype(np.float64)-b.astype(np.float64))
    # Monotone ordered IEEE codes, retaining signed-zero distinction separately.
    ai = np.where(au>>31, np.uint32(0xffffffff)-au, au+np.uint32(0x80000000)).astype(np.int64)
    bi = np.where(bu>>31, np.uint32(0xffffffff)-bu, bu+np.uint32(0x80000000)).astype(np.int64)
    ulp = np.abs(ai-bi)
    bins = [int(np.count_nonzero((ulp>=lo)&(ulp<=hi)&finite)) for lo,hi in [(0,0),(1,1),(2,4),(5,16),(17,256),(257,65536),(65537,2**32)]]
    mismatch = au != bu
    rows = mismatch.reshape((-1,a.shape[-1]))
    return dict(words=int(a.size),word_matches=int(np.count_nonzero(~mismatch)),bitwise_mismatches=int(np.count_nonzero(mismatch)),
        full_case_matches=int(np.count_nonzero(~rows.any(axis=-1))),cases=int(rows.shape[0]),
        maxAbs=float(np.max(d[finite])) if finite.any() else None,
        rms=float(np.sqrt(np.mean(d[finite]**2))) if finite.any() else None,
        ulp_bins_0_1_2to4_5to16_17to256_257to65536_above65536=bins,
        signed_zero_mismatches=int(np.count_nonzero((a==0)&(b==0)&mismatch)),
        actual_subnormal=int(np.count_nonzero((au&0x7f800000)==0))-int(np.count_nonzero(a==0)),
        model_subnormal=int(np.count_nonzero((bu&0x7f800000)==0))-int(np.count_nonzero(b==0)),
        model_overflow=int(np.count_nonzero(np.isinf(b))),
        model_underflow_to_zero=int(np.count_nonzero((b==0)&(a!=0))),
        nonfinite_actual=int(np.count_nonzero(~np.isfinite(a))))


def multiply_models(x,y):
    return dict(separate_fp32=np.float32(x*y),fp16_operands_fp16_result=half(half(x)*half(y)),
        fp16_operands_fp32_result=half(x)*half(y),bf16_operands_bf16_result=bf(bf(x)*bf(y)),
        bf16_operands_fp32_result=bf(x)*bf(y))


def analyze():
    old=ROOT/'build/reports/g1-native-saved-p-backward-dsp'
    sr=old/'pilot-softmax-final/saved-p-words.bin'
    dr=old/'pilot-dvalue/saved-p-words.bin'
    assert sha(sr)=='7dd188e8852e8e68e86bd1d6dee0c9524ecb79ea4cfa94b0f9c446c4765e333f'
    assert sha(dr)=='b77137d09aeee414214a602acb1aeeb3bc2ce03d604b72ecae960124e407cf49'
    all_s=list(read_raw(sr)); all_d=list(read_raw(dr))
    assert len(all_s)==len(all_d)==9824
    report=dict(schema=1,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
                baseline_raw_sha256=dict(softmax=sha(sr),dvalue=sha(dr)),attribution={},host_candidates={},
                inference='Model agreement supports a hypothesis; internal implementation remains unresolved without backend proof.')
    for obs in (1,0):
        selected=[(k,v) for k,v in all_s if k[0]==obs and k[1]==1]
        keys=[k for k,v in selected]
        arrays={i:np.stack([v[i] for k,v in selected]) for i in range(1,7)}
        p,g,final,prod,dot,center=[arrays[i] for i in range(1,7)]
        local={}
        for name,actual,x,y,sub in [('product',prod,p,g,False),('subtract',center,g,np.repeat(dot,32,axis=1),True),('final',final,p,center,False)]:
            models=multiply_models(x,y) if not sub else dict(separate_fp32=x-y,fp16_operands_fp16_result=half(half(x)-half(y)),fp16_operands_fp32_result=half(x)-half(y),bf16_operands_bf16_result=bf(bf(x)-bf(y)),bf16_operands_fp32_result=bf(x)-bf(y))
            exact=x.astype(np.float64)-y.astype(np.float64) if sub else x.astype(np.float64)*y.astype(np.float64)
            err=np.abs(actual.astype(np.float64)-exact)
            pos=np.unravel_index(np.argmax(err),err.shape)
            local[name]=dict(models={n:metric(actual,v) for n,v in models.items()},received_double_maxAbs=float(err[pos]),received_double_rms=float(np.sqrt(np.mean(err**2))),
                worst=dict(case=list(keys[pos[0]]),coordinate=int(pos[1])),
                single_product_fma='No accumulation in a binary multiply; FMA cannot explain local multiplication rounding.',
                large_gradient_words=int(np.count_nonzero(np.abs(g)>32)))
        pr=prod.reshape(-1,32,32)
        reductions=dict(ordered_fp32=ordered(pr),reverse_fp32=ordered(pr,reverse=True),pairwise_fp32=pairwise(pr),
            fp16_accumulate=ordered(half(pr),'half'),fp16_inputs_fp32_accumulate=ordered(half(pr)),
            bf16_accumulate=ordered(bf(pr),'bf'),bf16_inputs_fp32_accumulate=ordered(bf(pr)),
            double_final_fp32=pr.astype(np.float64).sum(axis=-1).astype(np.float32),
            fp16_output_double_sum=half(pr.astype(np.float64).sum(axis=-1)),fp16_output_pairwise_sum=half(pairwise(pr)))
        exact=pr.astype(np.float64).sum(axis=-1)
        local['reduce']=dict(models={n:metric(dot,v) for n,v in reductions.items()},received_double_maxAbs=float(np.max(np.abs(dot-exact))),
                             upstream_product_dot_error=float(np.max(np.abs(exact-(p.astype(np.float64)*g.astype(np.float64)).reshape(-1,32,32).sum(axis=-1)))))
        report['attribution'][f'Observed{obs}Tap1']=local
    # Full frozen set, independent selection matrix. Increasing-index reductions explicit.
    base_s=[(k,v) for k,v in all_s if k[:2]==(1,1)]
    base_d=[(k,v) for k,v in all_d if k[:2]==(1,1)]
    p=np.stack([v[1] for k,v in base_s]).reshape(-1,32,32)
    g=np.stack([v[2] for k,v in base_s]).reshape(-1,32,32)
    dc=np.stack([v[2] for k,v in base_d]).reshape(-1,32,32)
    pd,gd=p.astype(np.float64),g.astype(np.float64)
    exact_dot=(pd*gd).sum(axis=-1)
    exact_ds=pd*(gd-exact_dot[...,None])
    product=np.float32(p*g)
    ordered_ds=np.float32(p*np.float32(g-ordered(product)[...,None]))
    double_ds=exact_ds.astype(np.float32)
    host={}
    for mode,actual,oracle in [('softmax_ordered_fp32',ordered_ds,exact_ds),('softmax_full_double',double_ds,exact_ds)]:
        errors=np.abs(actual.astype(np.float64)-oracle).reshape(-1,1024)
        maxima=errors.max(axis=1)
        core=np.array([k[2]<44 for k,v in base_s])
        host[mode]=dict(core_maxAbs=float(maxima[core].max()),holdout_maxAbs=float(maxima[~core].max()),
                        core_pairs=int(core.sum()),holdout_pairs=int((~core).sum()),gate=.002,
                        accuracy_pass=bool((maxima<=.002).all()))
    # TN32 term array [case,i,j,t]. No BLAS, independent double and rounding models.
    terms=p.transpose(0,2,1)[:,:,None,:]*dc.transpose(0,2,1)[:,None,:,:]
    dexact=(pd.transpose(0,2,1)[:,:,None,:]*dc.astype(np.float64).transpose(0,2,1)[:,None,:,:]).sum(axis=-1)
    dmodels=dict(ordered_fp32=ordered(terms),reverse_fp32=ordered(terms,reverse=True),pairwise_fp32=pairwise(terms))
    for label,q in [('fp16',half),('bf16',bf)]:
        qt=q(p).transpose(0,2,1)[:,:,None,:]*q(dc).transpose(0,2,1)[:,None,:,:]
        dmodels[label+'_operand_fp32_sum']=ordered(qt)
        dmodels[label+'_product_fp32_sum']=ordered(q(qt))
        dmodels[label+'_accumulation']=ordered(q(qt),'half' if label=='fp16' else 'bf')
    actual=np.stack([v[3] for k,v in base_d]).reshape(-1,32,32)
    # Output conversion is a separately preregistered attribution dimension.
    dmodels.update({n+'_half_output':half(v) for n,v in list(dmodels.items())})
    report['attribution']['dvalue_models']={n:metric(actual.reshape(-1,1024),v.reshape(-1,1024)) for n,v in dmodels.items()}
    core=np.array([k[2]<44 for k,v in base_d])
    for name,val in [('dvalue_ordered_fp32',dmodels['ordered_fp32']),('dvalue_full_double',dexact.astype(np.float32))]:
        errors=np.abs(val.astype(np.float64)-dexact).reshape(-1,1024).max(axis=-1)
        host[name]=dict(core_maxAbs=float(errors[core].max()),holdout_maxAbs=float(errors[~core].max()),
            core_pairs=int(core.sum()),holdout_pairs=int((~core).sum()),gate=.01,accuracy_pass=bool((errors<=.01).all()))
    report['host_candidates']=host
    # All observer pairs independently from output words, including signed zero.
    anchor={k[2:]:v[3] for k,v in all_s if k[:2]==(1,1)}
    observer={}
    for o,t in [(1,1),(1,0),(0,1),(0,0)]:
        xs=[v[3] for k,v in all_s if k[:2]==(o,t)]
        ys=[anchor[k[2:]] for k,v in all_s if k[:2]==(o,t)]
        observer[f'O{o}T{t}']=metric(np.stack(xs),np.stack(ys))
    report['observer']=observer
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
    report=analyze()
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(host_candidates=report['host_candidates'],baseline='FAIL_PRESERVED',internal_precision='UNRESOLVED')))


if __name__=='__main__':main()
