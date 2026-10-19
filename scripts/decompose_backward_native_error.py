"""Signed host analysis of captured DSP outputs; no device arithmetic or tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from assess_backward_repair import ROOT, exact, read, require, words
from prepare_saved_p_backward import sha


def decompose(directory, assessment):
    identity = read(directory / 'identity-private.json')
    result = read(assessment)
    require(result['classification'] == 'PASS' and result['mode'] == identity['mode'], 'complete independent PASS required')
    require(identity['exit_code'] == 0 and sha(directory/'saved-p-words.bin') == identity['raw_evidence_sha256'], 'captured output')
    require(identity['mode'] in ('native_softmax', 'native_dvalue', 'dual'), 'native role')
    fixture = (ROOT / 'build/fp32-double-qk/double-inputs.bin').read_bytes()
    require(__import__('hashlib').sha256(fixture).hexdigest() == 'f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2', 'fixture')
    peaks = {}; counts = {}
    with (directory / 'saved-p-words.bin').open('rb') as raw:
        header = tuple(words(raw, 4)); require(header[:2] == (0x42525001, 1) and header[3] == 60, 'raw header')
        while raw.peek(1):
            o,t,fi,gi,h,v,n = map(int, words(raw, 7)); fields = {}
            for _ in range(n):
                code,head,count = map(int, words(raw, 3)); require(head == h, 'head')
                fields[code] = words(raw, count)
            p,g,y = [fields[k].view('<f4').reshape(32,32).astype(np.float64) for k in (1,2,3)]
            local = exact(p,g,v); counts[str(v)] = counts.get(str(v),0)+1
            for ri in (0,1):
                ideal_p = np.frombuffer(fixture, dtype='<f8', count=1024, offset=12+fi*122888+90120+(ri*2+h)*8192).reshape(32,32)
                ideal_y = exact(ideal_p,g,v)
                local_error = y-local; induced = local-ideal_y; total = y-ideal_y
                for label, error in (('local',local_error),('p_induced_actual',induced),('total',total)):
                    k = label+'_'+str(v); coordinate = int(np.argmax(np.abs(error))); maximum = float(np.abs(error).flat[coordinate])
                    if k not in peaks or maximum > peaks[k]['maxAbs']:
                        at = lambda x: float(x.flat[coordinate])
                        peaks[k] = dict(maxAbs=maximum, case=[o,t,fi,gi,h,v], reference=ri, coordinate=coordinate,
                            p_directly_observed=bool(o or t), actual=at(y), original_double=at(ideal_y), conditioned_double=at(local),
                            signed_total=at(total), signed_consumer_local=at(local_error), signed_p_induced=at(induced),
                            decomposition_residual=at(total-local_error-induced))
    for k, value in peaks.items():
        require(abs(value['maxAbs']-result['metrics'][k]['maxAbs']) < 1e-14, 'independent peak mismatch')
        require(abs(value['decomposition_residual']) < 1e-12, 'signed decomposition')
    require(all(count == 9824 for count in counts.values()), 'full per-consumer coverage')
    return dict(status='PASS', mode=identity['mode'], raw_sha256=identity['raw_evidence_sha256'], assessment_sha256=sha(assessment),
        counts=counts, peaks=peaks, observer_free_P='Anchor only when neither receipt nor tap exists; final DSP output remains actual.',
        interpretation='At each stored coordinate: total = consumer_local + P_induced. Maxima at different coordinates must not be summed.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report-root',type=Path,required=True);parser.add_argument('--assessment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=decompose(args.report_root,args.assessment)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print('signed_native_decomposition=PASS mode='+result['mode'])
