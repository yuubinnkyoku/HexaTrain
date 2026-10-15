"""Allowlisted aggregate only; private ledgers/tensors/identifiers never serialize."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
from export_public_softmax_projection import require, digest, FIXTURES, BINARIES

LANES = ('oneClosureA', 'oneClosureB', 'oneClosureC')


def required(mode, tensor):
    if 'Next' in mode:
        return False
    if mode.startswith('closureRaw'):
        return mode != 'closureRaw2'
    if mode.startswith('closureChain'):
        # Last 3 digits = Softmax domain / raw implementation / observer mode.
        return mode[-2] != '0' or 'raw' not in tensor
    if mode.startswith('closureHead'):
        return True
    if mode.startswith('closureFFN'):
        return mode != 'closureFFN02'
    if mode.startswith('projection'):
        return mode != 'projection2' and not tensor.startswith('next_')
    return mode.startswith('producer') and mode.endswith('Local')


def process(j, stdout, lane, source):
    require(j['variant'] == lane and j['exit_code'] == 0, 'lane/exit')
    require(j['source_sha'] == source == j['build']['source_sha'], 'source')
    require(j['mode'] == 'BACKGROUND_CORRECTNESS' and j['focus_takeover_count'] == 0, 'mode/focus')
    require(j['status'] == 'COMPLETE' and j['protected_identity'] == 'PASS', 'complete/protected')
    b = j['build']
    require(b['qairt_build_id'] == '2.48.40.260702151143', 'SDK')
    require(b['architecture'] == 'v81' and b['fp_contract'] == 'off' and b['vectorization'] == 'disabled', 'compiler contract')
    for k in BINARIES:
        require(j['hashes'][k] == b['binary_hashes'][k] and re.fullmatch('[0-9a-f]{64}', j['hashes'][k]), 'binary')
    fixtures = ['one-layer-inputs.bin']
    if lane != 'oneClosureA':
        fixtures.append('ln1-parameters.bin')
    if lane == 'oneClosureC':
        fixtures.append('ln1-producers.bin')
    for k in fixtures:
        require(j['hashes'][k] == FIXTURES[k], 'fixture')
    lines = stdout.splitlines()
    summary = [s for s in lines if s.startswith('summary,')]
    require(summary == [f'summary,{lane},failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0'], 'summary')
    apis = [s for s in lines if s.startswith('api,')]
    require(apis and all(s.endswith(',0') for s in apis), 'API')
    require(all('api,'+k+',0' in lines for k in ('contextFree', 'deviceFree', 'backendFree')), 'teardown')
    require(any(s.startswith('segmentreceipt,marker=4838311') for s in lines), 'DSP receipt audit')
    groups = {}
    metrics = []
    nrequired = 0
    for line in lines:
        if not line.startswith('onemetric,'):
            continue
        p = line.split(',')
        v = dict(s.split('=', 1) for s in p[4:])
        for k in ('abs', 'rms', 'relative_l2', 'max_relative', 'fp_abs', 'cosine', 'limit'):
            require(math.isfinite(float(v[k])), 'finite metric')
        require(int(v['nonfinite']) == 0, 'nonfinite')
        require(int(v['gate']) == int(float(v['abs']) <= float(v['limit'])), 'gate')
        if required(p[1], p[3]):
            require(v['gate'] == '1', 'required accuracy')
            nrequired += 1
        g = groups.setdefault(p[1]+'/'+p[3], dict(cases=0, failures=0, max_abs=0., max_fp_abs=0., max_fp_word_mismatch=0, limit=float(v['limit'])))
        require(g['limit'] == float(v['limit']), 'gate changed')
        g['cases'] += 1
        g['failures'] += v['gate'] == '0'
        g['max_abs'] = max(g['max_abs'], float(v['abs']))
        g['max_fp_abs'] = max(g['max_fp_abs'], float(v['fp_abs']))
        g['max_fp_word_mismatch'] = max(g['max_fp_word_mismatch'], int(v['fp_word_mismatch']))
        metrics.append(line)
    expected = {
        'oneClosureA': ['closureRaw'+str(m) for m in range(3)]+['closureChain'+str(d)+str(n)+str(m) for d in (1, 2) for n in range(3) for m in range(4)],
        'oneClosureB': ['closureHeadStandaloneObserved', 'closureHeadStandaloneFree']+['closureHeads'+str(n)+str(m) for n in range(2) for m in range(4)],
        'oneClosureC': ['closureFFN'+str(s)+str(m) for s in range(3) for m in range(3 if s == 0 else 4)],
    }[lane]
    for mode in expected:
        require(any(k.startswith(mode+'/') for k in groups), 'missing variant '+mode)
    require(nrequired > 100, 'missing required metrics')
    return dict(execute_count=sum(s == 'api,graphExecute,0' for s in lines), metric_count=len(metrics), required_metric_count=nrequired, numerical_sha256=digest(metrics), groups=groups)


def export(cohort, source):
    require(re.fullmatch('[0-9a-f]{40}', source), 'source format')
    c = json.loads(Path(cohort).read_text())
    result = dict(schema='g1-fp32-segment-closure-v1', source_commit=source, qairt_build_id='2.48.40.260702151143', fixtures=FIXTURES, lanes={})
    paths, starts = set(), set()
    identity = None
    for lane in LANES:
        require(len(c[lane]) == 3, 'three processes')
        records = []
        for folder in c[lane]:
            p = Path(folder).resolve()
            require(p not in paths, 'duplicate process path')
            paths.add(p)
            j = json.loads((p/'identity-private.json').read_text(encoding='utf-8-sig'))
            require(j['started_utc'] not in starts, 'duplicate process')
            starts.add(j['started_utc'])
            b = j['build']
            anchor = dict(binary_hashes={k: v for k, v in j['hashes'].items() if k in BINARIES or k.startswith('libQnn')}, compiler_hashes={k: b[k] for k in ('hexagon_compiler_sha256', 'ndk_compiler_sha256')})
            if identity is None:
                identity = anchor
            require(identity == anchor, 'binary/compiler cohort mismatch')
            records.append(process(j, (p/'stdout.txt').read_text(), lane, source))
        require(records[0] == records[1] == records[2], 'numerical cohort mismatch')
        result['lanes'][lane] = dict(processes=[{k: v for k, v in s.items() if k != 'groups'} for s in records], metrics=records[0]['groups'])
    result['provenance'] = identity
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = export(args.cohort, args.source)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('segment_closure_allowlist=PASS independent_processes=9 required_accuracy=1')
