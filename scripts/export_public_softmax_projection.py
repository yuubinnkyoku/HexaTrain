"""Strict aggregate exporter. Never serialize private ledger/paths or raw words."""
from pathlib import Path
import argparse, collections, hashlib, json, math, re

LANES = ('oneSoftmax', 'oneProjection', 'oneProducer')
FIXTURES = {
    'one-layer-inputs.bin': '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f',
    'ln1-parameters.bin': '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f',
    'ln1-producers.bin': 'ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580',
}
BINARIES = ('libHexaFp32Dsp.so', 'libHexaFp32Host.so', 'one_layer_probe')

def require(value, reason):
    if not value:
        raise ValueError(reason)

def digest(lines):
    return hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()

def required_metric(mode, tensor):
    if mode.startswith('softStage'):
        return not mode.endswith('M0')
    if mode.startswith('productChain'):
        return not any(x in tensor for x in ('dot', 'centered', '_ds'))
    if mode.startswith('softSegment'):
        return not any(x in tensor for x in ('qraw', 'kraw'))
    if mode.startswith('projection'):
        return mode != 'projection2' and not tensor.startswith('next_')
    return mode.startswith('producer') and mode.endswith('Local')

def process(ledger, stdout, lane, expected_source):
    require(ledger['variant'] == lane and ledger['exit_code'] == 0, 'lane/exit')
    require(ledger['source_sha'] == expected_source, 'source')
    require(ledger['build']['source_sha'] == expected_source, 'build source')
    require(ledger['mode'] == 'BACKGROUND_CORRECTNESS', 'mode')
    require(ledger['focus_takeover_count'] == 0, 'focus')
    require(ledger['protected_identity'] == 'PASS' and ledger['status'] == 'COMPLETE', 'protected/complete')
    require(ledger['build']['qairt_build_id'] == '2.48.40.260702151143', 'sdk')
    require(ledger['build']['architecture'] == 'v81' and ledger['build']['fp_contract'] == 'off', 'arithmetic build')
    hashes = ledger['hashes']
    for k in BINARIES:
        require(hashes[k] == ledger['build']['binary_hashes'][k], 'binary')
        require(re.fullmatch('[0-9a-f]{64}', hashes[k]), 'hash format')
    for k in ('one-layer-inputs.bin',) + (() if lane == 'oneSoftmax' else ('ln1-parameters.bin',)) + (('ln1-producers.bin',) if lane == 'oneProducer' else ()):
        require(hashes[k] == FIXTURES[k], 'fixture')
    lines = stdout.splitlines()
    summary = [x for x in lines if x.startswith('summary,')]
    require(len(summary) == 1 and summary[0] == f'summary,{lane},failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0', 'summary')
    apis = [x for x in lines if x.startswith('api,')]
    require(apis and all(x.endswith(',0') for x in apis), 'api')
    require(all('api,'+k+',0' in lines for k in ('contextFree', 'deviceFree', 'backendFree')), 'teardown')
    metrics = []; groups = {}; required_count = 0
    for line in lines:
        if not line.startswith('onemetric,'):
            continue
        p = line.split(','); v = dict(x.split('=', 1) for x in p[4:])
        for k in ('abs', 'rms', 'relative_l2', 'max_relative', 'fp_abs', 'cosine', 'limit'):
            require(math.isfinite(float(v[k])), 'finite metric')
        require(int(v['nonfinite']) == 0, 'nonfinite')
        require(int(v['gate']) == int(float(v['abs']) <= float(v['limit'])), 'gate consistency')
        if required_metric(p[1], p[3]):
            require(v['gate'] == '1', 'required accuracy')
            required_count += 1
        g = groups.setdefault(p[1]+'/'+p[3], dict(cases=0, failures=0, max_abs=0., max_fp_abs=0., max_fp_word_mismatch=0, limit=float(v['limit'])))
        require(g['limit'] == float(v['limit']), 'gate changed')
        g['cases'] += 1; g['failures'] += v['gate'] == '0'
        g['max_abs'] = max(g['max_abs'], float(v['abs']))
        g['max_fp_abs'] = max(g['max_fp_abs'], float(v['fp_abs']))
        g['max_fp_word_mismatch'] = max(g['max_fp_word_mismatch'], int(v['fp_word_mismatch']))
        metrics.append(line)
    require(required_count > 0 and len(metrics) > 50, 'missing metrics')
    for prefix in {'oneSoftmax': ('softStage', 'softSegment'), 'oneProjection': ('projection0', 'projection5'), 'oneProducer': ('producer0', 'observerCut')}[lane]:
        require(any(k.startswith(prefix) for k in groups), 'missing variant')
    markers = ('segmentreceipt,', 'product_receipt,', 'softmaxreceipt,', 'oneownership,', 'dprob_receipt,')
    require(any(x.startswith(markers) for x in lines), 'missing receipt audit')
    numerical = [x for x in lines if x.startswith(('onemetric,', 'cutDifference,', 'productObserver,'))]
    return dict(execute_count=sum(x == 'api,graphExecute,0' for x in lines), metric_count=len(metrics), required_metric_count=required_count, numerical_sha256=digest(numerical), groups=groups)

def export(cohort_path, expected_source):
    require(re.fullmatch('[0-9a-f]{40}', expected_source), 'source format')
    cohort = json.loads(cohort_path.read_text())
    result = dict(schema='g1-softmax-ln1-projection-v1', source_commit=expected_source, qairt_build_id='2.48.40.260702151143', fixtures=FIXTURES, lanes={})
    anchor = None; paths = set(); starts = set()
    for lane in LANES:
        require(len(cohort[lane]) == 3, 'three processes required')
        records = []
        for folder in cohort[lane]:
            p = Path(folder).resolve(); require(p not in paths, 'duplicate process path'); paths.add(p)
            j = json.loads((p/'identity-private.json').read_text(encoding='utf-8-sig'))
            require(j['started_utc'] not in starts, 'duplicate process'); starts.add(j['started_utc'])
            hashes = {k: v for k, v in j['hashes'].items() if k in BINARIES or k.startswith('libQnn')}
            compiler = {k: j['build'][k] for k in ('hexagon_compiler_sha256', 'ndk_compiler_sha256')}
            identity = dict(binary_hashes=hashes, compiler_hashes=compiler)
            if anchor is None:
                anchor = identity
            require(identity == anchor, 'cohort binary/compiler mismatch')
            records.append(process(j, (p/'stdout.txt').read_text(), lane, expected_source))
        require(records[0] == records[1] == records[2], 'numerical cohort mismatch')
        result['lanes'][lane] = dict(processes=[{k: v for k, v in p.items() if k != 'groups'} for p in records], metrics=records[0]['groups'])
    result['provenance'] = anchor
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--cohort', type=Path, required=True); p.add_argument('--source', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); data = export(a.cohort, a.source)
    a.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('allowlist_export=PASS processes=9 binary_identity=1 numerical_reproducibility=1 required_accuracy=1')
