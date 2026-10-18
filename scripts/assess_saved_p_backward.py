"""Independent offline audit of private standalone DSP words. No native claim."""
import argparse
import hashlib
import json
import math
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]


def word_hash(words):
    value = 14695981039346656037
    for word in words:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((word >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return f'{value:016x}'


def floats(words):
    return [struct.unpack('<f', struct.pack('<I', w))[0] for w in words]


def distance(actual, expected):
    if len(actual) != len(expected) or not all(map(math.isfinite, actual + expected)):
        raise ValueError('nonfinite/shape')
    diffs = [abs(a - b) for a, b in zip(actual, expected)]
    index = max(range(len(diffs)), key=diffs.__getitem__)
    return {'maxAbs': diffs[index], 'coordinate': index}


def assess(path, mode, limit, raw_binary=None):
    records, operands, summaries = {}, {}, []
    current = None
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            line = line.strip()
            if line.startswith('{'):
                data = json.loads(line)
                if data.get('kind') == 'backward_pair':
                    key = tuple(data[k] for k in ('observed', 'tap', 'forward', 'gradient', 'head'))
                    if key in records or data['mode'] != mode:
                        raise ValueError('duplicate/wrong numerical identity')
                    records[key] = data
            elif line.startswith('case,'):
                fields = line.split(',')
                if fields[1] != mode or len(fields) != 7:
                    raise ValueError('case identity')
                current = tuple(map(int, fields[2:]))
                if current in operands:
                    raise ValueError('duplicate raw case')
                operands[current] = {}
            elif line.startswith('words,'):
                fields = line.split(',')
                if current is None or int(fields[2]) != current[-1] or fields[1] in operands[current]:
                    raise ValueError('raw identity/head')
                operands[current][fields[1]] = [int(w, 16) for w in fields[3:]]
            elif line.startswith('summary,'):
                summaries.append(line)
    if raw_binary is None and (path.parent / 'saved-p-words.bin').exists():
        raw_binary = path.parent / 'saved-p-words.bin'
    if raw_binary is not None:
        operands = {}
        names = {1: 'P', 2: 'gradient', 3: 'output', 4: 'product', 5: 'dot', 6: 'centered'}
        with raw_binary.open('rb') as f:
            def read_words(n):
                b = f.read(n * 4)
                if len(b) != n * 4:
                    raise ValueError('truncated binary evidence')
                return struct.unpack('<' + str(n) + 'I', b)
            if read_words(4) != (0x53425002, 1, 0 if mode == 'standalone_softmax' else 1, limit):
                raise ValueError('binary evidence header')
            while f.peek(1):
                o, t, fi, gi, h, fields = read_words(6)
                key = (o, t, fi, gi, h)
                if key in operands or fields != (6 if mode == 'standalone_softmax' and t else 3):
                    raise ValueError('binary record identity')
                raw = {}
                for _ in range(fields):
                    field, head, count = read_words(3)
                    if field not in names or head != h or names[field] in raw or count != (32 if field == 5 else 1024):
                        raise ValueError('binary field identity')
                    raw[names[field]] = read_words(count)
                operands[key] = raw
    expected = {(o, t, fi, gi, h) for o in (1, 0) for t in (1, 0)
                for fi in range(limit) for gi in range(25 if fi < 44 else 8) for h in range(2)}
    if set(records) != expected or set(operands) != expected or len(summaries) != 1:
        raise ValueError('incomplete coverage or terminal summary')
    all_worst, local_worst, counts, failed, canonical = {}, {}, {}, 0, []
    accuracy_failures = observer_pairs = observer_words = 0
    plan = json.loads((ROOT / 'host_tests/fp32_custom/saved_p_backward_device_plan.json').read_text(encoding='utf-8-sig'))
    frozen_p = (ROOT / 'build/fp32-double-qk/double-inputs.bin').read_bytes()
    frozen_g = (ROOT / 'build/reports/g1-native-saved-p-backward-dsp/gradients.bin').read_bytes()
    if (hashlib.sha256(frozen_p).hexdigest() != plan['protected_forward_identities']['double_fixture'] or
            hashlib.sha256(frozen_g).hexdigest() != plan['gradient_sha256']):
        raise ValueError('frozen operand identity changed')

    def update(target, field, result, key):
        if field not in target or result['maxAbs'] > target[field]['maxAbs']:
            target[field] = dict(result, case=list(key))

    for key in sorted(expected):
        o, t, fi, gi, h = key
        record, raw = records[key], operands[key]
        p_offset = 12 + fi * 122888 + 65544 + h * 4096
        g_index = gi if fi < 44 else 25 + gi
        g_offset = 16 + ((g_index * 2 + h) * 2 + (mode == 'standalone_dvalue')) * 4096
        if (tuple(raw['P']) != struct.unpack_from('<1024I', frozen_p, p_offset) or
                tuple(raw['gradient']) != struct.unpack_from('<1024I', frozen_g, g_offset) or
                word_hash(raw['output']) != record['output_hash']):
            raise ValueError('raw P/gradient/head/coordinate/hash does not match frozen record')
        p, grad, output = (floats(raw[name]) for name in ('P', 'gradient', 'output'))
        if any(len(v) != 1024 for v in (p, grad, output)):
            raise ValueError('raw matrix shape')
        if mode == 'standalone_softmax':
            dot = [sum(p[r*32+c] * grad[r*32+c] for c in range(32)) for r in range(32)]
            oracle = [p[i] * (grad[i] - dot[i//32]) for i in range(1024)]
            if t:
                product, dot_actual, centered = (floats(raw[name]) for name in ('product', 'dot', 'centered'))
                local = {
                    'product_received_input': distance(product, [a*b for a, b in zip(p, grad)]),
                    'reduction_received_product': distance(dot_actual, [sum(product[r*32:r*32+32]) for r in range(32)]),
                    'subtraction_received_dot': distance(centered, [grad[i]-dot_actual[i//32] for i in range(1024)]),
                    'final_product_received_centered': distance(output, [p[i]*centered[i] for i in range(1024)]),
                }
                for field, result in local.items():
                    update(local_worst, field, result, key)
            gate = .002
        else:
            oracle = [sum(p[k*32+r] * grad[k*32+d] for k in range(32)) for r in range(32) for d in range(32)]
            gate = .01
        result = distance(output, oracle)
        if abs(result['maxAbs'] - record['maxAbs']) > 1e-10 * max(1., result['maxAbs']):
            raise ValueError('C++ vs independent Python oracle mismatch')
        update(all_worst, 'actual_vs_received_input_double', result, key)
        set_name = 'core' if fi < 44 else 'holdout'
        update(all_worst, set_name, result, key)
        counts.setdefault(f'O{o}T{t}', {'core': 0, 'holdout': 0})[set_name] += 1
        if record['variant_word_mismatches'] or not record['pass']:
            failed += 1
        accuracy_failures += result['maxAbs'] > gate
        observer_pairs += record['variant_word_mismatches'] > 0
        observer_words += record['variant_word_mismatches']
        canonical.append({k: record[k] for k in ('mode', 'observed', 'tap', 'forward', 'gradient', 'head', 'maxAbs',
                                                'rms', 'coordinate', 'ordered_fp32_delta', 'variant_word_mismatches', 'output_hash', 'pass')})
        if fi in (16, 17, 20, 21, 22) and gi in (9, 12):
            update(all_worst, f'critical_F{fi}H{h}G{gi}', result, key)
    terminal = dict(field.split('=', 1) for field in summaries[0].split(',')[2:] if '=' in field)
    if int(terminal['pairs']) != len(expected) or int(terminal['required_failures']) != failed:
        raise ValueError('terminal failure/coverage disagreement')
    for field, value in {'cpu_arithmetic_fallback': '0', 'input_mutation': '0', 'finite': '1', 'poison_remaining': '0',
                         'canary': '1', 'teardown': '1', 'native_execution': '0'}.items():
        if terminal[field] != value:
            raise ValueError('invalid execution invariant ' + field)
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    classification = 'NUMERICAL_FAIL' if accuracy_failures else 'OBSERVER_FAIL' if observer_pairs else 'PASS'
    return {'mode': mode, 'classification': classification, 'required_failed_pairs': failed,
            'accuracy_failed_pairs': accuracy_failures, 'observer_mismatch_pairs': observer_pairs,
            'observer_mismatch_words': observer_words,
            'pairs': len(expected), 'coverage': counts, 'worst': all_worst, 'local_boundary_worst': local_worst,
            'independent_python_double_recheck': 'PASS', 'numerical_digest': digest,
            'execution_invariants': {k: terminal[k] for k in ('cpu_arithmetic_fallback', 'input_mutation', 'finite',
                'poison_remaining', 'canary', 'teardown', 'native_execution')},
            'interpretation': 'DSP standalone on frozen external P; no actual Forward-produced native P certification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdout', type=pathlib.Path, required=True)
    parser.add_argument('--mode', choices=('standalone_softmax', 'standalone_dvalue'), required=True)
    parser.add_argument('--forward-limit', type=int, default=60, choices=range(1, 61))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = assess(args.stdout, args.mode, args.forward_limit)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('mode', 'classification', 'required_failed_pairs', 'pairs', 'numerical_digest')}))
