"""Lossless Forward-only raw transport into the existing joint P assessor."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from assess_backward_repair import ROOT, fnv, read, require, words
from prepare_saved_p_backward import sha


def assess(directory):
    identity = read(directory / 'identity-private.json')
    completion = read(directory / 'execution-before-readback-private.json')
    require(identity['mode'] == 'forward' and identity['forward_limit'] == 60, 'Forward role/coverage')
    require(identity['exit_code'] == completion['exit_code'] == 0, 'actual process completion')
    require(identity['focus_takeover_count'] == 0 and identity['protected_identity'] == 'PASS', 'safety')
    require(identity['orchestration_file'] == 'scripts/run_backward_repair_trace_probe.ps1', 'trace source')
    require(identity['orchestration_sha256'] == completion['orchestration_sha256'] == sha(ROOT / identity['orchestration_file']), 'trace SHA')
    build = identity['build']
    require(build['probe_sha256'] == '23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05', 'frozen probe')
    for name, value in build['binary_hashes'].items():
        require(sha(ROOT / 'build/fp32-saved-p-backward-repair-final' / name) == value, 'binary SHA')
    for name, value in build['source_hashes'].items():
        require(sha(ROOT / name) == value, 'source SHA')
    require(sha(directory / 'saved-p-words.bin') == identity['raw_evidence_sha256'], 'raw SHA')
    require(sha(directory / 'saved-p-words.bin.gz') == identity['compressed_evidence_sha256'], 'gzip SHA')
    text = (directory / 'stdout.txt').read_text(encoding='utf-8-sig')
    terminal = 'summary,backward_repair,mode=forward,pairs=0,required_failures=0,input_mutation=0,finite=1,complete_writes=1,poison_remaining=0,canary=1,teardown=1,cpu_arithmetic_fallback=0'
    require(text.splitlines().count(terminal) == 1, 'Forward terminal/teardown')
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    outputs = [r for r in rows if r.get('kind') == 'repair_forward']
    require(len(outputs) == 480, 'P output coverage')
    # API lines are checked with the same explicit nonzero rejection as the
    # original independent backward assessor; final finite/guards stay separate.
    api_lines = [line for line in text.splitlines() if line.startswith('api,')]
    require(api_lines and all(len(line.split(',')) == 3 and line.endswith(',0') for line in api_lines), 'QNN API')
    apis = [line.split(',')[1] for line in api_lines]
    require(apis.count('graphFinalize') == 4 and apis.count('graphExecute') == 240, 'Forward graph coverage')
    require({'registerRepairHost', 'registerRepairDSP', 'registerOldDSP', 'registerS2DSP', 'registerMixedDSP', 'registerDoubleDSP', 'contextFree', 'deviceFree', 'backendFree'} <= set(apis), 'HTP/teardown APIs')
    derived = directory / 'derived-forward-p-words.txt'
    with (directory / 'saved-p-words.bin').open('rb') as raw, derived.open('x', encoding='ascii') as target:
        require(tuple(words(raw, 4)) == (0x42525001, 1, 2, 60), 'Forward raw header')
        index = 0
        for observed in (1, 0):
            for tap in (1, 0):
                for case in range(60):
                    for head in range(2):
                        row = outputs[index]
                        require(tuple(row[k] for k in ('observed', 'tap', 'forward', 'head')) == (observed, tap, case, head), 'P order')
                        require(tuple(words(raw, 3)) == (1, head, 1024), 'P record identity/length')
                        value = words(raw, 1024)
                        require(fnv(value) == row['hash'], 'P stdout/raw words')
                        require(np.isfinite(value.view('<f4')).all(), 'P finite')
                        target.write('p_words,' + ','.join(map(str, (observed, tap, case, head))) + ',' + ','.join(f'{int(v):08x}' for v in value) + '\n')
                        index += 1
        require(raw.read(1) == b'', 'trailing raw')
    executable = ROOT / 'build/reports/g1-s2-double-qk-native-p-closure/host-final/double_qk_assessment.exe'
    require(sha(executable) == '10eff7ddc5897848943a38ad664626e3be149323f345fcc9c4d54455622640ce', 'independent assessor SHA')
    result = subprocess.run([str(executable), str(ROOT / 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'), str(ROOT / 'build/fp32-double-qk/double-inputs.bin'), str(derived)], capture_output=True, timeout=120)
    with (directory / 'forward-joint.jsonl').open('xb') as target:
        target.write(result.stdout)
    with (directory / 'forward-joint-stderr.txt').open('xb') as target:
        target.write(result.stderr)
    summaries = [json.loads(line) for line in result.stdout.decode().splitlines() if line.startswith('{') and json.loads(line).get('kind') == 'assessment_summary']
    require(result.returncode == 0 and len(summaries) == 1 and summaries[0]['joint_pass'], 'independent joint contract')
    require(summaries[0]['p_outputs'] == 480 and summaries[0]['required_failed_predicates'] == summaries[0]['variant_word_mismatches'] == 0, 'joint coverage/observer')
    return dict(status='PASS', role='forward', actual_exit=identity['exit_code'], raw_sha256=identity['raw_evidence_sha256'], derived_sha256=sha(derived), assessor_sha256=sha(executable), joint=summaries[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-root', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.report_root)
    with (args.report_root / 'joint-assessment-private.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
