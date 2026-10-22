"""Allow-list projection of the single frozen exit139 diagnostic; never certifies a cohort."""
import argparse
import copy
import json
import re
from pathlib import Path

from assess_backward_repair import ROOT, assess, read, require
from prepare_saved_p_backward import sha

REPORT = ROOT / 'build/reports/g1-backward-repair-native-completion'
START = 'fad9d19a669032aa6d3e29651eb547920ca76506'
TRACE = 'ddaec3470adea03c287ef3138f64a0d04ac9a0ca408d096c24303385d0c9f811'
DIGEST = 'fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267'


def expected():
    # A closed schema for this observed outcome; other outcomes need their own evidence projection.
    return dict(starting_commit=START, classification='EXIT139_REPRODUCED',
        diagnostic_processes=1, completed_processes=0, actual_exit=139,
        retries=0, formal_processes_added=0, formal_eligibility='WITHHELD',
        formal_status='BLOCKED_FORMAL_COHORT', training='NO-GO',
        visible_competing_workloads=0, competition_monitor_samples=46,
        competition_monitor_failures=0, reviewed_own_readonly_candidates=2,
        all_npu_absence_proven=False, preflight='PASS', awake='Awake',
        top_resumed='KNOWN', focus_takeover_count=0,
        completed_roles=['standalone_softmax'], incomplete_role='standalone_dvalue',
        unstarted_roles=['native_softmax', 'native_dvalue', 'dual'],
        completed_pairs=9824, core_pairs_per_observer=2200,
        holdout_pairs_per_observer=256, observers=4, graph_execute_successes=4912,
        softmax_independent_accuracy='PASS', softmax_numerical_digest=DIGEST,
        softmax_core_maxAbs=6.335392901668513e-5,
        softmax_holdout_maxAbs=2.2067718461471486e-6,
        arithmetic_word_mismatches=0, observer_word_mismatches=0,
        finite=True, complete_writes=True, softmax_teardown='PASS',
        process_teardown='INCOMPLETE', dvalue_pairs=0,
        dvalue_graphFinalize='ENTERED_NO_RETURN', last_returned_api='graphAddNode',
        last_returned_code=0, last_backend_stage='VTCM_CACHE_ACQUIRED',
        api6031_occurrences=0, token6033_occurrences=0,
        signal='SIGSEGV', signal_code='SEGV_MAPERR',
        faulting_library='libQnnHtpPrepare.so', faulting_library_offset='0x3007bac',
        crash_backtrace_obtained=True, old_tombstone_recovered=True,
        matching_backtrace_frames=17, protected_state='PASS',
        old_evidence_files_unchanged=234, candidate_rebuilt=False,
        formal_resume='NO-GO', root_cause='NOT_PROVEN')


def validate_idle(value):
    require(value == expected(), 'unverified idle-NPU projection or unknown public field')


def project(directory):
    directory = directory.resolve()
    require(directory.parent == REPORT.resolve() and directory.name.startswith('idle-npu-'), 'private diagnostic root')
    audit = read(directory / 'starting-audit-private.json')
    plan = read(directory / 'preregistered-plan-private.json')
    require(audit['status'] == 'PASS' and audit['clean_tree'], 'starting audit')
    require(plan['starting_commit'] == START and plan['maximum_probe_processes'] == 1 and plan['retries'] == 0, 'one-shot preregistration')
    require(len(audit['existing_evidence']) == 234, 'prior evidence coverage')
    for path, digest in {**audit['existing_evidence'], **audit['protected']}.items():
        require(sha(ROOT / path) == digest, 'previous evidence/protected artifact changed')
    for name, bundle in audit['bundles'].items():
        manifest = ROOT / 'build' / name / 'build-identity.json'
        require(sha(manifest) == bundle['manifest_sha256'], 'manifest changed')
        for path, digest in read(manifest)['source_hashes'].items():
            require(sha(ROOT / path) == digest, 'frozen source changed')
        for path, digest in bundle['binary_hashes'].items():
            require(sha(manifest.parent / path) == digest, 'binary changed')
    require(sha(ROOT / plan['runner']) == audit['runner_sha256'] == TRACE, 'runner changed')
    completion = read(directory / 'runner-completion-private.json')
    run = Path(completion['run_directory']).resolve()
    require(run.parent == directory, 'run containment')
    require(completion['actual_runner_exit'] == 139 and completion['second_launch'] is False, 'runner completion')
    identity = read(run / 'identity-private.json')
    require(identity['exit_code'] == read(run / 'execution-before-readback-private.json')['exit_code'] == 139, 'actual process completion')
    require(identity['source_sha'] == START and identity['mode'] == 'cohort' and identity['forward_limit'] == 60, 'candidate process')
    require(identity['orchestration_sha256'] == TRACE and identity['focus_takeover_count'] == 0, 'runtime orchestration')
    require(identity['build'] == read(ROOT / 'build/fp32-saved-p-backward-repair-final/build-identity.json'), 'executed frozen build')
    from export_public_backward_native_completion import HASHES
    require(identity['build']['probe_sha256'] == HASHES['probe'] and
        identity['hashes']['libHexaBackwardRepairDsp.so'] == HASHES['DSP'] and
        identity['hashes']['libHexaBackwardRepairHost.so'] == HASHES['Host'], 'executed binary identities')
    for path in (directory / 'preflight', run / 'preflight'):
        preflight = read(path / 'device-preflight-private.json')
        require(preflight['status'] == 'PASS' and preflight['gates']['awake'] == preflight['gates']['top_resumed'] == 'PASS', 'safety preflight')
    stdout = (run / 'stdout.txt').read_text(encoding='utf-8-sig')
    stderr = (run / 'stderr.txt').read_text(encoding='utf-8-sig')
    owned = re.findall(r'^owned_probe_pid=(\d+)\s*$', stdout, re.M)
    require(len(owned) == 1, 'one actual probe PID')
    require(re.findall(r'^cohort_begin,mode=(\w+)', stdout, re.M) == plan['roles'][:2], 'role order')
    require(re.findall(r'^cohort_end,mode=(\w+),exit=0', stdout, re.M) == ['standalone_softmax'], 'completed roles')
    unfinished = stdout.split('cohort_begin,mode=standalone_dvalue\n')[1]
    require('api,graphFinalize,' not in unfinished and 'api,graphExecute,' not in unfinished and 'repair_pair' not in unfinished, 'dValue incomplete')
    apis = re.findall(r'^api,(.+)$', stdout, re.M)
    require(apis[-1] == 'graphAddNode,0' and apis.count('graphExecute,0') == 4912 and all(a.endswith(',0') for a in apis), 'API accounting')
    require('QnnGraph_finalize started' in stderr and 'Acquired Cache' in stderr and 'Segmentation fault' in stderr, 'crash stage')
    for token in ('6031', '6033'):
        require(not re.search(r'(?<!\d)' + token + r'(?!\d)', stdout + stderr), 'separate runtime error observed')
    samples = [json.loads(line) for line in (directory / 'competition-monitor-private.jsonl').read_text().splitlines()]
    require(len(samples) == completion['samples'] == 46 and not completion['monitor_command_failures'], 'monitor coverage')
    candidates = []
    observed_owned = 0
    for sample in samples:
        require(sample['exit_code'] == 0, 'monitor gap')
        detected = []
        for line in sample['stdout'].splitlines()[1:]:
            fields = line.split(None, 4)
            require(len(fields) == 5, 'incomplete process row')
            pid, _, _, name, args = fields
            if name.startswith('[') or name in ('hexlpservice', 'cdsprpcd', 'traced_probes', 'ps'):
                continue
            if name == 'sh':
                require(not re.search(r'(?i)qnn|htp|hmx|train|probe|llama|snpe|mnn', args) or
                    ('hexasbp-23616e58144eb23a-' + run.name in args and './saved_p_backward_probe' in args), 'unrecognized shell workload')
                continue
            if name == 'saved_p_backward_probe' and pid == owned[0]:
                observed_owned += 1
                continue
            if re.search(r'(?i)qnn|htp|hmx|hexatrain|phonelm|train|probe|llama|snpe|mnn', name + ' ' + args):
                detected.append(line)
        require(detected == sample['suspects'], 'unaccounted recognizable workload')
        candidates += sample['suspects']
    require(observed_owned > 0, 'missing runtime competition observation')
    require(candidates == completion['competition'] and len(candidates) == 2, 'competition observations')
    require(re.fullmatch(r'\d+ \d+ shell\s+sha256sum\s+sha256sum /data/local/tmp/hexasbp-23616e58144eb23a-' + re.escape(run.name) + r'/libQnnHtpPrepare.so', candidates[0]), 'unexpected competitor')
    require(re.fullmatch(r'\d+ \d+ shell\s+run-as\s+run-as com\.yuubinnkyoku\.phonelm cat files/headless/status\.json', candidates[1]), 'unexpected competitor')
    receipt = read(directory / 'readback-receipt-private.json')
    raw = directory / 'preserved-completed-roles/standalone_softmax.bin'
    require(receipt['status'] == 'PASS' and receipt['physical_identity_verified'] and receipt['additional_dsp_executes'] == 0, 'readback')
    require(sha(raw) == receipt['sha256'] and raw.stat().st_size == receipt['bytes'] == 284827248, 'completed role raw')
    assessment = read(directory / 'completed-softmax-independent-assessment-private.json')
    role_view = directory / 'completed-softmax-role-only'
    require(sha(role_view / 'saved-p-words.bin') == receipt['sha256'] and
        read(role_view / 'identity-private.json')['parent_process_exit'] == 139, 'role-only evidence binding')
    require(assess(role_view) == assessment, 'independent assessment stale')
    require(assessment['classification'] == 'PASS' and assessment['pairs'] == 9824 and assessment['numerical_digest'] == DIGEST, 'independent full role audit')
    require(assessment['arithmetic_word_mismatches'] == assessment['observer_mismatch_words'] == 0 and assessment['api_success'] and assessment['execution_invariants'], 'independent words/runtime')
    require(assessment['metrics']['core_0']['maxAbs'] == expected()['softmax_core_maxAbs'] and
        assessment['metrics']['holdout_0']['maxAbs'] == expected()['softmax_holdout_maxAbs'], 'independent maxima')
    new = (directory / 'tombstone_06-private.txt').read_text()
    old = (directory / 'tombstone_05-private.txt').read_text()
    old_stdout = (REPORT / 'cohort-1-awake-93dd0a34/stdout.txt').read_text(encoding='utf-8-sig')
    old_owned = re.findall(r'^owned_probe_pid=(\d+)\s*$', old_stdout, re.M)
    require(len(old_owned) == 1 and 'pid: ' + old_owned[0] + ',' in old, 'historical crash identity')
    frames = lambda text: re.findall(r'#(\d+) pc ([0-9a-f]+)\s+\S*/([^/\s]+)', text.split('backtrace:', 1)[1].split('\nmemory near', 1)[0])
    require('pid: ' + owned[0] + ',' in new and 'signal 11 (SIGSEGV), code 1 (SEGV_MAPERR)' in new, 'crash identity/signal')
    require(frames(new) == frames(old) and len(frames(new)) == 17 and frames(new)[0] == ('00', '0000000003007bac', 'libQnnHtpPrepare.so'), 'old/new backtrace')
    result = expected()
    validate_idle(result)
    return result


def selftest():
    validate_idle(expected())
    changes = dict(classification='EXIT139_NOT_REPRODUCED_ONCE', actual_exit=0,
        visible_competing_workloads=1, competition_monitor_failures=1,
        formal_processes_added=1, all_npu_absence_proven=True, dvalue_pairs=1,
        softmax_independent_accuracy='FAIL', observer_word_mismatches=1,
        api6031_occurrences=1, token6033_occurrences=1, crash_backtrace_obtained=False,
        protected_state='FAIL', candidate_rebuilt=True, formal_resume='GO',
        root_cause='VERIFIED', private_path='private', softmax_numerical_digest='a'*64)
    for key, value in changes.items():
        candidate = copy.deepcopy(expected()); candidate[key] = value
        try:
            validate_idle(candidate)
        except ValueError:
            continue
        raise AssertionError('false idle-NPU claim accepted: ' + key)
    print('idle_npu_exporter_selftest=PASS rejection_cases=' + str(len(changes)) + ' device_calls=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        from export_public_backward_native_completion import validate
        require(args.report_root is not None and args.output is not None, 'explicit source/output required')
        aggregate = read(ROOT / 'docs/results/g1-backward-repair-native-completion-summary.json')
        validate(aggregate)
        aggregate['idle_npu_reproduction'] = project(args.report_root)
        validate(aggregate)
        with args.output.open('x', encoding='utf-8') as target:
            json.dump(aggregate, target, indent=2, allow_nan=False)
        print('idle_npu_public=PASS classification=EXIT139_REPRODUCED formal_processes_added=0 training=NO-GO')
