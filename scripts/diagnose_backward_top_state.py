"""Bounded read-only Android observations. Never supplies a DSP safety permit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'build/reports/g1-backward-repair-native-completion'
PLAN = ROOT / 'host_tests/fp32_custom/backward_native_monitor_plan.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def classify(exit_code, text, expected=None):
    if exit_code != 0:
        return dict(classification='COMMAND_FAILURE', identity=None)
    if not isinstance(text, str) or not text.strip():
        return dict(classification='EMPTY_RESPONSE', identity=None)
    if 'ACTIVITY MANAGER ACTIVITIES' not in text:
        return dict(classification='TOP_RESUMED_UNKNOWN', identity=None)
    fields = re.findall(r'^\s*topResumedActivity=(.*)$', text, re.M)
    if not fields:
        return dict(classification='VALID_RESPONSE_NO_FIELD', identity=None)
    parsed = [re.fullmatch(r'ActivityRecord\{\S+ u\d+ ([^\s/]+/\S+) t\d+[^}]*\}', x.strip()) for x in fields]
    if any(p is None for p in parsed):
        return dict(classification='FIELD_PRESENT_UNPARSEABLE', identity=None)
    identities = [p.group(1) for p in parsed]
    if len(identities) != 1:
        return dict(classification='TOP_RESUMED_UNKNOWN', identity=None)
    identity = identities[0]
    state = 'TOP_RESUMED_NOT_MATCHED' if expected and expected != identity else 'TOP_RESUMED_VERIFIED'
    return dict(classification=state, identity=identity)


def safety(parsed, power):
    return (parsed['classification'] == 'TOP_RESUMED_VERIFIED'
            and bool(re.search(r'^\s*mWakefulness=Awake\s*$', power, re.M))
            and parsed['identity'].split('/')[0] != 'com.yuubinnkyoku.phonelm')


class ParserTests(unittest.TestCase):
    def test_states(self):
        good = 'ACTIVITY MANAGER ACTIVITIES\n topResumedActivity=ActivityRecord{abc u0 example.app/.Main t12}\n'
        cases = [
            (1, good, None, 'COMMAND_FAILURE'),
            (0, '', None, 'EMPTY_RESPONSE'),
            (0, None, None, 'EMPTY_RESPONSE'),
            (0, 'ACTIVITY MANAGER ACTIVITIES\n mCurrentFocus=example.app/.Main', None, 'VALID_RESPONSE_NO_FIELD'),
            (0, 'ACTIVITY MANAGER ACTIVITIES\n topResumedActivity=null', None, 'FIELD_PRESENT_UNPARSEABLE'),
            (0, good, None, 'TOP_RESUMED_VERIFIED'),
            (0, good, 'other.app/.Main', 'TOP_RESUMED_NOT_MATCHED'),
            (0, good + good, None, 'TOP_RESUMED_UNKNOWN'),
            (0, good.replace('ACTIVITY MANAGER ACTIVITIES', 'partial'), None, 'TOP_RESUMED_UNKNOWN'),
            (0, good.replace(' t12}', ''), None, 'FIELD_PRESENT_UNPARSEABLE'),
        ]
        for code, text, expected, result in cases:
            with self.subTest(result=result, text=text):
                self.assertEqual(classify(code, text, expected)['classification'], result)
        parsed = classify(0, good)
        self.assertTrue(safety(parsed, 'mWakefulness=Awake'))
        self.assertFalse(safety(parsed, 'mWakefulness=Asleep'))
        self.assertFalse(safety(classify(0, 'mWakefulness=Awake'), 'mWakefulness=Awake'))
        self.assertFalse(safety(classify(0, good.replace('example.app', 'com.yuubinnkyoku.phonelm')), 'mWakefulness=Awake'))


def diagnose():
    plan = read(PLAN)
    out = REPORT / 'monitor-diagnosis-1'
    out.mkdir(parents=True, exist_ok=False)
    adb = Path(os.environ['LOCALAPPDATA']) / 'Android/Sdk/platform-tools/adb.exe'
    started = time.monotonic()
    records = []
    def call(args, endpoint=None):
        remaining = plan['maximum_seconds'] - (time.monotonic() - started)
        if remaining <= 0 or len(records) >= plan['maximum_adb_commands']:
            raise RuntimeError('DIAGNOSTIC_BUDGET_EXHAUSTED')
        command = [str(adb)] + (['-s', endpoint] if endpoint else []) + args
        begin = time.monotonic()
        utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            p = subprocess.run(command, capture_output=True, timeout=min(remaining, plan['command_timeout_seconds']))
            code, stdout, stderr = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            code, stdout, stderr = 124, e.stdout or b'', e.stderr or b''
        text = stdout.decode('utf-8', errors='replace')
        item = dict(command_private=command, utc=utc, exit_code=code, duration_seconds=time.monotonic()-begin,
                    stdout_private=text, stderr_private=stderr.decode('utf-8', errors='replace'),
                    stdout_bytes=len(stdout), response_characters=len(text), stdout_sha256=hashlib.sha256(stdout).hexdigest())
        records.append(item)
        save(out / f'command-{len(records):02d}-private.json', item)
        return item
    summary = dict(status='BLOCKED_TOP_RESUMED_STATE', device_arithmetic_executes=0, alternative_adopted=False,
                   plan_sha256=sha(PLAN), source_sha256=sha(Path(__file__)), commands=0, rounds=[])
    try:
        listing = call(['devices', '-l'])
        if listing['exit_code']:
            raise RuntimeError('DEVICE_LIST_FAILED')
        endpoints = re.findall(r'^(\S+)\s+device\b', listing['stdout_private'], re.M)
        if not endpoints or len(endpoints) > 3:
            raise RuntimeError('DEVICE_ALIAS_COUNT_UNSAFE')
        identities = []
        for endpoint in endpoints:
            values = []
            for prop in ('ro.serialno', 'ro.boot.serialno', 'ro.kernel.qemu'):
                r = call(['shell', 'getprop', prop], endpoint)
                if r['exit_code']:
                    raise RuntimeError('IDENTITY_READ_FAILED')
                values.append(r['stdout_private'].strip())
            if not values[0] or values[0] != values[1] or values[2] == '1':
                raise RuntimeError('PHYSICAL_IDENTITY_UNKNOWN')
            identities.append(values[0])
        original = read(ROOT / 'build/reports/g1-long-linear-200k-primary/run-plan.json')['device_serial_private']
        if set(identities) != {original}:
            raise RuntimeError('PHYSICAL_IDENTITY_MISMATCH')
        endpoint = sorted(endpoints, key=lambda x: (':' in x, x))[0]
        summary['same_physical_device'] = True
        for args in (['getprop', 'ro.build.fingerprint'], ['getprop', 'ro.build.version.sdk'],
                     ['cat', '/proc/sys/kernel/random/boot_id'], ['cat', '/proc/uptime']):
            if call(['shell'] + args, endpoint)['exit_code']:
                raise RuntimeError('BUILD_BOOT_READ_FAILED')
        services = call(['shell', 'dumpsys', '-l'], endpoint)
        call(['shell', 'dumpsys', 'activity', '-h'], endpoint)
        names = set(services['stdout_private'].splitlines())
        names = {x.strip() for x in names}
        expected = None
        for i in range(plan['diagnostic_rounds']):
            power = call(['shell', 'dumpsys', 'power'], endpoint)
            control = call(['shell', 'dumpsys', 'activity', 'activities'], endpoint)
            parsed = classify(control['exit_code'], control['stdout_private'], expected)
            if expected is None and parsed['classification'] == 'TOP_RESUMED_VERIFIED':
                expected = parsed['identity']
            row = dict(round=i+1, control_command=len(records), parser_private=parsed,
                       awake=bool(re.search(r'^\s*mWakefulness=Awake\s*$', power['stdout_private'], re.M)),
                       safety_verified=power['exit_code'] == 0 and safety(parsed, power['stdout_private']), comparisons=[])
            for service in ('activity', 'activity_task'):
                if service in names:
                    result = call(['shell', 'dumpsys', service], endpoint)
                    row['comparisons'].append(dict(service=service, command=len(records),
                        parser_private=classify(result['exit_code'], result['stdout_private'], expected)))
            summary['rounds'].append(row)
        summary['status'] = ('READ_ONLY_CONTROLS_VERIFIED' if all(x['safety_verified'] for x in summary['rounds'])
                             else 'BLOCKED_TOP_RESUMED_STATE')
        summary['equivalence'] = 'NOT_ESTABLISHED_SEQUENTIAL_OBSERVATIONS_ONLY'
    except Exception as e:
        summary['reason'] = str(e)
    finally:
        summary['commands'] = len(records)
        summary['duration_seconds'] = time.monotonic()-started
        save(out / 'summary-private.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'commands', 'duration_seconds', 'device_arithmetic_executes', 'alternative_adopted')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        unittest.main(argv=['monitor-tests'])
    else:
        diagnose()
