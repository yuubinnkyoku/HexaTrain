"""Exclusive local provenance freeze. No device operations or checkpoint decode."""
import json
import pathlib
import subprocess
from prepare_saved_p_backward import sha, git, ROOT

OUT = ROOT / 'build/reports/g1-backward-primitive-precision-repair'


def main():
    base = git('rev-parse', '49d91b6')
    assert base == '49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2'
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', base, 'HEAD'], check=True)
    assert git('branch', '--show-current') == 'investigation/g1-backward-primitive-precision-repair'
    for ancestor in ('e05f353', 'be4271e'):
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', ancestor, base], check=True)
    prior = json.loads((ROOT / 'build/reports/g1-native-saved-p-backward-dsp/initial-private.json').read_text(encoding='utf-8-sig'))
    for p, h in prior['protected'].items():
        assert sha(ROOT / p) == h, ('PROTECTED_MISMATCH', p)
    for line in prior['refs'].splitlines():
        h, ref = line.split()
        # This was the then-active branch in the old initial ledger. Its
        # authorized additive commits ended at the user's new exact base.
        expected = base if ref == 'refs/heads/investigation/g1-native-saved-p-backward-dsp' else h
        assert git('rev-parse', ref) == expected, ('PRIOR_REF_MISMATCH', ref)
    bundles = {}
    for name in ('fp32-custom', 'fp32-selective-ln1', 'fp32-mixed-qkv', 'fp32-double-qk', 'fp32-saved-p-backward-final'):
        manifest = json.loads((ROOT / 'build' / name / 'build-identity.json').read_text(encoding='utf-8-sig'))
        for p, h in manifest['source_hashes'].items():
            assert sha(ROOT / p) == h, ('SOURCE_MISMATCH', p)
        for p, h in manifest.get('binary_hashes', {}).items():
            assert sha(ROOT / 'build' / name / p) == h, ('BINARY_MISMATCH', p)
        if 'probe_sha256' in manifest:
            assert sha(ROOT / 'build' / name / 'saved_p_backward_probe') == manifest['probe_sha256']
        bundles[name] = manifest
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / 'initial-private.json').open('x', encoding='utf-8') as f:
        json.dump(dict(root=str(ROOT), common_dir=git('rev-parse', '--git-common-dir'), base=base,
                       branch=git('branch', '--show-current'), refs=git('show-ref'),
                       protected=prior['protected'], bundles=bundles, device_calls=0), f, indent=2)
    print('backward_precision_provenance=PASS protected=5 previous_refs_preserved=1 device_calls=0')


if __name__ == '__main__':
    main()
