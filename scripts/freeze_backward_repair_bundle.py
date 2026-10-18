"""Freeze identical compiled bytes with current orchestration provenance.

The compiler inputs must match the immutable v4 build exactly. Only the
noncompiled runner provenance is refreshed, with both source identities retained.
"""
import json
import shutil
import subprocess
from prepare_saved_p_backward import sha,git,ROOT


def main():
    if git('status','--porcelain'):raise ValueError('committed source required')
    old=ROOT/'build/fp32-saved-p-backward-repair-v4';out=ROOT/'build/fp32-saved-p-backward-repair-final'
    build=json.loads((old/'build-identity.json').read_text(encoding='utf-8-sig'))
    for p,h in build['source_hashes'].items():
        if p!='scripts/run_backward_repair_probe.ps1' and sha(ROOT/p)!=h:raise ValueError('compiled/Host-tested source changed: '+p)
    for p,h in {**build['binary_hashes'],**build['codegen_host_hashes']}.items():
        if sha(old/p)!=h:raise ValueError('immutable bundle changed: '+p)
    # Host/DSP bodies were not changed by the cohort/evidence harness revision.
    earlier=json.loads((ROOT/'build/fp32-saved-p-backward-repair-v3/build-identity.json').read_text(encoding='utf-8-sig'))
    for p in ('libHexaBackwardRepairDsp.so','libHexaBackwardRepairHost.so'):
        if earlier['binary_hashes'][p]!=build['binary_hashes'][p]:raise ValueError('arithmetic package bytes changed')
    shutil.copytree(old,out)  # exclusive destination; preserves every older build.
    updated=dict(build,bundle_leaf=out.name,compiled_source_sha=build['source_sha'],bundle_source_sha=git('rev-parse','HEAD'),
                 runner_source_sha=git('rev-parse','HEAD'),binary_recompile=False,arithmetic_candidate_count=1,
                 previous_runner_sha256=build['source_hashes']['scripts/run_backward_repair_probe.ps1'])
    updated['source_hashes']=dict(build['source_hashes'])
    for p in ('scripts/run_backward_repair_probe.ps1','scripts/freeze_backward_repair_bundle.py'):
        updated['source_hashes'][p]=sha(ROOT/p)
    # The copied build identity is preserved under an immutable compilation name;
    # the current distribution manifest is a distinct new file.
    (out/'build-identity.json').rename(out/'compilation-identity-private.json')
    with (out/'build-identity.json').open('x',encoding='utf-8') as f:json.dump(updated,f,indent=2)
    report=ROOT/'build/reports/g1-backward-primitive-precision-repair/final-bundle-private.json'
    with report.open('x',encoding='utf-8') as f:json.dump(dict(compiled_bundle=str(old),frozen_bundle=str(out),identity=updated,production_changed=False),f,indent=2)
    print('backward_repair_bundle_freeze=PASS identical_package_bytes=1 arithmetic_candidates=1 no_evidence_overwrite=1')


if __name__=='__main__':main()
