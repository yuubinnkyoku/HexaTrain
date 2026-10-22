"""Exact-source audit: no numerical edits are authorized by lifetime expansion."""
from pathlib import Path
import hashlib
import json
from assess_backward_repair import ROOT, require, read


def audit():
    src = ROOT/'host_tests/fp32_custom'
    old = (src/'backward_shared_lifetime_probe.cpp').read_text(encoding='utf-8')
    new = (src/'backward_shared_lifetime_formal_probe.cpp').read_text(encoding='utf-8')
    # Every worker/oracle/observer/graph call before the new entry point is exact.
    prefix = old.split('int main(int argc,char** argv)', 1)[0]
    prefix = prefix.replace('backward_shared_lifetime.h', 'backward_shared_lifetime_formal.h').replace('sharedlife::', 'formalsharedlife::')
    fail_closed='     if(!pass){std::fclose(savedp::rawEvidence);closeProbe();return 4;} // Fail closed before another DSP execution.\n'
    require(new.count(fail_closed)==1 and new.replace(fail_closed,'').startswith(prefix), 'worker/oracle/observer/layout exact source except immediate failure stop')
    owner = (src/'backward_shared_lifetime.h').read_text(encoding='utf-8')
    require((src/'backward_shared_lifetime_formal.h').read_text(encoding='utf-8') ==
            owner.replace('namespace sharedlife {', 'namespace formalsharedlife {').replace('roles>=2', 'roles>=5'), 'proven owner only capacity/namespace')
    # Independent numerical body also stays exact; only role and process entry change.
    oracle = (ROOT/'scripts/assess_backward_shared_lifetime.py').read_text(encoding='utf-8')
    formal = (ROOT/'scripts/assess_backward_shared_lifetime_formal.py').read_text(encoding='utf-8')
    require(oracle.split('    fixture=', 1)[1].replace("candidate='shared-lifetime-boundary-v1'", "candidate='shared-lifetime-five-role-final'") ==
            formal.split('    fixture=', 1)[1], 'independent numerical body exact')
    plan = read(src/'backward_shared_lifetime_formal_plan.json')
    historical = src/'backward_repair_cohort_plan.json'
    require(hashlib.sha256(historical.read_bytes()).hexdigest()==plan['original_formal_plan']['sha256'], 'historical plan immutable')
    return dict(status='PASS', worker_math='EXACT', numerical_oracle='EXACT',
                owner_delta='namespace and capacity 2 to 5 only', original_plan='UNCHANGED', device_calls=0)


if __name__=='__main__':
    print(json.dumps(audit()))
