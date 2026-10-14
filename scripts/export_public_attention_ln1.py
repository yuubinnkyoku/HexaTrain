"""Allowlisted aggregate export. Private ledgers/tensors/paths never leave build/."""
import argparse, hashlib, json, math, re
from pathlib import Path

def export(cohort, output):
    entries=json.loads(Path(cohort).read_text(encoding="utf-8-sig"))
    assert set(entries)=={"oneAttention","oneLn1","oneObserver"}
    summary={"schema":1,"lanes":{}}
    identities=[]
    metric=re.compile(r"^onemetric,([^,]+),(\d+),([^,]+),(.+)$")
    for lane, paths in entries.items():
        assert len(paths)==3 and len(set(paths))==3
        processes=[]; baseline=None; groups={}
        for path in paths:
            ledger=json.loads((Path(path)/"identity-private.json").read_text(encoding="utf-8-sig"))
            assert ledger["variant"]==lane and ledger["status"]=="COMPLETE"
            assert ledger["exit_code"]==0 and ledger["focus_takeover_count"]==0 and ledger["protected_identity"]=="PASS"
            assert ledger["branch"]=="investigation/g1-fp32-attention-ln1-frontier"
            build=ledger["build"]
            assert ledger["source_sha"]==build["source_sha"] and re.fullmatch(r"[0-9a-f]{40}",build["source_sha"])
            assert build["qairt_build_id"]=="2.48.40.260702151143" and build["architecture"]=="v81"
            selected={key:ledger["hashes"][key] for key in ("libHexaFp32Host.so","libHexaFp32Dsp.so","one_layer_probe","libQnnHtp.so","libQnnHtpPrepare.so","libQnnHtpV81Stub.so","libQnnHtpV81Skel.so","one-layer-inputs.bin")}
            for key in ("libHexaFp32Host.so","libHexaFp32Dsp.so","one_layer_probe"):assert selected[key]==build["binary_hashes"][key]
            if lane=="oneLn1":
                fixtures={key:ledger["hashes"][key] for key in ("ln1-parameters.bin","ln1-producers.bin")}
            else: fixtures={}
            identity={"source_commit":build["source_sha"],"binary_hashes":selected,"sdk_build":build["qairt_build_id"],"hexagon_compiler_sha256":build["hexagon_compiler_sha256"],"ndk_compiler_sha256":build["ndk_compiler_sha256"]}
            identities.append(identity)
            lines=(Path(path)/"stdout.txt").read_text(encoding="utf-8",errors="strict").splitlines()
            assert f"summary,{lane},failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0" in lines
            for line in lines:
                if line.startswith("api,"):assert line.rsplit(",",1)[1]=="0"
                assert not re.search(r"^(?:CANARY_FAIL|POISON_FAIL|INPUT_MUTATION|\w*RECEIPT_FAIL|\w*NONFINITE\w*)",line)
            for teardown in ("contextFree","deviceFree","backendFree"):assert f"api,{teardown},0" in lines
            canonical=[line for line in lines if line.startswith(("onemetric,","dpObserver,","observerDifference,"))]
            digest=hashlib.sha256(("\n".join(canonical)+"\n").encode()).hexdigest()
            assert baseline is None or baseline==canonical,"numerical digest mismatch"
            if baseline is None:
                baseline=canonical
                seen=set()
                for line in canonical:
                    match=metric.fullmatch(line)
                    if not match:continue
                    mode, case, stage, tail=match.groups();key=(mode,case,stage);assert key not in seen;seen.add(key)
                    values=dict(piece.split("=",1) for piece in tail.split(","))
                    nonfinite=int(values["nonfinite"])
                    if nonfinite:assert mode in ("ln1V0O0","ln1V1O0") and values["gate"]=="0"
                    else:
                        for name in ("abs","rms","relative_l2","max_relative","fp_abs","limit","cosine"):assert math.isfinite(float(values[name]))
                    assert int(values["gate"])==int(not nonfinite and float(values["abs"])<=float(values["limit"]))
                    group=groups.setdefault(mode+":"+stage,{"mode":mode,"stage":stage,"cases":0,"failures":0,"max_abs":0,"max_fp_abs":0,"fp_word_mismatches":0,"gate":float(values["limit"])})
                    assert group["gate"]==float(values["limit"])
                    group["cases"]+=1;group["failures"]+=values["gate"]=="0";group["nonfinite_cases"]=group.get("nonfinite_cases",0)+bool(nonfinite)
                    if not nonfinite:
                        group["max_abs"]=max(group["max_abs"],float(values["abs"]));group["max_fp_abs"]=max(group["max_fp_abs"],float(values["fp_abs"]))
                    group["fp_word_mismatches"]+=int(values["fp_word_mismatch"])
            executes=sum(line=="api,graphExecute,0" for line in lines)
            assert executes>0 and len(canonical)>0
            processes.append({"ordinal":len(processes)+1,"numerical_digest":digest,"execute_count":executes,"metric_count":sum(line.startswith("onemetric,") for line in canonical),"api_success":True,"finite":True,"protected_checks":True})
        summary["lanes"][lane]={"processes":processes,"aggregate":list(groups.values())}
    assert all(identity==identities[0] for identity in identities)
    summary["provenance"]=identities[0]
    summary["ln1_fixture_hashes"]=fixtures if fixtures else {key:json.loads((Path(entries['oneLn1'][0])/"identity-private.json").read_text(encoding="utf-8-sig"))["hashes"][key] for key in ("ln1-parameters.bin","ln1-producers.bin")}
    text=json.dumps(summary,ensure_ascii=False,indent=2)+"\n"
    assert not re.search(r"[A-Za-z]:[\\/]|/data/|device_serial|endpoint|checkpoint_path",text)
    Path(output).write_text(text,encoding="utf-8",newline="\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("cohort");parser.add_argument("output");args=parser.parse_args()
    export(args.cohort,args.output)
