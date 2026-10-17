"""Strict aggregate exporter. Raw tensor/private identifiers are never copied."""
import argparse, collections, copy, hashlib, json, math, pathlib, re, subprocess

GATES={"p_maxAbs":1e-6,"row_error":2e-6,"ds_bound":.00025,"dv_bound":.001,"ds_actual":.002,"dv_actual":.01}
MASKS=[0,1,2,4,3,5,6,7]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))
def records(p):
    with open(p,encoding="utf-8-sig") as f:
        for line in f:yield json.loads(line)
def require(b,message):
    if not b:raise ValueError(message)
def finite(x):
    if isinstance(x,float):require(math.isfinite(x),"nonfinite aggregate")
    if isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)

def validate_group(cases,pairs,s):
    expected={"frozen":(44,25),"holdout":(16,8),"confirmation":(16,8)}[s["set"]]
    nf,ng=expected
    require(len(cases)==nf*2 and len(pairs)==nf*2*ng,"coverage count")
    require({(x["forward"],x["head"]) for x in cases}=={(i,h) for i in range(nf) for h in range(2)},"case coverage")
    require({(x["forward"],x["head"],x["gradient"]) for x in pairs}=={(i,h,g) for i in range(nf) for h in range(2) for g in range(ng)},"Cartesian coverage")
    mapping={"p_maxAbs":(cases,"p_maxAbs"),"row_error":(cases,"row_error"),"ds_bound":(pairs,"ds_bound"),"dv_bound":(pairs,"dv_bound"),"ds_signed":(pairs,"ds_signed"),"dv_signed":(pairs,"dv_signed"),"ds_actual":(pairs,"ds_fp_actual"),"dv_actual":(pairs,"dv_fp_actual")}
    for name,(rows,key) in mapping.items():require(max(x[key] for x in rows)==s[name],"summary maximum "+name)
    counts=[sum(x["p_maxAbs"]>1e-6 for x in cases),sum(x["row_error"]>2e-6 for x in cases),sum(x["ds_bound"]>.00025 for x in pairs),sum(x["dv_bound"]>.001 for x in pairs),sum(x["ds_fp_actual"]>.002 for x in pairs)+sum(x["dv_fp_actual"]>.01 for x in pairs)]
    for key,value in zip(["p_fail","row_fail","ds_fail","dv_fail","actual_fail"],counts):require(value==s[key],"predicate count "+key)
    require(sum(counts)==s["failures"] and s["joint_pass"]==(sum(counts)==0),"joint classification")
    require(s["pairs"]==len(pairs),"pairs")
    failed={(x["forward"],x["head"]) for x in cases if x["p_maxAbs"]>1e-6 or x["row_error"]>2e-6}
    failed|={(x["forward"],x["head"]) for x in pairs if x["ds_bound"]>.00025 or x["dv_bound"]>.001 or x["ds_fp_actual"]>.002 or x["dv_fp_actual"]>.01}
    require(len(failed)==s["unique_failed_pairs"],"unique failure count")
    for x in pairs:
        require(x["ds_signed"]<=x["ds_bound"]+1e-10 and x["dv_signed"]<=x["dv_bound"]+1e-10,"signed bound")
    return failed

def analyze(p,model):
    groups=collections.defaultdict(lambda:{"case":[],"pair":[],"summary":None})
    saved=[];consistent=[];same=[]
    for x in records(p):
        finite(x);k=x["kind"]
        if k in ("case","pair","summary"):
            if x["set"]=="same_frozen":same.append(x);continue
            key=(x["candidate"],x["set"],x["reference"])
            if k=="summary":require(groups[key][k] is None,"duplicate summary");groups[key][k]=x
            else:groups[key][k].append(x)
        elif k=="saved_ln":saved.append(x)
        elif k=="consistency":
            require(x["anchor_word_mismatches"]==0 and x["saved_qk_word_mismatches"]==0 and x["local_qk_gate"],"operand/anchor consistency")
            consistent.append(x)
    summaries=[];failures=[];critical=[]
    watched={(16,1,9),(17,1,9),(20,1,9),(21,1,9),(21,1,12),(22,0,9),(22,0,12),(22,1,9)}
    for key,g in groups.items():
        s=g["summary"];require(s is not None,"missing summary");validate_group(g["case"],g["pair"],s)
        summaries.append({**s,"model":model,"p_rms_max":max(x["rms"] for x in g["case"])})
        for x in g["case"]:
            if x["p_maxAbs"]>1e-6 or x["row_error"]>2e-6:failures.append({**x,"model":model})
            if x["set"]=="frozen" and (x["forward"],x["head"])==(21,1):critical.append({**x,"model":model})
        for x in g["pair"]:
            # Only numeric aggregate/coordinates; no raw arrays or tensor words.
            safe={k:x[k] for k in ("candidate","set","reference","forward","head","gradient","ds_bound","dv_bound","ds_signed","dv_signed","ds_fp_actual","dv_fp_actual","bound_coordinate","signed_t","absolute_t","bound_term1","bound_term2")}
            if x["ds_bound"]>.00025 or x["dv_bound"]>.001 or x["ds_fp_actual"]>.002 or x["dv_fp_actual"]>.01:failures.append({**safe,"model":model})
            if x["set"]=="frozen" and (x["forward"],x["head"],x["gradient"]) in watched:critical.append({**safe,"model":model})
    for s in same:
        g=groups[(s["candidate"],"frozen",s["reference"])];pairs=[x for x in g["pair"] if x["forward"]<24 and x["gradient"]==x["forward"]]
        cases=[x for x in g["case"] if x["forward"]<24]
        require(len(pairs)==48 and s["pairs"]==48,"same-frozen coverage")
        for dst,rows,key in [("p_maxAbs",cases,"p_maxAbs"),("row_error",cases,"row_error"),("ds_bound",pairs,"ds_bound"),("dv_bound",pairs,"dv_bound"),("ds_actual",pairs,"ds_fp_actual"),("dv_actual",pairs,"dv_fp_actual")]:require(max(x[key] for x in rows)==s[dst],"matched maximum")
        require(s["joint_pass"]==all(s[k]<=v for k,v in GATES.items()),"matched gate")
    return {"summaries":summaries,"same_frozen":[{**s,"model":model} for s in same],"failures":failures,"critical_original_predicates":critical,"saved_ln":saved,"consistency":consistent}

def cost(mask):
    r,n,a=bool(mask&1),bool(mask&2),bool(mask&4)
    return {"ln_double_add_sub":32*(192*r+(1 if r and n else 0)+64*a),"ln_double_mul":2048*(r+n+a),"ln_double_div_by_64":64*r,"ln_double_reciprocal":32*n,"ln_double_sqrt":32*(r),"ln_double_hypot":32*(n and not r),"epsilon_sqrt":"compile-time constant; not a measured runtime call",
        "matmul_double_mul":327680,"matmul_double_add":327680,"fp32_output_stores_ln":4128,"extra_saved_bytes":0,"source_center_temporary_bytes":512,"compiled_stack_excludes_callees":568 if r and n else 560,
        "conversion_note":"Actual static conversion sites and spills in codegen; source double center[64] includes float->double->float in unselected paths; compiler removes many. Dynamic counts depend on branches and are not equated to source casts.","qnn_launches":"LN1 + mixed QKV + paired QK = 3 proposed; split QK/V = 4; no graph built"}

def codegen(root):
    root=root/"codegen-complete"
    ident=read(root/"identity-private.json")
    for name,d in ident["outputs"].items():require(sha(root/name)==d,"codegen hash")
    asm=(root/"selective.s").read_text(encoding="utf-8-sig")
    funcs=[]
    for match in re.finditer(r"\.type\s+([^,\s]+),@function(.*?)\.size\s+\1,",asm,re.S):
        name,body=match.groups()
        if "_ZN9selective2lnILj" in name:
            mask=int(re.search(r"ILj(\d+)E",name).group(1));label="S"+str(MASKS.index(mask))
        elif name.startswith("selective_qk_"):label=name
        else:continue
        funcs.append({"function":label,"static_dfadd_sites":len(re.findall(r"dfadd\(",body)),"static_df_partial_multiply_sites":len(re.findall(r"dfmpy(?:fix|ll|lh|hh)\(",body)),"static_sf2df_sites":body.count("convert_sf2df"),"static_df2sf_sites":body.count("convert_df2sf"),"folded_spill_sites":body.count("Folded Spill"),"folded_reload_sites":body.count("Folded Reload"),"calls":sorted(set(re.findall(r"\bcall\s+([A-Za-z0-9_]+)",body)))})
    require(len(funcs)==10,"all actual arithmetic functions")
    require(not re.search(r"llvm\.fma|fadd fast|fmul fast",(root/"selective.ll").read_text()),"strict codegen")
    require(sha(root/"link-private.txt")==ident["link_log_sha256"] and sha(root/"link-explicit-private.txt")==ident["explicit_link_log_sha256"],"link evidence")
    return {"source_commit":ident["source_commit"],"compiler_sha256":ident["compiler_sha256"],"source_sha256":ident["source_sha256"],"arithmetic_sha256":ident["arithmetic_sha256"],"sdk_manifest_sha256":ident["sdk_manifest_sha256"],"flags":ident["flags"],"output_hashes":ident["outputs"],"functions":funcs,"runtime_library_hashes":ident["runtime_library_hashes"],"pic_archive_hashes":ident["pic_archive_hashes"],"static_no_undefined_link_exit":ident["static_no_undefined_link_exit"],"explicit_pic_archive_link_exit":ident["explicit_pic_archive_link_exit"],"link_log_sha256":ident["link_log_sha256"],"explicit_link_log_sha256":ident["explicit_link_log_sha256"],"runtime_linkage_verified":False,"latency_measured":False,"native_float64_required":False}

def export(root,repo):
    identity=read(root/"host/identity-private.json")
    require(sha(root/"host/selective_ln1_test.exe")==identity["exe_sha256"],"Host binary")
    for p,d in identity["source_hashes"].items():require(sha(repo/p)==d,"final arithmetic source changed")
    selection=read(repo/"host_tests/fp32_custom/selective_ln1_selection.json")
    require(selection["selected"]=="S2" and identity["exe_sha256"]==selection["development_executable_sha256"],"selection")
    analyses={};processes={}
    for phase in ("development","confirmation"):
        process=read(root/f"host/{phase}-processes-private.json")
        hashes=[sha(root/f"host/{phase}-{i}.jsonl") for i in (1,2,3)]
        require(hashes==process["numerical_digests"] and len(set(hashes))==1,"3 process digests")
        require(sha(root/f"host/{phase}-fp32-shift.jsonl")==process["shift_digest"],"shift digest")
        require(process["exe_sha256"]==identity["exe_sha256"],"final binary fixed")
        processes[phase]={k:process[k] for k in ("exe_sha256","numerical_digests","shift_digest","source_commit")}
        for suffix,model in (("1","legacy_r5"),("fp32-shift","fp32_shift")):
            analyses[phase+"_"+model]=analyze(root/f"host/{phase}-{suffix}.jsonl",model)
    require(processes["development"]["numerical_digests"][0]==selection["development_numerical_digest"] and processes["development"]["shift_digest"]==selection["development_shift_digest"],"selection before confirmation")
    for key,a in analyses.items():
        ids={s["candidate"] for s in a["summaries"]}
        require(ids==({"S0","S2","S7"} if key.startswith("confirmation") else {"S"+str(i) for i in range(8)}),"candidate coverage")
        for s in a["summaries"]:
            if s["candidate"] in ("S2","S7"):require(s["joint_pass"],"selected/anchor joint contract")
    saved_id=read(root/"host/saved-final-identity-private.json");require(saved_id["exit_code"]==0 and sha(root/"host/saved-final.jsonl")==saved_id["log_sha256"] and sha(root/"host/selective_ln1_saved_final.exe")==saved_id["exe_sha256"],"actual saved harness identity")
    saved=list(records(root/"host/saved-final.jsonl"));require(len(saved)==16,"saved coverage")
    for s in saved:require(s["backward_gate"] and s["dx"]<=.02 and s["dg"]<=.02 and s["db"]<=.02 and s["complete_finite"] and s["input_mutation"]==0,"actual saved gate")
    baseline=read(root/"baseline/analysis-identity-private.json")
    require(baseline["tests"]["probability_reference_test"]["exit_code"]==4,"old expected fail")
    legacyhash=sha(root/"baseline/attribution.jsonl");require(legacyhash=="c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7","original 9/R0-R5 unchanged")
    oldfail=[{k:x[k] for k in ("forward","head","gradient","predicate","value","threshold","excess")} for x in records(root/"baseline/attribution.jsonl") if x["kind"]=="failure"]
    require(len(oldfail)==9,"9 predicates")
    prior=read(root/"reproduced/host-identity-private.json")
    require(len(set(prior["tests"]["upstream_precision_test"]["logs"][f"matrix-{i}.jsonl"] for i in (1,2,3)))==1,"old cohort")
    require(sha(root/"reproduced/matrix-1.jsonl")=="daa9f90c6506f808c516110f64c1887ea099dae657d16a89051065ed144dc126","old C0-C7 matrix and holdout unchanged")
    safety=read(root/"initial-preflight/device-preflight-private.json")
    require(safety["device_executes"]==0 and safety["read_only"],"no unsafe device work")
    profiles={}
    for profile in ("Fast","Host"):
        p=root/(profile.lower()+"-profile.txt");text=p.read_text(encoding="utf-8-sig")
        require(re.search(r"total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$",text),"profile incomplete "+profile)
        profiles[profile]={"status":"PASS","log_sha256":sha(p)}
    initial=read(root/"initial-private.json")
    for p,d in initial["protected"].items():require(sha(repo/p)==d,"protected artifact")
    for p,d in initial["previous_build"]["source_hashes"].items():require(sha(repo/p)==d,"old DSP source")
    for p,d in initial["previous_build"]["binary_hashes"].items():require(sha(repo/"build/fp32-custom"/p)==d,"old binary preserved")
    data={"schema":1,"branch":"investigation/g1-c7-selective-ln1-dsp-feasibility","base":initial["base"],"host_source_commit":identity["source_commit"],"plan_sha256":identity["plan_sha256"],"selected":"S2","classifications":["SELECTIVE_LN1_MATRIX_VERIFIED","SELECTIVE_LN1_JOINT_CONTRACT_PASS","C7_HOST_REFERENCE_REPRODUCED","MINIMAL_HOST_PRECISION_SUBSET_IDENTIFIED","LN1_SAVED_STATE_CONTRACT_CHARACTERIZED","QKV_MIXED_PRECISION_ABI_DESIGNED","DSP_DOUBLE_CODEGEN_AUDITED","BLOCKED_DEVICE_SAFETY"],"gates_unchanged":GATES,"original_baseline":{"status":"FAIL","expected_exit":4,"attribution_sha256":legacyhash,"failed_predicates":oldfail},"analyses":analyses,"actual_old_ln_backward_host":saved,"saved_test_provenance":saved_id,"processes":processes,"codegen":codegen(root),"cost_model":[{"candidate":"S"+str(i),**cost(m)} for i,m in enumerate(MASKS)],"regression":profiles,"safety":{"status":safety["status"],"reason":safety["reason"],"device_executes":0},"device":{"prototype":"NOT_IMPLEMENTED","accuracy":"NOT_MEASURED","performance":"NOT_MEASURED","new_package_hvx_w8":"NOT_RUN","cumulative_p":"UNRESOLVED"},"previous_device_tested_source":initial["previous_build"]["source_sha"],"previous_binary_hashes":initial["previous_build"]["binary_hashes"],"protected_hashes_unchanged":True,"training":"NO-GO","actual_training_matched":"UNAVAILABLE"}
    finite(data);encoded=json.dumps(data,ensure_ascii=False,indent=2)+"\n"
    require(not re.search(r"[A-Za-z]:[\\/]|/data/|/Users/|device_serial|device_private|endpoint",encoded,re.I),"private path/identifier leak")
    return encoded

def selftest():
    cases=[{"forward":i,"head":h,"p_maxAbs":1e-7,"row_error":0} for i in range(16) for h in range(2)]
    pairs=[{"forward":i,"head":h,"gradient":g,"ds_bound":.0003,"dv_bound":.0001,"ds_signed":.0002,"dv_signed":.00005,"ds_fp_actual":.0002,"dv_fp_actual":.0001} for i in range(16) for h in range(2) for g in range(8)]
    s={"set":"confirmation","p_maxAbs":1e-7,"row_error":0,"ds_bound":.0003,"dv_bound":.0001,"ds_signed":.0002,"dv_signed":.00005,"ds_actual":.0002,"dv_actual":.0001,"p_fail":0,"row_fail":0,"ds_fail":256,"dv_fail":0,"actual_fail":0,"failures":256,"unique_failed_pairs":32,"joint_pass":False,"pairs":256}
    validate_group(cases,pairs,s);n=0
    for key in ["p_maxAbs","ds_bound","ds_actual","p_fail","ds_fail","failures","unique_failed_pairs","joint_pass","pairs"]:
        bad=copy.deepcopy(s);bad[key]=True if key=="joint_pass" else 999
        try:validate_group(cases,pairs,bad)
        except ValueError:n+=1
        else:raise ValueError("accepted corrupt "+key)
    try:validate_group(cases,pairs[:-1],s)
    except ValueError:n+=1
    else:raise ValueError("accepted missing pair")
    print("allowlist_selftest=PASS rejected_corruptions="+str(n)+" allocation_vs_actual_gate_separate=1")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--private-root");parser.add_argument("--output");parser.add_argument("--selftest",action="store_true");args=parser.parse_args()
    if args.selftest:selftest()
    else:
        repo=pathlib.Path(__file__).resolve().parents[1];root=pathlib.Path(args.private_root).resolve();require(root.is_relative_to(repo/"build/reports"),"private root")
        text=export(root,repo);out=pathlib.Path(args.output).resolve();require(out.parent==repo/"docs/results","public aggregate path")
        out.write_text(text,encoding="utf-8");print("public_selective_ln1=PASS aggregate_only=1")
