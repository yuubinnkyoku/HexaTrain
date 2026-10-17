"""Strict allow-list export of Host precision research; never certifies DSP."""
import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

IDS = [f"C{i}" for i in range(8)] + ["LN_reduce", "LN_norm", "LN_affine", "QK_sum64", "QK_Kahan", "QK_Neumaier", "LN_QK_Kahan", "Q_sum64", "Q_Kahan", "Q_Neumaier", "Q_QK_Kahan", "B1", "B2", "B3", "B4"]
METRICS = ["p_maxAbs", "row_error", "ds_bound", "dv_bound", "ds_signed", "dv_signed", "ds_actual", "dv_actual"]
COUNTS = ["p_fail", "row_fail", "ds_fail", "dv_fail", "actual_fail"]
GATES = dict(p_maxAbs=1e-6, row_error=2e-6, ds_bound=.00025, dv_bound=.001, ds_actual=.002, dv_actual=.01)
SUMMARY_KEYS = ["candidate", "set", "reference", "pairs"] + METRICS + COUNTS + ["failures", "unique_failed_pairs", "joint_pass"]
FOCUS = {(16, 1), (17, 1), (20, 1), (21, 1), (22, 0), (22, 1)}

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def need(ok, message):
    if not ok: raise ValueError(message)

def fresh():
    return {**{k: 0. for k in METRICS}, **{k: 0 for k in COUNTS}, "pairs": 0, "failed": set()}

def add_case(t, x):
    for k in ("p_maxAbs", "row_error"): t[k] = max(t[k], x[k])
    t["p_fail"] += x["p_maxAbs"] > GATES["p_maxAbs"]
    t["row_fail"] += x["row_error"] > GATES["row_error"]
    if x["p_maxAbs"] > GATES["p_maxAbs"] or x["row_error"] > GATES["row_error"]: t["failed"].add((x["forward"], x["head"]))

def add_pair(t, x):
    t["pairs"] += 1
    for k in ("ds_bound", "dv_bound", "ds_signed", "dv_signed"): t[k] = max(t[k], x[k])
    for k, src in (("ds_actual", "ds_fp_actual"), ("dv_actual", "dv_fp_actual")): t[k] = max(t[k], x[src])
    t["ds_fail"] += x["ds_bound"] > GATES["ds_bound"]
    t["dv_fail"] += x["dv_bound"] > GATES["dv_bound"]
    t["actual_fail"] += x["ds_fp_actual"] > GATES["ds_actual"]
    t["actual_fail"] += x["dv_fp_actual"] > GATES["dv_actual"]
    if x["ds_bound"] > GATES["ds_bound"] or x["dv_bound"] > GATES["dv_bound"] or x["ds_fp_actual"] > GATES["ds_actual"] or x["dv_fp_actual"] > GATES["dv_actual"]: t["failed"].add((x["forward"], x["head"]))

def validate_summary(x, t):
    failures = sum(t[k] for k in COUNTS)
    for k in METRICS + COUNTS + ["pairs"]: need(x[k] == t[k], f"summary mismatch {x['candidate']} {k}")
    need(x["failures"] == failures and x["unique_failed_pairs"] == len(t["failed"]) and x["joint_pass"] is (failures == 0), "joint-contract classification mismatch")

def scan(p):
    totals = defaultdict(fresh); cases = {}; seen = set(); summaries = {}; saved = defaultdict(lambda: defaultdict(float)); boundary = {}; focus = {}; completed = False
    with Path(p).open(encoding="utf-8-sig") as f:
        for line in f:
            x = json.loads(line)
            need(all(not isinstance(v, float) or math.isfinite(v) for v in x.values()), "nonfinite evidence")
            if "candidate" in x: need(x["candidate"] in IDS, "unknown candidate")
            kind = x["kind"]
            if kind in ("case", "pair", "summary"):
                need(x["set"] in ("frozen", "holdout", "same_frozen") and x["reference"] in ("source", "legacy"), "unknown coverage")
                key = (x["candidate"], x["set"], x["reference"])
                if kind == "case":
                    ci = key + (x["forward"], x["head"])
                    need(ci not in cases, "duplicate case")
                    cases[ci] = x; add_case(totals[key], x)
                    if x["set"] == "frozen" and x["reference"] == "source" and (x["forward"], x["head"]) in FOCUS:
                        focus[(x["candidate"], x["forward"], x["head"])] = {k: x[k] for k in ("candidate", "forward", "head", "p_maxAbs", "coordinate")}
                elif kind == "pair":
                    pi = key + (x["forward"], x["head"], x["gradient"])
                    limit = 44 if x["set"] == "frozen" else 16
                    glimit = 25 if x["set"] == "frozen" else 8
                    need(pi not in seen and 0 <= x["forward"] < limit and x["head"] in (0, 1) and 0 <= x["gradient"] < glimit, "pair coverage mismatch")
                    seen.add(pi); add_pair(totals[key], x)
                    if x["set"] == "frozen" and x["reference"] == "source" and x["forward"] < 24 and x["forward"] == x["gradient"]:
                        mk = (x["candidate"], "same_frozen", "source"); add_case(totals[mk], cases[key + (x["forward"], x["head"])]); add_pair(totals[mk], x)
                    fk = (x["candidate"], x["forward"], x["head"])
                    if x["set"] == "frozen" and x["reference"] == "source" and fk in focus and x["gradient"] in (9, 12):
                        focus[fk]["gradient_" + str(x["gradient"])] = {k: x[k] for k in ("ds_bound", "dv_bound", "ds_signed", "ds_fp_actual", "dv_fp_actual")}
                else:
                    need(key not in summaries, "duplicate summary"); validate_summary(x, totals[key]); summaries[key] = {k: x[k] for k in SUMMARY_KEYS}
            elif kind == "saved_ln":
                t = saved[(x["candidate"], x["set"])]
                for k in ("y", "xhat", "inverse", "dx", "dg", "db", "saved_affine_replay_delta"): t[k] = max(t[k], x[k])
                t["cases"] += 1; t["backward_gate_failures"] += not x["representation_backward_gate"]
            elif kind == "boundary":
                boundary[(x["candidate"], x["set"])] = {k: x[k] for k in ("reference_delta_p", "saved_q_rounding", "qk_local_error", "keep_y", "keep_q", "keep_raw")}
            elif kind == "completed": need(x["candidates"] == 23 and not x["dsp_execution"] and not x["actual_training_matched"], "completion mismatch"); completed = True
            else: raise ValueError("unknown evidence record")
    expected = {(i, s, r) for i in IDS for s in ("frozen", "holdout") for r in ("source", "legacy")} | {(i, "same_frozen", "source") for i in IDS}
    need(completed and set(summaries) == expected, "incomplete candidate matrix")
    for (i, s, r), x in summaries.items():
        need(x["pairs"] == (2200 if s == "frozen" else 256 if s == "holdout" else 48), "incomplete Cartesian coverage")
        if s != "same_frozen": need(sum(k[:3] == (i,s,r) for k in cases) == (88 if s == "frozen" else 32), "incomplete case coverage")
    for i in IDS:
        for s, n in (("frozen", 44), ("holdout", 16)): need(saved[(i,s)]["cases"] == n, "incomplete saved-state audit")
    return dict(summaries=summaries, saved=saved, boundary=boundary, focus=list(focus.values()))

def selftest():
    x = {"candidate": "test", **{k: 0. for k in METRICS}, **{k: 0 for k in COUNTS}, "pairs": 0, "failures": 0, "unique_failed_pairs": 0, "joint_pass": True}
    t = fresh(); validate_summary(x, t); n=0
    for k in METRICS + COUNTS + ["pairs", "failures", "unique_failed_pairs", "joint_pass"]:
        bad = copy.deepcopy(x); bad[k] = False if k == "joint_pass" else 1
        try: validate_summary(bad,t)
        except ValueError: n += 1
        else: raise AssertionError("corruption accepted " + k)
    b = {"forward":0,"head":1,"ds_bound":.0003,"dv_bound":0,"ds_signed":.00026,"dv_signed":0,"ds_fp_actual":.0002,"dv_fp_actual":0}
    q=fresh(); add_pair(q,b); need(q["ds_fail"] == 1 and q["actual_fail"] == 0, "budget and actual gates conflated")
    print(f"upstream_public_export_selftest=PASS summary_corruptions_rejected={n} immutable_allocation_check=PASS")

def export(root, out):
    ident=read(root/"final-host/host-identity-private.json")
    need(ident["device_executes"] == 0, "unexpected device claim")
    for name, test in ident["tests"].items():
        need(test["exit_code"] == 0 and digest(root/"final-host"/(name+".exe")) == test["executable_sha256"], "executable identity mismatch")
        for log, h in test["logs"].items(): need(digest(root/"final-host"/log) == h, "log identity mismatch")
    digests=[digest(root/"final-host"/f"matrix-{i}.jsonl") for i in (1,2,3)]
    need(len(set(digests)) == 1, "independent Host process mismatch")
    need(digest(root/"final-host/synthetic-inputs.bin") == ident["synthetic_inputs_sha256"], "holdout identity mismatch")
    initial=read(root/"initial-private.json"); need(initial["plan_sha256"] == ident["plan_sha256"], "prospective plan changed")
    baseline=read(root/"baseline/analysis-identity-private.json")
    need(baseline["tests"]["probability_attribution_test"]["exit_code"] == 0 and baseline["tests"]["probability_reference_test"]["exit_code"] == 4, "original contract not reproduced")
    need(digest(root/"baseline/attribution.jsonl") == "c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7", "old nine failures changed")
    old=[]; paths=defaultdict(lambda:dict(p_maxAbs=0,ds_bound=0,ds_signed=0))
    with (root/"baseline/attribution.jsonl").open(encoding="utf-8-sig") as f:
        for line in f:
            x=json.loads(line)
            if x["kind"] == "failure": old.append({k:x[k] for k in ("forward","head","gradient","predicate","value","threshold")})
            if x["kind"] in ("path","impact"):
                for k in paths[x["path"]]: paths[x["path"]][k]=max(paths[x["path"]][k],x.get(k,0))
    need(len(old) == 9, "old failed predicates")
    models={"legacy_R5_softmax":scan(root/"final-host/matrix-1.jsonl"),"FP32_shift_Host_libm":scan(root/"final-host/matrix-fp32-shift.jsonl")}
    rows={};saved={};boundaries={}
    for model, data in models.items():
        rows[model]=list(data["summaries"].values())
        saved[model]=[{"candidate":k[0],"set":k[1],**v} for k,v in data["saved"].items()]
        boundaries[model]=[{"candidate":k[0],"set":k[1],**v} for k,v in data["boundary"].items()]
        c7=[x for x in rows[model] if x["candidate"] == "C7" and x["set"] in ("frozen","holdout")]
        need(len(c7) == 4 and all(x["joint_pass"] for x in c7), "selected candidate does not pass both references and sets")
    c0=models["legacy_R5_softmax"]["summaries"][("C0","frozen","legacy")]; need(c0["failures"] == 9, "baseline not preserved")
    cons=[json.loads(x) for x in (root/"final-host/consistency.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    need(len(cons)==4 and cons[0]["heads"]==120 and cons[0]["baseline_word_mismatches"]==0 and cons[0]["double_long_double_p_error"]<=1e-10,"independent reference checks")
    floor_id=read(root/"floor-identity-private.json")
    need(floor_id["exit_code"] == 0 and digest(root/"floor.jsonl") == floor_id["log_sha256"] and digest(root/"upstream_precision_floor_test.exe") == floor_id["exe_sha256"],"floor identity mismatch")
    floors=[json.loads(x) for x in (root/"floor.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    expected_floor={(i,s) for i in ("nearest_FP32_P","isolated_LN_Y_storage","isolated_QK_storage","isolated_raw_storage") for s in ("frozen","holdout")}
    need({(x["candidate"],x["set"]) for x in floors} == expected_floor and len(floors)==8,"floor coverage")
    for x in floors:
        need(x["pairs"] == (2200 if x["set"]=="frozen" else 256),"floor pairs")
        need(all(math.isfinite(x[k]) and x[k]<=v for k,v in GATES.items()) and x["joint_pass"] and x["failures"]==0,"floor gates")
    floor_public=[{k:x[k] for k in SUMMARY_KEYS} for x in floors]
    cg=read(root/"codegen/codegen-identity-private.json"); need(cg["status"]=="COMPILE_ONLY_FEASIBILITY" and cg["device_executes"]==0,"compile-only status")
    safety=read(root/"initial-preflight/device-preflight-private.json"); need(safety["read_only"] and safety["device_executes"]==0,"safety record")
    profiles={}
    import re
    for name in ("Fast","Host"):
        log=root/(name.lower()+"-profile.txt"); txt=log.read_text(encoding="utf-8-sig")
        match=re.search(r"total ([0-9.]+)s\s+PASS=([1-9][0-9]*) FAIL=0 SKIP=0\s*$",txt)
        need(match is not None,"incomplete profile "+name);profiles[name]={"status":"PASS","seconds":float(match[1]),"log_sha256":digest(log)}
    doc={"schema":1,"scope":"Host counterfactual research only; no new DSP package or high-precision device execution","base":initial["base"],"branch":"investigation/g1-forward-p-upstream-precision-budget","analysis_source":ident["source_commit"],"preregistered_plan_sha256":ident["plan_sha256"],"immutable_gates":GATES,"old_contract":"FAIL","original_failed_predicates":old,"R0_R5":[{"path":i,**paths[i]} for i in range(6)],"candidates":rows,"focus_cases":models["legacy_R5_softmax"]["focus"],"saved_LN":saved,"boundaries":boundaries,"selected_candidate":"C7","minimum_claim":"Minimum within evaluated stage subsets and named alternatives, not a proof over all algorithms","selection_reason":"All stage outputs remain FP32, no extra cross-boundary state; strict subsets C0-C6 fail the allocation on the immutable Cartesian set","independent_validation":{"kind":"preregistered synthetic; not a new checkpoint-derived set","forward_cases":16,"heads":2,"gradient_cases":8,"pairs":256,"inputs_sha256":ident["synthetic_inputs_sha256"]},"same_frozen_pairs":48,"actual_training_matched":"ACTUAL_TRAINING_MATCHED_UNAVAILABLE","reference_checks":cons,"precision_feasibility":{k:cg[k] for k in ("source_sha256","compiler_sha256","sdk_manifest_sha256","assembly_sha256","llvm_ir_sha256","scalar_dfadd_emitted","scalar_partial_product_df_multiply_emitted","division_helper","ln_sqrt_external","qhpi_float64_enum","generic_qnn_float64_enum","runtime_linkage_verified","runtime_rounding_verified","hardware_latency_measured","status")},"previous_device_package_hashes":initial["previous_build"]["binary_hashes"],"previous_device_tested_source":"65148e19237b162da14042b500892ef5716e9b8d","previous_results_reused_as_new_device_pass":False,"initial_readonly_safety":{"status":safety["status"],"gates":safety["gates"],"device_executes":0},"prototype":{"status":"NOT_IMPLEMENTED_NOT_RUN","reason":"QK-only candidates fail. Winning design changes LN1, Q/K, QK and LN saved-state arithmetic; separate staged contract/linkage/runtime design is needed before one bounded geometry can be certified.","training_ready":False},"validation":profiles,"Host_three_process_numerical_digests":digests,"Host_executable_hashes":{k:v["executable_sha256"] for k,v in ident["tests"].items()},"host_compiler_sha256":ident["compiler_sha256"],"fixture_sha256":ident["fixture_sha256"],"classifications":["UPSTREAM_PRECISION_ATTRIBUTION_VERIFIED","FP32_BOUNDARY_FLOOR_CHARACTERIZED","MINIMAL_PRECISION_ISLAND_IDENTIFIED","HOST_P_JOINT_CONTRACT_PASS"],"classification_scope":"C7 Host candidate; existing ordered FP32 and previous device cumulative contract remain FAIL","new_Qnn_Device_HVX_W8":"NOT_RUN_NO_NEW_DEVICE_PACKAGE","training":"NO_GO_40k_to_60k_AND_200k"}
    doc["representation_floor"]=floor_public
    doc["representation_floor_limitation"]="Only nearest-FP32 final P establishes an elementwise output representation floor. Isolated upstream boundary paths are counterfactuals, not unavoidable cumulative error floors."
    doc["floor_executable_sha256"]=floor_id["exe_sha256"]
    sdk=(root/"sdk-audit.txt").read_text(encoding="utf-8-sig")
    apk=(root/"apk-audit.txt").read_text(encoding="utf-8-sig")
    need("qnn_core_headers_present=true" in sdk and "qnn_required_phone_libraries_present=true" in sdk and "expected_build_id_match=true" in sdk,"current SDK core audit")
    need("status=SUCCESS" in apk and "forbidden_2_47_strings=false" in apk and "host_sdk_path_present=false" in apk,"current APK audit")
    doc["validation"]["SDK_readonly"]={"status":"CORE_AND_BUILD_ID_PASS_OPTIONAL_INVENTORY_ADVISORY","log_sha256":digest(root/"sdk-audit.txt")}
    doc["validation"]["existing_APK_readonly"]={"status":"PASS","new_build":False,"log_sha256":digest(root/"apk-audit.txt")}
    first=(root/"fast-first-gradle-daemon-failure.txt").read_text(encoding="utf-8-sig")
    need("Could not connect to the Gradle daemon" in first,"Fast initial failure identity")
    doc["Fast_attempts"]=[{"status":"INFRASTRUCTURE_FAILURE","reason":"GRADLE_DAEMON_CONNECTION_FAILURE","log_sha256":digest(root/"fast-first-gradle-daemon-failure.txt")},{"status":"PASS","same_profile_and_conditions":True}]
    doc["sdk_identity"]={"QAIRT":"2.48.40.260702151143","QNN_API":"2.37.0","HTP":"V81","Hexagon_SDK":"6.6.0.0","Hexagon_tools":"19.0.07","NDK":"26.2.11394342","NDK_compiler_sha256":initial["previous_build"]["ndk_compiler_sha256"]}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("upstream_public_export=PASS old_contract=FAIL C7_host_contract=PASS new_device_certification=NOT_RUN")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--private-root",type=Path);p.add_argument("--output",type=Path);p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.private_root is not None and a.output is not None,"paths required");export(a.private_root,a.output)
