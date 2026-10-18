"""Allow-list export of the saved-P provenance recovery. No device calls.
Rejects missing keys, extra keys, misclassified fields and unapproved values
by construction; --self-test proves each rejection path."""
import argparse
import hashlib
import json
import pathlib
import re

ALLOWED_CLASSIFICATIONS = {
    "RESEARCH_HISTORY_VERIFIED", "RESEARCH_SOURCE_RECOVERED", "RESEARCH_EVIDENCE_RECOVERED",
    "RESEARCH_FIXTURES_RECOVERED", "HOST_AUDIT_RECONCILED", "HOST_BACKWARD_REFERENCE_SELFTEST_PASS",
    "RECOVERED_SOURCE_REVALIDATION_REQUIRED", "REMOTE_BASE_ONLY_RECOVERED",
    "BLOCKED_RESEARCH_PROVENANCE", "BLOCKED_HOST_TOOLCHAIN", "UNRESOLVED",
}
ALLOWED_COMMIT_STATUS = {
    "FOUND_LOCAL_RESEARCH_CLONE", "FOUND_LOCAL_AND_REMOTE", "FOUND_LOCAL_AUDIT_CLONE",
    "MISSING_ALL_SEARCHED_REPOSITORIES",
}
ALLOWED_FAST_STATUS = {"PASS", "FAIL", "NOT_RUN", "RUNNING"}
TOP_KEYS = {
    "schema", "study", "commits", "ancestry", "fixtures", "artifact_hashes",
    "numerical_evidence", "host_reference_selftest", "toolchain", "fast_profile",
    "device_executed", "protected_state", "classifications",
}
FORBIDDEN_SUBSTRINGS = ("C:\\", "D:\\", "Users\\", "@", "serial", "endpoint", "token=")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scan(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            require(key not in {"path", "abs_path", "local_path", "apk_sha256", "env"},
                    "unapproved field name " + str(key))
            scan(item, path + "/" + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan(item, path + "/" + str(index))
    elif isinstance(value, str):
        for bad in FORBIDDEN_SUBSTRINGS:
            require(bad.lower() not in value.lower(),
                    "forbidden substring %r at %s" % (bad, path))
    elif isinstance(value, float):
        require(value == value and abs(value) != float("inf"), "nonfinite at " + path)


def validate(e):
    require(set(e) == TOP_KEYS,
            "top-level mismatch missing=%s extra=%s" %
            (sorted(TOP_KEYS - set(e)), sorted(set(e) - TOP_KEYS)))
    require(e["schema"] == "g1-saved-p-provenance-recovery-private-v1", "schema")
    require(e["study"] == "g1-saved-p-provenance-recovery", "study")
    require(set(e["commits"]) == {"984bea4", "b2ce127", "e05f353", "024aaef", "e075d8c"},
            "commit set")
    for name, commit in e["commits"].items():
        require(set(commit) == {"status", "found", "full_sha"}, "commit fields " + name)
        require(commit["status"] in ALLOWED_COMMIT_STATUS, "commit status " + name)
        if commit["found"]:
            require(bool(HEX40.match(commit["full_sha"] or "")), "full sha " + name)
        else:
            require(commit["full_sha"] is None, "missing sha must be null " + name)
    require(e["ancestry"], "ancestry required")
    for link in e["ancestry"]:
        require(set(link) == {"ancestor", "descendant", "is_ancestor"}, "ancestry fields")
        require(bool(HEX40.match(link["ancestor"])) and bool(HEX40.match(link["descendant"])),
                "ancestry sha")
    require(set(e["fixtures"]) == {"double_qk", "mixed_qkv", "s2_ln1", "one_layer_inputs"},
            "fixture set")
    for name, fixture in e["fixtures"].items():
        require(set(fixture) == {"sha256", "verified"}, "fixture fields " + name)
        require(bool(HEX64.match(fixture["sha256"])) and fixture["verified"] is True,
                "fixture " + name)
    require(e["artifact_hashes"], "artifact hashes required")
    for name, digest in e["artifact_hashes"].items():
        require(bool(HEX64.match(digest)), "artifact hash " + name)
    require(set(e["numerical_evidence"]) == {"local_digest", "p_assessment_digest"},
            "numerical evidence set")
    for name, digest in e["numerical_evidence"].items():
        require(bool(HEX64.match(digest)), "digest " + name)
    selftest = e["host_reference_selftest"]
    require(selftest["status"] in {"HOST_BACKWARD_REFERENCE_SELFTEST_PASS", "NOT_RUN"},
            "selftest status")
    metrics = ("softmax_fp32_vs_double", "dvalue_fp32_vs_double", "softmax_rowsum")
    if selftest["status"].endswith("PASS"):
        for metric in metrics:
            value = selftest[metric]
            require(isinstance(value, float) and 0.0 <= value <= 1.0e-4,
                    "selftest metric " + metric)
    else:
        require(all(selftest.get(k) is None for k in metrics),
                "selftest metrics must be null when not run")
    toolchain = e["toolchain"]
    require(set(toolchain) == {"compiler", "on_session_path", "resolution_issue",
                               "resolved_for_selftest"}, "toolchain fields")
    require(isinstance(toolchain["on_session_path"], bool) and
            isinstance(toolchain["resolved_for_selftest"], bool), "toolchain flags")
    require(e["fast_profile"]["status"] in ALLOWED_FAST_STATUS, "fast status")
    require(e["device_executed"] is False, "device execution must be false")
    protected = e["protected_state"]
    require(protected["all_preserved"] is True and protected["training"] == "NO-GO",
            "protected state")
    require(bool(HEX64.match(protected["checkpoint_sha256_recorded"])),
            "checkpoint digest record")
    require(set(e["classifications"]) <= ALLOWED_CLASSIFICATIONS and e["classifications"],
            "classifications")


def export(e):
    validate(e)
    public = {
        "schema": "g1-saved-p-provenance-recovery-v1",
        "study": e["study"],
        "commits": {name: {"status": c["status"], "found": c["found"], "full_sha": c["full_sha"]}
                    for name, c in sorted(e["commits"].items())},
        "ancestry": e["ancestry"],
        "fixtures": {name: {"sha256": f["sha256"], "verified": f["verified"]}
                     for name, f in sorted(e["fixtures"].items())},
        "artifact_hashes": dict(sorted(e["artifact_hashes"].items())),
        "numerical_evidence": dict(sorted(e["numerical_evidence"].items())),
        "host_reference_selftest": e["host_reference_selftest"],
        "toolchain": e["toolchain"],
        "fast_profile": e["fast_profile"],
        "device_executed": e["device_executed"],
        "protected_state": e["protected_state"],
        "classifications": sorted(e["classifications"]),
        "overall": "PROVENANCE_RECOVERED"
                    if {"RESEARCH_SOURCE_RECOVERED", "RESEARCH_EVIDENCE_RECOVERED"}
                    <= set(e["classifications"]) else "PROVENANCE_PARTIAL",
    }
    public["summary_digest"] = canonical(public)
    scan(public)
    return json.dumps(public, indent=2, allow_nan=False) + "\n"


def minimal_valid():
    zero40 = "0" * 40
    zero64 = "0" * 64

    def commit(found):
        return {"status": "FOUND_LOCAL_RESEARCH_CLONE" if found
                else "MISSING_ALL_SEARCHED_REPOSITORIES",
                "found": found, "full_sha": zero40 if found else None}

    return {
        "schema": "g1-saved-p-provenance-recovery-private-v1",
        "study": "g1-saved-p-provenance-recovery",
        "commits": {"984bea4": commit(True), "b2ce127": commit(True), "e05f353": commit(True),
                    "024aaef": commit(True), "e075d8c": commit(True)},
        "ancestry": [{"ancestor": zero40, "descendant": zero40, "is_ancestor": True}],
        "fixtures": {name: {"sha256": zero64, "verified": True}
                     for name in ("double_qk", "mixed_qkv", "s2_ln1", "one_layer_inputs")},
        "artifact_hashes": {"probe": zero64},
        "numerical_evidence": {"local_digest": zero64, "p_assessment_digest": zero64},
        "host_reference_selftest": {"status": "HOST_BACKWARD_REFERENCE_SELFTEST_PASS",
                                    "softmax_fp32_vs_double": 4.23e-9,
                                    "dvalue_fp32_vs_double": 5.14e-8,
                                    "softmax_rowsum": 1.23e-8},
        "toolchain": {"compiler": "g++ 15.2.0 MSYS2 ucrt64", "on_session_path": False,
                      "resolution_issue": "PATH_DLL_RESOLUTION_0xC0000135",
                      "resolved_for_selftest": True},
        "fast_profile": {"status": "PASS", "prior_total_seconds": 54.7},
        "device_executed": False,
        "protected_state": {"all_preserved": True, "training": "NO-GO",
                            "checkpoint_sha256_recorded": zero64},
        "classifications": ["RESEARCH_SOURCE_RECOVERED", "RESEARCH_EVIDENCE_RECOVERED"],
    }


def reject(label, mutate):
    data = minimal_valid()
    mutate(data)
    try:
        export(data)
    except ValueError:
        return
    raise ValueError(label + " was not rejected")


def selftest():
    first = export(minimal_valid())
    require(first == export(minimal_valid()), "stable digest")
    reject("extra key", lambda d: d.update(extra=1))
    reject("missing key", lambda d: d.pop("classifications"))
    reject("misclassification", lambda d: d["classifications"].append("FAKE_PASS"))
    reject("absolute path", lambda d: d["toolchain"].update(compiler="C:\\g++"))
    reject("device execution claim", lambda d: d.update(device_executed=True))
    reject("oversized selftest metric",
           lambda d: d["host_reference_selftest"].update(softmax_fp32_vs_double=1.0))
    changed = minimal_valid()
    changed["classifications"] = ["UNRESOLVED"]
    require(first != export(changed), "output identity")
    print("provenance_recovery_exporter_selftest=PASS rejections=6 device_calls=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        repo = pathlib.Path(__file__).resolve().parents[1]
        evidence_path = pathlib.Path(args.evidence).resolve()
        require(evidence_path.is_relative_to(repo / "build" / "reports"),
                "private evidence containment")
        payload = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        output_path = pathlib.Path(args.output).resolve()
        require(output_path.is_relative_to(repo / "docs" / "results"),
                "public output containment")
        output_path.write_text(export(payload), encoding="utf-8")
        print("provenance_recovery_public_export=PASS")
