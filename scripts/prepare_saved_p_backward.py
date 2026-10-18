"""Host-only provenance audit and exclusive preregistration; never runs ADB."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "build/reports/g1-native-saved-p-backward-dsp"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    base = git("rev-parse", "be4271e")
    forward = git("rev-parse", "e05f353")
    assert base == "be4271efd1b8fb71f0c66f3ae7c35f04571aa78e"
    assert forward == "e05f3539e2e0c0e461157a3abf1d9833d74b392b"
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", forward, base], check=True)
    bundles = {}
    for name in ("fp32-custom", "fp32-selective-ln1", "fp32-mixed-qkv", "fp32-double-qk"):
        manifest = json.loads((ROOT / "build" / name / "build-identity.json").read_text(encoding="utf-8-sig"))
        for path, digest in manifest["source_hashes"].items():
            assert sha(ROOT / path) == digest, ("SOURCE_CHANGED", path)
        for path, digest in manifest["binary_hashes"].items():
            assert sha(ROOT / "build" / name / path) == digest, ("BINARY_CHANGED", path)
        bundles[name] = manifest
    prior = json.loads((ROOT / "build/reports/g1-s2-double-qk-native-p-closure/initial-private.json").read_text(encoding="utf-8-sig"))
    for path, digest in prior["protected"].items():
        assert sha(ROOT / path) == digest, ("PROTECTED_CHANGED", path)
    fixed = {
        "double_qk_dsp": ("build/fp32-double-qk/libHexaDoubleQkDsp.so", "962a98c6f30e4e8f610abba22f69cd2f08fb216d5ed111586b45d7c73f8991e7"),
        "double_qk_host": ("build/fp32-double-qk/libHexaDoubleQkHost.so", "e2df381e1682ab12e68784ee22ddc468fe553d5cf12c778a167c75990a83ea0a"),
        "forward_probe": ("build/fp32-double-qk/double_qk_probe", "911d66e8b0fc3ba56afd544b5c588f3ef4f0027ee08f92a96b809ee4d44b56f6"),
        "double_fixture": ("build/fp32-double-qk/double-inputs.bin", "f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2"),
        "original_fixture": ("build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin", "388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f"),
    }
    for path, digest in fixed.values():
        assert sha(ROOT / path) == digest
    source = "app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc"
    plan = {
        "schema": 1, "plan_id": "g1-native-saved-p-backward-dsp", "base": base,
        "forward_base": forward, "branch": "investigation/g1-native-saved-p-backward-dsp",
        "production_source_sha256": sha(ROOT / source),
        "protected_forward_identities": {name: digest for name, (_, digest) in fixed.items()},
        "gradient_sha256": sha(OUT / "gradients.bin"),
        "gradient_selection": "Unchanged fullUpstream(original)->reference(fp32=true) 25, syntheticGradients() 8; exact existing FP32 words, exclusive new derivative file; no rescale",
        "geometry": {"T": 32, "D": 64, "H": 2, "Dh": 32, "rank4": [1, 1, 32, 32], "row_major": True},
        "softmax_backward": {"nodes": ["hN_softmax_product:ElementWiseMultiply(dp,P)", "hN_softmax_dot:ReduceSum(product,axis=3,keep_dims=true)", "hN_softmax_centered:ElementWiseSubtract(dp,dot)", "hN_dscores:ElementWiseMultiply(P,centered)"],
            "output_boundary": "dMaskedLogits == historical dScores; no scale applied here", "package": "qti.aisw", "dtype": "FLOAT_32"},
        "dvalue": {"operator": "qti.aisw:MatMul", "transpose0": True, "transpose1": False, "dtype": "FLOAT_32", "formula": "P^T*dContext"},
        "rounding_contract": "Separate FP32 declared tensor boundaries. Production does not specify standard HTP reduction/MatMul internal order or precision. Serial FP32 oracle B is a diagnostic comparison, never a bitwise hardware gate. No FMA/fast-math in Host oracle.",
        "references": {"A": "Independent ideal double P and original frozen gradients", "B": "Recovered separate FP32 product, serial sum, subtraction, product; serial FP32 MatMul", "C": "Host on actual device P (never DSP certification)", "D": "Actual HTP output"},
        "gates": {"p_maxAbs": 1e-6, "p_row_error": 2e-6, "p_induced_ds_bound": 2.5e-4, "p_induced_dv_bound": .001, "actual_dMaskedLogits": .002, "actual_dValue": .01,
            "standalone_softmax_vs_received_input_double": .002, "standalone_dvalue_vs_received_input_double": .01, "observer_final_word_mismatches": 0},
        "local_boundary_comparisons": "Report product/reduction/centered deltas and received-input independent errors. No invented bitwise gate or numerical threshold for unspecified internal reduction order.",
        "coverage": {"core": [44, 2, 25], "core_pairs": 2200, "holdout": [16, 2, 8], "holdout_pairs": 256, "P_predicates": "once per forward/head, separate core/holdout"},
        "modes": ["standalone_softmax", "standalone_dvalue", "forward", "native_softmax", "native_dvalue", "dual"],
        "observer_conditions": ["Observed+Tap", "Observed+NoTap", "NoReceipt+Tap", "NoReceipt+NoTap"],
        "ownership": "One existing two-output ForwardSoftmax node, one P output ID per head; product/final-product/dValue share that ID. NoTap native P, no graph input, no Host reinjection, no duplicate producer. Full graph context retained until consumers finish; native mutation/lifetime requires device evidence.",
        "observability_limit": "Standard QNN primitives have no received-word receipts. Tap can expose P/intermediates but cannot independently prove an invisible read. Do not substitute existing fused Custom SoftmaxBackward or add arithmetic kernels for receipts.",
        "sequence": "Standalone gate before native. Standalone FAIL blocks native certification. Pilot separate from final cohort. Stop device on unsafe preflight, preserve Host progress.",
        "digest": "SHA256 of canonical sorted numerical records and output word hashes keyed by mode/observer/forward/head/gradient; exclude timing,counters,IDs,timestamps. All required coverage independently checked before three-process claim.",
        "processes": 3,
        "performance": "Only after accuracy: same final bundle, common Forward/LN-control nodes, first+finalize, discard warmup,31 steady samples median/p10/p90; no-receipt internal timers null; memory/node/tensor inventory; no layer extrapolation.",
        "negative_controls": ["C0 nine FAIL", "old device cumulative P 36 FAIL", "standard Scale known FAIL", "standard Mask 176 predicate FAIL", "standard Softmax 224 local FAIL", "35/40 Tap REPRODUCED/CAUSE UNRESOLVED", "6031 UNRESOLVED/DORMANT/WATCH"],
        "failure_classes": ["BLOCKED_SOFTMAX_BACKWARD_ACCURACY", "BLOCKED_DVALUE_ACCURACY", "BLOCKED_NATIVE_P_FANOUT", "BLOCKED_OBSERVER_INDEPENDENCE", "BLOCKED_DEVICE_SAFETY", "UNRESOLVED"],
        "safety": "Existing research monitoring; ordinary battery/temp/no external power recorded, critical/emergency protection retained. Fresh identity/awake/top/no-training/no-QNN/APK/SDK/storage/protected checks per process. No wake/unlock/install/force-stop/root/power changes.",
        "toolchain": {"qairt": "2.48.40.260702151143", "api": "2.37.0", "hexagon_sdk": "6.6.0.0", "tools": "19.0.07", "arch": "V81", "ndk": "26.2.11394342", "host": "g++ 15.2.0", "host_compiler_sha256": sha(pathlib.Path("C:/msys64/ucrt64/bin/g++.exe"))},
        "training": "NO-GO", "full_attention_or_layer_certification": False,
    }
    # Compiler location is inspected locally, never exported as data.
    path = ROOT / "host_tests/fp32_custom/saved_p_backward_device_plan.json"
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(plan, f, indent=2, sort_keys=True); f.write("\n")
    private = {"root": str(ROOT), "base": base, "head": git("rev-parse", "HEAD"),
               "git_dir": git("rev-parse", "--absolute-git-dir"), "bundles": bundles,
               "protected": prior["protected"], "refs": git("show-ref"),
               "plan_sha256": sha(path), "device_arithmetic_executed": False}
    with (OUT / "initial-private.json").open("x", encoding="utf-8") as f:
        json.dump(private, f, indent=2)
    print("saved_p_backward_preregistration=PASS protected=PASS forward_sources_binaries=UNCHANGED device_arithmetic=0")


if __name__ == "__main__":
    main()
