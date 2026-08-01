# First nonfinite results

Public, aggregate-only evidence for the FFN372 LayerNorm2 square boundary and its semantics-preserving scale correction. See `manifest.json` for status and the CSV files for evidence.

The five-seed post-fix formal is now complete for both FFN372 (5/5 finite, oracle 20/20, free 20/20) and L19 (5/5 finite; the pre-fix seed-2 final-evaluation nonfinite did not recur). L19 generation quality did not meet the device-side exact-rollout threshold (13/20 oracle, 13/20 free) while staying numerically finite; that is a learning-quality depth question, not a numeric failure. A one-seed L64 smoke also completed 320 steps finite at the corrected scale.

Formals run one seed per Android process with resumable storage: each seed result is pulled immediately, integrity-hashed and atomically promoted on the host, and only identity-matching completed results are reused on resume. Run ids, ADB endpoints, APK hashes and device identifiers remain private and are not exported.
