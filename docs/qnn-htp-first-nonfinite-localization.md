# QNN HTP first-nonfinite localization

QAIRT 2.48.40.260702 (Build ID `2.48.40.260702151143`) was used throughout. A reproducible application-visible nonfinite boundary occurred during successful QNN HTP execution for B1/T16/V32/D256/FFN372/L3/H4, Adam 0.003, no clipping: seed 1 failed at step 32 in 3/3 unchanged runs. The step-31 checkpoint replayed deterministically.

The same checkpoint produced finite CPU forward/backward/Adam results. HTP replay reproduced the failure 3/3 with QNN return success. Coarse-to-fine taps localized the first nonfinite application-visible tensor to layer 2, LayerNorm2, `ElementWiseMultiply(square)`, output `layer_002_ln2_square`. Its finite input reached 279.75; the CPU float expression produced 78260.0625 while the HTP-path output was positive infinity. Shape, byte count, APP_WRITE/APP_READ roles, full writes, poison count, immutability, alignment and storage non-aliasing were checked. Untapped and scoped-tap final hashes/failure locations agreed (`NO_OBSERVER_EFFECT`). A one-node, shape `[1]` procedural graph reproduced CPU-finite/QNN-success/HTP-nonfinite 3/3.

The application had scaled centered LayerNorm values by `s=8` before squaring. It now uses `s=1`. This is real-number equivalent for positive `s` when variance and epsilon are scaled consistently:

`v_s=s²v`, `epsilon_s=s²epsilon`, and `rsqrt(v_s+epsilon_s)·s = 1/sqrt(v+epsilon)`.

The transform removes the unnecessary 64-fold square range expansion without clipping, changing epsilon, lowering the learning rate, or falling back to CPU. Same-checkpoint post-fix forward/backward/HTP Adam replay was finite and hash-stable 3/3; square maximum was 1223. This is classified `APP_NUMERIC_RANGE_DEFECT`; it does not assert an undocumented internal precision.

The L19 depth observation was corrected: seed 1 was finite 3/3, while seed 2 reproduced a final-evaluation `layer_018_output` nonfinite 3/3 (pre-fix). Post-fix, the five-seed formal is now complete for both FFN372 and L19 at 320 steps, Adam 0.003, no clipping, one seed per Android process with resumable per-seed storage: FFN372 is 5/5 finite with oracle 20/20 and free-running 20/20 exact rollouts; L19 is 5/5 finite and the pre-fix seed-2 final-evaluation nonfinite did not recur, though its device-side exact-rollout thresholds were only partially met (13/20 oracle, 13/20 free) while all steps stayed finite. A one-seed L64 smoke finished 320 steps finite; the pre-fix step-301 failure did not recur at scale 1. Result: `FIRST_NONFINITE_OPERATION_IDENTIFIED_AND_FIXED_FORMALLY_VALIDATED`. L2/H2 and L6/H8 seed-1 320-step regressions passed finite with zero nonzero QNN returns.

Raw checkpoints, parameters, Adam state, tensor dumps, logcat, APKs, device identifiers and local paths remain private and are not exported.
