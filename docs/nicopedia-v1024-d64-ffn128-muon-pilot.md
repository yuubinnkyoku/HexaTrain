# Nicopedia V1024 / D64 / FFN128 Muon research pilot

## Motivation

This research-only pilot asks whether Muon is promising for HexaTrain's
758,528-parameter model before any QNN/HTP-native Muon implementation is
attempted. The production optimizer remains Adam. The comparison fixes
V1024/T32/D64/FFN128/L19/H2, batch 8, seed, tokenizer, cache, data order, and
article/chunk exposure. The final-test split is closed and is not used.

## Algorithm and reference

The semantic source is KellerJordan/Muon `muon.py` revision
`64560829f70a216f8263ea8ae0fefeac8b451d36` (2025-12-04 UTC); the repository
HEAD observed during the pilot was
`21275da9e7ca53ef5cc8df6780989e366fe75c5e`.

- momentum: `m = beta*m + (1-beta)*g`, with `beta=0.95`
- Nesterov input: `(1-beta)*g + beta*m`
- FP32 Newton--Schulz reference: five iterations with
  `(a,b,c)=(3.4445,-4.7750,2.0315)` and normalization
  `X = G/(||G||F + 1e-7)`
- aspect handling: transpose internally when rows exceed columns, then
  transpose back
- Keller-original update scaling: `sqrt(max(1, fan_out/fan_in))`
- Muon and auxiliary-Adam weight decay: zero

This deliberately does not mix in Moonlight's
`0.2*sqrt(max(fan_out,fan_in))` LR adjustment or its differently scaled
momentum-state convention. Sources: [current Muon implementation](https://github.com/KellerJordan/Muon/blob/64560829f70a216f8263ea8ae0fefeac8b451d36/muon.py),
[original write-up](https://kellerjordan.github.io/posts/muon/), and
[Muon scaling literature](https://arxiv.org/abs/2502.16982).

## Parameter split

Classification is fail-closed and comes from the parameter registry's semantic
role and shape, not substring matching.

- Muon: Wq, Wk, Wv, Wo, FFN W1, and FFN W2 in each of 19 layers; 114 matrices
  and 622,592 parameters.
- Auxiliary Adam: token embedding, LM head/output projection, norm gains and
  offsets, and every non-hidden parameter; 78 registry entries and 135,936
  parameters.
- Total: 192 registry entries and 758,528 parameters, each assigned exactly
  once.

Auxiliary Adam remains beta1=.9, beta2=.999, epsilon=1e-8, weight decay=0.

## Implementation

The CPU reference owns Muon momentum independently from auxiliary Adam m/v and
checks gradient, momentum, normalized input, each NS stage/output, and final
parameters for finite values. Exact-zero gradient with exact-zero momentum is a
no-op. A zero gradient with existing momentum is intentionally a momentum
continuation, matching the source implementation.

The research execution boundary is:

- forward/backward backend: QNN HTP
- Muon optimizer backend: CPU
- auxiliary Adam backend: CPU
- production/default Adam path: unchanged QNN HTP Adam

`cpu_fallback=false` only states that the QNN graph did not fall back; it does
not describe optimizer placement.

## CPU reference validation

`nicopedia_muon_optimizer_test` and `nicopedia_muon_checkpoint_test` pass. The
optimizer test covers 2x2, 2x3, 3x2, 64x64, 128x64, and 64x128 matrices,
deterministic repeated updates, zero/tiny gradients, finite checks, NS
transpose equivalence, aspect-scaled full updates, and the exact semantic
parameter classification. The checkpoint test covers V4 round-trip, finite and
registry validation, wrong-optimizer rejection, Adam/Muon cross-resume
rejection, and bit-identical CPU fresh-8 versus fresh-4/resume-to-8 parity.

## Checkpoint semantics

NPRTCKPTV3 retains its Adam m/v meaning. Mixed Muon state uses a separate
versioned identity and records parameters, per-Muon-matrix momentum, per-aux
parameter Adam m/v, optimizer identity/hyperparameters, global step, tokenizer,
dataset/order cursor, and registry roles/shapes. Adam-to-Muon and Muon-to-Adam
resume are rejected.

## Experimental protocol

The frozen auxiliary-Adam schedule is .0022 through step 4000 and then linear
to .0001 at step 8000. Muon uses the same schedule shape and relative endpoint;
only its peak LR varies.

- Stage A: fresh seed-1 runs at Muon LR .005, .010, and .020 through step 1000.
- Stage B: at most the best two candidates continue to step 2000.
- Promotion: at most one candidate continues, with sparse evaluation at steps
  4000, 6000, and 8000.
- Evaluation: existing fixed Val and Dev samples only; no final-test access.

The verified Adam references are:

| step | Val | Dev | Balanced |
| ---: | ---: | ---: | ---: |
| 1000 | 3.241813540 | 3.544772203 | 3.393292872 |
| 2000 | 2.519542361 | 2.798246775 | 2.658894568 |
| 4000 | 2.717130022 | 3.245737228 | 2.981433625 |
| 8000 | 2.284370268 | 2.541861852 | 2.413116060 |

The matching Adam step-2000 anchor is now verified and evaluated with the same
256+256 protocol used for the recovered Muon endpoint. It is the exact
step-2000 checkpoint from the completed constant-LR `.0022` seed-1 trajectory,
not an inference from the step-4000 or step-8000 result. Its V3 header,
checkpoint SHA-256, parameter hash, tokenizer, cache identity, optimizer
identity, and fixed canonical data-order identity were checked before reuse.

## Stage A

Started on 2026-09-04 after one NX741J/SM8850 physical device was resolved by
matching its two stable identity properties. Each active-run gate was clean:
no test process, foreground service, active Activity, or live heartbeat from a
different run. TCP ADB and the host PC disconnected during the .005 and .010
long runs respectively, but the device-side instrumentation continued to a
terminal `PASSED` state. Recovery accepted results only after the expected run
identity, four interval checkpoints, V4/data identity, parameter hash,
finiteness, QNN return status, and 1000-row LR telemetry all matched.

All three fixed 256+256 Val/Dev evaluations are complete. LR .005 leads LR
.010 by 0.059019711 bpb Balanced; LR .020 is the slowest but remains healthy.

| Muon LR | step | Val | Dev | Balanced | ms/update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| .005 | 1000 | 2.623903850 | 2.903580359 | 2.763742105 | 7,557.222 |
| .010 | 1000 | 2.677503609 | 2.968020022 | 2.822761816 | 7,009.188 |
| .020 | 1000 | 2.767109946 | 3.007209504 | 2.887159725 | 6,502.949 |

The .005 training loss moved from 7.489538193 to 4.937286377; .010 moved from
the identical initial loss to 5.038092136; .020 moved to 5.180727005. All three
1000-step reports executed 8,000 QNN graphs, reported zero QNN failures,
remained finite, and did not fall back.

## Stage B

LR .005 was the best two-candidate Stage-A survivor and was continued on the
same Muon checkpoint. The original 1000->2000 device invocation produced
valid V4 checkpoints at 1250, 1500, and 1750 plus 1000 telemetry rows, but its
instrumentation process crashed before producing a terminal report or step
2000 checkpoint. No QNN/native cause is inferred from that runner crash alone.

The retained step-1750 checkpoint was independently resumed for 250 updates
to step 2000 with a fresh run ID. That recovery report passed its own gates:
2,000 QNN executions, zero QNN failures, finite outputs/state, and no CPU
fallback. The resulting step-2000 checkpoint was V4-decoded and evaluated on
the fixed 256+256 samples:

| Muon LR | endpoint | Val | Dev | Balanced | endpoint evidence |
| ---: | ---: | ---: | ---: | ---: | --- |
| .005 | 2000 | 2.491017807 | 2.756343366 | 2.623680587 | recovered 1750->2000; QNN 2000/2000, finite, fallback=false |

This is a valid recovered endpoint. The original segment still has no terminal
training report, which remains a run-accounting caveat rather than a reason to
invalidate the checkpoint. The matching Adam anchor closes the comparison
gap:

| Optimizer | LR | Val | Dev | Balanced |
| --- | ---: | ---: | ---: | ---: |
| Adam | .0022 | 2.519542361 | 2.798246775 | 2.658894568 |
| Muon | .005 | 2.491017807 | 2.756343366 | 2.623680587 |

Paired deltas are Muon minus Adam: Val `-0.028524554`, Dev `-0.041903409`,
and Balanced `-0.035213982` bpb. Both held-out splits improve, and the
Balanced delta is below the preregistered `-0.01` PROMISING threshold. The
pilot verdict is therefore `PROMISING`; the original Stage-B terminal-report
gap is retained explicitly in the manifest and comparison artifact.

Machine-readable comparison evidence is under
`build/muon-pilot/nicopedia-v1024-d64-f128/comparison/adam-step2000.json` and
`build/muon-pilot/nicopedia-v1024-d64-f128/comparison/muon-vs-adam-step2000.json`.

## Promoted run

The frozen Muon LR `.005` seed-1 survivor was extended from step 2000 through
step 4000 and step 6000 with the same V1024/T32/D64/FFN128/L19/H2, batch 8,
linear-decay schedule (`.0022` through step 4000, target `.0001` at step 8000),
and Muon `beta=.95`, Nesterov, five Newton--Schulz steps. Both segments passed
their training health gates independently:

| endpoint | segment | Val bpb | Dev bpb | Balanced bpb | endpoint evidence |
| ---: | --- | ---: | ---: | ---: | --- |
| 4000 | recovered 2000->4000 | 2.385215485 | 2.652586649 | 2.518901067 | QNN 16,000/16,000, finite, fallback=false |
| 6000 | recovered 4000->6000 | 2.277414560 | 2.561163638 | 2.419289099 | QNN 16,000/16,000, finite, fallback=false |

The step-4000 checkpoint parameter hash is
`fnv1a64:ac92421c4ddd2503`; step 6000 is
`fnv1a64:0ef06bc3058c4fd9`. The matching Adam anchors are 2.981433625 and
2.939889743 Balanced bpb respectively, so the paired Muon deltas are
`-0.462532558` at step 4000 and `-0.520600644` at step 6000. These comparisons
use the same fixed 256+256 HTP evaluation protocol and independently verify QNN
return success, tensor finiteness, and no fallback.

The first 6000->8000 attempt emitted valid telemetry through step 6769 and a
V4 checkpoint at step 6750, then its instrumentation was force-stopped after a
checkpoint stall; no 8000 checkpoint or terminal training report exists. A
fresh 6750->8000 retry reproduced a native-initialization stall and was also
force-stopped by the runner after 3600 seconds. The step-6750 checkpoint is
host-decoded, finite, and has parameter hash
`fnv1a64:7af51ee4a40fd508`, but its fixed 256+256 HTP evaluation likewise
ended in an instrumentation crash after the bounded four-hour poll window, so
no step-6750 quality number is accepted. These are runner/instrumentation
outcomes; they are not evidence of a QNN return-code or tensor-numerical
failure. No seed-2 run was started because the promoted 8000-step endpoint and
its fixed evaluation were not completed.

## Learning curves and efficiency

The primary efficiency outputs are tokens-to-quality and wall-time-to-quality,
including host optimizer and parameter-transfer overhead. Exact step-2000
training-time comparison is not reported because the reused Adam anchor has no
standalone terminal step-2000 training report; no incompatible average is
substituted. No interpolation across evaluation sample identities is allowed.

## Systems overhead

The short 8--32-update smoke records forward/backward QNN time, Muon CPU time,
auxiliary Adam CPU time, parameter-transfer time, total wall time, QNN execute
counts/failures, finite state, and fallback state. No hybrid result is labeled
as fully on HTP.

The two smoke observations were CPU-optimizer dominated:

- LR .010: HTP forward/backward 799.690 ms total, Muon CPU 45,958.278 ms,
  Aux Adam CPU 547.963 ms, total 59,412.504 ms for eight updates.
- LR .005: HTP forward/backward 1,008.058 ms total, Muon CPU 98,028.792 ms,
  Aux Adam CPU 1,135.789 ms, total 126,691.878 ms for eight updates.

Both runs reported 64/64 QNN executes, zero QNN failures, finite tensors,
fallback=false, focus takeover zero, and thermal status zero. The large
between-run CPU timing variation means these are overhead diagnostics, not a
stable performance comparison. `parameter_transfer_ms=0` is the current
host-visible binding telemetry value and is not interpreted as proof that data
movement has zero cost. Consequently no tokens/sec or wall-time advantage is
claimed. The fixed-QAIRT build and APK audit did pass:
QAIRT `2.48.40.260702151143`, arm64-v8a/V81 only, all required runtime hashes
matching the pinned SDK, no 2.47 string, and no host SDK path embedded.

The recovered 1000-step reports make the CPU bottleneck unambiguous: LR .005
recorded 55,931.834 ms total HTP forward/backward, 6,036,188.754 ms Muon CPU,
71,890.536 ms Aux Adam CPU, and 7,557,222.056 ms total update time; LR .010
recorded 57,765.405 ms, 5,559,300.851 ms, 68,648.054 ms, and 7,009,188.481 ms
respectively. These are system-overhead observations, not evidence of an HTP
native Muon implementation. LR .020 recorded 99,016.602 ms HTP
forward/backward, 5,154,571.586 ms Muon
CPU, 62,462.369 ms Aux Adam CPU, and 6,502,948.861 ms total update time.
The recovered .005 continuation recorded 25,225.944 ms HTP forward/backward
and 1,475,501.346 ms Muon CPU over its 250 updates; its report's
`run_completed_steps=250` is intentionally retained rather than rewritten as
a synthetic 1000-step report.

## Verification

- fixed-QAIRT `assembleDebug` and `assembleDebugAndroidTest`: PASS
- QNN APK ABI/hash/path/2.47 audit: PASS
- Muon optimizer host test: PASS
- Muon V4 checkpoint/resume host test: PASS
- all C++ host tests, including the graph shape validator: PASS
- Muon research runner parser/self-test and plan generation: PASS
- QNN training runner self-test: PASS
- `verify_local.ps1 -SkipAndroidBuild`: 24 PASS, 1 FAIL, 2 SKIP; its only
  failure is the pre-existing public evidence exporter rejecting the changed
  `tiny_language_model_cpu.cpp` source hash. The regenerated private anchors,
  JVM tests, and all host tests passed. A public-bundle evidence refresh is not
  part of this uncommitted research pilot.
- physical-device hybrid smoke: PASS for LR .010 and .005
- physical-device fresh 8 versus fresh 4/resume-to-8: PASS; step-4 and step-8
  NPRTCKPTV4 files were byte-identical between trajectories
- Stage A LR .005 and .010 1000-step training/evaluation: PASS
- Stage A LR .020 smoke and 1000-step training/evaluation: PASS
- Stage B LR .005 endpoint recovery 1750->2000 and fixed 256+256 evaluation:
  PASS for endpoint/QNN/finiteness evidence; original terminal-report gap
  retained as a run-accounting caveat
- Promoted Muon 2000->4000 and 4000->6000 training/evaluation: PASS for both
  endpoints; QNN return success and tensor finiteness are independently
  verified
- Promoted 6000->8000 first attempt: instrumentation/runner force-stop after
  valid step-6750 checkpoint and telemetry through step 6769; no numeric or
  QNN cause inferred
- Promoted 6750->8000 retry: repeated native-initialization stall and
  runner force-stop after the configured 3600-second checkpoint-stall guard
- Step-6750 fixed 256+256 evaluation: instrumentation crash after the
  four-hour poll window; no quality metric accepted
- Matching Adam step-2000 checkpoint identity and fixed 256+256 HTP evaluation:
  PASS; QNN return success and tensor finiteness are independently verified
- Paired Muon-vs-Adam step-2000 comparison: `PROMISING`

No commit or push was performed. Confirmed completed new training steps through
the three fresh Stage-A 1000-step segments, the recovered .005 250-step
continuation, and the valid promoted 2000->4000->6000 segments. The exact Adam
step-2000 checkpoint and its 256+256 evaluation were reused/evaluated for the
paired comparison; no new Adam training was needed. No final-test input was
opened. The promoted 8000-step endpoint remains unavailable because both
6000->8000 attempts terminated in runner/instrumentation failure before a
terminal checkpoint/report.

## Caveats

- FP32 reference behavior is intentional and differs from the source's BF16
  accelerator implementation while retaining the same algorithmic formula.
- NS transpose equivalence is tested before aspect scaling. The complete
  Keller-original update is not transpose invariant because its LR adjustment
  is directional.
- Seed 1 can establish only `PROMISING`, `FLAT`, or `NEGATIVE`; confirmation
  requires the frozen recipe to pass a paired seed-2 run.

## Step6750 stall isolation and 8000 recovery (2026-09-08)

Checkpoint audit (host-only, throwaway `build/ckpt_audit_muon.cpp`): step 6000
(`eebfd9b4...`), step 6750 (`b9bb9522...`), and step 4000 all decode PASS as
`NPRTCKPTV4`/`muon_aux_adam`, with a stable layout (192 entries: 114 Muon /
78 aux, 758,528 parameters, identical 6,622,141 bytes), all parameter, Muon
momentum, and Aux Adam m/v finite, and `validateCheckpoint` PASS. No step-6500
checkpoint exists. Step 6750 carries record_index 54000, exposed 1,728,000
tokens, order seed 20260806, and the frozen schedule
(Muon .005 / aux .0022 peak, linear decay 4000->8000, aux target .0001).
Parser failure is ruled out; step 6750 is a complete verified checkpoint and a
valid resume source.

Fresh-process isolation on the idle device: 6750->6751 one-update smoke PASS
in 39 s (new checkpoint `1e345e8b...` verified; aux LR at 6751 exactly the
linear-decay value, Muon scaled proportionally), 6000->6001 PASS in 35 s, and
a 4+4-window HTP eval smoke on the same 6750 bytes PASS in 24 s
(param hash `fnv1a64:7af51ee4a40fd508`). Classification is therefore C
(device/backend residual/transient state): checkpoint-content (A) and
instrumentation-only (D) causes are ruled out by evidence, and no
deterministic native bug reproduced, so no native code was changed per the
minimal-fix policy. The V4 decode path was statically audited and is
allocation-bounded with fail-closed errors.

Recovery `muon-recover-8000-r6750` resumed from verified step 6750 with the
unchanged frozen recipe and reached step 8000 (1250 new updates, 1250-row LR
telemetry fully schedule-verified). Mid-run the USB ADB endpoint vanished
(host `ADB_COMMAND_FAILURE`; a transport interruption, not a numeric
failure); the device-side run continued, and the host reattached over TCP
after re-verifying the stable identity `324753221196`, polling the same run
to terminal and performing the same pulls/assertions into the same recovery
directory. Checkpoints 7000/7250/7500/7750/8000 pulled and host-decode
verified. Step 8000 (`2c2b05c6...`) is step=8000, seed=1, full Muon/AuxAdam
state, finite, with record_index 64000 and exposed 2,048,000 tokens.
Report gates: QNN return success, finite tensors, fallback=false, zero graph
failures, HTP forward/backward with CPU Muon/AuxAdam. Systems: HTP fwd/bwd
64,364 ms total (51.49 ms/update), Muon CPU 7,107,303 ms (5,685.84
ms/update), Aux Adam 13,054 ms (10.44 ms/update), total 8,955,786 ms.

Fixed 256+256 HTP evaluation of step 8000 (`muon-diag-eval8000-1`):
Val 2.207289387, Dev 2.501556184, Balanced 2.354422786. Against frozen Adam
seed 1 at 8000 (Val 2.284370268, Dev 2.541861852, Balanced 2.413116060):
dVal -0.077080881, dDev -0.040305668, dBalanced -0.058693274 (Muon better).
Muon step curve (Balanced): 2000 2.623680587, 4000 2.518901067, 6000
2.419289099, 8000 2.354422786. Muon at 6000 remains +0.006173039 above Adam
at 8000, and the first observed Muon checkpoint at or below Adam-8000 quality
is step 8000 itself, so there is no fewer-tokens arrival; STRONG is via the
final -0.0587 bpb margin with both splits improved.

Machine-readable evidence: `build/muon-pilot/nicopedia-v1024-d64-f128/`
`reliability/{checkpoint-audit,resume-smoke,recovery,qairt-pin}.json` plus
`recovery-8000-r6750/`, `reliability/resume-smoke-{6000,6750}/`, and
`reliability/eval-{smoke-6750,8000}/`. Existing trial artifacts were not
modified; recovery ran in separate directories with new run IDs. No 6750
quality number was ever accepted (that eval never completed). Final_test
remains closed. Seed 2 was not started.

## Verdict

`MUON SEED1 STRONG`. Muon seed 1 at 8000 reaches Balanced 2.354422786 against
frozen Adam seed 1 at 8000 Balanced 2.413116060 (dBalanced -0.058693274, both
splits improved), satisfying the preregistered `-0.01` STRONG threshold via
final margin. Earlier Muon endpoints beat their matching Adam anchors by
`-0.462532558` balanced bpb at step 4000 and `-0.520600644` at step 6000.
The original Stage-B terminal-report gap is retained as a run-accounting
caveat. Seed-2 confirmation is still required before any production claim.

## Next step

Seed-2 confirmation recommended: freeze the Muon recipe and run the paired
seed-2 confirmation as a separate task. Do not add LR candidates, retune
momentum/NS/architecture, or open final_test. This pilot does not implement
QNN Newton--Schulz, batched NS, HMX tuning, or an all-HTP optimizer.

## Status update (later)

This CPU Muon pilot is historical reference only. Production-quality
promotion used the HVX W8 path with two 8000-step seeds and is recorded in
`docs/nicopedia-hvx-muon.md` under "Formal baseline freeze (HVX Muon)".
Do not treat this pilot's CPU-backend wall times or single-seed quality as
the current formal baseline.
