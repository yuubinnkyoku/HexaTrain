# Nicopedia HPO: Schedule-v2b lower target boundary

Status: **Schedule-v2b COMPLETE** (seed 1, one selected seed, exact step-8000
256+256 held-out evaluations).

The only varied independent variable was the linear-decay target learning rate.
The validated C2200 constant-LR step-6000 checkpoint was reused as the common
parent for the proxy sweep. A qualifying proxy winner received the one allowed
S4000 full-tail confirmation.

## Motivation and Schedule-v1/v2a evidence

Schedule-v1 showed that linear decay improved the constant-LR anchors. Schedule-v2a
then lowered only the target and improved monotonically in its 2,000-step tail
proxy:

| Target LR | Val256 | Dev256 | Balanced |
| ---: | ---: | ---: | ---: |
| `.0015` | 2.330305757 | 2.600519830 | 2.465412794 |
| `.0010` | 2.313034397 | 2.587311777 | 2.450173087 |
| `.0007` | 2.303790595 | 2.568418868 | 2.436104732 |
| `.0004` | 2.294507954 | 2.559579893 | **2.427043924** |

The v2a S4000 confirmation also improved both splits at target `.0004`:
Val `2.291483826`, Dev `2.560856598`, Balanced `2.426170212`, versus the
S4000 `.0015` anchor Balanced `2.462892624`. v2b therefore extended only the
lower target boundary to `.0002`, `.0001`, and `.0000`.

## Why the tail proxy

Each new target forked the validated C2200 constant-LR step-6000 checkpoint and
ran only steps 6001--8000. Three fresh 8,000-step trials would cost 24,000
steps; the proxy sweep used 6,000 nominal new steps, a 75% proxy saving. This
is a selection mechanism, not an assumption that S6000 ranking is invariant to
decay start. Only the proxy winner received the S4000 full-tail confirmation.

## Fixed configuration

* `V=1024`, `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; 758,528 parameters
* batch `8`, seed `1`, byte-BPE tokenizer
* tokenizer SHA-256 `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* dataset cache `fnv1a64:0c7b2826f5f26fea`; canonical order identity
  `fnv1a64:0e2e15196d851431`
* Adam `beta1=.9`, `beta2=.999`, `eps=1e-8`; gradient clip disabled; weight
  decay `0`
* QAIRT build `2.48.40.260702151143`; QNN HTP backend
* exact step-8000 exposure for every terminal artifact: 2,048,000 target
  tokens, 5,491,256 original UTF-8 bytes, 46,616 chunks, 1,949 articles

The global step is 1-indexed:

```text
step <= 6000: lr = 0.0022
6000 < step <= 8000:
    p = (step - 6000) / 2000
    lr = 0.0022 + p * (target - 0.0022)
```

For T0000, the existing formula naturally produces `lr=0` at step 8000. No
epsilon LR and no optimizer-skip special case were introduced.

## Common parent and explicit fork

The common parent was the validated C2200 constant-LR step-6000 checkpoint:

* step `6000`
* checkpoint SHA-256
  `ee8c122612b832c89e7c6a3c3d360c76f38a28c06f260f1b0a40d969dc309063`
* parameter hash `fnv1a64:bde59040980ad8ce`
* identity `V/T/D/FFN/L/H=1024/32/64/128/19/2`, batch 8, seed 1
* prefix LR `.0022`, inherited Adam m/v state, finite state
* QNN return success, finite outputs, graph failures `0`, fallback `false`

The local artifact revalidated the header, model/data/order/tokenizer identity,
parent hash, optimizer state count, and host finite probe. Each child manifest
records the explicit fork flag, parent trial/hash/step, peak and target LR,
decay bounds/type, and model/data/optimizer identity. Cross-target resume is
rejected.

## Proxy targets and step-8000 results

T0400 is the existing v2a artifact and was not retrained. Negative deltas are
improvements relative to the `.0004` proxy anchor.

| Target LR | Val256 | Dev256 | Balanced | Δ vs `.0004` |
| ---: | ---: | ---: | ---: | ---: |
| `.0004` (existing) | 2.294507954 | 2.559579893 | 2.427043924 | 0 |
| `.0002` | 2.288719170 | 2.550702739 | 2.419710955 | -0.007332969 |
| `.0001` | **2.286752948** | **2.543557534** | **2.415155241** | **-0.011888683** |
| `.0000` | 2.287054128 | 2.548642313 | 2.417848221 | -0.009195703 |

Proxy winner: **target `.0001`**. It improved Val and Dev versus `.0004`; the
margin is greater than 0.005 bpb in this seed-1 run.

Runtime telemetry was checked at steps 6000, 6250, 6500, 6750, 7000, 7250,
7500, 7750, and 8000 for every new candidate. Actual-vs-expected differences
were within the 2e-8 fail-closed tolerance. T0000 recorded exactly zero at
step 8000 while retaining valid Adam state, graph execution, tensors, and
checkpoint health.

## Full-tail confirmation

The proxy winner was confirmed from the validated C2200 step-4000 parent using
the same linear shape and target replacement:

| S4000 target | Val256 | Dev256 | Balanced |
| ---: | ---: | ---: | ---: |
| `.0004` (existing v2a) | 2.291483826 | 2.560856598 | 2.426170212 |
| `.0001` | **2.284370268** | **2.541861852** | **2.413116060** |

Full-tail delta for `.0001` versus `.0004`: ΔVal `-0.007113558`, ΔDev
`-0.018994746`, ΔBalanced `-0.013054152`. Proxy and full-tail rankings agree:
`.0001` is best in both comparisons.

Two earlier attempts were interrupted by infrastructure limits after valid
partial checkpoints (checkpoint-stall and then the host poll limit). The final
run used the same full-tail child and the validated step-4000 parent; no
cross-target or cross-parent resume was used. These retries are recovery
overhead and are excluded from the nominal compute ledger below.

## Boundary verdict

Target sensitivity is **strong** over the tested range: lowering from `.0004`
to `.0001` improved Balanced by 0.0119 bpb in the proxy and 0.0131 bpb in the
S4000 confirmation. `.0000` was worse than `.0001`, so the observed optimum is
near `.0001`, but the exact lower boundary is not mathematically closed by this
grid. The ledger records `boundary_verdict=inconclusive`; no finer target grid
was added.

This is one seed, not a statistical-significance claim. It is a quality result
under the fixed configuration and evaluation protocol.

## Compute accounting

| Quantity | Value |
| --- | ---: |
| naive fresh comparison | 24,000 steps |
| nominal new training | 10,000 steps (6,000 proxy + 4,000 confirmation) |
| reused parent prefixes | 22,000 steps (18,000 proxy parents + 4,000 S4000 parent) |
| nominal saved vs fresh | 14,000 steps |
| nominal saving | 58.33% |

The terminal training reports record 42,722.3 seconds in aggregate across
T0200, the resumed T0100 segment, T0000, and the successful S4000 confirmation.
This excludes failed/retried device work and host finalization time; the ledger
therefore reports the nominal step saving, not a claim about elapsed wall time.

## Systems and safety

All terminal proxy training and full-tail/eval artifacts report:

* QNN return code success and finite tensors independently verified
* graph execute failures `0`, `cpu_fallback=false`, fallback attempted/succeeded `false`
* QAIRT runtime build `v2.48.40.260702151143`, backend `HTP`
* thermal status `0` during observed runs; battery/CPU temperature and device
  scheduling varied over the long run and are treated as system noise, not as
  a target-LR effect

Execution was single-flight. Transport interruptions were handled by
reconnect, status inspection, and checkpoint/header/hash validation. No broad
force-stop, app-data deletion, endpoint/serial disclosure, parallel trial, or
cross-target resume was used. The only stop issued during recovery targeted the
exact owned duplicate run after verifying its run id.

## Verification

* PowerShell parser check: PASS
* `run_nicopedia_hpo_schedule_v2b.ps1 -SelfTest` with explicit fixed QAIRT:
  PASS
* zero-target endpoint test and host tests: PASS
* fixed-QAIRT Android build and APK audit: PASS
* `verify_local.ps1 -SkipAndroidBuild`: 25 PASS, 2 intentional Android-build
  SKIP (no native/QNN change after that build)
* final proxy/full-tail manifests, reports, exposures, runtime LR telemetry,
  and 256+256 HTP evaluations: PASS

Full gate, Compose, generation, commit, and push were intentionally not run.

## Verdict and next step

**Schedule-v2b COMPLETE.** The current best target is linear-decay target
`.0001`, confirmed in both the S6000 proxy and the one allowed S4000 full-tail
run. Keep `.0001` fixed for the next cheap decay-start/shape experiment, then
validate the resulting recipe with seed 2. Do not start architecture changes or
an additional fine target grid yet.
