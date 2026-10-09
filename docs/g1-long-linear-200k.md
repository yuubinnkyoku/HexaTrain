# HexaTrain G1 Long-Horizon Linear Decay, 200k

**Status: BLOCKED BEFORE START — preflight and the 8k fresh sanity passed; the primary trajectory has not started because the required storage headroom gate is unmet.**

This is a fresh-initialization schedule comparison with a hard ceiling of 200,000 updates. The independent variable is the learning-rate schedule. No quality result may change the schedule or stop the run early. A single seed describes this trajectory and does not establish statistical generalization.

## Source and preserved baseline

- New branch: `experiment/g1-long-linear-200k`.
- Source branch: `experiment/g1-expanded-data-100k`.
- User-specified 100k source SHA: `a3e3ce3339914d00a59673af29357f5ce180f209`.
- Source branch tip when this work began: `712a3495d3f18cc5ed66979b0f0d8266073e1fb5` (the specified SHA is an ancestor; the tip adds the verified 100k gates and trajectory figure).
- 100k experiment implementation commit: `5730d735c3a32604c16d132914397f2d56c01caf`.
- Exact experiment tooling source SHA used for preflight and sanity: `55a26b4fba486f3107ab97d7a22bc4e460813795`.
- The original 100k branch, working tree, run plan, cache, checkpoints, reports, and telemetry are preserved. The 200k run uses a separate output root and a byte-verified copy of the original cache inputs.
- Original result: `SUCCESS / STILL_IMPROVING`, seed 1, 100,000 updates, 25,600,000 target tokens.
- Final 100k baseline: Val BPB 2.026486192, Dev BPB 2.135994049, Val NLL 3.728428748, Dev NLL 3.705932880.
- Fresh initialization anchor: `fnv1a64:01b81108b8c8b842`.

## Fixed model, optimizer, and data

- G1 `headwise_g1_sigmoid`; V1024 / T32 / D64 / FFN128 / L19 / H2; 760,960 parameters; batch 8; initialization seed 1.
- Original Muon, `HVX_W8`, peak LR 0.0075, terminal LR 0.0003409090909090909, momentum 0.95, Nesterov enabled, Newton–Schulz 5 iterations.
- Auxiliary Adam, peak LR 0.0033, terminal LR 0.00015, beta1 0.9, beta2 0.999, epsilon `1e-8`.
- Weight decay 0; gradient clipping disabled. Auxiliary Adam update remains on CPU; HTP executes forward/backward and HVX executes Muon. This is not an NPU-only claim.
- Tokenizer: V1024 byte-BPE, SHA-256 `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`.
- Full-train cache: 407,726,880 target BPE tokens; 12,741,465 T32 records; 942,868,468 bytes; SHA-256 `bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf`.
- Cache manifest SHA-256: `e8a77e06abab3da5ff7fe26eb9744c3bffad8cf5fbf006bbf4c7f582393961e4`.
- Source aggregate SHA-256: `b3185ea689ffa64c71f5ea8fe25f3779dbf1f0cd0d103b6e2fe05792e635bf86`.
- Validation cache: 26,652 records, SHA-256 `fd3fce2f61837435338127f165c15137051acaf8edf39d18f45c762bd310fd4f`.
- Development cache: 25,952 records, SHA-256 `a74bc959328fc6602d87f7e420ae9c4cb771084b8965c3d37e01e824ee3f028b`.
- Training order is `SplitMix64(global_selection_index + state) modulo record_count; with replacement`, order seed `20260806`. The legacy first 800,000 selection hash remains `fnv1a64:be3269f541236dc2`. The deterministic 200k extension contains 1,600,000 selections and hashes to `fnv1a64:83f9c96cf10261a0`; the 100k prefix is checked against the original manifest.
- The cache, tokenizer, and held-out files were copied without re-tokenization or reselection. Their destination SHA-256 values match the original artifacts above. The original 100k cache and reports remain untouched.
- Final test is excluded from training and quality evaluation. `6031` remains `UNRESOLVED / DORMANT / WATCH`.

## Frozen schedule

Both optimizers use the same linear multiplier and the existing 1-indexed global optimizer-step convention:

```text
LR(s) = peak + (terminal - peak) * clamp((s - 4000) / 196000, 0, 1)
```

Steps 1–4,000 receive peak LR. Step 4,001 begins decay. Step 200,000 receives the exact terminal LR; later steps clamp to terminal but are outside this run. At 100,000 the multiplier is `287/539` (53.246753% of peak): Aux Adam LR 0.001757142857 and Muon LR 0.003993506494. At 200,000 the multiplier is `1/22` (4.545455% of peak). Boundary and LR telemetry checks cover both optimizer columns. The schedule is frozen before any quality result is inspected.

## Preregistered execution and evaluation

- Fresh initialization for both the sanity trajectory and the primary trajectory. Sanity checkpoints are never used to initialize the primary run.
- Resume-equivalence gate: compare uninterrupted 0→2,000 with fresh 0→1,000→checkpoint/resume→2,000 under the exact 200k schedule. Checkpoint bytes, optimizer state, data cursor, and loss/order evidence must agree before primary training.
- Fresh sanity: 0→1,000→resume→8,000. Inspect QNN status, tensor finiteness, HVX status/fallbacks, optimizer and gate health, checkpoint identity, order identity, and both LR telemetry columns. Sanity quality is not used for HPO.
- Primary segment targets: 1,000; 5,000; then 10,000 through 200,000 in 5,000-update increments. Checkpoints and health telemetry are collected every 1,000 updates. Quality-based early stopping is prohibited.
- Fixed full-cap Val/Dev evaluation boundaries: 1k, 2k, 4k, 8k, 16k, 32k, 50k, 64k, 80k, 100k, 120k, 140k, 160k, 180k, 200k. The first ten reuse the baseline's exact full-cap boundary plan. Each evaluation uses all 26,652 Val and 25,952 Dev records and the same cache, tokenizer, and calculation.
- At 100k: 25,600,000 target tokens. At 200k: 51,200,000 target tokens. Final test is not run.
- Pinned runtime: QAIRT `2.48.40.260702151143`; no fallback, mixed SDK, or SDK substitution.

## Preflight gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Source branch, clean state, baseline preservation | PASS | New branch starts from the 100k branch history; original 100k checkout is clean and unchanged. |
| Full cache SHA and cache identity | PASS | Full cache, tokenizer, Val, and Dev SHA-256 values match the source manifest; 100k order prefix matches. |
| Extended 200k order hash | PASS | Independent SplitMix64/FNV implementation returns `fnv1a64:83f9c96cf10261a0`; 100k prefix remains `fnv1a64:be3269f541236dc2`. |
| LR boundary/unit tests | PASS | Boundaries 4k, 4,001, 100k, 199,999, 200k, and post-end clamp pass; Muon uses the same decay multiplier as Aux Adam. |
| Checkpoint codec and 200k cursor representation | PASS | Host round-trip covers global step 200k, 1,600,000 selections, 51,200,000 exposed tokens, and the 200k schedule identity. |
| Fast / Host / Android | PASS | `verify.ps1` Fast, Host, and Android profiles passed. |
| Pinned QNN build and APK audit | PASS | QAIRT `2.48.40.260702151143`; QNN APK audit succeeded with required library hashes matching and no forbidden 2.47 strings or host SDK paths. The QNN profile reported optional SDK samples as `QAIRT_INVENTORY_INCOMPLETE`; this is advisory in the repository checker and did not fail the pinned phone-build audit. |
| QNN+HVX build and audit | PASS | Fresh debug and androidTest APKs built with Hexagon SDK 6.6.0.0 / Tools 19.0.07. Audited training APK SHA-256: `4cebd539f89e46df7d72cfb4f654d5af072dd0edddc9d0f15171d2f54a49f4bc`. SDK manifest SHA-256: `8bb7a783a59b7b929355f643e78580addf80bbc1e5f4762c244ce2e06be9478f`; compiler SHA-256: `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`. |
| Device smoke and G1 health | PASS | Device profile passed. Stable physical device identity and clean active-run gates were confirmed; sanity used HTP with HVX_W8, Aux Adam on CPU, no fallback, and pinned QAIRT. Thermal status stayed 0; observed battery temperature was 31–38 °C. |
| Resume equivalence | PASS | Fresh uninterrupted 0→2k and segmented 0→1k→resume→2k produced byte-identical step-2k checkpoints (SHA-256 `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d`). |
| Fresh 8k sanity | PASS | Fresh 0→1k→resume→8k reached step 8k with matching initialization, cache, order, QNN, finite, HVX, and checkpoint evidence. Details below. |
| Primary storage headroom | **BLOCKED** | The run requires at least 20 GiB free before each segment; measured free space was 19.71 GiB. No primary update or evaluation was started. |

The primary run must not start if any data, optimizer, LR, order, checkpoint, resume, QNN/HVX, or storage gate fails. All technical preflight and sanity gates passed, but the unmet storage gate keeps the primary run blocked.

## Primary result

**Status: NOT STARTED — blocked before step 1 by storage headroom.** No production checkpoint, segment, or full-cap evaluation exists. The sanity and resume-equivalence trajectories are separate fresh runs and are not part of the primary trajectory.

### Resume-equivalence and sanity evidence

| Check | Result |
| --- | --- |
| Uninterrupted 0→2,000 vs segmented 0→1,000→resume→2,000 | PASS; final checkpoint SHA-256 matched exactly: `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d` (6,652,678 bytes). |
| Fresh sanity initialization | `fnv1a64:01b81108b8c8b842`, matching the preserved 100k initialization. |
| Cache and order | Cache SHA-256 matches the registered cache; 200k order hash `fnv1a64:83f9c96cf10261a0`. |
| Sanity checkpoints | Step 1,000 SHA-256 `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b`; step 8,000 SHA-256 `d3e983c3898ad3bcd7c5e2e8fc03d1e34cdec92e361d43447555f7e0a37c9`. |
| Runtime correctness through step 8,000 | QNN return-code success and tensor/parameter finiteness both passed; Muon backend `HVX_W8`; HVX RPC status 0; HVX output finite; fallback false; CPU fallback false; clipping disabled and clipped steps 0. Aux Adam remained on CPU. |
| LR telemetry | Steps 1 and 1,000: Aux Adam 0.003299999982 / Muon 0.007499999832. Step 4,001: Aux Adam 0.003299983917 / Muon 0.007499963045. Step 8,000: Aux Adam 0.003235714277 / Muon 0.00735389581. The schedule and both optimizer columns agreed through the 8k segment. |
| Optimizer health | Eight health windows through 8k; invalid windows 0. At 8k, gradient L2 1.041447873, parameter L2 241.2259175, Muon parameter delta 0.4826847443, and Adam parameter delta 0.2645275856; all finite flags passed. |
| Gate health at 8k | Weighted over 19 layers × 2 heads: mean 0.0236041490, stddev 0.0265079508, min 0, max 0.5107421875; 97.5085% below 0.1 and 0% above 0.9. Head means (head 0 / 1) were 0.02382294 / 0.02338536; below-0.1 fractions were 0.98207463 / 0.96809519. Layer means, 1-based: 1 0.02396268, 2 0.00135725, 3 0.01126861, 4 0.01081457, 5 0.01840681, 6 0.01931146, 7 0.03009081, 8 0.01347663, 9 0.09853161, 10 0.00891189, 11 0.00953160, 12 0.01733664, 13 0.02065813, 14 0.03988367, 15 0.02030457, 16 0.02195575, 17 0.02072140, 18 0.02885371, 19 0.03310104. No head or layer had values above 0.9. This is strong near-zero saturation in the sanity window; it does not justify stopping or changing the registered schedule or architecture. |
| Sanity schedule selection | No quality result was used to select or change the schedule. |

Resume-equivalence and sanity work covered separate fresh trajectories totaling 12,000 updates (3.072M target-token exposures across those validation runs); none of these updates count toward the primary 200k trajectory.

Checkpoint SHA-256 by global step (all checkpoints were 6,652,678 bytes):

| Step | SHA-256 | Trajectory |
| ---: | --- | --- |
| 1,000 | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | Uninterrupted, segmented, sanity |
| 2,000 | `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d` | Uninterrupted, segmented, sanity |
| 3,000 | `84072c9a87d736f6b63664daa17e48a5e87c4d60f594d9b9fc0c428477147c53` | Sanity |
| 4,000 | `61a8331d58323bbcba4ba86db6ae01ef6f3f3910f89b1d4ead44d3398db3bd6f` | Sanity |
| 5,000 | `279fa455067cec0d0f7067a4eddfd35c90637e438c77a6114c7098ceef87c360` | Sanity |
| 6,000 | `91806658956a0df4fd1a5117fbdd1a1fef7723115a4ad5d7821a654912d8e444` | Sanity |
| 7,000 | `f8bee9c6432e5eb8617fee634f11bb260f5dfa8fc66ceb1839e83798b745e6d7` | Sanity |
| 8,000 | `d3e983c3898ad3bcd7c5e2e8fc03d1e34cdec92e361d43447555f7e0a37c9` | Sanity |

### Segment and checkpoint history

| Trajectory segment | Status | End checkpoint SHA-256 | Target tokens in segment | Wall time | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Resume-equivalence: 0→1k uninterrupted | SUCCESS | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | 256,000 | — | Same fresh initialization and schedule. |
| Resume-equivalence: 1k→2k uninterrupted | SUCCESS | `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d` | 256,000 | — | Final 2k checkpoint byte-matches segmented trajectory. |
| Resume-equivalence: 0→1k segmented | SUCCESS | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | 256,000 | — | Separate fresh initialization. |
| Resume-equivalence: 1k→2k resumed | SUCCESS | `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d` | 256,000 | — | Resumed same checkpoint; final bytes match uninterrupted. |
| Sanity: 0→1k | SUCCESS | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | 256,000 | — | Fresh initialization. |
| Sanity: 1k→8k resumed | SUCCESS | `d3e983c3898ad3bcd7c5e2e8fc03d1e34cdec92e361d43447555f7e0a37c9` | 1,792,000 | — | Checkpoint saved each 1k; per-step hashes listed above. |
| Primary: planned 0→1k | BLOCKED before start | — | 0 | — | Required headroom is ≥20 GiB; measured 19.71 GiB. Primary trajectory remains fresh and separate. |

### Fixed full-cap quality

Delta convention: new result minus the 100k baseline at the same 100k step; negative NLL/BPB delta is better. Top-1 delta is new minus baseline. 200k is reported as the final Long Linear result, without claiming that it beats a 100k baseline at matched training duration.

| Step | Val NLL | Val BPB | Val top-1 | Dev NLL | Dev BPB | Dev top-1 | QNN / finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100k baseline | 3.728428748 | 2.026486192 | 0.248132176 | 3.705932880 | 2.135994049 | 0.245884229 | PASS / PASS |
| 100k Long Linear | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |
| 120k | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |
| 140k | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |
| 160k | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |
| 180k | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |
| 200k Long Linear final | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not run |

### Schedule comparison at equal budget

| Measure at 100k / 25.6M target tokens | Baseline | Long Linear | Delta (new − baseline) |
| --- | ---: | ---: | ---: |
| Val BPB | 2.026486192 | Not measured | Not measured |
| Dev BPB | 2.135994049 | Not measured | Not measured |
| Val NLL | 3.728428748 | Not measured | Not measured |
| Dev NLL | 3.705932880 | Not measured | Not measured |
| Time-to-quality | 40.226 h to best measured baseline | Not measured | Not measured |

### Gate, optimizer, and systems telemetry

Primary-boundary telemetry was not produced because the run did not start. Sanity telemetry is recorded above. If resumed, every fixed evaluation boundary must report gate mean/stddev, layer/head distribution, fraction below 0.1, fraction above 0.9, and observed min/max. Report optimizer finite checks, QNN return-code success separately from tensor/parameter finiteness, HVX success/fallbacks, CPU fallback, clipped steps, and both LR values. Gate saturation alone does not stop the run or change architecture.

### Incidents and negative findings

The first setup attempt used a QNN-only APK and failed native initialization with `HVX_MUON_NOT_BUILT` at step 0; zero updates were applied. The preflight was corrected to build and audit the pinned QNN+HVX APK before device use, and the successful r3 resume/sanity gates used that APK. A second attempt correctly built and audited QNN+HVX, but a wrapper check misread CRLF line endings in the audit report; it stopped before installation or training. The wrapper now normalizes line endings, and the final preflight passed.

The primary storage gate then measured 19.71 GiB free against its 20 GiB minimum. A request to remove task-generated build intermediates to reclaim space was rejected by automatic approval review; no deletion occurred and no alternate execution route was tried. This is the current blocking condition. The primary runner must resume only when its storage check passes. If a future transport or infrastructure failure occurs after launch, classify it separately from numerical or QNN/HVX failures and recover only from a verified checkpoint with matching schedule, dataset/order, optimizer, and device identity. No result may be omitted because it is unfavorable.

### Convergence and next research decision

**Current classification: `INCOMPLETE`.** The primary trajectory has no updates or quality measurements, so no statement about the 100k matched-step result, 200k final quality, or late-stage slope is supported. The 8k sanity gate result is not a convergence measurement. The 200k hard ceiling does not presume convergence.

The next action is to resume this registered primary trajectory once the existing ≥20 GiB storage gate passes; do not alter the schedule, data, optimizer, or seed. No recommendation about whether Linear improves 100k quality or whether to run the matched fixed-terminal 200k control can be made until the primary trajectory has results. The strict 200k schedule comparison still requires a separate matched control: continue the 100k baseline checkpoint to 200k while holding terminal LR fixed. Linear 200k versus Baseline 100k alone is not a superiority claim. `6031` remains `UNRESOLVED / DORMANT / WATCH`.
