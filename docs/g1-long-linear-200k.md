# HexaTrain G1 Long-Horizon Linear Decay, 200k

**Status: RUNNING from the verified step-10,000 primary checkpoint — all registered full-cap evaluations through 8,000 passed. Before resume, AC charging was confirmed at 36% battery, 38 °C, thermal status 0; C: had 70,684,344,320 bytes free and device `/data` had 50,243,301,376 bytes free. Both unchanged storage gates pass.**

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

## C: migration and execution environment

- The execution workspace is a separate checkout on C: on branch `experiment/g1-long-linear-200k`. Its source SHA at the first primary update was `024aaef44784355e8e278ef642e2752ae79846bb` and the working tree was clean. The committed report omits the local absolute workspace path.
- The branch was fetched from the D: local repository, so it contains both required local commits: `55a26b4` (tooling) and `024aaef` (experiment record). D: and the 100k workspace remain intact and unchanged.
- C: free space was 74.54 GiB before migration, about 70.76 GiB immediately before the first segment, and 68.07 GiB after final Fast verification. After the user's cleanup, C: had 66.05 GiB at the step-10,000 boundary and 65.84 GiB at resume. D: had 19.63 GiB before cleanup and was not used for output. Codex did not delete or modify D: data or the preserved 100k workspace.
- The host gate checks the workspace volume. C: holds the source checkout, Android build intermediates, CMake output, Gradle cache, Android SDK, private cache, checkpoints, reports, telemetry, and temporary build output. The runner observed 75,980,914,688 host-free bytes at segment start against the unchanged 20 GiB minimum.
- Measured C: sizes during migration: Android `app/build` 1.77 GiB; `app/.cxx` 2.73 GiB; the copied expanded-train cache 0.88 GiB; primary reports and telemetry about 0.01 GiB after the first segment; pinned MNN source 0.32 GiB. The existing 100k C: workspace was about 7.75 GiB and was not reused as an output directory.
- The C: cache copy matches the registered cache SHA-256 `bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf`, manifest file SHA-256 `e8a77e06abab3da5ff7e26eb9744c3bffad8cf5fbf006bbf4c7f582393961e4`, tokenizer SHA-256 `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`, validation SHA-256 `fd3fce2f61837435338127f165c15137051acaf8edf39d18f45c762bd310fd4f`, and development SHA-256 `a74bc959328fc6602d87f7e420ae9c4cb771084b8965c3d37e01e824ee3f028b`. No re-tokenization or data reselection occurred.
- C: build and audit used pinned QAIRT Build ID `2.48.40.260702151143`, QNN API 2.37.0, Hexagon SDK 6.6.0.0 / Tools 19.0.07, and HTP V81. The QNN+HVX APK audit passed; required libraries matched the pinned SDK. The APK hash remains in ignored private preflight evidence and is not committed.
- The C: device preflight confirmed a single stable physical device and no active run. Device free space was 13,802,455,040 bytes before staging the cache. After staging and the first segment it was 12,826,243,072 bytes, 58,658,816 bytes below the unchanged 12 GiB minimum. A later read-only resume preflight measured 12,726,960,128 bytes, 157,941,760 bytes below the minimum. After the user's cleanup, device `/data` reported 42,824,628 KiB available at the 1,000 boundary; it later measured about 46.75 GiB at step 10,000, above the unchanged 12 GiB gate. The fixed full-cap evaluations at steps 1,000, 2,000, 4,000, and 8,000 all passed.
- Existing C: preflight, Android/QNN/HVX build audits, 8k sanity, and resume-equivalence evidence were retained under ignored build output. No checkpoint, private cache, raw telemetry, device identifier, or APK hash was added to Git.

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
| QNN+HVX build and audit | PASS | Fresh debug and androidTest APKs built with Hexagon SDK 6.6.0.0 / Tools 19.0.07. The audit passed; APK hash is retained in ignored private evidence. SDK manifest SHA-256: `8bb7a783a59b7b929355f643e78580addf80bbc1e5f4762c244ce2e06be9478f`; compiler SHA-256: `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`. |
| Device smoke and G1 health | PASS | Device profile passed. Stable physical device identity and clean active-run gates were confirmed; sanity used HTP with HVX_W8, Aux Adam on CPU, no fallback, and pinned QAIRT. Thermal status stayed 0; observed battery temperature was 31–38 °C. |
| Resume equivalence | PASS | Fresh uninterrupted 0→2k and segmented 0→1k→resume→2k produced byte-identical step-2k checkpoints (SHA-256 `92930af7b1f3bb80ebb92ba631063ae6f931723fdf082097fed599bf8f6cad1d`). |
| Fresh 8k sanity | PASS | Fresh 0→1k→resume→8k reached step 8k with matching initialization, cache, order, QNN, finite, HVX, and checkpoint evidence. Details below. |
| C: host and device storage headroom | PASS after cleanup and rechecked before resume | C: had 70,684,344,320 bytes free and device `/data` had 50,243,301,376 bytes free immediately before resume. Both exceed the unchanged 20 GiB host and 12 GiB device gates; the runner rechecks before each segment. |

The C: migration and the user's cleanup cleared the host/device storage blockers without changing either gate. The step-1,000 pending evaluation completed, and the primary trajectory reached 10,000 updates. The user selected charger-connected execution, and the device's AC-powered status was confirmed before resume. Both storage gates pass with margin; no threshold or recipe was changed.

## Primary result

**Status: RUNNING from step 10,000.** The production trajectory started from fresh G1 seed 1 and ran 0→1,000→5,000→10,000. QNN/HVX, finite, optimizer, data-order, initialization, and checkpoint checks passed. Full-cap Val/Dev evaluations at 1,000, 2,000, 4,000, and 8,000 passed. The step-10,000 checkpoint (`a8f62923c5bf272c7432c3093355075ba0665e5ca42c471f7bb395ca90e3b70f`) is the resume point; no sanity checkpoint was used.

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
| 1,000 | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | Uninterrupted, segmented, sanity, primary |
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
| Primary: 0→1k | HEALTHY_COMPLETE; initial evaluation deferred by storage, later PASS | `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b` | 256,000 | 107.990 s training | Fresh initialization, QNN/HVX and finite checks passed; step-1,000 full-cap evaluation later passed after the user freed device space. |
| Primary: 1k→5k | HEALTHY_COMPLETE | `279fa455067cec0d0f7067a4eddfd35c90637e438c77a6114c7098ceef87c360` | 1,024,000 | 451.583 s training; 531.954 s runner wall time | Checkpoints at 2k–5k verified; full-cap evaluations at 2k and 4k passed. |
| Primary: 5k→10k | HEALTHY_COMPLETE | `a8f62923c5bf272c7432c3093355075ba0665e5ca42c471f7bb395ca90e3b70f` | 1,280,000 | 619.801 s training; 700.088 s runner wall time | Checkpoints at 6k–10k verified; full-cap evaluation at 8k passed. Resumed from this checkpoint after AC charging and both storage gates were confirmed. |

### Fixed full-cap quality

Delta convention: new result minus the 100k baseline at the same 100k step; negative NLL/BPB delta is better. Top-1 delta is new minus baseline. 200k is reported as the final Long Linear result, without claiming that it beats a 100k baseline at matched training duration.

| Step | Val NLL | Val BPB | Val top-1 | Dev NLL | Dev BPB | Dev top-1 | QNN / finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100k baseline | 3.728428748 | 2.026486192 | 0.248132176 | 3.705932880 | 2.135994049 | 0.245884229 | PASS / PASS |
| 1k Long Linear | 4.740780648 | 2.576722575 | 0.1110130103 | 4.877831245 | 2.811442854 | 0.1054555044 | QNN PASS / finite PASS |
| 2k Long Linear | 4.520167723 | 2.456814411 | 0.1297111849 | 4.639855020 | 2.674280143 | 0.1285582518 | QNN PASS / finite PASS |
| 4k Long Linear | 4.326372417 | 2.351482235 | 0.1529517016 | 4.407774785 | 2.540515713 | 0.1470924688 | QNN PASS / finite PASS |
| 8k Long Linear | 4.160503968 | 2.261328944 | 0.1763763038 | 4.209927866 | 2.426482389 | 0.1702072576 | QNN PASS / finite PASS |
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

At step 10,000, cumulative exposure was 2,560,000 target tokens across 80,000 records; 79,767 records and 52,122 articles were unique, and cache traversal fraction was 0.006278712848. The latest training loss was 4.201113701 (step 5,000: 4.191999435). Active model-training time summed to 1,179.373 s through step 10,000. The step-10,000 segment had 40,000 successful QNN graph executions, zero QNN failures, zero HVX RPC failures/fallbacks/nonfinite outputs, no CPU fallback, and finite outputs/parameters. Gradient clipping remained disabled with zero clipped steps.

| Step | Aux Adam LR | Muon LR | Gradient L2 | Parameter L2 | Muon Δ L2 | Aux Adam Δ L2 | Health |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,000 | 0.003299999982 | 0.007499999832 | 1.215008242 | 121.4618655 | 0.5808333385 | 0.2635671950 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 2,000 | 0.003299999982 | 0.007499999832 | 1.008321383 | 145.0968900 | 0.5292620506 | 0.2503667038 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 4,000 | 0.003299999982 | 0.007499999832 | 1.369613396 | 182.8969511 | 0.4987918135 | 0.2600771705 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 8,000 | 0.003235714277 | 0.007353895810 | 1.041447873 | 241.2259175 | 0.4826847443 | 0.2645275856 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 10,000 | 0.003203571308 | 0.007280843798 | 1.125344220 | 265.6270575 | 0.4642483703 | 0.2511263691 | QNN/HVX/finite PASS; fallback 0; clipped 0 |

LR telemetry matched the frozen schedule at the decay boundary: step 4,000 remained at peak, step 4,001 was Aux/Muon 0.003299983917 / 0.007499963045, and step 5,000 was 0.003283928614 / 0.007463473827. The 8,000 and 10,000 values are shown above.

Gate telemetry is weighted over 19 layers × 2 heads:

| Step | Mean | Stddev | Min | Max | Below 0.1 | Above 0.9 | Head 0 / Head 1 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.122361645 | 0.117555923 | 0 | 0.943359375 | 58.3824% | 0.0003701% | 0.118403 / 0.126320 |
| 2,000 | 0.074435355 | 0.069969324 | 0 | 0.925781250 | 76.1764% | 0.0000514% | 0.070702 / 0.078169 |
| 4,000 | 0.040580056 | 0.040425261 | 0 | 0.767089844 | 92.9634% | 0% | 0.039744 / 0.041416 |
| 8,000 | 0.023604149 | 0.026507951 | 0 | 0.510742188 | 97.5085% | 0% | 0.023823 / 0.023385 |
| 10,000 | 0.020207683 | 0.023376378 | 0 | 0.640625000 | 98.2149% | 0% | 0.019800 / 0.020615 |

Gate values continue to saturate near zero; after step 4,000 none were above 0.9. This does not justify stopping or changing the registered schedule or architecture.

Each full-cap boundary executed 52,604 QNN calls with return-code success, finite outputs, and no CPU fallback. Full-cap evaluation elapsed times were 1,273.775 s (1k), 1,288.993 s (2k), 1,609.658 s (4k), and 1,488.082 s (8k), totaling 5,660.509 s. At the 8k evaluation, battery changed from 35% to 31%, battery temperature fell from 47 °C to 38 °C, and thermal status stayed 0. Cumulative active training time is 1,179.373 s; the current primary run has not yet reached a quality target or the 100k comparison. Exact per-layer/head and per-window telemetry remains in ignored local evidence.

### Incidents and negative findings

The first setup attempt used a QNN-only APK and failed native initialization with `HVX_MUON_NOT_BUILT` at step 0; zero updates were applied. The preflight was corrected to build and audit the pinned QNN+HVX APK before device use, and the successful r3 resume/sanity gates used that APK. A second attempt correctly built and audited QNN+HVX, but a wrapper check misread CRLF line endings in the audit report; it stopped before installation or training. The wrapper now normalizes line endings, and the final preflight passed.

The original D: host-space block measured 19.71 GiB against the 20 GiB host minimum; C: migration resolved it without Codex deleting D: data. The primary 0→1,000 checkpoint SHA-256 was `aee1d40468ad553891c40a72b06fc3c01da4425f65ffba23556ef3a7384ee05b`. After dataset staging, device free space was 12,826,243,072 bytes against the 12,884,901,888-byte gate, and a later preflight measured 12,726,960,128 bytes. The runner initially stopped before step-1,000 evaluation. After the user's cleanup, that evaluation and later fixed evaluations through step 8,000 passed. The primary trajectory completed 5,000 and 10,000 segment boundaries with verified checkpoints and matching initialization/order identities. Checkpoint at step 5,000 SHA-256: `279fa455067cec0d0f7067a4eddfd35c90637e438c77a6114c7098ceef87c360`; step 10,000: `a8f62923c5bf272c7432c3093355075ba0665e5ca42c471f7bb395ca90e3b70f`. Training and evaluation had no QNN/HVX, finiteness, or fallback failures. At the step-10,000 boundary, the device had 31% battery, was unplugged, and had thermal status 0 at 38 °C. The user selected charger-connected execution; AC-powered status was then confirmed at 36% battery and 38 °C, with thermal status 0. The verified checkpoint was resumed without changing the trajectory. The original runner initially wrote the healthy segment ledger after the post-segment storage assertion, so the first stopped run-plan omitted step 1,000. We reconciled it only after rechecking result, checkpoint, log, LR, optimizer, and gate evidence. Tooling was corrected to persist validated boundaries before post-segment capacity checks and to preflight pending evaluations on resume. Runner SelfTest and Fast passed after those changes. Codex did not delete D: data or 100k artifacts, and neither storage gate was altered. Continue only from a verified primary checkpoint; never restart from step 0 or use a sanity/100k checkpoint.

### Convergence and next research decision

**Current classification: `INCOMPLETE`.** The primary trajectory has completed 10,000 updates and four full-cap evaluations through step 8,000. Val and Dev NLL/BPB improved at each measured early boundary, but this is too early to classify late-stage slopes or infer a 100k schedule comparison. The 8k sanity run is separate from the primary trajectory. The 200k hard ceiling does not presume convergence.

The run has resumed from the verified step-10,000 checkpoint. Full-cap evaluations through step 8,000 are complete and will not be repeated. The next registered segment is 10,000→15,000; its evaluation plan remains fixed. Do not change the schedule, data, optimizer, seed, or safety gate. No conclusion about Long Linear versus the 100k baseline or the need for a matched fixed-terminal 200k control is supported yet. Linear 200k versus Baseline 100k alone is not a superiority claim. `6031` remains `UNRESOLVED / DORMANT / WATCH`.
