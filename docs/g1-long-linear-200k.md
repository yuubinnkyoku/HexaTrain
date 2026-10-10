# HexaTrain G1 Long-Horizon Linear Decay, 200k

**Status: `UNSTABLE`; the fresh primary trajectory reached a verified step-40,000 checkpoint. The 40k→45k segment stopped on non-finite QNN output (`logits`, 243,059 non-finite values) despite QNN return-code success. The last verified checkpoint is step 40,000; the failed trajectory is not resumable.**

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
- The C: run-control source HEAD for the verified 40,000 boundary and the failed 40k→45k attempt was `464690aad73b08e3bdbcccf844595dea17df3bc3`; the failure record and no-resume guard are committed afterward as audit/tooling changes only.
- C: execution source HEAD at the 10k resume was `d8f68b12a8207cdaa3346855a6c09620afdeaae1`; the subsequent report-only commit is `2a054bd4811e6d76e7bbe803992248642128bf7d`.
- The branch was fetched from the D: local repository, so it contains both required local commits: `55a26b4` (tooling) and `024aaef` (experiment record). D: and the 100k workspace remain intact and unchanged.
- C: free space was 74.54 GiB before migration, about 70.76 GiB immediately before the first segment, and 68.07 GiB after final Fast verification. After the user's cleanup, C: had 66.05 GiB at the step-10,000 boundary, 65.84 GiB at resume, 65.56 GiB after the 30k segment, and 65.27 GiB after the verified 40k segment. Device `/data` then had 46.07 GiB free. D: had 19.63 GiB before cleanup and was not used for output. Codex did not delete or modify D: data or the preserved 100k workspace.
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
| C: host and device storage headroom | PASS through step 40,000 | Both unchanged gates passed before each completed segment. The 40,000 checkpoint is present and verified. The failed 40k→45k segment stopped on non-finite output, not storage. |

The C: migration and the user's cleanup cleared the host/device storage blockers without changing either gate. The primary trajectory reached a verified step-40,000 boundary. The user confirmed charger-connected execution; the runner’s power checks also captured a transient AC-false/discharging sample during the 16k evaluation, then later AC-true/charging samples. The 40k→45k failure was numerical and occurred while storage headroom remained above gate. No threshold or recipe was changed.

## Primary result

**Status: `UNSTABLE`; 40,000 primary updates completed and verified, then the 40k→45k segment failed its application-visible finiteness check.** The production trajectory began from fresh G1 seed 1. Segments through 40,000 passed QNN/HVX, optimizer, data-order, initialization, checkpoint, and finite-output checks. Full-cap Val/Dev evaluations at 1,000, 2,000, 4,000, 8,000, 16,000, and 32,000 passed. During the next segment, all 13,170 QNN graph calls returned success, but the residual output contained 243,059 non-finite values, including 7,428 NaNs in `logits`. The last verified checkpoint is step 40,000 (`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`). The exact failing optimizer step is not available; no step after 40,000 is treated as verified, and this run must not resume.

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
| Primary: 10k→15k | HEALTHY_COMPLETE | `4d9776fa12d85ccc2644311bb97d6cf1f5698b77fa32c1208db662816d13c6ed` | 1,280,000 | 686.080 s training; 773.020 s runner wall time | Checkpoints at 11k–15k verified; zero QNN/HVX failure, fallback, or nonfinite events. |
| Primary: 15k→20k | HEALTHY_COMPLETE | `425ec9031aff9459afb501126f5cd035ad9ccb6d5773fb8aa09a0c6a0d532e3e` | 1,280,000 | 575.080 s training; 672.570 s runner wall time | Checkpoints at 16k–20k verified; fixed 16k full-cap evaluation passed. |
| Primary: 20k→25k | HEALTHY_COMPLETE | `6ebb62a390a5ab61daab97e3aa01c9b4308423365c1a9e98f9e6f53b47b0eada` | 1,280,000 | 1,108.500 s training; 1,237.630 s runner wall time | Checkpoints at 21k–25k verified; zero QNN/HVX failure, fallback, or nonfinite events. |
| Primary: 25k→30k | HEALTHY_COMPLETE | `e9f20a6bc5ffea7ad4ae97ba196440e5965152d4698ee86acb9b31a612a7a4ef` | 1,280,000 | 771.131 s training; 902.930 s runner wall time | Checkpoints at 26k–30k verified; zero QNN/HVX failure, fallback, or nonfinite events. |
| Primary: 30k→35k | HEALTHY_COMPLETE | `a46a54722d92cd65a5f110e448f626dd8314698d1639bec8e9a44d26bc312685` | 1,280,000 | 703.875 s training | Checkpoints through 35k verified; step-32k full-cap evaluation passed. |
| Primary: 35k→40k | HEALTHY_COMPLETE | `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d` | 1,280,000 | 670.065 s training | Checkpoints through 40k verified; QNN/HVX and finite checks passed. |
| Primary: 40k→45k requested | FAILED; `UNSTABLE_NONFINITE` | No verified checkpoint after step 40k | Not reconciled | QNN execute returned 0 on all 13,170 calls, but 243,059 application-visible outputs were non-finite (7,428 NaNs in `logits`). Exact failing optimizer step was not recorded. The run is fail-closed and must not resume. Result SHA-256 `dd3ba99aa6b86f422fe9826e2857a724ed69dec81b45a2fbab11b611f488b37c`; segment-log SHA-256 `31ae3c97419e5c03c1ba460692b1d9eb056c62f02c32e28345c7b2cec933bb13`. |

### Fixed full-cap quality

Delta convention: new result minus the 100k baseline at the same 100k step; negative NLL/BPB delta is better. Top-1 delta is new minus baseline. 200k is reported as the final Long Linear result, without claiming that it beats a 100k baseline at matched training duration.

| Step | Val NLL | Val BPB | Val top-1 | Dev NLL | Dev BPB | Dev top-1 | QNN / finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100k baseline | 3.728428748 | 2.026486192 | 0.248132176 | 3.705932880 | 2.135994049 | 0.245884229 | PASS / PASS |
| 1k Long Linear | 4.740780648 | 2.576722575 | 0.1110130103 | 4.877831245 | 2.811442854 | 0.1054555044 | QNN PASS / finite PASS |
| 2k Long Linear | 4.520167723 | 2.456814411 | 0.1297111849 | 4.639855020 | 2.674280143 | 0.1285582518 | QNN PASS / finite PASS |
| 4k Long Linear | 4.326372417 | 2.351482235 | 0.1529517016 | 4.407774785 | 2.540515713 | 0.1470924688 | QNN PASS / finite PASS |
| 8k Long Linear | 4.160503968 | 2.261328944 | 0.1763763038 | 4.209927866 | 2.426482389 | 0.1702072576 | QNN PASS / finite PASS |
| 16k Long Linear | 4.015671717 | 2.182609308 | 0.2003156424 | 4.024119574 | 2.319387787 | 0.1968237034 | QNN PASS / finite PASS |
| 32k Long Linear | 3.913179639 | 2.126902522 | 0.218395899 | 3.945325825 | 2.273973317 | 0.2133722834 | QNN PASS / finite PASS |
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

At step 40,000, cumulative exposure was 10,240,000 target tokens across 400,000 selections; 316,047 records and 122,487 articles were unique, and cache traversal fraction was 0.02511485139. The latest verified training loss was 4.200420856. Active primary model-training time summed to 5,694.099 s through step 40,000; full-cap evaluation elapsed time summed to 10,015.052 s through step 32,000. The completed 0→40k trajectory made 320,000 successful QNN graph executions with zero QNN or HVX failures, zero fallback, and finite outputs/parameters. The failed 40k→45k attempt added 13,170 successful QNN calls before the application-visible non-finite check stopped it; no exact optimizer step or exposure from that attempt is claimed. Gradient clipping remained disabled with zero clipped steps.

| Step | Aux Adam LR | Muon LR | Gradient L2 | Parameter L2 | Muon Δ L2 | Aux Adam Δ L2 | Health |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,000 | 0.003299999982 | 0.007499999832 | 1.215008242 | 121.4618655 | 0.5808333385 | 0.2635671950 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 2,000 | 0.003299999982 | 0.007499999832 | 1.008321383 | 145.0968900 | 0.5292620506 | 0.2503667038 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 4,000 | 0.003299999982 | 0.007499999832 | 1.369613396 | 182.8969511 | 0.4987918135 | 0.2600771705 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 8,000 | 0.003235714277 | 0.007353895810 | 1.041447873 | 241.2259175 | 0.4826847443 | 0.2645275856 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 10,000 | 0.003203571308 | 0.007280843798 | 1.125344220 | 265.6270575 | 0.4642483703 | 0.2511263691 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 16,000 | 0.003107142868 | 0.007061688229 | 1.249991656 | 321.5758544 | 0.4419821133 | 0.2444944547 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 30,000 | 0.002882142784 | 0.006550324615 | 1.511364094 | 411.0655347 | 0.3845620431 | 0.2286621103 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 32,000 | 0.002850000048 | 0.006477272604 | 1.713088651 | 420.7603798 | 0.3736130145 | 0.2236003808 | QNN/HVX/finite PASS; fallback 0; clipped 0 |
| 40,000 | 0.002721428638 | 0.006185064558 | 1.636061420 | 454.6030940 | 0.3558501781 | 0.2092847265 | QNN/HVX/finite PASS at verified boundary; fallback 0; clipped 0 |

LR telemetry matched the frozen schedule: step 4,000 remained at peak; step 4,001 was Aux/Muon 0.003299983917 / 0.007499963045; step 15,000 was 0.003123214236 / 0.007098213769; step 16,000 was 0.003107142868 / 0.007061688229; step 20,000 was 0.003042857163 / 0.006915584672; step 25,000 was 0.002962500090 / 0.006732954644; step 30,000 was 0.002882142784 / 0.006550324615; step 32,000 was 0.002850000048 / 0.006477272604; and step 40,000 was 0.002721428638 / 0.006185064558. Both optimizer columns followed the same preregistered decay multiplier through the last verified step.

Gate telemetry is weighted over 19 layers × 2 heads:

| Step | Mean | Stddev | Min | Max | Below 0.1 | Above 0.9 | Head 0 / Head 1 mean | Per-layer mean range |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.122361645 | 0.117555923 | 0 | 0.943359375 | 58.3824% | 0.0003701% | 0.118403 / 0.126320 |
| 2,000 | 0.074435355 | 0.069969324 | 0 | 0.925781250 | 76.1764% | 0.0000514% | 0.070702 / 0.078169 |
| 4,000 | 0.040580056 | 0.040425261 | 0 | 0.767089844 | 92.9634% | 0% | 0.039744 / 0.041416 |
| 8,000 | 0.023604149 | 0.026507951 | 0 | 0.510742188 | 97.5085% | 0% | 0.023823 / 0.023385 |
| 10,000 | 0.020207683 | 0.023376378 | 0 | 0.640625000 | 98.2149% | 0% | 0.019800 / 0.020615 |
| 16,000 | 0.015810860 | 0.022009158 | 0 | 0.411132813 | 98.1440% | 0% | 0.016974 / 0.014647 | 0.000001–0.086991 |
| 30,000 | 0.010984270 | 0.017955467 | 0 | 0.577636719 | 99.0146% | 0% | 0.012518 / 0.009451 | 0–0.067974 |
| 32,000 | 0.010788207 | 0.018029211 | 0 | 0.247192383 | 98.9712% | 0% | 0.012368 / 0.009208 | 0–0.068101 |
| 40,000 | 0.009597224 | 0.016669796 | 0 | 0.489501953 | 99.1943% | 0% | 0.011559 / 0.007635 | 0–0.060915 |

Gate values continue to saturate near zero; after step 4,000 none were above 0.9. This does not justify stopping or changing the registered schedule or architecture.

Each full-cap boundary executed 52,604 QNN calls with return-code success, finite outputs, and no CPU fallback. The step-32k full-cap evaluation elapsed 2,220.579 s; cumulative evaluation time through 32k is 10,015.052 s. From 16k to 32k, Val NLL improved by 0.102492078 and Dev NLL by 0.078793749; BPB improved by 0.055706786 and 0.045414470, respectively. These early changes do not supply the registered 100k comparison. Exact per-layer/head and per-window telemetry remains in ignored local evidence.

### Incidents and negative findings

The first setup attempt used a QNN-only APK and failed native initialization with `HVX_MUON_NOT_BUILT` at step 0; zero updates were applied. Preflight was corrected to build and audit the pinned QNN+HVX APK. A later audit wrapper check misread CRLF and stopped before install/training; line-ending normalization fixed it. Both issues were resolved before the primary run. D: initially had 19.71 GiB against the unchanged 20 GiB host minimum. The C: migration and the user's cleanup resolved the storage shortage without Codex deleting D: data. Device storage initially missed the 12 GiB floor after cache staging; the user's cleanup cleared that blocker. The step-1,000 evaluation later completed, and all registered full-cap evaluations through 32,000 passed.

The primary 10k boundary had 31% battery and no external power. The user chose charger-connected execution; AC power was verified before the 10k resume. One later power sample during the 16k evaluation showed AC/USB/wireless false at 53%; subsequent reads reported AC true/Charging, and no training update began while the sample was false. The user later confirmed the charger remained connected. No storage gate was changed. Codex did not alter D: or the preserved 100k experiment.

At step 40,000, the verified checkpoint SHA-256 was `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`. The subsequent 40k→45k attempt failed the application-visible finite-output check: 243,059 outputs were non-finite, including 7,428 NaNs in `logits`; poison count was zero. QNN return code was 0 on all 13,170 graph executions, with 13,170 successes, zero QNN call failures, no CPU backend, and no fallback attempt. Thus successful QNN return codes do not explain away the separate tensor-finiteness failure. The log reached 229 seconds and `completed=1/2`, but the exact optimizer step was not recorded; no update count, data exposure, or checkpoint after 40,000 is claimed. Result SHA-256 is `dd3ba99aa6b86f422fe9826e2857a724ed69dec81b45a2fbab11b611f488b37c`; segment-log SHA-256 is `31ae3c97419e5c03c1ba460692b1d9eb056c62f02c32e28345c7b2cec933bb13`. The run plan is marked `UNSTABLE`, its resume pointer is cleared, and the runner now rejects an `UNSTABLE` plan. No training was resumed after the non-finite event.

### Convergence and next research decision

**Final classification: `UNSTABLE`.** This fresh single-seed trajectory reached 40,000 verified updates and then produced non-finite model outputs. It is incomplete relative to the 200,000-update ceiling, but the required classification for this run is `UNSTABLE`; it is neither a convergence nor plateau result. Val/Dev quality improved through the last completed evaluation at 32k, while gate values continued to saturate near zero. The 40k→45k numerical failure is the stopping reason, not a quality-based early stop.

The 100k same-budget comparison was not reached, so this run cannot establish whether Long Linear beats the 100k baseline. There is no 200k final quality or 200k convergence analysis, and no claim that Linear 200k outperforms Baseline 100k is possible. First investigate the non-finite failure and the progressive near-zero gate saturation with a separate, explicitly reviewed study; do not resume or mutate this frozen trajectory. A matched fixed-terminal 200k control should follow only after numerical stability can be established, since a control with different failure risk would not answer the schedule comparison cleanly. `6031` remains `UNRESOLVED / DORMANT / WATCH`.
