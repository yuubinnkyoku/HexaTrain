# G1 identity-init 500-step A/B (2026-09)

## Status

**DEVICE A/B BLOCKED** at authoring time: no physical HTP device is attached
(`adb devices` shows only an offline emulator). Implementation and host
identity proof are complete and committed. The 500-step Current vs Identity
run must be executed when the NX741J-class device is available.

## Hypothesis

Current G1 early optimization gain may come from either

1. initializing the attention branch near a 0.5 gate (soft shrink), or
2. the learnable head-wise gate itself.

This A/B isolates those factors.

## Arms (factor isolation)

Only two factors differ. Everything else is identical.

| | Arm A = Current G1 | Arm B = Identity-init G1 |
|---|---|---|
| gate math | `G = sigmoid(N @ Wg)` | `G = 2 * sigmoid(N @ Wg)` |
| Wg init | current deterministic linear | **zero** |
| step-0 G | ~0.5 neighborhood | **exactly 1** (`Yh = Ah`) |
| Wg shape | `[64,2] × 19` | same |
| params | 760,960 | 760,960 |
| optimizer | Wg → AUX_ADAM, 114 matrices → Original Muon | same |
| checkpoint | NPRTCKPTV5 | NPRTCKPTV5 |
| attention_gate id | `headwise_g1_sigmoid` (=1) | `headwise_g1_scale2_identity` (=2) |

Forbidden (unchanged): reduced channel, shared Wg, bias, Wv, head count,
residual, RMSNorm, optimizer partition, LR, batch, Muon, gate position.

## Math

```text
G = scale * sigmoid(z),  scale = 1 (current) or 2 (identity)
dG/dz = G * (1 - G/scale)

scale=1: dG/dz = G*(1-G) = s*(1-s)          max 0.25
scale=2: dG/dz = G*(1-G/2) = 2s(1-s)        at z=0: 0.5
```

Wg LR is **not** halved. Same optimizer/LR for both arms (user B4).

## Step-0 identity proof (host)

`host_tests/headwise_g1_gate_test.cpp` (PASS):

```text
identity_init_gate_mean=1
identity_init_gate_min=1
identity_init_gate_max=1
identity_init_wg_zero=true
identity_init_dwg_nonzero=true
identity_checkpoint_cross_resume_rejected=true
```

Also verified: ungated context parity at Wg=0 / scale=2, finite backward,
correct dWg shape, V5 encode/decode roundtrip, and fail-closed resume when
checkpoint `attention_gate=2` is extracted as gate=1 or ungated.

## Architecture identity

NPRTCKPTV5 stores `attention_gate` as u32. Decode accepts only 1 or 2.
`sameConfig` compares the enum, so a Current-G1 checkpoint cannot be resumed
as Identity-init and vice versa.

## Run recipe (pending device)

```powershell
# smoke (Identity, 8 updates)
.\scripts\run_g1_identity_ab.ps1 -Mode Smoke -Arm Identity `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'

# fresh matched A/B at 1.5x, 500 steps
.\scripts\run_g1_identity_ab.ps1 -Mode Ab `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143'
```

LR (fixed 1.5x):

| | |
|---|---:|
| Muon | 0.00750 |
| Aux Adam | 0.00330 |
| target | 0.000150 |

Eval: Val/Dev first 256 chunks, canonical original UTF-8 byte bpb at
100/200/300/400/500. Final split forbidden. Seed 1, step-0 fresh.

## Scale-aware gate telemetry

Identity-init gates live in `(0, 2)` with identity at 1. Keep legacy
`g<0.1` / `g>0.9` fields, and record `mean_abs(g-1)`. Do not treat
`g>0.9=1` at step-0 identity as pathology.

## Decision logic (pending results)

- `PROMOTE_IDENTITY_G1` / `PROMOTE_IDENTITY_G1_GRID`
- `KEEP_CURRENT_G1`
- `HOLD_IDENTITY_G1`

See task brief B15. Wall-time is secondary; step-to-bpb is primary when
checkpoint cumulative timing is unavailable (same rule as stress closure).

## Artifacts (after device run)

`docs/results/g1-identity-init-500-2026-09/` — quality, time-to-bpb, gate,
Wg telemetry, runtime, health, `current/`, `identity/`.

## Verification performed tonight

```text
Fast:  PASS
Host:  PASS (including identity-init gate tests)
runner self-test: PASS
device smoke / 500-step A/B: BLOCKED (no physical device)
```

## Commits

```text
feat(research): add identity-init headwise G1 variant
docs(research): correct G1 stress time-to-target metrics
```

push: NOT PERFORMED.
