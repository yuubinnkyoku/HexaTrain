# R2 S1/S2 device implementation plan (CPU-only, not authorised)

Status: **PLAN_ONLY / NOT AUTHORISED**, 2026-10-25. This document fixes the
implementation shape of the minimal R2 device test. It does **not** add a DSP
mode, change DSP source, build a DSP image, contact a device, or measure
performance. The DSP/Host diff below is a specification for a future
authorised change.

Companion documents:

* [`hmx-v81-r2-acc-reuse-feasibility.md`](hmx-v81-r2-acc-reuse-feasibility.md)
  — spec audit, CPU state model, cost model, verdict.
* [`hmx-v81-r2-device-prereg.md`](hmx-v81-r2-device-prereg.md) — frozen
  expectations and decision procedure.

## 0. What this test is and is not

The test isolates **ACC reuse, bias-set selection, store ordering and
DSP-authored HVX reconstruction** with a single dot per subcase. It is not a
B3 accuracy run: the G8/G9 1,024-cell B3 comparison is CPU-only evidence, and
its 3 MMA packets per tile are a different scope. In this minimal test:

* S1 uses **one MMA packet** (`a × a`), two converts, two stores.
* S2 uses **one MMA packet** (three spatial lanes), two converts, two stores.
* Total: 2 MMA packets, 4 converts, 4 stores, 2 HVX recon loops.
* No B3 hi/lo 3-product synthesis, no output-channel duplication.

A packet means one activation+weight HMX pair. The static audit counts the
pair as one semantic MMA and reports the raw instruction mnemonics separately.

## 1. Frozen fixture

```text
a  = 1 + 2^-6 = 1.015625              FP16 0x3C10
x  = a*a      = 4225/4096 = 1.031494140625
64x = 66.015625                       0x5420 (RN16), 0x5421 (ceiling)
32x = 33.0078125                      0x5020 (RN16), 0x5021 (ceiling)
-64x = -66.015625                     0xD420 (RN16)
```

S1 tile: `A(0,0)=+a`, `W(0,0)=+a`, all else zero → `ACC(0,0)=+x`.

S2 tile: `A(0,0)=+a`, `A(1,0)=-a`, `A(2,0)=0`, `W(0,0)=+a` →
`ACC(0,0)=+x`, `ACC(1,0)=-x`, `ACC(2,0)=0`.

## 2. VTCM map (frozen)

| region | offset | size | alignment | contents |
| --- | ---: | ---: | --- | --- |
| `act` | `+0` | 2048 B | 2 KiB | activation crouton |
| `wgt` | `+2048` | 2048 B | 2 KiB | weight crouton |
| `out_pos` | `+4096` | 2048 B | 2 KiB | convert #1 store |
| `out_neg` | `+6144` | 2048 B | 2 KiB | convert #2 store |
| `out_recon` | `+8192` | 2048 B | 2 KiB | DSP HVX `vsub.hf` result |
| `bias` set 0 | `+10240` | 256 B | 256 B | `+64`, shape 0, bo 0 |
| `bias` set 1 | `+10496` | 256 B | 256 B | `+64`, shape 4, bo 0 |
| `bias` set 2 | `+10752` | 256 B | 256 B | `+32`, shape 0, bo 0 |
| total | | 11008 B | | 1280 B free of 12288 B |

All three output regions are disjoint and 2 KiB aligned. `out_recon` is
written only by the HVX loop; the two convert stores must not touch it. The
HVX loop must not modify `out_pos` or `out_neg`; the run checks both buffers
after the loop. The bias rows are 256 B aligned.

The layout shifts the bias rows from the earlier +8192/+8448/+8704 plan to
make room for a third 2 KiB region. This is a pre-implementation change; no
existing device mode or measured result uses these offsets.

## 3. Frozen 11-slot readback

The host result buffer grows from 4 to 11 slots. The RPC IDL already uses an
unbounded `sequence<long>`; only the host-side array changes.

| slot | subcase | buffer | lane | source | expected |
| ---: | --- | --- | --- | --- | --- |
| 0 | S1 | first | `(s0,o0)` | convert store | `0x5420` |
| 1 | S1 | second | `(s0,o0)` | convert store | `0x5020` |
| 2 | S2 | positive | `(s0,o0)` | convert store | `0x5420` |
| 3 | S2 | positive | `(s1,o0)` index 1 | convert store | `0x0000` |
| 4 | S2 | positive | `(s2,o0)` index 64 | convert store | `0x0000` |
| 5 | S2 | negative | `(s0,o0)` | convert store | `0x0000` |
| 6 | S2 | negative | `(s1,o0)` index 1 | convert store | `0x5420` |
| 7 | S2 | negative | `(s2,o0)` index 64 | convert store | `0x0000` |
| 8 | S2 | recon | `(s0,o0)` | DSP `out_recon` | `0x5420` |
| 9 | S2 | recon | `(s1,o0)` index 1 | DSP `out_recon` | `0xD420` |
| 10 | S2 | recon | `(s2,o0)` index 64 | DSP `out_recon` | `0x0000` |

Frozen pattern:

```text
0x5420 0x5020 0x5420 0x0000 0x0000 0x0000 0x5420 0x0000 0x5420 0xD420 0x0000
```

`recon_source` must be `DSP_HVX`. A host-side `pos - neg` is a fallback
analysis, not a registered device result.

## 4. Instruction sequence (frozen)

### Prologue (once per invocation)

```text
bias load set 0  (+10240)
bias load set 1  (+10496)
bias load set 2  (+10752)
```

`mxclracc` is issued once before each subcase's MMA (2 total), not in the
prologue. The three bias loads are counted once; they are not repeated
between converts.

### S1 (retention + bias-set control)

```text
mxclracc
MMA (one packet)                 # ACC(0,0) = +x
convert #1: Rs = 0x0001         # set 0, retain, no feedback
store out_pos
convert #2: Rs = 0x2000         # set 2 (+32), clear+swap, no feedback
store out_neg
HVX recon loop: out_recon = out_pos - out_neg
read slots 0..1
```

### S2 (sign split)

```text
mxclracc
MMA (one packet)                 # +x, -x, 0 lanes
convert #1: Rs = 0x0001         # set 0, retain, no feedback
store out_pos
convert #2: Rs = 0x1000         # set 1, shape 4, clear+swap, no feedback
store out_neg
HVX recon loop: out_recon = out_pos - out_neg
read slots 2..10
```

The HVX loop is 16 iterations of 64 FP16 lanes for a 32x32 tile:

```text
for v in 0..15:
    p = load(out_pos  + v*128)
    n = load(out_neg  + v*128)
    r = vsub.hf(p, n)
    store(out_recon + v*128, r)
```

The loop reads both inputs before writing `out_recon`; it must not use
`out_pos` or `out_neg` as the destination.

### Counts to prove

| item | planned count | proof required |
| --- | ---: | --- |
| `mxclracc` | 2 (one before each MMA) | static audit + DSP counter |
| MMA packets | 2 (S1, S2) | static audit shows no third packet; DSP counter = 2 |
| bias loads | 3 (sets 0/1/2, prologue) | static audit + DSP counter + load status |
| converts | 4 | static audit + DSP counter + `Rs` echo |
| stores | 4 | static audit + DSP counter + sentinel |
| HVX recon loops | 2 | static audit + DSP counter + recon region write |
| trials | 2 (S1, S2) | host metadata + DSP counter |
| returned slots | 11 | host array size and zero-extension |
| `Rs` values | `0x0001, 0x2000, 0x0001, 0x1000` | DSP echo + static immediate audit |

The static audit must inspect the **built DSP binary**, not just the source.
It reports the raw instruction mnemonics, the semantic packet/convert/store
counts, and the four `Rs` immediates. A source diff alone is not evidence.

## 5. Evidence provenance

| source | what it establishes | limitation |
| --- | --- | --- |
| Host-planned counts | intended recipe | not execution evidence |
| DSP instruction counters | the instrumented path ran | self-reported; no instruction-identity proof |
| Static disassembly | the built binary contains the registered sequence | does not prove it ran |
| `Rs` echo | the reported `Rs` values are the registered ones | self-reported |
| Store sentinel | a store wrote the region | no count/order by itself |
| 11 result slots | numerical output | not bitwise ACC retention |

`R2_ACC_REUSE_SUPPORTED` requires the result pattern **and** a changed
sentinel **and** build-id match **and** `Rs` echo **and** DSP counters **and**
a static disassembly audit of the same build. The counters and static audit
must agree. Host-planned counts alone are `R2_HOST_CONTRACT_FAILED`; a single
counter or a single disassembly is not enough. The CPU model tests this
contract with `r2.evidence.*`, but it does not produce device evidence.

## 6. DSP/Host diff outline (not implemented)

### DSP side (future, not in this change)

* Add one mode (proposed name `R2S1S2`; the numeric mode is chosen when the
  change is authorised) with the fixture and VTCM map above.
* Write the three bias rows and the two tiles.
* Issue the exact sequence in §4; do not add feedback bits or a third MMA.
* Run the HVX loop into `out_recon`; never overwrite `out_pos`/`out_neg`.
* Return 11 zero-extended slots, counters, `Rs` echo, bias-load status,
  sentinel status, build id and close status.
* Do not reuse an existing measured mode's expectations; the new mode has its
  own frozen table.

### Host side (future, not in this change)

* Widen the result buffer from 4 to 11; keep the IDL `sequence<long>`.
* Validate slot count, zero-extension, sentinel, counts, `Rs`, bias loads,
  region disjointness, recon source, build id, RPC and close status.
* Require the static-audit/counter cross-check; reject host-planned counts.
* Do not compute slots 8..10 on the host; read them from the DSP result.

## 7. Fail-closed conditions

Before interpreting any slot, all of the following must hold:

* one clean stamped V81 build, one deploy/run session, one attempt, no retry;
* `results.size() == 11`, each slot zero-extended 16-bit;
* no `0xA5A5` sentinel in any returned slot;
* `metadata`: 2 MMA, 4 convert, 4 store, 3 bias load, 2 HVX recon loops,
  2 trials, `hmx_executed=1`;
* `Rs` values exactly `0x0001, 0x2000, 0x0001, 0x1000`, and the DSP echo
  matches;
* `out_pos`, `out_neg`, `out_recon` distinct and 2 KiB aligned;
* `recon_source == DSP_HVX`;
* DSP counters and static disassembly counts agree, both present;
* host/DSP build id match, RPC status 0, close status 0.

Any failure is `R2_HOST_CONTRACT_FAILED` or an infrastructure failure. A
failed contract is never interpreted as success. An S1 second slot of
`0x0000` or `0x5420` is `R2_CAUSE_INCONCLUSIVE`, not a cause verdict.

## 8. CPU-only verification of this plan

The CPU model already covers the decision procedure and the evidence
contract:

```text
r2_cpu_model_checks=114
hmx_signed_output_oracle checks=269 fail=0
r2_slot_contract=11/11 sentinel=0 host_fail_closed=PASS
r2_evidence recon=DSP_HVX counts=dsp_counter+static_audit rs_echo=yes sentinel=yes
r2_device_verdict=NOT_RUN
```

`r2_pack_slots()` fixes the 11-slot order. `r2.evidence.*` rejects a host-only
recon, host-planned counts, a counter-only or static-only source, a missing
`Rs` echo and a missing sentinel. `r2.layout.*` checks the new VTCM offsets,
alignment and the 1,024-cell store bijection. `r2.recon.*` checks that the
HVX writer leaves `out_pos`/`out_neg` unchanged. The oracle is CPU-only and
does not prove the DSP implementation.

The contract facts are read back from the executed model, not typed in as
constants: the four registered `Rs` values come from the converts that
actually ran (`r2.contract.rs_*`, `r2.rs_model.*`, including the rejection of
the unregistered `0x1001`), the per-subcase HVX recon loop count comes from
the model (`r2.contract.s1_recon_*`, `r2.contract.recon_twice`), and
`bias_loads` counts the prologue load events of the one shared bias bank that
both subcases reuse (`r2.bias.*`).

## 9. Stop conditions and authorization

After the future run: record the 11 raw slots, metadata, counters, `Rs` echo,
build id and static-audit report; stop. No retry, no second session, no
expectation change.

* `R2_STORE_INTERFERENCE` blocks R2 on this runtime.
* `R2_CAUSE_INCONCLUSIVE` stops the decision; record diagnostics and do not
  assert ACC destruction or bias-switch failure.
* `R2_ROUNDING_MISMATCH` means the RN16 assumption does not hold for this
  path; record and diagnose.
* `R2_HOST_CONTRACT_FAILED` invalidates the run; do not interpret the slots.
* `R2_ACC_REUSE_SUPPORTED` certifies only this minimal sequence; it does not
  certify a full tile, real-tile accuracy, HVX subnormal behaviour, or
  performance.

DSP implementation, a new mode, a DSP build, device execution and performance
measurement are **not authorised**. PR #14 stays Draft. The research commits
through `fbe665a` (11 commits) are pushed to
`origin/experiment/hmx-acc-residual-feedback`.
