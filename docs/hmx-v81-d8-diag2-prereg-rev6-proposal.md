# D8 Diagnostic 2 — Preregistration (Revision 6 Proposal)

Status: **`D8_DIAG2_PROPOSED`**, 2026-11-23. This document authorises nothing.
It is a proposal for the owner to accept or reject.

**This is a diagnostic, not a D8 observation.** It never issues `mxclracc` or
anything after it, so the critical interval does not run and the D8 question is
not put to the hardware.

Companion documents:

- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (§41 is the specification finding)
- `docs/hmx-v81-cand-b-d8-device-prereg-rev5-proposal.md` (the last registered build)
- `docs/hmx-v81-d8-disasm-audit-spec.md` (Appendix A)

## 0. Why a second diagnostic exists

The first diagnostic captured the Ahi tile and the USR. Its data was lost
(§40.2), but before that it established two things:

1. The device's Ahi tile differs from the CPU's in 931 of 1,024 words.
2. **The anchors differ.** The device reads `0xCB76` where the CPU has `0x3C00`.

`0xCB76` is about −14.9 in fp16. No conversion path, rounding mode, FTZ setting
or implementation can turn `1.0` into −14.9. So the device is reading a different
byte sequence, not a differently-rounded one.

Analysis 4 cleared the pack: all four `vmem` stores go to `r3+0`, `r3+0x800`,
`r3+0x1000` and `r3+0x1800`, the registered offsets, from one base register.

Analysis 5 found the specification that explains how a correct pack can still be
read wrong.

## 1. The two hypotheses this diagnostic discriminates

### 1.1 Cache asymmetry — persistent staleness

> Scalar data memory accesses go through a two-level cache hierarchy, while HVX
> memory accesses only transit through the L2 memory.

If a stale line sits in L1D and an HVX store does not invalidate it, a scalar
read returns the stale value **forever**. Time does not fix it.

### 1.2 `(R0 & 127)` truncation — base misalignment

> `VMEM(R0) = V0` // (R0 & 127) bits are ignored

If the runtime base pointer is not 128 B aligned, a `vmem` store lands at
`floor128(base + off)` while a scalar read uses the exact address. The two see
different bytes.

**This hypothesis explains every observation at once:**

| Observation | Explained |
| --- | --- |
| sim reproduces the registered constants | sim's base is aligned |
| guard (scalar write) matches | scalar write and scalar read agree |
| tiles (vmem write) do not match | vmem truncates, scalar read does not |
| session 3's FNV changed with the link recipe | the read value tracks the generator's output, just shifted |

A purely stale read would not change when the link recipe changes. It did change,
so the read is seeing the tile content through a shifted or partial view.

### 1.3 The weak hypothesis, kept only for completeness

> Expect a delay of tens of cycles between the HVX write to memory and the scalar
> read.

**This alone does not explain the failure.** Thousands of cycles pass between the
pack and the FNV (guard init, `syncht`, loop setup), and session 3 ran with
`syncht` and still failed. It is recorded so the delayed re-read can rule it out
rather than assume it away.

## 2. What the diagnostic measures

### 2.1 Payload layout

`results` is 1,024 words. The allocation:

| Words | Content |
| --- | --- |
| `0..1023` | **Ahi tile, all 2,048 B** (the tile that differs) |

The other three tiles do not fit. **Ahi is the one that matters**: it is the tile
whose anchors are wrong, and its first 16 words are the anchor row. Capturing it
whole gives the byte-level comparison the first diagnostic could not finish.

### 2.2 Metadata words

| Word | Content |
| ---: | --- |
| 9 | diagnostic stamp `0x0D1A6002` |
| 10 | `base & 127` — **the single most important word** |
| 11 | `base` low 32 bits |
| 12 | `base` high 32 bits |
| 13 | FNV of Ahi **immediately after the pack** |
| 14 | FNV of Ahi **immediately before return** (delayed re-read) |
| 15 | first word of `ahi_local` (the generator's output, before the pack) |
| 16 | second word of `ahi_local` |
| 17 | first word of `[base-0x80]` (the window below base) |
| 18 | second word of `[base-0x80]` |
| 19 | first word of `[base-0x40]` |
| 20 | second word of `[base-0x40]` |
| 21 | scalar write/read sanity: wrote `0xA5A5` to `base+0`, read back |
| 22 | USR raw value |
| 23 | `ahi_local` FNV (the generator's output as a whole) |
| 24 | guard pre FNV (control: scalar write, scalar read) |
| 25 | guard status |
| 26 | failure code |
| 27 | schema |
| 28 | readback |
| 29 | reserved (0) |
| 30 | cleanup |
| 31 | sentinel `0x0D1A6002` |

**Word 10 alone nearly decides hypothesis 1.2.** If `base & 127 != 0`, the vmem
stores are landing at a truncated address and the scalar reads are not.

### 2.3 The window below base

Words 17..20 capture `[base-0x80]` and `[base-0x40]`. **If the misalignment
hypothesis is right, the tile's anchor row (`3C00 BC00 1000 9000 …`) appears
there**, because the vmem store landed below the scalar read's view.

This is the direct test. One run finds it or does not.

### 2.4 The delayed re-read

Word 13 is the FNV right after the pack; word 14 is the FNV just before return.
**If they differ, the store lands late** and hypothesis 1.3 is confirmed in its
strong form. If they are equal and wrong, it is persistent (1.1 or 1.2).

## 3. Pre-declared interpretation

**Declared before the run. No other reading is permitted.**

| Observation | Reading |
| --- | --- |
| `base & 127 != 0` | **Misalignment confirmed.** Combined with an anchor found in the window below base, hypothesis 1.2 is settled. |
| `base & 127 == 0` and word 13 != word 14 | **Late landing confirmed** (§24.3 strong form). |
| `base & 127 == 0` and word 13 == word 14 and neither is the registered FNV | **Persistent wrong view**: cache asymmetry (1.1) or another alias effect. |
| `ahi_local` FNV != the CPU's | **Generator output differs.** FP hypothesis revives, but it contradicts the anchor distance argument, so this is low confidence. |
| `base+0` scalar write/read sanity fails | The scalar path itself is broken; every scalar read in the handler is suspect. |

## 4. Build

| Item | Value |
| --- | --- |
| Link recipe | **identical to `0x4D8B1125`** (12 extracted libc objects + `-Bsymbolic` + `libgcc.a` + `libqurt.a`) |
| Source | the prologue is identical to `0x4D8B1125`; the diagnostic code sits inside the FNV gate |
| Stamp | **`0x0D1A6002`** |
| skel SHA-256 | `e0a4b4d9d5bc230c038f20813160df664c5c2dab590246de6ded7849c7146d8d` |
| Size | 34,712 B |
| `mxclracc` and below | **never issued** |

## 5. Budget

| Item | Value |
| --- | --- |
| Builds | 1 |
| Device sessions | **1** |
| Runs | **1** (q=6, mode 46) |
| Retries | **0** |
| NPU gate | **required** |

## 6. Evidence preservation

**This is the lesson from §40.2, where the first diagnostic's data was lost.**

1. The run writes to `/data/local/tmp/d8/evidence/diag2-<stamp>/`, **not `/tmp`**.
2. The full stdout, the 1,024-word payload, the 32-word metadata, the observed
   SHA-256, the build id double echo and the UTC timestamps are all saved there.
3. **The deployed skel binary and its SHA-256 are committed to the repository**
   under `docs/evidence/`, because the build is not reproducible
   (`3d9a46d7…` was rebuilt from the same recipe and did not match `d0b3e302…`).
   The recovered object is the only durable guarantee that the audited object and
   the device object are the same.
4. The evidence is committed and pushed **on the day of the contact**.

## 7. Gates before the session

| Gate | Requirement |
| --- | --- |
| Disassembly audit | four conditions PASS on the stamped object |
| Link binding | `math_plt=0`, `math_und=0`, UND within the allowlist |
| Handler | 352 segments (280 + the diagnostic block); the registered sequence is unchanged |
| `.init` / `.fini` | `0x6c` / `0x44`, same as `0x4D8B1125` |
| CPU suite | `run_hmx_d8_sim.ps1` PASS |
| `verify.ps1 -Profile Fast` | all PASS |
| Identity | `build_id_status=MATCH`, `IDENTITY_OK=PASS` before the run |

## 8. What this does not do

It does not run the critical interval, does not produce a D8 numeric result, and
does not retire `B3_R2_SYNC_UNPROVEN`. `C2` stays `NOT_RUN`.

## 9. What the owner is being asked

Accept or reject one diagnostic run under §5, with the payload in §2, the
pre-declared interpretation in §3, and the evidence rules in §6.
