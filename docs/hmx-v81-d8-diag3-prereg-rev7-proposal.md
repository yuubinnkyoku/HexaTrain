# D8 Diagnostic 3 — VTCM Pointer Verification (Revision 7 Proposal)

Status: **`D8_DIAG3_PROPOSED`**, 2026-11-23. This document authorises nothing.
It is a proposal for the owner to accept or reject.

**This is a diagnostic, not a D8 observation.** It never issues `mxclracc` or
anything after it, so the critical interval does not run and the D8 question is
not put to the hardware.

Companion documents:

- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (§44–§49 are the investigation)
- `docs/hmx-v81-d8-diag2-prereg-rev6-proposal.md` (the previous diagnostic)
- `docs/hmx-v81-cand-b-d8-device-prereg-rev5-proposal.md` (the last registered build)

## 0. What is established, and what is not

### 0.1 Established

| Fact | Evidence |
| --- | --- |
| The device's Ahi tile differs from the CPU's in 931 of 1,024 words | diag2 payload |
| The anchors differ: `0xCB76` where the CPU has `0x3C00` | diag2 payload |
| `base` = `0x02080000` | diag2 metadata word 11 |
| `base + 0x20` = `0xA320` = `hmx_d8_diag_local0`, `base + 0x28` = `0xA328` = `ahi_local` | `nm -S` on the committed object |
| The payload words 16..19 are `local0`/`local1`/`local_fnv`; words 20.. are `ahi_local` | payload vs symbol layout |
| `get_vtcm_ptr` returns `attr+0x8` unchanged and validates nothing | framework disassembly |
| `attr_init` does not clear `attr+0x8` | framework disassembly |
| `attr+0x8` is written at exactly one place, `0x1848`, and every path reaching `0x1844` goes through it | framework disassembly |
| That write reads `memw(r29+#0x18)`, an acquire-frame stack word | framework disassembly |
| The VTCM-failure path `0x1604` clears stack `+0x8` but not `+0x18` | framework disassembly |

### 0.2 Not established

**Whether the VTCM-failure path returns a non-zero context id.** The call at
`0x1630` has no relocation — the instruction is the unresolved `5a004000` form
and objdump prints it as pointing at itself — so its return value cannot be
determined statically.

**Scope note.** The framework analysed is the simulator's `test_util.a` (v68).
The device's v81 runtime is a different binary that the SDK does not ship. The
structural findings about `attr+0x8` likely carry over; the conclusion about what
`acquire` writes is about the simulator implementation. **The device diagnostic
is the decisive arbiter.**

### 0.3 The two hypotheses

**(A) Stale write.** VTCM allocation failed, `acquire` still returned a non-zero
context id, and `attr+0x8` received the uninitialised stack word
`memw(r29+#0x18)`, which happened to be a pointer into the skel's own `.bss`.

**(B) Legitimate grant.** `0x02080000` is the VA of a real VTCM grant, and it
collides with the skel image from the scalar view.

## 1. What this diagnostic measures

### 1.1 The primary discriminator

**The `acquire()` return value (context id).**

- **Zero** → allocation failed, so hypothesis (A) cannot be the mechanism and
  the run stops before the handler.
- **Non-zero** → allocation reported success. Combined with the pointer check
  below, this separates (A) from (B).

### 1.2 The pointer check (main defence)

**Skel self-image exclusion.** Compare `vtcm` against the skel's own image
range. If `vtcm` falls inside it, the pointer is not VTCM and the run stops
unconsumed with `HMX_ERR_VTCM`.

**The range is defined by the linker, not guessed.** The upper bound is the
linker symbol `__end`, which sits just past `.bss` (`0xC331` in the diag2
object, with `.bss` running `0x6318`..`0xC330`), so **`.bss` is inside
the range**. The lower bound is the page containing the check function's own
address, which is inside `.text`. The observed failure pointed at `0xA300`,
inside `.bss`, and this is exactly what the range must cover.

This is the check that would have caught the observed failure directly.

### 1.3 Bidirectional coherence

Two directions, because one alone cannot distinguish a wrong address from a
write that did not land:

| Direction | Test |
| --- | --- |
| scalar write → `vmem` read | write a pattern with scalar stores, read it back with a vector load |
| `vmem` write → scalar read | write a pattern with a vector store, read it back with scalar loads |

**Both must agree.** If either fails, the address is not coherently usable and
the run stops unconsumed.

### 1.4 `attr` capture

`attr+0x0..+0x3f` is 16 words. **The D8 metadata table has only two free words
(29 and 31)**, so the full 16 words cannot go in metadata.

**Resolution: the diagnostic writes them into `results`.** The diagnostic does
not need the 1,024-word payload for a tile this time, because the question is
the pointer, not the tile contents. Layout:

| Words | Content |
| --- | --- |
| `0..15` | `attr+0x0..+0x3f`, verbatim |
| `16` | gate FNV Ahi |
| `17` | gate FNV Alo |
| `18` | gate FNV Whi |
| `19` | gate FNV Wlo |
| `20` | `acquire()` return value (context id) |
| `21` | `base & 127` |
| `22` | coherence: scalar write -> vmem read (0 = pass) |
| `23` | coherence: vmem write -> scalar read (0 = pass) |
| `24..31` | **reserved, written as zero** |
| `32..1023` | zero |

**This avoids the diag2 mistake of overwriting the gate FNV slots.** The gate
FNVs get their own words and are never reused.

### 1.5 Metadata words

| Word | Content |
| ---: | --- |
| 9 | diagnostic stamp `0x0D1A6003` |
| 10 | `acquire()` return value (context id) |
| 11 | `vtcm` low 32 bits |
| 12 | `vtcm` high 32 bits |
| 13 | `vtcm & 127` |
| 14 | distance from `vtcm` to `&ahi_local` (signed) |
| 15 | self-image check result: 0 = outside, 1 = inside |
| 16 | coherence: scalar write → vmem read (0 = pass) |
| 17 | coherence: vmem write → scalar read (0 = pass) |
| 18 | `attr+0x4` (the requested size) |
| 19 | `attr+0x0` (the magic/version word) |
| 20 | gate FNV Ahi |
| 21 | gate FNV Alo |
| 22 | gate FNV Whi |
| 23 | gate FNV Wlo |
| 24 | guard pre FNV |
| 25 | guard status |
| 26 | failure code |
| 27 | schema |
| 28 | readback |
| 29 | reserved (0) |
| 30 | cleanup |
| 31 | sentinel `0x0D1A6003` |

**The sentinel is bumped to `0x0D1A6003`**, fixing the diag2 known-issue where
`HMX_D8_DIAG_SENTINEL` still said `0x0D1A6001`.

## 2. Pre-declared interpretation

**Declared before the run. No other reading is permitted.**

| Observation | Reading |
| --- | --- |
| `acquire()` returns 0 | Allocation failed. The run stops unconsumed; **(A) is not the mechanism** and the failure is a plain VTCM acquisition failure. |
| `acquire()` non-zero and self-image check says **inside** | **(A) confirmed.** The pointer is not VTCM; it is the skel's own image. The stale-write mechanism is the explanation. |
| `acquire()` non-zero, self-image check says **outside**, both coherence tests pass | **(B) confirmed.** `0x02080000` is a legitimate VTCM VA. The collision is in how the scalar view maps it. |
| `acquire()` non-zero, self-image check says outside, a coherence test fails | The address is not coherently usable. Neither (A) nor (B) as stated; the VTCM mapping itself is broken. |
| `attr+0x4` != 12,288 | The attribute structure was not populated as requested. **This is corroboration only, not proof of grant** — it shows `set_vtcm_param` recorded the request, not that VTCM was granted. |

**`attr+0x4` is deliberately not a primary discriminator.** It records the
request, not the grant.

## 3. What changes in `run()`

### 3.1 The gate

Inserted after `get_vtcm_ptr()` and before `hmx_lock2`:

1. **`memset(&attr, 0, sizeof(attr))` before `attr_init`** — hygiene only. It
   does **not** fix the observed mechanism, because `acquire` writes `attr+0x8`
   on every path that reaches `0x1844`. It is included so that a future
   framework that leaves the field untouched fails closed.
2. **`acquire()` return value check** — zero stops unconsumed.
3. **Skel self-image exclusion** — the main defence.
4. **Bidirectional coherence** — both directions must pass.
5. **`get_vtcm_ptr_v2` size** — secondary consistency check only, because the
   size is read from the pointer's own `+0x20` and may be garbage.

**Failure stops unconsumed with `HMX_ERR_VTCM`.**

### 3.2 What does NOT change

| Item | Status |
| --- | --- |
| the D8 handler (`run_mode46_d8_one_trial`) | **unchanged** |
| the four registered operand FNVs | **unchanged** |
| the four reference SHA-256 values | **unchanged** |
| the issued order, offsets, fill form, `Rs`, sync ban | **unchanged** |
| the acquisition order (`acquire()` → `get_vtcm_ptr()`) | **unchanged** — documented as correct for the non-cacheable flow |
| `hmx_d8_contract.h` | **unchanged** |

## 4. Build

| Item | Value |
| --- | --- |
| Link recipe | identical to `0x4D8B1125` (12 extracted libc objects + `-Bsymbolic` + `libgcc.a` + `libqurt.a`) |
| Stamp | **`0x0D1A6003`** |
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

1. The run writes to `/data/local/tmp/d8/evidence/diag3-<stamp>/`, **not `/tmp`**.
2. Full stdout, the 1,024-word payload, the 32-word metadata, the observed
   SHA-256, the build id double echo and the UTC timestamps are saved there.
3. **The deployed skel binary and its SHA-256 are committed** under
   `docs/evidence/`, because the build is not reproducible.
4. The evidence is committed and pushed **on the day of the contact**.

## 7. Gates before the session

| Gate | Requirement |
| --- | --- |
| Disassembly audit | four conditions PASS on the stamped object |
| Link binding | `math_plt=0`, `math_und=0`, UND within the allowlist |
| Handler | unchanged segment count for the registered sequence |
| `.init` / `.fini` | `0x6c` / `0x44`, same as `0x4D8B1125` |
| CPU suite | `run_hmx_d8_sim.ps1` PASS |
| `verify.ps1 -Profile Fast` | all PASS |
| Identity | `build_id_status=MATCH`, `IDENTITY_OK=PASS` before the run |

## 8. What this does not do

It does not run the critical interval, does not produce a D8 numeric result, and
does not retire `B3_R2_SYNC_UNPROVEN`. `C2` stays `NOT_RUN`.

## 9. What the owner is being asked

Accept or reject one diagnostic run under §5, with the measurements in §1, the
pre-declared interpretation in §2, and the `run()` changes in §3.
