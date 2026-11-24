# D8 Diagnostic 3 — VTCM Pointer Verification (Revision 8 Proposal)

Status: **`D8_DIAG3_REV8_PROPOSED`**, 2026-11-24. This document authorises
nothing. It is a proposal for the owner to accept or reject.

**This is a diagnostic, not a D8 observation.** It never issues `mxclracc` or
anything after it, so the critical interval does not run and the D8 question is
not put to the hardware.

Companion documents:

- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (§44–§50 are the investigation)
- `docs/hmx-v81-d8-diag3-prereg-rev7-proposal.md` (the rejected revision)
- `docs/hmx-v81-d8-diag2-prereg-rev6-proposal.md` (the previous diagnostic)
- `docs/hmx-v81-d8-fastrpc-failure-contract.md` (the return-value rule)

## 0. Why revision 7 was withdrawn

Revision 7 was implemented and built. The §7 gates all passed, and the build
still could not have produced a measurement. Six defects were found; four by
the implementer, two by the owner. They are recorded in worklog §50 and
summarised here because revision 8 exists to fix them.

| # | Defect | Consequence |
| ---: | --- | --- |
| 1 | The gate ran in `run()`, before the handler | On gate failure the handler never ran, `results[]` stayed zeroed, `run()` returned non-zero, and the registered contract discards `rout` on a non-zero return. **The diagnostic recorded nothing about the failure it exists to catch.** |
| 2 | `__end` does not resolve at link time | The reference stays `*UND*` with an `R_HEX_GLOB_DAT` relocation. The DSP loader does not export `__end`, so the GOT slot keeps the loader's value and the check takes its unknown-range branch every run. |
| 3 | The image check compared a runtime pointer against a link-time constant | The diag2 payload shows the load base is not zero, so the two values are not in the same address space. The comparison is meaningless. |
| 4 | The coherence test issued `vmemu` with no HVX lock held | `run()` never calls `qurt_hvx_lock`; the handler does, and the gate ran before it. The same mistake fixed as a device bug in worklog §24.4. |
| 5 | The FNV-mismatch return leaked the HVX lock | `d5191cb` removed the `qurt_hvx_unlock()` that preceded it. The most frequently taken failure path returned with the context still held. |
| 6 | The `attr` snapshot was transported through a `.bss` static | Under (A) the pack's vector stores land inside the image, and the diagnostic's own statics sit in that window. The measurement destroys its own source before the echo. |

**The lesson, which the owner asked to be made an audit item:** the §7 gates
check agreement with the registration. They do not check whether the
registration can produce a measurement. A gate that passes while the design is
blind is worse than a gate that fails.

## 1. What this diagnostic measures

### 1.1 The primary discriminator

**The `acquire()` return value (context id).**

- **Zero** → allocation failed, so hypothesis (A) cannot be the mechanism and
  the run stops before the handler.
- **Non-zero** → allocation reported success. Combined with the pointer check
  below, this separates (A) from (B).

### 1.2 The pointer check — runtime against runtime

**This is the change that makes the check meaningful.** Revision 7 compared a
runtime pointer against the link-time `__end`, which defect 3 shows is invalid.
Revision 8 compares two **runtime** addresses and embeds no link-time constant
in the comparison.

The anchor is `&hmx_d8_diag_attr`, a static the diagnostic owns. It is always in
`.bss`, so its runtime address moves with the load base exactly as the image
does.

```
inside = (base - &hmx_d8_diag_attr >= -8) && (base - &hmx_d8_diag_attr < 0x60B8)
```

The two constants are the anchor's offset from the image start and end, taken
from the linked object:

| Quantity | Link-time value | Offset from the anchor |
| --- | ---: | ---: |
| `.bss` start | `0x6310` | `-8` |
| `.bss` end | `0xC3D0` | `0x60B8` |
| anchor `&hmx_d8_diag_attr` | `0x6318` | `0` |

**These two offsets are the only embedded constants, and they are relative, not
absolute.** They are correct for any load base because both terms of the
subtraction move together. The audit script checks them against the object.

Validation, computed before the run:

| Case | `base − anchor` | Verdict | Correct? |
| --- | ---: | --- | --- |
| (A), load base `0x02075D00`, `base` = `0x02080000` | `0x3FE8` | inside | yes |
| (B), load base `0`, `base` = `0x02080000` | `0x02079CE8` | outside | yes |
| (A), `base` = anchor | `0` | inside | yes |
| (A), `base` = image end − 1 | `0x60B7` | inside | yes |
| (A), `base` = image end | `0x60B8` | outside | yes |

### 1.3 The load base, measured rather than assumed

Revision 7's defect 3 rested on an **estimate** of the load base
(`0x02075D00`) derived from the diag2 payload. Revision 8 measures it.

The diagnostic echoes `&hmx_vtcm_inside_image` — a runtime address of a function
whose link-time address is known from the object. The host computes

```
load_base = echoed_runtime_address − link_time_address
```

and can then check the whole picture arithmetically. **No estimate is needed and
none is used.**

### 1.4 Bidirectional coherence

Two directions, because one alone cannot distinguish a wrong address from a
write that did not land:

| Direction | Test |
| --- | --- |
| scalar write → `vmem` read | write a pattern with scalar stores, read it back with a vector load |
| `vmem` write → scalar read | write a pattern with a vector store, read it back with scalar loads |

**Both must agree.** Each direction gets its own result word, so a failure says
which one broke.

**The owner's warning is recorded here and governs the interpretation:** under
(A) both directions land in the same `.bss` RAM, so **the coherence test can
pass while the pointer is wrong**. It is not a discriminator. `inside_image` is
the only effective discriminator, and the load-base measurement is what makes
`inside_image` trustworthy.

### 1.5 `attr` and context transport — direct, not through a static

Revision 7 copied `attr` into a `.bss` static and read it back in the handler.
Defect 6 shows why that fails: under (A) the pack's stores land in the image,
and the diagnostic's statics are in that window.

Measured on the revision-7 build:

| Symbol | Link address | Inside the pack window `[0xA300, 0xC300)`? |
| --- | ---: | --- |
| `hmx_d8_diag_attr` | `0x6318` | no |
| `hmx_d8_diag_context` | `0x63B8` | no |
| `hmx_d8_diag_local0` | `0xA3C0` | **yes** |
| `hmx_d8_diag_local1` | `0xA3C2` | **yes** |
| `ahi_local` | `0xA3C8` | **yes** |
| `alo_local` | `0xABC8` | **yes** |
| `whi_local` | `0xB3C8` | **yes** |
| `wlo_local` | `0xBBC8` | **yes** |

So the specific statics revision 7 used for `attr` and the context id happen to
sit **outside** the window, while the generator's own tiles sit **inside** it.
The owner's general point stands and is adopted regardless: **transport through
`results`, which is host-shared memory and is not affected by writes into the
`base` window.**

`run()` writes `results[0..15]` (the 16 `attr` words) and `results[20]` (the
context id) **directly**, before the handler is called. The handler never copies
them and never needs to.

## 2. Where the gate runs

**In the handler, not in `run()`.** This is the fix for defects 1 and 4 at once.

The handler already takes `qurt_hvx_lock` at its top, so the coherence test's
`vmemu` is issued with the lock held — defect 4 is structurally impossible.
And the handler's diagnostic block already returns 0, so `rout` survives —
defect 1 is structurally impossible.

The gate is evaluated in the diagnostic block, which is reached on the
operand-FNV mismatch. Its verdict goes into `results[]` and `metadata[]`, and
the handler returns 0.

**`run()` keeps no gate.** The `memset` before `attr_init` stays as hygiene
only, and the `execution_error` assignment added in `d5191cb` is reverted with
the gate.

## 3. Results layout

Fixed and versioned by `HMX_D8_DIAG_LAYOUT = 4`.

| Words | Content | Written by |
| --- | --- | --- |
| `0..15` | `attr+0x0..+0x3f`, verbatim | `run()`, before the handler |
| `16` | gate FNV Ahi | handler |
| `17` | gate FNV Alo | handler |
| `18` | gate FNV Whi | handler |
| `19` | gate FNV Wlo | handler |
| `20` | `acquire()` return value (context id) | `run()`, before the handler |
| `21` | `base & 127` | handler |
| `22` | coherence: scalar write → `vmem` read (0 = pass) | handler |
| `23` | coherence: `vmem` write → scalar read (0 = pass) | handler |
| `24` | `inside_image` verdict (0 = outside, 1 = inside) | handler |
| `25` | `&hmx_vtcm_inside_image` at run time | handler |
| `26` | `&hmx_d8_diag_attr` at run time | handler |
| `27` | `base` low 32 bits | handler |
| `28..31` | reserved, written as zero | handler |
| `32..1023` | zero | handler |

**Word 25 is what makes the load base measurable** (§1.3). Word 26 lets the host
recompute the window test independently of the handler's own arithmetic.

## 4. Metadata words

| Word | Content |
| ---: | --- |
| 9 | diagnostic stamp |
| 10 | `acquire()` return value (context id) |
| 11 | `base` low 32 bits |
| 12 | `base` high 32 bits |
| 13 | `base & 127` |
| 14 | `inside_image` verdict |
| 15 | coherence, scalar → vector |
| 16 | coherence, vector → scalar |
| 17 | `&hmx_vtcm_inside_image` at run time |
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
| 31 | sentinel |

### 4.1 The stamp

**A new stamp is required: `0x0D1A6004`.** Revision 7's `0x0D1A6003` was built
and its binary exists, so reusing the stamp would make two different binaries
claim the same identity. The layout version moves to 4 with it.

## 5. Pre-declared interpretation

**Declared before the run. No other reading is permitted.**

| Observation | Reading |
| --- | --- |
| `acquire()` returns 0 | Allocation failed. The run stops unconsumed; **(A) is not the mechanism** and the failure is a plain VTCM acquisition failure. |
| `acquire()` non-zero, `inside_image` says **inside** | **(A) confirmed.** The pointer is not VTCM; it is the skel's own image. |
| `acquire()` non-zero, `inside_image` says **outside**, both coherence tests pass | **(B) confirmed.** `0x02080000` is a legitimate VTCM VA. |
| `acquire()` non-zero, `inside_image` says outside, a coherence test fails | The address is not coherently usable. Neither (A) nor (B) as stated; the VTCM mapping itself is broken. |
| `inside_image` says inside **and** a coherence test fails | Consistent with (A): the region is ordinary writable RAM, so a coherence failure would mean the vector and scalar views disagree even within the image. Recorded, not interpreted further. |
| `attr+0x4` != 12,288 | The attribute structure was not populated as requested. **Corroboration only, not proof of grant.** |
| word 25 is 0 | The load base could not be measured. **The run is inconclusive** and no reading is permitted. |

**`attr+0x4` is deliberately not a primary discriminator.** It records the
request, not the grant.

**The worst failure mode is a false "all clear".** A run that reports
`inside_image = outside` because the comparison was broken would read as (B)
and would be wrong. That is why word 25 exists and why a zero there voids the
run.

## 6. What changes in the code

### 6.1 `probe_dsp.c`

| Change | Reason |
| --- | --- |
| Remove the gate from `run()` | defects 1 and 4 |
| Revert the `execution_error` assignment added with the gate | it existed only to carry the gate's code |
| Keep `memset(&attr, 0, sizeof(attr))` before `attr_init` | hygiene only; it does not fix the observed mechanism |
| Write `results[0..15]` and `results[20]` directly after `acquire()` | defect 6 |
| `hmx_vtcm_inside_image` compares runtime against runtime | defects 2 and 3 |
| `hmx_vtcm_coherent` returns one bit per direction | each direction gets its own word |

### 6.2 `hmx_mode46_d8.h`

| Change | Reason |
| --- | --- |
| Evaluate the gate in the diagnostic block | defects 1 and 4 |
| Echo words 24–27 | §3 |
| Restore `qurt_hvx_unlock()` before the diagnostic `return 0` | defect 5 |
| Bump the sentinel and layout to `0x0D1A6004` / `4` | §4.1 |

### 6.3 What does NOT change

| Item | Status |
| --- | --- |
| the four registered operand FNVs | **unchanged** |
| the four reference SHA-256 values | **unchanged** |
| the issued order, offsets, fill form, `Rs`, sync ban | **unchanged** |
| the critical interval | **unchanged** — the diagnostic never reaches it |
| `hmx_d8_contract.h` | **unchanged** |
| the acquisition order (`acquire()` → `get_vtcm_ptr()`) | **unchanged** |

## 7. Build

| Item | Value |
| --- | --- |
| Link recipe | identical to `0x4D8B1125` (12 PIC libc objects + `-Bsymbolic` + `libgcc.a` + `libqcc.a` + `libqurt.a`) |
| Stamp | **`0x0D1A6004`** |
| `mxclracc` and below | **never issued** |

**The PIC `libc.a` is required.** The non-PIC archive fails to link a shared
object with `R_HEX_32_6_X` relocation errors; this was found while building
revision 7 and is recorded so it is not rediscovered.

## 8. Budget

| Item | Value |
| --- | --- |
| Builds | 1 |
| Device sessions | **1** |
| Runs | **1** (q=6, mode 46) |
| Retries | **0** |
| NPU gate | **required** |

## 9. Evidence preservation

1. The run writes to `/data/local/tmp/d8/evidence/diag3-<stamp>/`, **not `/tmp`**.
2. Full stdout, the 1,024-word payload, the 32-word metadata, the observed
   SHA-256, the build id double echo and the UTC timestamps are saved there.
3. **The deployed skel binary and its SHA-256 are committed** under
   `docs/evidence/`, because the build is not reproducible.
4. The evidence is committed and pushed **on the day of the contact**.
5. **Each evidence directory records the commit at which its binary can be
   retrieved**, since `.so` files are no longer tracked.

## 10. Gates before the session

| Gate | Requirement |
| --- | --- |
| Disassembly audit | four conditions PASS on the stamped object |
| Link binding | `math_plt=0`, `math_und=0`, UND within the allowlist |
| Handler | the registered issued order unchanged |
| `.init` / `.fini` | `0x6c` / `0x44`, same as `0x4D8B1125` |
| CPU suite | `run_hmx_d8_sim.ps1` PASS |
| `verify.ps1 -Profile Fast` | all PASS |
| Identity | `build_id_status=MATCH`, `IDENTITY_OK=PASS` before the run |
| **Evidence-survival** | **NEW.** A synthetic run of the failure path must show the payload reaching the host. A gate that passes while the payload is discarded is the defect this revision exists to fix. |
| **Window constants** | **NEW.** The two offsets in §1.2 are checked against the linked object by the audit script. |

## 11. What this does not do

It does not run the critical interval, does not produce a D8 numeric result, and
does not retire `B3_R2_SYNC_UNPROVEN`. `C2` stays `NOT_RUN`.

## 12. What the owner is being asked

Accept or reject one diagnostic run under §8, with the measurements in §1, the
pre-declared interpretation in §5, the code changes in §6, and the two new gates
in §10.
