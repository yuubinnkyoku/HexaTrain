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
Revision 8 compares two **runtime** addresses and embeds no absolute value in
the comparison.

The anchor is `&hmx_d8_diag_attr`, a static the handler owns. It is always in
`.bss`, so its runtime address moves with the load base exactly as the image
does.

```
inside = (base - &hmx_d8_diag_attr >= LO) && (base - &hmx_d8_diag_attr < HI)
```

**The lower bound covers the whole image, not just `.bss`.** This is the owner's
condition C1 and it closes a real hole. A window that started at `.bss` would
report a stale pointer into `.text` or `.data` as "outside"; the coherence test
would then pass, because that region is ordinary RAM in both directions, and the
run would read as **(B) confirmed** — a false all-clear, which is the worst
failure mode this diagnostic has.

| Quantity | Link-time value | Offset from the anchor |
| --- | ---: | ---: |
| address 0 (covers the ELF header) | `0x0000` | `-0xA318` |
| `.init` (first section) | `0x0FC0` | `-0x5358` |
| `.bss` start | `0x6310` | `-0x4008` |
| anchor `&hmx_d8_diag_attr` | `0xA318` | `0` |
| `.bss` end / image end | `0xC3C8` | `+0x20B0` |

The lower bound is the offset to **address 0**, not to `.init`. That is
deliberately stricter: it also covers the ELF header and any padding below the
first section, so a stale pointer anywhere below the image is caught.

**The two bounds are not hardcoded, and that matters.** They depend on the link:
the anchor moved from `0x6318` to `0xA318` between two builds of the same source
when the diagnostic block stopped using a stack local. A hardcoded pair would
have silently measured the wrong window after that change, and a wrong window
reads as "outside" — the false all-clear again. So the build script links once
with placeholders, reads the real layout from the object, relinks with the
measured values, and then **confirms the layout did not move between the two
passes**. The audit script re-derives both bounds from the final object and
fails if they disagree with what was compiled in.

Validation, computed before the run:

| Case | `base − anchor` | Verdict | Correct? |
| --- | ---: | --- | --- |
| (A), load base `0x02075D00`, `base` = `0x02080000` | `-0x18` | inside | yes |
| (B), load base `0`, `base` = `0x02080000` | `0x02075CE8` | outside | yes |
| (A), `base` = anchor | `0` | inside | yes |
| (A), `base` = image end − 1 | `0x20AF` | inside | yes |
| (A), `base` = image end | `0x20B0` | outside | yes |
| (A), `base` = address 0 (ELF header) | `-0xA318` | inside | yes |
| (A), `base` = address −1 | `-0xA319` | outside | yes |
| (A), `base` = `.text` (`0x11A0`) | `-0x9178` | inside | yes |
| (A), `base` = `.data` (`0x6188`) | `-0x4190` | inside | yes |

The last four are the cases C1 added. Without them the `.text` and `.data` rows
would have read "outside" and produced a false all-clear.

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
| `acquire()` returns 0 | Allocation failed. **(A) is not the mechanism** and the failure is a plain VTCM acquisition failure. **The host observes this directly**: `metadata[6] == -1`, `metadata[26] == HMX_D8_FAIL_VTCM`, `metadata[7] == 0`, and `results[0..15]`/`results[20]` carry the attribute words and the zero context id. See §5.1. |
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

### 5.1 The `acquire() == 0` path must be observable

This is the owner's condition C2. Revision 7's reading said "the run stops
before the handler", which is true but **not observable**: a non-zero `run()`
return discards `rout` data under the registered contract, so the host would
never see the zero context id it is supposed to be reading.

Revision 8 makes that path evidence-preserving:

| Item | Value on the `acquire() == 0` path |
| --- | --- |
| `run()` return | **0** |
| `metadata[6]` | `-1` (the acquire failed) |
| `metadata[7]` | `0` (the handler did not run; no D8 observation exists) |
| `metadata[26]` | `HMX_D8_FAIL_VTCM` |
| `metadata[31]` | the diagnostic sentinel |
| `results[0..15]` | the attribute words, written before the acquire |
| `results[20]` | `0`, the context id |

The payload is written **before** `acquire()` is called, so it exists on every
path. A cleanup failure on this path also returns 0 rather than discarding it.

**The evidence-survival principle therefore covers this path too**, which is
what the owner asked for.

## 6. What changes in the code

### 6.1 `probe_dsp.c`

| Change | Reason |
| --- | --- |
| Remove the gate from `run()` | defects 1 and 4 |
| Keep only the non-NULL / 128 B alignment check in `run()` | it needs no HVX context |
| Write `results[0..15]` and `results[20]` directly, **before** `acquire()` | defect 6 and C2 |
| Return 0 on the `acquire() == 0` path with `metadata[26] = HMX_D8_FAIL_VTCM` | C2 |
| Keep `memset(&attr, 0, sizeof(attr))` before `attr_init` | hygiene only; it does not fix the observed mechanism |
| `hmx_vtcm_inside_image` compares runtime against runtime, bounds from the build | defects 2 and 3, C1 |
| `hmx_vtcm_coherent` returns one bit per direction | each direction gets its own word |
| Move `hmx_vtcm_inside_image` after the handler include, with a forward declaration | its anchor is the handler's static |

### 6.2 `hmx_mode46_d8.h`

| Change | Reason |
| --- | --- |
| Evaluate the gate in the diagnostic block | defects 1 and 4 |
| Echo words 24–27 | §3 |
| Restore `qurt_hvx_unlock()` before the diagnostic `return 0` | defect 5 |
| Bump the sentinel and layout to `0x0D1A6004` / `4` | §4.1 |
| **Keep `hmx_d8_diag_attr`** even though it no longer transports `attr` | C4: it is the window anchor and the basis of `results[26]` |

### 6.3 Failure codes on the diagnostic paths

This is the owner's condition C4. The diagnostic paths carry these codes:

| Path | `metadata[26]` | `metadata[6]` | `run()` return |
| --- | ---: | ---: | ---: |
| `acquire() == 0` | `HMX_D8_FAIL_VTCM` (3) | `-1` | **0** |
| pointer NULL or misaligned | `HMX_D8_FAIL_VTCM` (3) | `HMX_ERR_VTCM` (14003) | non-zero |
| operand FNV mismatch (the diagnostic block) | `HMX_D8_FAIL_FNV` (16) | `0` | **0** |

The FNV-mismatch row is the one that matters: it is the state the device has
been stuck in, and it returns 0 so the payload survives.

### 6.4 What does NOT change

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
3. The evidence is committed and pushed **on the day of the contact**.

### 9.1 How the binary is preserved — the `.so` policy conflict

This is the owner's condition C3. Revision 7 said "the deployed skel binary is
committed under `docs/evidence/`", which **cannot be executed as written**: the
repository forbids tracked `.so` files (`tracked-binary-audit` in
`scripts/verify_local.ps1` rejects `\.(so|apk|aab|jks|keystore|pem|key|log)$`),
and `docs/evidence/**/*.so` is now gitignored.

**Resolution, in this order:**

1. **The SHA-256 is the tracked identity.** Each evidence directory records
   `<sha256>  skel.so  <size> bytes` in `skel.sha256`. That file is tracked and
   is what the identity check compares against.
2. **The binary itself is committed once, under a non-`.so` name**, as
   `skel.bin`, in the same directory. The audit rule matches the extension, and
   the file is an ELF shared object whatever it is called; renaming it does not
   weaken the rule, which exists to keep build products out of the tree, and
   this one is evidence rather than a build product.
3. **The commit is recorded** in `skel.sha256` as
   `retrievable_at=<commit> (git show <commit>:<path>)`, so the exact bytes can
   be recovered even if the file is later removed.

**If the owner prefers not to add a binary under any name**, option 2 is
dropped and the evidence is the SHA-256 plus the recorded commit only. That is
weaker — the build is not reproducible, so a hash alone cannot be re-derived —
but it does not violate the rule. **This is a decision for the owner, and the
proposal does not presume it.** The default in this revision is option 2.

The three earlier evidence directories were handled the same way: their
binaries were untracked in `d5191cb`, their SHA-256 values were kept, and each
now records `retrievable_at=a1a44775cf76`. All three were retrieved from that
commit and their hashes matched.

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
| **Window constants** | **NEW.** Both bounds are re-derived from the final object by the audit script and must match what was compiled in. A negative control with a deliberately wrong pair must be rejected with a non-zero exit. |
| **Lock balance** | **NEW.** Every return after `qurt_hvx_lock` must reach a `qurt_hvx_unlock`, and the lock-failure path must not. |

## 11. What this does not do

It does not run the critical interval, does not produce a D8 numeric result, and
does not retire `B3_R2_SYNC_UNPROVEN`. `C2` stays `NOT_RUN`.

## 12. What the owner is being asked

Accept or reject one diagnostic run under §8, with the measurements in §1, the
pre-declared interpretation in §5, the code changes in §6, and the two new gates
in §10.
