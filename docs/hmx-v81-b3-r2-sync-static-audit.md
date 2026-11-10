# HMX / HVX Synchronization Static Audit — B3+R2 Sequential Operand Supply

Status: **read-only audit. No DSP source was modified, nothing was built, no ADB
or device contact, no timing measured, no commit.** PR #14 stays Draft.

Scope: `experiment/hmx-acc-residual-feedback`, HEAD `969e26e`.

This document answers one question only:

> On which documented or demonstrated basis can software know that an HMX MMA
> packet has finished reading a VTCM operand tile, so that the same tile can be
> overwritten for the next MMA in the B3 three-product stream?

It does **not** authorise Mode 46, and it does not close
`B3_R2_SYNC_UNPROVEN`.

---

## 1. Method and evidence classes

Four independent kinds of evidence were collected:

| Class | What it establishes | What it never establishes |
|---|---|---|
| **SPEC** | A primary Qualcomm document states the guarantee | — (none obtained; see §2) |
| **BINARY** | The shipped DSP object contains (or does not contain) an instruction | That the hardware honours the intent |
| **SOURCE** | What the code asks for, and its lexical order | What the hardware does in which order |
| **DEVICE** | A previously executed run produced the registered output | Anything about an untested instruction interleaving |

Explicit rule applied throughout: **source order, emitted packet order, issue
order and completion order are four different things.** Only the fourth is what
the B3 stream needs, and only a specification or a device experiment can speak
about it.

### 1.1 Artifacts examined

| Artifact | Identity |
|---|---|
| Built DSP skel | `host_tests/hmx_acc_residual_rpc/hexagon_ReleaseG_toolv19_v81/ship/libhexatrain_hmx_acc_residual_skel.so`, SHA-256 `0d402c587dd94067e6a2248918831f347097f19bb9afd8812b2c27e7e31f7618` |
| Disassembler | `hexagon-llvm-objdump` from Hexagon Tools 19.0.07 (Hexagon SDK 6.6.0.0) |
| Assembler opcode table | `hexagon-llvm-mc` from the same toolchain (string extraction only; nothing was assembled) |
| QAIRT | 2.48.40.260702 (fixed root); bundled documentation inspected without modification |
| DSP source | `host_tests/hmx_acc_residual_rpc/probe_dsp.c` (`run_mode43_one_trial`, `run_mode44_one_trial`, `run_mode45_one_trial`, `run_one`) |
| CPU contract | `hmx_b3_r2.h`, `docs/hmx-v81-b3-r2-full-tile-prereg.md` §4 |

Pre-existing build output was read as an artifact. It was **not** rebuilt: no
new DSP object was produced by this audit.

---

## 2. Primary documentation: what could and could not be established

**No primary-source guarantee for HMX operand-read completion was obtained.**

| Source | Attempt | Result |
|---|---|---|
| Qualcomm Hexagon V81 HMX PRM, 80-N2040-62 Rev. AA | PDF fetch | Text not extractable by the reader; no section/page could be read |
| Qualcomm Hexagon V81 PRM index (80-N2040-64) | HTML fetch | Returned a documentation shell, no manual body |
| Qualcomm Hexagon V79 PRM "Memory" topic | HTML fetch | Shell only |
| QAIRT SDK 2.48.40.260702 bundled docs | `grep -rli -E "vtcm"` and `-E "hmx|syncht"` over all bundled `.html` | **0 hits** — the installed documentation set is the QNN/Genie API set, not the HTP op-package internals |
| Hexagon Tools 19.0.07 `include/` | `grep -rl "syncht"`, `grep -rl -E "mxmem|mxclracc"` | **0 hits** |
| `target/hexagon/include/hexagon_protos.h` | case-insensitive `synch` | 6 hits, all `dmsyncht` / `dmtlbsynch` (DMA). **No bare `syncht` intrinsic, no MX intrinsic.** |
| `target/hexagon/include/hvx_hexagon_protos.h` | case-insensitive `synch` | **0 hits** |

`syncht` itself does exist as an instruction mnemonic (`MCK_syncht` appears in the clang instruction tables), alongside `dmsyncht` and `dmtlbsynch`. So: the ISA has the instruction, the DMA variants have C intrinsics (`Q6_dmstart_A`, `Q6_R_dmsyncht`, `Q6_R_dmtlbsynch`), and **`syncht` and every MX operation have no C intrinsic in the headers this project builds against** — they are reachable only through inline assembly.

A citation supplied during review — V81 HMX PRM §2.2 "HMX reads operands from
VTCM" — is recorded here as **reviewer-supplied, not independently re-read in
this audit**, because the PDF could not be read here. The same applies to the
V73 HVX PRM §3.7 sequential-consistency statement: the architecture generation
in this project is V81, so a V73 statement is at best evidence, never a V81
guarantee.

Consequence: **every dependency below is at most `IMPLEMENTATION_SUPPORTED` or
`UNRESOLVED`. None is `SPEC_CONFIRMED`.**

---

## 3. ISA surface: is there an HMX wait instruction at all?

String extraction from the Hexagon 19.0.07 assembler image yields exactly five
`mx`-prefixed mnemonics:

```
mxclracc
mxmem
mxmem2
mxshl
mxswapacc
```

A search of the same image for any `mx`/`hm`/`acc` mnemonic containing
`wait|sync|fence|flush|bar|done|idle` returned **nothing**.

Observations, stated at the strength the evidence supports:

- The toolchain's MX surface is small and contains **no obvious completion or
  fence primitive**. That is evidence of absence *in this toolchain's opcode
  table*, not proof that no such mechanism exists in silicon (the MX decode
  space could contain instructions the MC layer does not expose).
- `mxswapacc` is the one mnemonic whose name suggests an ACC state transition.
  Its semantics are **unknown to this audit**; it is a candidate to investigate,
  not a solution.
- Because there is no C intrinsic for MX (§2), a B3 implementation has no
  compiler-visible way to express "wait for MX"; the only expressible ordering
  is inline-asm order plus clobbers. `syncht` is likewise assembly-only here,
  whereas the DMA syncs (`dmsyncht`, `dmtlbsynch`) do have intrinsics.

---

## 4. What the shipped binary actually contains

### 4.1 No synchronization instruction exists anywhere in the skel

Disassembling the whole `.so` and counting `syncht`, `barrier`, `dmwait`,
`dmstart` and bare `sync` mnemonics gives **0**.

Therefore **every device result this project has ever recorded — Modes 40, 41,
42, 43, 44 and 45 — was produced with no explicit HMX/HVX synchronization
instruction at all.**

### 4.2 The mode45 MMA → convert → store → read-back sequence (BINARY)

`run_mode45_one_trial`, instruction packets in order:

```
1ca8  { call memset }                     <- DSP writes the operand tiles
1cb8  { mxclracc.hf }
1cbc  { r2 = #0x77c ; r3 = #0x7ff }       <- mxmem rt operands
1cc4  { activation.hf = mxmem(r18,r2)
1cc8    weight.hf = mxmem(r27,r3) }       <- MMA packet (loads share one packet)
1ccc  { cvt.hf = acc(r19) }               <- Rs = 0x0001, next packet, no wait
1cd4  { mxmem(r20,r2) = cvt }             <- HMX store into out_pos
1cd8  { r3 = #0x2000 }
1cdc  { cvt.hf = acc(r3) }                <- Rs = 0x1000
1ce0  { mxmem(r25,r2) = cvt }             <- HMX store into out_neg
1cec  { r3 = r20 }
1cfc  { r5 = memub(r3++#0x1) }            <- scalar byte read of the region
...                                        just stored by HMX (FNV-1a checksum)
```

Registered counts for this symbol match the disassembly audit's expectations:
`mxclracc` 2, MMA packets 2, `bias = mxmem2` 3, `cvt.hf = acc` 4,
`mxmem(...)= cvt` 4, HVX `vsub` 2.

### 4.3 The three-product stream, as built (BINARY)

Inside `run_one`, the B3 branch emits three MMA packets with **nothing between
them**:

```
26d0  { mxclracc.hf }
26d4  { r3 = #0x77c ; r4 = #0x7ff }
26dc  { activation.hf = mxmem(r20,r3)      MMA #1  Ahi * Whi
26e0    weight.hf   = mxmem(r23,r4) }
26e4  { activation.hf = mxmem(r20,r3)      MMA #2  Ahi * Wlo
26e8    weight.hf   = mxmem(r18,r4) }
26ec  { activation.hf = mxmem(r21,r3)      MMA #3  Alo * Whi
26f0    weight.hf   = mxmem(r23,r4) }
26f4  { bias = mxmem2(r19) }
26f8  { bias = mxmem2(r2) }
```

Read the four distinct base registers: `r20`/`r23`/`r18`/`r21` — that is
`Ahi`, `Whi`, `Wlo`, `Alo`, **each in its own resident VTCM tile**. Three MMA
packets, four operands, zero copies and zero overwrites between them.

### 4.4 Compiler behaviour (SOURCE vs BINARY)

Each inline-asm block became its own packet, in source order, and no memory
operation was moved across a block boundary. The `"memory"` clobber is doing
exactly what it claims: it stops the **compiler**. It emits no instruction, so
it also does nothing to the **hardware**, which is why §4.1 finds no fence in
the object.

---

## 5. The decisive scope finding: what the existing device success does cover

The device results already registered (`B2_THREE_PRODUCT_CONFIRMED`,
`R2_ACC_REUSE_SUPPORTED`, the Mode 40/41/42 bit-exact cells, the Mode 45 S1/S2
slots) all share one structural property:

> Every operand tile an MMA reads is either written **before any MMA in the same
> symbol is issued**, or is never written again.

| Behaviour | Exercised on device? | Meaning |
|---|---|---|
| DSP scalar store → HMX operand read (tile written first, MMA reads it) | **Yes** — every mode: `memset`/`memh` then `mxmem` | Store→read visibility works on this part with no explicit sync |
| MMA → `cvt.hf = acc` with no intervening instruction | **Yes** — modes 41, 42, 45, bit-exact | ACC dependency is resolved by hardware, not by software |
| HMX `mxmem(...) = cvt` store → later scalar read of the same bytes | **Yes** — mode 45 FNV checksum loop, matched | HMX store completion vs a later load, one direction |
| HMX `mxmem(...) = cvt` store → later **HVX vector** read of the same bytes | **Yes** — mode 45 `Q6_Vhf_vsub_VhfVhf` recon loop, S1/S2 matched | HMX store completion vs a later HVX load, one direction |
| **HVX store overwriting a tile an already-issued MMA is reading** | **No — never** | This is the B3 stream hazard |
| **Any store overwriting `wgt`/`act` between two MMA packets** | **No — never** | Idem |
| Convert #1/#2 issued while an operand copy is in flight | **No — never** | Idem |

The last three rows are precisely `B3_R2_SYNC_UNPROVEN`. The first four rows
show the hardware behaves sensibly in the easy direction; they are **not**
evidence for the hard direction, exactly as the review required.

---

## 6. Dependency-by-dependency classification

| # | Dependency (B3 stream) | SPEC | BINARY / SOURCE | DEVICE | Class |
|---|---|---|---|---|---|
| D1 | MMA #1 has finished reading `wgt` = `Whi` before the `Wlo` copy overwrites `wgt` | none | no fence present; nothing enforces it | not exercised | **UNRESOLVED → DEVICE_ONLY** |
| D2 | MMA #2 has finished reading `wgt` = `Wlo` before the `Whi` restore overwrites `wgt` | none | idem | not exercised | **UNRESOLVED → DEVICE_ONLY** |
| D3 | MMA #2 has finished reading `act` = `Ahi` before the `Alo` copy overwrites `act` | none | idem (the `act` tile has the same hazard as `wgt`) | not exercised | **UNRESOLVED → DEVICE_ONLY** |
| D4 | `Alo`/`Whi` stores are visible to HMX when MMA #3 is issued | none | no fence; store precedes packet in issue order | **partly**: store→read is routine (tile writes before MMA), but never *overwriting a live operand* | **IMPLEMENTATION_SUPPORTED (fresh tile) / DEVICE_ONLY (overwrite)** |
| D5 | MMA #3's ACC update is reflected in Convert #1 | none | `cvt.hf = acc` immediately after the MMA packet, no wait | **yes**, bit-exact, modes 41/42/45 | **IMPLEMENTATION_SUPPORTED** (spec still unknown) |
| D6 | Convert store completes before the HVX recon reads `out_pos`/`out_neg` | none | no fence; mode 45 shows the read-back loop | **yes**, mode 45 S1/S2 matched | **IMPLEMENTATION_SUPPORTED** |
| D7 | HVX recon result visible to the host readback | none | RPC return boundary | yes (registered slots) | **IMPLEMENTATION_SUPPORTED** |

D5–D7 are the comfortable ones and they are already carrying the project's
results. D1–D3 are the entire remaining risk of the **streaming** design, and
they are the only place where "it worked before" is worthless, because no
previous run has ever performed the operation.

### 6.1 Re-evaluation under candidate B (§7)

| # | Streaming | Candidate B (phase reuse) |
|---|---|---|
| D1 `Whi` overwritten by `Wlo` | UNRESOLVED / DEVICE_ONLY | **DESIGN_ELIMINATED** — `Wlo` lives at `+6144`, `Whi` at `+4096`, no overwrite exists |
| D2 `Wlo` overwritten by the `Whi` restore | UNRESOLVED / DEVICE_ONLY | **DESIGN_ELIMINATED** — there is no restore |
| D3 `Ahi` overwritten by `Alo` | UNRESOLVED / DEVICE_ONLY | **DESIGN_ELIMINATED** — `Alo` lives at `+2048` |
| D4 store visible to HMX before the MMA | IMPLEMENTATION_SUPPORTED / DEVICE_ONLY | unchanged direction, one occurrence: the four prologue tiles before MMA #1 |
| D5 MMA #3 ACC into Convert #1 | IMPLEMENTATION_SUPPORTED | IMPLEMENTATION_SUPPORTED (unchanged) |
| D6 Convert store before the HVX recon read | IMPLEMENTATION_SUPPORTED | IMPLEMENTATION_SUPPORTED (unchanged) |
| D7 HVX recon visible to the host | IMPLEMENTATION_SUPPORTED | IMPLEMENTATION_SUPPORTED (unchanged) |
| **D8** MMA #2's operand reads finished before `out_neg`/`out_recon` reuse `Wlo`/`Ahi` | — | **UNRESOLVED → DEVICE_ONLY**, but one MMA and one convert later than streaming's overwrite, and guarded by assumption A1 (in-order MX) if one is willing to lean on it |

Single-run caveat: every DEVICE entry above is one build, one run, one part. It
is registered evidence, not a stress campaign, and it is not independently
re-verifiable from the repository (raw device logs are not committed).

---

## 7. Can the hazard be designed away instead of synchronized?

**Yes — by lifetime reuse. The earlier "cannot be avoided" conclusion is
withdrawn.** It assumed that all three output tiles must be reserved while the
four operand tiles are live, but the outputs only exist after the last MMA.

### 7.1 Why the earlier arithmetic was wrong

The impossibility result below still holds for *simultaneous* residency, and
only for that:

| Variant (all regions live at the same time) | Bytes | Fits 12,288 B? |
|---|---:|---|
| 4 operands + 3 outputs + 2 bias sets | 8,192 + 6,144 + 512 = 14,848 | no |
| 4 operands + 2 outputs (`out_recon` out of VTCM) + bias | 8,192 + 4,096 + 512 = 12,800 | no, over by 512 B |
| 4 operands + 2 outputs + one 256 B bias set | 8,192 + 4,096 + 256 = 12,544 | no, over by 256 B |

Phasing the outputs changes the answer entirely, because during the MMA phase
there are no outputs at all.

### 7.2 Candidate B — phase-reused layout (evaluated, CPU-only)

Four operand tiles resident for the whole MMA phase; outputs land afterwards in
regions whose owner has already been consumed.

| VTCM offset | MMA phase | Output phase | Owner's last use |
|---:|---|---|---|
| `+0` | `Ahi` | `out_recon` | MMA #2 |
| `+2048` | `Alo` | — (never written) | MMA #3 |
| `+4096` | `Whi` | — (never written) | MMA #3 |
| `+6144` | `Wlo` | `out_neg` | MMA #2 |
| `+8192` | — | `out_pos` | none (fresh region) |
| `+10240` | bias set 0 | bias set 0 | — |
| `+10496` | bias set 1 | bias set 1 | — |

| Quantity | Streaming (registered) | Candidate B |
|---|---:|---:|
| VTCM during the MMAs | 10,752 B | **8,704 B** |
| VTCM peak | 10,752 B | 10,752 B |
| Mid-kernel operand transfers | 3 tiles / 48 vectors / 6,144 B | **0** |
| Prologue operand transfers | 5 tiles / 80 vectors | 4 tiles / 64 vectors |
| Total operand transfers | 8 tiles / 128 vectors | **4 tiles / 64 vectors** |
| DSP L2/TCM outside VTCM | 6,144 B (3 staged tiles) | **2,048 B** (the Plain subcase's finished output) |
| Operand written between MMAs | 2 tiles, twice | **0** |

Candidate B also drops the `Whi` restore entirely: with `Wlo` resident at
`+6144`, nothing overwrites `Whi` at `+4096`, so the staging tile, its FNV
checksum and fault 14 have no counterpart. That is less machinery, not more.

A second variant (out_pos `+0`, out_recon `+2048`, out_neg `+6144`, `Whi`
unused) peaks even lower at 8,704 B, but it recycles `Alo`, an operand of the
*last* MMA. Candidate B never touches a MMA #3 operand in-kernel, which is why
it is the one proposed.

### 7.3 What candidate B does and does not remove

The D1–D3 overwrite dependencies are **removed by construction**: no operand
region is written between MMA #1 and MMA #3, so there is no window in which a
live tile can be replaced.

What remains is a single, narrower dependency (**D8**): `out_neg` (over `Wlo`)
and `out_recon` (over `Ahi`) are written after MMA #3 and both converts, so the
hardware must have finished MMA #2's operand reads by then. In the streaming
design the equivalent overwrite sits *directly between* MMA #2 and MMA #3, with
nothing in between; here an MMA and a convert intervene.

A tempting argument — "Convert #1 returned the correct accumulated value, so
MMA #1–#3 all consumed their operands, so MMA #2's reads are certainly done" —
rests on an **in-order MX execution** assumption (A1) that is itself
undocumented. It is recorded as an assumption, not as proof.

### 7.4 Candidate B, as executed in the CPU model

`hmx_b3_r2_full_tile_sim.c` now models both layouts and asserts:

- layout fit and alignment for both phases; four pairwise-disjoint operand
tiles; bias sets outside the operand block;
- `last_use(owner) < first_write(output)` for every aliased output;
- operand immutability across the whole MMA phase, by FNV-1a over each of the
four tiles before and after;
- the Plain subcase's output survives the B3 prologue only because it is saved
out of VTCM first (2,048 B), and that save is shown to be load-bearing;
- **the 1,024-cell raw bits of both subcases are bit-identical to the streaming
model** — the same arithmetic, a different memory schedule.

New faults (17 total, all fail closed): 15 an output region written before its
owner's last MMA; 16 a live operand tile clobbered; 17 `out_pos`/`out_neg`
mapped to one region.

Still not addressed by any of this: whether D8 holds on silicon. Candidate B
narrows the hazard; it does not retire `B3_R2_SYNC_UNPROVEN`.

---

## 8. Candidate synchronization mechanisms and where they would go

None of these is authorised or implemented. They are the candidates a Mode 46
pre-registration would have to choose between, with the honest status of each.

| Candidate | Position | Status |
|---|---|---|
| `syncht` / `__syncht()` | After MMA #n, before the operand copy | Documented (outside V81, unverified here) as completing **outstanding memory transactions** for the thread. **Not documented to cover MX operand consumption.** Using it as "MMA is done" would be exactly the inference the review forbids. |
| `barrier` (packet attribute) | Same | Packet-level scheduling constraint for Hexagon threads; says nothing about the MX unit's VTCM read completion. |
| A dummy `cvt.hf = acc` between MMA #n and the overwrite | After MMA #n | **Empirically strong only for the ACC**: D5 shows convert-after-MMA works with no software wait. But `cvt` consumes ACC, not VTCM; there is no evidence it implies the operand reads finished, and no document says so. |
| `mxswapacc` | Investigate | Mnemonic exists in the assembler; semantics unknown here. |
| `mxmem(...) = cvt` then read that byte back, and only then overwrite | After MMA #n | Would chain two empirically-supported behaviours (HMX store → later load) into a software "round trip". Whether MX reads are ordered *strictly* before MX stores retire is **not** established. |
| `Q6_R_dmsyncht` / `Q6_R_dmtlbsynch` / `Q6_dmstart_A`, if staging by DMA instead of HVX | Around the copies | These are the DMA-synchronisation intrinsics exposed by `hexagon_protos.h`; the headers give their assembly form, not a prose guarantee about other hardware agents. They constrain the **DMA transfer**, not the MMA's VTCM read. Switching the copies from HVX to DMA would turn the hazard into a DMA-vs-MX question, which is equally undocumented here. |
| Split into two kernel launches with a host boundary between MMA #2 and MMA #3 | Between phases | Removes the intra-kernel hazard by construction, at the cost of an extra RPC and loss of the single-attempt invariant. Would validate the *numbers* but not the streaming design. |

There is no candidate here that a document currently backs for D1–D3. Picking
one and shipping it would be a guess with a plausible name.

---

## 9. What this audit closes, and what only a device can close

| Question | Closed by this audit? |
|---|---|
| Is a compiler barrier doing anything at hardware level? | **Closed: no.** `"memory"` clobber emits nothing; the shipped object contains zero fences. |
| Does the project already synchronize HMX anywhere? | **Closed: no.** 0 sync mnemonics in the whole skel; all device results came from unsynchronized issue order. |
| Has the operand-overwrite interleaving ever run on device? | **Closed: no.** All executed modes use resident, never-rewritten operand tiles. |
| Is there an MX wait instruction exposed by the toolchain? | **Scoped finding only**: five MX mnemonics in this SDK's assembler, none of them a wait/sync. Nothing here proves silicon has no such decode. |
| Does `syncht` guarantee MX operand consumption? | **Not closed.** Not documented; not testable statically. |
| Do D1–D3 hold on the part? | **Still unanswered, but no longer required**: candidate B removes the operation those dependencies describe. |
| Does D8 hold on the part? | **Not closable by static analysis.** It is a timing property of the MX unit against HVX stores, one step further from the MMA than streaming's hazard. |

Therefore the minimal honest gate before any Mode 46 work is a **device
experiment that deliberately commits the hazard**, not a code review:

1. `mxclracc`; issue MMA #1 with weight tile payload `P`; then overwrite the
   same `wgt` bytes with payload `Q` using HVX vector stores; issue MMA #2; single
   convert; store.
2. If MMA #1's operand read completed before the overwrite, the registered
   result corresponds to `P` consumed and `Q` untouched by MMA #1; a corrupted
   read shows up as a wrong cell or a non-finite value, not as a plausible
   number.
3. Repeat with a swept distance between the MMA packet and the overwrite
   (varying NOP/packet count) to expose a window instead of sampling one
   instant.
4. A single clean run does **not** prove safety; only a documented guarantee
   can, and a stress campaign can only bound the risk.

This design is recorded as a candidate. It is **not authorised**, not
implemented, not built, and not scheduled.

---

## 10. Verdict

Unchanged, and now with its cost quantified and its scope narrowed:

- `B3_R2_FULL_TILE_CPU_READY` — CPU-scope verdict, untouched. Candidate B
  reproduces all 1,024 cells of both subcases bit-identically.
- `B3_R2_SYNC_UNPROVEN` — **confirmed as the correct classification**, on the
  narrower ground of D8. For D8 there is no specification, no wait primitive in
  the examined toolchain, no binary evidence and no device exercise. D1–D3 no
  longer apply, because candidate B never overwrites a live operand.
- `R2_ACC_REUSE_SUPPORTED` — untouched, limited to the Mode 45 S1/S2
  configuration.
- Candidate B: `CPU_VALIDATED_PREFERRED_DESIGN` — a **design-comparison**
  label only. It records that the phase-reused layout passed the CPU model and
  is preferred over streaming as the basis for a *future* Mode 46
  pre-registration. It does **not** mean a device run succeeded, does not make
  Candidate B the registered layout, and does not authorise Mode 46.
- Full-tile device accuracy `NOT_RUN`; E-3C2 `UNTRIGGERED`; Mode 46 not
  implemented, not built, not run.

The uncomfortable result of this audit is that the B3 stream's risk is **not**
in the arithmetic, which is already frozen and reproduced, and **not** in the
VTCM budget, which is proven arithmetically once the output phase is allowed to
reuse operand regions. It is in a single hardware question — *when does an
issued MMA stop reading its operand tile* — for which the public documentation
available here says nothing, and which no amount of CPU modelling can answer.

### 10.1 Design recommendation

**Candidate B (phase-reused VTCM) is the preferred basis for any future Mode 46
work**, ahead of the registered sequential-streaming design:

- it removes the D1–D3 hazard instead of trying to synchronise around it;
- it halves the operand transfers (4 tiles / 64 vectors vs 8 / 128) and removes
  all 6,144 B of L2 operand staging;
- it removes the `Whi` restore, its checksum check and fault 14's whole class of
  failure;
- it produces bit-identical results in the CPU model, so adopting it costs no
  numerical evidence;
- it leaves exactly one unproven hardware dependency (D8), one MMA + one
  convert further from the MMA than the streaming hazard.

Its own costs: the layout must be written as a lifetime contract (a region's
first output write is ordered after its owner's last MMA), the between-subcase
save becomes load bearing (2,048 B), and the streaming design's 12 KiB margin
shrinks from 1,536 B to the same 1,536 B but with the operand phase at 8,704 B.

This is a recommendation, not an authorisation: candidate B is not registered,
not implemented in DSP, and does not change any verdict.
