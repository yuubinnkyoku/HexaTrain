# Candidate B — D8 Implementation Feasibility Audit

Status: **STATIC AUDIT, CPU-ONLY — `D8_IMPLEMENTATION_BLOCKED`**, 2026-11-10.
Scope: whether the revision-2 D8 minimal experiment can be carried by the
existing V81 DSP / FastRPC implementation, and which design defects must be
resolved before any DSP source change is written.

This document authorises nothing. No DSP source was modified, no DSP binary was
built, no QAIRT command was run, no ADB/device/FastRPC call was made, and Mode
46 was not registered or implemented. The registered states are unchanged:

| Label / state | Value after this audit |
| --- | --- |
| D8 pre-registration | **revision 2** (unchanged) |
| D8 CPU preparation | `D8_CPU_PREREG_READY` (unchanged) |
| D8 device | `NOT_RUN` |
| C2 device | `NOT_RUN` |
| Synchronisation | `B3_R2_SYNC_UNPROVEN` (unchanged) |
| Mode 46 | unregistered, `NOT_RUN` |
| PR #14 | Draft, open (Draft maintained) |
| Device-experiment approval | **`NOT GRANTED`** |

The verdict is a statement about the **registered experiment's implementability
on the current interface**, not about hardware behaviour, and not a claim that
D8 is true or false. It is `BLOCKED` because the registered evidence contract
and the existing `run()` metadata/selector machinery conflict in ways that must
be amended before an implementation can be honest; it is **not** blocked by the
IDL, by a missing instruction, or by a proven hardware limitation.

---

## 1. Baseline, scope and evidence read

### 1.1 Repository state at audit start

| Item | Value |
| --- | --- |
| Repository | `https://github.com/yuubinnkyoku/HexaTrain` |
| PR | **#14**, Draft, open, base `main`, head `experiment/hmx-acc-residual-feedback` |
| PR head / audit base HEAD | `5a20fd858a72c252c20606fc55f318e14c127189` |
| Existing worktrees | main worktree on the research branch with uncommitted C2 changes; three other worktrees, all clean |
| Audit worktree | `D:/ghq/github.com/yuubinnkyoku/HexaTrain-d8-audit-20261110`, new branch `codex/hmx-d8-feasibility-audit-20261110` at the same HEAD |
| Worktree status at start | clean; the existing main-worktree changes were not touched |

The audit used an independent worktree because the main worktree carries
pre-existing uncommitted C2 edits. Nothing in this audit reads from or writes
to that worktree.

### 1.2 Documents read

- `docs/hmx-v81-cand-b-d8-device-prereg.md` (revision 2) — condition table,
  fixed sequence, references, classifier, stop conditions, §10 CPU record.
- `docs/hmx-v81-cand-b-d8-evidence-contract.md` (revision 2) — 32-word metadata
  table, length/identity requirements, disassembly plan.
- `docs/hmx-v81-d8-vs-c2-experiment-order.md` — the D8-first rationale and the
  condition-selector risk.
- `docs/hmx-v81-b3-r2-sync-static-audit.md` — D8's evidence classes, the four
  orders, the absence of a completion primitive.
- `docs/hmx-v81-b3-r2-full-tile-prereg.md` — §4.6 phase-reused layout, the
  Candidate B order, the withdrawn timing claim.
- `docs/hmx-cb-fix-1-cpu-access-audit.md` — the CPU access-contract scope.
- `docs/hmx-v81-cand-c-c2-device-evidence-contract.md` — the parallel C2 ABI
  design (used only to show that the metadata-word collision is systemic, not
  D8-specific).
- `docs/hmx-v81-r2-device-prereg.md`, `docs/hmx-v81-r2-device-implementation-plan.md`,
  `host_tests/hmx_acc_residual_rpc/README.md` — the mode 41/42/43/45 device and
  CPU records used as implementation precedent.

### 1.3 Code read

| File | Relevant content |
| --- | --- |
| `host_tests/hmx_acc_residual_rpc/hexatrain_hmx_acc_residual.idl` | one method, `run(rout sequence<long> results, rout sequence<long> metadata, in long q, in long mode)` |
| `probe_dsp.c` | `run()` gates and metadata ownership; mode 40/41/42/43/44/45 handlers; VTCM acquire/lock/cleanup; bias/convert/store assembly; HVX 16 × 128 B loops |
| `probe_host.c` | CLI q/mode gates; rpcmem allocation; identity pre-check; result/metadata lengths; mode-45 contract gate |
| `hmx_mode45.h` | 11-slot / 32-word precedent; bias/hvx constants; VTCM map |
| `hmx_b3_r2.h` | Candidate B offsets, tile/VTCM constants, PRM store mapping |
| `hmx_identity.h` | build-id protocol, architecture echo, strict non-zero gate |
| `hmx_d8_min_sim.c` | CPU trace, fill-mask derivation, rev-1 rejection, classifier |
| `hmx_b3_r2_full_tile_sim.c` | Candidate B phase-reused order, post-MMA FNV checks, store mapping |
| `hmx_d8_independent_test.py` | independent numeric/classifier audit (29 checks) |
| `scripts/audit_hmx_mode45_disasm.ps1` | the only existing disassembly audit tool |
| `scripts/run_hmx_d8_sim.ps1` | the D8 CPU gate (extended by this audit) |

### 1.4 Existing binaries and disassembly artefacts

**None were available in this checkout.** There is no Hexagon SDK
(`HEXAGON_SDK_ROOT` unset), no `hexagon-llvm-objdump`, and no built
`libhexatrain_hmx_acc_residual_skel.so` under the worktree. The `run_experiment.ps1`
build expects Hexagon SDK `6.6.0.0` / toolv19; QAIRT is pinned separately at
`2.48.40.260702` (`docs/agent/qairt-policy.md`). This audit therefore could not
re-disassemble the shipped skel. It uses the **documented** prior disassembly
findings as method precedent only, never as D8 evidence.

### 1.5 What this audit adds

- An independent, CPU-only implementation-feasibility verifier:
  `host_tests/hmx_acc_residual_rpc/hmx_d8_implementation_feasibility_test.py`,
  wired into `scripts/run_hmx_d8_sim.ps1` (Fast and Host profiles).
- Machine checks for the layout, fill ledger, mask table, sequence, metadata
  completeness, current interface baseline, and a re-derivation of the
  mode-42 DSP fixture generator against the registered operand FNVs.
- This document, which records every pre-implementation defect found and the
  future disassembly procedure.

No registered value, digest, expected bit, mask, reference array or approval
state was changed.

---

## 2. The registered experiment in one page

Conditions (revision 2):

| Condition | Operand fills after MMA #3 | Mask (bit0 out, bit1 Wlo, bit2 Ahi) |
| --- | --- | ---: |
| A | none | `1` |
| B | Wlo and Ahi | `7` |
| C | Wlo only | `3` |
| D | Ahi only | `5` |
| E | CPU-only fault control | never a device condition |

Fixed VTCM map (bytes from the 12 KiB reserve base):

| Offset | Size | MMA phase | Output phase | Last MMA use |
| ---: | ---: | --- | --- | --- |
| `+0` | 2,048 | `Ahi` | `out_recon` alias / Ahi fill (B/D) | MMA #2 |
| `+2048` | 2,048 | `Alo` | never written | MMA #3 |
| `+4096` | 2,048 | `Whi` | never written | MMA #3 |
| `+6144` | 2,048 | `Wlo` | `out_neg` alias / Wlo fill (B/C) | MMA #2 |
| `+8192` | 2,048 | — | `out_pos` | none |
| `+10240` | 256 | bias set 0 | bias set 0 | — |
| `+10496` | 256 | bias set 1 | bias set 1 | — |
| **used** | **10,752** | | | |
| `+10752` | 1,536 | guard tail (must stay `0x00`) | | |

Registered critical order:

```text
pack Ahi/Alo/Whi/Wlo; bias set 0; bias set 1
mxclracc
MMA #1 Ahi*Whi; MMA #2 Ahi*Wlo; MMA #3 Alo*Whi
fill out_pos (+8192, 16x128 B, 0xA5A5)
condition fills: Wlo (+6144, B/C), Ahi (+0, B/D)
Convert #1: cvt.hf = acc(0x0001), bias set 0, shape 0, scale 64
Store #1: mxmem(+8192) = cvt
read back 1,024 cells + 32 metadata words
```

The map is byte-for-byte the Candidate B §4.6 phase-reused layout, and the
three MMA operand pairs are the registered B3 pairs. The audit checks that the
map, masks and order are internally consistent (they are), and then asks whether
the *existing interface* can carry the run.

---

## 3. D8 condition selection — Phase 1

### 3.1 What exists

- The IDL type is not an obstacle: `q` is `in long`, so `0..3` is expressible.
- The transport is observable: identity mode 30 echoes the received `q` in
  `results[1]`, and `hmx_identity.h` rejects `identity[1] != q`
  (`HMX_IDENTITY_FAILURE_Q`). The host runs identity mode first on the same
  handle with the same `q`.
- The condition selector is **not** a free selector today. `probe_dsp.c`
  rejects `q < 6 || q > 10` before dispatch, and `probe_host.c` mirrors the same
  gate. Existing modes bind `q` to fixed fixture/encoding values (6, 8 or 10);
  mode 30 (identity), modes 40–45 all require `q = 6`. `q = 0..3` is rejected
  before any mode handler runs.

### 3.2 Consequence for the registered design

The evidence contract (§3.1) proposes `q = 0=A, 1=B, 2=C, 3=D`, and its
metadata word 3 expects `0/1/2/3`. As written, that cannot be carried:

1. the DSP `run()` rejects `q < 6`;
2. the host CLI rejects `q < 6` before the RPC;
3. the identity pre-check runs with the same `q`, so it would also be rejected
   (or the host would have to run identity with a different `q`, weakening the
   same-q transport evidence);
4. widening the global gate to `0..10` would change the accepted argument
   surface of every legacy mode, several of which currently interpret `q` as a
   fixture parameter.

Three encodings are possible. They are not equivalent registrations:

| Option | What changes | Cost / risk |
| --- | --- | --- |
| **q = 6..9**, condition = `q - 6`; word 3 echoes `0..3` | no gate change; no identity change; a mode-46-only mapping | requires a documented selector-encoding amendment (the contract currently says `q = 0..3`) |
| **q = 0..3**, widen the gate for mode 46 *and* identity only | DSP + host special cases; identity must accept 0..3 | more surface area; identity q evidence becomes mode-dependent |
| new `run()` argument / input sequence | IDL/stub/skel and ABI change | violates the pre-registration's "no ABI change" premise |

The minimal-change path is the first: it keeps both gates, keeps the identity
pre-check, and yields the registered word-3 echo. It still requires a
**pre-implementation registration amendment**, because the selector encoding is
part of what the run means. This audit does not amend revision 2 unilaterally.

### 3.3 Mode dispatch

There is no `mode == 46` handler. `MODE_MAX = 45` in the DSP and
`HMX_PROBE_MODE_MAX = 45` in the host; `run()` writes its generic metadata block
and dispatches modes 43/44/45/`run_one`. Adding a D8 mode requires:

- a new mode number and its registration (Mode 46 is **unregistered**);
- a new DSP handler and a `MODE_MAX`/host-max update;
- a result-length gate of at least 1,024 (today the generic gate is `< 4`, and
  only mode 45 raises it to 11);
- a host result buffer of 1,024 slots (today 11) and a metadata length of 32
  (already the mode-45 size);
- mode-specific metadata ownership (next section);
- the host-side D8 contract/classifier, which does not exist yet.

### 3.4 Phase 1 verdict

| Question | Verdict |
| --- | --- |
| Can `q` carry the condition without an IDL change? | **Yes for the type and transport; no for the current gate as written.** |
| Can `0..3` be expressed in the existing IDL? | **Yes** (`long`); rejected by DSP/host validation, not by the IDL. |
| Does `mode`/`q` affect existing modes? | **Yes** — `q` is a fixture selector and is bound per mode; a global widening changes legacy behaviour. |
| What is needed for four conditions in one build? | A mode-46 handler with a condition switch, a selector encoding decision, 1,024-slot readback, and the metadata layout fix. |
| Is an ABI change required? | **No IDL signature change.** The ABI *behaviour* (lengths, metadata layout, error path) changes; see §6. |

**Classification: `IMPLEMENTATION_CHANGE_REQUIRED`** (selector + dispatch +
evidence layout), with **`BLOCKED`** on the Mode 46 registration prerequisite.

---

## 4. Registered instruction sequence — Phase 2

### 4.1 The three MMA packets already exist, on device

Mode 41 (B-2) is the strongest existing precedent:

- three MMA packets `Ahi·Whi → Ahi·Wlo → Alo·Whi` into the **same ACC**, with
  no convert, store or clear between them;
- one `mxclracc` at the start of each subcase;
- one bias-0, scale-64 convert and one store at the end;
- device result `B2_THREE_PRODUCT_CONFIRMED` at the pre-registered bits
  (`0x5401` / `0x53ff`, build id `0xe99839a1`).

Mode 42's B3 trial has exactly the D8 operand pairs and the D8 tile offsets:

| Packet | Activation | Weight | Expected offsets |
| ---: | --- | --- | --- |
| #1 | `act_hi` | `wgt_hi` | `+0`, `+4096` |
| #2 | `act_hi` | `wgt_lo` | `+0`, `+6144` |
| #3 | `act_lo` | `wgt_hi` | `+2048`, `+4096` |

The source uses the same register encodings as modes 43/45:
`spatial_mask = 14 << 7`, `act_rt = spatial_mask | (31 << 2)`,
`wgt_rt = 2048 - 1`, `wr_rt = spatial_mask`, `cvt.hf = acc(1)` and
`mxmem(addr, wr_rt) = cvt`. These are the registered Convert #1 and Store #1
forms.

Caveat: mode 42's device run (E-3C1) was inconclusive for transport reasons, and
its *packing* was row-major (the mode-43 correction). Its arithmetic path is a
code template, not a device confirmation of the D8 layout. The project record
attributes the 3-packet ACC chain to mode 41 (`B2_THREE_PRODUCT_CONFIRMED`), a
PRM-packed 32×32 tile at four-slot scope to mode 43
(`E3C1R_LAYOUT_CONFIRMED_CLAMP`), and the bias-load-before-`mxclracc` /
16 × 128 B HVX patterns to mode 45 (`R2_ACC_REUSE_SUPPORTED`, Mode 45 only).
This audit did not re-verify those device transcripts (none are in the repo);
it uses them as design precedent. D8 is the composition, and the composition is
not device-tested.

### 4.2 The operand fixture is already in the DSP source

This is the most useful positive finding of the audit. The mode-42 `t8_build`
generator (same xorshift64 seed `0xE3C0117E5EEDF00D`, same anchor rows/lanes,
same `RN16(x)` / float-subtract residual split) is bit-identical to the G8 /
`g_ops_deploy` fixture that D8 registers. The new verifier re-derives the
generator independently and checks the four packed-tile FNVs:

```text
gen_fnv ahi=c6231266 alo=3b2d22d2 whi=8b88609f wlo=a66af695
```

These equal the four registered D8 FNVs. So the values are not a missing input
path. What is missing is the **composition**:

- mode 42 computes the four logical arrays but writes them **row-major**
  (mode 43's correction records this);
- mode 43 has the PRM pair-interleaved pack loop, but only for the two plain
  hi tiles (`act`, `wgt`);
- no existing mode writes the **packed lo** tiles at `+2048` and `+6144`.

D8 therefore needs the mode-43 pack loop extended to the lo tiles and to four
destinations. That is a small, well-understood code change, not an ABI change.
The runtime FNV echo (words 17–20) then verifies the packing on the device.

### 4.3 Bias placement: the template that must not be copied

- Mode 42 loads the two bias sets **after MMA #3** and before Convert #1.
- Mode 45 loads three bias sets **before `mxclracc`** and before the MMA.
- D8 revision 2 registers `BIAS 0`, `BIAS 1` **before `mxclracc`** and before
  MMA #1.

The target order is mode 45's prologue, not mode 42's. Copying mode 42's block
verbatim would produce a numerically equivalent convert (the bias register is
read at convert time) but a **different registered sequence**, which the
pre-registration explicitly treats as a different experiment. This is a
sequence-identity issue, not an arithmetic issue.

### 4.4 Between MMA #3 and the first fill

The C source can place the three `asm volatile` MMA packets, then the out_pos
fill, adjacently. What that guarantees:

- the three MMA asm statements are `volatile` and carry a `"memory"` clobber;
  GCC/Clang cannot delete them and cannot move a later normal store above a
  memory-clobber asm, so the fill stores cannot be hoisted before MMA #3;
- the compiler may still insert non-memory instructions (address arithmetic,
  register spills, branch setup) between the last MMA asm and the first fill
  store. Those are not memory accesses, but they change the instruction
  distance from MMA #3 to the first fill;
- the compiler may lower the fill loop to a counted loop, unrolled stores, a
  `memset`, or a helper call. Only the last is explicitly forbidden by the
  contract; the counted loop is a different *static* form from 16 explicit
  stores, which the contract's wording does not yet distinguish.

Only the stamped object can settle this. The source is not evidence that the
future machine order is the registered order. This is the central
`INSUFFICIENT_EVIDENCE` item of Phase 2.

### 4.5 Phase 2 verdict

| Item | Verdict |
| --- | --- |
| Three MMA packets with the registered operand pairs | **`STATICALLY_SUPPORTED`** (code exists; mode 41 proves the ACC chain on device) |
| ACC retention across the three packets | **`STATICALLY_SUPPORTED`** (mode 41 device result) |
| Bias loads before `mxclracc` | **`STATICALLY_SUPPORTED`** as a pattern (mode 45), but mode 42's template must not be reused |
| Convert #1 / Store #1 forms | **`STATICALLY_SUPPORTED`** (modes 42/43/45) |
| Exact emitted order and spacing after MMA #3 | **`INSUFFICIENT_EVIDENCE`** until disassembly |
| Fixture values | **`STATICALLY_SUPPORTED`**; packed-lo composition is new code |
| Condition branches | **`IMPLEMENTATION_CHANGE_REQUIRED`** (new mode handler + selector) |

---

## 5. HVX fill feasibility — Phase 3

### 5.1 The write form exists

Mode 45 already executes 16 iterations of a 128 B HVX loop into VTCM:

```c
for (int v = 0; v < 16; ++v) {
  HVX_Vector pv = *(const HVX_Vector *)((const uint8_t *)pos + v * 128);
  HVX_Vector nv = *(const HVX_Vector *)((const uint8_t *)neg + v * 128);
  *(HVX_Vector *)((uint8_t *)recon + v * 128) =
      Q6_Vhf_vsub_VhfVhf(pv, nv);
}
```

with `qurt_hvx_lock(QURT_HVX_MODE_128B)` held. The D8 fill needs the same
16 × 128 B store shape with a splat value:

```text
HVX_Vector sentinel = Q6_V_vsplat_R(0xA5A5A5A5);   /* every 16-bit lane = 0xA5A5 */
for v in 0..15: *(HVX_Vector *)(dst + 128*v) = sentinel;
```

`0xA5A5` is a normal negative FP16 (sign 1, exponent 20, non-zero mantissa), not
NaN/Inf/subnormal. The byte pattern is `A5 A5`, so a byte-splat/memset with
`0xA5` would also produce `0xA5A5` cells — which is exactly why the *machine
form* matters: a `memset` is not the registered 16 vector stores.

### 5.2 Addressing and coverage

- The VTCM pointer is required to be 2,048 B aligned by `run()`
  (`(uintptr_t)vtcm & (TILE_BYTES - 1u)`), so `+0`, `+6144`, `+8192` are all
  128 B aligned. The reserve request is 12,288 B.
- One fill is `16 × 128 = 2,048 B`. Three fills cover 6,144 B.
- The three fill ranges are pairwise disjoint: `+0..2048`, `+6144..8192`,
  `+8192..10240`; the out fill ends exactly where `bias set 0` begins.
- No fill writes `+2048` (Alo) or `+4096` (Whi); those are MMA #3 operands and
  must remain untouched.
- The fill loop's bound is a compile-time 16, so the compiler can unroll it. A
  loop that executes 16 times is dynamically equivalent but statically
  different; the contract's "16 × 128 B stores" must be defined as either form
  before the audit can accept a loop.

### 5.3 What is not proven

- That the compiler will not replace the splat loop with `memset`, a library
  call, or a scalar store sequence. The `asm volatile("" ::: "memory")` barrier
  stops reordering, not idiom recognition.
- That 128 B HVX stores to `+0` and `+6144` are accepted with the same cost and
  bank behaviour as mode 45's accesses to `+4096..+10240`. The address window is
  structurally identical, but only the device can show bank conflicts,
  throughput, or a fault.
- That holding the HVX context across the three MMA packets and the convert
  does not perturb the HMX path. Mode 45 holds the lock across its MMA and
  convert in the source; this is precedent, not proof for D8.

### 5.4 Phase 3 verdict

| Item | Verdict |
| --- | --- |
| 16 × 128 B stores cover 2,048 B exactly | **`STATICALLY_SUPPORTED`** (machine-checked) |
| 128 B alignment of all three fill ranges | **`STATICALLY_SUPPORTED`** |
| Splat value `0xA5A5` is a normal negative FP16 | **`STATICALLY_SUPPORTED`** |
| The compiler emits the registered vector-store form | **`INSUFFICIENT_EVIDENCE`** until disassembly |
| VTCM acceptance, bank behaviour, execution | **`DEVICE_ONLY`** |

---

## 6. FastRPC ABI and evidence retrieval — Phase 4

### 6.1 Shape

The IDL remains:

```text
long run(rout sequence<long> results, rout sequence<long> metadata,
         in long q, in long mode);
```

`rout sequence<T>` maps to `(T* data, int dataLen)`, where `dataLen` is the
**client-specified maximum number of elements** (the `rout` qualifier is the
bounded analogue of `read()`; see the QAIC IDL documentation,
["Bounded output parameters"](https://github.com/qualcomm/QAIC)).
The generated DSP signature is
already `(int *results, int resultsLen, int *metadata, int metadataLen, int q,
int mode)`. Consequences:

- results: 1,024 slots × 4 B = 4,096 B; metadata: 32 × 4 = 128 B; total
  4,224 B. The host allocates with `rpcmem_alloc`; today it allocates 11 slots
  (44 B) for results and 32 words (128 B) for metadata. A D8 host path needs a
  1,024-slot allocation.
- **The callee cannot return a different length.** The "short or long return"
  in evidence-contract §3.3 is not observable through this ABI. The length is a
  host allocation fact; a DSP "readbacks" word is a self-report. The correct
  host rule is: allocate exactly 1,024, pass exactly 1,024, validate that every
  slot is a zero-extended u16, and compare all 1,024 cells.
- On a non-zero method return, FastRPC/QAIC does **not** propagate the `rout`
  data (the QAIC documentation's "Return Values" rule: any data in `rout`
  parameters is not propagated back when a non-zero code is returned). The
  contract's "preserve the raw bytes on failure" therefore only works if the
  method returns 0 once the experiment has started, carrying the first-failure
  reason in metadata. Non-zero should be reserved for pre-execution failures
  (power, acquire, lock), where there is no experiment data to preserve.
  The C2 evidence contract has the same length wording and the same latent
  issue; this audit fixes neither document, but the D8 amendment should not
  copy the C2 wording.
- The DSP `run()` currently zeroes the entire results buffer with
  `memset(results, 0, resultsLen * sizeof(int))` before dispatch. A host-side
  poison-slot check cannot distinguish "DSP wrote zero" from "DSP never wrote";
  the numeric comparison is the only length-adjacent evidence.

### 6.2 Metadata layout — the blocking design defect

The contract's 32-word table assigns:

| Word | D8 contract | What `run()` already writes | Conflict |
| ---: | --- | --- | --- |
| 0 | Build ID echo | `HMX_MAGIC` | **yes** |
| 1 | DSP architecture | `__HEXAGON_ARCH__` | compatible value, different meaning |
| 2 | mode echo | `VTCM_BYTES` | **yes** |
| 3 | condition (`q` echo) | `elapsed_us` **after the handler** | **yes (overwritten)** |
| 4 | `mxclracc` count | power status | **yes** |
| 5 | MMA packets | mode-A lock status | **yes** |
| 6 | Convert count | mode-B lock status | **yes** |
| 7 | HMX Store count | hmx-executed flag **after the handler** | **yes (overwritten; value 1 coincides)** |
| 8 | operand packs | trial count **after the handler** | **yes (overwritten)** |
| 9 | bias sets | free (mode 45 uses it for schema) | mode-specific |
| 10 | fill mask | free | mode-specific |
| 11 | fill word | free | mode-specific |
| 12 | store target | free | mode-specific |
| 13 | out fill offset | free | mode-specific |
| 14 | Wlo fill offset | free | mode-specific |
| 15 | Ahi fill offset | free | mode-specific |
| 16 | trace status | free | mode-specific |
| 17–20 | operand FNVs | free | mode-specific |
| 21 | readbacks | free | mode-specific |
| 22 | cleanup status | free | mode-specific |
| 23 | first failure | free | mode-specific |
| 24 | Build ID second echo | free (mode 45 uses it) | mode-specific |
| 25–31 | reserved 0 | free | mode-specific |

Words 0, 2, 4, 5, 6 are written before dispatch and could be overwritten by a
handler. Words 3, 7 and 8 are written **after** the handler returns and cannot
be handler-owned. Word 0 conflicts with the magic check used by the identity
and host paths; word 2 conflicts with the VTCM-bytes meaning. A mode-46-aware
`run()` could stop writing the conflicting words, but that is a change to the
shared metadata contract, not a handler-local change.

Two clean resolutions exist:

1. **Renumber into 9..31.** 23 words are free; the D8 table needs 25 fields, but
   word 15 (Ahi offset) and word 24 (second build id) are derivable/redundant.
   Drop them and the remaining 23 fit. The generic words 0..8 keep their
   existing meanings, so the host's magic/arch/lock/trial checks continue to
   work and no legacy mode is affected. This is the recommended fix.
2. **Make `run()` mode-46 aware**: do not write words 3/7/8 (and define words
   0/2/4/5/6 as D8-owned for mode 46). The host must then skip the generic
   preamble checks for mode 46. This is more invasive and loses the shared
   magic/identity invariants.

The audit does not choose unilaterally; it records that the table as written
cannot be implemented against the current `run()`.

### 6.3 Guard tail

Contract §2 item 7 requires the 1,536 B tail (`+10752..+12288`) to stay `0x00`.
There is no metadata word for a tail/guard status (all 32 are assigned), and
VTCM zero-initialisation by `HAP_compute_res` is not established. A DSP-side
"tail is zero" flag would be a self-report; a pre/post FNV pair would be a real
measurement but needs two words, and the pre-run FNV must be taken before the
MMAs. Copying mode 45's three-bias prologue would write `+10752..+11008`
(bias set 2), directly violating the guard before D8 does anything.

### 6.4 Evidence independence

The contract asks for counts, masks and offsets. These are not equally strong:

| Evidence | During-run measurement | Self-report | Static disassembly can verify | Device-only |
| --- | --- | --- | --- | --- |
| `mxclracc`/MMA/Convert/Store/pack/bias counts | no (unless performance counters, absent) | yes — counters incremented next to the asm | yes — instruction counts per path | no |
| Store/fill target offsets | no | yes — constants echoed | yes — address computation/immediates | no |
| Fill mask | no | yes — derived from `q` | yes — branch conditions + store addresses | no |
| Fill word | no | yes — constant | yes — splat immediate | no |
| Operand FNVs | **yes**, if hashed from VTCM before MMA #1 | the FNV routine itself | only the hash loop's position | no |
| Condition echo | no | q transport echo | branch use of `q` | no |
| Build ID / architecture | no | compiled constants | symbol/stamp | no |
| Trace status (word 16) | no (always 0) | honest "not provided" | n/a | the internal read completion itself |
| Cleanup status | run() return code | handler cannot see release | n/a | resource release |
| 1,024-cell output | **yes** — the actual numeric outcome | n/a | n/a | yes (the outcome) |
| Tail/guard | no channel today | would be self-report | static "no write" | actual memory |

Key point: a counter or mask echo proves **the source path executed**, not that
the instruction did what the constant says. The independent evidence is the
disassembly (static) plus the 1,024-cell numeric outcome (behavioural). The
contract's `SUSPECT` language is correct; the audit should not upgrade it.

### 6.5 Phase 4 verdict

| Item | Verdict |
| --- | --- |
| IDL signature unchanged | **`STATICALLY_SUPPORTED`** |
| 1,024 + 32 length/transport | **`IMPLEMENTATION_CHANGE_REQUIRED`** (host allocation/gate; wording) |
| Metadata 32-word layout | **`IMPLEMENTATION_CHANGE_REQUIRED`** (word ownership) |
| Error-path raw preservation | **`IMPLEMENTATION_CHANGE_REQUIRED`** (return-0 rule) |
| Guard-tail evidence | **`IMPLEMENTATION_CHANGE_REQUIRED`** (no channel) |
| Count/mask evidence independence | **documented above**; no device upgrade |

---

## 7. VTCM layout and allocation — Phase 5

### 7.1 Statically proven

- The D8 map is the Candidate B §4.6 map: four 2 KiB operand tiles at
  `0/2048/4096/6144`, `out_pos` at `8192`, two 256 B bias sets at
  `10240/10496`.
- Every tile offset is 2,048 B aligned; every bias offset is 256 B aligned;
  every fill offset is 128 B aligned.
- The seven regions are pairwise disjoint; the used total is
  `10496 + 256 = 10752`; the margin to 12,288 is 1,536 B = 12 × 128 B.
- The reservation request is 12,288 B, exactly the size mode 42 already
  requests (and mode 42 uses all 12,288 B); mode 45 requests the same 12,288 B
  and uses 11,008 B. So the *request size and the 2 KiB pointer alignment check*
  are already exercised by shipped modes.
- The 128 B vector granularity is `2048 / 128 = 16`, and mode 45 executes
  16 × 128 B HVX loops in the same window.

### 7.2 Not statically provable

- That the allocator grants 12,288 B on the D8 session and returns a pointer
  with the required alignment (the existing check is fail-closed, so this is a
  device assertion, not an assumption).
- That 128 B HVX stores to `+0`/`+6144` are accepted and conflict-free.
- That the tail is `0x00` at entry and stays `0x00` (no evidence channel).
- Bank conflicts, throughput, memory synchronisation, execution time.

### 7.3 Cleanup

`run()` acquires an HMX-only resource (lock/unlock/release) and a combined
VTCM+HMX resource (lock, execute, unlock2, release), then powers HMX down.
Cleanup failures are returned as non-zero status (`HMX_ERR_UNLOCK2`,
`HMX_ERR_RELEASE`, `HMX_ERR_POWER_DOWN`). The D8 contract's word 22 "cleanup
status 0" would have to be written **after** `HAP_compute_res_release`, i.e. in
`run()`, not in the handler. On a cleanup failure the status is non-zero, so the
`rout` data is not propagated; the host sees only the status. That is acceptable
if the stop rule is "non-zero status = contract failure, no numeric reading",
but it makes word 22 redundant on the failure path and requires a `run()` change
on the success path.

### 7.4 Phase 5 verdict

| Item | Verdict |
| --- | --- |
| Offsets, alignment, disjointness, sizes, capacity arithmetic | **`STATICALLY_SUPPORTED`** |
| 12 KiB request / 2 KiB alignment pattern | **`STATICALLY_SUPPORTED`** (existing modes), acceptance **`DEVICE_ONLY`** |
| 128 B access to the two reused operand tiles | **`DEVICE_ONLY`** |
| Tail zero / guard | **`IMPLEMENTATION_CHANGE_REQUIRED`** (no channel) |
| Cleanup evidence | **`IMPLEMENTATION_CHANGE_REQUIRED`** (run() ownership) |

---

## 8. Disassembly audit procedure — Phase 6

### 8.1 Why the source is not enough

The pre-registration itself says source is not machine code, and static
structure is not execution order. The extra D8 problems are:

- a fill loop can be unrolled, kept as a loop, or turned into `memset`;
- the compiler can insert address arithmetic or spills between MMA #3 and the
  first fill;
- branches for A/B/C/D can be shared, duplicated, if-converted or laid out with
  a jump table;
- the three MMA packets can be adjacent in source but separated in the object.

### 8.2 Existing tool and its limits

`scripts/audit_hmx_mode45_disasm.ps1` is the right starting point: it takes the
built skel, hashes it, checks the symbol, counts `mxclracc`, `activation.hf =
mxmem`, `weight.hf = mxmem`, `bias = mxmem2`, `cvt.hf = acc`, `mxmem(...) =
cvt`, `vsub`, extracts the Rs immediate from the register's earlier assignment,
and checks a fixed order. It has **no** fill-address/extent parser, no HVX store
mnemonic handling, no branch/CFG traversal and no per-condition path model.
Using it unchanged for D8 would silently miss the fill form and the condition
paths.

### 8.3 Required D8 audit (future stamped object)

Run only after a future authorisation and build. The audit must:

1. hash the `.so` and record the hash, build id, symbol and compiler version;
2. locate the D8 handler symbol and disassemble it; recover the CFG;
3. count per path: `mxclracc` = 1, MMA packets = 3, `bias = mxmem2` = 2,
   `cvt.hf = acc` = 1, `mxmem(...) = cvt` = 1, and the condition's fill vector
   stores (48 on B, 32 on C/D, 16 on A; or a verified loop of that length);
4. verify the three MMA packets reference the registered addresses
   `(Ahi,Whi)`, `(Ahi,Wlo)`, `(Alo,Whi)` in order, and that all three precede
   every fill store on every path;
5. verify the bias loads precede `mxclracc` (the registered order), not merely
   the convert;
6. verify the out fill is the first fill, then Wlo, then Ahi, and that each
   fill covers exactly 2,048 B at `+8192`, `+6144`, `+0`;
7. verify the fill value is a 0xA5A5A5A5 splat and the stores are 128 B
   (`vmem`/`vstore`/equivalent), with no `memset`/helper call;
8. verify Convert #1 uses `Rs = 0x0001`, and Store #1 targets `+8192`;
9. verify no `syncht`, `barrier`, DMA sync, `mxswapacc`, checksum/scalar read,
   or host RPC appears between MMA #3 and the fills or between the fills and
   Convert #1;
10. verify the operand FNV loop (if present) is before MMA #1, not after MMA #3;
11. verify no write outside the declared regions, including `+10752..+12288`;
12. verify the condition paths have masks `1/7/3/5` and that A executes no
    operand fill;
13. preserve the disassembly and the hash with the run report.

Two honesty rules bound this audit:

- **A clean object is not an execution-order proof.** It shows the sequence is
  present and no forbidden construct exists; it cannot show when the MX unit
  finished reading.
- **A loop is not 16 static stores.** The contract must say which form is
  accepted; this audit recommends accepting either a fully unrolled 16-store
  sequence or a loop whose bound, stride and store width are all proven, and
  recording which one was found.

### 8.4 Existing disassembly evidence and its scope

The prior docs record: no `syncht`/barrier/`dmwait`/`dmstart` in the shipped
skel; three B3 MMA packets back-to-back; the mode-45 Rs values and HVX loops
counted by the audit script. Those are **method and precedent**, not D8
evidence: the D8 handler does not exist, and no D8 object was built. This audit
could not re-run the disassembler (no SDK/objdump/skel in this checkout).

### 8.5 Phase 6 verdict

| Item | Verdict |
| --- | --- |
| A concrete D8 audit procedure | **`STATICALLY_SUPPORTED`** (specified above) |
| Ability to execute it in this audit | **`INSUFFICIENT_EVIDENCE`** (no build environment) |
| Existing script usable unchanged | **no** — needs fill/CFG/address checks |

---

## 9. Design problems found

Each row is a pre-implementation issue, with the class and the evidence that
establishes it. `Blocking` means "must be resolved before writing the D8 DSP
handler"; it does not mean the problem is physically unsolvable.

| # | ID | Class | Problem | Required resolution |
| ---: | --- | --- | --- | --- |
| 1 | `mode46_unregistered` | **BLOCKED** | No trigger mode exists; `MODE_MAX = 45`, host max 45, no handler; Mode 46 is unregistered/NOT_RUN. | Register Mode 46 and its evidence contract (user authorisation), or re-register D8 under an approved mode. |
| 2 | `metadata_ownership_collision` | **IMPLEMENTATION_CHANGE_REQUIRED** | The 32-word D8 table reuses words 0–8; `run()` writes magic at 0, VTCM at 2, and overwrites 3/7/8 after the handler. | Renumber D8 evidence into 9..31 (drop the derivable Ahi offset and second build id), or make `run()` mode-46 aware. |
| 3 | `q_selector_encoding` | **IMPLEMENTATION_CHANGE_REQUIRED** | Contract proposes `q=0..3`; DSP and host reject `q<6`; identity pre-check uses the same `q`. | Register `q=6..9` with `condition=q-6` (recommended), or widen the gate for mode 46 + identity only. |
| 4 | `results_length_unobservable` | **IMPLEMENTATION_CHANGE_REQUIRED** | `rout sequence` length is client-bounded; the callee cannot return a short/long length. | Amend §3.3 to "host allocates exactly 1,024 / 32; validate all slots"; add the 4,096 B host allocation and `resultsLen >= 1024` DSP gate. |
| 5 | `error_path_raw_loss` | **IMPLEMENTATION_CHANGE_REQUIRED** | Non-zero method return discards `rout` data; the contract asks to preserve raw bytes on failure. | Return 0 after execution starts; carry the first-failure reason in metadata; non-zero only for pre-execution acquisition failures. |
| 6 | `guard_tail_no_channel` | **IMPLEMENTATION_CHANGE_REQUIRED** | The 1,536 B tail must stay `0x00`, but no metadata word carries a tail/guard status and VTCM zero-init is unproven; mode 45's bias2 writes into the tail. | Add a tail/guard word (and a pre-run baseline FNV or a documented zero-init guarantee); do not copy the three-bias prologue. |
| 7 | `fixture_packing_new` | **IMPLEMENTATION_CHANGE_REQUIRED** | The values exist (mode-42 generator reproduces the registered FNVs), but no mode writes the packed lo tiles at `+2048`/`+6144`. | Extend mode 43's PRM pack loop to hi/lo × act/wgt; emit the four runtime FNVs before MMA #1. |
| 8 | `fnv_timing` | **IMPLEMENTATION_CHANGE_REQUIRED** | Candidate B's model hashes the operands after MMA #3; copied to the device that is an 8 KiB read in the critical interval. | Take the FNV echo before MMA #1; the disassembly audit must verify the loop's position. |
| 9 | `bias_position_template` | **IMPLEMENTATION_CHANGE_REQUIRED** | Mode 42 loads bias after MMA #3; D8 registers bias before `mxclracc`. | Use the mode-45 prologue order; audit `bias = mxmem2` < `mxclracc`. |
| 10 | `fill_form_unproven` | **INSUFFICIENT_EVIDENCE** | The C loop can lower to unrolled stores, a loop or `memset`; the CPU model only records one 2,048 B event. | Build and disassemble; define the accepted static/dynamic form in the contract; reject `memset`/helper. |
| 11 | `disasm_audit_fill_gap` | **INSUFFICIENT_EVIDENCE** | The existing audit script cannot parse fill addresses/extents, HVX store forms, branches or the CFG. | Extend the audit tool before judging a future object. |
| 12 | `dsp_build_env_absent` | **INSUFFICIENT_EVIDENCE** | No Hexagon SDK/objdump/skel in this checkout; no DSP build or object inspection was possible. | Run the build and audit in the authorised build environment (not this audit). |
| 13 | `mode45_bias2_tail` | **IMPLEMENTATION_CHANGE_REQUIRED** | Mode 45's bias set 2 at `+10752` is inside D8's guard tail. | D8 writes only two bias sets; the disassembly audit verifies `+10752..+12288` is untouched. |
| 14 | `trace_validator_bias_gap` | **IMPLEMENTATION_CHANGE_REQUIRED** (CPU) | `d8_trace_valid` checks the bias count (2) but not the bias position; it also does not validate `mxclracc`/Convert counts (the model records Convert+Store as one HMX store event). | Before a device build, extend the CPU validator with bias-before-MMA and `mxclracc == 1`/Convert == 1 checks; re-run the 16-fault battery. |
| 15 | `out_fill_overwritten` | note | The out_pos sentinel is fully overwritten by Store #1; its evidence value is the registered *order/distance*, not a surviving byte. | Keep the contract language that the fill makes an unwritten store detectable and fixes the distance; do not claim a post-run sentinel check. |
| 16 | `condition_mask_self_report` | note | On a device the mask echo is derived from `q`; the trace validator is CPU-only. | The independent mask evidence is the disassembly branch/address check, not the echo. |

---

## 10. CPU-side verification performed

### 10.1 New verifier

`host_tests/hmx_acc_residual_rpc/hmx_d8_implementation_feasibility_test.py`

- reads the two D8 documents, the IDL, `probe_dsp.c`, `probe_host.c`,
  `hmx_mode45.h`, `hmx_b3_r2.h`, `hmx_identity.h`, the D8 CPU model and the
  disassembly script;
- checks the layout, alignment, disjointness, capacity and margin; the fill
  ledger, 16 × 128 coverage and masks; the registered sequence; the 32-word
  metadata table; the current ABI/interface baseline; the mode-46 absence;
  the build-id and identity gates;
- re-derives the mode-42 generator and verifies the four packed-tile FNVs
  equal the registered values;
- emits 13 findings and the verdict. Exit 0 means **the encoded invariants
  hold**, not that the experiment is ready.

Result: **183 checks, 0 fail, 13 findings** (`BLOCKED=1`,
`IMPLEMENTATION_CHANGE_REQUIRED=9`, `INSUFFICIENT_EVIDENCE=3`),
`d8_feas_gate=PASS`, `d8_feas_verdict=D8_IMPLEMENTATION_BLOCKED`.

### 10.2 Existing D8 CPU validation (unchanged)

`scripts/run_hmx_d8_sim.ps1` (extended with the verifier) still produces:

- `d8_checks=34 fail=0`, `d8_verdict=D8_CPU_READY`;
- four reference SHA-256 values unchanged; six pairwise differences unchanged;
- `d8_fault=01..16` unchanged, including revision-1 rejection with identical
  numbers;
- independent Python: `checks=29 fail=0`, `D8_INDEPENDENT_AUDIT_PASS`;
- differential: 125 cases; mutation: 5/5 detected;
- gcc `-O0`, gcc `-O2`, clang `-O2`: PASS;
- `d8_feasibility=PASS (static audit)`.

### 10.3 What was fixed / added

- Added the verifier and its gate to the D8 runner. It runs in Fast and Host
  because `run_host_contract_tests.ps1` invokes the D8 runner before its Fast
  early return.
- Fixed the verifier's own interface facts after checking them against the
  source (q gate, identity call, metadata ownership, mode-42 packet order,
  fixture generator). No registered value was changed.
- Deliberately **did not** modify `hmx_d8_min_sim.c`: adding checks would change
  the recorded `d8_checks=34` count of the revision-2 CPU record. The validator
  gap is instead recorded as finding 14 and must be closed before a device
  build.

### 10.4 Regression status

- D8: PASS (see above; `d8_checks=34`, `d8_feas_checks=183`).
- B3/R2: `b3_r2_checks=12828 fail=0`, 30/30 fault injections, six evidence
  digests unchanged, `B3_R2_FULL_TILE_CPU_READY`.
- C2: independent classifier 81 checks, differential 5,460 cases on gcc `-O2`,
  gcc `-O0` and clang `-O2`, 2/2 mutants; `C2_DIFFERENTIAL_PASS`.
- CB-FIX-1: trace controls and all-six-array parity PASS as part of the B3/R2
  battery.
- Fast: `scripts/verify.ps1 -Profile Fast` exit 0,
  `PASS=10 FAIL=0 SKIP=21` (211.0 s), including the D8 feasibility audit and
  the D8/B3/C2 gates.
- Host: `scripts/verify.ps1 -Profile Host` exit 0,
  `run_host_tests=PASS (contract+diagnostic)`, including the same gates and the
  full host object session.
- No FAIL/SKIP is reported as PASS. A SKIP (e.g. missing clang, the optional
  nicopedia checkpoint) is recorded as SKIP, not converted.

The verifier and runner changes are commit
`b2a3e347c1649ba8b4545477ed8a4d31b84ae26e`; this document is committed on top
of it. The final audit HEAD is the branch tip at push time and is recorded in
the PR #14 comment.

---

## 11. Blockers before implementation

Ordered by dependency:

1. **Mode 46 registration/authorisation.** Without a registered trigger mode,
   no D8 build may exist. This is a governance blocker, not a technical one.
2. **Metadata ownership amendment.** Decide renumbering into 9..31 (preferred)
   or a mode-46-aware `run()`. Until then, word 3 cannot carry the condition.
3. **Selector encoding amendment.** Register `q=6..9, condition=q-6` (or the
   widened-gate variant). The word-3 echo stays `0..3`.
4. **Length/error contract amendment.** Replace "short/long return" with the
   host allocation rule; add the 4,096 B buffer and `resultsLen >= 1024` gate;
   adopt the return-0-after-start rule.
5. **Guard-tail channel.** Add a tail/guard word or a documented zero-init
   guarantee; forbid bias set 2 in the D8 prologue.
6. **Fixture composition.** Add the packed-lo writes and the pre-MMA FNV echo.
7. **FNV timing and bias order.** Both are implementation-order constraints
   that the disassembly audit must verify.
8. **Fill-form definition and audit tooling.** Decide static 16 stores vs a
   proven 16-iteration loop; extend the disassembly script for fill addresses,
   extents, branches and CFG.
9. **CPU validator extension.** Bias-position and `mxclracc`/Convert counts in
   the trace validator, with new faults, before a device build.
10. **Build environment.** One clean stamped V81 build with
    `-DHMX_PROBE_BUILD_ID`, the host built with the same id, and the object
    hash preserved. This audit could not perform it.

None of these is an IDL change. None is evidence that D8 is violated. They are
the conditions under which a D8 implementation would be *the registered
experiment* rather than a look-alike.

---

## 12. What only the device can establish

- Whether `HAP_compute_res` grants the 12,288 B VTCM window on the D8 session
  and returns the expected alignment.
- Whether 128 B HVX stores to `+0` and `+6144` are accepted, and their bank
  behaviour/throughput.
- Whether the tail remains `0x00` (if a guard channel is added).
- **D8 itself**: whether the MMA operand reads have completed before the fills
  overwrite `Wlo`/`Ahi`. The device reports no MX pipeline trace; word 16 is
  always 0. Only the 1,024-cell outcome is observable.
- The later reuse writes (Convert #2 into `Wlo`, the HVX recon into `Ahi`) are
  later than the fills and are not measured by the minimal experiment.
- Timing, power states, other instruction distances, repeated runs.
- The actual compiler output (the future object), which this audit could not
  build.

A safe device result would mean: in that build, at that registered distance, in
that run, the fill did not beat the read for any cell carrying a difference. It
would **not** retire `B3_R2_SYNC_UNPROVEN`, prove a universal ordering, or
validate the full Candidate B run.

---

## 13. Verdict and reproduction

### 13.1 Verdict

**`D8_IMPLEMENTATION_BLOCKED`.**

Why not `D8_IMPLEMENTATION_STATIC_READY`: the implementation plan and the CPU
evidence contract are not yet coherent. The metadata table as written cannot be
implemented against `run()`, the `q=0..3` selector is rejected by both gates,
the results-length clause describes an unobservable, the error path discards the
data the contract asks to preserve, the guard tail has no evidence channel, and
the fixture/packing and FNV timing require explicit implementation decisions.
These are pre-implementation defects, not device unknowns.

Why not `D8_IMPLEMENTATION_INSUFFICIENT_EVIDENCE`: the defects are established
by the source and the registered documents; the missing DSP build/disassembly is
a separate, expected limitation (recorded as `INSUFFICIENT_EVIDENCE` items).

Why not `BLOCKED_BY_ABI`: the IDL signature is unchanged and `rout sequence<long>`
can carry 1,024 + 32 slots. The blockers are metadata ownership, selector
encoding, evidence semantics, mode registration and the missing build — not the
IDL type system.

### 13.2 Reproduction

CPU only; no SDK, no device, no ADB:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
python host_tests\hmx_acc_residual_rpc\hmx_d8_implementation_feasibility_test.py
.\scripts\run_hmx_d8_sim.ps1
```

Key new output:

```text
d8_feas_layout=PASS
d8_feas_fill=PASS
d8_feas_mask=PASS
d8_feas_sequence=PASS
d8_feas_metadata=PASS
d8_feas_abi=PASS
d8_feas_fixture=PASS
d8_feas_findings=13 BLOCKED=1 IMPLEMENTATION_CHANGE_REQUIRED=9 INSUFFICIENT_EVIDENCE=3
d8_feas_checks=183 fail=0
d8_feas_gate=PASS
d8_feas_verdict=D8_IMPLEMENTATION_BLOCKED
```

### 13.3 Change inventory

| Path | Change |
| --- | --- |
| `host_tests/hmx_acc_residual_rpc/hmx_d8_implementation_feasibility_test.py` | new independent static verifier (183 checks, 13 findings) |
| `scripts/run_hmx_d8_sim.ps1` | runs the verifier; adds `d8_feasibility`; fails the gate on a verifier failure |
| `docs/hmx-v81-d8-implementation-feasibility-audit.md` | this document |

No registered expectation, digest, mask, reference array, CPU check count or
approval state was modified. No DSP source, DSP binary, Mode 46 registration,
device command or QAIRT invocation was made.

### 13.4 PR status

PR #14 remains **Draft**. The audit result is reported as a PR comment; the
research branch receives only the verified CPU-side changes above. No merge,
no Draft release, no approval-state change.

### 13.5 Commits

| Commit | Content |
| --- | --- |
| `b2a3e347c1649ba8b4545477ed8a4d31b84ae26e` | independent feasibility verifier + `run_hmx_d8_sim.ps1` integration |
| commit containing this document | audit record; it is the audit branch tip at push time |

The final HEAD SHA, the PR head update and the Fast/Host results are reported
in the PR #14 comment and the session report. The audit base remains
`5a20fd858a72c252c20606fc55f318e14c127189`.
