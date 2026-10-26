# B3+R2 Integrated 32x32 Full-Tile Precision Pre-Registration

Status: **PRE-REGISTERED, NOT AUTHORISED**, revision 2, 2026-10-26.
Scope: CPU-only pre-registration, VTCM 12 KiB streaming design with an explicit
operand-restore path, 1,024-cell numeric oracle, an FP32-input comparison
condition, reproducible evidence digests, and a fail-closed decision procedure.
No DSP code added, no DSP build made, no device contacted, no ADB interaction.

**Revision 2 (2026-10-26)** closes the four CPU-only review items and changes no
frozen result:

1. The `Whi` restore source for MMA #3 is specified: a third 2 KiB DSP L2/TCM
   staging tile, i.e. **6 KiB** of staging (`Alo`, `Wlo`, `Whi`), not 4 KiB
   (§4.3). A `Whi` shadow inside VTCM is arithmetically impossible, and a
   mid-kernel host re-fetch is forbidden.
2. The cycles / ns / "< 0.1 %" transfer-cost estimate is **withdrawn**. Only the
   static transfer count (3 mid-kernel tiles = 48 vectors of 128 B) remains
   registered (§4.4). MMA-versus-transfer synchronization is an **unproven,
   device-only gate** (§4.5) and blocks the device trial.
3. The 1.845321 improvement is now evaluated under an explicit **FP32-input
   condition** in addition to the historical double-derived G8 condition (§6.3).
   The frozen G8 fixture, the pre-registered expected bit arrays and the Mode 45
   device evidence are untouched.
4. The fixture, the operand conditions and the Plain/B3 expected raw-bit arrays
   are frozen by canonical serialisation plus SHA-256, both in the simulator and
   in §6.4, and the `7.084104e-04` versus `7.084101e-04` MAE difference is
   explained and reproduced (§6.2).

`B3_R2_FULL_TILE_CPU_READY` is a **CPU-scope** verdict. It does not authorise
Mode 46, a DSP build, or any device contact: §4.5 keeps the device gate closed.

---

## 0. Purpose and Baseline Protection

This document pre-registers the future full-tile device verification mission integrating:
1. **B3:** High/low decomposition accumulating 3 MMA product packets ($A_{hi}W_{hi}$, $A_{hi}W_{lo}$, $A_{lo}W_{hi}$) into one 37-bit accumulator without intermediate rounding.
2. **R2:** Accumulator retention across two convert operations (`Rs[0]=1` on Convert #1, `Rs=0x1000` on Convert #2).
3. **Shape 0 / 4 sign separation:** Downstream clamping zeroes negative convert outputs, yielding non-negative positive and negative halves.
4. **DSP HVX `vsub.hf` signed reconstruction:** Vector subtraction on DSP (`out_recon = out_pos - out_neg`).

### Baseline Device Provenance (Protected)

The minimal 11-slot R2 feasibility test is established on device:
```text
HEAD: 8dfc5eb82c6079fbed6796f0d9dbb014384a0a45
r2_verdict=R2_ACC_REUSE_SUPPORTED
observed_slots=11
matching_slots=11
attempts=1
retry=0
MMA_packets=2
Convert=4
Store=4
HVX_recon_loops=2
rpc_status=0
close_status=0
```
This minimal result proves accumulator retention, bias switching, and HVX vector subtraction on isolated scalar slots. **It does NOT prove full-tile accuracy or multi-MMA accumulation.** This pre-registration defines the gates required to verify the full 32x32 tile.

---

## 1. Git HEAD and Target Source Inventory

- **Repository:** `yuubinnkyoku/HexaTrain` (workspace directory `PhoneLM`)
- **Branch:** `experiment/hmx-acc-residual-feedback` (PR #14, Draft maintained)
- **Baseline Device Provenance Commit:** `8dfc5eb82c6079fbed6796f0d9dbb014384a0a45`
- **Revision 1 (initial pre-registration):** `89bcec2811bcde5c4dd682db0ad038d5df6606fb`
- **Revision 2 (this document):** CPU-only corrections of §4.3-§4.5, §6.2-§6.4
- **Shared Contract Header:** `host_tests/hmx_acc_residual_rpc/hmx_b3_r2.h`
- **CPU Full-Tile Simulator:** `host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c`
- **Runner Script:** `scripts/run_hmx_b3_r2_sim.ps1`
- **1,024-Cell Oracle CSV (generated, gitignored):** `build/hmx_b3_r2_1024_cells.csv`
- **Byte-level audit dump (generated, gitignored):** `build/hmx_b3_r2_registered_arrays.txt`

---

## 2. Fixture Seed and Operand Generation

### Seed and PRNG
- **PRNG:** 64-bit Xorshift (`x ^= x << 13; x ^= x >> 7; x ^= x << 17;`)
- **Seed:** `0xE3C0117E5EEDF00DULL` (verbatim G8 deterministic seed)
- **Distribution:** Uniform $U(-2.0, +2.0)$ across $32 \times 32$ activation matrix $X$ and $32 \times 32$ weight matrix $W$.

### Pinned Anchor Rows and Lanes
- **Weight matrix $W$ (output channels $o \in [0, 31]$):**
  - Anchor rows $o \in \{0, 1, 2, 3\}$ and $o \in \{28, 29, 30, 31\}$ zeroed across all 32 input channels, then populated:
    - $W(0, 0) = 1.0 + 2^{-11}$, $W(0, 1) = 2^{-11}$, $W(0, 2) = 2^{-13}$
    - $W(1, 0) = 1.0 - 2^{-12}$ (B-2 anchor)
    - $W(2, 0) = 2^{-9}$ ($64 \times \text{acc} = 2^{-14}$ boundary)
    - $W(3, 0) = 1.0$, $W(3, 1) = 1.0 + 2^{-12}$ (near-zero cancellation)
    - $W(28, 0) = 2^{-10}$
    - $W(29, 0) = 2^{-10} - 2^{-21}$
    - $W(30, 0) = 3 \times 2^{-12}$
    - $W(31, 0) = 2^{-13}$
  - Rows $o \in [4, 27]$ stay seeded realistic ($N=32$ channels active).
- **Activation matrix $X$ (spatial lanes $s \in [0, 31]$):**
  - Anchor lanes $s \in \{0, 1, 2, 3\}$ zeroed across all 32 input channels, then populated:
    - $X(0, 0) = 1.0 + 2^{-11}$, $X(0, 1) = 2^{-11}$, $X(0, 2) = 2^{-13}$
    - $X(1, i) = -X(0, i)$ for all $i$ ($s_1 = -s_0$, negative anchor)
    - $X(2, 0) = 1.0 - 2^{-12}$, $X(2, 1) = -(1.0 - 2^{-12})$
    - $X(3, 0) = 2^{-10}$ (subnormal edge)
  - Lanes $s \in [4, 31]$ stay seeded realistic ($N=32$ channels active).

### Registered Operand Conditions

Three operand conditions share the seed, the pinned anchors and the same double
true oracle. They differ only in how the operands are rounded and split, and the
simulator publishes all three:

| ID | Condition | `hi` | `lo` | Role |
|---|---|---|---|---|
| A | `g8_double_split` | `RN16(x)` | `RN16(x - hi)` in double arithmetic | Historical G8/G9 oracle. Reproduced only to explain the MAE delta (§6.2). |
| B | `deploy_float_split` | `RN16(x)` | `RN16((float)x - (float)hi)` | **Registered simulation condition.** This is the residual the DSP actually computes (mode-42 `split_float_parts`). |
| C | `fp32_input` | `RN16((float)x)` | `RN16((float)x - (float)hi)` | FP32-input comparison condition (§6.3). `x` is rounded to FP32 first, as a real FP32 caller produces it. |

Condition B differs from A only in the residual subtraction precision. Condition C
differs from B only in that the input is rounded to FP32 before the FP16 rounding.

### Six Separated Oracle Levels
1. **$Y_{\text{true}}$ (Double True Oracle):** Exact double-precision product sum $\sum_{i=0}^{31} X(s, i) W(o, i)$ without intermediate float truncation.
2. **Plain FP16 MMA Expected:** Single chained float accumulator over $\text{RN16}(X) \times \text{RN16}(W)$, scaled by $+64.0$.
3. **B3 3-Product ACC Expected:** Single chained float accumulator over $A_{hi}W_{hi} + A_{hi}W_{lo} + A_{lo}W_{hi}$, scaled by $+64.0$.
4. **B3 Direct RN16 Expected:** Stored FP16 bit pattern directly converted with RN16 from the 3-product ACC.
5. **Reconstructed FP16 Expected:** Signed value after Convert #1 (Shape 0), Convert #2 (Shape 4), and HVX subtraction: $\text{RN16}(\text{pos} - \text{neg})$.
6. **Future HMX Measured Raw Bits:** Actual 16-bit integers read back from DSP VTCM `out_recon`.

---

## 3. Fixed B3 Accumulation Order and Constraints

### B3 3-Product Accumulation Sequence
```text
mxclracc.hf

MMA #1: Ahi × Whi   (primary accumulator initialized with dominant product)
MMA #2: Ahi × Wlo   (accumulated into SAME 37-bit ACC)
MMA #3: Alo × Whi   (accumulated into SAME 37-bit ACC)

[ACC retained across Convert #1]

Convert #1: scale = +64.0, shape = 0 (x),  bo = 0, Rs = 0x0001 (retain ACC)
Store #1:   out_pos (VTCM offset 4096)

Convert #2: scale = +64.0, shape = 4 (-x), bo = 0, Rs = 0x1000 (clear+swap)
Store #2:   out_neg (VTCM offset 6144)

DSP HVX:
out_recon = out_pos - out_neg (VTCM offset 8192, 16 vectors of 128 B)
```

### Operand Lifetime Per MMA (Registered)

A 2 KiB `wgt` buffer can hold exactly one weight tile, so MMA #2 necessarily
destroys the VTCM copy of `Whi`. The lifetime table is therefore part of the
contract, not an implementation detail:

| MMA | `act` (VTCM +0) | `wgt` (VTCM +2048) | Where `Whi` comes from | Stream tiles before this MMA |
|---|---|---|---|---|
| #1 | `Ahi` (prologue) | `Whi` (prologue) | prologue pack | 0 |
| #2 | `Ahi` (still resident) | `Wlo` (overwrites `Whi`) | — | 1 (2 KiB `Wlo`) |
| #3 | `Alo` | `Whi` **restored** | L2 `stWhi` staging tile | 2 (2 KiB `Alo` + 2 KiB `Whi`) |

The operand identity of every MMA is registered and checked:
`operand_log = AHI:WHI, AHI:WLO, ALO:WHI`.
→ fault `WHI_RESTORE_SOURCE_LOST` (fault 14) covers a staging tile that is lost
between MMA #2 and MMA #3.

### Inviolable B3 Invariants
- **NEVER** rely on the VTCM `Whi` bytes after the MMA #2 stream update: they
  are overwritten by `Wlo` by construction.
- **NEVER** restore `Whi` from a mid-kernel host/DDC re-fetch, and **NEVER**
  assume a VTCM `Whi` shadow: §4.3 shows it does not fit.
- **NEVER** accept a restore whose payload integrity was not checked (the CPU
  model checksums the 2 KiB staging tile; a device implementation must provide
  an equivalent check).
- **NEVER** clear ACC between MMA #1, #2, and #3.
- **NEVER** round or convert intermediate products to FP16 between MMAs.
- **NEVER** store intermediate products separately to add them in HVX.
- **NEVER** re-execute MMAs between Convert #1 and Convert #2.
- **NEVER** enable feedback (`Rs[4]=0`, `Rs[3:2]=0`).
- **NEVER** alter the product accumulation order ($A_{hi}W_{hi} \rightarrow A_{hi}W_{lo} \rightarrow A_{lo}W_{hi}$).

---

## 4. VTCM 12 KiB Layout and Staged Operand Streaming

### 4.1 Why the operands cannot all be resident
- A 32x32 FP16 tile occupies $32 \times 32 \times 2 = 2,048\text{ B}$ (2 KiB).
- R2 output buffers require:
  - `out_pos`: 2 KiB
  - `out_neg`: 2 KiB
  - `out_recon`: 2 KiB
  - Bias sets 0, 1: $2 \times 256\text{ B} = 512\text{ B}$
  - Total output + bias: $6,656\text{ B}$ (6.5 KiB).
- Total HMX VTCM reservation is **12,288 B** (12 KiB).
- Remaining VTCM space for operands: $12,288 - 6,656 = 5,632\text{ B}$ (fits at most two 2 KiB operand buffers).
- Storing four resident operand tiles ($A_{hi}, A_{lo}, W_{hi}, W_{lo} = 8\text{ KiB}$) simultaneously with the R2 outputs would require $14.5\text{ KiB} > 12\text{ KiB}$ and is **physically impossible**.

### 4.2 VTCM Map (verified): sequential operand streaming
Keep the exact R2 non-overlapping memory map and stream operand tiles into the 2 KiB `act` and 2 KiB `wgt` buffers between MMAs:

| Region | VTCM Offset | Size | Alignment | Purpose |
|---|---:|---:|---:|---|
| `act` | `+0` | 2048 B | 2 KiB | Activation buffer (holds $A_{hi}$, then $A_{lo}$) |
| `wgt` | `+2048` | 2048 B | 2 KiB | Weight buffer (holds $W_{hi}$, then $W_{lo}$, then $W_{hi}$) |
| `out_pos` | `+4096` | 2048 B | 2 KiB | Positive converted half (Convert #1, Shape 0) |
| `out_neg` | `+6144` | 2048 B | 2 KiB | Negative converted half (Convert #2, Shape 4) |
| `out_recon` | `+8192` | 2048 B | 2 KiB | Reconstructed signed output (DSP HVX `vsub.hf`) |
| `bias` set 0 | `+10240` | 256 B | 256 B | Scale `0x5400` (+64), Shape 0, $bo=0$ |
| `bias` set 1 | `+10496` | 256 B | 256 B | Scale `0x5400` (+64), Shape 4, $bo=0$ |
| **Total Used** | | **10,752 B** | | **of 12,288 B (1,536 B margin)** |

### 4.3 DSP L2/TCM Operand Staging (the `Whi` restore source)

The mid-kernel operand copies need a source that survives the MMA phase. That
source is a **third** 2 KiB staged tile in DSP L2/TCM, outside the 12 KiB HMX
VTCM window:

| Stage region | L2 offset | Size | Alignment | Contents | Lifetime |
|---|---:|---:|---:|---|---|
| `stAlo` | `+0` | 2048 B | 2 KiB | Packed $A_{lo}$ | written in prologue, read once (MMA #3) |
| `stWlo` | `+2048` | 2048 B | 2 KiB | Packed $W_{lo}$ | written in prologue, read once (MMA #2) |
| `stWhi` | `+4096` | 2048 B | 2 KiB | Packed $W_{hi}$ **restore copy** | written in prologue, **read-only**, read once (MMA #3) |
| **Total** | | **6144 B (6 KiB)** | | 3 tiles | |

Answering the review question directly:

- The original `Whi` is **not** available from VTCM after MMA #2. `wgt` was
  overwritten by `Wlo` and there is no room for a shadow: 10,752 B used +
  2,048 B = 12,800 B > 12,288 B.
- `Whi` is therefore kept in an independent external buffer, and the amount of
  staging is **6 KiB, not 4 KiB**: 2 KiB `Alo` + 2 KiB `Wlo` + 2 KiB `Whi`
  restore copy. This is a data-lifetime correction, not a VTCM capacity failure.
- Restoring `Whi` by re-fetching it from the host mid-kernel is forbidden: it
  would add an RPC between MMAs and destroy the single-attempt invariant.

### Operand Transfer Timeline
1. **Prologue (5 tiles = 80 vectors):**
   - Pack $A_{hi}$ into `act` (VTCM +0) and $W_{hi}$ into `wgt` (VTCM +2048).
   - Pack $A_{lo}$ into `stAlo`, $W_{lo}$ into `stWlo`, and copy the packed
     $W_{hi}$ tile into `stWhi` (byte copy of the `wgt` contents loaded for MMA #1).
   - Checksum `stWhi` (FNV-1a over 2,048 B in the CPU model) and keep it for the restore check.
   - Pre-fill `out_pos`, `out_neg`, `out_recon` with sentinel `0xA5A5`.
   - Load Bias sets 0 and 1.
2. **MMA #1 ($A_{hi} \times W_{hi}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }`.
3. **Stream Update 1 (for MMA #2), 1 tile:**
   - $A_{hi}$ remains resident in `act`.
   - DSP HVX copies `stWlo` (16 vectors $\times 128\text{ B}$) into `wgt` (+2048).
     The VTCM copy of $W_{hi}$ is destroyed here by design.
4. **MMA #2 ($A_{hi} \times W_{lo}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }` into the retained ACC.
5. **Stream Update 2 (for MMA #3), 2 tiles:**
   - DSP HVX copies `stAlo` into `act` (+0).
   - DSP HVX copies `stWhi` back into `wgt` (+2048) — the only registered
     restore source — after verifying the staging checksum.
6. **MMA #3 ($A_{lo} \times W_{hi}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }` into the retained ACC.
7. **Convert & HVX Reconstruction:**
   - Convert #1 stores to `out_pos` (offset 4096).
   - Convert #2 stores to `out_neg` (offset 6144).
   - DSP executes 16 iterations of `Q6_Vhf_vsub_VhfVhf` writing to `out_recon` (offset 8192).

### 4.4 Static Transfer Accounting (no timing claim)

| Item | Static count |
|---|---:|
| Prologue tiles | 5 (10,240 B = 80 vectors) |
| Mid-kernel stream tiles | **3** (`Wlo`, `Alo`, `Whi`) |
| Mid-kernel stream bytes | **6,144 B** |
| Mid-kernel stream vectors (128 B) | **48** |

**Withdrawn (2026-10-26):** the earlier claim "1 cycle per 128 B vector, so
$\approx 48$ cycles $\approx 48$ ns at 1 GHz, $< 0.1\%$ overhead" is retracted.
The static number of 128 B vector transfers is computable and remains registered;
the *execution* cost is not. It depends on issue width, store/load dependencies,
L2-to-VTCM bandwidth, VTCM bank conflicts, and the synchronization scheme, none
of which is established on this design. No cycle, nanosecond, or percentage
overhead figure may be quoted for this kernel until it is measured on a device
under the §4.5 protocol.

### 4.5 Synchronization: UNPROVEN (device-only gate)

`asm volatile("" ::: "memory")` is a **compiler** barrier. It does not order
HMX issue/completion against HVX stores, and it does not flush or fence VTCM
writes. The Qualcomm HMX PRM describes HMX reading operands from VTCM; it does
not, by itself, establish that overwriting an operand buffer after an MMA packet
is safe for this streaming schedule.

The following must be proven on device before any full-tile result is accepted:

1. That MMA #2 has completed its VTCM `wgt`/`act` reads before the `Wlo` stream
   overwrites `wgt` (or that a documented double-buffer/ownership rule makes the
   overwrite safe without one).
2. That MMA #3 has completed before `out_pos`/`out_neg` are written, and that the
   HVX `stAlo`/`stWhi` stores are visible to HMX before MMA #3 is issued.
3. What the actual, documented completion primitive is (`mxmem`/HMX barrier,
   thread `syncht`, VTCM fence, or a DA-style handshake), with the instruction
   encoding verified in the static disassembly audit.
4. That the staging checksum still matches at restore time (no staging reuse).

Status: **OPEN**. `B3_R2_SYNC_UNPROVEN` is the correct classification of this
design today, and the device trial stays unauthorised (§9).

### Architectural Safety Audits
1. **Accumulator Immunity:** The HMX 37-bit ACC resides in internal execution registers, not in VTCM. Overwriting bytes at offsets 0..4095 has zero impact on accumulator contents.
2. **Output Buffer Protection:** Offsets 4096..10239 are never touched during the MMA phase. Sentinel patterns `0xA5A5` confirm non-interference.
3. **Transfer Accounting:** See §4.4. Static counts only; no timing claim is registered.
4. **PRM Crouton Invariant:** All inputs and outputs strictly follow pair-interleaved crouching:
   $$\text{act}(s, i) = (s \gg 1) \times 64 + i \times 2 + (s \& 1)$$
   $$\text{wgt}(o, i) = (i \gg 1) \times 64 + o \times 2 + (i \& 1)$$
   $$\text{store}(s, o) = (s \gg 1) \times 64 + o \times 2 + (s \& 1)$$
   Bijective mapping over all 1,024 elements is verified with zero collisions.

---

## 5. Subcases and Execution Plan

The trial consists of two sequential, independent subcases executed on the DSP:

### Subcase P: Plain + R2 (Control Baseline)
- Operands: Unsplit $\text{RN16}(X)$ and $\text{RN16}(W)$
- Execution:
  - 1 `mxclracc`
  - 1 MMA packet
  - 2 Converts (Convert #1 $Rs=0x0001$, Convert #2 $Rs=0x1000$)
  - 2 Stores (`out_pos`, `out_neg`)
  - 1 DSP HVX reconstruction loop (16 vectors)

### Subcase B: B3 + R2 (Precision Evaluation)
- Operands: Split $A_{hi}, A_{lo}, W_{hi}, W_{lo}$
- Execution:
  - 1 `mxclracc`
  - 3 MMA packets (staged streaming, `Whi` restored from `stWhi` for MMA #3)
  - 2 Converts (Convert #1 $Rs=0x0001$, Convert #2 $Rs=0x1000$)
  - 2 Stores (`out_pos`, `out_neg`)
  - 1 DSP HVX reconstruction loop (16 vectors)

### Total Trial Execution Budget
```text
Total mxclracc      = 2
Total MMA packets   = 4  (1 Plain + 3 B3)
Total Converts      = 4  (2 Plain + 2 B3)
Total Stores        = 4  (2 Plain + 2 B3)
Total HVX loops     = 2  (32 vectors x 128 B = 4 KiB)
Bias sets loaded    = 2  (Set 0 at 10240, Set 1 at 10496)
L2 staging tiles    = 3  (stAlo, stWlo, stWhi = 6 KiB)
Mid-kernel streams  = 3  (Wlo, Alo, Whi restore = 48 vectors)
Whi restores        = 1  (checksum-verified)
Attempts allowed    = 1 clean run (NO retries)
```

---

## 6. 1,024-Cell CPU Reference Results (Pre-Registered)

### 6.1 Registered condition (B, `deploy_float_split`)

The following baseline metrics were produced by
`host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c` under operand
condition B and are **frozen**. Every entry below is asserted by the simulator
on each run, so the table cannot drift away from the code. Medians use the
conventional definition for an even sample count (mean of the two middle order
statistics, `errors[511]` and `errors[512]`).

### Global Error Metrics
| Metric | Subcase P (Plain+R2) | Subcase B (B3+R2) | Ratio (Plain / B3) |
|---|---:|---:|---:|
| **Mean Absolute Error (MAE)** | `1.307244e-03` | `7.084101e-04` | **`1.845321`** |
| **Max Absolute Error** | `1.113717e-02` | `7.217338e-03` | `1.543149` |
| **Root Mean Squared Error (RMSE)** | `1.978930e-03` | `1.230952e-03` | `1.607642` |
| **Median Absolute Error** | `8.406054e-04` | `3.204800e-04` | `2.622957` |
| **95th Percentile Error** | `4.384638e-03` | `2.955180e-03` | `1.483713` |

### Pairwise Cell Distribution
- **Improved Cells:** `427` ($41.70\%$)
- **Equal Cells:** `597` ($58.30\%$)
- **Worsened Cells:** `0` ($0.00\%$)
- **Both Non-Zero Violations:** `0` (Strict mutual exclusivity preserved across all 1,024 cells)
- **Direct Bit Mismatches:** `0` ($100\%$ raw bit match vs direct CPU reference)
- **Inf / NaN Count:** `0`

### Fixture Cell Class Breakdown
| Class | Cells | Plain MAE | B3 MAE | Improved | Equal | Worsened |
|---|---:|---:|---:|---:|---:|---:|
| `s0_positive` | 32 | `4.168487e-04` | `1.890148e-04` | 18 | 14 | 0 |
| `s1_negative` | 32 | `4.168487e-04` | `1.890148e-04` | 18 | 14 | 0 |
| `s2_mixed` | 32 | `4.322749e-04` | `2.624828e-04` | 14 | 18 | 0 |
| `s3_edge` | 32 | `1.424011e-07` | `1.424011e-07` | 0 | 32 | 0 |
| `realistic` | 896 | `1.448775e-03` | `7.867311e-04` | 377 | 519 | 0 |

### Pinned Diagnostic Cells
- `(s0, o0)`: Plain `0x5400`, B3 `0x5401` (Positive discrimination)
- `(s1, o0)`: Plain `0xD400`, B3 `0xD401` (Negative discrimination)
- `(s2, o1)`: Plain `0x5400`, B3 `0x53FF` (B-2 case-1 anchor)
- `(s2, o3)`: Plain `0x0000`, B3 `0xA400` (Cancellation anchor)
- `(s3, o28)`: Plain `0x0400`, B3 `0x0400` (Smallest normal)
- `(s3, o29)`: Plain `0x0400`, B3 `0x0400`
- `(s3, o30)`: Plain `0x0300`, B3 `0x0300` (Subnormal boundary)
- `(s3, o31)`: Plain `0x0080`, B3 `0x0080` (Deep subnormal)

### 6.2 Why the B3 MAE is `7.084101e-04` here and `7.084104e-04` in G8

The two numbers are the **same experiment with two different residual-split
arithmetics**, and the simulator now reproduces both to prove it:

| Condition | `lo` residual | Plain MAE | B3 MAE | Ratio | Improved / Equal / Worsened |
|---|---|---:|---:|---:|---|
| A `g8_double_split` (historical G8 oracle) | `RN16(x - hi)` in double | `1.307244e-03` | `7.084104e-04` | `1.845321` | 427 / 597 / 0 |
| B `deploy_float_split` (registered) | `RN16((float)x - (float)hi)` | `1.307244e-03` | `7.084101e-04` | `1.845321` | 427 / 597 / 0 |

Measured cause (printed as `split_residual` by the simulator):

- `lo_bits_differ_act = 162`, `lo_bits_differ_wgt = 152` of 1,024 elements
  each: the FP32 residual subtraction rounds the 11-bit `lo` tail differently
  on the seeded non-dyadic cells. All pinned dyadic cells
  (`1.0 ± 2^-k`, `2^-k`) are identical under both arithmetics.
- `stored_cells_differ = 1` of 1,024 cells changes its stored FP16 bits; the
  aggregate MAE therefore moves by `3e-10`, i.e. only the 6th significant
  digit. Plain MAE, max error, RMSE, the improvement ratio, and the
  427/597/0 split are unchanged.
- The G8 fixture, its expected values and the Mode 45 device evidence are
  untouched. The registered condition is **B**, because B is what the DSP can
  compute; A is kept only as the historical oracle reproduction.

### 6.3 FP32-Input Comparison Condition (C)

Condition A/B derive `hi`/`lo` from the double-precision truth value. A real
FP32 caller hands the split an FP32 operand, so the same 1,024-cell experiment
is repeated with `x` and `w` rounded to FP32 first (condition C). This is a
separate, additional evaluation; it changes no frozen result.

| Metric | Value |
|---|---:|
| Operand-level divergence vs condition B (`hi`/`lo`, 2,048 elements) | `0` / `0` |
| Cells whose stored bits differ from condition B | `0` / 1024 |
| Plain MAE | `1.307244e-03` |
| B3 MAE | `7.084101e-04` |
| Improvement ratio | `1.845321` |
| Improved / Equal / Worsened | 427 / 597 / 0 |
| Plain MAE vs FP32-input-exact oracle | `1.307255e-03` |
| B3 MAE vs FP32-input-exact oracle | `7.084098e-04` |
| Ratio vs FP32-input-exact oracle | `1.845338` |

Interpretation, stated conservatively:

- On this fixture the FP32 rounding of the input is a **bit-level no-op** for
  both `hi` and `lo`. That is expected rather than lucky: `hi` keeps 11
  significand bits and `lo` is quantised to FP16 *after* the residual
  subtraction, so a `2^-24` relative input perturbation only matters at an
  exact rounding tie. The fixture contains no such tie.
- Therefore the `1.845321` figure is **unchanged** under the FP32-input
  condition, and it is now a measured result rather than an assumption carried
  over from the double-derived fixture.
- The input-rounding floor is visible separately in the FP32-input-exact scores:
  against the exact product of the FP32 operands the ratio is `1.845338`, i.e.
  the input rounding contributes a ~`1e-7` relative shift to the ratio and does
  not drive the comparison.
- Limitation, registered: this fixture does not contain double-rounding ties, so
  it cannot prove that C always equals B. A dedicated tie-stress tile would be a
  separate pre-registration if the deployment needs that guarantee.

### 6.4 Reproducibility: canonical serialisation and SHA-256

`build/hmx_b3_r2_1024_cells.csv` is generated output and is gitignored, so the
registered evidence is frozen here as **canonical byte serialisation plus
SHA-256**, not as a `build/` file:

| Artifact | Canonical serialisation | Bytes | SHA-256 |
|---|---|---:|---|
| Truth fixture | `X[0..1023]` then `W[0..1023]`, IEEE-754 binary64, little-endian | 16,384 | `49b0c69cb559e575c75d3d804d5ed95a42ad402ac8c07ceaaa3b666a44f236d0` |
| Double true oracle | `Y_true[0..1023]`, binary64 LE (row-major `s*32+o`) | 8,192 | `5f61ff948a808595521b4b428aac916b62b6b70e6653fe9dd6686b7f0ada02c0` |
| Operand array, condition B | `ahi, alo, whi, wlo, a_plain, w_plain`, each 1,024 × binary16 LE | 12,288 | `e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3` |
| **Plain expected raw bits** | `out_recon` from the Plain+R2 subcase, 1,024 × binary16 LE, row-major | 2,048 | `0e58d8f88c8ac00d005a3a93e743fedc90bdc3cb0e163d0a23b18c118e9336bb` |
| **B3 expected raw bits** | `out_recon` from the B3+R2 subcase, 1,024 × binary16 LE, row-major | 2,048 | `687509a2c5999541ee15d5da21aecd28e41d37a4417ed22b2f0a13184e030939` |
| Operand array, condition C | same layout as B | 12,288 | `e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3` |

Note that the condition C digest is **identical** to condition B: that is the
measured FP32 no-op of §6.3, recorded as a digest rather than as prose.

Procedure to re-derive:

```powershell
# CPU only; no SDK, no device, no ADB.
.\scripts\run_hmx_b3_r2_sim.ps1
# -> b3_r2_sha256_input_truth=...  (six digest lines, keys above)
```

The simulator compares all six digests against compiled-in constants and fails
closed on any difference, so a fixture or expectation change cannot pass
silently. It also writes `build/hmx_b3_r2_registered_arrays.txt` (gitignored)
with the same bytes as hex for independent checking, e.g.
`sha256sum` over the `input_truth_x_then_w_f64le` section reproduces
`49b0c69c…`. The digests were cross-checked against an independent SHA-256
implementation (Python `hashlib`) during this revision.

---

## 7. Normal vs Subnormal Boundary Partition

The 1,024 output cells are partitioned into two distinct evaluation sets:

1. **Normal Partition (1,022 cells):**
   - All cells where $|64 \times \text{acc}| \ge 2^{-14} = 6.103515625 \times 10^{-5}$ or $\text{acc} = 0$.
   - **Contract Requirement:** Strict bit-exact match against CPU oracle.
2. **Subnormal Boundary Partition (2 cells):**
   - Target cells: `(s3, o30)` (raw bits `0x0300`) and `(s3, o31)` (raw bits `0x0080`).
   - Hexagon HVX `vsub.hf` subnormal handling (whether round-to-nearest even or flush-to-zero / denormals-are-zero) is not formally documented.
   - **Contract Decision:** A device mismatch restricted strictly to `(s3, o30)` and `(s3, o31)` resulting from hardware FTZ/DAZ **shall NOT invalidate the normal partition results**. It shall be recorded as `B3_R2_HVX_SUB_FTZ_OBSERVED`, keeping the normal partition verified while preventing unearned full-tile promotion.

---

## 8. Fault Injection Suite (Verified Fail-Closed)

The following 14 fault models are implemented and verified in `hmx_b3_r2_full_tile_sim.c`. All 14 fail closed:

| # | Fault Model | Injected Defect | Detection Mechanism |
|---|---|---|---|
| 1 | `FAULT_MMA2_MISSING` | MMA #2 omitted | MAE regression, bit mismatch, MMA count |
| 2 | `FAULT_MMA3_MISSING` | MMA #3 omitted | MAE regression, bit mismatch, MMA count |
| 3 | `FAULT_MMA_ORDER` | MMA #3 executed before #2 | Sequence tracking contract fail |
| 4 | `FAULT_ACC_EARLY_CLEAR` | ACC cleared between MMAs | Output zeroed, pinned anchors fail |
| 5 | `FAULT_OPERAND_SWAP` | $W_{hi}$ used instead of $W_{lo}$ | Numerical mismatch, pinned cells fail |
| 6 | `FAULT_CROUTON_CORRUPT` | Row-major written to VTCM | Crouton decode failure, cell mismatch |
| 7 | `FAULT_SHAPE4_NOT_SELECTED` | Convert #2 uses Shape 0 | Pos/neg cancellation, zero recon |
| 8 | `FAULT_BO_NONZERO` | $bo = 1.0$ injected in bias | `both_nonzero > 0`, mutual exclusivity broken |
| 9 | `FAULT_BUFFER_OVERLAP` | VTCM region collision | Layout bounds validator |
| 10 | `FAULT_HVX_SUB_REVERSED` | `out_neg - out_pos` | Sign inverted across all cells |
| 11 | `FAULT_RECON_STORE_FAIL` | Recon unwritten | Sentinel `0xA5A5` unwritten detection |
| 12 | `FAULT_HOST_FAKE_RECON` | Host computed recon | `recon_source != DSP_HVX` contract fail |
| 13 | `FAULT_COUNTER_MISMATCH` | Counter vs disasm mismatch | Disassembly static audit comparison |
| 14 | `FAULT_WHI_RESTORE_SOURCE_LOST` | L2 `stWhi` staging tile lost between MMA #2 and #3 | Restore checksum mismatch **and** stored-bit mismatch |

---

## 9. Future Device Trial Decision Procedure

Upon eventual authorization of the device trial, results shall be classified in the following exact order:

```text
                                [ Device Execution ]
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ Host Contract ]               [ Static Disasm ]
                 - Execution counts              - Opcode sequence
                 - Rs encodings                  - Register map
                 - Build ID match                - Instruction counts
                 - Sentinel words = 0            - Staging buffers
                 - operand_log                   - Restore copy present
                 - restore checksum
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                                   { All Agree? }
                                    /          \
                                  No            Yes
                                 /                \
          [ B3_R2_HOST_CONTRACT_FAILED ]           ▼
                                        [ Operand Supply / Sync Gate ]
                                        - operand_log == AHI:WHI,AHI:WLO,ALO:WHI
                                        - mid-kernel tiles == 3 (48 vectors)
                                        - Whi restored from L2 stWhi only
                                        - restore checksum matched
                                        - MMA #2 / #3 vs stream order proven
                                        - completion primitive in the disasm
                                                   │
                                          { Supply proven? }
                                           /             \
                                         No               Yes
                                        /                   \
   [ B3_R2_OPERAND_SUPPLY_BLOCKED ]                            ▼
   [ B3_R2_SYNC_UNPROVEN ]                            [ Implementation Check ]
                                                      - Mutual exclusivity (both_nonzero == 0)
                                                      - Sentinel clean (0xA5A5 absent)
                                                      - Positive/Negative separation
                                                             │
                                                      { Correct Layout? }
                                                       /             \
                                                     No               Yes
                                                    /                   \
                         [ B3_R2_LAYOUT_OR_RESOURCE_BLOCKED ]            ▼
                                                                 [ Numerical Accuracy ]
                                                                 - Plain MAE ~ 1.307e-3
                                                                 - B3 MAE ~ 7.084e-4
                                                                 - Improvement >= 1.80x
                                                                 - Worsened cells == 0
                                                                 - Normal bit match = 100%
                                                                         │
                                                                 { Accuracy Met? }
                                                                  /             \
                                                                No               Yes
                                                               /                   \
                                       [ B3_R2_NUMERICAL_BLOCKED ]                  ▼
                                                                      [ Subnormal Boundary Check ]
                                                                      - s3 cells match RN-even?
                                                                       /                       \
                                                                     No                         Yes
                                                                    /                             \
                                                     [ B3_R2_HVX_SUB_FTZ_OBSERVED ]  [ B3_R2_FULL_TILE_SUPPORTED ]
```

A CPU-only re-run of the model can never produce
`B3_R2_FULL_TILE_SUPPORTED`: the operand-supply/sync gate needs device evidence
(§4.5), so the strongest CPU classification remains
`B3_R2_FULL_TILE_CPU_READY`.

---

## 10. Pre-Registration Verdict

**`B3_R2_FULL_TILE_CPU_READY` (CPU scope only)**

- The 3-product B3 accumulation and R2 two-convert sign separation are
  architecturally proven to co-exist within 10,752 B of the 12 KiB VTCM via
  staged sequential operand streaming, **with `Whi` restored for MMA #3 from a
  third 2 KiB L2/TCM staging tile (6 KiB staged, not 4 KiB)**.
- Operand lifetimes are registered per MMA (`AHI:WHI, AHI:WLO, ALO:WHI`) and
  asserted; the restore is checksum-verified; a lost staging tile fails closed.
- The 1,024-cell numeric oracle is frozen under a registered operand condition,
  together with the fixture, operand and expected-bit SHA-256 digests (§6.4), so
  the numbers can be recomputed from the source tree alone.
- The improvement ratio is now measured under the FP32-input condition as well
  (§6.3), not carried over from a double-derived fixture.
- Only static transfer counts are registered: **48 mid-kernel vectors of 128 B**.
  The earlier cycles/ns/0.1 % estimate is withdrawn.
- MMA-versus-transfer **synchronization is unproven** (§4.5); the device gate is
  therefore closed with `B3_R2_SYNC_UNPROVEN`, and all 14 fault modes fail closed.
- **No DSP implementation, Mode 46 build, or device contact is authorised by
  this document.**
