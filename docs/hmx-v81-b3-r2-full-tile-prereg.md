# B3+R2 Integrated 32x32 Full-Tile Precision Pre-Registration

Status: **PRE-REGISTERED, NOT AUTHORISED**, 2026-10-25.
Scope: CPU-only pre-registration, VTCM 12 KiB streaming design, 1,024-cell numeric oracle, and fail-closed decision procedure.
No DSP code added, no DSP build made, no device contacted, no ADB interaction.

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
- **Base Commit:** `8dfc5eb82c6079fbed6796f0d9dbb014384a0a45`
- **Shared Contract Header:** `host_tests/hmx_acc_residual_rpc/hmx_b3_r2.h`
- **CPU Full-Tile Simulator:** `host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c`
- **Runner Script:** `scripts/run_hmx_b3_r2_sim.ps1`
- **1,024-Cell Oracle CSV:** `build/hmx_b3_r2_1024_cells.csv`

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

### Inviolable B3 Invariants
- **NEVER** clear ACC between MMA #1, #2, and #3.
- **NEVER** round or convert intermediate products to FP16 between MMAs.
- **NEVER** store intermediate products separately to add them in HVX.
- **NEVER** re-execute MMAs between Convert #1 and Convert #2.
- **NEVER** enable feedback (`Rs[4]=0`, `Rs[3:2]=0`).
- **NEVER** alter the product accumulation order ($A_{hi}W_{hi} \rightarrow A_{hi}W_{lo} \rightarrow A_{lo}W_{hi}$).

---

## 4. VTCM 12 KiB Layout and Staged Operand Streaming

### Problem Statement
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

### Verified Solution: Sequential Operand Streaming
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

### Operand Transfer Timeline
1. **Prologue:**
   - Pre-pack $A_{hi}$ at `act` (offset 0) and $W_{hi}$ at `wgt` (offset 2048).
   - Pre-pack $A_{lo}$ and $W_{lo}$ in host-allocated DSP L2/TCM memory.
   - Pre-fill `out_pos`, `out_neg`, `out_recon` with sentinel `0xA5A5`.
   - Load Bias sets 0 and 1.
2. **MMA #1 ($A_{hi} \times W_{hi}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }`.
3. **Stream Update 1 (for MMA #2):**
   - $A_{hi}$ remains resident in `act`.
   - DSP HVX vector copy streams $W_{lo}$ (16 vectors $\times 128\text{ B} = 2,048\text{ B}$) into `wgt` (offset 2048).
   - Memory synchronization barrier (`asm volatile("" ::: "memory")`).
4. **MMA #2 ($A_{hi} \times W_{lo}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }` into the retained ACC.
5. **Stream Update 2 (for MMA #3):**
   - DSP HVX vector copy streams $A_{lo}$ into `act` (offset 0).
   - DSP HVX vector copy streams $W_{hi}$ back into `wgt` (offset 2048).
   - Memory synchronization barrier.
6. **MMA #3 ($A_{lo} \times W_{hi}$):**
   - HMX executes `{ activation.hf = mxmem(act); weight.hf = mxmem(wgt); }` into the retained ACC.
7. **Convert & HVX Reconstruction:**
   - Convert #1 stores to `out_pos` (offset 4096).
   - Convert #2 stores to `out_neg` (offset 6144).
   - DSP executes 16 iterations of `Q6_Vhf_vsub_VhfVhf` writing to `out_recon` (offset 8192).

### Architectural Safety Audits
1. **Accumulator Immunity:** The HMX 37-bit ACC resides in internal execution registers, not in VTCM. Overwriting bytes at offsets 0..4095 has zero impact on accumulator contents.
2. **Output Buffer Protection:** Offsets 4096..10239 are never touched during the MMA phase. Sentinel patterns `0xA5A5` confirm non-interference.
3. **Streaming Cost:** Total data streamed mid-kernel is $3 \text{ tiles} = 6\text{ KiB} = 48\text{ HVX vectors}$. On Hexagon V81, vector loads/stores take 1 cycle per 128 bytes ($\approx 48\text{ cycles} \approx 48\text{ ns}$ at 1 GHz), adding $< 0.1\%$ execution overhead.
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
  - 3 MMA packets (staged streaming)
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
Attempts allowed    = 1 clean run (NO retries)
```

---

## 6. 1,024-Cell CPU Reference Results (Pre-Registered)

The following baseline metrics were produced by `host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c` and are **frozen**:

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

The following 13 fault models are implemented and verified in `hmx_b3_r2_full_tile_sim.c`. All 13 fail closed:

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
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                                   { All Agree? }
                                    /          \
                                  No            Yes
                                 /                \
          [ B3_R2_HOST_CONTRACT_FAILED ]           ▼
                                            [ Implementation Check ]
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
                         [ B3_R2_NUMERICAL_BLOCKED ]                      ▼
                                                            [ Subnormal Boundary Check ]
                                                            - s3 cells match RN-even?
                                                             /                       \
                                                           No                         Yes
                                                          /                             \
                                           [ B3_R2_HVX_SUB_FTZ_OBSERVED ]  [ B3_R2_FULL_TILE_SUPPORTED ]
```

---

## 10. Pre-Registration Verdict

**`B3_R2_FULL_TILE_CPU_READY`**

- The 3-product B3 accumulation and R2 two-convert sign separation are architecturally proven to co-exist within 10,752 B of the 12 KiB VTCM via staged sequential operand streaming.
- The 1,024-cell numeric oracle is frozen with exact bit patterns recorded in `build/hmx_b3_r2_1024_cells.csv`.
- All 13 failure modes are verified to fail closed.
- **No DSP implementation, build, or device contact is authorised by this document.**
