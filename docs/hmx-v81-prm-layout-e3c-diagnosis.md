# HMX V81 VTCM layout and the E-3C1 mode-42 mismatch

Status: CPU-only diagnosis, 2026-10-22. No DSP change, no device run, no
production path for the diagnosis itself; the separately approved mode-43
follow-up was implemented and executed on the same day (§8). Classification:
**`MULTIPLE_CAUSES_IDENTIFIED`** with
sub-findings `LAYOUT_MISMATCH_RESOLVED` (both operands) and
`CONVERT_STORE_MAPPING_RESOLVED`; `ISSUE_GEOMETRY_MISMATCH_NOT_FOUND`.

Primary source: *Qualcomm Hexagon V81 HMX PRM*, 80-N2040-62 Rev. AA
(Qualcomm Hexagon SDK 6.6.0.0 documentation set,
`docs/pdf/80-N2040-62_AA_Qualcomm_Hexagon_v81_HMX_PRM.pdf`). Section and
figure numbers below refer to that document. The replay model is
`host_tests/hmx_acc_residual_rpc/hmx_prm_layout_sim.c`, run by
`scripts/run_prm_layout_sim.ps1` (CPU only, not part of the Fast gate).

## 0. Conclusion

Mode 42 did not write the activation and weight croutons in the layout
HMX reads. It wrote both 32x32 logical tiles **row-major**, while the
HMX crouton is **pair-interleaved**: one 128-byte VTCM vector holds two
spatials (activation) or two input channels (weight) across all 32 of
the other dimension. In addition, the readback interpreted `out[32]` as
`(s1,o0)` (L1) or `(s0,o1)` (L2), but the store layout makes
`out[32]` = `(s0,o16)`.

Replaying the recorded mode-42 fixture through the PRM layout with the
row-major buffers reproduces **all four measured slots exactly**:

| slot | model | measured |
| --- | --- | --- |
| plain `out[0]` pre-clamp | `0xcf88` (-30.125) | - |
| plain `out[0]` stored | `0x0000` (negative-store clamp) | `0x0000` |
| B3 `out[0]` stored | `0x0000` (same pre-clamp `0xcf88`) | `0x0000` |
| plain `out[32]` | `0x59f6` | `0x59f6` |
| B3 `out[32]` | `0x59f7` | `0x59f7` |

`out[32]` is the device cell `(s0,o16)`; under the row-major buffers it
is a **gather artifact**, not the output of any logical tile cell. That
is why the earlier CPU sweep over logical tile cells never found
`0x59f6`: the search space was wrong, not the device.

The register values (`act_rt=0x77C`, `wgt_rt=0x7FF`, `wr_rt=0x700`)
decode to a correct 4x8 spatial plane (`SM=11100`) with 32 channels.
There is no issue-geometry mismatch.

## 1. Spec-derived layout

### 1.1 Bit skeleton

All three VTCM structures use the same 16-bit cell index skeleton.
Index bits mean:

| structure | bit 0 | bits 5:1 | bits 9:6 |
| --- | --- | --- | --- |
| activation | spatial LSB `s&1` | input channel `i` | spatial `s>>1` (vector) |
| weight | input-channel LSB `i&1` | output channel `o` | input channel `i>>1` (vector) |
| convert store | spatial LSB `s&1` | output channel `o` | spatial `s>>1` (vector) |

In u16 units (byte offset = 2 x index):

```text
act_u16(s,i)   = (s >> 1) * 64 + i * 2 + (s & 1)
wgt_u16(o,i)   = (i >> 1) * 64 + o * 2 + (i & 1)
store_u16(s,o) = (s >> 1) * 64 + o * 2 + (s & 1)
```

All three indexes stay within `[0, 1023]` (2 KB). The last cells are
`act(31,31) = 1023`, `wgt(31,31) = 1023`, `store(31,31) = 1023`, so the
crouton is exactly 2 KB with no padding.

### 1.2 Activation crouton

PRM §2.2.1, Figure 4 ("Activation VTCM organization"): two activations
spatials share a 32-bit VTCM word; the blue cells are the first 32 bits,
then vector 1, etc. The figure annotates the first vector as
`0, 4, ... 124` for spatial 0 and `2, 6, ... 126` for spatial 1, i.e.
spatial even at the low half and spatial odd at the high half of every
32-bit word, with the input channel advancing every 4 bytes. §3.1
`write_act_vtcm` corroborates this directly:

```c
for (vec = 0; vec < 16; vec++)
  for (ch = 0; ch < 32; ch++)
    for (sp_in_vec = 0; sp_in_vec < 2; sp_in_vec++) {
      spatial = vec * 2 + sp_in_vec;
      *addr++ = fp16(spatial + ch * 0.01);
    }
```

which is exactly `(s>>1)*64 + i*2 + (s&1)` in u16 units.

### 1.3 Weight crouton

PRM §2.2.2, Figure 8 ("Weights VTCM organization") and §3.1
`write_wgt_vtcm`:

```c
for (vec = 0; vec < 16; vec++)          /* two input channels per vector */
  for (out = 0; out < 32; out++)
    for (in_ch_vec = 0; in_ch_vec < 2; in_ch_vec++) {
      input_channel = vec * 2 + in_ch_vec;
      *addr++ = fp16(...);
    }
```

which is `(i>>1)*64 + o*2 + (i&1)`: the output channel varies fastest
(every 4 bytes, 32 outputs = 128 bytes) and the two input channels sit
at byte offsets 0 and 2 inside the vector.

**Figure 7 caution.** The weight address-bit diagram labels bits 9,8,7
as input channel, bits 6..2 as output channel, bit 1 as input channel,
and omits bit 10. With only three vector bits that decode gives 8
vectors = 16 input channels, contradicting "32 input channels" and the
2 KB size. The §3.1 writer and the 2 KB arithmetic require bits 10:7 =
`i>>1` (16 vectors x 2 channels = 32). The figure is missing one input
label at bit 10; the code is authoritative. Do not derive a weight
packing from Figure 7 alone.

### 1.4 Accumulator and convert

PRM §2.2.3, Figure 9: one accumulator is 32 spatial x 32 output-channel
x 37-bit floating point. §2.2.4, Figure 10 and Equation 3: a single
multiply instruction computes, for every spatial and output channel,

```text
ACC(s,o) = sum_i A(s,i) * W(o,i)
```

with the same `i` range selected by the activation/weight registers.
The spatial mask only defines the 2D ordering of the 32 spatials; it
does not permute output channels or input channels.

PRM §2.3, Figure 16: "Every cell in the accumulator goes through the
same convert operation and is stored in the corresponding location in
the convert state register" and "accumulator (0,0) is converted with
the parameters from bias (0) and stored at convert state (0,0)". The
mapping is therefore **identity**:

```text
ACC(s,o) -> CVT(s,o)
```

There is no convert permutation. Mode 42's `cvt.hf = acc(1)` reads
bias set 0 with `Rs[0]=1` (retain); the bias register is scale 64, shape
0, input bias 0, output bias 0, so the convert target is `64 * ACC`.

### 1.5 Write convert state

PRM §2.4 and Figure 18/19: the convert state (spatial x output channel)
is stored as a 3D structure (spatial-x, spatial-y, output channel) and
"the result is organized similarly to the activation crouton in VTCM
with two spatials stored in a vector" (Figure 19). Applying the
activation pair rule to the convert state gives
`store_u16(s,o) = (s>>1)*64 + o*2 + (s&1)`. The bit skeleton in §1.1
shows the key consequence at index 32:

```text
u16 index 32 = 0b0000100000
             = vector 0 (bits 9:6 = 0)
             + output channel 16 (bit 5 = o bit 4)
             + spatial even 0 (bit 0 = 0)
             -> (s0, o16)
```

### 1.6 Register fields and issue geometry

PRM Table 1 (activation) / Table 2 (weight) / Table 8 (write convert
state), and the §3.2 reference code:

* 5-bit spatial mask `rt_sp_mask = 0b11100` = `YYYXX` = 4 spatial-x by
  8 spatial-y. The register field spreads it as
  `((mask >> 1) << 7) | ((mask & 1) << 1)`.
* Mode 42 passes `spatial_mask = 14 << 7`, which is exactly
  `(0b11100 >> 1) << 7`; with `Rt[6:2] = 31` (channel stop) this yields
  `act_rt = 0x77C`. `Rt[31:11] = 0` (dY/dC).
* `wr_rt = 14 << 7 = 0x700` is the same encoded mask with all store
  reserved bits zero.
* `wgt_rt = 2048 - 1 = 0x7FF` is `dW = 15` (last 128-byte vector of a
  2 KB crouton) with the required reserved `[6:0] = 0x7F`; this is the
  PRM's own `wgt_size - 1`.
* For `SM = 11100` the flattened spatial index is `s = y*4 + x`
  (low two bits X, high three bits Y). Mode 42 uses `s` as a flat
  0..31 index, so this only fixes the 2D interpretation, not the dot
  product.

All of these are correct in mode 42. No issue-geometry mismatch was
found.

## 2. Mapping tables

### 2.1 Activation

| u16 idx | byte | vector | cell |
| --- | --- | --- | --- |
| 0 | 0 | 0 | (s0, i0) |
| 1 | 2 | 0 | (s1, i0) |
| 2 | 4 | 0 | (s0, i1) |
| 3 | 6 | 0 | (s1, i1) |
| 32 | 64 | 0 | (s0, i16) |
| 33 | 66 | 0 | (s1, i16) |
| 64 | 128 | 1 | (s2, i0) |
| 65 | 130 | 1 | (s3, i0) |

### 2.2 Weight

| u16 idx | byte | vector | cell |
| --- | --- | --- | --- |
| 0 | 0 | 0 | (o0, i0) |
| 1 | 2 | 0 | (o0, i1) |
| 2 | 4 | 0 | (o1, i0) |
| 3 | 6 | 0 | (o1, i1) |
| 32 | 64 | 0 | (o16, i0) |
| 33 | 66 | 0 | (o16, i1) |
| 64 | 128 | 1 | (o0, i2) |
| 65 | 130 | 1 | (o0, i3) |

### 2.3 Convert store, and DSP slot -> CPU oracle cell

CPU oracle index is row-major `cell = s*32 + o`. Decoding a store u16
index `dst`: `v = dst >> 6`, `r = dst & 63`, `s = 2v + (r & 1)`,
`o = r >> 1`.

| dst u16 | byte | vector | device cell | oracle index |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | (s0, o0) | 0 |
| 1 | 2 | 0 | (s1, o0) | 32 |
| 2 | 4 | 0 | (s0, o1) | 1 |
| 3 | 6 | 0 | (s1, o1) | 33 |
| 32 | 64 | 0 | (s0, o16) | 16 |
| 33 | 66 | 0 | (s1, o16) | 48 |
| 64 | 128 | 1 | (s2, o0) | 64 |
| 65 | 130 | 1 | (s3, o0) | 96 |

## 3. Replaying mode 42

### 3.1 Fixture and model

The model reuses the exact mode-42 fixture: xorshift64 seed
`0xE3C0117E5EEDF00D`, the same literal rows/lanes, the same
float-arithmetic hi/lo split, and the same instruction sequence (plain:
one packet; B3: three packets into one accumulator, same convert).

"Row-major buffer" means the DSP wrote `buffer[k] = tile[k]` with
`k = row*32 + col`, exactly as mode 42 did. The model then reads that
buffer through §1's indexes and computes the PRM dot product. The
convert is `RN16(64 * acc)`; the negative-store clamp is the empirical
README law 4 (`store(x)=0 for x<=0`), not a PRM rule.

One honesty note: both a float32 accumulation chain and a float64 chain
produce identical stored bits for every slot and every combination
below. The measurement therefore identifies the layout, **not** the
internal accumulator precision. The PRM says 37-bit floating point; no
FP32 claim is made.

### 3.2 Measured slots

Mode 42 wrote four operand tiles row-major. The device read them through
the pair-interleaved indexes, so the activation vector for device
spatial `s` is a strided gather of the row-major array:

```text
logical index read for device (s,i) = (s>>1)*64 + 2i + (s&1)
```

That selection alternates between the two logical rows of the pair, so
it is not any logical row. The weight read is likewise a gather, not
`W[o]`. The computed values are therefore new numbers, not a
permutation of the logical tile.

| readback | device cell | plain | B3 |
| --- | --- | --- | --- |
| `out[0]` | (s0,o0) | `0xcf88` -> clamp `0x0000` | `0xcf88` -> clamp `0x0000` |
| `out[32]` | (s0,o16) | `0x59f6` | `0x59f7` |

Exact match at all four slots. The plain/B3 1-ulp difference at
`out[32]` confirms that both packet variants reached the same convert;
it says nothing about tile-wide accuracy.

### 3.3 Uniqueness

Scanning the full 32x32 as-written grid, exactly **one** device cell has
plain `0x59f6` **and** B3 `0x59f7`: `(s0,o16)`, store index 32. The
pre-registered L1/L2 candidates are different values
(plain `0xa388` at `(s1,o0)`, plain `0xd5ef` at `(s0,o1)`), so neither
candidate family could ever have matched.

### 3.4 Four packing combinations

| activation | weight | `out[0]` plain (store) | `out[32]` plain | `out[32]` B3 | matches measurement |
| --- | --- | --- | --- | --- | --- |
| row-major | row-major | `0xcf88` -> `0x0000` | `0x59f6` | `0x59f7` | **yes (4/4)** |
| row-major | PRM | `0x5400` | `0x54e8` | `0x54e8` | no |
| PRM | row-major | `0x5400` | `0x5400` | `0x5400` | no |
| PRM | PRM | `0x5400` | `0x55e3` | `0x55e3` | no (correct model) |

Only row-major/row-major produces the measured signature
(`out[0]` stored zero **and** `out[32]` = `0x59f6`/`0x59f7`). Note that
three of the four combinations give a positive `out[0] = 0x5400`; the
measured `0x0000` alone already rules them out, because it requires a
negative pre-clamp value at `(s0,o0)`.

### 3.5 Correctly packed predictions

With both operands written at the PRM offsets, the same model predicts
the intended logical cells:

| slot | device cell | plain | B3 |
| --- | --- | --- | --- |
| `out[0]` | (s0,o0) | `0x5400` | `0x5401` |
| `out[1]` | (s1,o0) | `0xd400` (signed) / `0x0000` (clamp) | `0xd401` / `0x0000` |
| `out[2]` | (s0,o1) | `0x5400` | `0x5400` |
| `out[32]` | (s0,o16) | `0x55e3` | `0x55e3` |

`out[1]` is the negative twin of `(s0,o0)` and is the signed-store
discriminator. These are CPU predictions only; none was measured.

## 4. Diagnosis

| # | cause | evidence |
| --- | --- | --- |
| 1 | activation crouton written row-major instead of pair-interleaved | §2.2.1/Fig. 4 + §3.1 writer; exact `out[0]`/`out[32]` replay |
| 2 | weight crouton written row-major instead of pair-interleaved | §2.2.2/Fig. 8 + §3.1 writer; four-combination table |
| 3 | readback assumed a row-major store family (L1/L2) | §2.4/Fig. 18-19; `out[32]` = `(s0,o16)` |
| 4 | negative final store clamps to zero | README law 4; `out[0]` pre-clamp `0xcf88` vs stored `0x0000` |

This is `MULTIPLE_CAUSES_IDENTIFIED` for the observed mismatch. In
device terms there is one root cause (operand packing); the readback
assumption and the clamp are separate test/provenance causes. The
issue geometry and convert mapping are correct, so no
`ISSUE_GEOMETRY_MISMATCH` exists.

What the earlier CPU search missed: it enumerated logical tile cells
and axis transpositions, but the device was computing dot products of
pair-interleaved gathers of a row-major buffer. No logical-cell search
could produce `0x59f6`.

## 5. Code problems (mode 42)

`host_tests/hmx_acc_residual_rpc/probe_dsp.c`, mode-42 block:

1. Operand tiles are derived into linear arrays at
   `act_hi[idx]`, `act_lo[idx]`, `wgt_hi[idx]`, `wgt_lo[idx]` with
   `idx = row*32 + col` and passed directly to `mxmem`. This is the
   root cause; the instruction constants are correct.
2. The pre-registered comment says `out[0]` is `(s0,o0) "under BOTH
   layout candidates"` (true) but claims `out[32]` is `(s1,o0)` (L1) or
   `(s0,o1)` (L2) (false). The store layout gives `(s0,o16)`.
3. The variable named `spatial_mask = (14u << 7)` is the *encoded
   register field* for the 5-bit mask `0x1C`, not the mask value.
   The value is right; the name invites the wrong decode.
4. `probe_host.c` implements only the four pre-registered candidates,
   so the catch-all `E3C1_INCONCLUSIVE_DIAGNOSE_PACKET_LANE_CONVERT`
   was the correct output of that decision procedure. The recorded
   device verdict is historical and is not rewritten.

No DSP, host-gate, or production code was changed by this diagnosis.

## 6. CPU validation

```powershell
.\scripts\run_prm_layout_sim.ps1       # gcc gate + optional clang
.\scripts\run_fp16x2_split_sim.ps1     # existing G8 regression
.\scripts\verify.ps1 -Profile Fast
```

The model asserts, and the runner prints, at least:

* the three mapping tables for indexes 0/1/2/3/32/33/64(/65);
* `act_rt = 0x77C`, `wgt_rt = 0x7FF`, `wr_rt = 0x700`, mask `0x1C`;
* the as-written `out[0]`/`out[1]`/`out[2]`/`out[32]` bits for plain and
  B3, including the pre-clamp value;
* uniqueness of `(s0,o16)` for `0x59f6`/`0x59f7`;
* all four packing combinations and the corrected-packing predictions;
* `prm_layout_sim=PASS`.

Result (2026-10-22): `prm_layout_checks=72 fail=0`,
`prm_layout_sim_gcc_exit=0`, `prm_layout_sim_clang=PASS`
(x86_64-pc-windows-msvc19.44.35229), script exit 0. The expected
literals were derived from the PRM and the recorded mode-42 slots
before the C model was written; every one matched on the first compute
(no retuning). The existing G8 regression still passes
(`fp16x2_split_sim=PASS cases=115563`).

This is a model check, not independent device evidence. The device
evidence is the recorded mode-42 data.

## 7. Unresolved items

1. **Negative-store clamp.** The PRM's convert equation and IEEE FP16
   rounding predict a negative stored value for `(s0,o0)`. The device
   stores zero. This diagnosis extends README law 4 from fine convert
   targets to a large negative final value (-30.125), but the mechanism
   (output-bias clamp, store saturation, or another undocumented stage)
   is not specified.
2. **Figure 7 bit 10.** The weight address-bit figure omits the
   input-channel vector bit; §3.1 code and 2 KB arithmetic require it.
   Treat the figure as ambiguous.
3. **ACC precision.** 37-bit floating point per §2.2.3. The stored bits
   examined here are identical under float32 and float64 chains, so no
   internal-precision conclusion follows.
4. **Store layout beyond index 0.** Index 0 was previously
   device-verified as `(s0,o0)`. Index 32 matches the derived formula in
   this run, but the same run is the evidence, so it is one model-data
   agreement rather than an independent probe of the full store map.
   Indexes 1, 2, 64 and an odd-spatial slot are untested on device.
5. **Accuracy.** Mode 42's four slots say nothing about tile-wide plain
   vs B3 accuracy. E-3C1 device accuracy remains
   `INSUFFICIENT_EVIDENCE`; E-3C2 stays untriggered.

## 8. Mode 43 (E-3C1R) — implemented and executed 2026-10-22

A sparse layout-identifier probe is **not needed**: the layout is
already uniquely determined by the recorded mode-42 slots and the PRM.
The remaining device question is narrower: does correctly packed input
produce the intended cells, and is the negative final store signed or
clamped? The approved single confirmation run answered the packing and
slot-1 question at four-slot scope.

**Mode 43 — corrected-packing confirmation (E-3C1R), EXECUTED under
separate approval (PR #14 comment 6335830744): one clean stamped build
plus one `-DeployAndRun` session, no retry.**

* Reused the exact mode-42 fixture and instruction sequence; the only
  intended data change was the plain Ahi/Whi packing at
  `act_u16`/`wgt_u16` offsets. No instruction, register or bias change.
  Mode 42's code path and historical verdict were left intact.
* One plain trial; a dedicated DSP handler returned four store slots
  through the existing `results[4]` ABI: `out[0]`, `out[1]`, `out[2]`,
  `out[32]`. Plain-only was deliberate: B3 is not an independent
  variable for layout, and its 1-ulp sensitivity was already observed
  at `out[32]` in mode 42. `trials=1`; no IDL/stub change. The host
  branch used the shared `hmx_mode43.h` decision procedure and failed
  closed unless the trial count was exactly one.
* Pre-registered expectations (fixed before the run):

| slot | expected | classification meaning |
| --- | --- | --- |
| `out[0]` | `0x5400` | base cell / positive path |
| `out[1]` | `0xd400` | signed final store |
| `out[1]` (alternative) | `0x0000` | clamp final store |
| `out[2]` | `0x5400` | output-channel stride |
| `out[32]` | `0x55e3` | output-channel 16 stride |

* Verdicts: `E3C1R_LAYOUT_CONFIRMED_SIGNED` (all four as signed),
  `E3C1R_LAYOUT_CONFIRMED_CLAMP` (`out[1]=0x0000`, others as above),
  `E3C1R_RESIDUAL_MISMATCH` (anything else; stop and re-diagnose, no
  retuning).
* Result (raw, 2026-10-22; build `d7c1603` / build id `0xd7c1603c`):
  `out[0]=0x5400`, `out[1]=0x0000`, `out[2]=0x5400`,
  `out[32]=0x55e3`; positive checks 3/3; `hmx_executed=1`, `trials=1`,
  `rpc_status=0`, `close_status=0`, exit 0 →
  **`E3C1R_LAYOUT_CONFIRMED_CLAMP`**. This confirms the corrected
  packing at the four readback slots and records the clamp candidate.
  It does not locate the clamp stage (ACC, convert or store), does not
  establish a signed store, and does not measure full-tile accuracy.
* Cost/risk: same VTCM reserve (12 KB); mode 43 uses act+wgt+out plus
  two 256-byte bias sets (6.5 KB). One plain trial (1 MMA packet)
  instead of mode 42's two trials (4 packets). One build, one device
  run; the approved one-run gate was applied with no retry and no B3
  expansion.
* Explicitly out of scope: full-tile accuracy measurement (E-3C2
  remains untriggered), throughput, signed-store law, clamp-stage
  attribution, production integration.

The run preserved mode 42's historical measurement and verdict. Further
device work (B3 expansion, clamp-stage or signed-store discriminator)
requires new explicit authorization. E-3C2 remains untriggered.

## 9. Files and next gate

* `host_tests/hmx_acc_residual_rpc/hmx_prm_layout_sim.c` — CPU model,
  assertions and mapping tables, all-1024 packing contract and mode-43
  decision-procedure tests.
* `scripts/run_prm_layout_sim.ps1` — gcc/clang runner, wired into the
  Fast host-contract suite.
* `host_tests/hmx_acc_residual_rpc/hmx_mode43.h` — shared mode-43
  literals, one-trial contract and four-slot classifier.
* `host_tests/hmx_acc_residual_rpc/probe_dsp.c` — dedicated one-trial
  mode-43 handler; `probe_host.c` — host gate and raw-slot reporting.
* `host_tests/hmx_acc_residual_rpc/README.md` — corrected E-3C1
  diagnosis, E-3C1R pre-registration and measured result.
* This document.

Next gate: E-3C1 accuracy remains `INSUFFICIENT_EVIDENCE`; mode 43
resolved the four-slot packing and the slot-1 clamp candidate but not
the clamp stage or a signed-store law. E-3C2 remains untriggered.
Further device work requires explicit authorization.
