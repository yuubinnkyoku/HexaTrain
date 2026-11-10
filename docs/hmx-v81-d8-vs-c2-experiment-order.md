# D8 vs C2 — Experiment Order and Decision Value

Status: **CPU-ONLY RESEARCH NOTE**, 2026-11-08. This document compares the two
minimal device experiments already pre-registered:

- **C2** — `docs/hmx-v81-cand-c-c2-device-prereg.md` (revision 2, `NOT_RUN`);
- **D8** — `docs/hmx-v81-cand-b-d8-device-prereg.md` (revision 2, `NOT_RUN`).

Neither is authorised. This note does not change either pre-registration and
does not change the Candidate B preference. It only records which experiment
should be attempted first, and why.

---

## 1. The two questions in one line each

| Experiment | Question | Design it serves |
| --- | --- | --- |
| **C2** | If software issues HMX Store #1 → HVX read of the same buffer → HMX Store #2, does the read see Store #1's bytes, or can Store #2 overtake it? | Candidate C (single shared OUT) |
| **D8** | When MMA #2 has read `Ahi`/`Wlo`, does the output-phase fill of those same regions land before the read completes? | Candidate B (phase-reused VTCM) |

They are different directions:

- C2 is **read-after-store then store-after-read** on one shared buffer: the
  output of one MX store is read by HVX, then overwritten by a second MX store.
- D8 is **store-after-MMA-read** on two reused operand tiles: the MMA reads an
  operand, then software fills the same region.

A C2 result does not answer D8, and a D8 result does not answer C2. They are not
substitutes at any confidence level.

---

## 2. Comparison

| Dimension | C2 | D8 |
| --- | --- | --- |
| MMAs | **1** | **3** |
| Operand tiles | 2 (`act`, `wgt`) | 4 (`Ahi`, `Alo`, `Whi`, `Wlo`) |
| Critical operations | 2 Convert, 2 Store to the same X, 1 HVX copy | 3 MMA, 1 Convert, 1 Store, 2 conditional fills |
| Dependency under test | HVX read → MX store over the same bytes (W-A-R) | MX operand read → software fill over the same bytes (R-W) |
| VTCM | 8,960 B used, 3,328 B margin | 10,752 B used, 1,536 B margin |
| L2 | 2,560 B arena (P + guards) | none |
| Observation | P (from L2) and X (VTCM), 2,048 cells | one 1,024-cell output + metadata |
| Input identity | registered fixture, sentinel + negative payload | registered fixture, four packed-tile FNVs |
| Reference outcomes | nominal, C1, C1+C2 mixed, zero-loss, `0xDEAD`, X-broken | safe, Wlo-corrupt, Ahi-corrupt, both-corrupt |
| Discrimination | 475 non-zero negative cells; 2 ambiguous subnormal cells; zero agreement is weak | four pairwise-distinct arrays; Wlo vs Ahi = 726 cells |
| Attribution | C1 vs C2 from P's content; zero-loss cannot separate FTZ from C2 | D8a vs D8b from conditions C/D; partial corruption is inconclusive |
| Runs per attempt | 1 | **4** (A/B/C/D) |
| Condition selector | not needed | proposed via `q`; not implemented |
| Evidence complexity | higher: P/X, guards, sentinels, L2 lifetimes, 2 store addresses | lower per run: one output, one store, four FNVs |
| Failure modes | contract (RPC, map, guards), C1, C2, FTZ/zero, inconclusive | contract (counts, FNV, mask), A/B/C/D, inconclusive |
| CPU validation | 12,828 C checks, 81 Python checks, 5,460 differential comparisons, 30/30 faults, 2-mutant regression | 34 C checks, 29 Python checks, 125 differential cases, 16 control families, 5-mutant regression |
| Path to the full tile | Candidate C's C3/C4/C5 and full-tile accuracy remain | the safe sequence **is** the B3 positive path; the full B run extends it |
| Design priority | Candidate C is third (`B > streaming > C`) | Candidate B is first |
| Existing device evidence | Mode 45 covers C1's direction only | Mode 45 covers none of D8 |

---

## 3. Decision value

### 3.1 D8 first

1. **It tests the preferred design's only remaining device-specific blocker.**
   Candidate B is `CPU_VALIDATED_PREFERRED_DESIGN`; D8 is the one property its
   whole layout rests on. A safe result supports proceeding with the full B
   build; a violation redirects the project to the streaming design before more
   B-specific work is invested.

2. **The sequence is reusable.** The D8 safe reference is cell-for-cell equal to
   the shipped B3 subcase's positive output (`diff=0`). The build, the operand
   packing, the bias setup and the first convert are the same machinery a full B
   run needs. C2 is a separate layout with a separate evidence chain.

3. **C2 answers a lower-priority design question.** Even a clean C2 result does
   not establish Candidate C: C3, C4, C5 and a full-tile run remain. A clean C2
   result changes no ranking; a D8 violation changes the recommended design.

4. **C2's efficiency advantage is real but bounded.** C2 is one run and one
   MMA; D8 is four runs and three MMAs. Per attempt, C2 buys more. But the
   project's decision is which design to pursue, and that decision turns on D8.

### 3.2 When C2 first would be better

The order should be reversed if any of these holds:

- **Only one device run is approved.** A single D8 condition (say C) answers
  only D8a; presenting it as D8 would be an overclaim. A single C2 run answers
  its full question. With a one-run budget, C2 is the better use of the run.
- **The D8 condition selector is not implementable** in a stamped build without
  a new mode registration. If the four conditions cannot be expressed, the D8
  registration is not executable and C2 proceeds while D8 is redesigned.
- **The goal is generic V81 ordering knowledge**, independent of the design
  choice. C2's read→store question is the more fundamental hazard, and its
  sentinel classification is more informative per observation.

None of these currently holds: the registered D8 budget is 4 runs, the selector
is a design proposal that a future build can carry through `q`, and the project
is design-driven (Candidate B first).

---

## 4. What each result would establish

### C2

| Result | Establishes | Does not establish |
| --- | --- | --- |
| `C2_MINIMAL_SEQUENCE_SUPPORTED` | the sequence is not disproved in that build/interleaving | Candidate C as a design; C3/C4/C5; full-tile accuracy |
| `C2_ORDERING_VIOLATED` | the shared-buffer premise is unsafe as written | which mechanism; whether a fence would fix it |
| `C1_STORE_VISIBILITY_SUSPECT` | a build/measurement problem first | new physics |
| `C2_SUBNORMAL_OR_ZERO_LOSS` | a payload cell read as zero; cause unknown | C2 or FTZ specifically |
| contract failure | nothing numeric | anything about ordering |

### D8

| Result | Establishes | Does not establish |
| --- | --- | --- |
| `D8_NO_VIOLATION_OBSERVED` | in that build, distance and run, every operand element the MMA read was pre-fill | other distances/builds/power states; the later Convert #2 / recon stores; full B; `B3_R2_SYNC_UNPROVEN` retired |
| `D8_VIOLATION_*_SUSPECT` | the fill beat the read for every cell carrying the difference (contract clean) | the internal mechanism; whether a fence would fix it; a universal rate |
| `D8_CONTROL_ANOMALY` | the minimal sequence is not correct in that build | a D8 result |
| `D8_INCONCLUSIVE_*` | nothing attributable | anything |

Both experiments share the same honesty rule: one run is a sample, not a
guarantee. Only a documented hardware guarantee could prove the ordering; a
device experiment can only fail to disprove it.

---

## 5. Recommended order

**1. D8 (conditions A, B, C, D), then 2. C2**, under the registered budgets.

Reasons, in order of weight:

1. D8 tests the preferred design; C2 tests the third-ranked one.
2. The D8 sequence is the B3 path's first convert, so its build and evidence
   chain carry over to the full B run; C2 does not.
3. A D8 result changes the design recommendation; a C2 result does not change
   the ranking.
4. C2's one-run efficiency does not compensate for answering the lower-priority
   question first, given the four-run D8 budget.

Conditions that would flip the order are listed in §3.2. If the user later
approves only one device run, run C2 and re-register D8 for a four-run budget.

Revision 2 of the D8 pre-registration corrected the fill-mask table and moved
the out_pos fill into Candidate B's output phase. Neither correction changes
the comparison above: D8 still uses four runs, three MMAs and one 1,024-cell
output, and it still tests Candidate B's only remaining device-specific
blocker. The revision makes the registered sequence match the design it is
supposed to measure; it does not change the design ranking or the run budget.

This recommendation is about experiment order only. It does not register a
mode, authorise a build, approve a device session, or change any label.
`CPU_VALIDATED_PREFERRED_DESIGN` remains a design-comparison label;
`B3_R2_SYNC_UNPROVEN` remains in force; C2 and D8 are both `NOT_RUN`.
