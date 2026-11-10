# D8 FastRPC Failure and Evidence-Preservation Contract (Revision 3 Proposal)

Status: **`D8_REV3_PROPOSED_CPU_ONLY`**, 2026-11-10. Part of
`D8_REV3_PROPOSED_CPU_ONLY`. This document authorises nothing: no DSP source,
no DSP build, no QAIRT, no ADB, no device contact, no Mode 46.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md`
- `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md`

## 1. The ABI facts this contract must respect

The IDL is unchanged:

```text
long run(rout sequence<long> results, rout sequence<long> metadata,
         in long q, in long mode);
```

1. **`rout` length is the client maximum.** `resultsLen` and `metadataLen` are
   supplied by the host; the callee can check them but cannot return a
   different count. There is no "short return" or "long return" observable.
2. **A non-zero method return discards the `rout` data.** FastRPC/QAIC does not
   propagate the output arrays when the method returns non-zero. The host then
   has only the status code.
3. **The DSP `run()` zeroes both buffers before dispatch.** A host-side poison
   slot cannot distinguish "written zero" from "never written".
4. **The current `run()` returns non-zero on any cleanup failure**, even after
   a successful handler. That would discard a completed experiment's arrays.

Revision 2's "preserve the raw bytes on failure" and "short/long return is a
contract failure" are therefore not implementable as written. Revision 3
replaces them with the rules below.

## 2. Failure states

<!-- d8-failure-states:begin -->
S0 preexec_arg 14001 no_metadata D8_CONTRACT_FAILED_PREEXEC none
S1 preexec_resource nonzero no_metadata D8_CONTRACT_FAILED_PREEXEC none
S2 contract_after_start 0 full D8_CONTRACT_FAILED none
S3 guard_after_numeric 0 full D8_CONTRACT_FAILED none
S4 cleanup_after_start 0 full D8_CONTRACT_FAILED_CLEANUP none
S5 transport_after_invocation unknown unavailable D8_EVIDENCE_LOST none
S6 host_post_processing unknown partial D8_HOST_EVIDENCE_FAILED none
<!-- d8-failure-states:end -->

Fields: state, trigger, `run()` return, `rout` evidence, host class, numeric
reading.

### S0 — pre-execution argument failure

Examples: `resultsLen != 1024`, `metadataLen != 32`, `q` outside 6..9,
`mode != 46`, null buffer.

- `run()` returns `HMX_ERR_ARG` (14001) before power, lock or VTCM;
- no metadata is propagated;
- the host records `D8_CONTRACT_FAILED_PREEXEC` with the status and stops;
- there is no experiment data to preserve, and no numeric reading;
- this is a host/contract bug, not a D8 result.

### S1 — pre-execution resource failure

Examples: `HAP_power_set` (14006), HMX lock (14004), VTCM reserve (14003),
pointer alignment (14002), HVX lock (14010).

- `run()` returns the existing non-zero code;
- the handler has not written `metadata[9]`, so there is no D8 evidence;
- the host records `D8_CONTRACT_FAILED_PREEXEC` and stops;
- **no retry** under the registered budget. The environment failed; a retry
  would be a new session and needs a new registration.

### S2 — contract failure after execution started

Examples: a count, mask, address, FNV, selector or issued-order mismatch.

- "Execution started" means the handler acquired VTCM and HVX and wrote
  `metadata[9]`;
- the handler writes `metadata[26]` (failure code) and returns 0;
- `run()` sets `metadata[7] = 1`, `metadata[8] = 1` and returns 0;
- the host receives the full arrays and records `D8_CONTRACT_FAILED` with the
  first failing field and the word-26 reason;
- no numeric reading. The raw arrays are preserved before `rpcmem_free`.

### S3 — guard failure after the numeric result exists

Examples: post digest differs, first bad byte set, guard status inconsistent.

- the numeric result exists but the memory window is not the registered one;
- the handler returns 0 with `metadata[26] = 8` (GUARD_POST);
- the host records `D8_CONTRACT_FAILED` and **does not compare the array with
  a reference**. A guard failure invalidates the run even when the numbers
  equal `safe`;
- raw arrays preserved; no retry.

### S4 — cleanup failure after execution started

Examples: `hmx_unlock2` (14007), `compute_res_release` (14008), power-down
(14009).

The current `run()` returns the cleanup code, which discards the arrays.
Revision 3 requires a **mode-46-only exception**:

1. after the handler, `run()` writes `metadata[30]` with the first cleanup
   error (0 if none);
2. if `metadata[7] == 1`, `run()` returns 0 even when `metadata[30] != 0`;
3. the host checks word 30 and records `D8_CONTRACT_FAILED_CLEANUP`, with no
   numeric reading;
4. the arrays survive because the metadata buffer is host memory and is not
   affected by releasing VTCM.

If the implementer refuses the exception, a cleanup failure after execution is
**`D8_EVIDENCE_LOST`**: the host sees only the non-zero status and cannot know
whether the four MMAs ran. The run stops, and the experiment must be
re-registered before another attempt.

The exception is scoped to mode 46 and only when `metadata[7] == 1`. A cleanup
failure before execution keeps the normal non-zero return.

### S5 — transport failure after invocation

Examples: FastRPC error, handle failure, process death.

- the host cannot know whether the handler ran;
- no arrays, no metadata, no failure code;
- record `D8_EVIDENCE_LOST` (or `D8_TRANSPORT_FAILED` only when the failure is
  provably before invocation);
- **no retry**. A second run would be an unregistered fifth run.

### S6 — host post-processing failure

Examples: allocation, copy, digest or record failure after a successful RPC.

- the RPC data may already be lost when the host fails to copy it;
- record `D8_HOST_EVIDENCE_FAILED`;
- if the raw buffer still exists, preserve it before any parsing;
- no numeric reading.

## 3. Return-value rule

| Situation | Return |
| --- | ---: |
| argument / power / lock / VTCM / alignment / HVX lock, before `metadata[9]` | existing non-zero code |
| any contract failure after `metadata[9]` | **0** |
| success | **0** |
| cleanup failure after `metadata[7] = 1` | **0** (mode-46 exception), word 30 set |
| cleanup failure before `metadata[7] = 1` | existing non-zero code |
| transport failure | not a `run()` return |

"Always return 0" is **not** the rule. The rule is: preserve evidence once the
experiment has started; keep the transport status for environment failures.

## 4. Host decision order

1. If the RPC status is non-zero and is one of the documented pre-execution or
   cleanup-before-start codes: `D8_CONTRACT_FAILED_PREEXEC`; stop.
2. If the RPC status is non-zero but the invocation may have started:
   `D8_EVIDENCE_LOST`; stop.
3. If the status is 0: copy the raw arrays immediately.
4. Check `resultsLen == 1024`, `metadataLen == 32`, every slot `0..65535`.
5. Check metadata words 0..8, 9..31 per the evidence contract.
6. Check word 30 (cleanup); a non-zero value is
   `D8_CONTRACT_FAILED_CLEANUP`, no numeric reading.
7. Check word 26 (failure); a non-zero value is `D8_CONTRACT_FAILED`.
8. Check the guard words; a failure is `D8_CONTRACT_FAILED`, no numeric
   reading even if the array matches `safe`.
9. Only then compare the 1,024 cells exactly with the four references and
   apply the classification order.

## 5. Evidence availability matrix

| Evidence | S0 | S1 | S2 | S3 | S4 (exception) | S4 (no exception) | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RPC status | yes | yes | yes (0) | yes (0) | yes (0) | yes (non-zero) | maybe | yes |
| metadata 0..8 | no | no | yes | yes | yes | no | no | maybe |
| metadata 9..31 | no | no | yes | yes | yes | no | no | maybe |
| results 1,024 | no | no | yes | yes | yes | no | no | maybe |
| failure code | no | no | yes | yes | no | no | no | no |
| cleanup code | no | no | no | no | yes | no | no | no |
| numeric interpretation | no | no | no | no | no | no | no | no |

"maybe" means the host may have copied the buffer before the failure; the
contract gives it no guarantee.

## 6. What this contract does not fix

1. It does not make a non-zero return preserve `rout` data. Only the return-0
   rule does that.
2. It does not make the cleanup exception exist in the current `run()`; that is
   an implementation change, forbidden in this CPU-only phase.
3. It does not make a transport failure recoverable.
4. It does not let a host detect "the DSP wrote fewer than 1,024 slots" beyond
   the slot-value rule, the `slots_written` self-report, the exact comparison
   and the disassembly.
5. It does not authorise a retry. The registered budget is four runs, zero
   retries.

## 7. CPU-side checks

The failure states are exercised at the contract level, not on a device:

- `hmx_d8_rev3_selftest.c` checks that a non-zero failure code, a non-zero
  cleanup word, a short slot count, a claimed trace, an HVX-lock failure and a
  missing readback bit are all rejected before any numeric reading;
- `hmx_d8_rev3_contract_test.py` parses the state block above and checks that
  every state has a return class, an evidence class, a host class and "no
  numeric reading", and that the return-0 rule and the cleanup exception are
  stated;
- `hmx_d8_min_sim.c` checks that numeric-safe runs with an order or guard defect
  are still rejected.

No RPC is made. The C/Python check is a contract test, not a transport test.
