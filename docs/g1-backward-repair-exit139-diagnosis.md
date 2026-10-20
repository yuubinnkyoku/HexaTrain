# G1 Backward Repair — Formal Cohort exit139 Root Cause Diagnosis

Offline diagnosis of the 2026-10-20 formal cohort process that exited 139
(Segmentation fault) during `standalone_dvalue` graph construction.

Status: **`EXIT139_ROOT_CAUSE = STRONGLY_SUPPORTED`**, formal cohort
**`BLOCKED_FORMAL_COHORT`**, `FORMAL_COMPLETED = 0/3`, `TRAINING_READY = NO`,
40k→60k / 200k **NO-GO**. No status was promoted by this work.

No DSP arithmetic was executed. The formal cohort was not retried. No frozen
DSP/Host/probe binary was modified or rebuilt. No role order, coverage,
fixture, oracle or numeric threshold was changed. No existing private evidence
was modified, deleted or overwritten.

Parent report and the current 41-item answers:
[`docs/g1-backward-repair-native-completion.md`](g1-backward-repair-native-completion.md).

---

## A. Environment and evidence authenticity

| Item | Observed |
| --- | --- |
| Checkout | the G1 research checkout named in the mission (separate clone, own `.git`); local absolute paths are intentionally not recorded |
| Branch | `investigation/g1-backward-repair-native-completion` |
| HEAD at start | `30d0cb3bc5eb9e632ac869a122438b3fab64ae06` |
| Working tree at start | clean |
| Git common dir | `<checkout>/.git` (not a linked worktree of any other clone) |

Frozen artifact hashes re-verified independently against the bundle
`build-identity.json`:

| Artifact | SHA256 | Match |
| --- | --- | --- |
| `saved_p_backward_probe` | `23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05` | yes |
| `libHexaBackwardRepairDsp.so` | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` | yes |
| `libHexaBackwardRepairHost.so` | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` | yes |

Completed-role raw evidence re-verified:
`preserved-completed-roles/standalone_softmax.bin` is 284,827,248 bytes with
SHA256 `b6465b0737696d3a2b5f342a772a059ccab5d75e357bac1a054ad918d33d3b8b`,
identical to the recorded remote-before, remote-after and local hashes. The
partial dValue raw is 0 bytes with the empty-file hash. Nothing was mutated.

The frozen binary was compiled from source `bb6aaceb42de505b29689a362ce8293687e7ab89`.
`git diff bb6aaceb..30d0cb3` over `host_tests/fp32_custom` shows **no change**
to `probe_support.h`, `double_qk_initialize.h`, `saved_p_backward_graph.h`,
`backward_repair_graph.h`, `backward_repair_probe.cpp` or
`saved_p_backward_probe.cpp`. The source analysed below is therefore the source
that produced the failing binary.

---

## B. Phase A — EXIT139_EVIDENCE_TIMELINE

Time base: the probe's own `api_us` instrumentation, which records the
wall-clock duration of each QNN call and is re-zeroed at each `cohort_begin`.
Absolute wall-clock times are only available for file mtimes and for the total
process duration, so the sequence below is **relative unless stated**.

### B.1 Role `standalone_softmax` (completed, exit 0)

| Relative (µs) | Event | Source |
| ---: | --- | --- |
| 0.2 | `getProviders` returned 0 | stdout |
| 1,034.0 | `logCreate` returned 0 — stderr `QnnLog_create` → **`Initializing HtpProvider`** (first and only occurrence) | stdout + stderr 3 |
| 2,075.2 | `backendCreate` returned 0, backend `0x6da91ed0` | stdout + stderr 11–13 |
| 61,774.5 | 6 forward op-package registrations returned 0 | stdout |
| 174,955.8 | `deviceCreate` returned 0 (113,181.4 µs) | stdout |
| 191,839.0 | `contextCreate` returned 0 (16,883.1 µs) | stdout |
| 194,909.2 | `registerRepairDSP` returned 0 — last setup call | stdout |
| 262,191.7 | `graphCreate` #1 (registry id 256) | stdout + stderr 97–103 |
| 283,861.7 | `graphFinalize` #1 returned 0 (21,304.0 µs) | stdout |
| 4,298,199.9 | `graphFinalize` #2 returned 0 (35,932.1 µs) | stdout |
| 8,180,684.0 | `graphFinalize` #3 returned 0 (43,356.0 µs) | stdout |
| 12,159,008.3 | `graphFinalize` #4 returned 0 (48,954.2 µs) | stdout |
| …→15,738,710.2 | 4,912 × `graphExecute` returned 0 | stdout |
| 15,824,607.6 | `contextFree` returned 0 (85,897.4 µs) → `Number of existing contexts: 0, graphs: 0` | stdout + stderr 30,589–30,611 |
| 15,857,356.8 | `deviceFree` returned 0 (32,749.1 µs) → `Exiting thread` / `exits with 0` | stdout + stderr 30,613–30,619 |
| 15,866,627.9 | `backendFree` returned 0 (9,271.2 µs) → 5 packages unloaded, `Closing environment: 0xb4000078b0e2d850`, **`PrepareLibLoader Unloading libQnnHtpPrepare.so`** | stdout + stderr 30,621–30,635 |
| +0 | `summary,backward_repair,mode=standalone_softmax,pairs=9824,...,teardown=1` then `cohort_end,mode=standalone_softmax,exit=0` | stdout 24,716–24,717 |

Measured absolute: stdout/stderr files last written 2026-10-20 19:41:00;
identity record written 19:41:04; total probe duration 20.500856 s
(`execution-before-readback-private.json`).

### B.2 Role `standalone_dvalue` (crashed)

| Relative (µs) | Event | Source |
| ---: | --- | --- |
| 0.6 | `getProviders` returned 0 | stdout |
| 111.3 | `logCreate` returned 0 — stderr shows **no `Initializing HtpProvider`** | stdout + stderr 30,637–30,639 |
| 197.6 | `backendCreate` returned 0, backend `0x6da91ed0` → `PrepareLibLoader Loading libQnnHtpPrepare.so`, `First connection to QNN HTP Prepare library established!`, `Creating environment …`, **`New env: 0xb4000078b0e2d850`** (same address as role 1) | stdout + stderr 30,641–30,655 |
| 4,177.8 | 6 forward op-package registrations returned 0 | stdout |
| 89,515.4 | `deviceCreate` returned 0 (85,337.6 µs) | stdout |
| 106,627.6 | `contextCreate` returned 0 (17,112.2 µs) | stdout |
| 109,598.9 | `registerRepairDSP` returned 0 — **last returning QNN API** | stdout |
| 171,692.4 | `graphCreate` (registry id 260, DSP handle 0x1) returned 0 | stdout + stderr 30,725–30,733 |
| 171,792.1 | 5 × `tensorCreate` returned 0 | stdout |
| 171,853.8 | `graphAddNode` #1 returned 0 | stdout |
| 171,894.3 | 3 × `tensorCreate` returned 0 | stdout |
| 171,919.6 | `graphAddNode` #2 returned 0 — **last returning API of the whole process** | stdout |
| 171,919.6 | `QnnGraph_finalize started, graph = 0x1` — the call is **entered** | stderr 30,791 |
| — | `os_linux.cc:55`, `prepare_options.cc:29/342`, `graph_prepare.cc:1213`, `vtcm_wrapper.cc:135/160/180/268/278` | stderr 30,793–30,809 |
| — | `Segmentation fault` — the call never returned | stderr 30,811 |

Remaining roles `native_softmax`, `native_dvalue`, `dual` never started. Whole
process teardown incomplete.

### B.3 Correction to the recorded "last API" description

The previously recorded statement — "dValue graphFinalize / graphExecute = 0,
last successful API = graphAddNode / code 0" — is accurate for the *returned*
API log, but it must not be read as "the fault happened after `graphAddNode`
returned". The stderr shows `QnnGraph_finalize started`, so `graphFinalize`
**was invoked** and the fault occurred **inside** it. The stdout API log only
records calls that returned, which is why no `api,graphFinalize,<rc>` line
exists.

### B.4 Faulting PC, faulting address, stack and thread

**Not recoverable from existing evidence.** Verified by searching the complete
private evidence tree (333 files) for `tombstone`, `TOMBSTONE`, `signal 11`,
`SIGSEGV`, `backtrace`, `#00 pc`, `libc DEBUG`, `Abort message`,
`fault address` and `si_addr`: zero hits. The runner
`scripts/run_backward_repair_trace_probe.ps1` captures stdout, stderr, the
exit code, the monitoring samples and the raw evidence; it never runs
`logcat`, never reads `/data/tombstones` and never attaches a debugger.
Therefore:

- faulting PC: unavailable
- faulting address: unavailable
- signal detail beyond "Segmentation fault" / exit 139: unavailable
- unwound stack: unavailable
- thread name: unavailable

The tid printed in `vtcm_wrapper.cc:268/278` (`0x7fd090d970`) identifies the
thread that was executing `QnnGraph_finalize`. The corresponding tid in the
successful standalone dValue pilot (`0x7fee72c770`) is of the same kind
(stack-address pthread handle), so **thread identity is not a differentiator**
between the passing and the failing run.

**The faulting region is nonetheless bounded by the log.** In every successful
finalize — the four role-1 graphs in the crash run and the four graphs of both
successful pilots — the log line immediately after
`vtcm_wrapper.cc:278 "Acquired Cache"` is
`graph_prepare.cc:1777 "do_prepare2 with num_nssp = 1"`. In the failing
finalize that line never appears. The fault is therefore inside the
host-side QNN HTP graph-prepare library, in the code region between
`vtcm_wrapper.cc:278` and `graph_prepare.cc:1777`.

---

## C. Phase B — Multi-role lifecycle and ownership

### C.1 What the frozen Host does per cohort role

`repairWorker` is invoked five times inside one OS process by the cohort
`main`. Each invocation performs, in order:

1. `initialize()` → `initializeDoubleQk()`:
   `logCreate` (new handle) → `backendCreate` (new backend) →
   6 × `backendRegisterOpPackage` → `deviceCreate` (new device) →
   `contextCreate` (new context); then 2 × `backendRegisterOpPackage` for the
   repair packages.
2. Four `repairprobe::Graph` objects (observer × tap ∈ {1,0}²). Each one
   calls `graphCreate` (new graph in the current context),
   `tensorCreateGraphTensor` ×N, `graphAddNode` ×2, `graphFinalize`, then
   1,228 × `graphExecute` with canary / poison / input-immutability checks.
3. `closeProbe()`: `contextFree` → `deviceFree` → `backendFree`.

Per-role Host API calls measured by the host-only audit: **17 lifecycle calls**
plus per-graph construction, identical for all five roles.

### C.2 Ownership facts established from the frozen source and the QAIRT headers

| Object | Owner | Released by | Verdict |
| --- | --- | --- | --- |
| `Qnn_LogHandle_t` | `initializeDoubleQk` local | **never** — no `logDelete` | real leak, 5 per cohort process |
| `Qnn_BackendHandle_t` | global `backend` | `backendFree` | correct |
| `Qnn_DeviceHandle_t` | global `device` | `deviceFree` | correct |
| `Qnn_ContextHandle_t` | global `context` | `contextFree` | correct |
| `Qnn_GraphHandle_t` | `DiagnosticGraph::graph` | **no destructor, no `graphFree` call** — but QNN 2.37 exposes **no** graph-free API; `Qnn_Context_free` is documented to "free the context and all associated graphs, operations & tensors" | not a defect |
| `Qnn_Tensor_t.v1.name` | points into `DiagnosticGraph::names` | — | `names.reserve(512)` in the ctor, so the pointer never dangles |
| `Qnn_Tensor_t.v1.dimensions` | points into `DiagnosticGraph::shapes` | — | `shapes.reserve(512)`, so the pointer never dangles |
| `Qnn_OpConfig_t.v1.name` | points into a **destroyed temporary** (`(n+"dvalue").c_str()`) | — | safe only because QNN copies the op config during the call; latent risk, not the differentiator (role 1 produced correct results with the identical pattern) |

### C.3 The cross-role state that the Host does not control

The Host hands **no** role-1 pointer, handle or descriptor into role 2. Every
QNN handle is re-created per role. The only state that survives the role
boundary lives inside the QNN libraries, and the device log shows it is
**not symmetric**:

- `libQnnHtp.so` is `dlopen`ed with `RTLD_GLOBAL` and is **never `dlclose`d**.
  Its process globals therefore survive every backend teardown.
- `libQnnHtpPrepare.so` is **not** a `NEEDED` dependency of `libQnnHtp.so`
  (verified from the ELF dynamic section: `libQnnHtp.so` needs only
  `libc`, `libm`, `libdl`, `liblog`). It is loaded and unloaded at runtime by
  the backend's `PrepareLibLoader`.
- The HTP provider global is initialised **once per process**:
  `Initializing HtpProvider` appears at role 1's `QnnLog_create` and is
  **absent** at role 2's `QnnLog_create`.
- The hexnn environment is destroyed at `backendFree`
  (`Closing environment: 0xb4000078b0e2d850`) and recreated at the next
  `backendCreate` **at the identical address**
  (`New env: 0xb4000078b0e2d850`).

So the cohort process unloads and reloads the graph-prepare library four times
while a provider-level global in a library that is never unloaded survives
those cycles.

### C.4 Host-only measurement of that structure

`host_tests/fp32_custom/backward_repair_cohort_lifecycle_test.cpp` drives the
**frozen** `DiagnosticGraph` and `repairprobe::Graph` against an instrumented
mock QNN API that models the documented QNN ownership contract plus the HTP
process-global behaviour observed above.

| Property | Certified cohort plan (current) | Single-session variant | Backend-shared variant (candidate fix) |
| --- | ---: | ---: | ---: |
| `backendCreate` / `backendFree` | 5 / 5 | 1 / 1 | 1 / 1 |
| `deviceCreate` / `deviceFree` | 5 / 5 | 1 / 1 | 1 / 1 |
| `contextCreate` / `contextFree` | 5 / 5 | 1 / 1 | 5 / 5 |
| graphs created | 20 | 20 | 20 |
| HTP provider initialisations | 1 | 1 | 1 |
| prepare-library load / unload | 5 / 5 | 1 / 1 | 1 / 1 |
| **prepare-library reload cycles** | **4** | **0** | **0** |

---

## D. Phase C — Host-only reproduction and regression

### D.1 What the host-only audit proves

`scripts/test_backward_repair_cohort_lifecycle.ps1` →
`build/reports/g1-backward-repair-exit139-diagnosis/cohort_lifecycle_audit.txt`

```
cohort_lifecycle_audit=PASS
roles=5 role_lifecycle_symmetric=1 lifecycle_calls_per_role=17
role=standalone_softmax api_calls=76
role=standalone_dvalue api_calls=63
role=native_softmax api_calls=247
role=native_dvalue api_calls=235
role=dual api_calls=275
graphs_per_role=4 graphs_total=20 contexts_created=5
per_role_lifetimes: backend_create=5 device_create=5 context_create=5 backend_free=5 device_free=5 context_free=5
htp_provider_initializations=1 prepare_library_loads=5 prepare_library_unloads=5 prepare_library_reload_cycles=4 env_created=5 env_destroyed=5
log_handles_created=5 log_handles_destroyed=0
descriptor_faults=0 handle_faults=0
single_session_variant: backend_create=1 device_create=1 context_create=1 contexts=1 graphs=20 prepare_library_loads=1 prepare_library_unloads=1
backend_shared_variant: backend_create=1 device_create=1 context_create=5 contexts=5 graphs=20 prepare_library_loads=1 prepare_library_unloads=1
cohort_lifecycle_audit_rc=0 HOST_ONLY=1 NO_DEVICE=1 NO_DSP=1 NOT_FORMAL_EVIDENCE=1
```

1. **Role symmetry.** All five roles issue an identical QNN lifetime
   pattern (17 lifecycle calls: logCreate, backendCreate, 8 registrations,
   deviceCreate, contextCreate, then contextFree, deviceFree, backendFree).
   The frozen Host cannot distinguish role 1 from role 2.
2. **No dangling descriptor.** The audit re-reads every tensor's `name` and
   `dimensions` pointers at `graphFinalize` and compares them against the
   values captured at `tensorCreateGraphTensor`:
   `descriptor_faults = 0` over all 20 graphs.
3. **No stale handle.** The audit detects use-after-free, double-free and
   use of a handle whose context was already freed:
   `handle_faults = 0`.
4. **No cross-role pointer flow.** The audit proves structurally that the Host
   passes no role-1 object into role 2 — the only cross-role state is the
   QNN library's own process-global state.
5. **The exposure is quantified.** The current plan forces 4 prepare-library
   reload cycles per cohort process; both candidate variants force 0, while
   building the same 20 graphs in the same role order.

### D.2 What the host-only audit cannot do

The fault is inside `libQnnHtpPrepare.so` (host-side QNN graph prepare). That
library only runs on the Android/aarch64 target with the QNN HTP backend. A
host-only harness **cannot reproduce the crash** and no such reproduction was
claimed. The audit establishes the Host-side invariants and the process-level
exposure; it does not establish the faulting instruction.

Sanitizers were not available in this environment (MSYS2 UCRT64 g++ 15.2.0;
no clang), so no ASan/UBSan run was performed. The descriptor and handle
checks above are deterministic equivalents for the specific lifetimes in
question and are not a substitute for a sanitizer build.

---

## E. Phase D — Root cause classification

### E.1 Candidate matrix

| ID | Hypothesis | Supporting evidence | Refuting evidence | Still unverified | Next minimal step |
| --- | --- | --- | --- | --- | --- |
| H1 | Use-after-free of role-1 memory by role 2 | role 1 frees the backend, env and prepare library; role 2 reuses the same env address | the Host passes no role-1 pointer or handle into role 2 (host audit, `handle_faults=0`); every QNN handle is re-created | whether the *library's* provider global holds a pointer into the destroyed env | §F diagnostic run |
| H2 | Double-free / teardown corruption | role 1's `contextFree`/`deviceFree`/`backendFree` release 4 graphs, 1 device, 1 backend and the env | all three frees returned 0; no Host double free; the same teardown is performed in the successful standalone pilots | whether the library double-frees something internally on the second cycle | §F diagnostic run |
| H3 | Dangling tensor descriptor | `op.v1.name` points into a destroyed temporary | host audit `descriptor_faults=0`; `names`/`shapes` are `reserve(512)`; QNN copies op config and tensor data during the call; role 1 produced correct results with the identical pattern | whether a non-copying backend path exists for the failing call | none needed |
| H4 | Context / graph handle reuse inconsistency | the DSP-side graph handle restarts at `0x1` while the registry id continues 256→260 | the Host creates a fresh backend/device/context per role; `Number of existing contexts: 1, graphs: 0` before the failing graph | — | none needed |
| H5 | Global state leakage between roles | the HTP provider global is initialised once and never re-initialised, while the prepare library is unloaded and reloaded four times | the leakage is inside the QNN library, not the Host | whether the surviving provider global is what the failing finalize dereferences | §F diagnostic run |
| H6 | Memory exhaustion / fragmentation | 284 MB of role-1 raw output; 4 graphs accumulate per role | role-2 setup was **faster** than role-1 (`registerOldHost` 58,634→3,388 µs, `deviceCreate` 113,181→85,338 µs); the raw is a file, not resident heap; the failing finalize reached a successful 8 MB VTCM allocation | whether the *DSP-side* pool was short at finalize | §F diagnostic run |
| H7 | Invalid memory access specific to graph construction | the fault is inside `graphFinalize`, after `graphAddNode` returned 0 | the failing graph (8 tensors, 2 nodes, obs=1/tap=1) is structurally identical to the graph that finalized successfully in the standalone dValue pilot | the exact instruction | §F diagnostic run |
| H8 | QNN/FastRPC library internal state across backend destroy/recreate in one process | `libQnnHtpPrepare.so` is loaded/unloaded at runtime and is not a `NEEDED` dependency; the provider global survives while the env is destroyed and recreated at the same address; the Host is role-symmetric and defect-free; the fault is inside the library's finalize path | none found | the faulting instruction and the causal link from the reload to the fault | §F diagnostic run |

### E.2 Classification

**`ROOT_CAUSE_STRONGLY_SUPPORTED`** — *the QNN HTP backend's
backend-destroy / backend-recreate cycle inside one OS process leaves the
second backend's graph-prepare path in an invalid state, and the second role's
`QnnGraph_finalize` dereferences it.*

Independent evidence lines that agree:

1. The fault is inside the QNN HTP host-side graph-prepare library, not in
   the probe, not in the DSP skel and not in the custom op package
   (stderr, bounded region).
2. The frozen Host is **role-symmetric** and free of dangling-descriptor and
   stale-handle defects over the whole 5-role sequence (host-only audit).
3. The Host hands no cross-role pointer into role 2 (host-only audit plus
   source reading).
4. The device log shows a concrete, non-symmetric process-global state change
   at exactly the role boundary: provider global initialised once, prepare
   library unloaded and reloaded, hexnn environment destroyed and recreated at
   the same address.
5. The certified cohort plan itself mandates this pattern — "Each role
   creates/tears down its own QNN backend/device/context" — so the plan's
   `backend_lifetime` premise, not the arithmetic, is what is being tested
   for the first time.

Why not `ROOT_CAUSE_VERIFIED`: there is no stack trace, no reproduction and no
faulting PC, so the causal link from the reload to the fault remains
inferred.

Why not `ROOT_CAUSE_HYPOTHESIS`: the localisation, the Host-side exclusion
and the process-global asymmetry are each established by direct evidence, not
by plausibility alone.

### E.3 Separate events

The historical API6031 episode, the DMA6033 token and the top-resumed
deficiencies remain **separate, unresolved** events. No 6031 or 6033 token
occurs anywhere in the captured stdout/stderr of the exit139 run. The exit139
is **not** attributed to 6031/6033, and 6031/6033 are **not** exonerated by
this episode. The QAIRT 2.48.40 release notes contain an unrelated entry
("Fixed a race condition in the HexNNv2 backend performance driver that could
cause a SIGSEGV during SNPE instance teardown … {163626}"); that concerns the
SNPE API and is recorded only as a non-causal prior, not as support for this
diagnosis.

---

## F. Candidate fix and its scope

### F.1 Minimal change

Keep **one QNN backend and one device for the entire cohort OS process** and
create one context per role, so that:

- the per-role graphs are still released at the end of each role by
  `contextFree` (same peak graph residency as today: 4 live graphs),
- `libQnnHtpPrepare.so` is loaded once and never unloaded, and the hexnn
  environment and the HTP provider global survive intact for all five roles,
- the prepare-library reload cycles drop from 4 to 0.

Host-only verified (`backend_shared_variant`): 1 backend, 1 device, 5
contexts, 20 graphs, 1 prepare-library load, 1 unload, 0 reload cycles, with
the same five roles in the same order and the same per-role graph
construction.

A wider variant (one context for the whole process) also removes the reload
cycles but keeps 20 graphs live at once. It is **not** recommended: role-1
evidence shows `graphFinalize` cost growing with context residency
(21.3 → 35.9 → 43.4 → 49.0 ms for graphs 1→4), and 20 resident graphs would
raise device memory pressure for no benefit.

### F.2 Why the fix does not touch the mathematics or the formal contract

- Arithmetic, oracle, fixtures, numeric thresholds, observer definition,
  tap semantics, coverage (Core 2,200 / Holdout 256 per observer), role
  order, role set, raw evidence format and the digest definition are all
  untouched.
- The change is confined to *when* the QNN backend and device are created
  and destroyed inside one OS process.
- The formal requirement of 5 roles × 3 independent OS processes is
  unchanged; the fix alters the QNN lifetime *inside* one process.

### F.3 New candidate binary is required

`host_tests/fp32_custom/backward_repair_probe.cpp` is hash-verified inside
the frozen bundle (`build-identity.json` → `source_hashes`). Any change to
the Host lifecycle therefore requires a **new candidate bundle**: new source
hash, new `build-identity.json`, new `probe_sha256`, and a new approval
gate. The frozen bundle must not be edited in place.

The certified plan `host_tests/fp32_custom/backward_repair_cohort_plan.json`
must also be amended explicitly: its `backend_lifetime` clause ("Each role
creates/tears down its own QNN backend/device/context") is the clause that
this diagnosis invalidates. **The plan is not rewritten silently here**; the
amendment is a separate, approved act.

### F.4 Host-only regression test for the fix

`scripts/test_backward_repair_cohort_lifecycle.ps1` is that regression test.
It must be extended, in the new candidate bundle, to assert the candidate's
invariants on the candidate binary: `backendCreate == 1`, `deviceCreate == 1`,
`contextCreate == 5`, `contextFree == 5`, `backendFree == 1`,
`deviceFree == 1`, prepare-library reload cycles `== 0`, graphs `== 20`, and
the per-role API ordering unchanged.

### F.5 Minimum device confirmation after a fix

Before any formal cohort attempt, the new candidate binary must complete, in
one OS process and on the same device, the full five-role sequence with
`exit 0` and complete teardown, under the unchanged preflight and monitoring
rules. That single run is the device confirmation; it is **not** a formal
process and must not be counted as one.

---

## G. Minimum device reproduction plan (designed, not executed)

Not authorised here. No device was contacted for this work.

**One diagnostic case**, chosen because it separates the teardown variable
from every other candidate:

> In one OS process, run `standalone_softmax` to completion, then run
> `standalone_dvalue` **without** freeing the backend and device between the
> two roles, with crash-dump capture enabled.

| Item | Requirement |
| --- | --- |
| Binary | a **new diagnostic-only** Host probe; the frozen `saved_p_backward_probe` bytes are not reused as the executable. The same frozen `libHexaBackwardRepairHost.so` / `libHexaBackwardRepairDsp.so` bytes are used so the numerics stay comparable |
| Identity | the diagnostic binary's SHA256 must differ from the frozen probe's and must be recorded as a diagnostic artifact |
| Evidence directory | a new exclusive directory under `build/reports/`, separate from `g1-backward-repair-native-completion/` |
| Device identity | the same physical device identity must be verified and recorded privately |
| Crash capture | logcat / tombstone / dropbox capture around the run is **mandatory**, because the 2026-10-20 run produced no faulting PC. This is the single most important addition |
| Preflight | fresh Awake + top-resumed + safety preflight, unchanged semantics |
| Monitoring | unchanged stop conditions, unchanged 2 s foreground monitoring, no wake/unlock/force-stop/foreground takeover |
| Retries | **none**. One attempt. Stop on the first failure |
| Formal counting | **not counted** toward the 5 × 3 formal cohort |
| Interpretation | PASS ⇒ the backend destroy/recreate cycle is necessary for the crash (H8a confirmed, H6/H1-work largely excluded). FAIL ⇒ the teardown cycle is not the trigger; the captured stack then discriminates H6/H7/QNN-internal |

Simply re-running the same cohort three times is explicitly excluded.

---

## H. Formal cohort resumption: **NO-GO**

The formal cohort may be resumed only when **all** of the following hold.
None of them currently holds.

| # | Condition | Current state |
| --- | --- | --- |
| 1 | The exit139 cause is sufficiently identified and suppressed | cause localised and strongly supported, but **not** verified (no stack, no reproduction); no suppression exists |
| 2 | Host-only regression of the fix holds | the *audit* holds; the *fix* does not exist yet in any binary |
| 3 | Binary / compiled source / fixture provenance re-confirmed | frozen bytes re-verified unchanged; a new candidate bundle would need its own manifest |
| 4 | Any minimum device diagnostic is separately approved | **not approved**; §G is a design only |
| 5 | A new safety preflight passes | the last preflight passed, but it is not a preflight for a future candidate |
| 6 | Role order, coverage, digest and observer contract unchanged | unchanged; the candidate fix does not touch them |
| 7 | Complete process exit and teardown obtainable | **not yet**; the crashed process never tore down |
| 8 | Failed processes excluded from formal counts | satisfied; this exit139 process is `FAILED_PARTIAL` and formally ineligible |

The formal cohort still requires **5 roles × 3 independent OS processes**. The
existing single-role pilots do not substitute for it.

---

## I. Unchanged status

| Field | Value |
| --- | --- |
| Formal status | `BLOCKED_FORMAL_COHORT` |
| `FORMAL_COMPLETED` | `0/3` |
| `EXIT139_ROOT_CAUSE` | `STRONGLY_SUPPORTED` (was `UNRESOLVED` — raised on evidence, not on a formal certification) |
| `TRAINING_READY` | `NO` |
| 40k→60k / 200k | `NO-GO` |
| API6031 | `UNRESOLVED / DORMANT / WATCH`, unchanged and unrelated |
| DMA6033 | not observed, unchanged |
| Project Formal | `BLOCKED_SCOPE / NOT_EXECUTED` |
