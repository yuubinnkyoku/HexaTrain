# G1 exit139 — shared Backend/Device lifetime diagnostic

Latest result, **2026-10-23**: **SHARED_LIFETIME_BOUNDARY_PILOT_PASS**.
One diagnostic OS process completed Softmax 9,824 pairs and the registered dValue
200 pairs, including the formerly crashing first dValue finalize, with actual
exit0. Independent double/FP32 oracles, word contracts, observed resource counts
and teardown passed. The hypothesis is strengthened; root cause is not proven.
The resumed evidence and current answers follow the preserved initial record below.
Formal cohort remains **0/3**, higher-layer certification absent, training **NO-GO**.

## Initial preparation decision (preserved)

**BLOCKED_DEVICE_SAFETY: ADB_NO_ONLINE_DEVICE.** The independent candidate and
Host gates are complete. The fresh read-only preflight could not resolve an
online physical device. **Zero diagnostic ARM processes and zero DSP executes**
were launched. There was no staging, crash-collection session, wake, reconnect,
retry, foreground takeover, APK installation, performance measurement or training.
This is not a negative result for the intervention: its device effect is unmeasured.

`BLOCKED_FORMAL_COHORT = 0/3` remains. Full Attention, Full1-layer and Full19-layer
are not certified. 40k→60k and 200k training remain **NO-GO**.

## A. Provenance and preservation

Work started in the designated G1 research clone, with its own `.git` common
directory, a clean working tree, no upstream, and full HEAD
`6c4058af820151f89996065380b12778fbd965bc`. Its original branch remains
`investigation/g1-backward-repair-native-completion`. The new branch
`investigation/g1-backward-shared-lifetime-diagnostic` was created from that exact
commit. Neither existing follow-up work nor conflicting changes existed there.

Both `30d0cb3bc5eb9e632ac869a122438b3fab64ae06` and
`fad9d19a669032aa6d3e29651eb547920ca76506` are ancestors of the starting HEAD.
All pre-existing local and remote refs were recorded privately and rechecked;
the original branch and every protected ref retain their original targets.
No reset, rebase, amend, history rewrite, push, PR or merge was used.

The original compiled source remains
`bb6aaceb42de505b29689a362ce8293687e7ab89`. All five original bundle manifests,
their declared sources/binaries, protected artifacts, the prior 333-file evidence
ledger and 288 pre-existing files in the native-completion/exit139 evidence roots
were rehashed unchanged. These sets overlap; their counts must not be added as
independent evidence counts. Old raw/tombstones and five-role pilots were retained.
The protected 40k checkpoint retains SHA256
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
No checkpoint was decoded or written. Dataset, tokenizer, cache/order, production
optimizer/Muon ABI and checkpoint codec were not changed.

| Artifact | SHA256 |
| --- | --- |
| Original and copied repair DSP | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` |
| Original and copied repair Host | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` |
| Original ARM probe, retained | `23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05` |
| New diagnostic ARM probe | `2ba5640f5842f9477ad165bae4d49e3a59d19135b60751ff101de508f46ce94d` |
| Exact original-to-diagnostic source diff | `a951c384fd6887ad410ebe10f811d7bde7568f296cf82b4d18040a22885644f4` |

The independent plan and candidate sources were committed in
`658df776aff26b2297e9d4c8703e6978878d6b7d`, before building the ARM candidate and
before any possible execution. The separate build manifest identifies that
compilation commit, hashes all candidate/frozen dependencies and the relevant SDK
headers, and records the exact diff, compiler identities, flags and diagnostic-only
classification. Subsequent reporting commits do not change its compiled identity.
The old compiled bundle and its manifests were not edited. DSP/Host were copied,
not rebuilt; there is no new DSP kernel.

## B. API and lifetime audit

The reused original lifecycle audit freshly confirmed five Backend creates,
five Device creates, five Context creates and 20 graphs in the original five-role
structure. Its modeled Prepare reload count is four. That model represents the
historical load/unload observations; it is not a new observation inside QNN.

The candidate has one process owner for Backend/Device and two sequential role
Contexts. Graph construction is unchanged; each role still builds four observer
graphs, which are released with that role's Context. No tensor, graph, descriptor,
Context handle or DSP tensor is passed to the next role. Backend packages are
registered once for their owning Backend and reused with it, avoiding duplicate
registration against the same Backend. The original first-role ordering is
preserved: eight Forward packages before Device/Context creation, two repair
packages after first Context creation.

The fixed SDK header contract was read directly, not inferred from Host mocks:

- `QnnContext.h`, lines 678–710: Context creation accepts Backend and Device
  handles and owns graphs, operations and tensors.
- `QnnContext.h`, lines 1510–1531: Context free releases its associated graphs,
  operations and tensors; it does not free the supplied Backend/Device.
- `QnnDevice.h`, lines 439–457: Device free rejects a Device associated with an
  unfreed Context. Contexts must therefore be freed before Device free.
- `QnnBackend.h`, lines 404–423: Backend free releases Backend resources.
- `QnnBackend.h`, lines 193–198, and `QnnLog.h`, lines 150–164: Backend does not
  own the Logger. Its handle must remain valid; Logger release is a separate API.

Sequential Context creation/free under retained valid Backend/Device handles is
consistent with this API contract. There is no requirement here to destroy them
after each Context. This establishes API-level legality, not freedom from an HTP
implementation defect. The Logger callback and its handle remain valid through
Device/Backend cleanup. The original Logger leak is deliberately retained; no
`logFree` or unrelated Logger repair was mixed into this comparison. Sharing does
not require freeing or replacing that Logger between roles.

Ownership code rejects repeated initialization/cleanup and overlapping role
Contexts, clears successfully freed handles, frees only successfully created
resources, and uses Context→Device→Backend dependency order. If a free fails,
ownership is ambiguous: it is not retried, dependent parents are retained, and
cleanup fails. An `atexit` hook covers the frozen worker's existing `std::exit`
API-failure paths; exceptions return to the process cleanup path. SIGSEGV cannot
be assumed to run cleanup. Graph/descriptor storage still uses the frozen owner.

### Exact scope of the source change

The new ARM source is independent of `backward_repair_probe.cpp`. Its worker,
oracles, record layout, arithmetic, graph usage, observer loops and gates are a
verbatim copy, with only `initialize()` rebound to the shared role Context owner.
An automated exact source comparison verifies this restriction. The new process
entry point restricts the roles, selects the registered limits, traces finalize
entry/actual return and owns whole-process cleanup. The shared lifetime header
contains the ownership changes. The existing production builder, graph headers,
Custom Op ABI, Forward P producer, DSP arithmetic and frozen fixtures are unchanged.

Softmax remains `P × (dP − rowSum(P × dP))`, gate 0.002. dValue remains
`Pᵀ × dContext`, gate 0.01. Compiler flags retain ordered FP32, disabled FMA
contraction and no fast-math. No rescale, hidden double DSP arithmetic or optimizer
change was introduced. The independent Host double oracle is a reference, not DSP
arithmetic. HTP internal precision is not asserted to be FP32.

## C. Fresh Host/build evidence

The dedicated Host test reuses the original lifecycle/descriptor audit and the
same new ownership class compiled into the ARM candidate. It passes normal
Softmax→dValue and injected Backend create, Device create, both Context create,
both finalize, both teardown, dValue execute, Forward package registration and
repair package registration failures. Early exit, repeated cleanup rejection,
stale graph/role handle rejection and Logger validity through Backend free are
also asserted. Descriptor/handle validity and legal dependency order are checked
by mocks. A teardown failure deliberately leaves its uncertain Context and parents
for process termination; it is not mislabeled complete cleanup.

| Resource | Host normal case | Device measurement |
| --- | ---: | --- |
| OS process | simulated two-role session | 0 launched |
| Backend create/free | 1 / 1 | null / null |
| Device create/free | 1 / 1 | null / null |
| Context create/free | 2 / 2 | null / null |
| Graphs | 8 | null |
| Inter-role Backend/Device teardown/recreate | 0 | null |
| Inter-role Prepare reload | 0 in lifetime model | null |

Fresh build gates also passed repair arithmetic contract, rejection of unsupported
prepare, repair graph shape/ownership, Saved-P reference, independent oracle and
Saved-P graph checks. Fixed SDK ARM compilation passed. SDK identities are QAIRT
`2.48.40.260702151143`, QNN API `2.37.0`, Hexagon SDK `6.6.0.0`, Tools `19.0.07`,
HTP V81, NDK `26.2.11394342`, UCRT64 g++ `15.2.0`. Compiler hashes were checked.

Fresh Qnn profile: **PASS**, 442.3 seconds, 4 PASS / 0 FAIL / 0 SKIP, including
Host contract+diagnostic suites, QNN-enabled Android build and APK audit. Missing
optional SDK samples remained advisory; core and Build ID passed. The approved
APK was not replaced or installed. The standalone Host profile was NOT_EXECUTED;
its historical result is not promoted to a new profile invocation. The actual
fresh Host tests above and those inside Qnn are recorded separately.

Initial PS7 Fast: **PASS**, 89.5 seconds, outer 1 PASS / 0 FAIL / 0 SKIP; inner
10 PASS / 0 FAIL / 21 SKIP. Final PS7 closeout Fast also **PASS**: 83.0 seconds,
outer 1 PASS / 0 FAIL / 0 SKIP; inner 10 PASS / 0 FAIL / 21 SKIP. The independent
allow-list exporter passes 26 rejection cases. No Project Formal, generic Device
smoke, W8 regression or performance suite was run.

Host success does not prove that the real QNN library cannot crash.

## D. Registered device diagnostic and actual stop

The committed plan requires exactly one possible process, no retry, and
`standalone_softmax → standalone_dvalue` only. Softmax must finish all four
observers × (2,200 Core + 256 Holdout) = 9,824 pairs / 4,912 executes, with the
same word/numerical contracts. Failure stops before dValue.

The bounded dValue selection is the unchanged worker with `ForwardLimit=1`:
Forward index 0, frozen Core gradient indices 0–24, heads 0–1, all four observers.
Its maximum is 100 executes / 200 pairs, four graphs. The first dValue finalize
entry and actual return are separate observations. Full dValue 9,824-pair coverage,
third role, formal cohort, performance and training are prohibited by this entry
point. No old formal plan clause, coverage, digest or three-process rule was edited.

The fresh existing read-only preflight passed Host analysis, fixed SDK/APK Host
checks, storage, compiler/package and protected-artifact checks, then stopped at
**ADB_NO_ONLINE_DEVICE**. Same physical device, active-run absence, Awake,
top-resumed, installed approved APK, storage and critical protection on the device
could not be established. There are no fresh device safety or competing-workload
claims. No endpoint search/reconnect, automatic wake or safety waiver followed.
The blocked preflight's Host wrapper remained waiting after its report was saved;
only that owned wrapper was closed after checking its identity and absence of
child work. The shared ADB server and all device processes were untouched. The
saved preflight decision, not that wrapper's termination, is the safety evidence.

The new runner preserves the existing preflight, approved-APK staging/hash checks,
two-second top-resumed monitor, critical protection and owned-PID cancellation.
Its independent candidate directory/staging path does not overwrite the original
bundle or approved APK. It adds an exclusive mission-level one-process claim,
read-only crash collection and process snapshots. Its device path was NOT_EXECUTED
and is not claimed verified end-to-end. Before any later launch, a new fresh safety
preflight and a complete review of the unexecuted runner remain necessary.

## E. Crash evidence

The original two independent episodes remain `EXIT139_REPRODUCED` and
`FAULT_SITE_MATCHED`: SIGSEGV / SEGV_MAPERR inside first dValue finalize, starting
at `libQnnHtpPrepare.so + 0x3007bac`, with matching 17-frame main-thread stacks.
Both completed Softmax and its role teardown, then recreated Backend/Device/
Context. Last returned API was `graphAddNode / 0`; last visible Prepare stage was
successful 8MiB VTCM cache acquisition. Probe stdout/stderr contained no 6031/6033.
The second episode's 46 process snapshots found no recognizable separate NPU
experiment; complete absence of all possible clients was not proven.

For the new candidate, logcat, QNN runtime log, tombstone, DropBox metadata,
signal/fault address, relative offset/stack, finalize return, process exit and
cleanup are all **null / NOT_EXECUTED**. Crash-collection preparation was not
attempted against an unidentified device. The runner's registered read-only
collection must establish usable logcat before any launch; unavailable optional
crash artifacts remain null. No log clear, root or SELinux change is permitted.
Old raw evidence was not overwritten. API6031/6033 are separate unresolved watches,
not inferred absent from a process that never started.

## F. Scientific interpretation

`BACKEND_REINITIALIZATION_HYPOTHESIS_SUPPORTED` describes the original evidence;
`ROOT_CAUSE_NOT_PROVEN` remains. The shared-lifetime intervention has no device
observation yet. It neither strengthens nor refutes the hypothesis. There is no
new exit139 reproduction and no evidence that the candidate avoids it.

Even one future successful boundary pilot would only strengthen the hypothesis,
not establish a unique root cause or formal closure. Context/Graph cleanup,
process-global provider/Prepare state, package registration, Logger lifetime,
heap corruption and a vendor runtime defect remain alternative questions. They
were not collectively repaired here. The generic SDK contract and Host mocks
must not be substituted for the missing intervention experiment.

## G. Research decision and all requested answers

First restore availability of the original physical device, then perform a fresh
safety/crash-capture/identity review before deciding on the still-unexecuted
one-shot diagnostic. No automatic continuation or retry is scheduled. Full formal
candidate research is not authorized by a Host-only result. If the boundary pilot
later passes, a separate plan must explicitly amend the formal lifetime clause,
freeze the new ARM identity and cover all five roles, all Core/Holdout/observers,
three independent processes, both digests, fresh regression and performance.

| # | Answer |
| --- | --- |
| 1 | Yes: designated clean research clone, exact `6c4058a` HEAD; full ancestry recorded. |
| 2 | Yes: all original DSP/Host/probe bytes unchanged. |
| 3 | `2ba5640f5842f9477ad165bae4d49e3a59d19135b60751ff101de508f46ce94d`. |
| 4 | Runtime delta is Backend/Device ownership and necessary context/package cleanup binding; bounded entry point/tracing are diagnostic controls. Worker math/topology/gates are exact-source verified. |
| 5 | Sequential Contexts under retained handles are consistent with the pinned API contract; real HTP behavior remains untested. Logger repair was excluded. |
| 6 | Yes by source and Host mock; no device measurement. |
| 7 | Host 1/1; device null. |
| 8 | Host 1/1; device null. |
| 9 | Host 2/2; device null. |
| 10 | Zero in ownership model; actual Prepare loading between roles unmeasured. |
| 11 | PASS: dedicated lifecycle/fault injection/descriptor checks. |
| 12 | NOT_EXECUTED for this candidate. |
| 13 | NOT_MEASURED; old anchor retained, no new digest claimed. |
| 14 | NOT_EXECUTED; actual return null. |
| 15 | NOT_EXECUTED; no real DSP bounded-case PASS. |
| 16 | No process launched; recurrence untested, not a no-crash PASS. |
| 17 | New signal/library/offset/stack null; original signature above is retained. |
| 18 | New API6031/6033 observations null; historical independent watches retained. |
| 19 | NOT_EXECUTED on device; Host normal cleanup PASS. |
| 20 | null: no diagnostic OS process. |
| 21 | Unknown: no online device; hard safety stop honored. |
| 22 | Device competition unobserved/unknown; no fresh absence claim. |
| 23 | Yes by exact worker source and bit-identical DSP/Host; real candidate numerics unmeasured. |
| 24 | Yes: old raw/tombstones rehashed unchanged. |
| 25 | Yes: protected files, all pre-existing refs and the 333-file ledger unchanged. |
| 26 | Original evidence supports the hypothesis; intervention and root cause remain unresolved. |
| 27 | Yes: zero added formal processes; always excluded. |
| 28 | Not yet: perform the bounded intervention first; no formal-candidate advancement from Host-only evidence. |
| 29 | Yes: Backward performance remains NOT_MEASURED. |
| 30 | Full Attention/Full1-layer/Full19-layer NOT_CERTIFIED; 40k→60k and 200k training NO-GO. |

Public output is generated only through
`scripts/export_public_backward_shared_lifetime.py`. Its closed schema rejects
old-binary replacement/impersonation, formal promotion, missing finalize/exit/
teardown/resource counts, hidden arithmetic mismatch, unavailable crash evidence
treated as successful capture, private fields, a second process and protected
mismatch. The private evidence root is independent of every old report root;
raw data, local paths, runtime addresses, app/device identities, APK hashes and
endpoints are not exported.

## 2026-10-23 — Resumed one-process boundary pilot

### Provenance and prelaunch-only corrections

The user explicitly resumed the mission after an online endpoint became visible.
Resume started clean at `8de526883ccada2b62b90ea7b6d4c94a59c36c18`, on the same
diagnostic branch and research clone. The original `6c4058a` branch, all protected
refs, original source/binaries and old private evidence remained unchanged.

The first fresh read-only safety preflight passed physical identity, active-run
absence, Awake, known top-resumed, installed approved APK, SDK/toolchain, storage
and critical protection. A subsequent read-only crash-availability check lost
its ADB endpoint before any probe launch. A currently online endpoint was then
matched by stable device identity to the original phone, without a reconnect,
wake or other device-state command. A new safety preflight passed. This transport
episode is preserved separately; it is not an arithmetic trial or numerical failure.

Prelaunch Host testing exposed the old, never-executed diagnostic runner's use
of PowerShell's automatic `$Args` variable as a declared parameter: it discarded
the crash command array. The old runner and ARM compilation manifest were kept
frozen. A separate runner, `scripts/run_backward_shared_lifetime_device.ps1`, and
committed `backward_shared_lifetime_runner_addendum.json` correct that binding.
The addendum keeps the original roles, case identities, limits, mathematics,
source/binary identities, hard gates and one-process cap. Its launch receipt
binds the exact fresh Host inventory, runner and addendum hashes, with a bounded
review handoff and another safety/competition recheck before the process claim.

An additional prelaunch-only stop falsely matched two system processes by the
substring `probe`: the exact `policy_probe_wq` kernel worker and `traced_probes`.
Read-only `/proc` evidence established Kthread1/PPID2/root/empty cmdline for the
former, and Kthread0/PPID1/UID9999 with exact system cmdline for the latter.
Perfetto documents `traced_probes` as its system/kernel tracing daemon, consistent
with the observed identity. [Perfetto reference](https://perfetto.dev/docs/reference/traced_probes).
Executable symlink access was unavailable and remains null; root was not used.

The corrected classifier accepts only those exact, attested identities, pins
PID/PPID/user/name/full-command tuples, saves the evidence, and still stops on
unknown probe/training/HMX matches. This applies the existing distinction between
system infrastructure and recognizable NPU experiments; it does not waive a real
competing workload. Both prelaunch stops had **zero diagnostic processes** and
no consumed process claim. They were retained, not overwritten as successes.

The executed orchestration source was committed in
`da614e5403d71b83aaeb2a2365af899de5d0ddc2`. Runner SHA256:
`8499bcb577a310f3b11f188c9694ae95b660b34af23c8efa6f20fa5bfb5a9e18`.
The ARM candidate stayed exactly
`2ba5640f5842f9477ad165bae4d49e3a59d19135b60751ff101de508f46ce94d`,
compiled from `658df776aff26b2297e9d4c8703e6978878d6b7d`.
No ARM, DSP, Host, graph or arithmetic source changed during resume.

### Fresh gates and safety

All seven dedicated lifecycle/arithmetic/prepare/independent-reference/graph
executables were rerun successfully with new private outputs. Capture-function
Host tests passed six command/error-preservation cases; infrastructure tests
passed nine cases, including rejection of unknown workloads, wrong user/parent,
wrong Kthread status, wrong cmdline and unreadable identity. Fresh PS7 Fast passed
after each substantive runner change; the executed runner's gate was 70.9 seconds,
outer 1 PASS / 0 FAIL / 0 SKIP. The frozen candidate's prior fresh Qnn profile
remained applicable; Qnn/standalone Host profiles were not invoked again on resume.
The actual runner performed fresh approved-APK and staged-file audits.

Final runner preflight and immediate prelaunch safety checks passed on the same
physical device. No active training or recognizable competing NPU experiment was
found. Five successful process snapshots covered prelaunch and runtime; only the
owned diagnostic, two attested system identities and reviewed persistent RPC/
kernel infrastructure matched the relevant terms. This does not prove absence
of all in-process clients or activity between snapshots. Two-second top-resumed
monitoring remained in place, with takeover count0 and no safety stop. Before/
after critical thermal status was0 and battery health was healthy; ordinary power,
temperature and battery telemetry retained the existing monitoring-only policy.
No UI takeover, install, permission change, force-stop, root, reboot or training
occurred. The approved APK was not replaced.

### Actual execution, numerical and lifetime evidence

An exclusively created mission-level claim allowed exactly **one** ARM process.
Its claim time was **2026-10-23 00:05:47 JST**. There was no automatic retry,
second process, third role, full dValue run or formal/performance experiment.
The controller reviewed the exact fresh Host inventory before issuing its bound
launch receipt. Probe output, shell completion and captured completion metadata
agree on **actual OS exit0**.

| Measurement | Result |
| --- | --- |
| Backend create/free | 1 / 1, both return0 |
| Device create/free | 1 / 1, both return0 |
| Context create/free | 2 / 2, all return0 |
| Graphs / successful finalize returns | 8 / 8; four per independent role Context |
| Successful executes | 5,012 = 4,912 Softmax + 100 bounded dValue |
| First dValue finalize | Actual entry and return0 captured |
| Prepare load/unload | 1 / 1 for the entire process; unload only after final Backend free starts |
| Inter-role Prepare unload/reload | 0 observed; no inter-role Backend/Device free or recreate |
| HTP provider initialization | 1 |
| Whole-process required cleanup | Both Contexts, then Device, then Backend; return0 throughout |
| Runtime QNN nonzero API results | 0 |
| API6031 / token6033 | 0 / 0 in the complete probe stdout/stderr, independently scanned |
| SIGSEGV / owned-process tombstone | Not reproduced / no matching owned-process tombstone |

SDK stderr additionally reports zero remaining Contexts/Graphs before final Device
free and successful Device free. Only then are packages unloaded, the environment
closed and Prepare unloaded. Logger repair remains excluded; this result does not
claim that the known original Logger leak has been fixed.

| Independent numerical audit | Softmax | Bounded dValue |
| --- | ---: | ---: |
| Pairs | 9,824 | 200 |
| Observer conditions | All four | All four |
| Core coverage | 2,200 per observer | Registered Forward0 × gradients0–24 × heads0–1 per observer |
| Holdout coverage | 256 per observer | NOT_EXECUTED |
| Total/Core maxAbs | `6.335392901668513e-5` | `6.955715726064682e-6` |
| Holdout maxAbs | `2.2067718461471486e-6` | null |
| Unchanged maxAbs gate | 0.002 | 0.01 |
| Ordered FP32 word mismatches | 0 | 0 |
| Observer word mismatches | 0 | 0 |
| Independent double recheck | PASS | PASS |
| Finite / full writes / guards / poison / input immutability | PASS | PASS |

Softmax numerical digest is
`fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267`,
equal to the old anchor and both completed crash-episode Softmax observations.
The bounded dValue digest is
`1808c34fc0011796c1693355a3bccce5efa08794c5aafdf0820499430f7bb76e`;
it is a 200-pair diagnostic digest, not the full standalone dValue anchor or a
formal legacy/supplemental cohort digest.

The independently read raw sizes are 284,827,248 bytes for Softmax and 3,713,216
bytes for dValue. Each compressed transfer was hashed, losslessly decoded into
an exclusively created local file, and checked against remote-before/local/
remote-after identities. Independent assessment checks exact frozen P/gradient
words, receipts, all expected case keys, numerical reference and output words.

### Crash capture and final stream audit

Before/after crash/main/system logcat snapshots, probe QNN stderr, accessible
Android tombstones and DropBox metadata were collected successfully without
root, clearing logs or changing SELinux. All 32 accessible text tombstones were
read; none matched the owned diagnostic PID. Signal/fault address, new faulting
library/relative offset and new main-thread backtrace are null because this
process did not crash. The old two matching 17-frame stacks at
`libQnnHtpPrepare.so + 0x3007bac` remain intact.

After collector shutdown and Host redirection drain, the continuous logcat file
contains **1,737,957 bytes / 11,960 lines**, including 139 owned-process lines;
its stderr is empty. The owned-process stream and post-snapshot lines each have
zero 6031/6033 tokens and zero SIGSEGV/fatal-signal 11 records. Continuous capture
is therefore exported as AVAILABLE. Complete probe stderr, successful snapshots,
tombstone/DropBox access and actual API/exit/cleanup evidence provide additional
comparison sources; completeness of every Android log message is not asserted.

An interim inspection saw the still-active redirection file at zero bytes and
was prematurely described as the final stream state. That commentary and the
provisional private review are corrected by a separate final stream audit receipt;
the earlier observation is retained rather than overwritten. The final projection
uses the completed, nonempty saved stream. The exporter also requires successful,
nonempty snapshots/stderr and successful tombstone/DropBox collections for this
observed PASS, so missing capture is not silently treated as normal evidence.

### Scientific conclusion and next decision

**SHARED_LIFETIME_BOUNDARY_PILOT_PASS.**
**BACKEND_REINITIALIZATION_HYPOTHESIS_STRENGTHENED. ROOT_CAUSE_NOT_PROVEN.**

With the same physical device, fixed QAIRT and unchanged DSP/Host/operands/math,
the old candidate crashed twice at first dValue finalize after Backend/Device
recreation; the new candidate retained them and passed that boundary, its bounded
arithmetic and required teardown in one process. This is an intervention result,
not a failure-rate estimate or unique-cause proof. It jointly retains Backend,
Device and their package/Prepare state; it does not isolate Backend from Device,
package registration or process-global vendor state. Other lifetime/corruption/
vendor-runtime explanations remain unresolved.

A separate full-candidate research plan is now justified. It must explicitly
amend the original formal lifetime clause and freeze the new ARM identity, then
cover full standalone Softmax/dValue, native singles and dual, all four observers,
full Core/Holdout, three independent processes, legacy/supplemental agreement,
fresh regression and subsequently performance. None of that was started here.
The original formal plan is unchanged. **BLOCKED_FORMAL_COHORT = 0/3** remains;
Full Attention/Full1-layer/Full19-layer are NOT_CERTIFIED; 40k→60k and 200k training
remain **NO-GO**. Performance remains NOT_MEASURED.

### Current answers to all 30 final questions

| # | Current answer after the one-process intervention |
| --- | --- |
| 1 | Yes; exact clean research base and ancestry retained; resume on its dedicated branch. |
| 2 | Yes; original frozen DSP/Host/probe unchanged. |
| 3 | `2ba5640f5842f9477ad165bae4d49e3a59d19135b60751ff101de508f46ce94d`. |
| 4 | ARM delta remains lifetime/ownership plus registered entry/tracing controls; no arithmetic/topology change. Resume changed collection/orchestration only. |
| 5 | Yes under the audited pinned API contract; generic legality and actual tested success do not prove vendor-runtime defect absence. |
| 6 | Yes: two independently created/freed role Contexts; no cross-role graph/tensor/descriptor handoff. |
| 7 | Actual Backend create/free 1/1, return0. |
| 8 | Actual Device create/free 1/1, return0. |
| 9 | Actual Context create/free 2/2, return0. |
| 10 | No inter-role reload observed; one Prepare load, one final Backend-time unload. |
| 11 | PASS, including fresh resume reruns and the separately tested runner corrections. |
| 12 | PASS: Softmax 9,824 pairs across full Core/Holdout and all four observers. |
| 13 | Yes: digest exactly equals the old anchor. |
| 14 | Yes: first dValue graphFinalize returned0; all four dValue finalizes returned0. |
| 15 | PASS: registered 200 pairs / 100 executes, both independent oracles and word contracts. |
| 16 | Not reproduced in this single diagnostic process; no general no-crash guarantee. |
| 17 | New fault library/offset/stack null; original two matching signatures retained. |
| 18 | API6031 and token6033 each0 in complete probe stdout/stderr; historical watches remain separate and unresolved. |
| 19 | Required Context→Device→Backend cleanup PASS; the deliberately excluded Logger leak is not repaired. |
| 20 | Actual Android child/process exit0, captured through shell wait/completion metadata. |
| 21 | Yes: fresh Awake/known-top checks and runtime monitor passed, takeover count0. |
| 22 | No recognizable separate NPU experiment in five reviewed snapshots; complete absence is not proven. |
| 23 | Yes: exact worker source, unchanged DSP/Host and fixtures, actual ordered-word and independent numerical PASS. |
| 24 | Yes: old failure raw/tombstones preserved and rehashed. |
| 25 | Yes: protected artifacts/refs and prior 333 ledger unchanged; new evidence is separate. |
| 26 | Strengthened by the intervention; unique root cause still unresolved, not verified. |
| 27 | Yes: this pilot is excluded; zero formal processes added. |
| 28 | A separate full-candidate plan is justified; no formal launch or certification approval is inferred. |
| 29 | Yes: Backward performance remains NOT_MEASURED. |
| 30 | Full Attention/Full1-layer/Full19-layer NOT_CERTIFIED; both requested training paths NO-GO. |

Closeout: fresh PS7 Fast **PASS**, 73.3 seconds, outer 1 PASS / 0 FAIL / 0 SKIP;
inner 10 PASS / 0 FAIL / 21 SKIP. The final allow-list exporter passes 30 rejection
cases, including nonzero first finalize, missing graph-count evidence and missing
observed Prepare lifetime. All original frozen bundles, protected artifacts,
prior 333 ledger, 288 earlier native/exit139 evidence files and pre-existing
nonworking refs passed final preservation checks. No additional device process
was launched for closeout. The result is committed on the dedicated diagnostic
branch; no push, PR, merge or history rewrite is performed.
