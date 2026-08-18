# Android long-running WorkManager training

## Decision

HexaTrain の production standalone GUI training は、独自 foreground service を execution owner にせず、unique WorkManager work と long-running `CoroutineWorker` で所有する。Worker は training 開始前に `setForeground()` を呼び、WorkManager の `SystemForegroundService` が foreground lifecycle と通知を管理する。

短い Android 16 実機 probe と QNN smoke はこの構成で動作した。ただし Android 16 では long-running Worker も JobScheduler quota の影響を受け得るため、この結果は数時間の training が quota 制限を受けないことを証明しない。独自 direct FGS への自動 fallback は実装しない。

以前検討した user-initiated data transfer job (UIJ) は、local ML training が data transfer ではないため対象外とした。pure on-device training を `dataSync` として宣言しない。

## Architecture

以前:

```text
Activity / repository
  -> app-owned foreground service
  -> native QNN training core
```

現在:

```text
Activity
  -> TrainingWorkCoordinator
  -> WorkManager unique work (KEEP)
  -> StandaloneTrainingWorker : CoroutineWorker
  -> setForeground(ForegroundInfo)
  -> WorkManager SystemForegroundService
  -> existing repository / native QNN training core
  -> checkpoint + progress
```

WorkManager は execution ownership と lifecycle だけを担当する。training algorithm、model、optimizer、tokenizer、training order、checkpoint format、generation lifecycle、および `HeadlessDeviceTestRunner` は変更しない。既存 repository/session は observable state と cooperative control の facade として残す。

既存 foreground service は benchmark path でも使われているため削除していない。standalone Worker path では起動せず、`SystemForegroundService` と二重に動作させない。

## Foreground service declaration

- type: `specialUse`
- permissions: `FOREGROUND_SERVICE` と `FOREGROUND_SERVICE_SPECIAL_USE`
- merged service: `androidx.work.impl.foreground.SystemForegroundService`
- subtype: user-initiated on-device ML model training の説明を `PROPERTY_SPECIAL_USE_FGS_SUBTYPE` に設定
- API 34 以降: `ForegroundInfo` に `FOREGROUND_SERVICE_TYPE_SPECIAL_USE` を明示

既存 `dataSync` は benchmark の既存 service path がまだ使用するため現時点では残す。standalone training の runtime owner は `specialUse` の WorkManager service だけである。

## Lifecycle and persistence

`phonelm-standalone-training` を unique work name とし、`ExistingWorkPolicy.KEEP` を使う。scheduler-level の unique work と既存 process/native arbiter を二層の single-flight とする。

app-private durable metadata は `runId`、WorkRequest UUID、model/config identity、dataset identity、phase、step、latest checkpoint、terminal state、開始時刻、stop reason を結び付ける。WorkInfo の progress step は resume point に使わない。Worker が再生成された場合は、latest complete/compatible/finite checkpoint を選択し、checkpoint がない最初期だけ fresh initialization を許す。

UI Stop と通知 Stop は WorkManager cancellation を発生させる。Worker の coroutine cancellation は repository/native stop request に伝播し、native safe boundary、任意の safe checkpoint、resource release、terminal state を待ってから終了する。Pause は cancellation と分離し、同一 Worker 内の既存 pause/resume semantics を維持する。interrupted run は compatible durable checkpoint がある場合だけ Resume を表示する。

Compose は WorkManager を直接操作せず `TrainingWorkCoordinator` を介する。navigation と Activity recreation は Worker を cancel/再作成しない。rich live telemetry は repository state、WorkManager progress は coarse fields のみに限定する。

## Probe and device observations

2026-08-18 に Android 16 / API 36 の物理端末で、private endpoint/serial を保存せず次を確認した。

- isolated CPU probe 45 秒: enqueue、RUNNING、progress、notification、`SystemForegroundService`、runtime `specialUse`、SUCCESS、service/notification cleanup が PASS
- cancel probe: cancel action、STOPPED、app cancel stop reason、service/notification cleanup が PASS
- Activity recreation: 同一 WorkRequest UUID のまま RUNNING を維持
- screen-off probe 180 秒: manual wake lock なしで Dozing 中も progress 継続。途中で外部操作により screen が復帰したため、全時間の deep-idle 継続を意味しない
- forced-idle diagnostic: 端末利用と競合するため未実施
- JobScheduler: probe 中は quota 内。短時間なので quota exhaustion は未検証

Production QNN smoke は QNN-enabled APK で GUI Start から実施した。WorkManager Worker が `SystemForegroundService` の `specialUse` として動作し、旧 standalone service との二重 FGS はなかった。step 8 で Stop を要求し、native safe checkpoint boundary の step 39 で `INTERRUPTED` となった。checkpoint 保存、QNN return success、finite output tensors、CPU fallback 未観測をそれぞれ確認し、service/notification は終了した。これは lifecycle/correctness smoke であり、学習品質評価ではない。

## Android 16 quota and diagnostics

Worker は WorkInfo state、coarse progress、および利用可能な stop reason を app-private diagnostics に残す。private prompt、dataset text、ADB endpoint、device serial は記録しない。

短い probe で判断できるのは foreground promotion、manifest/runtime type、notification、cancellation、cleanup、および短時間の background progress までである。数時間 run で `STOP_REASON_QUOTA`、timeout、または再現する system stop が観測された場合は、training core を変更せず direct `specialUse` FGS を別 architecture として評価する。WorkManager から direct FGS へ自動 fallback はしない。

## Remaining validation

- controlled multi-hour run は下記の結果で完了した。Android 16 の JobScheduler timeout 履歴は下記のT2調査でWorker非停止を確認したが、OEM固有の発火理由は未特定
- normal process death 後の Worker/repository/checkpoint reattachment は、下記の短い安全な run で確認済み。`force-stop` は同等の試験として扱わない
- forced idle は端末を占有できる development window で短時間だけ実施し、必ず device state を復元する

### Controlled multi-hour soak

Configuration: private run identity `workmanager-soak-20260818`; stable production GUI preset D32/FFN32/L19/H2/T32, seed 1, batch 8, target step 8000. No model, tokenizer, optimizer, dataset, training-order, QNN graph, or checkpoint-semantics changes were made for this run.

Duration: 4 h 05 min 07 s until the GUI Stop request (4 h 05 min 09 s until the worker terminal boundary). The hard-ceiling stop attempt woke the screen but was blocked by the keyguard; after the user unlocked the device, the same production Stop control was used. Screen-off/Dozing was observed for at least 4 h 00 min. `dumpsys deviceidle` remained non-forced (`mForceIdle=false`, state `ACTIVE`); no forced-idle or stay-awake override was used.

WorkManager: UUID `b5ddb797-fdf2-489a-bf9b-0ae506fa973c`; one continuous run, `runAttemptCount=1`, final WAL-inclusive WorkInfo state `CANCELLED`, native phase `INTERRUPTED`, no duplicate work. The low-frequency monitor lagged at the terminal transition, so the final state was confirmed from the WorkManager database after the stop.

FGS: WorkManager `SystemForegroundService`, runtime type `specialUse`, present throughout the active samples. After the cooperative stop the service was absent, the training notification was absent from the active notification list (the record remained only in notification history), and the legacy custom training service was absent.

JobScheduler: active samples remained `WITHIN_QUOTA`; `STOP_REASON_QUOTA` was not observed. Android 16 nevertheless recorded repeated `SystemJobService timeout` stop/start history (`Num failures=5`, `Num system stops=1`) while the same WorkRequest UUID, process, and progress continued. This is non-quota caveat evidence, not a quota exhaustion result, and needs a separate follow-up.

Training health: progress continued while the screen was off and the device reported `Dozing`; sampled progress advanced through step 8000 without a WorkManager restart. QNN success/failure counters, tensor-finite counters, and fallback state were not available from the private monitor, so this soak is not used as a new model-quality or QNN numerical result.

Checkpoint: latest native checkpoint reached step 8000. Existing checkpoint inspection verified `NPRTCKPTV2`, V256/T32/D32/FFN32/L19/H2, seed 1, finite parameter/Adam registries, and zero trailing bytes. The checkpoint identity matched the stable GUI configuration and dataset identity.

Stop latency: request → native cancellation observed = 1.562 s; request → Worker terminal = 1.562 s; steps after request = 0. A new post-request safe-checkpoint interval was not required because the step-8000 checkpoint had already completed before the Stop request.

Resume: `SKIP`. The latest compatible checkpoint was already at the fixed plan target (step 8000), so the production UI correctly did not offer a meaningful Resume action. No Start-over or second long run was created.

Verdict at soak completion: **B — ADOPT WITH CAVEAT**. WorkManager-managed `specialUse` FGS sustained the multi-hour screen-off run, remained within quota, and cooperatively stopped with checkpoint and lifecycle cleanup. At that point the caveats were the Android 16 JobScheduler timeout history, lack of a deep/forced-idle observation, untested normal process-death recovery, and unavailable QNN counter telemetry; the follow-up sections below update the first two lifecycle questions.

### SystemJobService timeout investigation

The four-hour evidence was correlated with the WorkManager database snapshot
captured while the run was active. The production WorkRequest
`b5ddb797-fdf2-489a-bf9b-0ae506fa973c` was also the `WorkSpec` ID; its
`generation` was `0`, `runAttemptCount` was `1`, and `SystemIdInfo` mapped that
generational ID to JobScheduler job ID `6`. This rules out treating the timeout
records as an unrelated package job or as a different WorkManager generation.

The JobScheduler history for the same package/job showed repeated
`SystemJobService timeout` stop/start records, ending at `Num failures=5` and
`Num system stops=1`. Across the monitor samples, however, the WorkInfo state
remained `RUNNING`, `runAttemptCount` remained `1`, the app PID remained
unchanged, native progress advanced monotonically from step 368 through step
8000, and checkpoint cadence continued. There was no observed
`STOP_REASON_QUOTA`; the job remained within the recorded quota/foreground
exemption state. The only terminal stop was the later GUI cooperative Stop,
which produced the expected cancellation record.

Classification: **T2 — BACKING JOB EVENT, WORKER UNAFFECTED**. AndroidX
WorkManager encodes the WorkSpec ID and generation into the JobInfo extras and
`SystemJobService` maps them back on start/stop; a delivered `onStopJob()` would
call the WorkManager processor's stop path. The absence of the corresponding
Worker/Processor transition in the synchronized WorkInfo, PID, native, and
checkpoint evidence means the observed strings are scheduler-level backing-job
timeout history, not proof that the production Worker stopped. The exact OEM /
Android 16 trigger for the historical entries is not recoverable from the
saved per-event callback data, so this remains a diagnostic caveat rather than
a quota or architecture failure.

The reconnected short recovery run reproduced the same shape: its mapped Job ID
recorded two timeout entries while WorkInfo stayed `RUNNING`, the checkpoint
advanced to step 1750, and the process identity remained stable until the
separate induced crash. After that crash, the same WorkSpec was rescheduled as
attempt 2 and later canceled by the GUI Stop. This is corroborating T2 evidence,
not a second long-soak result.

### Process death recovery

Method: production GUI Start on the stable D32/FFN32/L19/H2/T32 configuration,
then Android's `am crash` shell-induced VM crash. This is an induced crash
semantics check, not an LMK/natural-kill claim; neither `force-stop` nor
`kill -9` was used. The first transport-disconnected attempt reached step 120
before step 250 and is retained as external-inconclusive evidence. After the
device reconnected, the same active run provided the valid recovery pair.

Before death, the durable metadata named WorkRequest
`06f25f01-6619-417e-b8b6-aebad0e90bdb`, runId
`56aef38b-b698-448a-bae3-f54f60e1cd9a`, and a complete compatible finite
checkpoint at step 1750. The process was in `TRAINING` beyond that checkpoint.
`ApplicationExitInfo` recorded an app crash and the process identity changed.
WorkManager kept the same WorkRequest UUID and generation; the final
WAL-inclusive database showed `runAttemptCount` advancing from 1 to 2, with no
second WorkSpec. The foreground service/job were recreated once and the
production Training screen displayed `Training resumed` from step 1750, then
progressed beyond it (step 1800 observed). This rules out fresh initialization
and WorkInfo-progress-only resume for this run.

The post-recovery GUI Stop produced a complete terminal checkpoint at step
2014. WorkInfo then reached `CANCELLED`, the SystemForegroundService and
WorkManager job mapping were absent, and no active training notification or
legacy training service remained. Checkpoint metadata remained finite and
identity-compatible; detailed QNN success/failure counters were unavailable
from the existing monitor.

Result: **PASS** for induced crash → checkpoint recovery → continued training
→ cooperative GUI Stop. This does not cover LMK or every natural process-death
cause.

### Deep Idle

**SKIP.** The completed soak already demonstrated continued progress during
screen-off / natural Dozing. No additional forced-idle diagnostic was started
while the targeted recovery run was incomplete.

### Follow-up architecture verdict

**B — ADOPT WITH CAVEAT.** The timeout history is classified T2: the active
WorkSpec/job mapping is known, but the OEM-specific reason for the backing
`SystemJobService timeout` entries is not recoverable beyond the AndroidX
semantics. The Worker itself continued through those entries, quota was never
exhausted, and the induced process-death recovery passed from the exact finite
checkpoint. WorkManager-managed `specialUse` FGS remains the production
architecture; no direct-FGS fallback or quota bypass is warranted.

Remaining caveats are limited to the OEM timeout trigger, natural LMK-style
process death (the test used an induced VM crash), and deep/forced-idle
behavior. None is a reason to rerun a multi-hour soak or start another ML
experiment in this task.

## References

- [Support for long-running workers](https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running)
- [Foreground service types: special use](https://developer.android.com/develop/background-work/services/fgs/service-types#special-use)
- [WorkManager release notes](https://developer.android.com/jetpack/androidx/releases/work)
