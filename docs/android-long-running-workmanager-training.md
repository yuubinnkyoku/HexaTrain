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

- controlled multi-hour run で Android 16 JobScheduler quota を観測する（今回の作業範囲外）
- normal process death 後の Worker/repository/checkpoint reattachment を短い安全な run で確認する。`force-stop` は同等の試験として扱わない
- forced idle は端末を占有できる development window で短時間だけ実施し、必ず device state を復元する

## References

- [Support for long-running workers](https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running)
- [Foreground service types: special use](https://developer.android.com/develop/background-work/services/fgs/service-types#special-use)
- [WorkManager release notes](https://developer.android.com/jetpack/androidx/releases/work)
