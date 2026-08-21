package com.yuubinnkyoku.phonelm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.SystemClock
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.UUID

data class TrainingWorkerIdentity(
    val modelConfigIdentity: String,
    val datasetIdentity: String,
    val latestCheckpointUri: String?,
    val modelConfigEncoding: String = modelConfigIdentity,
)

enum class TrainingWorkerStartMode { FRESH, RESUME }

data class DurableTrainingWork(
    val runId: String,
    val workId: String,
    val modelConfigIdentity: String,
    val datasetIdentity: String,
    val phase: String,
    val currentStep: Int,
    val maxStep: Int,
    val latestCheckpointUri: String?,
    val terminalStatus: String?,
    val runStartedAtMs: Long,
    val executionStarted: Boolean,
    val stopReason: Int,
    val modelConfigEncoding: String = modelConfigIdentity,
)

class TrainingWorkMetadataStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    @Synchronized
    fun create(runId: String, request: OneTimeWorkRequest, identity: TrainingWorkerIdentity) {
        check(prefs.edit()
            .putString(KEY_RUN_ID, runId)
            .putString(KEY_WORK_ID, request.id.toString())
            .putString(KEY_CONFIG_IDENTITY, identity.modelConfigIdentity)
            .putString(KEY_CONFIG_ENCODING, identity.modelConfigEncoding)
            .putString(KEY_DATASET_IDENTITY, identity.datasetIdentity)
            .putString(KEY_PHASE, TrainingPhase.IDLE.name)
            .putInt(KEY_CURRENT_STEP, 0)
            .putInt(KEY_MAX_STEP, 0)
            .remove(KEY_CHECKPOINT_URI)
            .remove(KEY_TERMINAL_STATUS)
            .remove(KEY_STOP_REQUESTED_AT_ELAPSED_MS)
            .remove(KEY_NATIVE_STOP_OBSERVED_AT_ELAPSED_MS)
            .remove(KEY_SAFE_CHECKPOINT_STARTED_AT_ELAPSED_MS)
            .remove(KEY_SAFE_CHECKPOINT_FINISHED_AT_ELAPSED_MS)
            .remove(KEY_WORKER_TERMINAL_AT_ELAPSED_MS)
            .remove(KEY_STOP_BASE_CHECKPOINT_URI)
            .putLong(KEY_STARTED_AT_MS, System.currentTimeMillis())
            .putBoolean(KEY_EXECUTION_STARTED, false)
            .putInt(KEY_STOP_REASON, STOP_REASON_UNAVAILABLE)
            .commit()) { "training work metadata could not be created" }
    }

    @Synchronized
    fun load(): DurableTrainingWork? {
        val runId = prefs.getString(KEY_RUN_ID, null) ?: return null
        val workId = prefs.getString(KEY_WORK_ID, null) ?: return null
        val configIdentity = prefs.getString(KEY_CONFIG_IDENTITY, null) ?: return null
        val configEncoding = prefs.getString(KEY_CONFIG_ENCODING, null) ?: runCatching {
            TrainingModelConfigCodec.encode(TrainingModelConfigCodec.decodeLegacyCompatibilityKey(configIdentity))
        }.getOrNull() ?: return null
        return DurableTrainingWork(
            runId = runId,
            workId = workId,
            modelConfigIdentity = configIdentity,
            datasetIdentity = prefs.getString(KEY_DATASET_IDENTITY, null) ?: return null,
            phase = prefs.getString(KEY_PHASE, TrainingPhase.IDLE.name)!!,
            currentStep = prefs.getInt(KEY_CURRENT_STEP, 0),
            maxStep = prefs.getInt(KEY_MAX_STEP, 0),
            latestCheckpointUri = prefs.getString(KEY_CHECKPOINT_URI, null),
            terminalStatus = prefs.getString(KEY_TERMINAL_STATUS, null),
            runStartedAtMs = prefs.getLong(KEY_STARTED_AT_MS, 0L),
            executionStarted = prefs.getBoolean(KEY_EXECUTION_STARTED, false),
            stopReason = prefs.getInt(KEY_STOP_REASON, STOP_REASON_UNAVAILABLE),
            modelConfigEncoding = configEncoding,
        )
    }

    @Synchronized
    fun markExecutionStarted(): DurableTrainingWork? {
        val current = load() ?: return null
        check(prefs.edit().putBoolean(KEY_EXECUTION_STARTED, true).commit()) {
            "training work start could not be persisted"
        }
        return current
    }

    @Synchronized
    fun update(state: TrainingUiState) {
        val edit = prefs.edit()
            .putString(KEY_PHASE, state.phase.name)
            .putInt(KEY_CURRENT_STEP, state.progress?.completedSteps ?: 0)
            .putInt(KEY_MAX_STEP, state.progress?.totalSteps ?: 0)
        state.lastCheckpoint?.uri?.let { edit.putString(KEY_CHECKPOINT_URI, it) }
        if (state.phase in TERMINAL_PHASES) edit.putString(KEY_TERMINAL_STATUS, state.phase.name)
        val stopRequestedAt = prefs.getLong(KEY_STOP_REQUESTED_AT_ELAPSED_MS, UNSET_ELAPSED_MS)
        if (stopRequestedAt != UNSET_ELAPSED_MS) {
            val now = SystemClock.elapsedRealtime()
            val stopCheckpointBase = prefs.getString(KEY_STOP_BASE_CHECKPOINT_URI, null)
            val checkpointStarted = prefs.getLong(
                KEY_SAFE_CHECKPOINT_STARTED_AT_ELAPSED_MS,
                UNSET_ELAPSED_MS,
            )
            if (state.phase == TrainingPhase.SAVING_CHECKPOINT &&
                checkpointStarted == UNSET_ELAPSED_MS
            ) {
                edit.putLong(KEY_SAFE_CHECKPOINT_STARTED_AT_ELAPSED_MS, now)
            }
            val checkpointFinished = prefs.getLong(
                KEY_SAFE_CHECKPOINT_FINISHED_AT_ELAPSED_MS,
                UNSET_ELAPSED_MS,
            )
            val checkpointUri = state.lastCheckpoint?.uri
            if (checkpointFinished == UNSET_ELAPSED_MS &&
                checkpointUri != null && checkpointUri != stopCheckpointBase
            ) {
                edit.putLong(KEY_SAFE_CHECKPOINT_FINISHED_AT_ELAPSED_MS, now)
            }
            if (state.phase == TrainingPhase.INTERRUPTED &&
                prefs.getLong(KEY_NATIVE_STOP_OBSERVED_AT_ELAPSED_MS, UNSET_ELAPSED_MS) == UNSET_ELAPSED_MS
            ) {
                // This is the app-side observation of the native CANCELLED
                // terminal result, not a claim about a particular QNN core.
                edit.putLong(KEY_NATIVE_STOP_OBSERVED_AT_ELAPSED_MS, now)
            }
        }
        check(edit.commit()) { "training work progress could not be persisted" }
    }

    /** Records the user/notification cancellation request using boot-relative time. */
    @Synchronized
    fun markStopRequested() {
        if (prefs.getLong(KEY_STOP_REQUESTED_AT_ELAPSED_MS, UNSET_ELAPSED_MS) != UNSET_ELAPSED_MS) return
        check(prefs.edit()
            .putLong(KEY_STOP_REQUESTED_AT_ELAPSED_MS, SystemClock.elapsedRealtime())
            .putString(KEY_STOP_BASE_CHECKPOINT_URI, prefs.getString(KEY_CHECKPOINT_URI, null))
            .commit()) { "training stop request could not be persisted" }
    }

    /** Records the WorkManager coroutine's terminal boundary after cooperative cleanup. */
    @Synchronized
    fun markWorkerTerminal() {
        if (prefs.getLong(KEY_STOP_REQUESTED_AT_ELAPSED_MS, UNSET_ELAPSED_MS) == UNSET_ELAPSED_MS) return
        if (prefs.getLong(KEY_WORKER_TERMINAL_AT_ELAPSED_MS, UNSET_ELAPSED_MS) != UNSET_ELAPSED_MS) return
        check(prefs.edit().putLong(KEY_WORKER_TERMINAL_AT_ELAPSED_MS, SystemClock.elapsedRealtime()).commit()) {
            "training worker terminal timestamp could not be persisted"
        }
    }

    @Synchronized
    fun recordStopReason(reason: Int) {
        prefs.edit().putInt(KEY_STOP_REASON, reason).commit()
    }

    companion object {
        const val STOP_REASON_UNAVAILABLE = -1
        const val UNSET_ELAPSED_MS = -1L
        private const val PREFS = "phonelm_standalone_training_work"
        private const val KEY_RUN_ID = "run_id"
        private const val KEY_WORK_ID = "work_id"
        private const val KEY_CONFIG_IDENTITY = "config_identity"
        private const val KEY_CONFIG_ENCODING = "config_encoding_v1"
        private const val KEY_DATASET_IDENTITY = "dataset_identity"
        private const val KEY_PHASE = "phase"
        private const val KEY_CURRENT_STEP = "current_step"
        private const val KEY_MAX_STEP = "max_step"
        private const val KEY_CHECKPOINT_URI = "checkpoint_uri"
        private const val KEY_TERMINAL_STATUS = "terminal_status"
        private const val KEY_STOP_REQUESTED_AT_ELAPSED_MS = "stop_requested_at_elapsed_ms"
        private const val KEY_NATIVE_STOP_OBSERVED_AT_ELAPSED_MS = "native_stop_observed_at_elapsed_ms"
        private const val KEY_SAFE_CHECKPOINT_STARTED_AT_ELAPSED_MS = "safe_checkpoint_started_at_elapsed_ms"
        private const val KEY_SAFE_CHECKPOINT_FINISHED_AT_ELAPSED_MS = "safe_checkpoint_finished_at_elapsed_ms"
        private const val KEY_WORKER_TERMINAL_AT_ELAPSED_MS = "worker_terminal_at_elapsed_ms"
        private const val KEY_STOP_BASE_CHECKPOINT_URI = "stop_base_checkpoint_uri"
        private const val KEY_STARTED_AT_MS = "started_at_ms"
        private const val KEY_EXECUTION_STARTED = "execution_started"
        private const val KEY_STOP_REASON = "stop_reason"
        private val TERMINAL_PHASES = setOf(
            TrainingPhase.COMPLETED,
            TrainingPhase.ERROR,
            TrainingPhase.INTERRUPTED,
        )
    }
}

class TrainingWorkCoordinator(
    private val context: Context,
    private val repository: StandaloneTrainingRepository,
    private val workManager: WorkManager = WorkManager.getInstance(context),
    private val metadata: TrainingWorkMetadataStore = TrainingWorkMetadataStore(context),
) {
    @Synchronized
    fun start(mode: TrainingWorkerStartMode = TrainingWorkerStartMode.FRESH): Boolean {
        val identity = repository.workerIdentity() ?: return false
        val unfinished = workManager.getWorkInfosForUniqueWork(UNIQUE_WORK_NAME).get()
            .any { !it.state.isFinished }
        if (unfinished) return true
        val runId = UUID.randomUUID().toString()
        val request = request(runId, identity, mode)
        metadata.create(runId, request, identity)
        workManager.enqueueUniqueWork(UNIQUE_WORK_NAME, ExistingWorkPolicy.KEEP, request)
        return true
    }

    fun stop(): Boolean {
        metadata.markStopRequested()
        workManager.cancelUniqueWork(UNIQUE_WORK_NAME)
        return repository.cancelWorkerExecution()
    }

    fun resume(): Boolean = if (repository.snapshot().phase == TrainingPhase.PAUSED) {
        repository.resume()
    } else {
        start(TrainingWorkerStartMode.RESUME)
    }

    companion object {
        const val UNIQUE_WORK_NAME = "phonelm-standalone-training"
        const val TAG = "phonelm-standalone-training"
        const val KEY_RUN_ID = "run_id"
        const val KEY_CONFIG_IDENTITY = "config_identity"
        const val KEY_CONFIG_ENCODING = "config_encoding_v1"
        const val KEY_DATASET_IDENTITY = "dataset_identity"
        const val KEY_START_MODE = "start_mode"

        fun request(
            runId: String,
            identity: TrainingWorkerIdentity,
            mode: TrainingWorkerStartMode,
        ): OneTimeWorkRequest = OneTimeWorkRequestBuilder<StandaloneTrainingWorker>()
            .setInputData(workDataOf(
                KEY_RUN_ID to runId,
                KEY_CONFIG_IDENTITY to identity.modelConfigIdentity,
                KEY_CONFIG_ENCODING to identity.modelConfigEncoding,
                KEY_DATASET_IDENTITY to identity.datasetIdentity,
                KEY_START_MODE to mode.name,
            ))
            .addTag(TAG)
            .build()
    }
}

class StandaloneTrainingWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {
    private val repository = StandaloneTrainingRepositoryRegistry.get(appContext)
    private val metadata = TrainingWorkMetadataStore(appContext)
    private var lastNotificationAtMs = 0L

    override suspend fun doWork(): Result {
        val runId = inputData.getString(TrainingWorkCoordinator.KEY_RUN_ID) ?: return Result.failure()
        val configIdentity = inputData.getString(TrainingWorkCoordinator.KEY_CONFIG_IDENTITY) ?: return Result.failure()
        val configEncoding = inputData.getString(TrainingWorkCoordinator.KEY_CONFIG_ENCODING) ?: return Result.failure()
        val runConfig = runCatching { TrainingModelConfigCodec.decode(configEncoding) }.getOrNull()
            ?: return Result.failure()
        if (runConfig.compatibilityKey != configIdentity) return Result.failure()
        val datasetIdentity = inputData.getString(TrainingWorkCoordinator.KEY_DATASET_IDENTITY) ?: return Result.failure()
        val mode = runCatching {
            TrainingWorkerStartMode.valueOf(inputData.getString(TrainingWorkCoordinator.KEY_START_MODE).orEmpty())
        }.getOrNull() ?: return Result.failure()
        val durable = metadata.load()?.takeIf { it.runId == runId && it.workId == id.toString() }
            ?: return Result.failure()
        if (repository.selectedDatasetIdentity() != datasetIdentity ||
            durable.modelConfigIdentity != configIdentity ||
            durable.modelConfigEncoding != configEncoding ||
            durable.datasetIdentity != datasetIdentity
        ) return Result.failure()

        setForeground(createForegroundInfo(applicationContext, id, "Preparing training", 0, 0))
        val prior = metadata.markExecutionStarted() ?: return Result.failure()
        val updates = Channel<TrainingUiState>(Channel.CONFLATED)
        val subscription = repository.subscribe { updates.trySend(it) }
        try {
            val started = repository.attachOrStartWorkerRun(
                mode = mode,
                checkpointUri = prior.latestCheckpointUri,
                recoverStartedRun = prior.executionStarted,
                runStartedAtMs = prior.runStartedAtMs,
                runConfig = runConfig,
            )
            if (!started) return Result.failure(workDataOf(KEY_ERROR_CODE to "START_REJECTED"))
            while (true) {
                val state = updates.receive()
                metadata.update(state)
                setProgress(progressData(runId, state))
                updateForegroundIfNeeded(state)
                when (state.phase) {
                    TrainingPhase.COMPLETED -> {
                        metadata.markWorkerTerminal()
                        return Result.success(progressData(runId, state))
                    }
                    TrainingPhase.ERROR -> {
                        metadata.markWorkerTerminal()
                        return Result.failure(
                            progressData(runId, state, state.message ?: "TRAINING_FAILED"),
                        )
                    }
                    TrainingPhase.INTERRUPTED -> {
                        metadata.markWorkerTerminal()
                        return Result.success(progressData(runId, state))
                    }
                    else -> Unit
                }
            }
        } catch (cancelled: CancellationException) {
            repository.cancelWorkerExecution()
            withContext(NonCancellable) {
                while (repository.snapshot().phase in ACTIVE_PHASES) delay(CANCEL_POLL_MS)
                metadata.update(repository.snapshot())
                metadata.recordStopReason(stopReasonOrUnknown())
                metadata.markWorkerTerminal()
            }
            throw cancelled
        } finally {
            subscription.close()
            updates.close()
        }
    }

    private suspend fun updateForegroundIfNeeded(state: TrainingUiState) {
        val now = System.currentTimeMillis()
        if (now - lastNotificationAtMs < NOTIFICATION_INTERVAL_MS && state.phase !in TERMINAL_PHASES) return
        lastNotificationAtMs = now
        val progress = state.progress
        setForeground(createForegroundInfo(
            applicationContext,
            id,
            if (state.phase == TrainingPhase.PAUSED) "Paused" else state.phase.name,
            progress?.completedSteps ?: 0,
            progress?.totalSteps ?: 0,
        ))
    }

    private fun stopReasonOrUnknown(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) stopReason
        else TrainingWorkMetadataStore.STOP_REASON_UNAVAILABLE

    companion object {
        const val KEY_PROGRESS_RUN_ID = "run_id"
        const val KEY_PHASE = "phase"
        const val KEY_STEP = "step"
        const val KEY_MAX_STEP = "max_step"
        const val KEY_LAST_CHECKPOINT = "last_checkpoint"
        const val KEY_ERROR_CODE = "error_code"
        private const val CHANNEL_ID = "phonelm_training"
        private const val NOTIFICATION_ID = 0x504d32
        private const val NOTIFICATION_INTERVAL_MS = 1_000L
        private const val CANCEL_POLL_MS = 100L
        private val ACTIVE_PHASES = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
            TrainingPhase.PAUSED,
        )
        private val TERMINAL_PHASES = setOf(
            TrainingPhase.COMPLETED,
            TrainingPhase.ERROR,
            TrainingPhase.INTERRUPTED,
        )

        fun progressData(runId: String, state: TrainingUiState, error: String? = null): Data {
            val builder = Data.Builder()
                .putString(KEY_PROGRESS_RUN_ID, runId)
                .putString(KEY_PHASE, state.phase.name)
                .putInt(KEY_STEP, state.progress?.completedSteps ?: 0)
                .putInt(KEY_MAX_STEP, state.progress?.totalSteps ?: 0)
            state.lastCheckpoint?.uri?.let { builder.putString(KEY_LAST_CHECKPOINT, it) }
            error?.let { builder.putString(KEY_ERROR_CODE, it) }
            return builder.build()
        }

        fun createForegroundInfo(
            context: Context,
            workId: UUID,
            phase: String,
            step: Int,
            maxStep: Int,
        ): ForegroundInfo {
            context.getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "HexaTrain training", NotificationManager.IMPORTANCE_LOW),
            )
            val contentIntent = PendingIntent.getActivity(
                context,
                workId.hashCode(),
                Intent(context, StandaloneTrainingActivity::class.java),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            val cancelIntent = WorkManager.getInstance(context).createCancelPendingIntent(workId)
            val notification = Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher)
                .setContentTitle("HexaTrain training")
                .setContentText(if (maxStep > 0) "$phase — step $step / $maxStep" else phase)
                .setOnlyAlertOnce(true)
                .setOngoing(true)
                .setContentIntent(contentIntent)
                .addAction(Notification.Action.Builder(null, "Stop", cancelIntent).build())
                .apply { if (maxStep > 0) setProgress(maxStep, step, false) else setProgress(0, 0, true) }
                .build()
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else ForegroundInfo(NOTIFICATION_ID, notification)
        }
    }
}

private object TrainingWorkCoordinatorRegistry {
    private val coordinators = java.util.WeakHashMap<Any, TrainingWorkCoordinator>()

    @Synchronized
    fun get(context: Context, repository: StandaloneTrainingRepository): TrainingWorkCoordinator {
        val appContext = context.applicationContext
        return coordinators.getOrPut(appContext) {
            TrainingWorkCoordinator(appContext, repository)
        }
    }
}

fun trainingWorkCoordinator(
    context: Context,
    repository: StandaloneTrainingRepository,
): TrainingWorkCoordinator = TrainingWorkCoordinatorRegistry.get(context, repository)
