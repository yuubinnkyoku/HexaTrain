package com.yuubinnkyoku.phonelm

import java.io.Closeable
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/** Non-Android asynchronous session. Observers are invoked on the worker thread. */
class TrainingSession(
    private val backend: TrainingBackend = UnavailableTrainingBackend,
    private val clock: TrainingClock = TrainingClock { System.nanoTime() / 1_000_000L },
    private val cpuMetrics: CpuProcessMetricSource = UnavailableCpuProcessMetricSource,
    private val worker: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-training").apply { isDaemon = true }
    },
) : Closeable {
    private val lock = Any()
    private val backendOperationLock = Any()
    private var state = TrainingState()
    private var closed = false
    private var nextRunId = 0L
    private var activeRunId = 0L
    private var dashboardRecorder = TrainingDashboardRecorder(0)
    private val listeners = CopyOnWriteArrayList<TrainingStateListener>()

    /** Compatibility setter for existing tests; repositories use [subscribe]. */
    fun setListener(listener: TrainingStateListener?) {
        listeners.clear()
        if (listener != null) subscribe(listener)
    }

    fun subscribe(listener: TrainingStateListener): AutoCloseable {
        listeners += listener
        listener.onStateChanged(snapshot())
        return AutoCloseable { listeners.remove(listener) }
    }

    fun snapshot(): TrainingState = synchronized(lock) { state }

    fun resolveTotalSteps(config: TrainingModelConfig, dataset: TrainingDataset): Int? =
        backend.resolveTotalSteps(config, dataset)?.takeIf { it > 0 }

    fun canPause(): Boolean = synchronized(lock) {
        !closed && backend.supportsPause && state.phase == TrainingPhase.TRAINING
    }

    fun canResume(): Boolean = synchronized(lock) {
        !closed && backend.supportsResume && backend.hasCompatibleResumeCheckpoint && state.phase == TrainingPhase.PAUSED
    }

    fun start(request: TrainingRequest): Boolean = start(request, null)

    /**
     * Starts a run and invokes [onAccepted] before the worker is queued.  The
     * repository uses this to start the foreground notification before a very
     * fast backend can reach a terminal state.
     */
    fun start(request: TrainingRequest, onAccepted: (() -> Unit)?): Boolean =
        synchronized(backendOperationLock) { startInternal(request, onAccepted) }

    private fun startInternal(request: TrainingRequest, onAccepted: (() -> Unit)?): Boolean {
        val runId: Long
        val validationError: String?
        synchronized(lock) {
            if (closed || state.phase in activePhases) return false
            validationError = request.validationError()
            if (validationError != null) {
                state = TrainingState(TrainingPhase.ERROR, message = validationError)
                runId = 0L
            } else {
                runId = ++nextRunId
                activeRunId = runId
                val runStartStep = request.resumeFrom?.completedStep ?: 0
                dashboardRecorder = TrainingDashboardRecorder(runStartStep).apply {
                    recordPhase(TrainingPhase.PREPARING, runStartStep)
                    if (request.resumeFrom != null) recordResume(runStartStep)
                }
                val progress = TrainingProgress(runStartStep, request.totalSteps)
                state = TrainingState(
                    TrainingPhase.PREPARING,
                    progress,
                    dashboard = dashboardRecorder.snapshot(progress, null),
                )
            }
        }
        notifyListeners()
        if (validationError != null) return false
        try {
            backend.prepareForRun()
        } catch (failure: Throwable) {
            publish(
                TrainingState(
                    TrainingPhase.ERROR,
                    message = "training backend could not be prepared: " +
                        (failure.message ?: failure.javaClass.simpleName),
                ),
                runId,
            )
            return false
        }
        val acceptanceFailure = onAccepted?.let { callback ->
            runCatching { callback() }.exceptionOrNull()
        }
        if (acceptanceFailure != null) {
            runCatching { backend.cancelPreparedRun() }
            publish(
                TrainingState(
                    TrainingPhase.ERROR,
                    message = "foreground training lifetime could not be started: " +
                        (acceptanceFailure.message ?: acceptanceFailure.javaClass.simpleName),
                ),
                runId,
            )
            return false
        }
        try {
            worker.execute { execute(request, runId) }
        } catch (failure: Throwable) {
            runCatching { backend.cancelPreparedRun() }
            publish(
                TrainingState(
                    TrainingPhase.ERROR,
                    message = "training worker could not be queued: " +
                        (failure.message ?: failure.javaClass.simpleName),
                ),
                runId,
            )
            return false
        }
        return true
    }

    fun requestStop(): Boolean {
        synchronized(backendOperationLock) {
            // The native bridge resets its stop flag when the JNI call starts.
            // Do not claim that a stop was accepted while preparation or HTP
            // initialization is still in flight; only a safe training/checkpoint
            // boundary is cancellable through the current backend contract.
            val running = synchronized(lock) { !closed && state.phase in stopAcceptingPhases }
            if (running) backend.requestStop()
            return running
        }
    }

    /** WorkManager owns execution and must propagate cancellation in every active phase. */
    fun requestOwnerStop(): Boolean {
        synchronized(backendOperationLock) {
            val active = synchronized(lock) { !closed && state.phase in activePhases }
            if (active) backend.requestStop()
            return active
        }
    }

    fun pause(): Boolean {
        synchronized(backendOperationLock) {
            val runId = synchronized(lock) {
                if (closed || !backend.supportsPause || !backend.hasCompatibleResumeCheckpoint || state.phase != TrainingPhase.TRAINING) null else activeRunId
            } ?: return false
            if (!backend.pause()) return false
            return transition(runId, setOf(TrainingPhase.TRAINING), TrainingPhase.PAUSED, "paused")
        }
    }

    fun resume(): Boolean {
        synchronized(backendOperationLock) {
            val runId = synchronized(lock) {
                if (closed || !backend.supportsResume || state.phase != TrainingPhase.PAUSED) null else activeRunId
            } ?: return false
            if (!backend.resume()) return false
            return transition(runId, setOf(TrainingPhase.PAUSED), TrainingPhase.TRAINING, null)
        }
    }

    private fun execute(request: TrainingRequest, runId: Long) {
        // Activity recreation and repository close can leave a queued worker
        // behind.  Never enter JNI for a run that is no longer owned by this
        // session.
        val stillOwned = synchronized(lock) {
            !closed && activeRunId == runId && state.phase == TrainingPhase.PREPARING
        }
        if (!stillOwned) {
            runCatching { backend.cancelPreparedRun() }
            return
        }
        val startedAt = clock.elapsedRealtimeMs()
        val cpuAtStart = cpuMetrics.read()
        val accumulator = TimingAccumulator()
        val dashboard = synchronized(lock) { dashboardRecorder }
        var latestCheckpoint: TrainingCheckpointMetadata? = null
        var latestHtpWindow: HtpActivityWindow? = null
        var latestProgress: TrainingBackendProgress? = null
        var malformedProgress: String? = null
        val initializingProgress = TrainingProgress(request.resumeFrom?.completedStep ?: 0, request.totalSteps)
        dashboard.recordPhase(TrainingPhase.INITIALIZING_HTP, initializingProgress.completedSteps)
        publish(
            TrainingState(
                TrainingPhase.INITIALIZING_HTP,
                initializingProgress,
                dashboard = dashboard.snapshot(initializingProgress, null),
            ),
            runId,
        )

        val result = try {
            backend.run(request) { progress ->
                if (!acceptingProgress(runId)) return@run
                try {
                    val validation = progressValidationError(
                        progress,
                        request.totalSteps,
                        request.resumeFrom?.completedStep ?: 0,
                    )
                    if (validation != null) {
                        malformedProgress = validation
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    val trusted = progress.withTrustedTiming()
                    val checkpointError = trusted.checkpoint?.let {
                        progressCheckpointValidationError(it, request, trusted.completedSteps)
                    }
                    if (checkpointError != null) {
                        malformedProgress = checkpointError
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    if (latestProgress != null && trusted.completedSteps < latestProgress!!.completedSteps) {
                        malformedProgress = "native progress step moved backwards"
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    if (latestCheckpoint != null &&
                        trusted.checkpoint != null &&
                        trusted.checkpoint.completedStep < latestCheckpoint!!.completedStep
                    ) {
                        malformedProgress = "native checkpoint step moved backwards"
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    if (latestProgress?.phase == TrainingPhase.SAVING_CHECKPOINT &&
                        trusted.phase == TrainingPhase.TRAINING &&
                        trusted.completedSteps == latestProgress!!.completedSteps
                    ) {
                        malformedProgress = "native progress phase moved backwards"
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    latestProgress = trusted
                    latestCheckpoint = trusted.checkpoint ?: latestCheckpoint
                    // Each progress record owns its observation window. Do not
                    // carry a previous authoritative HTP ratio into a record
                    // whose timing was stripped by the fail-closed evidence gate.
                    latestHtpWindow = trusted.htpActivity
                    accumulator.add(
                        trusted.timingSample,
                        trusted.checkpointIoMs?.toDouble(),
                        trusted.timingSampleWeight,
                    )
                    val phase = trusted.phase.takeIf { it in progressPhases } ?: TrainingPhase.TRAINING
                    val timing = timingSnapshot(
                        startedAt, clock.elapsedRealtimeMs(), latestHtpWindow,
                        cpuAtStart, cpuMetrics.read(), latestProgress, accumulator,
                    )
                    val stateProgress = TrainingProgress(trusted.completedSteps, trusted.totalSteps, trusted.loss)
                    dashboard.recordProgress(trusted, timing)
                    dashboard.recordPhase(phase, trusted.completedSteps)
                    publishProgress(
                        TrainingState(
                            phase = phase,
                            progress = stateProgress,
                            timing = timing,
                            lastCheckpoint = latestCheckpoint,
                            runtimeEvidence = trusted.runtimeEvidence,
                            dashboard = dashboard.snapshot(stateProgress, timing),
                        ),
                        runId,
                    )
                } catch (error: Throwable) {
                    malformedProgress = "invalid native progress: ${error.message ?: error.javaClass.simpleName}"
                    runCatching { backend.requestStop() }
                }
            }
        } catch (error: Throwable) {
            TrainingBackendResult.Failed(error.message ?: error.javaClass.simpleName)
        }

        val effectiveResult = malformedProgress?.let { TrainingBackendResult.Failed(it) } ?: result

        val finalProgress = when (effectiveResult) {
            is TrainingBackendResult.Completed -> effectiveResult.finalProgress
            is TrainingBackendResult.Cancelled -> effectiveResult.finalProgress
            is TrainingBackendResult.Failed -> null
        }
        val finalValidationError = when (effectiveResult) {
            is TrainingBackendResult.Completed -> finalProgressValidationError(
                effectiveResult.finalProgress,
                request,
                requireHtpEvidence = true,
                runtimeEvidence = effectiveResult.runtimeEvidenceOrNull(),
                requireTargetReached = true,
            )
            is TrainingBackendResult.Cancelled -> effectiveResult.finalProgress?.let {
                finalProgressValidationError(
                    it,
                    request,
                    requireHtpEvidence = false,
                    runtimeEvidence = effectiveResult.runtimeEvidenceOrNull(),
                    requireTargetReached = false,
                )
            }
            is TrainingBackendResult.Failed -> null
        }
        val trustedFinalProgress = finalProgress
            ?.takeIf { finalValidationError == null }
            ?.withTrustedTiming(effectiveResult.runtimeEvidenceOrNull())
        val terminalResultEvidence = effectiveResult.runtimeEvidenceOrNull()
        val terminalTiming = safeTimingSnapshot(
            startedAt = startedAt,
            endedAt = runCatching { clock.elapsedRealtimeMs() }.getOrDefault(startedAt),
            // Terminal current activity is present only when the terminal
            // payload itself supplies both an observation and authoritative
            // HTP evidence. The last good window remains in dashboard history.
            htpWindow = effectiveResult.htpActivityOrNull()
                ?.takeIf { finalValidationError == null && terminalResultEvidence?.isAuthoritativelyHtp == true },
            cpuAtStart = cpuAtStart,
            cpuAtEnd = runCatching { cpuMetrics.read() }.getOrNull(),
            progress = trustedFinalProgress ?: latestProgress,
            accumulator = accumulator,
        )
        // Terminal payloads are authoritative. Never promote an earlier
        // successful progress sample when terminal evidence is missing,
        // mismatched, or explicitly failed.
        val terminalEvidence = terminalResultEvidence
        dashboard.recordRuntimeEvidence(
            trustedFinalProgress?.completedSteps ?: latestProgress?.completedSteps ?: request.resumeFrom?.completedStep ?: 0,
            terminalEvidence,
        )
        trustedFinalProgress?.let { dashboard.recordProgress(it, terminalTiming, terminalEvidence) }
        fun terminalDashboard(
            phase: TrainingPhase,
            progress: TrainingProgress?,
            error: String? = null,
        ): TrainingDashboardSnapshot {
            if (error != null) dashboard.recordError(progress?.completedSteps, error)
            dashboard.recordPhase(phase, progress?.completedSteps)
            return dashboard.snapshot(progress, terminalTiming)
        }
        when (effectiveResult) {
            is TrainingBackendResult.Completed -> {
                val invalid = finalValidationError
                if (invalid != null) {
                    publish(TrainingState(TrainingPhase.ERROR, message = invalid, timing = terminalTiming, lastCheckpoint = latestCheckpoint, runtimeEvidence = terminalEvidence, dashboard = terminalDashboard(TrainingPhase.ERROR, latestProgress?.let { TrainingProgress(it.completedSteps, it.totalSteps, it.loss) }, invalid)), runId)
                } else {
                    publish(
                        TrainingState(
                            TrainingPhase.COMPLETED,
                            TrainingProgress(effectiveResult.finalProgress.completedSteps, request.totalSteps, effectiveResult.finalProgress.loss),
                            timing = terminalTiming,
                            lastCheckpoint = effectiveResult.finalProgress.checkpoint ?: latestCheckpoint,
                            runtimeEvidence = terminalEvidence,
                            dashboard = terminalDashboard(TrainingPhase.COMPLETED, TrainingProgress(effectiveResult.finalProgress.completedSteps, request.totalSteps, effectiveResult.finalProgress.loss)),
                        ),
                        runId,
                    )
                }
            }
            is TrainingBackendResult.Cancelled -> {
                val invalid = finalValidationError
                if (invalid != null) {
                    publish(TrainingState(TrainingPhase.ERROR, message = invalid, timing = terminalTiming, lastCheckpoint = latestCheckpoint, runtimeEvidence = terminalEvidence, dashboard = terminalDashboard(TrainingPhase.ERROR, latestProgress?.let { TrainingProgress(it.completedSteps, it.totalSteps, it.loss) }, invalid)), runId)
                } else {
                    publish(
                        TrainingState(
                            TrainingPhase.INTERRUPTED,
                            effectiveResult.finalProgress?.let { TrainingProgress(it.completedSteps, request.totalSteps, it.loss) },
                            timing = terminalTiming,
                            lastCheckpoint = effectiveResult.finalProgress?.checkpoint ?: latestCheckpoint,
                            runtimeEvidence = terminalEvidence,
                            dashboard = terminalDashboard(TrainingPhase.INTERRUPTED, effectiveResult.finalProgress?.let { TrainingProgress(it.completedSteps, request.totalSteps, it.loss) }),
                        ),
                        runId,
                    )
                }
            }
            is TrainingBackendResult.Failed -> publish(
                TrainingState(
                    TrainingPhase.ERROR,
                    message = effectiveResult.reason,
                    timing = terminalTiming,
                    lastCheckpoint = latestCheckpoint,
                    runtimeEvidence = terminalEvidence,
                    dashboard = terminalDashboard(TrainingPhase.ERROR, latestProgress?.let { TrainingProgress(it.completedSteps, it.totalSteps, it.loss) }, effectiveResult.reason),
                ),
                runId,
            )
        }
    }

    private fun acceptingProgress(runId: Long): Boolean = synchronized(lock) {
        !closed && activeRunId == runId && state.phase in progressAcceptingPhases
    }

    private fun transition(
        runId: Long,
        expected: Set<TrainingPhase>,
        phase: TrainingPhase,
        message: String?,
    ): Boolean {
        val accepted = synchronized(lock) {
            if (closed || activeRunId != runId || state.phase !in expected) false
            else {
                val step = state.progress?.completedSteps
                if (phase == TrainingPhase.TRAINING && TrainingPhase.PAUSED in expected && step != null) {
                    dashboardRecorder.recordResume(step)
                }
                dashboardRecorder.recordPhase(phase, step)
                state = state.copy(
                    phase = phase,
                    message = message,
                    dashboard = dashboardRecorder.snapshot(state.progress, state.timing),
                )
                true
            }
        }
        if (accepted) notifyListeners()
        return accepted
    }

    private fun progressValidationError(
        progress: TrainingBackendProgress,
        totalSteps: Int,
        minimumCompletedSteps: Int = 0,
    ): String? = when {
        progress.totalSteps != totalSteps -> "native progress totalSteps does not match the request"
        progress.completedSteps !in 0..totalSteps -> "native progress step is outside the requested range"
        progress.completedSteps < minimumCompletedSteps -> "native progress step is before the resume checkpoint"
        progress.loss != null && !progress.loss.isFinite() -> "native progress loss is non-finite"
        progress.currentStepMs != null && progress.currentStepMs < 0L -> "native current step timing is negative"
        progress.averageStepMs != null && (!progress.averageStepMs.isFinite() || progress.averageStepMs < 0.0) -> "native average step timing is invalid"
        progress.cumulativeHtpActivityMs != null && progress.cumulativeHtpActivityMs < 0L -> "native cumulative HTP timing is negative"
        progress.htpExecuteCount != null && progress.htpExecuteCount < 0L -> "native HTP execute count is negative"
        progress.checkpointIoMs != null && progress.checkpointIoMs < 0L -> "native checkpoint I/O timing is negative"
        progress.timingSampleWeight <= 0L -> "native timing sample weight is not positive"
        else -> null
    }

    private fun finalProgressValidationError(
        progress: TrainingBackendProgress,
        request: TrainingRequest,
        requireHtpEvidence: Boolean,
        runtimeEvidence: TrainingRuntimeEvidence?,
        requireTargetReached: Boolean,
    ): String? {
        progressValidationError(
            progress,
            request.totalSteps,
            request.resumeFrom?.completedStep ?: 0,
        )?.let { return it }
        if (requireTargetReached && progress.completedSteps != request.totalSteps) {
            return "training completed before reaching the requested target step"
        }
        if (requireHtpEvidence && runtimeEvidence?.isAuthoritativelyHtp != true) {
            return "training completed without authoritative QNN HTP evidence"
        }
        progress.checkpoint?.let { checkpoint ->
            val checkpointPolicy = CheckpointFormatPolicy.forConfig(request.modelConfig)
            if (!checkpoint.finite) return "terminal checkpoint is not finite"
            if (checkpoint.modelConfig != request.modelConfig) return "terminal checkpoint model configuration differs"
            if (checkpoint.format != checkpointPolicy.format ||
                checkpoint.formatVersion != checkpointPolicy.version
            ) return "terminal checkpoint format differs"
            if (checkpoint.completedStep > progress.completedSteps) return "terminal checkpoint is ahead of progress"
            if (checkpoint.datasetIdentity == null || request.dataset.identity == null ||
                checkpoint.datasetIdentity != request.dataset.identity
            ) return "terminal checkpoint dataset identity differs"
        }
        return null
    }

    private fun progressCheckpointValidationError(
        checkpoint: TrainingCheckpointMetadata,
        request: TrainingRequest,
        completedSteps: Int,
    ): String? {
        val checkpointPolicy = CheckpointFormatPolicy.forConfig(request.modelConfig)
        if (!checkpoint.finite) return "native checkpoint is not finite"
        if (checkpoint.completedStep > completedSteps) return "native checkpoint is ahead of progress"
        if (checkpoint.modelConfig != request.modelConfig) return "native checkpoint model configuration differs"
        if (checkpoint.format != checkpointPolicy.format ||
            checkpoint.formatVersion != checkpointPolicy.version
        ) return "native checkpoint format differs"
        if (checkpoint.datasetIdentity == null || request.dataset.identity == null ||
            checkpoint.datasetIdentity != request.dataset.identity
        ) return "native checkpoint dataset identity differs"
        return null
    }

    private fun TrainingBackendProgress.withTrustedTiming(
        evidenceOverride: TrainingRuntimeEvidence? = runtimeEvidence,
    ): TrainingBackendProgress {
        val hasHtp = timingSample?.entries()?.values?.any { it.backend == TimingBackend.HTP } == true ||
            htpActivity != null || htpExecuteCount != null || cumulativeHtpActivityMs != null
        if (!hasHtp || evidenceOverride?.isAuthoritativelyHtp == true) return this
        return copy(
            htpActivity = null,
            cumulativeHtpActivityMs = null,
            htpExecuteCount = null,
            timingSample = timingSample?.withoutHtpEvidence(),
        )
    }

    private fun timingSnapshot(
        startedAt: Long,
        endedAt: Long,
        htpWindow: HtpActivityWindow?,
        cpuAtStart: CpuProcessMetrics?,
        cpuAtEnd: CpuProcessMetrics?,
        progress: TrainingBackendProgress?,
        accumulator: TimingAccumulator,
    ): TrainingTiming = TrainingTiming(
        startedAtMs = startedAt,
        endedAtMs = endedAt,
        htpActivity = htpWindow,
        cpuAtStart = cpuAtStart,
        cpuAtEnd = cpuAtEnd,
        currentStepMs = progress?.currentStepMs,
        averageStepMs = progress?.averageStepMs,
        cumulativeHtpActivityMs = progress?.cumulativeHtpActivityMs,
        htpExecuteCount = progress?.htpExecuteCount,
        checkpointIoMs = progress?.checkpointIoMs,
        aggregate = accumulator.snapshot(),
    )

    private fun safeTimingSnapshot(
        startedAt: Long,
        endedAt: Long,
        htpWindow: HtpActivityWindow?,
        cpuAtStart: CpuProcessMetrics?,
        cpuAtEnd: CpuProcessMetrics?,
        progress: TrainingBackendProgress?,
        accumulator: TimingAccumulator,
    ): TrainingTiming = runCatching {
        timingSnapshot(
            startedAt,
            endedAt.coerceAtLeast(startedAt),
            htpWindow,
            cpuAtStart,
            cpuAtEnd,
            progress,
            accumulator,
        )
    }.getOrElse {
        // A malformed terminal payload must never strand the worker in a
        // non-terminal state merely because its timing fields are unusable.
        TrainingTiming(startedAt, endedAt.coerceAtLeast(startedAt))
    }

    private fun TrainingBackendResult.htpActivityOrNull() = when (this) {
        is TrainingBackendResult.Completed -> htpActivity
        is TrainingBackendResult.Cancelled -> htpActivity
        is TrainingBackendResult.Failed -> htpActivity
    }

    private fun TrainingBackendResult.runtimeEvidenceOrNull() = when (this) {
        is TrainingBackendResult.Completed -> {
            val progressEvidence = finalProgress.runtimeEvidence
            if (runtimeEvidence != null && progressEvidence != null && runtimeEvidence != progressEvidence) null
            else runtimeEvidence ?: progressEvidence
        }
        is TrainingBackendResult.Cancelled -> {
            val progressEvidence = finalProgress?.runtimeEvidence
            if (runtimeEvidence != null && progressEvidence != null && runtimeEvidence != progressEvidence) null
            else runtimeEvidence ?: progressEvidence
        }
        is TrainingBackendResult.Failed -> runtimeEvidence
    }

    /**
     * Re-check the session phase at the publication boundary. A progress
     * callback may already be in flight while pause() changes the state; that
     * callback remains useful as history but must never overwrite PAUSED.
     */
    private fun publishProgress(next: TrainingState, runId: Long): Boolean {
        val accepted = synchronized(lock) {
            if (closed || activeRunId != runId || state.phase !in progressAcceptingPhases) false
            else {
                state = next
                true
            }
        }
        if (accepted) notifyListeners()
        return accepted
    }

    private fun publish(next: TrainingState, runId: Long? = null): Boolean {
        val accepted = synchronized(lock) {
            if (runId != null && (closed || activeRunId != runId)) false
            else if (state.isTerminal && !next.isTerminal) false
            else {
                state = next
                true
            }
        }
        if (accepted) notifyListeners()
        return accepted
    }

    private fun notifyListeners() {
        val snapshot = snapshot()
        listeners.forEach { listener -> runCatching { listener.onStateChanged(snapshot) } }
    }

    override fun close() {
        synchronized(backendOperationLock) {
            val shouldStop = synchronized(lock) {
                if (closed) false else {
                    closed = true
                    state.phase in activePhases
                }
            }
            if (shouldStop) backend.requestStop()
            worker.shutdown()
        }
    }

    private companion object {
        val activePhases = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
            TrainingPhase.PAUSED,
        )
        val progressPhases = setOf(TrainingPhase.TRAINING, TrainingPhase.SAVING_CHECKPOINT)
        val progressAcceptingPhases = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
        )
        val stopAcceptingPhases = setOf(
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
        )
    }
}

interface TrainingSelectionPersistence {
    fun loadDataset(): TrainingDataset?
    fun saveDataset(dataset: TrainingDataset)
    fun loadModelConfig(): TrainingModelConfig = ModelConfigurationCatalog.defaultConfig
    fun saveModelConfig(config: TrainingModelConfig) = Unit
}

class InMemoryTrainingSelectionPersistence : TrainingSelectionPersistence {
    private var dataset: TrainingDataset? = null
    private var modelConfig: TrainingModelConfig = ModelConfigurationCatalog.defaultConfig
    override fun loadDataset(): TrainingDataset? = dataset
    override fun saveDataset(dataset: TrainingDataset) { this.dataset = dataset }
    override fun loadModelConfig(): TrainingModelConfig = modelConfig
    override fun saveModelConfig(config: TrainingModelConfig) { modelConfig = config }
}

interface TrainingRunLifecycle {
    fun onStarted(totalSteps: Int)
    fun onStateChanged(state: TrainingState)
    fun onFinished(state: TrainingState)
}

object NoOpTrainingRunLifecycle : TrainingRunLifecycle {
    override fun onStarted(totalSteps: Int) = Unit
    override fun onStateChanged(state: TrainingState) = Unit
    override fun onFinished(state: TrainingState) = Unit
}

/**
 * Application-scoped adapter for UI reconnection. It owns a single session and
 * keeps the opaque selected URI outside the Activity lifecycle.
 */
class StandaloneTrainingRepository(
    private val session: TrainingSession = TrainingSession(),
    private val datasetStore: TrainingDatasetUriStore = InMemoryTrainingDatasetUriStore(),
    private val selectionPersistence: TrainingSelectionPersistence = InMemoryTrainingSelectionPersistence(),
    private val checkpointStore: TrainingCheckpointStore = InMemoryTrainingCheckpointStore(),
    private val runLifecycle: TrainingRunLifecycle = NoOpTrainingRunLifecycle,
) : Closeable {
    @Volatile private var latestRequest: TrainingRequest? = null
    @Volatile private var selectedDataset: TrainingDataset? = selectionPersistence.loadDataset()
    @Volatile private var selectedConfig: TrainingModelConfig = selectionPersistence.loadModelConfig()
    @Volatile private var datasetCompatibilityMessage: String? = null
    @Volatile private var resumeMessage: String? = null
    private val commandLock = Any()
    private val datasetSelectionGeneration = AtomicLong(0L)
    private val listeners = CopyOnWriteArrayList<(TrainingUiState) -> Unit>()
    private val sessionSubscription = session.subscribe { state ->
        runCatching {
            state.lastCheckpoint?.let { checkpoint ->
                checkpointStore.save(checkpoint)
            }
        }
        runCatching { runLifecycle.onStateChanged(state) }
        runCatching { publishUi() }
        if (state.isTerminal) runCatching { runLifecycle.onFinished(state) }
    }

    fun subscribe(listener: (TrainingUiState) -> Unit): AutoCloseable {
        listeners += listener
        listener(snapshot())
        return AutoCloseable { listeners.remove(listener) }
    }

    fun snapshot(): TrainingUiState = toUiState(session.snapshot())

    /** Allocates a monotonic token so a slow, stale SAF callback cannot win. */
    fun nextDatasetSelectionToken(): Long = datasetSelectionGeneration.incrementAndGet()

    fun selectDataset(dataset: TrainingDataset): Boolean =
        selectDataset(dataset, nextDatasetSelectionToken())

    fun selectDataset(dataset: TrainingDataset, selectionGeneration: Long): Boolean {
        synchronized(commandLock) {
            if (selectionGeneration < datasetSelectionGeneration.get()) return false
            if (session.snapshot().phase in activePhases) return false
            val result = datasetStore.persistReadAccess(dataset.uri, selectedConfig)
                .map { persisted -> dataset.copy(identity = dataset.identity ?: persisted.identity) }
            val selected = result.getOrNull() ?: return false
            if (selectionGeneration != datasetSelectionGeneration.get()) {
                if (selected.uri != selectedDataset?.uri) {
                    runCatching { datasetStore.releaseReadAccess(selected.uri) }
                }
                return false
            }
            val previous = selectedDataset
            selectedDataset = selected
            selectionPersistence.saveDataset(selected)
            if (previous != null && previous.uri != selected.uri) {
                runCatching { datasetStore.releaseReadAccess(previous.uri) }
            }
            resumeMessage = null
            datasetCompatibilityMessage = null
            publishUi()
            return true
        }
    }

    fun selectModelConfig(config: TrainingModelConfig): Boolean = synchronized(commandLock) {
        if (SupportedTrainingModelPolicy.validationError(config) != null) return@synchronized false
        if (session.snapshot().phase in activePhases) return@synchronized false
        selectedConfig = config
        selectionPersistence.saveModelConfig(config)
        selectedDataset?.let { dataset ->
            val reinspected = datasetStore.reinspect(dataset, config)
            selectedDataset = reinspected.getOrNull() ?: dataset.copy(identity = null)
            selectionPersistence.saveDataset(selectedDataset!!)
            datasetCompatibilityMessage = reinspected.exceptionOrNull()?.let {
                if (config.vocabularySize == 1024) {
                    "Selected cache is not compatible with V1024 byte-BPE; select an NPRTBPEV1 cache."
                } else {
                    "Selected cache is not compatible with V256 byte format; select an NPRTBYTEV1 cache."
                }
            }
        }
        resumeMessage = null
        publishUi()
        true
    }

    fun start(request: TrainingRequest): Boolean {
        synchronized(commandLock) {
            if (session.snapshot().phase in activePhases) return false
            if (datasetStore.validateReadAccess(request.dataset).isFailure) return false
            latestRequest = request
            resumeMessage = null
            return session.start(request) { runLifecycle.onStarted(request.totalSteps) }
        }
    }

    fun start(config: TrainingModelConfig, dataset: TrainingDataset, totalSteps: Int): Boolean =
        start(TrainingRequest(config, dataset, totalSteps))

    /** Uses the backend-resolved target; the UI does not own a magic 8000 value. */
    fun start(): Boolean {
        val dataset = selectedDataset ?: return false
        if (datasetStore.validateReadAccess(dataset).isFailure) return false
        val totalSteps = session.resolveTotalSteps(selectedConfig, dataset) ?: return false
        return start(TrainingRequest(selectedConfig, dataset, totalSteps))
    }

    fun pause(): Boolean = session.pause()

    fun stop(): Boolean = session.requestStop()

    fun cancelWorkerExecution(): Boolean = session.requestOwnerStop()

    fun workerIdentity(): TrainingWorkerIdentity? {
        val dataset = selectedDataset ?: return null
        val datasetIdentity = dataset.identity ?: return null
        return TrainingWorkerIdentity(
            modelConfigIdentity = selectedConfig.compatibilityKey,
            datasetIdentity = datasetIdentity,
            latestCheckpointUri = compatibleCheckpoint()?.uri,
            modelConfigEncoding = TrainingModelConfigCodec.encode(selectedConfig),
        )
    }

    fun selectedDatasetIdentity(): String? = selectedDataset?.identity

    /**
     * Starts or reattaches the execution owned by WorkManager. Recovery uses
     * checkpoint commit identity only; WorkInfo progress is never a resume point.
     */
    fun attachOrStartWorkerRun(
        mode: TrainingWorkerStartMode,
        checkpointUri: String?,
        recoverStartedRun: Boolean,
        runStartedAtMs: Long,
        runConfig: TrainingModelConfig = selectedConfig,
    ): Boolean {
        synchronized(commandLock) {
            if (session.snapshot().phase in activePhases) return true
            val dataset = selectedDataset ?: return false
            val identity = dataset.identity ?: return false
            if (datasetStore.validateReadAccess(dataset).isFailure) return false
            val plan = runCatching { TrainingPlan.forConfig(runConfig) }.getOrNull() ?: return false
            val candidates = when {
                checkpointUri != null -> checkpointStore.list().filter { it.uri == checkpointUri }
                mode == TrainingWorkerStartMode.RESUME -> checkpointStore.list()
                recoverStartedRun -> checkpointStore.list().filter { it.createdAtMs >= runStartedAtMs }
                else -> emptyList()
            }
            val selection = TrainingCheckpointCatalog.latestCompatible(
                candidates,
                runConfig,
                plan.checkpointFormat,
                plan.checkpointFormatVersion,
                identity,
                expectedTotalSteps = plan.targetSteps,
            )
            val resumeFrom = when (selection) {
                is TrainingCheckpointSelection.Selected -> {
                    if (!checkpointStore.isUsableForResume(selection.checkpoint)) {
                        resumeMessage = "Worker resume unavailable: checkpoint payload is unavailable"
                        publishUi()
                        return false
                    }
                    selection.checkpoint
                }
                is TrainingCheckpointSelection.Incompatible -> {
                    resumeMessage = "Worker resume unavailable: ${selection.reason}"
                    publishUi()
                    return false
                }
                TrainingCheckpointSelection.None -> {
                    if (mode == TrainingWorkerStartMode.RESUME) {
                        resumeMessage = "Worker resume unavailable: compatible checkpoint not found"
                        publishUi()
                        return false
                    }
                    null
                }
            }
            latestRequest = TrainingRequest(runConfig, dataset, plan.targetSteps, resumeFrom)
            resumeMessage = null
            return session.start(latestRequest!!)
        }
    }

    /** In-run pause/resume is only enabled when the native backend proves support. */
    fun resume(): Boolean {
        val dataset = selectedDataset ?: return false
        if (datasetStore.validateReadAccess(dataset).isFailure) {
            resumeMessage = "Resume unavailable: persisted dataset access is missing"
            publishUi()
            return false
        }
        if (session.canResume()) return session.resume()
        val plan = TrainingPlan.forConfig(selectedConfig)
        val selection = if (dataset.identity == null) {
            TrainingCheckpointSelection.Incompatible("dataset identity is unavailable")
        } else {
            TrainingCheckpointCatalog.latestCompatible(
                checkpointStore.list(), selectedConfig,
                plan.checkpointFormat, plan.checkpointFormatVersion, dataset.identity,
                expectedTotalSteps = plan.targetSteps,
            ).let { selection ->
                if (selection is TrainingCheckpointSelection.Selected &&
                    !checkpointStore.isUsableForResume(selection.checkpoint)
                ) {
                    TrainingCheckpointSelection.Incompatible("checkpoint payload is unavailable")
                } else selection
            }
        }
        val checkpoint = (selection as? TrainingCheckpointSelection.Selected)?.checkpoint
        if (checkpoint == null) {
            resumeMessage = when (selection) {
                is TrainingCheckpointSelection.Incompatible -> "Resume unavailable: ${selection.reason}"
                else -> "Resume unavailable: compatible checkpoint not found"
            }
            publishUi()
            return false
        }
        resumeMessage = null
        return start(TrainingRequest(selectedConfig, dataset, plan.targetSteps, TrainingResumeRequest(checkpoint).checkpoint))
    }

    /** Confirmation is owned by the Activity; old checkpoints remain archived/preserved. */
    fun startOver(): Boolean {
        val dataset = selectedDataset ?: return false
        val totalSteps = session.resolveTotalSteps(selectedConfig, dataset) ?: return false
        return start(TrainingRequest(selectedConfig, dataset, totalSteps, resumeFrom = null))
    }

    override fun close() {
        val beforeClose = session.snapshot()
        sessionSubscription.close()
        listeners.clear()
        session.close()
        if (beforeClose.phase in activePhases) {
            runCatching {
                runLifecycle.onFinished(
                    beforeClose.copy(phase = TrainingPhase.INTERRUPTED, message = "training session closed"),
                )
            }
        }
    }

    private fun publishUi() {
        val ui = snapshot()
        listeners.forEach { listener -> runCatching { listener(ui) } }
    }

    private fun toUiState(state: TrainingState): TrainingUiState {
        val compatible = compatibleCheckpoint()
        val checkpoint = state.lastCheckpoint
            ?.takeIf(::isDisplayableCheckpoint)
            ?: compatible
        val config = latestRequest?.modelConfig ?: selectedConfig
        val progress = state.progress
        val overview = buildString {
            append("${state.phase}  ")
            if (progress != null) {
                append("${progress.completedSteps} / ${progress.totalSteps}  ")
                append("${formatPercent(progress.fraction)} %  ")
                append("loss=${progress.loss?.let(::formatLoss) ?: "Unavailable"}")
            } else {
                val target = selectedDataset?.let { session.resolveTotalSteps(config, it) }
                append("step target=${target ?: "Unavailable"}")
            }
            append("\nL${config.layers} H${config.heads} T${config.tokens} D${config.dimension} ")
            append("FFN${config.feedForwardDimension} V${config.vocabularySize} B${config.batchSize} ")
            append("LR${config.learningRate} checkpoint_interval=${config.checkpointInterval}")
            (state.message ?: resumeMessage ?: datasetCompatibilityMessage)?.let { append("\nerror=$it") }
        }
        val timing = state.timing
        val timingText = formatTiming(timing)
        val activityText = formatActivity(timing?.htpActivity, timing?.aggregate)
        val modelReadinessMessage = StandaloneTrainingCapabilityPolicy.blockingError(selectedConfig)
        return TrainingUiState(
            phase = state.phase,
            progress = progress,
            overview = overview,
            overviewText = overview,
            message = state.message ?: resumeMessage ?: datasetCompatibilityMessage,
            timing = timing,
            timingText = timingText,
            htpActivity = timing?.htpActivity,
            activityText = activityText,
            lastCheckpoint = checkpoint,
            checkpointText = checkpoint?.let {
                "Latest checkpoint: step ${it.completedStep}  next=${it.completedStep + config.checkpointInterval}  " +
                    "format=${it.format}v${it.formatVersion}"
            } ?: "Latest checkpoint: Unavailable (native checkpoint integration pending)",
            datasetUri = selectedDataset?.uri,
            datasetDisplayName = selectedDataset?.displayName,
            canStart = state.phase !in activePhases && selectedDataset?.identity != null &&
                SupportedTrainingModelPolicy.validationError(selectedConfig) == null &&
                modelReadinessMessage == null,
            canStop = state.phase in stopAcceptingPhases,
            canPause = session.canPause(),
            canResume = session.canResume() ||
                (state.phase !in activePhases && compatible != null),
            modelConfig = config,
            dashboard = state.dashboard,
            selectedModelConfig = selectedConfig,
            canEditModelConfig = state.phase !in activePhases,
            modelReadinessMessage = modelReadinessMessage,
        )
    }

    private fun compatibleCheckpoint(): TrainingCheckpointMetadata? {
        val dataset = selectedDataset ?: return null
        val identity = dataset.identity ?: return null
        val plan = TrainingPlan.forConfig(selectedConfig)
        return (TrainingCheckpointCatalog.latestCompatible(
            checkpointStore.list(), selectedConfig,
            plan.checkpointFormat, plan.checkpointFormatVersion, identity,
            expectedTotalSteps = plan.targetSteps,
        ) as? TrainingCheckpointSelection.Selected)?.checkpoint?.takeIf {
            checkpointStore.isUsableForResume(it)
        }
    }

    private fun isDisplayableCheckpoint(checkpoint: TrainingCheckpointMetadata): Boolean {
        val dataset = selectedDataset ?: return false
        val plan = TrainingPlan.forConfig(selectedConfig)
        return checkpoint.finite &&
            checkpoint.modelConfig == selectedConfig &&
            checkpoint.format == plan.checkpointFormat &&
            checkpoint.formatVersion == plan.checkpointFormatVersion &&
            checkpoint.datasetIdentity != null &&
            checkpoint.datasetIdentity == dataset.identity &&
            checkpointStore.isUsableForResume(checkpoint)
    }

    private fun formatTiming(timing: TrainingTiming?): String {
        if (timing == null) return "CURRENT STEP / AVERAGE / CUMULATIVE\nUnavailable — native timing integration pending"
        val aggregate = timing.aggregate
        val current = aggregate?.current
        val average = aggregate?.average
        val cumulative = aggregate?.cumulative
        return buildString {
            append("CURRENT STEP\n")
            append(formatSample(current))
            append("\nAVERAGE / STEP\n")
            append(formatSample(average))
            append("\nCUMULATIVE\n")
            append(formatSample(cumulative))
            append("\nElapsed                  ${formatSeconds(timing.elapsedMs / 1000.0)}")
            append("\nCheckpoint I/O          ${aggregate?.checkpointIoMs?.let(::formatMs) ?: timing.checkpointIoMs?.toDouble()?.let(::formatMs) ?: "Unavailable —"}")
            append("\nHTP execute count       ${aggregate?.htpExecuteCount ?: timing.htpExecuteCount ?: "Unavailable —"}")
            append("\nHTP execute time        ${aggregate?.htpExecuteTimeMs?.let(::formatMs) ?: "Unavailable —"}")
            append("\nCPU host time            ${aggregate?.cpuHostTimeMs?.let(::formatMs) ?: "Unavailable —"}")
            append("\nCPU process              ${timing.cpuProcessPercent?.let { "%.1f %%".format(java.util.Locale.ROOT, it) } ?: "Unavailable —"}")
        }
    }

    private fun formatSample(sample: TrainingTimingSample?): String {
        if (sample == null) return "Forward        —\nBackward       —\nFwd+Backward   —\nAdam           —\nHost           —"
        fun line(label: String, value: PhaseTiming?): String {
            val backend = value?.backend?.name ?: "Unavailable"
            val ms = when (value?.backend) {
                TimingBackend.HTP -> value.qnnExecuteMs
                TimingBackend.CPU -> value.hostMs
                else -> null
            }
            return "${label.padEnd(15)}$backend  ${ms?.let(::formatMs) ?: "—"}"
        }
        return listOf(
            line("Forward", sample.forward),
            line("Backward", sample.backward),
            line("Fwd+Backward", sample.fusedForwardBackward),
            line("Adam", sample.adam),
            line("Host", sample.host),
        ).joinToString("\n")
    }

    private fun formatActivity(window: HtpActivityWindow?, aggregate: TrainingTimingAggregate?): String {
        val percent = window?.activityPercent
        return buildString {
            append("HTP activity (QNN execute / observation window)  ")
            append(percent?.let { "%.1f %%".format(java.util.Locale.ROOT, it) } ?: "Unavailable —")
            append("\nThis is not NPU, DSP, or compute-unit utilization.")
            append("\nHTP execute time=${aggregate?.htpExecuteTimeMs?.let(::formatMs) ?: "Unavailable —"}")
        }
    }

    private fun formatPercent(value: Float): String = "%.1f".format(java.util.Locale.ROOT, value * 100.0f)
    private fun formatLoss(value: Float): String = "%.4f".format(java.util.Locale.ROOT, value)
    private fun formatMs(value: Double): String = "%.2f ms".format(java.util.Locale.ROOT, value)
    private fun formatSeconds(value: Double): String = "%.1f s".format(java.util.Locale.ROOT, value)

    private companion object {
        val activePhases = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
            TrainingPhase.PAUSED,
        )
        val stopAcceptingPhases = setOf(
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
        )
    }
}

/** Process-local singleton holder; Android supplies durable adapters through [get]. */
object StandaloneTrainingRepositoryRegistry {
    private val repositories = java.util.WeakHashMap<Any, StandaloneTrainingRepository>()
    private val fallback by lazy { StandaloneTrainingRepository() }

    @Synchronized
    fun get(applicationContext: Any): StandaloneTrainingRepository {
        if (applicationContext !is android.content.Context) return fallback
        val appContext = applicationContext.applicationContext
        return repositories.getOrPut(appContext) {
            val checkpointStore = AndroidTrainingCheckpointStore(appContext)
            val backend = if (BuildConfig.PHONELM_QNN_ENABLED) {
                NativeHtpTrainingBackend(appContext, checkpointStore)
            } else {
                UnavailableTrainingBackend
            }
            StandaloneTrainingRepository(
                session = TrainingSession(
                    backend = backend,
                    cpuMetrics = AndroidCpuProcessMetricSource(),
                ),
                datasetStore = AndroidTrainingDatasetUriStore(appContext),
                selectionPersistence = AndroidTrainingSelectionPersistence(appContext),
                checkpointStore = checkpointStore,
                // Standalone GUI execution lifetime is owned by WorkManager's
                // SystemForegroundService. Benchmark notifications retain the
                // legacy service independently.
                runLifecycle = NoOpTrainingRunLifecycle,
            )
        }
    }
}
