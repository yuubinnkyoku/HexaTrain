package com.yuubinnkyoku.phonelm

import java.io.Closeable
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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
                state = TrainingState(TrainingPhase.PREPARING, TrainingProgress(0, request.totalSteps))
            }
        }
        notifyListeners()
        if (validationError != null) return false
        val acceptanceFailure = onAccepted?.let { callback ->
            runCatching { callback() }.exceptionOrNull()
        }
        if (acceptanceFailure != null) {
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
            val running = synchronized(lock) { !closed && state.phase in activePhases }
            if (running) backend.requestStop()
            return running
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
        val startedAt = clock.elapsedRealtimeMs()
        val cpuAtStart = cpuMetrics.read()
        val accumulator = TimingAccumulator()
        var latestCheckpoint: TrainingCheckpointMetadata? = null
        var latestHtpWindow: HtpActivityWindow? = null
        var latestProgress: TrainingBackendProgress? = null
        var malformedProgress: String? = null
        publish(TrainingState(TrainingPhase.INITIALIZING_HTP, TrainingProgress(0, request.totalSteps)), runId)

        val result = try {
            backend.run(request) { progress ->
                if (!acceptingProgress(runId)) return@run
                try {
                    val validation = progressValidationError(progress, request.totalSteps)
                    if (validation != null) {
                        malformedProgress = validation
                        runCatching { backend.requestStop() }
                        return@run
                    }
                    val trusted = progress.withTrustedTiming()
                    latestProgress = trusted
                    latestCheckpoint = trusted.checkpoint ?: latestCheckpoint
                    latestHtpWindow = trusted.htpActivity ?: latestHtpWindow
                    accumulator.add(trusted.timingSample, trusted.checkpointIoMs?.toDouble())
                    val phase = trusted.phase.takeIf { it in progressPhases } ?: TrainingPhase.TRAINING
                    publish(
                        TrainingState(
                            phase = phase,
                            progress = TrainingProgress(trusted.completedSteps, trusted.totalSteps, trusted.loss),
                            timing = timingSnapshot(
                                startedAt, clock.elapsedRealtimeMs(), latestHtpWindow,
                                cpuAtStart, cpuMetrics.read(), latestProgress, accumulator,
                            ),
                            lastCheckpoint = latestCheckpoint,
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
        val terminalTiming = safeTimingSnapshot(
            startedAt = startedAt,
            endedAt = runCatching { clock.elapsedRealtimeMs() }.getOrDefault(startedAt),
            htpWindow = if (finalValidationError == null) {
                effectiveResult.htpActivityOrNull()
                    ?.takeIf { effectiveResult.runtimeEvidenceOrNull()?.isAuthoritativelyHtp == true }
                    ?: latestHtpWindow
            } else latestHtpWindow,
            cpuAtStart = cpuAtStart,
            cpuAtEnd = runCatching { cpuMetrics.read() }.getOrNull(),
            progress = trustedFinalProgress ?: latestProgress,
            accumulator = accumulator,
        )
        when (effectiveResult) {
            is TrainingBackendResult.Completed -> {
                val invalid = finalValidationError
                if (invalid != null) {
                    publish(TrainingState(TrainingPhase.ERROR, message = invalid, timing = terminalTiming, lastCheckpoint = latestCheckpoint), runId)
                } else {
                    publish(
                        TrainingState(
                            TrainingPhase.COMPLETED,
                            TrainingProgress(effectiveResult.finalProgress.completedSteps, request.totalSteps, effectiveResult.finalProgress.loss),
                            timing = terminalTiming,
                            lastCheckpoint = effectiveResult.finalProgress.checkpoint ?: latestCheckpoint,
                        ),
                        runId,
                    )
                }
            }
            is TrainingBackendResult.Cancelled -> {
                val invalid = finalValidationError
                if (invalid != null) {
                    publish(TrainingState(TrainingPhase.ERROR, message = invalid, timing = terminalTiming, lastCheckpoint = latestCheckpoint), runId)
                } else {
                    publish(
                        TrainingState(
                            TrainingPhase.INTERRUPTED,
                            effectiveResult.finalProgress?.let { TrainingProgress(it.completedSteps, request.totalSteps, it.loss) },
                            timing = terminalTiming,
                            lastCheckpoint = effectiveResult.finalProgress?.checkpoint ?: latestCheckpoint,
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
                state = state.copy(phase = phase, message = message)
                true
            }
        }
        if (accepted) notifyListeners()
        return accepted
    }

    private fun progressValidationError(progress: TrainingBackendProgress, totalSteps: Int): String? = when {
        progress.totalSteps != totalSteps -> "native progress totalSteps does not match the request"
        progress.completedSteps !in 0..totalSteps -> "native progress step is outside the requested range"
        progress.loss != null && !progress.loss.isFinite() -> "native progress loss is non-finite"
        progress.currentStepMs != null && progress.currentStepMs < 0L -> "native current step timing is negative"
        progress.averageStepMs != null && (!progress.averageStepMs.isFinite() || progress.averageStepMs < 0.0) -> "native average step timing is invalid"
        progress.cumulativeHtpActivityMs != null && progress.cumulativeHtpActivityMs < 0L -> "native cumulative HTP timing is negative"
        progress.htpExecuteCount != null && progress.htpExecuteCount < 0L -> "native HTP execute count is negative"
        progress.checkpointIoMs != null && progress.checkpointIoMs < 0L -> "native checkpoint I/O timing is negative"
        else -> null
    }

    private fun finalProgressValidationError(
        progress: TrainingBackendProgress,
        request: TrainingRequest,
        requireHtpEvidence: Boolean,
        runtimeEvidence: TrainingRuntimeEvidence?,
        requireTargetReached: Boolean,
    ): String? {
        progressValidationError(progress, request.totalSteps)?.let { return it }
        if (requireTargetReached && progress.completedSteps != request.totalSteps) {
            return "training completed before reaching the requested target step"
        }
        if (requireHtpEvidence && runtimeEvidence?.isAuthoritativelyHtp != true) {
            return "training completed without authoritative QNN HTP evidence"
        }
        progress.checkpoint?.let { checkpoint ->
            if (!checkpoint.finite) return "terminal checkpoint is not finite"
            if (checkpoint.modelConfig != request.modelConfig) return "terminal checkpoint model configuration differs"
            if (checkpoint.format != TrainingPlan.NICOPEDIA_L19.checkpointFormat ||
                checkpoint.formatVersion != TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion
            ) return "terminal checkpoint format differs"
            if (checkpoint.completedStep > progress.completedSteps) return "terminal checkpoint is ahead of progress"
            if (checkpoint.datasetIdentity == null || request.dataset.identity == null ||
                checkpoint.datasetIdentity != request.dataset.identity
            ) return "terminal checkpoint dataset identity differs"
        }
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
        is TrainingBackendResult.Failed -> null
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
    }
}

interface TrainingSelectionPersistence {
    fun loadDataset(): TrainingDataset?
    fun saveDataset(dataset: TrainingDataset)
}

class InMemoryTrainingSelectionPersistence : TrainingSelectionPersistence {
    private var dataset: TrainingDataset? = null
    override fun loadDataset(): TrainingDataset? = dataset
    override fun saveDataset(dataset: TrainingDataset) { this.dataset = dataset }
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
    @Volatile private var selectedConfig: TrainingModelConfig = TrainingModelConfig.NICOPEDIA_L19
    @Volatile private var resumeMessage: String? = null
    private val commandLock = Any()
    private val listeners = CopyOnWriteArrayList<(TrainingUiState) -> Unit>()
    private val sessionSubscription = session.subscribe { state ->
        runCatching {
            if (state.lastCheckpoint != null) checkpointStore.save(state.lastCheckpoint)
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

    fun selectDataset(dataset: TrainingDataset): Boolean {
        synchronized(commandLock) {
            if (session.snapshot().phase in activePhases) return false
            val result = datasetStore.persistReadAccess(dataset.uri)
                .map { persisted -> dataset.copy(identity = dataset.identity ?: persisted.identity) }
            val selected = result.getOrNull() ?: return false
            val previous = selectedDataset
            selectedDataset = selected
            selectionPersistence.saveDataset(selected)
            if (previous != null && previous.uri != selected.uri) {
                runCatching { datasetStore.releaseReadAccess(previous.uri) }
            }
            resumeMessage = null
            publishUi()
            return true
        }
    }

    fun selectModelConfig(config: TrainingModelConfig): Boolean {
        if (config.validationError() != null) return false
        if (session.snapshot().phase in activePhases) return false
        selectedConfig = config
        publishUi()
        return true
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

    /** In-run pause/resume is only enabled when the native backend proves support. */
    fun resume(): Boolean {
        val dataset = selectedDataset ?: return false
        if (datasetStore.validateReadAccess(dataset).isFailure) {
            resumeMessage = "Resume unavailable: persisted dataset access is missing"
            publishUi()
            return false
        }
        if (session.canResume()) return session.resume()
        val plan = TrainingPlan.NICOPEDIA_L19
        val selection = if (dataset.identity == null) {
            TrainingCheckpointSelection.Incompatible("dataset identity is unavailable")
        } else {
            TrainingCheckpointCatalog.latestCompatible(
                checkpointStore.list(), selectedConfig,
                plan.checkpointFormat, plan.checkpointFormatVersion, dataset.identity,
                expectedTotalSteps = plan.targetSteps,
            )
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
        val checkpoint = state.lastCheckpoint
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
            (state.message ?: resumeMessage)?.let { append("\nerror=$it") }
        }
        val timing = state.timing
        val timingText = formatTiming(timing)
        val activityText = formatActivity(timing?.htpActivity, timing?.aggregate)
        return TrainingUiState(
            phase = state.phase,
            progress = progress,
            overview = overview,
            overviewText = overview,
            message = state.message ?: resumeMessage,
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
            canStart = state.phase !in activePhases && selectedDataset != null,
            canStop = state.phase in activePhases,
            canPause = session.canPause(),
            canResume = session.canResume() || compatibleCheckpoint() != null,
        )
    }

    private fun compatibleCheckpoint(): TrainingCheckpointMetadata? {
        val dataset = selectedDataset ?: return null
        val identity = dataset.identity ?: return null
        val plan = TrainingPlan.NICOPEDIA_L19
        return (TrainingCheckpointCatalog.latestCompatible(
            checkpointStore.list(), selectedConfig,
            plan.checkpointFormat, plan.checkpointFormatVersion, identity,
            expectedTotalSteps = plan.targetSteps,
        ) as? TrainingCheckpointSelection.Selected)?.checkpoint
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
        if (sample == null) return "Forward        —\nBackward       —\nAdam           —\nHost           —"
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
            StandaloneTrainingRepository(
                session = TrainingSession(cpuMetrics = AndroidCpuProcessMetricSource()),
                datasetStore = AndroidTrainingDatasetUriStore(appContext),
                selectionPersistence = AndroidTrainingSelectionPersistence(appContext),
                checkpointStore = AndroidTrainingCheckpointStore(appContext),
                runLifecycle = AndroidTrainingRunLifecycle(appContext),
            )
        }
    }
}
