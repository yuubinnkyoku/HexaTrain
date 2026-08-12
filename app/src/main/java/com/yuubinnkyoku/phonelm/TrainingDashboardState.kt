package com.yuubinnkyoku.phonelm

/** One observed loss value. Values are never interpolated for presentation. */
data class TrainingLossHistoryEntry(val step: Int, val loss: Float) {
    init {
        require(step >= 0) { "loss history step must be non-negative" }
        require(loss.isFinite()) { "loss history loss must be finite" }
    }
}

/** Process and QNN observations captured at a real progress boundary. */
data class TrainingActivityHistoryEntry(
    val step: Int,
    val htpObservationRatioPercent: Double? = null,
    val processCpuPercent: Double? = null,
    val processMemoryBytes: Long? = null,
) {
    init {
        require(step >= 0) { "activity history step must be non-negative" }
        listOf(htpObservationRatioPercent, processCpuPercent).forEach {
            require(it == null || it.isFinite() && it >= 0.0) { "activity percentage must be finite and non-negative" }
        }
        require(processMemoryBytes == null || processMemoryBytes >= 0L) {
            "process memory must be non-negative"
        }
    }
}

enum class TrainingDashboardEventType {
    PHASE, CHECKPOINT, RESUME, ERROR, QNN_RETURN, TENSOR_FINITE, CPU_FALLBACK,
}

/** Human-readable event with no raw path, checkpoint payload, or device identifier. */
data class TrainingDashboardEvent(
    val type: TrainingDashboardEventType,
    val step: Int? = null,
    val message: String? = null,
) {
    init { require(step == null || step >= 0) { "event step must be non-negative" } }
}

/**
 * Immutable application-scoped dashboard projection. Null means no observation
 * is available; it is deliberately not substituted with an estimate or zero.
 */
data class TrainingDashboardSnapshot(
    val lossHistory: List<TrainingLossHistoryEntry> = emptyList(),
    val activityHistory: List<TrainingActivityHistoryEntry> = emptyList(),
    val eventTimeline: List<TrainingDashboardEvent> = emptyList(),
    val lossDelta: Float? = null,
    val etaMs: Long? = null,
    val currentStepWallTimeMs: Long? = null,
    val averageStepWallTimeMs: Double? = null,
    val checkpointCount: Int = 0,
    val currentMemoryBytes: Long? = null,
    val peakMemoryBytes: Long? = null,
    val runtimeEvidence: TrainingRuntimeEvidence? = null,
    val runStartStep: Int = 0,
) {
    init {
        require(lossDelta == null || lossDelta.isFinite()) { "loss delta must be finite" }
        require(etaMs == null || etaMs >= 0L) { "ETA must be non-negative" }
        require(currentStepWallTimeMs == null || currentStepWallTimeMs >= 0L) { "current step time must be non-negative" }
        require(averageStepWallTimeMs == null || averageStepWallTimeMs.isFinite() && averageStepWallTimeMs >= 0.0) {
            "average step time must be finite and non-negative"
        }
        require(checkpointCount >= 0) { "checkpoint count must be non-negative" }
        require(currentMemoryBytes == null || currentMemoryBytes >= 0L) { "current memory must be non-negative" }
        require(peakMemoryBytes == null || peakMemoryBytes >= 0L) { "peak memory must be non-negative" }
        require(runStartStep >= 0) { "run start step must be non-negative" }
    }
}

/** Thread-safe session recorder; all exposed values are immutable snapshots. */
class TrainingDashboardRecorder(
    private val runStartStep: Int,
    private val historyLimit: Int = DEFAULT_HISTORY_LIMIT,
    private val eventLimit: Int = DEFAULT_EVENT_LIMIT,
) {
    private val losses = ArrayDeque<TrainingLossHistoryEntry>()
    private val activities = ArrayDeque<TrainingActivityHistoryEntry>()
    private val events = ArrayDeque<TrainingDashboardEvent>()
    private var checkpointCount = 0
    private val checkpointKeys = mutableSetOf<Pair<String, Int>>()
    private var currentMemoryBytes: Long? = null
    private var peakMemoryBytes: Long? = null
    private var runtimeEvidence: TrainingRuntimeEvidence? = null
    private var lastPhase: TrainingPhase? = null

    init {
        require(runStartStep >= 0) { "run start step must be non-negative" }
        require(historyLimit > 0 && eventLimit > 0) { "dashboard limits must be positive" }
    }

    @Synchronized
    fun recordProgress(
        progress: TrainingBackendProgress,
        timing: TrainingTiming,
        evidence: TrainingRuntimeEvidence? = progress.runtimeEvidence,
    ) {
        progress.loss?.let { addLoss(TrainingLossHistoryEntry(progress.completedSteps, it)) }
        val memory = timing.cpuAtEnd?.memoryBytes
        currentMemoryBytes = memory
        if (memory != null) peakMemoryBytes = maxOf(peakMemoryBytes ?: memory, memory)
        addActivity(
            TrainingActivityHistoryEntry(
                step = progress.completedSteps,
                htpObservationRatioPercent = timing.htpActivity?.activityPercent,
                processCpuPercent = timing.cpuProcessPercent,
                processMemoryBytes = memory,
            ),
        )
        recordRuntimeEvidence(progress.completedSteps, evidence)
        progress.checkpoint?.let { recordCheckpoint(it) }
    }

    @Synchronized
    fun recordPhase(phase: TrainingPhase, step: Int?) {
        if (phase == lastPhase) return
        lastPhase = phase
        addEvent(TrainingDashboardEvent(TrainingDashboardEventType.PHASE, step, phase.name))
    }

    @Synchronized
    fun recordResume(step: Int) =
        addEvent(TrainingDashboardEvent(TrainingDashboardEventType.RESUME, step, "resumed"))

    @Synchronized
    fun recordError(step: Int?, message: String?) =
        addEvent(TrainingDashboardEvent(TrainingDashboardEventType.ERROR, step, message))

    @Synchronized
    fun recordRuntimeEvidence(step: Int, evidence: TrainingRuntimeEvidence?) {
        if (evidence == runtimeEvidence) return
        val previous = runtimeEvidence
        runtimeEvidence = evidence
        if (evidence == null) return
        evidence.qnnReturnCodeSuccess?.takeIf { it != previous?.qnnReturnCodeSuccess }?.let {
            addEvent(TrainingDashboardEvent(TrainingDashboardEventType.QNN_RETURN, step, it.toString()))
        }
        evidence.outputTensorsFinite?.takeIf { it != previous?.outputTensorsFinite }?.let {
            addEvent(TrainingDashboardEvent(TrainingDashboardEventType.TENSOR_FINITE, step, it.toString()))
        }
        evidence.cpuFallback?.takeIf { it != previous?.cpuFallback }?.let {
            addEvent(TrainingDashboardEvent(TrainingDashboardEventType.CPU_FALLBACK, step, it.toString()))
        }
    }

    @Synchronized
    fun recordCheckpoint(checkpoint: TrainingCheckpointMetadata) {
        if (checkpointKeys.add(checkpoint.uri to checkpoint.completedStep)) {
            checkpointCount += 1
            addEvent(TrainingDashboardEvent(TrainingDashboardEventType.CHECKPOINT, checkpoint.completedStep, "checkpoint saved"))
        }
    }

    @Synchronized
    fun snapshot(progress: TrainingProgress?, timing: TrainingTiming?): TrainingDashboardSnapshot {
        val previous = losses.dropLast(1).lastOrNull()?.loss
        val latest = losses.lastOrNull()?.loss
        val elapsed = timing?.elapsedMs
        val completed = progress?.completedSteps
        val eta = if (elapsed != null && completed != null) {
            val observedSteps = completed - runStartStep
            val remainingSteps = (progress.totalSteps - completed).coerceAtLeast(0)
            if (observedSteps > 0 && elapsed > 0L) (elapsed.toDouble() / observedSteps * remainingSteps).toLong() else null
        } else null
        return TrainingDashboardSnapshot(
            lossHistory = losses.toList(), activityHistory = activities.toList(), eventTimeline = events.toList(),
            lossDelta = if (previous != null && latest != null) latest - previous else null,
            etaMs = eta,
            currentStepWallTimeMs = timing?.currentStepMs,
            averageStepWallTimeMs = timing?.averageStepMs ?: elapsed?.let { elapsedMs ->
                completed?.let { completedSteps ->
                    val observedSteps = completedSteps - runStartStep
                    if (observedSteps > 0 && elapsedMs > 0L) elapsedMs.toDouble() / observedSteps else null
                }
            },
            checkpointCount = checkpointCount,
            currentMemoryBytes = currentMemoryBytes,
            peakMemoryBytes = peakMemoryBytes,
            runtimeEvidence = runtimeEvidence,
            runStartStep = runStartStep,
        )
    }

    private fun addLoss(value: TrainingLossHistoryEntry) = addDeduplicated(losses, value, historyLimit) { it.step }
    private fun addActivity(value: TrainingActivityHistoryEntry) = addDeduplicated(activities, value, historyLimit) { it.step }
    private fun addEvent(value: TrainingDashboardEvent) = addDeduplicated(events, value, eventLimit) { it }

    private fun <T, K> addDeduplicated(target: ArrayDeque<T>, value: T, limit: Int, key: (T) -> K) {
        val existing = target.indexOfFirst { key(it) == key(value) }
        if (existing >= 0) target.remove(target.elementAt(existing))
        target.addLast(value)
        while (target.size > limit) target.removeFirst()
    }

    private companion object {
        const val DEFAULT_HISTORY_LIMIT = 512
        const val DEFAULT_EVENT_LIMIT = 256
    }
}
