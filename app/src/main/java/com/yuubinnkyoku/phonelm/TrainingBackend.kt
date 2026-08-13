package com.yuubinnkyoku.phonelm

data class TrainingRequest(
    val modelConfig: TrainingModelConfig,
    val dataset: TrainingDataset,
    val totalSteps: Int,
    val resumeFrom: TrainingCheckpointMetadata? = null,
) {
    init { require(totalSteps > 0) { "totalSteps must be positive" } }
    fun validationError(): String? {
        modelConfig.validationError()?.let { return it }
        if (!dataset.uri.startsWith("content://")) return "dataset must be a content:// URI"
        val checkpoint = resumeFrom ?: return null
        if (!checkpoint.finite) return "resume checkpoint is not finite"
        if (checkpoint.modelConfig != modelConfig) return "resume checkpoint model configuration differs"
        if (checkpoint.format != TrainingPlan.NICOPEDIA_L19.checkpointFormat ||
            checkpoint.formatVersion != TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion
        ) return "resume checkpoint format differs"
        if (checkpoint.completedStep <= 0 || checkpoint.completedStep >= totalSteps) {
            return "resume checkpoint step is outside the requested range"
        }
        if (checkpoint.datasetIdentity == null || dataset.identity == null ||
            checkpoint.datasetIdentity != dataset.identity
        ) return "resume checkpoint dataset identity differs"
        return null
    }
}

data class TrainingBackendProgress(
    val completedSteps: Int,
    val totalSteps: Int,
    val loss: Float? = null,
    val htpActivity: HtpActivityWindow? = null,
    val checkpoint: TrainingCheckpointMetadata? = null,
    val phase: TrainingPhase = TrainingPhase.TRAINING,
    val currentStepMs: Long? = null,
    val averageStepMs: Double? = null,
    val cumulativeHtpActivityMs: Long? = null,
    val htpExecuteCount: Long? = null,
    val checkpointIoMs: Long? = null,
    val timingSample: TrainingTimingSample? = null,
    /** Number of completed native steps represented by this throttled sample. */
    val timingSampleWeight: Long = 1L,
    /** Native evidence is kept separate from timing; missing evidence is fail-closed. */
    val runtimeEvidence: TrainingRuntimeEvidence? = null,
)

/** Evidence required before a QNN phase or successful HTP run is presented as HTP. */
data class TrainingRuntimeEvidence(
    val qnnReturnCodeSuccess: Boolean?,
    val outputTensorsFinite: Boolean?,
    val cpuFallback: Boolean?,
    val backend: String? = "HTP",
    val error: String? = null,
) {
    val isAuthoritativelyHtp: Boolean
        get() = backend.equals("HTP", ignoreCase = true) &&
            error == null && qnnReturnCodeSuccess == true && outputTensorsFinite == true && cpuFallback == false
}

sealed interface TrainingBackendResult {
    data class Completed(
        val finalProgress: TrainingBackendProgress,
        val htpActivity: HtpActivityWindow? = null,
        val runtimeEvidence: TrainingRuntimeEvidence? = null,
    ) : TrainingBackendResult
    data class Cancelled(
        val finalProgress: TrainingBackendProgress? = null,
        val htpActivity: HtpActivityWindow? = null,
        val runtimeEvidence: TrainingRuntimeEvidence? = null,
    ) : TrainingBackendResult
    data class Failed(
        val reason: String,
        val htpActivity: HtpActivityWindow? = null,
        /** Terminal evidence is optional and never inferred from an earlier successful progress sample. */
        val runtimeEvidence: TrainingRuntimeEvidence? = null,
    ) : TrainingBackendResult
}

/** JNI contract only. Implementations must separately validate QNN return codes and tensor finiteness. */
interface TrainingBackend {
    fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult
    fun requestStop()
    /** Arms cancellation for the small window before the worker enters [run]. */
    fun prepareForRun() = Unit
    /** Clears the arm when foreground acceptance/worker queueing fails. */
    fun cancelPreparedRun() = Unit
    /** Returns null when the backend cannot authoritatively determine a run length. */
    fun resolveTotalSteps(config: TrainingModelConfig, dataset: TrainingDataset): Int? =
        TrainingPlan.NICOPEDIA_L19.targetSteps.takeIf { config == TrainingPlan.NICOPEDIA_L19.modelConfig }
    val supportsPause: Boolean get() = false
    val supportsResume: Boolean get() = false
    /** True only when pause created a finite, compatible checkpoint for resume. */
    val hasCompatibleResumeCheckpoint: Boolean get() = false
    fun pause(): Boolean = false
    fun resume(): Boolean = false
}

/** Fail-closed diagnostic backend used by QNN-disabled builds and pure JVM tests. */
object UnavailableTrainingBackend : TrainingBackend {
    private const val MESSAGE = "HTP training backend is unavailable: no JNI backend has been configured"
    override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult =
        TrainingBackendResult.Failed(MESSAGE)
    override fun requestStop() = Unit
}

/** Typed JNI boundary retained for a future handle-based/reconnect API. */
interface StandaloneTrainingJniApi {
    fun startTraining(request: TrainingRequest, observer: StandaloneTrainingJniObserver): StandaloneTrainingJniHandle
    fun getTrainingState(): StandaloneTrainingJniState
}

interface StandaloneTrainingJniObserver {
    fun onProgress(progress: TrainingBackendProgress)
    fun onTerminal(result: TrainingBackendResult)
}

interface StandaloneTrainingJniHandle {
    fun stop()
    fun close()
}

data class StandaloneTrainingJniState(
    val sessionId: String?,
    val phase: TrainingPhase,
    val qnnReturnCodeSuccess: Boolean?,
    val outputTensorsFinite: Boolean?,
    val cpuFallback: Boolean?,
    val error: String? = null,
    val backend: String? = "HTP",
) {
    /** A QNN success code is never sufficient without a finite-output result. */
    val isAuthoritativelySuccessful: Boolean
        get() = backend.equals("HTP", ignoreCase = true) && error == null &&
            qnnReturnCodeSuccess == true && outputTensorsFinite == true && cpuFallback == false

    fun runtimeEvidence(): TrainingRuntimeEvidence = TrainingRuntimeEvidence(
        qnnReturnCodeSuccess = qnnReturnCodeSuccess,
        outputTensorsFinite = outputTensorsFinite,
        cpuFallback = cpuFallback,
        backend = backend,
        error = error,
    )
}
