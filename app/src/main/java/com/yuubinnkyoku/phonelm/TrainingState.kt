package com.yuubinnkyoku.phonelm

enum class TrainingPhase {
    IDLE, PREPARING, INITIALIZING_HTP, TRAINING, SAVING_CHECKPOINT, PAUSED,
    COMPLETED, ERROR, INTERRUPTED,
}

data class TrainingProgress(val completedSteps: Int, val totalSteps: Int, val loss: Float? = null) {
    init {
        require(totalSteps > 0) { "totalSteps must be positive" }
        require(completedSteps in 0..totalSteps) { "completedSteps must be within totalSteps" }
        require(loss == null || loss.isFinite()) { "loss must be finite when present" }
    }
    val fraction: Float get() = completedSteps.toFloat() / totalSteps
}

data class TrainingState(
    val phase: TrainingPhase = TrainingPhase.IDLE,
    val progress: TrainingProgress? = null,
    val message: String? = null,
    val timing: TrainingTiming? = null,
    val lastCheckpoint: TrainingCheckpointMetadata? = null,
    /** Kept separate so UI never conflates QNN return success with tensor finiteness. */
    val runtimeEvidence: TrainingRuntimeEvidence? = null,
    val dashboard: TrainingDashboardSnapshot = TrainingDashboardSnapshot(),
) {
    val isTerminal: Boolean get() = phase in setOf(TrainingPhase.COMPLETED, TrainingPhase.ERROR, TrainingPhase.INTERRUPTED)
}

/** Presentation-ready projection that does not expose Android or JNI types. */
data class TrainingUiState(
    val phase: TrainingPhase,
    val progress: TrainingProgress?,
    val overview: String,
    val overviewText: String,
    val message: String?,
    val timing: TrainingTiming?,
    val timingText: String,
    val htpActivity: HtpActivityWindow?,
    val activityText: String,
    val lastCheckpoint: TrainingCheckpointMetadata?,
    val checkpointText: String,
    val datasetUri: String?,
    val datasetDisplayName: String?,
    val canStart: Boolean,
    val canStop: Boolean,
    val canPause: Boolean,
    val canResume: Boolean,
    val modelConfig: TrainingModelConfig,
    val dashboard: TrainingDashboardSnapshot,
)

fun interface TrainingStateListener {
    fun onStateChanged(state: TrainingState)
}
