package com.yuubinnkyoku.phonelm.ui.training

import com.yuubinnkyoku.phonelm.TrainingActivityHistoryEntry
import com.yuubinnkyoku.phonelm.TrainingDashboardEvent
import com.yuubinnkyoku.phonelm.TrainingDashboardEventType
import com.yuubinnkyoku.phonelm.TrainingLossHistoryEntry
import com.yuubinnkyoku.phonelm.TrainingPhase
import com.yuubinnkyoku.phonelm.TrainingProgress

/** Pure layout policy so portrait overview density can be verified on the JVM. */
internal enum class TrainingOverviewDensity(
    val outerPaddingDp: Int,
    val sectionGapDp: Int,
    val cardPaddingDp: Int,
    val heroPercentSp: Int,
    val graphMinimumHeightDp: Int,
) {
    ACCESSIBLE(6, 3, 7, 24, 36),
    COMPACT(8, 6, 10, 38, 54),
    STANDARD(12, 8, 12, 46, 72),
    COMFORTABLE(16, 10, 14, 52, 88),
}

internal fun trainingOverviewDensity(heightDp: Int, fontScale: Float): TrainingOverviewDensity = when {
    fontScale >= 1.60f -> TrainingOverviewDensity.ACCESSIBLE
    heightDp < 720 || fontScale >= 1.20f -> TrainingOverviewDensity.COMPACT
    heightDp < 860 || fontScale > 1.05f -> TrainingOverviewDensity.STANDARD
    else -> TrainingOverviewDensity.COMFORTABLE
}

/** The overview shows the current accepted observation; missing values are never backfilled. */
internal fun currentHtpObservationRatio(history: List<TrainingActivityHistoryEntry>): Double? =
    history.lastOrNull()?.htpObservationRatioPercent

internal fun currentProcessCpuPercent(history: List<TrainingActivityHistoryEntry>): Double? =
    history.lastOrNull()?.processCpuPercent

/**
 * Maps the session phase and target-step availability to the hero's step line.
 *
 * A missing target is normal before a run (IDLE/READY) and after a terminal
 * state; it is only fail-closed while a run is active, where progress must
 * always carry a target. ERROR surfaces the repository message instead of
 * pretending the cause is a missing target.
 */
internal fun stepTargetLine(
    phase: TrainingPhase,
    progress: TrainingProgress?,
    message: String?,
): String? = when {
    phase == TrainingPhase.ERROR -> message?.takeIf { it.isNotBlank() } ?: "Training error"
    progress != null -> "${progress.completedSteps} / ${progress.totalSteps} steps"
    phase in stepTargetRequiredPhases -> "Step target unavailable"
    else -> null
}

private val stepTargetRequiredPhases = setOf(
    TrainingPhase.PREPARING,
    TrainingPhase.INITIALIZING_HTP,
    TrainingPhase.TRAINING,
    TrainingPhase.SAVING_CHECKPOINT,
    TrainingPhase.PAUSED,
)

/** Primary action shown in the expressive toolbar for a given session state. */
internal enum class TrainingToolbarPrimaryAction {
    PAUSE,
    RESUME,
    START,
    START_OVER,
    SELECT,
    DETAILS,
}

/**
 * Maps the existing state machine to one primary action without inventing a new state.
 * Stop is intentionally never returned here: it remains a secondary/destructive action.
 */
internal fun trainingToolbarPrimaryAction(
    phase: TrainingPhase,
    canStart: Boolean,
    canStop: Boolean,
    canPause: Boolean,
    canResume: Boolean,
): TrainingToolbarPrimaryAction = when {
    phase in terminalPhasesForToolbar -> TrainingToolbarPrimaryAction.START_OVER
    phase == TrainingPhase.PAUSED && canResume -> TrainingToolbarPrimaryAction.RESUME
    canPause -> TrainingToolbarPrimaryAction.PAUSE
    canResume && phase in activePhasesForToolbar -> TrainingToolbarPrimaryAction.RESUME
    canStart -> TrainingToolbarPrimaryAction.START
    !canStop && phase !in activePhasesForToolbar -> TrainingToolbarPrimaryAction.SELECT
    else -> TrainingToolbarPrimaryAction.DETAILS
}

/** A phase transition is useful in the full history but too noisy for the overview. */
internal fun latestImportantEvent(events: List<TrainingDashboardEvent>): TrainingDashboardEvent? =
    events.withIndex()
        .mapNotNull { indexed -> eventImportance(indexed.value)?.let { Triple(it, indexed.index, indexed.value) } }
        .maxWithOrNull(compareBy<Triple<Int, Int, TrainingDashboardEvent>> { it.first }.thenBy { it.second })
        ?.third

private fun eventImportance(event: TrainingDashboardEvent): Int? = when (event.type) {
    TrainingDashboardEventType.ERROR -> 3
    TrainingDashboardEventType.QNN_RETURN -> if (event.message?.equals("true", ignoreCase = true) == false) 3 else null
    TrainingDashboardEventType.TENSOR_FINITE -> if (event.message?.equals("true", ignoreCase = true) == false) 3 else null
    TrainingDashboardEventType.CPU_FALLBACK -> if (event.message?.equals("true", ignoreCase = true) == true) 3 else null
    TrainingDashboardEventType.CHECKPOINT,
    TrainingDashboardEventType.RESUME,
    -> 2
    TrainingDashboardEventType.PHASE -> null
}

internal data class LossSummary(
    val latest: TrainingLossHistoryEntry?,
    val minimum: Float?,
    val maximum: Float?,
)

internal fun summarizeLoss(history: List<TrainingLossHistoryEntry>): LossSummary = LossSummary(
    latest = history.lastOrNull(),
    minimum = history.minOfOrNull { it.loss },
    maximum = history.maxOfOrNull { it.loss },
)

/** Returns a non-negative age while remaining fail-closed for unavailable timestamps. */
internal fun checkpointAgeMs(nowMs: Long, createdAtMs: Long): Long? =
    if (nowMs < 0L || createdAtMs < 0L || createdAtMs > nowMs) null else nowMs - createdAtMs

private val activePhasesForToolbar = setOf(
    TrainingPhase.PREPARING,
    TrainingPhase.INITIALIZING_HTP,
    TrainingPhase.TRAINING,
    TrainingPhase.SAVING_CHECKPOINT,
    TrainingPhase.PAUSED,
)

private val terminalPhasesForToolbar = setOf(
    TrainingPhase.COMPLETED,
    TrainingPhase.ERROR,
    TrainingPhase.INTERRUPTED,
)
