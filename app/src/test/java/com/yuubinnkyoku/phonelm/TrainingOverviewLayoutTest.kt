package com.yuubinnkyoku.phonelm

import com.yuubinnkyoku.phonelm.ui.training.TrainingOverviewDensity
import com.yuubinnkyoku.phonelm.ui.training.currentHtpObservationRatio
import com.yuubinnkyoku.phonelm.ui.training.currentProcessCpuPercent
import com.yuubinnkyoku.phonelm.ui.training.stepTargetLine
import com.yuubinnkyoku.phonelm.ui.training.TrainingToolbarPrimaryAction
import com.yuubinnkyoku.phonelm.ui.training.checkpointAgeMs
import com.yuubinnkyoku.phonelm.ui.training.latestImportantEvent
import com.yuubinnkyoku.phonelm.ui.training.summarizeLoss
import com.yuubinnkyoku.phonelm.ui.training.trainingToolbarPrimaryAction
import com.yuubinnkyoku.phonelm.ui.training.trainingOverviewDensity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingOverviewLayoutTest {
    @Test fun normalPortraitUsesStandardDensity() {
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 800, fontScale = 1f))
    }

    @Test fun shortPortraitAndLargeFontsSelectCompactDensity() {
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 700, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.2f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.3f))
        assertEquals(TrainingOverviewDensity.ACCESSIBLE, trainingOverviewDensity(heightDp = 800, fontScale = 2f))
    }

    @Test fun densityThresholdsAreStableAtPortraitBoundaries() {
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 719, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 720, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 859, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMFORTABLE, trainingOverviewDensity(heightDp = 860, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMFORTABLE, trainingOverviewDensity(heightDp = 900, fontScale = 1.05f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 900, fontScale = 1.051f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.20f))
    }

    @Test fun densityPolicyKeepsGraphVisibleAtEveryTier() {
        TrainingOverviewDensity.entries.forEach { density ->
            assertTrue(density.graphMinimumHeightDp >= 36)
            assertTrue(density.cardPaddingDp >= 6)
            assertTrue(density.heroPercentSp >= 24)
        }
    }

    @Test fun currentActivityDoesNotBackfillMissingLatestObservation() {
        val history = listOf(
            TrainingActivityHistoryEntry(step = 8, htpObservationRatioPercent = 64.0, processCpuPercent = 17.0),
            TrainingActivityHistoryEntry(step = 16, htpObservationRatioPercent = null, processCpuPercent = null),
        )

        assertNull(currentHtpObservationRatio(history))
        assertNull(currentProcessCpuPercent(history))
    }

    @Test fun toolbarPromotesPauseResumeAndStartWithoutMakingStopPrimary() {
        assertEquals(TrainingToolbarPrimaryAction.PAUSE, trainingToolbarPrimaryAction(TrainingPhase.TRAINING, false, true, true, false))
        assertEquals(TrainingToolbarPrimaryAction.RESUME, trainingToolbarPrimaryAction(TrainingPhase.PAUSED, false, true, false, true))
        assertEquals(TrainingToolbarPrimaryAction.RESUME, trainingToolbarPrimaryAction(TrainingPhase.INTERRUPTED, true, false, false, true))
        assertEquals(TrainingToolbarPrimaryAction.START_OVER, trainingToolbarPrimaryAction(TrainingPhase.INTERRUPTED, true, false, false, false))
        assertEquals(TrainingToolbarPrimaryAction.START, trainingToolbarPrimaryAction(TrainingPhase.IDLE, true, false, false, false))
    }

    @Test fun freshIdleWithoutTargetIsNotAnError() {
        // Fresh launch / dataset not selected: no target is expected, and the
        // hero must not surface "Step target unavailable".
        assertNull(stepTargetLine(TrainingPhase.IDLE, progress = null, message = null))
        assertNull(stepTargetLine(TrainingPhase.IDLE, progress = null, message = "stale"))
    }

    @Test fun runningWithoutTargetFailsClosed() {
        assertEquals(
            "Step target unavailable",
            stepTargetLine(TrainingPhase.TRAINING, progress = null, message = null),
        )
        assertEquals(
            "Step target unavailable",
            stepTargetLine(TrainingPhase.INITIALIZING_HTP, progress = null, message = null),
        )
        assertEquals(
            "Step target unavailable",
            stepTargetLine(TrainingPhase.PAUSED, progress = null, message = null),
        )
    }

    @Test fun runningWithValidTargetShowsProgress() {
        assertEquals(
            "16 / 8000 steps",
            stepTargetLine(TrainingPhase.TRAINING, TrainingProgress(16, 8_000), null),
        )
    }

    @Test fun errorWithoutTargetSurfacesRepositoryMessageInstead() {
        assertEquals(
            "HTP training backend is unavailable: no JNI backend has been configured",
            stepTargetLine(
                TrainingPhase.ERROR,
                progress = null,
                message = "HTP training backend is unavailable: no JNI backend has been configured",
            ),
        )
        assertEquals("Training error", stepTargetLine(TrainingPhase.ERROR, progress = null, message = null))
    }

    @Test fun errorWithProgressStillSurfacesRepositoryMessage() {
        // A terminal validation error retains the last progress; the error
        // itself must still take precedence over the step count.
        assertEquals(
            "HTP training backend is unavailable: no JNI backend has been configured",
            stepTargetLine(
                TrainingPhase.ERROR,
                progress = TrainingProgress(16, 8_000),
                message = "HTP training backend is unavailable: no JNI backend has been configured",
            ),
        )
    }

    @Test fun terminalStatesWithoutTargetAreNotFailClosed() {
        assertNull(stepTargetLine(TrainingPhase.COMPLETED, progress = null, message = null))
        assertNull(stepTargetLine(TrainingPhase.INTERRUPTED, progress = null, message = null))
    }

    @Test fun latestImportantEventSkipsNoisyPhaseTransitionsAndFailsClosed() {
        val events = listOf(
            TrainingDashboardEvent(TrainingDashboardEventType.CHECKPOINT, 12),
            TrainingDashboardEvent(TrainingDashboardEventType.PHASE, 13, "TRAINING"),
        )
        assertEquals(TrainingDashboardEventType.CHECKPOINT, latestImportantEvent(events)?.type)
        assertNull(latestImportantEvent(listOf(TrainingDashboardEvent(TrainingDashboardEventType.PHASE, 1, "TRAINING"))))
    }

    @Test fun failureEvidenceOutranksNewerRoutineCheckpoint() {
        val events = listOf(
            TrainingDashboardEvent(TrainingDashboardEventType.TENSOR_FINITE, 24, "false"),
            TrainingDashboardEvent(TrainingDashboardEventType.CHECKPOINT, 250, "checkpoint saved"),
        )

        assertEquals(TrainingDashboardEventType.TENSOR_FINITE, latestImportantEvent(events)?.type)
    }

    @Test fun lossSummaryUsesObservedFiniteValuesOnly() {
        val summary = summarizeLoss(listOf(TrainingLossHistoryEntry(1, 1.5f), TrainingLossHistoryEntry(2, 0.5f)))
        assertEquals(0.5f, summary.minimum)
        assertEquals(1.5f, summary.maximum)
        assertEquals(2, summary.latest?.step)
    }

    @Test fun checkpointAgeFailsClosedForInvalidOrFutureClock() {
        assertEquals(2_000L, checkpointAgeMs(5_000L, 3_000L))
        assertNull(checkpointAgeMs(3_000L, 5_000L))
        assertNull(checkpointAgeMs(-1L, 3_000L))
    }
}
