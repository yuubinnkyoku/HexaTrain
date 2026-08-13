package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingDashboardStateTest {
    @Test fun historyReplacesDuplicateStepWithoutInventingSamples() {
        val recorder = TrainingDashboardRecorder(0)
        recorder.recordProgress(progress(0, 1f), timing(0, 10, memory = 100))
        recorder.recordProgress(progress(1, 0.5f), timing(0, 20, memory = 120))
        recorder.recordProgress(progress(1, 0.5f), timing(0, 20, memory = 120))

        val dashboard = recorder.snapshot(TrainingProgress(1, 3, 0.5f), timing(0, 20, memory = 120))
        assertEquals(2, dashboard.lossHistory.size)
        assertEquals(0.5f, dashboard.lossHistory.last().loss)
        assertEquals(2, dashboard.activityHistory.size)
        assertEquals(120L, dashboard.peakMemoryBytes)
        assertEquals(-0.5f, dashboard.lossDelta ?: Float.NaN)
    }

    @Test fun resumedEtaUsesOnlyObservedStepsAfterCheckpoint() {
        val recorder = TrainingDashboardRecorder(5)
        val progress = TrainingProgress(7, 10, 1f)
        val dashboard = recorder.snapshot(progress, timing(100, 300))

        assertEquals(5, dashboard.runStartStep)
        assertEquals(300L, dashboard.etaMs) // 200 ms / 2 observed steps * 3 remaining
        assertEquals(100.0, dashboard.averageStepWallTimeMs ?: -1.0, 0.0)
        assertNull(TrainingDashboardRecorder(5).snapshot(TrainingProgress(5, 10), timing(100, 300)).etaMs)
    }

    @Test fun eventsAreDeduplicatedAndEvidenceFieldsRemainSeparate() {
        val recorder = TrainingDashboardRecorder(0)
        val evidence = TrainingRuntimeEvidence(qnnReturnCodeSuccess = true, outputTensorsFinite = false, cpuFallback = false)
        recorder.recordPhase(TrainingPhase.TRAINING, 1)
        recorder.recordPhase(TrainingPhase.TRAINING, 2)
        recorder.recordProgress(progress(1, 1f, evidence = evidence), timing(0, 10))
        recorder.recordProgress(progress(1, 1f, evidence = evidence), timing(0, 20))
        recorder.recordError(1, "non-finite output")
        recorder.recordError(1, "non-finite output")

        val dashboard = recorder.snapshot(TrainingProgress(1, 2, 1f), timing(0, 20))
        assertEquals(true, dashboard.runtimeEvidence?.qnnReturnCodeSuccess)
        assertEquals(false, dashboard.runtimeEvidence?.outputTensorsFinite)
        assertEquals(false, dashboard.runtimeEvidence?.cpuFallback)
        assertEquals(1, dashboard.eventTimeline.count { it.type == TrainingDashboardEventType.ERROR })
        assertEquals(1, dashboard.eventTimeline.count { it.type == TrainingDashboardEventType.QNN_RETURN })
        assertEquals(1, dashboard.eventTimeline.count { it.type == TrainingDashboardEventType.PHASE })
        assertFalse(dashboard.runtimeEvidence!!.isAuthoritativelyHtp)
    }

    @Test fun checkpointCountAndMemoryPeakUseObservedValues() {
        val recorder = TrainingDashboardRecorder(0)
        val checkpoint = TrainingCheckpointMetadata(
            uri = "native-checkpoint:run:1", completedStep = 1,
            modelConfig = TrainingModelConfig.NICOPEDIA_L19,
            format = TrainingPlan.NICOPEDIA_L19.checkpointFormat,
            formatVersion = TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion,
            finite = true, createdAtMs = 1,
        )
        recorder.recordProgress(progress(1, 1f, checkpoint = checkpoint), timing(0, 10, memory = 200))
        recorder.recordProgress(progress(1, 1f, checkpoint = checkpoint), timing(0, 20, memory = 150))

        val dashboard = recorder.snapshot(TrainingProgress(1, 2, 1f), timing(0, 20, memory = 150))
        assertEquals(1, dashboard.checkpointCount)
        assertEquals(150L, dashboard.currentMemoryBytes)
        assertEquals(200L, dashboard.peakMemoryBytes)
        assertTrue(dashboard.eventTimeline.any { it.type == TrainingDashboardEventType.CHECKPOINT })
    }

    private fun progress(
        step: Int,
        loss: Float,
        evidence: TrainingRuntimeEvidence? = null,
        checkpoint: TrainingCheckpointMetadata? = null,
    ) = TrainingBackendProgress(step, 3, loss, checkpoint = checkpoint, runtimeEvidence = evidence)

    private fun timing(start: Long, end: Long, memory: Long? = null) = TrainingTiming(
        startedAtMs = start,
        endedAtMs = end,
        cpuAtStart = CpuProcessMetrics(0, memoryBytes = memory),
        cpuAtEnd = CpuProcessMetrics(10, memoryBytes = memory),
    )
}
