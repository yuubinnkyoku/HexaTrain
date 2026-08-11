package com.yuubinnkyoku.phonelm

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingSessionTest {
    @Test fun sessionRunsAsynchronouslyAndPublishesStructuredTiming() {
        val finished = CountDownLatch(1)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult {
                    val evidence = TrainingRuntimeEvidence(true, true, false)
                    onProgress(
                        TrainingBackendProgress(
                            1,
                            request.totalSteps,
                            0.5f,
                            HtpActivityWindow(11, 14),
                            runtimeEvidence = evidence,
                        ),
                    )
                    return TrainingBackendResult.Completed(
                        TrainingBackendProgress(
                            request.totalSteps,
                            request.totalSteps,
                            0.25f,
                            runtimeEvidence = evidence,
                        ),
                        HtpActivityWindow(11, 20),
                        evidence,
                    )
                }
            },
            clock = object : TrainingClock { private var value = 10L; override fun elapsedRealtimeMs() = value++ },
            cpuMetrics = CpuProcessMetricSource { CpuProcessMetrics(100) },
        )
        session.setListener { if (it.phase == TrainingPhase.COMPLETED) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        val result = session.snapshot()
        assertEquals(TrainingPhase.COMPLETED, result.phase)
        assertEquals(2, result.progress?.completedSteps)
        assertEquals(9L, result.timing?.htpActivity?.durationMs)
        assertEquals(0L, result.timing?.cpuProcessDeltaMs)
        session.close()
    }

    @Test fun unavailableBackendIsExplicitFailureNotAFakeRun() {
        val finished = CountDownLatch(1)
        val session = TrainingSession()
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 1)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("unavailable"))
        session.close()
    }

    @Test fun malformedTerminalProgressBecomesStructuredError() {
        val finished = CountDownLatch(1)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult =
                    TrainingBackendResult.Completed(
                        TrainingBackendProgress(-1, request.totalSteps, 0.1f),
                        runtimeEvidence = TrainingRuntimeEvidence(true, true, false),
                    )
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("outside the requested range"))
        session.close()
    }

    @Test fun invalidTerminalTimingIsRejectedBeforeTimingSnapshot() {
        val finished = CountDownLatch(1)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult =
                    TrainingBackendResult.Completed(
                        TrainingBackendProgress(
                            request.totalSteps,
                            request.totalSteps,
                            0.1f,
                            currentStepMs = -1L,
                            runtimeEvidence = TrainingRuntimeEvidence(true, true, false),
                        ),
                        runtimeEvidence = TrainingRuntimeEvidence(true, true, false),
                    )
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("timing is negative"))
        session.close()
    }

    @Test fun completedResultMustReachTargetStep() {
        val finished = CountDownLatch(1)
        val evidence = TrainingRuntimeEvidence(true, true, false)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult =
                    TrainingBackendResult.Completed(
                        TrainingBackendProgress(1, request.totalSteps, 0.1f, runtimeEvidence = evidence),
                        runtimeEvidence = evidence,
                    )
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("before reaching"))
        session.close()
    }

    @Test fun foregroundAcceptanceFailureStopsBeforeWorkerIsQueued() {
        val finished = CountDownLatch(1)
        var ran = false
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult {
                    ran = true
                    return TrainingBackendResult.Failed("must not run")
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertFalse(
            session.start(
                TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2),
            ) { error("foreground unavailable") },
        )
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertFalse(ran)
        assertTrue(session.snapshot().message!!.contains("foreground training lifetime"))
        session.close()
    }

    @Test fun htpTimingWithoutEvidenceIsNotPresentedAsHtp() {
        val finished = CountDownLatch(1)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult {
                    onProgress(
                        TrainingBackendProgress(
                            1,
                            request.totalSteps,
                            0.5f,
                            timingSample = TrainingTimingSample(
                                forward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 10.0, qnnExecuteCount = 1),
                            ),
                        ),
                    )
                    val evidence = TrainingRuntimeEvidence(true, true, false)
                    return TrainingBackendResult.Completed(
                        TrainingBackendProgress(request.totalSteps, request.totalSteps, 0.2f, runtimeEvidence = evidence),
                        runtimeEvidence = evidence,
                    )
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.COMPLETED) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertEquals(TimingBackend.UNAVAILABLE, session.snapshot().timing?.aggregate?.average?.forward?.backend)
        session.close()
    }

    @Test fun repositoryRequiresDatasetBeforeNoArgumentStart() {
        val repository = StandaloneTrainingRepository()
        assertFalse(repository.start())
        repository.selectDataset(TrainingDataset("content://dataset"))
        assertTrue(repository.snapshot().canStart)
        repository.close()
    }
}
