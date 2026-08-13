package com.yuubinnkyoku.phonelm

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
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
        assertEquals(true, result.runtimeEvidence?.qnnReturnCodeSuccess)
        assertEquals(true, result.runtimeEvidence?.outputTensorsFinite)
        assertEquals(false, result.runtimeEvidence?.cpuFallback)
        assertEquals(0, result.dashboard.runStartStep)
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

    @Test fun regressingNativeProgressBecomesStructuredError() {
        val finished = CountDownLatch(1)
        val evidence = TrainingRuntimeEvidence(true, true, false)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult {
                    onProgress(TrainingBackendProgress(2, request.totalSteps, 0.4f, runtimeEvidence = evidence))
                    onProgress(TrainingBackendProgress(1, request.totalSteps, 0.5f, runtimeEvidence = evidence))
                    return TrainingBackendResult.Cancelled()
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 3)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("moved backwards"))
        session.close()
    }

    @Test fun resumedProgressCannotMoveBeforeCheckpointStep() {
        val finished = CountDownLatch(1)
        val identity = "dataset-identity"
        val checkpoint = TrainingCheckpointMetadata(
            uri = "native-checkpoint:resume:5",
            completedStep = 5,
            modelConfig = TrainingModelConfig.NICOPEDIA_L19,
            format = TrainingPlan.NICOPEDIA_L19.checkpointFormat,
            formatVersion = TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion,
            finite = true,
            createdAtMs = 1,
            datasetIdentity = identity,
        )
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    onProgress(
                        TrainingBackendProgress(
                            4,
                            request.totalSteps,
                            0.5f,
                            runtimeEvidence = TrainingRuntimeEvidence(true, true, false),
                        ),
                    )
                    return TrainingBackendResult.Cancelled()
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(
            session.start(
                TrainingRequest(
                    TrainingModelConfig.NICOPEDIA_L19,
                    TrainingDataset("content://dataset", identity = identity),
                    totalSteps = 10,
                    resumeFrom = checkpoint,
                ),
            ),
        )
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("before the resume checkpoint"))
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

    @Test fun stopIsNotAcceptedBeforeNativeTrainingBoundaryIsObservable() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val finished = CountDownLatch(1)
        var stopCalls = 0
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() { stopCalls += 1 }
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    entered.countDown()
                    release.await(2, TimeUnit.SECONDS)
                    return TrainingBackendResult.Cancelled()
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.INTERRUPTED) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(entered.await(2, TimeUnit.SECONDS))
        assertEquals(TrainingPhase.INITIALIZING_HTP, session.snapshot().phase)
        assertFalse(session.requestStop())
        assertEquals(0, stopCalls)
        release.countDown()
        assertTrue(finished.await(2, TimeUnit.SECONDS))
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

    @Test fun rejectedCurrentHtpEvidenceDoesNotReusePreviousActivityWindow() {
        val finished = CountDownLatch(1)
        val goodEvidence = TrainingRuntimeEvidence(true, true, false)
        val nonFiniteEvidence = TrainingRuntimeEvidence(true, false, false)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    onProgress(
                        TrainingBackendProgress(
                            1, request.totalSteps, 0.5f,
                            htpActivity = HtpActivityWindow(0, 10, executeDurationMs = 5.0),
                            runtimeEvidence = goodEvidence,
                        ),
                    )
                    onProgress(
                        TrainingBackendProgress(
                            2, request.totalSteps, 0.4f,
                            htpActivity = HtpActivityWindow(10, 20, executeDurationMs = 5.0),
                            runtimeEvidence = nonFiniteEvidence,
                        ),
                    )
                    return TrainingBackendResult.Cancelled()
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.INTERRUPTED) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 3)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        val terminal = session.snapshot()
        assertNull(terminal.timing?.htpActivity)
        assertNull(terminal.dashboard.activityHistory.last().htpObservationRatioPercent)
        assertNull(terminal.runtimeEvidence)
        assertEquals(false, terminal.dashboard.eventTimeline.last { it.type == TrainingDashboardEventType.TENSOR_FINITE }.message.toBoolean())
        session.close()
    }

    @Test fun failedTerminalDoesNotReuseEarlierSuccessfulEvidence() {
        val finished = CountDownLatch(1)
        val goodEvidence = TrainingRuntimeEvidence(true, true, false)
        val failedEvidence = TrainingRuntimeEvidence(false, null, false, error = "graph execute failed")
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    onProgress(TrainingBackendProgress(1, request.totalSteps, 0.5f, runtimeEvidence = goodEvidence))
                    return TrainingBackendResult.Failed("QNN execution failed", runtimeEvidence = failedEvidence)
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        val terminal = session.snapshot()
        assertNull(terminal.timing?.htpActivity)
        assertEquals(false, terminal.runtimeEvidence?.qnnReturnCodeSuccess)
        assertNull(terminal.runtimeEvidence?.outputTensorsFinite)
        assertEquals(failedEvidence, terminal.dashboard.runtimeEvidence)
        session.close()
    }

    @Test fun mismatchedCompletedTerminalDoesNotReuseEarlierSuccessfulEvidence() {
        val finished = CountDownLatch(1)
        val goodEvidence = TrainingRuntimeEvidence(true, true, false)
        val conflictingEvidence = TrainingRuntimeEvidence(true, false, false)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    onProgress(TrainingBackendProgress(1, request.totalSteps, 0.5f, runtimeEvidence = goodEvidence))
                    return TrainingBackendResult.Completed(
                        finalProgress = TrainingBackendProgress(
                            request.totalSteps,
                            request.totalSteps,
                            0.25f,
                            runtimeEvidence = goodEvidence,
                        ),
                        runtimeEvidence = conflictingEvidence,
                    )
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertNull(session.snapshot().runtimeEvidence)
        assertNull(session.snapshot().dashboard.runtimeEvidence)
        session.close()
    }

    @Test fun inFlightProgressCannotOverwritePausedState() {
        val cpuReadBlocked = CountDownLatch(1)
        val releaseCpuRead = CountDownLatch(1)
        val callbackReturned = CountDownLatch(1)
        val releaseBackend = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val reads = AtomicInteger()
        val evidence = TrainingRuntimeEvidence(true, true, false)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override val supportsPause = true
                override val supportsResume = true
                override val hasCompatibleResumeCheckpoint = true
                override fun pause() = true
                override fun resume() = true
                override fun requestStop() = Unit
                override fun run(
                    request: TrainingRequest,
                    onProgress: (TrainingBackendProgress) -> Unit,
                ): TrainingBackendResult {
                    onProgress(TrainingBackendProgress(1, request.totalSteps, 0.5f, runtimeEvidence = evidence))
                    onProgress(TrainingBackendProgress(2, request.totalSteps, 0.4f, runtimeEvidence = evidence))
                    callbackReturned.countDown()
                    releaseBackend.await(2, TimeUnit.SECONDS)
                    return TrainingBackendResult.Cancelled(runtimeEvidence = evidence)
                }
            },
            cpuMetrics = CpuProcessMetricSource {
                if (reads.incrementAndGet() == 3) {
                    cpuReadBlocked.countDown()
                    releaseCpuRead.await(2, TimeUnit.SECONDS)
                }
                CpuProcessMetrics(100)
            },
        )
        session.setListener { if (it.phase == TrainingPhase.INTERRUPTED) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 3)))
        assertTrue(cpuReadBlocked.await(2, TimeUnit.SECONDS))
        assertTrue(session.pause())
        assertEquals(TrainingPhase.PAUSED, session.snapshot().phase)
        releaseCpuRead.countDown()
        assertTrue(callbackReturned.await(2, TimeUnit.SECONDS))
        assertEquals(TrainingPhase.PAUSED, session.snapshot().phase)
        releaseBackend.countDown()
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        session.close()
    }

    @Test fun repositoryRequiresDatasetBeforeNoArgumentStart() {
        val repository = StandaloneTrainingRepository()
        assertFalse(repository.start())
        repository.selectDataset(TrainingDataset("content://dataset"))
        assertTrue(repository.snapshot().canStart)
        repository.close()
    }

    @Test fun staleHeadlessStatusIsNeverRestoredAsGuiActiveTraining() {
        // The GUI repository owns its own in-process session and must not be
        // influenced by a stale headless status.json (RUNNING + dead PID) left
        // on disk by the instrumented headless runner.
        val repository = StandaloneTrainingRepository()
        val state = repository.snapshot()
        assertEquals(TrainingPhase.IDLE, state.phase)
        assertNull(state.progress)
        assertFalse(state.canStop)
        assertFalse(state.canPause)
        repository.close()
    }

    @Test fun missingRequiredTargetDuringRunFailsClosed() {
        // An active run whose progress loses its target is malformed and must
        // never be presented as a valid run.
        val finished = CountDownLatch(1)
        val session = TrainingSession(
            backend = object : TrainingBackend {
                override fun requestStop() = Unit
                override fun run(request: TrainingRequest, onProgress: (TrainingBackendProgress) -> Unit): TrainingBackendResult {
                    onProgress(
                        TrainingBackendProgress(
                            completedSteps = 1,
                            totalSteps = 0,
                            loss = 0.5f,
                            runtimeEvidence = TrainingRuntimeEvidence(true, true, false),
                        ),
                    )
                    return TrainingBackendResult.Cancelled()
                }
            },
        )
        session.setListener { if (it.phase == TrainingPhase.ERROR) finished.countDown() }
        assertTrue(session.start(TrainingRequest(TrainingModelConfig.NICOPEDIA_L19, TrainingDataset("content://dataset"), 2)))
        assertTrue(finished.await(2, TimeUnit.SECONDS))
        assertTrue(session.snapshot().message!!.contains("totalSteps"))
        session.close()
    }
}
