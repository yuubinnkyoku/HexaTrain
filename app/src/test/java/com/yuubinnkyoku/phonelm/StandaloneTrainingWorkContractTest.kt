package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class StandaloneTrainingWorkContractTest {
    @Test fun requestCarriesDurableIdentityAndResumeMode() {
        val identity = TrainingWorkerIdentity("config-key", "dataset-key", "checkpoint-key")
        val request = TrainingWorkCoordinator.request("run-key", identity, TrainingWorkerStartMode.RESUME)
        assertTrue(request.tags.contains(TrainingWorkCoordinator.TAG))
        assertEquals("run-key", request.workSpec.input.getString(TrainingWorkCoordinator.KEY_RUN_ID))
        assertEquals("config-key", request.workSpec.input.getString(TrainingWorkCoordinator.KEY_CONFIG_IDENTITY))
        assertEquals("dataset-key", request.workSpec.input.getString(TrainingWorkCoordinator.KEY_DATASET_IDENTITY))
        assertEquals("RESUME", request.workSpec.input.getString(TrainingWorkCoordinator.KEY_START_MODE))
        assertEquals("phonelm-standalone-training", TrainingWorkCoordinator.UNIQUE_WORK_NAME)
    }

    @Test fun ownerCancellationPropagatesDuringInitialization() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val stopped = AtomicBoolean(false)
        val backend = object : TrainingBackend {
            override fun requestStop() {
                stopped.set(true)
                release.countDown()
            }

            override fun run(
                request: TrainingRequest,
                onProgress: (TrainingBackendProgress) -> Unit,
            ): TrainingBackendResult {
                entered.countDown()
                release.await(2, TimeUnit.SECONDS)
                return TrainingBackendResult.Cancelled()
            }
        }
        val session = TrainingSession(backend)
        assertTrue(session.start(TrainingRequest(
            TrainingModelConfig.NICOPEDIA_L19,
            TrainingDataset("content://dataset"),
            2,
        )))
        assertTrue(entered.await(2, TimeUnit.SECONDS))
        assertTrue(session.requestOwnerStop())
        assertTrue(stopped.get())
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
        while (session.snapshot().phase !in setOf(TrainingPhase.INTERRUPTED, TrainingPhase.ERROR) &&
            System.nanoTime() < deadline
        ) Thread.yield()
        assertFalse(session.snapshot().phase in setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
        ))
        session.close()
    }
}
