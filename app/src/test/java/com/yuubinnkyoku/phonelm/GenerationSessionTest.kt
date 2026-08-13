package com.yuubinnkyoku.phonelm

import java.util.concurrent.Executor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationSessionTest {
    @Test fun latestCheckpointNonFiniteSelectsOlderFinite() {
        val repository = FakeCheckpointRepository(mutableListOf(checkpoint(8_250, finite = false), checkpoint(8_000)))
        val session = session(FakeBackend(), repository)
        assertEquals(8_000, session.snapshot().selectedCheckpoint?.step)
        assertEquals("Newer unusable checkpoint exists", session.snapshot().checkpointWarning)
    }

    @Test fun highestFiniteStepIsSelectedWithTimestampOnlyAsTieBreak() {
        val repository = FakeCheckpointRepository(
            mutableListOf(checkpoint(7_750, modified = 9_999), checkpoint(8_000, modified = 1), checkpoint(8_000, modified = 2)),
        )
        val session = session(FakeBackend(), repository)
        assertEquals("checkpoint-8000-2", session.snapshot().selectedCheckpoint?.path)
    }

    @Test fun incompatibleAndMalformedAreExcludedFromDefault() {
        val repository = FakeCheckpointRepository(
            mutableListOf(
                checkpoint(8_250, compatibility = GenerationCheckpointCompatibility.INCOMPATIBLE),
                checkpoint(0, compatibility = GenerationCheckpointCompatibility.INVALID, valid = false),
                checkpoint(7_500),
            ),
        )
        val session = session(FakeBackend(), repository)
        assertEquals(7_500, session.snapshot().selectedCheckpoint?.step)
        assertEquals(3, session.snapshot().checkpoints.size)
    }

    @Test fun userCanSelectOlderFiniteCheckpoint() {
        val repository = FakeCheckpointRepository(mutableListOf(checkpoint(8_000), checkpoint(7_500)))
        val session = session(FakeBackend(), repository)
        assertTrue(session.selectCheckpoint("checkpoint-7500-7500"))
        assertEquals(7_500, session.snapshot().selectedCheckpoint?.step)
    }

    @Test fun selectedNonFiniteCannotGenerate() {
        val repository = FakeCheckpointRepository(mutableListOf(checkpoint(8_000, finite = false)))
        val backend = FakeBackend()
        val session = session(backend, repository)
        session.updatePrompt("prompt")
        assertFalse(session.snapshot().canGenerate)
        assertFalse(session.generate())
        assertEquals(0, backend.generateCalls)
    }

    @Test fun selectedCheckpointDisappearsBeforeGenerationFailsClosed() {
        val candidate = checkpoint(8_000)
        val repository = FakeCheckpointRepository(mutableListOf(candidate))
        val backend = FakeBackend()
        val session = session(backend, repository)
        session.updatePrompt("prompt")
        repository.present = false
        assertTrue(session.generate())
        assertTrue(session.snapshot().execution is GenerationState.Failed)
        assertEquals(0, backend.generateCalls)
    }

    @Test fun emptyPromptDisablesGenerate() {
        val session = session(FakeBackend(), FakeCheckpointRepository(mutableListOf(checkpoint(8_000))))
        assertFalse(session.snapshot().canGenerate)
        assertFalse(session.generate())
    }

    @Test fun greedyAndSampleControlsBuildExpectedRequest() {
        val session = session(FakeBackend(), FakeCheckpointRepository(mutableListOf(checkpoint(8_000))))
        session.updatePrompt("こんにちは")
        assertEquals(GenerationMode.GREEDY, session.snapshot().requestOrNull()?.mode)
        session.updateMode(GenerationMode.SAMPLE)
        session.updateTemperature("0.8")
        session.updateTopK("32")
        session.updateSamplingSeed("1")
        session.updateMaxNewBytes("64")
        val request = session.snapshot().requestOrNull()!!
        assertEquals(GenerationMode.SAMPLE, request.mode)
        assertEquals(0.8f, request.temperature)
        assertEquals(32, request.topK)
        assertEquals(1L, request.samplingSeed)
        assertEquals(64, request.maxNewBytes)
        assertTrue(request.promptBytes.contentEquals("こんにちは".toByteArray(Charsets.UTF_8)))
    }

    @Test fun stateTransitionsIdleRunningSuccess() {
        val session = session(FakeBackend(), FakeCheckpointRepository(mutableListOf(checkpoint(8_000))))
        val observed = mutableListOf<GenerationState>()
        session.subscribe { observed += it.execution }
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        assertTrue(observed.any { it is GenerationState.Running })
        assertEquals("generated", (session.snapshot().execution as GenerationState.Success).result.displayText)
    }

    @Test fun backendProgressUpdatesRunningByteCount() {
        val session = session(FakeBackend(progressBytes = 37), FakeCheckpointRepository(mutableListOf(checkpoint(8_000))))
        val observed = mutableListOf<GenerationState>()
        session.subscribe { observed += it.execution }
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        val running = observed.filterIsInstance<GenerationState.Running>()
        assertTrue(running.any { it.progress.phase == GenerationPhase.CHECKPOINT_VALIDATION })
        assertTrue(running.any { it.progress.generatedBytes == 37 && it.progress.maxNewBytes == 64 })
    }

    @Test fun activeTrainingRejectsGenerationBeforeBackendEntry() {
        val backend = FakeBackend()
        val session = session(backend, FakeCheckpointRepository(mutableListOf(checkpoint(8_000))))
        session.updatePrompt("prompt")
        assertFalse(session.generate(trainingActive = true))
        assertEquals(0, backend.generateCalls)
    }

    @Test fun backendFailureBecomesFailedStateWithoutClosingSession() {
        val session = session(
            FakeBackend(failure = IllegalStateException("QNN init failed")),
            FakeCheckpointRepository(mutableListOf(checkpoint(8_000))),
        )
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        val failed = session.snapshot().execution as GenerationState.Failed
        assertEquals("QNN init failed", failed.message)
    }

    @Test fun successfulGenerationIsSavedOnceAndRecreationDoesNotDuplicate() {
        val history = FakeHistoryRepository()
        val checkpoints = FakeCheckpointRepository(mutableListOf(checkpoint(8_000)))
        val session = session(FakeBackend(), checkpoints, history)
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        assertEquals(1, history.records.size)
        assertEquals(GenerationHistoryStatus.SUCCESS, history.records.single().status)
        assertTrue("generated".toByteArray().contentEquals(history.records.single().generatedBytes))

        session(FakeBackend(), checkpoints, history)
        assertEquals(1, history.records.size)
    }

    @Test fun failedGenerationIsSavedOnce() {
        val history = FakeHistoryRepository()
        val session = session(
            FakeBackend(failure = IllegalStateException("QNN init failed")),
            FakeCheckpointRepository(mutableListOf(checkpoint(8_000))),
            history,
        )
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        assertEquals(1, history.records.size)
        assertEquals(GenerationHistoryStatus.FAILED, history.records.single().status)
        assertEquals("QNN init failed", history.records.single().failureMessage)
    }

    @Test fun useSettingsAgainRestoresSettingsAndOriginalCheckpoint() {
        val original = checkpoint(8_000)
        val history = FakeHistoryRepository(mutableListOf(historyRecord(original.parameterHash!!)))
        val session = session(FakeBackend(), FakeCheckpointRepository(mutableListOf(original)), history)

        assertTrue(session.useHistoryAgain("history"))
        val state = session.snapshot()
        assertEquals("saved prompt", state.prompt)
        assertEquals(GenerationMode.SAMPLE, state.mode)
        assertEquals("0.7", state.temperatureText)
        assertEquals("12", state.topKText)
        assertEquals("99", state.samplingSeedText)
        assertEquals("48", state.maxNewBytesText)
        assertEquals(original.path, state.selectedCheckpointPath)
        assertEquals(null, state.checkpointWarning)
    }

    @Test fun missingOriginalCheckpointWarnsAndDoesNotSilentlySubstitute() {
        val history = FakeHistoryRepository(mutableListOf(historyRecord("fnv1a64:1111111111111111")))
        val session = session(
            FakeBackend(),
            FakeCheckpointRepository(mutableListOf(checkpoint(8_250))),
            history,
        )

        assertTrue(session.useHistoryAgain("history"))
        assertEquals(null, session.snapshot().selectedCheckpointPath)
        assertEquals("Original checkpoint is no longer available", session.snapshot().checkpointWarning)
        session.refreshCheckpoints()
        assertEquals(null, session.snapshot().selectedCheckpointPath)
    }

    private fun session(
        backend: FakeBackend,
        repository: FakeCheckpointRepository,
        history: FakeHistoryRepository = FakeHistoryRepository(),
    ) = GenerationSession(
        backend = backend,
        checkpointRepository = repository,
        historyRepository = history,
        losslessByteDisplay = LosslessByteDisplay { String(it, Charsets.ISO_8859_1) },
        executor = Executor { it.run() },
    )

    private fun checkpoint(
        step: Int,
        finite: Boolean = true,
        compatibility: GenerationCheckpointCompatibility = GenerationCheckpointCompatibility.COMPATIBLE,
        valid: Boolean = true,
        modified: Long = step.toLong(),
    ) = GenerationCheckpoint(
        path = "checkpoint-$step-$modified",
        step = step,
        seed = 1,
        tokens = if (compatibility == GenerationCheckpointCompatibility.INCOMPATIBLE) 16 else 32,
        dimension = 32,
        feedForwardDimension = 32,
        layers = 19,
        heads = 2,
        finite = finite,
        parameterHash = if (valid) "fnv1a64:5d1d51359d00d17a" else null,
        modifiedAtMs = modified,
        fileSizeBytes = 100,
        compatibility = compatibility,
        formatValid = valid,
        diagnostic = if (valid) null else "malformed",
    )

    private fun historyRecord(hash: String) = GenerationHistoryRecord(
        id = "history",
        createdAtMs = 1,
        promptBytes = "saved prompt".toByteArray(),
        mode = GenerationMode.SAMPLE,
        temperature = 0.7f,
        topK = 12,
        samplingSeed = 99,
        maxNewBytes = 48,
        checkpointStep = 8_000,
        checkpointParameterHash = hash,
        vocabulary = 256,
        tokens = 32,
        dimension = 32,
        feedForwardDimension = 32,
        layers = 19,
        heads = 2,
        generatedBytes = "saved output".toByteArray(),
        elapsedMs = 10,
        backend = "HTP",
        qnnExecuteAttempts = 2,
        qnnExecuteSuccesses = 2,
        qnnExecuteFailures = 0,
        cpuFallback = false,
        finite = true,
        status = GenerationHistoryStatus.SUCCESS,
    )

    private class FakeCheckpointRepository(
        val checkpoints: MutableList<GenerationCheckpoint>,
    ) : GenerationCheckpointRepository {
        var present = true
        override fun listCheckpoints() = if (present) checkpoints.toList() else emptyList()
        override fun defaultCheckpoint(checkpoints: List<GenerationCheckpoint>) = GenerationCheckpointSelection.default(checkpoints)
        override fun validateCheckpoint(checkpoint: GenerationCheckpoint): Result<GenerationCheckpoint> = runCatching {
            require(present && checkpoints.any { it.path == checkpoint.path }) { "checkpoint file is missing" }
            require(checkpoint.usable) { "checkpoint is not usable" }
            checkpoint
        }
    }

    private class FakeHistoryRepository(
        val records: MutableList<GenerationHistoryRecord> = mutableListOf(),
    ) : GenerationHistoryRepository {
        override fun insert(record: GenerationHistoryRecord): Boolean {
            if (records.any { it.id == record.id }) return false
            records += record
            return true
        }

        override fun listNewestFirst() = records.sortedByDescending { it.createdAtMs }
        override fun delete(id: String) = records.removeAll { it.id == id }
        override fun clear() = records.clear()
    }

    private class FakeBackend(
        private val failure: Throwable? = null,
        private val progressBytes: Int? = null,
    ) : GenerationBackend {
        var generateCalls = 0
        override fun generate(
            request: GenerationRequest,
            checkpoint: GenerationCheckpoint,
            onProgress: (GenerationProgress) -> Unit,
        ): GenerationResult {
            generateCalls++
            failure?.let { throw it }
            progressBytes?.let {
                onProgress(
                    GenerationProgress(
                        GenerationPhase.GENERATING,
                        generatedBytes = it,
                        maxNewBytes = request.maxNewBytes,
                    ),
                )
            }
            val bytes = "generated".toByteArray()
            return GenerationResult(
                bytes, "generated", bytes.size, 12, "HTP", checkpoint.parameterHash!!,
                false, true, true, "status=SUCCESS",
            )
        }
    }
}
