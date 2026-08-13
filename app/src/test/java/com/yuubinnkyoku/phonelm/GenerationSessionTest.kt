package com.yuubinnkyoku.phonelm

import java.util.concurrent.Executor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationSessionTest {
    private val checkpoint = TrainingCheckpointMetadata(
        uri = "checkpoint-8000",
        completedStep = 8_000,
        modelConfig = TrainingModelConfig.NICOPEDIA_L19,
        format = "NPRTCKPTV2",
        formatVersion = 2,
        finite = true,
        createdAtMs = 1L,
    )

    @Test fun emptyPromptDisablesGenerate() {
        val session = session(FakeBackend())
        assertFalse(session.snapshot().canGenerate)
        assertFalse(session.generate())
    }

    @Test fun greedyAndSampleControlsBuildExpectedRequest() {
        val session = session(FakeBackend())
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

    @Test fun invalidSampleParametersDisableGenerate() {
        val session = session(FakeBackend())
        session.updatePrompt("prompt")
        session.updateMode(GenerationMode.SAMPLE)
        session.updateTemperature("0")
        assertFalse(session.snapshot().canGenerate)
        session.updateTemperature("0.8")
        session.updateTopK("257")
        assertFalse(session.snapshot().canGenerate)
        session.updateTopK("32")
        assertTrue(session.snapshot().canGenerate)
    }

    @Test fun stateTransitionsIdleRunningSuccess() {
        val session = session(FakeBackend())
        val observed = mutableListOf<GenerationState>()
        session.subscribe { observed += it.execution }
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        assertTrue(observed.any { it is GenerationState.Running })
        val success = session.snapshot().execution as GenerationState.Success
        assertEquals("generated", success.result.displayText)
        assertEquals(9, success.result.byteCount)
    }

    @Test fun backendFailureBecomesFailedStateWithoutEscaping() {
        val session = session(FakeBackend(failure = IllegalStateException("QNN init failed")))
        session.updatePrompt("prompt")
        assertTrue(session.generate())
        val failed = session.snapshot().execution as GenerationState.Failed
        assertEquals("QNN init failed", failed.message)
    }

    @Test fun noCheckpointStateDisablesGenerate() {
        val backend = FakeBackend(status = GenerationCheckpointStatus.Unavailable("No trained checkpoint"))
        val session = session(backend)
        session.updatePrompt("prompt")
        assertFalse(session.snapshot().canGenerate)
        assertFalse(session.generate())
    }

    @Test fun activeTrainingRejectsGenerationBeforeBackendEntry() {
        val backend = FakeBackend()
        val session = session(backend)
        session.updatePrompt("prompt")
        assertFalse(session.generate(trainingActive = true))
        assertEquals(0, backend.generateCalls)
    }

    private fun session(backend: FakeBackend) = GenerationSession(backend, Executor { it.run() })

    private inner class FakeBackend(
        private val status: GenerationCheckpointStatus = GenerationCheckpointStatus.Available(checkpoint),
        private val failure: Throwable? = null,
    ) : GenerationBackend {
        var generateCalls = 0
        override fun checkpointStatus() = status
        override fun generate(request: GenerationRequest, checkpoint: TrainingCheckpointMetadata): GenerationResult {
            generateCalls++
            failure?.let { throw it }
            val bytes = "generated".toByteArray()
            return GenerationResult(
                generatedBytes = bytes,
                displayText = "generated",
                byteCount = bytes.size,
                elapsedMs = 12,
                backend = "HTP",
                checkpointParameterHash = "fnv1a64:test",
                cpuFallback = false,
                finite = true,
                generationGate = true,
                debugReport = "status=SUCCESS",
            )
        }
    }
}
