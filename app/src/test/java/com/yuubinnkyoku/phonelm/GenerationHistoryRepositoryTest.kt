package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GenerationHistoryRepositoryTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun rawBytesAndSettingsRoundTripWithoutUtf8Decoding() {
        val repository = repository()
        val raw = byteArrayOf(0x61, 0x80.toByte(), 0xe3.toByte(), 0x81.toByte())
        val record = record("one", 1, generated = raw, mode = GenerationMode.SAMPLE)

        assertTrue(repository.insert(record))
        val loaded = repository.listNewestFirst().single()
        assertTrue(raw.contentEquals(loaded.generatedBytes))
        assertTrue(record.promptBytes.contentEquals(loaded.promptBytes))
        assertEquals(GenerationMode.SAMPLE, loaded.mode)
        assertEquals(0.8f, loaded.temperature)
        assertEquals(32, loaded.topK)
        assertEquals(7L, loaded.samplingSeed)
        assertEquals(64, loaded.maxNewBytes)
    }

    @Test fun insertIsIdempotentAndOrderingIsNewestFirst() {
        val repository = repository()
        assertTrue(repository.insert(record("old", 10)))
        assertTrue(repository.insert(record("new", 20)))
        assertFalse(repository.insert(record("new", 20)))
        assertEquals(listOf("new", "old"), repository.listNewestFirst().map { it.id })
    }

    @Test fun deleteOneAndClearAllDoNotTouchAnythingElse() {
        val repository = repository()
        repository.insert(record("one", 1))
        repository.insert(record("two", 2))
        assertTrue(repository.delete("one"))
        assertEquals(listOf("two"), repository.listNewestFirst().map { it.id })
        repository.clear()
        assertTrue(repository.listNewestFirst().isEmpty())
    }

    private fun repository() = FileGenerationHistoryRepository(temporary.root.resolve("history.bin"))

    private fun record(
        id: String,
        createdAtMs: Long,
        generated: ByteArray = "output".toByteArray(),
        mode: GenerationMode = GenerationMode.GREEDY,
    ) = GenerationHistoryRecord(
        id = id,
        createdAtMs = createdAtMs,
        promptBytes = "prompt".toByteArray(),
        mode = mode,
        temperature = 0.8f,
        topK = 32,
        samplingSeed = 7,
        maxNewBytes = 64,
        checkpointStep = 8_000,
        checkpointParameterHash = "fnv1a64:5d1d51359d00d17a",
        vocabulary = 256,
        tokens = 32,
        dimension = 32,
        feedForwardDimension = 32,
        layers = 19,
        heads = 2,
        generatedBytes = generated,
        elapsedMs = 42,
        backend = "HTP",
        qnnExecuteAttempts = 45,
        qnnExecuteSuccesses = 45,
        qnnExecuteFailures = 0,
        cpuFallback = false,
        finite = true,
        status = GenerationHistoryStatus.SUCCESS,
    )
}
