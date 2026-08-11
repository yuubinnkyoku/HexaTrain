package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingRepositoryTest {
    @Test fun datasetSelectionPersistsOpaqueContentUriAndNeverUsesFilesystemPath() {
        val persistence = object : TrainingSelectionPersistence {
            var value: TrainingDataset? = null
            override fun loadDataset() = value
            override fun saveDataset(dataset: TrainingDataset) { value = dataset }
        }
        val repository = StandaloneTrainingRepository(selectionPersistence = persistence)
        assertFalse(repository.start())
        assertTrue(repository.selectDataset(TrainingDataset("content://provider/cache", "nicopedia-cache.bin")))
        val state = repository.snapshot()
        assertEquals("content://provider/cache", state.datasetUri)
        assertEquals("nicopedia-cache.bin", state.datasetDisplayName)
        assertTrue(state.canStart)
        assertFalse(state.datasetUri!!.startsWith("/"))
        repository.close()

        val restored = StandaloneTrainingRepository(selectionPersistence = persistence)
        assertEquals("content://provider/cache", restored.snapshot().datasetUri)
        restored.close()
    }

    @Test fun unavailableBackendResolvesCanonicalPlanButFailsClosedWithoutFakeMetrics() {
        val repository = StandaloneTrainingRepository()
        repository.selectDataset(TrainingDataset("content://provider/cache"))
        assertTrue(repository.start())
        val deadline = System.nanoTime() + 2_000_000_000L
        while (repository.snapshot().phase in setOf(
                TrainingPhase.PREPARING,
                TrainingPhase.INITIALIZING_HTP,
                TrainingPhase.TRAINING,
                TrainingPhase.SAVING_CHECKPOINT,
            ) && System.nanoTime() < deadline
        ) {
            Thread.yield()
        }
        val state = repository.snapshot()
        assertEquals(TrainingPhase.ERROR, state.phase)
        assertTrue(state.message.orEmpty().contains("unavailable"))
        assertTrue(state.timingText.contains("Unavailable"))
        repository.close()
    }
}
