package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingRepositoryTest {
    @Test fun modelSelectionPersistsAcrossRepositoryRecreation() {
        val persistence = InMemoryTrainingSelectionPersistence()
        val selected = ModelConfigurationCatalog.config(256, 64, 48)
        StandaloneTrainingRepository(selectionPersistence = persistence).use { repository ->
            assertTrue(repository.selectModelConfig(selected))
            assertEquals(selected, repository.snapshot().selectedModelConfig)
        }
        StandaloneTrainingRepository(selectionPersistence = persistence).use { restored ->
            assertEquals(selected, restored.snapshot().selectedModelConfig)
            assertTrue(restored.snapshot().canEditModelConfig)
        }
    }

    @Test fun v1024SelectionIsVisibleButTrainingStartIsCapabilityBlocked() {
        val repository = StandaloneTrainingRepository()
        assertTrue(repository.selectModelConfig(ModelConfigurationCatalog.config(1024, 64, 64)))
        assertTrue(repository.selectDataset(TrainingDataset("content://provider/bpe")))
        val state = repository.snapshot()
        assertFalse(state.canStart)
        assertTrue(state.modelReadinessMessage.orEmpty().contains("tokenizer model"))
        repository.close()
    }

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

    @Test fun persistedCheckpointIsVisibleAfterRepositoryRecreation() {
        val identity = "NPRTBYTEV1;context=32;vocab=256;records=1;bytes=70;sha256=test"
        val checkpoint = TrainingCheckpointMetadata(
            uri = "native-checkpoint:run:250",
            completedStep = 250,
            modelConfig = TrainingPlan.NICOPEDIA_L19.modelConfig,
            format = TrainingPlan.NICOPEDIA_L19.checkpointFormat,
            formatVersion = TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion,
            finite = true,
            createdAtMs = 1L,
            datasetIdentity = identity,
        )
        val checkpointStore = InMemoryTrainingCheckpointStore().apply { save(checkpoint) }
        val repository = StandaloneTrainingRepository(checkpointStore = checkpointStore)
        assertTrue(repository.selectDataset(TrainingDataset("content://provider/cache", identity = identity)))
        val state = repository.snapshot()
        assertEquals(250, state.lastCheckpoint?.completedStep)
        assertTrue(state.canResume)
        repository.close()
    }

    @Test fun staleDatasetSelectionTokenCannotOverwriteNewerSelection() {
        val repository = StandaloneTrainingRepository()
        val oldToken = repository.nextDatasetSelectionToken()
        val newToken = repository.nextDatasetSelectionToken()
        assertFalse(
            repository.selectDataset(
                TrainingDataset("content://provider/old", "old.bin"),
                oldToken,
            ),
        )
        assertTrue(
            repository.selectDataset(
                TrainingDataset("content://provider/new", "new.bin"),
                newToken,
            ),
        )
        assertEquals("content://provider/new", repository.snapshot().datasetUri)
        repository.close()
    }
}
