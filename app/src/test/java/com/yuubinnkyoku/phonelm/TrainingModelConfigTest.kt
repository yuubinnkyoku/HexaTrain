package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Test

class TrainingModelConfigTest {
    @Test fun canonicalNicopediaL19PresetIsCompleteAndValid() {
        val config = TrainingModelConfig.NICOPEDIA_L19
        assertEquals(19, config.layers)
        assertEquals(2, config.heads)
        assertEquals(32, config.tokens)
        assertEquals(32, config.dimension)
        assertEquals(32, config.feedForwardDimension)
        assertEquals(256, config.vocabularySize)
        assertEquals(8, config.batchSize)
        assertEquals(0.003f, config.learningRate)
        assertEquals(250, config.checkpointInterval)
        assertNull(config.validationError())
    }

    @Test fun invalidHeadDimensionRelationshipFailsClosed() {
        assertNotNull(TrainingModelConfig.NICOPEDIA_L19.copy(dimension = 15).validationError())
        assertNotNull(TrainingModelConfig.NICOPEDIA_L19.copy(learningRate = Float.NaN).validationError())
    }

    @Test fun htpEvidenceRequiresBackendIdentityAndNoError() {
        assertEquals(
            true,
            TrainingRuntimeEvidence(true, true, false).isAuthoritativelyHtp,
        )
        assertEquals(
            false,
            TrainingRuntimeEvidence(true, true, false, backend = "CPU").isAuthoritativelyHtp,
        )
        assertEquals(
            false,
            TrainingRuntimeEvidence(true, true, false, error = "qnn warning").isAuthoritativelyHtp,
        )
    }

    @Test fun trainingRequestRejectsFilesystemDatasetUri() {
        val request = TrainingRequest(
            TrainingModelConfig.NICOPEDIA_L19,
            TrainingDataset("C:/private/train_pilot.bin"),
            8000,
        )
        assertNotNull(request.validationError())
    }
}
