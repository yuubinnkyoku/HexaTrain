package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingCheckpointTest {
    private fun checkpoint(
        step: Int,
        finite: Boolean = true,
        config: TrainingModelConfig = TrainingModelConfig.NICOPEDIA_L19,
        datasetIdentity: String? = null,
    ) = TrainingCheckpointMetadata(
        "checkpoint-$step", step, config, "NPRTCKPTV2", 2, finite, step.toLong(), datasetIdentity,
    )

    @Test fun catalogSortsByStepThenCreationTime() {
        assertEquals(listOf(500, 250, 1), TrainingCheckpointCatalog.sortedNewestFirst(listOf(checkpoint(1), checkpoint(500), checkpoint(250))).map { it.completedStep })
    }

    @Test fun latestIncompatibleCheckpointIsNotSilentlySkipped() {
        val result = TrainingCheckpointCatalog.latestCompatible(
            listOf(checkpoint(500, finite = false), checkpoint(250)),
            TrainingModelConfig.NICOPEDIA_L19, "NPRTCKPTV2", 2,
        )
        assertTrue(result is TrainingCheckpointSelection.Incompatible)
    }

    @Test fun matchingLatestCheckpointIsSelected() {
        val result = TrainingCheckpointCatalog.latestCompatible(listOf(checkpoint(250)), TrainingModelConfig.NICOPEDIA_L19, "NPRTCKPTV2", 2)
        assertEquals(250, (result as TrainingCheckpointSelection.Selected).checkpoint.completedStep)
    }

    @Test fun incompatibleDatasetDoesNotFallBackToOlderCheckpoint() {
        val result = TrainingCheckpointCatalog.latestCompatible(
            listOf(checkpoint(500, datasetIdentity = "dataset-b"), checkpoint(250, datasetIdentity = "dataset-a")),
            TrainingModelConfig.NICOPEDIA_L19,
            "NPRTCKPTV2",
            2,
            expectedDatasetIdentity = "dataset-a",
        )
        assertTrue(result is TrainingCheckpointSelection.Incompatible)
    }
}
