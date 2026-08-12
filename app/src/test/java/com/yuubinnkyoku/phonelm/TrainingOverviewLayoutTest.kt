package com.yuubinnkyoku.phonelm

import com.yuubinnkyoku.phonelm.ui.training.TrainingOverviewDensity
import com.yuubinnkyoku.phonelm.ui.training.currentHtpObservationRatio
import com.yuubinnkyoku.phonelm.ui.training.currentProcessCpuPercent
import com.yuubinnkyoku.phonelm.ui.training.trainingOverviewDensity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingOverviewLayoutTest {
    @Test fun normalPortraitUsesStandardDensity() {
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 800, fontScale = 1f))
    }

    @Test fun shortPortraitAndLargeFontsSelectCompactDensity() {
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 700, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.2f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.3f))
    }

    @Test fun densityThresholdsAreStableAtPortraitBoundaries() {
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 719, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 720, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 859, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMFORTABLE, trainingOverviewDensity(heightDp = 860, fontScale = 1f))
        assertEquals(TrainingOverviewDensity.COMFORTABLE, trainingOverviewDensity(heightDp = 900, fontScale = 1.05f))
        assertEquals(TrainingOverviewDensity.STANDARD, trainingOverviewDensity(heightDp = 900, fontScale = 1.051f))
        assertEquals(TrainingOverviewDensity.COMPACT, trainingOverviewDensity(heightDp = 900, fontScale = 1.20f))
    }

    @Test fun densityPolicyKeepsGraphVisibleAtEveryTier() {
        TrainingOverviewDensity.entries.forEach { density ->
            assertTrue(density.graphMinimumHeightDp >= 48)
            assertTrue(density.cardPaddingDp >= 8)
            assertTrue(density.heroPercentSp >= 36)
        }
    }

    @Test fun currentActivityDoesNotBackfillMissingLatestObservation() {
        val history = listOf(
            TrainingActivityHistoryEntry(step = 8, htpObservationRatioPercent = 64.0, processCpuPercent = 17.0),
            TrainingActivityHistoryEntry(step = 16, htpObservationRatioPercent = null, processCpuPercent = null),
        )

        assertNull(currentHtpObservationRatio(history))
        assertNull(currentProcessCpuPercent(history))
    }
}
