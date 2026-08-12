package com.yuubinnkyoku.phonelm.ui.training

import com.yuubinnkyoku.phonelm.TrainingActivityHistoryEntry

/** Pure layout policy so portrait overview density can be verified on the JVM. */
internal enum class TrainingOverviewDensity(
    val outerPaddingDp: Int,
    val sectionGapDp: Int,
    val cardPaddingDp: Int,
    val heroPercentSp: Int,
    val graphMinimumHeightDp: Int,
) {
    COMPACT(8, 6, 10, 38, 54),
    STANDARD(12, 8, 12, 46, 72),
    COMFORTABLE(16, 10, 14, 52, 88),
}

internal fun trainingOverviewDensity(heightDp: Int, fontScale: Float): TrainingOverviewDensity = when {
    heightDp < 720 || fontScale >= 1.20f -> TrainingOverviewDensity.COMPACT
    heightDp < 860 || fontScale > 1.05f -> TrainingOverviewDensity.STANDARD
    else -> TrainingOverviewDensity.COMFORTABLE
}

/** The overview shows the current accepted observation; missing values are never backfilled. */
internal fun currentHtpObservationRatio(history: List<TrainingActivityHistoryEntry>): Double? =
    history.lastOrNull()?.htpObservationRatioPercent

internal fun currentProcessCpuPercent(history: List<TrainingActivityHistoryEntry>): Double? =
    history.lastOrNull()?.processCpuPercent
