package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class BenchmarkConfigTest {
    @Test
    fun presetsAreValid() {
        assertNull(BenchmarkConfig.small().validationError())
        assertNull(BenchmarkConfig.benchmark().validationError())
    }

    @Test
    fun invalidDimensionsAreRejected() {
        assertNotNull(BenchmarkConfig.small().copy(batchSize = 0).validationError())
        assertNotNull(BenchmarkConfig.small().copy(dimension = -1).validationError())
        assertNotNull(BenchmarkConfig.small().copy(steps = 0).validationError())
        assertNotNull(BenchmarkConfig.small().copy(warmupSteps = -1).validationError())
        assertNotNull(BenchmarkConfig.small().copy(learningRate = Float.NaN).validationError())
        assertNotNull(BenchmarkConfig.small().copy(
            sampleCount = 3,
            checkpointSelectionMode = CheckpointSelectionMode.BEST_VALIDATION_V1,
        ).validationError())
    }

    @Test
    fun benchmarkPresetMatchesProtocol() {
        val config = BenchmarkConfig.benchmark(Backend.VULKAN)
        assertEquals(Backend.VULKAN, config.backend)
        assertEquals(32, config.batchSize)
        assertEquals(512, config.dimension)
        assertEquals(200, config.steps)
        assertEquals(20, config.warmupSteps)
    }

    @Test
    fun scalingModesHaveStableNativeCodes() {
        val expected = mapOf(
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T16D16_SMOKE to 89,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T32D32_SMOKE to 90,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T16D16_SMOKE to 91,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T32D32_SMOKE to 92,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T16D16_SMOKE to 93,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_SMOKE to 94,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_FORMAL to 95,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_FORMAL to 96,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_FORMAL to 97,
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_DIAGNOSTIC to 98,
        )
        assertEquals(expected.values.toSet().size, expected.size)
        expected.forEach { (mode, nativeCode) -> assertEquals(nativeCode, mode.nativeCode) }
    }
}
