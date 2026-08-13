package com.yuubinnkyoku.phonelm

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationCheckpointInspectorTest {
    private val config = TrainingModelConfig(
        layers = 1,
        heads = 1,
        tokens = 1,
        dimension = 2,
        feedForwardDimension = 2,
        vocabularySize = 2,
        batchSize = 1,
        learningRate = 0.1f,
        checkpointInterval = 1,
    )

    @Test fun completeV2IsInspectedAndHasParameterHash() {
        val bytes = checkpointBytes(nonFinite = false)
        val inspected = GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes), "test.ckpt", 10L, bytes.size.toLong(), config,
        )
        assertTrue(inspected.formatValid)
        assertTrue(inspected.finite)
        assertTrue(inspected.usable)
        assertEquals(250, inspected.step)
        assertTrue(inspected.parameterHash!!.matches(Regex("fnv1a64:[0-9a-f]{16}")))
    }

    @Test fun nonFinitePayloadIsNeverUsable() {
        val bytes = checkpointBytes(nonFinite = true)
        val inspected = GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes), "test.ckpt", 10L, bytes.size.toLong(), config,
        )
        assertTrue(inspected.formatValid)
        assertFalse(inspected.finite)
        assertFalse(inspected.usable)
    }

    @Test(expected = java.io.EOFException::class)
    fun truncatedV2FailsClosed() {
        val bytes = checkpointBytes(nonFinite = false)
        GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes.copyOf(bytes.size - 3)),
            "truncated.ckpt", 10L, (bytes.size - 3).toLong(), config,
        )
    }

    private fun checkpointBytes(nonFinite: Boolean): ByteArray {
        val output = ByteArrayOutputStream()
        DataOutputStream(output).use { data ->
            data.write("NPRTCKPTV2\n".toByteArray(Charsets.US_ASCII))
            listOf(2, 1, 2, 2, 1, 1, 1, 250).forEach(data::writeInt)
            val registry = listOf(
                "token_embedding" to 4,
                "layer_000.norm1_gamma" to 2,
                "layer_000.norm1_beta" to 2,
                "layer_000.wq" to 4,
                "layer_000.wk" to 4,
                "layer_000.wv" to 4,
                "layer_000.wo" to 4,
                "layer_000.norm2_gamma" to 2,
                "layer_000.norm2_beta" to 2,
                "layer_000.ffn_w1" to 4,
                "layer_000.ffn_w2" to 4,
                "output_projection" to 4,
            )
            repeat(3) { registryIndex ->
                data.writeInt(registry.size)
                registry.forEachIndexed { entryIndex, (name, count) ->
                    val nameBytes = name.toByteArray(Charsets.UTF_8)
                    data.writeInt(nameBytes.size)
                    data.write(nameBytes)
                    data.writeLong(count.toLong())
                    repeat(count) { valueIndex ->
                        val value = if (nonFinite && registryIndex == 0 && entryIndex == 0 && valueIndex == 0) {
                            Float.NaN
                        } else {
                            (registryIndex + entryIndex + valueIndex + 1).toFloat()
                        }
                        data.writeInt(Integer.reverseBytes(value.toBits()))
                    }
                }
            }
        }
        return output.toByteArray()
    }
}
