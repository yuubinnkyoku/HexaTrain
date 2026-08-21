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

    @Test fun v3BindsTokenizerHashAndRejectsMismatch() {
        val hash = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH
        val bpeConfig = TrainingModelConfig.nicopediaBpeV1024(hash).copy(
            layers = 1, heads = 1, tokens = 1, dimension = 1,
            feedForwardDimension = 1, batchSize = 1, checkpointInterval = 1,
        )
        val bytes = bpeCheckpointBytes(hash)
        val inspected = GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes), "bpe.ckpt", 10L, bytes.size.toLong(), bpeConfig,
        )
        assertTrue(inspected.usable)
        assertEquals("NPRTCKPTV3", inspected.format)
        assertEquals(hash, inspected.tokenizerHash)
        val mismatch = GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes), "bpe.ckpt", 10L, bytes.size.toLong(),
            bpeConfig.copy(tokenizerHash = "sha256:" + "cd".repeat(32)),
        )
        assertEquals(GenerationCheckpointCompatibility.INCOMPATIBLE, mismatch.compatibility)
        assertFalse(mismatch.usable)
    }

    @Test fun productionD64V1024CheckpointIsAcceptedFromItsOwnHeader() {
        val bytes = productionBpeCheckpointBytes(dimension = 64, feedForward = 64, layers = 19, heads = 2, tokens = 32)
        val inspected = GenerationCheckpointInspector.inspect(
            ByteArrayInputStream(bytes), "d64-v1024.ckpt", 10L, bytes.size.toLong(), expected = null,
        )
        assertTrue(inspected.usable)
        assertEquals(ModelConfigurationCatalog.config(1024, 64, 64).architecture, inspected.architecture)
        assertEquals(602_880L, inspected.architecture.parameterCount())
    }

    private fun productionBpeCheckpointBytes(
        dimension: Int,
        feedForward: Int,
        layers: Int,
        heads: Int,
        tokens: Int,
    ): ByteArray {
        val hash = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH
        val registry = buildList {
            add("token_embedding" to 1024L * dimension)
            repeat(layers) { layer ->
                val prefix = "layer_%03d.".format(layer)
                add(prefix + "norm1_gamma" to dimension.toLong())
                add(prefix + "norm1_beta" to dimension.toLong())
                repeat(4) { index -> add(prefix + listOf("wq", "wk", "wv", "wo")[index] to dimension.toLong() * dimension) }
                add(prefix + "norm2_gamma" to dimension.toLong())
                add(prefix + "norm2_beta" to dimension.toLong())
                add(prefix + "ffn_w1" to dimension.toLong() * feedForward)
                add(prefix + "ffn_w2" to feedForward.toLong() * dimension)
            }
            add("output_projection" to dimension.toLong() * 1024L)
        }
        val output = ByteArrayOutputStream()
        DataOutputStream(output).use { data ->
            data.write("NPRTCKPTV3\n".toByteArray(Charsets.US_ASCII))
            listOf(1024, tokens, dimension, feedForward, layers, heads, 1, 250).forEach(data::writeInt)
            val kind = "byte_bpe".toByteArray(Charsets.US_ASCII)
            val hashBytes = hash.toByteArray(Charsets.US_ASCII)
            data.writeInt(kind.size); data.write(kind)
            data.writeInt(hashBytes.size); data.write(hashBytes)
            repeat(3) { registryIndex ->
                data.writeInt(registry.size)
                registry.forEachIndexed { entryIndex, (name, count) ->
                    val nameBytes = name.toByteArray(Charsets.UTF_8)
                    data.writeInt(nameBytes.size); data.write(nameBytes); data.writeLong(count)
                    repeat(count.toInt()) { valueIndex ->
                        data.writeInt(Integer.reverseBytes((registryIndex + entryIndex + valueIndex + 1f).toBits()))
                    }
                }
            }
        }
        return output.toByteArray()
    }

    private fun bpeCheckpointBytes(hash: String): ByteArray {
        val output = ByteArrayOutputStream()
        DataOutputStream(output).use { data ->
            data.write("NPRTCKPTV3\n".toByteArray(Charsets.US_ASCII))
            listOf(1024, 1, 1, 1, 1, 1, 1, 250).forEach(data::writeInt)
            val kind = "byte_bpe".toByteArray(Charsets.US_ASCII)
            val hashBytes = hash.toByteArray(Charsets.US_ASCII)
            data.writeInt(kind.size); data.write(kind)
            data.writeInt(hashBytes.size); data.write(hashBytes)
            val registry = listOf(
                "token_embedding" to 1024,
                "layer_000.norm1_gamma" to 1, "layer_000.norm1_beta" to 1,
                "layer_000.wq" to 1, "layer_000.wk" to 1,
                "layer_000.wv" to 1, "layer_000.wo" to 1,
                "layer_000.norm2_gamma" to 1, "layer_000.norm2_beta" to 1,
                "layer_000.ffn_w1" to 1, "layer_000.ffn_w2" to 1,
                "output_projection" to 1024,
            )
            repeat(3) { registryIndex ->
                data.writeInt(registry.size)
                registry.forEachIndexed { entryIndex, (name, count) ->
                    val nameBytes = name.toByteArray(Charsets.UTF_8)
                    data.writeInt(nameBytes.size); data.write(nameBytes); data.writeLong(count.toLong())
                    repeat(count) { valueIndex ->
                        val value = (registryIndex + entryIndex + valueIndex + 1).toFloat()
                        data.writeInt(Integer.reverseBytes(value.toBits()))
                    }
                }
            }
        }
        return output.toByteArray()
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
