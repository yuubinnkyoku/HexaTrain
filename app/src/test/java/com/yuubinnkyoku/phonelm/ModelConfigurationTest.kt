package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ModelConfigurationTest {
    @Test fun catalogProducesEveryBoundedProductionCombination() {
        val configs = ModelConfigurationCatalog.vocabularySizes.flatMap { vocabulary ->
            ModelConfigurationCatalog.dimensions.flatMap { dimension ->
                ModelConfigurationCatalog.feedForwardDimensions.map { ffn ->
                    ModelConfigurationCatalog.config(vocabulary, dimension, ffn)
                }
            }
        }
        assertEquals(18, configs.size)
        configs.forEach { assertNull(SupportedTrainingModelPolicy.validationError(it)) }
    }

    @Test fun parameterCountMatchesCanonicalCapacityEvidence() {
        assertEquals(
            364_608L,
            ModelConfigurationCatalog.config(1024, 48, 48).architecture.parameterCount(),
        )
        assertEquals(
            602_880L,
            ModelConfigurationCatalog.config(1024, 64, 64).architecture.parameterCount(),
        )
        assertEquals(32, ModelConfigurationCatalog.config(1024, 64, 64).architecture.headDimension)
    }

    @Test fun codecRoundTripsWithoutChangingFloatBitIdentity() {
        val source = ModelConfigurationCatalog.config(1024, 64, 48)
        val encoded = TrainingModelConfigCodec.encode(source)
        assertEquals(source, TrainingModelConfigCodec.decode(encoded))
        assertEquals(source.learningRate.toBits(), TrainingModelConfigCodec.decode(encoded).learningRate.toBits())
    }

    @Test fun codecRejectsMalformedUnknownAndOverflowingValues() {
        val encoded = TrainingModelConfigCodec.encode(ModelConfigurationCatalog.defaultConfig)
        listOf(
            encoded.replace("NPRTMODEL1", "NPRTMODEL2"),
            encoded.replace(";D=32", ";UNKNOWN=32"),
            encoded.replace(";D=32", ";D=2147483648"),
            encoded.replace(";H=2", ";H=3"),
        ).forEach { malformed ->
            assertNotNull(runCatching { TrainingModelConfigCodec.decode(malformed) }.exceptionOrNull())
        }
    }

    @Test fun canonicalTokenizerIdentityAndCheckpointPolicyFailClosed() {
        val v256 = ModelConfigurationCatalog.config(256, 32, 32)
        val v1024 = ModelConfigurationCatalog.config(1024, 64, 64)
        assertEquals(CheckpointFormatPolicy("NPRTCKPTV2", 2), CheckpointFormatPolicy.forConfig(v256))
        assertEquals(CheckpointFormatPolicy("NPRTCKPTV3", 3), CheckpointFormatPolicy.forConfig(v1024))
        assertNotNull(v1024.copy(tokenizerHash = null).validationError())
        assertNotNull(v1024.copy(tokenizerHash = "sha256:" + "00".repeat(32)).validationError())
        assertNotNull(v256.copy(tokenizerHash = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH).validationError())
    }

    @Test fun legacyCompatibilityKeyMigratesOnlyExactD32Identity() {
        val legacy = TrainingModelConfig.NICOPEDIA_L19
        assertEquals(legacy, TrainingModelConfigCodec.decodeLegacyCompatibilityKey(legacy.compatibilityKey))
        assertNotNull(runCatching {
            TrainingModelConfigCodec.decodeLegacyCompatibilityKey(legacy.compatibilityKey.replace("D=32", "D=48"))
        }.exceptionOrNull())
    }
}
