package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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

    @Test fun generationPolicyAcceptsFfn128WithoutExpandingTrainingCatalog() {
        val ffn128 = ModelConfigurationCatalog.config(1024, 48, 64).architecture.copy(
            feedForwardDimension = 128,
        )
        assertNull(SupportedGenerationModelPolicy.validationError(ffn128))
        assertNotNull(SupportedTrainingModelPolicy.validationError(
            ModelConfigurationCatalog.config(1024, 48, 64).copy(feedForwardDimension = 128),
        ))
    }

    @Test fun generationPolicyRejectsUnsupportedFeedForwardDimension() {
        val unsupported = ModelConfigurationCatalog.config(1024, 48, 64).architecture.copy(
            feedForwardDimension = 96,
        )
        assertNotNull(SupportedGenerationModelPolicy.validationError(unsupported))
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

    private fun computeFromDefinitions(
        definitions: List<GeneratedParameterDefinition>,
        arch: ModelArchitecture,
        headwiseG1: Boolean,
    ): Long {
        var total = 0L
        for (def in definitions) {
            if (def.condition == GeneratedParameterCondition.HEADWISE_G1 && !headwiseG1) continue
            var elements = 1L
            for (dim in def.shape) {
                elements = Math.multiplyExact(elements, when (dim) {
                    GeneratedParameterDimension.VOCABULARY -> arch.vocabularySize.toLong()
                    GeneratedParameterDimension.MODEL -> arch.dimension.toLong()
                    GeneratedParameterDimension.FEED_FORWARD -> arch.feedForwardDimension.toLong()
                    GeneratedParameterDimension.HEADS -> arch.heads.toLong()
                })
            }
            val instances = when (def.placement) {
                GeneratedParameterPlacement.PER_LAYER -> arch.layers.toLong()
                GeneratedParameterPlacement.GLOBAL_PREFIX, GeneratedParameterPlacement.GLOBAL_SUFFIX -> 1L
            }
            total = Math.addExact(total, Math.multiplyExact(elements, instances))
        }
        return total
    }

    @Test fun generatedSsotMetadataIsGenericAndParameterCountDerivesFromIt() {
        // Traverse the generated Kotlin SSOT table directly. JSON artifact identity
        // with the C++ header is the exporter --check / stale-check responsibility;
        // JSON/Kotlin escaping is the exporter self-test responsibility.
        val definitions = GeneratedTransformerParameterMetadata.DEFINITIONS

        // Generic invariants only — no fixed parameter name/count registry may be
        // re-declared here. Adding a parameter to the C++ SSOT must only require
        // regenerating the artifacts, never touching this list.
        assertTrue("generated metadata must contain at least one definition", definitions.isNotEmpty())
        assertEquals(
            "definitions must not duplicate suffixes",
            definitions.size,
            definitions.map { it.suffix }.toSet().size,
        )
        for (def in definitions) {
            assertTrue("suffix must be non-empty", def.suffix.isNotBlank())
            assertTrue("shape must not be empty", def.shape.isNotEmpty())
            assertTrue("rank must be <= shape size", def.rank in 1..def.shape.size)
        }
        for (placement in GeneratedParameterPlacement.entries) {
            assertTrue(
                "missing definition in placement group $placement",
                definitions.any { it.placement == placement },
            )
        }

        val hasHeadwiseG1 = definitions.any { it.condition == GeneratedParameterCondition.HEADWISE_G1 }

        val configs = ModelConfigurationCatalog.vocabularySizes.flatMap { vocabulary ->
            ModelConfigurationCatalog.dimensions.flatMap { dimension ->
                ModelConfigurationCatalog.feedForwardDimensions.map { ffn ->
                    ModelConfigurationCatalog.config(vocabulary, dimension, ffn)
                }
            }
        }
        for (cfg in configs) {
            val arch = cfg.architecture
            val ungated = computeFromDefinitions(definitions, arch, headwiseG1 = false)
            val gated = computeFromDefinitions(definitions, arch, headwiseG1 = true)
            assertEquals(
                "Ungated parameter count mismatch for ${arch.displayLabel}",
                ungated,
                arch.parameterCount(headwiseG1 = false),
            )
            assertEquals(
                "Gated parameter count mismatch for ${arch.displayLabel}",
                gated,
                arch.parameterCount(headwiseG1 = true),
            )
            assertEquals(
                "Generated evaluator ungated mismatch for ${arch.displayLabel}",
                ungated,
                GeneratedTransformerParameterMetadata.calculateParameterCount(
                    vocabularySize = arch.vocabularySize.toLong(),
                    dimension = arch.dimension.toLong(),
                    feedForwardDimension = arch.feedForwardDimension.toLong(),
                    layers = arch.layers.toLong(),
                    heads = arch.heads.toLong(),
                    headwiseG1 = false,
                ),
            )
            assertEquals(
                "Generated evaluator gated mismatch for ${arch.displayLabel}",
                gated,
                GeneratedTransformerParameterMetadata.calculateParameterCount(
                    vocabularySize = arch.vocabularySize.toLong(),
                    dimension = arch.dimension.toLong(),
                    feedForwardDimension = arch.feedForwardDimension.toLong(),
                    layers = arch.layers.toLong(),
                    heads = arch.heads.toLong(),
                    headwiseG1 = true,
                ),
            )
            if (hasHeadwiseG1) {
                assertTrue(
                    "gated count must exceed ungated when HEADWISE_G1 definitions exist for ${arch.displayLabel}",
                    gated > ungated,
                )
            } else {
                assertEquals(
                    "gated and ungated counts must match when no HEADWISE_G1 definition exists",
                    ungated,
                    gated,
                )
            }
        }
    }
}
