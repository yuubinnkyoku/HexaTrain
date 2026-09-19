package com.yuubinnkyoku.phonelm

import java.io.File
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

    private data class ParsedDefinition(
        val suffix: String,
        val role: String,
        val placement: String,
        val condition: String,
        val shape: List<String>,
        val rank: Int,
    )

    @Test fun generatedSsotMetadataIsGenericAndParameterCountDerivesFromIt() {
        val candidatePaths = listOf(
            File("../metadata/transformer_parameter_metadata.json"),
            File("metadata/transformer_parameter_metadata.json"),
            File("../../metadata/transformer_parameter_metadata.json"),
        )
        val metadataFile = candidatePaths.firstOrNull { it.isFile }
        assertNotNull("metadata/transformer_parameter_metadata.json must exist", metadataFile)
        val json = metadataFile!!.readText()

        val paramRegex = Regex(
            """\{\s*"suffix":\s*"([^"]+)",\s*"role":\s*"([^"]+)",\s*"placement":\s*"([^"]+)",\s*"condition":\s*"([^"]+)",\s*"shape":\s*\[([^\]]*)\],\s*"rank":\s*(\d+)""",
            RegexOption.DOT_MATCHES_ALL,
        )
        val definitions = paramRegex.findAll(json).map { match ->
            val (suffix, role, placement, condition, shapeStr, rank) = match.destructured
            val dims = shapeStr.split(",").map { it.trim().removeSurrounding("\"") }.filter { it.isNotEmpty() }
            ParsedDefinition(suffix, role, placement, condition, dims, rank.toInt())
        }.toList()

        // Generic invariants only — no fixed parameter name/count registry may be
        // re-declared here. Adding a parameter to the C++ SSOT must only require
        // regenerating the artifacts, never touching this list.
        assertTrue("metadata must contain at least one definition", definitions.isNotEmpty())
        assertEquals(
            "definitions must not duplicate suffixes",
            definitions.size,
            definitions.map { it.suffix }.toSet().size,
        )
        for (def in definitions) {
            assertTrue("suffix must be non-empty", def.suffix.isNotBlank())
            assertTrue("role must be serialized without loss", def.role in setOf("MUON", "AUX_ADAM"))
            assertTrue("placement must be serialized without loss", def.placement in setOf("GLOBAL_PREFIX", "PER_LAYER", "GLOBAL_SUFFIX"))
            assertTrue("condition must be serialized without loss", def.condition in setOf("ALWAYS", "HEADWISE_G1"))
            assertTrue("rank must be <= shape size", def.rank in 1..def.shape.size)
            assertTrue("shape must not be empty", def.shape.isNotEmpty())
        }
        for (placement in setOf("GLOBAL_PREFIX", "PER_LAYER", "GLOBAL_SUFFIX")) {
            assertTrue(
                "missing definition in placement group $placement",
                definitions.any { it.placement == placement },
            )
        }

        fun computeFromMetadata(arch: ModelArchitecture, headwiseG1: Boolean): Long {
            var total = 0L
            for (def in definitions) {
                if (def.condition == "HEADWISE_G1" && !headwiseG1) continue
                var elements = 1L
                for (dim in def.shape) {
                    elements = Math.multiplyExact(elements, when (dim) {
                        "VOCABULARY" -> arch.vocabularySize.toLong()
                        "MODEL" -> arch.dimension.toLong()
                        "FEED_FORWARD" -> arch.feedForwardDimension.toLong()
                        "HEADS" -> arch.heads.toLong()
                        else -> error("Unknown dimension: $dim")
                    })
                }
                val instances = when (def.placement) {
                    "PER_LAYER" -> arch.layers.toLong()
                    "GLOBAL_PREFIX", "GLOBAL_SUFFIX" -> 1L
                    else -> error("Unknown placement: ${def.placement}")
                }
                total = Math.addExact(total, Math.multiplyExact(elements, instances))
            }
            return total
        }

        val configs = ModelConfigurationCatalog.vocabularySizes.flatMap { vocabulary ->
            ModelConfigurationCatalog.dimensions.flatMap { dimension ->
                ModelConfigurationCatalog.feedForwardDimensions.map { ffn ->
                    ModelConfigurationCatalog.config(vocabulary, dimension, ffn)
                }
            }
        }
        for (cfg in configs) {
            val arch = cfg.architecture
            assertEquals(
                "Ungated parameter count mismatch for ${arch.displayLabel}",
                computeFromMetadata(arch, headwiseG1 = false),
                arch.parameterCount(headwiseG1 = false),
            )
            assertEquals(
                "Gated parameter count mismatch for ${arch.displayLabel}",
                computeFromMetadata(arch, headwiseG1 = true),
                arch.parameterCount(headwiseG1 = true),
            )
        }

        // Gated evaluator fixture: gating must add the HEADWISE_G1 parameter only.
        val l19 = ModelArchitecture(
            layers = 19, heads = 2, tokens = 32, dimension = 64, feedForwardDimension = 128,
            vocabularySize = 1024, tokenizerKind = "byte_bpe", tokenizerHash = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH,
        )
        val gatedAdded = computeFromMetadata(l19, headwiseG1 = true) - computeFromMetadata(l19, headwiseG1 = false)
        assertTrue(
            "gated evaluator must add exactly the HEADWISE_G1 parameter bytes",
            gatedAdded > 0L,
        )
        assertEquals(
            "evaluator and generated Kotlin metadata must agree on gated L19 count",
            l19.parameterCount(headwiseG1 = true),
            GeneratedTransformerParameterMetadata.calculateParameterCount(
                vocabularySize = 1024L, dimension = 64L, feedForwardDimension = 128L,
                layers = 19L, heads = 2L, headwiseG1 = true,
            ),
        )
    }
}
