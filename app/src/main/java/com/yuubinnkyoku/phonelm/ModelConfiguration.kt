package com.yuubinnkyoku.phonelm

/** Architecture-only identity. Optimizer and run controls deliberately live elsewhere. */
data class ModelArchitecture(
    val layers: Int,
    val heads: Int,
    val tokens: Int,
    val dimension: Int,
    val feedForwardDimension: Int,
    val vocabularySize: Int,
    val tokenizerKind: String,
    val tokenizerHash: String?,
) {
    fun validationError(): String? = when {
        layers !in 1..128 -> "layers must be in 1..128"
        heads !in 1..128 -> "heads must be in 1..128"
        tokens !in 1..4096 -> "tokens must be in 1..4096"
        dimension !in 1..4096 || dimension % heads != 0 ->
            "dimension must be in 1..4096 and divisible by heads"
        feedForwardDimension !in 1..16384 -> "feedForwardDimension must be in 1..16384"
        vocabularySize !in 2..65536 -> "vocabularySize must be in 2..65536"
        vocabularySize == 1024 && tokenizerKind != ModelConfigurationCatalog.BYTE_BPE_TOKENIZER ->
            "V1024 requires byte_bpe tokenizer"
        vocabularySize == 1024 && tokenizerHash != ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH ->
            "V1024 requires the canonical tokenizer SHA-256"
        vocabularySize == 256 && tokenizerKind != ModelConfigurationCatalog.LEGACY_BYTE_TOKENIZER ->
            "V256 requires the legacy byte tokenizer"
        vocabularySize == 256 && tokenizerHash != null ->
            "legacy V256 must not carry a BPE tokenizer hash"
        else -> null
    }

    val headDimension: Int get() = dimension / heads

    /** Mirrors the native/checkpoint parameter registry exactly. */
    fun parameterCount(): Long {
        validationError()?.let { throw IllegalArgumentException(it) }
        val d = dimension.toLong()
        val ffn = feedForwardDimension.toLong()
        val vocabulary = vocabularySize.toLong()
        val embeddingAndProjection = Math.multiplyExact(Math.multiplyExact(vocabulary, d), 2L)
        val norms = Math.multiplyExact(4L, d)
        val attention = Math.multiplyExact(4L, Math.multiplyExact(d, d))
        val feedForward = Math.multiplyExact(2L, Math.multiplyExact(d, ffn))
        val perLayer = Math.addExact(Math.addExact(norms, attention), feedForward)
        return Math.addExact(embeddingAndProjection, Math.multiplyExact(layers.toLong(), perLayer))
    }

    val displayLabel: String
        get() = "V$vocabularySize/T$tokens/D$dimension/FFN$feedForwardDimension/L$layers/H$heads"
}

/** The bounded set exposed by production UI and accepted by production backends. */
object ModelConfigurationCatalog {
    const val LEGACY_BYTE_TOKENIZER = "byte"
    const val BYTE_BPE_TOKENIZER = "byte_bpe"
    const val LEGACY_TRAINING_TOKENIZER_ID = "nicopedia-byte-v1"
    const val CANONICAL_BPE_TOKENIZER_HASH =
        "sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798"

    val vocabularySizes = listOf(256, 1024)
    val tokenCounts = listOf(32)
    val layerCounts = listOf(19)
    val headCounts = listOf(2)
    val dimensions = listOf(32, 48, 64)
    val feedForwardDimensions = listOf(32, 48, 64)

    val defaultConfig: TrainingModelConfig get() = TrainingModelConfig.NICOPEDIA_L19

    fun config(vocabularySize: Int, dimension: Int, feedForwardDimension: Int): TrainingModelConfig =
        TrainingModelConfig.NICOPEDIA_L19.copy(
            dimension = dimension,
            feedForwardDimension = feedForwardDimension,
            vocabularySize = vocabularySize,
            tokenizerId = if (vocabularySize == 1024) BYTE_BPE_TOKENIZER else LEGACY_TRAINING_TOKENIZER_ID,
            tokenizerHash = if (vocabularySize == 1024) CANONICAL_BPE_TOKENIZER_HASH else null,
        ).also { require(SupportedTrainingModelPolicy.validationError(it) == null) }
}

object SupportedTrainingModelPolicy {
    fun validationError(config: TrainingModelConfig): String? {
        config.validationError()?.let { return it }
        if (config.layers !in ModelConfigurationCatalog.layerCounts ||
            config.heads !in ModelConfigurationCatalog.headCounts ||
            config.tokens !in ModelConfigurationCatalog.tokenCounts ||
            config.dimension !in ModelConfigurationCatalog.dimensions ||
            config.feedForwardDimension !in ModelConfigurationCatalog.feedForwardDimensions ||
            config.vocabularySize !in ModelConfigurationCatalog.vocabularySizes
        ) return "model configuration is not in the production catalog"
        if (config.batchSize != 8 || config.learningRate.toBits() != 0.003f.toBits() ||
            config.checkpointInterval != 250 || config.seed != 1L ||
            config.optimizerId != "adam-beta1-0.9-beta2-0.999-eps-1e-8"
        ) return "training controls are not supported by the production backend"
        return null
    }
}

object SupportedGenerationModelPolicy {
    fun validationError(architecture: ModelArchitecture): String? {
        architecture.validationError()?.let { return it }
        if (architecture.layers !in ModelConfigurationCatalog.layerCounts ||
            architecture.heads !in ModelConfigurationCatalog.headCounts ||
            architecture.tokens !in ModelConfigurationCatalog.tokenCounts ||
            architecture.dimension !in ModelConfigurationCatalog.dimensions ||
            architecture.feedForwardDimension !in ModelConfigurationCatalog.feedForwardDimensions ||
            architecture.vocabularySize !in ModelConfigurationCatalog.vocabularySizes
        ) return "checkpoint architecture is not supported by this build"
        return null
    }
}

/** Runtime capability is narrower than the architecture catalog until the app has a safe BPE-model import. */
object StandaloneTrainingCapabilityPolicy {
    fun blockingError(config: TrainingModelConfig): String? = when (config.vocabularySize) {
        1024 -> "V1024 training requires the canonical byte-BPE tokenizer model in app-private storage; model import is not available in this build"
        else -> null
    }
}

data class CheckpointFormatPolicy(val format: String, val version: Int) {
    companion object {
        fun forArchitecture(architecture: ModelArchitecture): CheckpointFormatPolicy = when {
            architecture.vocabularySize == 256 &&
                architecture.tokenizerKind == ModelConfigurationCatalog.LEGACY_BYTE_TOKENIZER &&
                architecture.tokenizerHash == null -> CheckpointFormatPolicy("NPRTCKPTV2", 2)
            architecture.vocabularySize == 1024 &&
                architecture.tokenizerKind == ModelConfigurationCatalog.BYTE_BPE_TOKENIZER &&
                architecture.tokenizerHash == ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH ->
                CheckpointFormatPolicy("NPRTCKPTV3", 3)
            else -> throw IllegalArgumentException("no checkpoint format for model tokenizer identity")
        }

        fun forConfig(config: TrainingModelConfig): CheckpointFormatPolicy = forArchitecture(config.architecture)
    }
}

/** Strict, deterministic durable encoding. Unknown versions/keys and malformed values fail closed. */
object TrainingModelConfigCodec {
    private const val VERSION = "NPRTMODEL1"
    private val keys = listOf("L", "H", "T", "D", "FFN", "V", "B", "LR", "I", "S", "TK", "TH", "O")

    fun encode(config: TrainingModelConfig): String {
        require(config.validationError() == null) { config.validationError() ?: "invalid model configuration" }
        val values = listOf(
            config.layers.toString(), config.heads.toString(), config.tokens.toString(),
            config.dimension.toString(), config.feedForwardDimension.toString(), config.vocabularySize.toString(),
            config.batchSize.toString(), config.learningRate.toBits().toString(), config.checkpointInterval.toString(),
            config.seed.toString(), config.tokenizerId, config.tokenizerHash ?: "-", config.optimizerId,
        )
        return buildList {
            add(VERSION)
            keys.zip(values).forEach { (key, value) -> add("$key=$value") }
        }.joinToString(";")
    }

    fun decode(encoded: String): TrainingModelConfig {
        val fields = encoded.split(';')
        require(fields.size == keys.size + 1 && fields.first() == VERSION) { "unknown or malformed model config version" }
        val values = fields.drop(1).mapIndexed { index, field ->
            val separator = field.indexOf('=')
            require(separator > 0 && field.substring(0, separator) == keys[index]) { "unknown or misplaced model config key" }
            field.substring(separator + 1).also { require(it.isNotEmpty()) { "empty model config value" } }
        }
        val config = TrainingModelConfig(
            layers = values[0].strictInt("layers"), heads = values[1].strictInt("heads"),
            tokens = values[2].strictInt("tokens"), dimension = values[3].strictInt("dimension"),
            feedForwardDimension = values[4].strictInt("feedForwardDimension"),
            vocabularySize = values[5].strictInt("vocabularySize"), batchSize = values[6].strictInt("batchSize"),
            learningRate = Float.fromBits(values[7].strictInt("learningRate bits")),
            checkpointInterval = values[8].strictInt("checkpointInterval"), seed = values[9].strictLong("seed"),
            tokenizerId = values[10], tokenizerHash = values[11].takeUnless { it == "-" }, optimizerId = values[12],
        )
        require(config.validationError() == null) { config.validationError() ?: "invalid model configuration" }
        return config
    }

    fun decodeLegacyCompatibilityKey(encoded: String): TrainingModelConfig {
        val legacy = TrainingModelConfig.NICOPEDIA_L19
        require(encoded == legacy.compatibilityKey) { "legacy model config identity differs" }
        return legacy
    }

    private fun String.strictInt(name: String): Int = toIntOrNull() ?: throw IllegalArgumentException("invalid $name")
    private fun String.strictLong(name: String): Long = toLongOrNull() ?: throw IllegalArgumentException("invalid $name")
}
