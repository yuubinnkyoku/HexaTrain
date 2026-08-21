package com.yuubinnkyoku.phonelm

/** Immutable model and optimizer contract shared by the UI and the future JNI bridge. */
data class TrainingModelConfig(
    val layers: Int,
    val heads: Int,
    val tokens: Int,
    val dimension: Int,
    val feedForwardDimension: Int,
    val vocabularySize: Int,
    val batchSize: Int,
    val learningRate: Float,
    val checkpointInterval: Int,
    val seed: Long = 1L,
    val tokenizerId: String = "nicopedia-byte-v1",
    val tokenizerHash: String? = null,
    val optimizerId: String = "adam-beta1-0.9-beta2-0.999-eps-1e-8",
) {
    val architecture: ModelArchitecture
        get() = ModelArchitecture(
            layers = layers,
            heads = heads,
            tokens = tokens,
            dimension = dimension,
            feedForwardDimension = feedForwardDimension,
            vocabularySize = vocabularySize,
            tokenizerKind = when (tokenizerId) {
                ModelConfigurationCatalog.LEGACY_TRAINING_TOKENIZER_ID -> ModelConfigurationCatalog.LEGACY_BYTE_TOKENIZER
                else -> tokenizerId
            },
            tokenizerHash = tokenizerHash,
        )

    fun validationError(): String? = when {
        architecture.validationError() != null -> architecture.validationError()
        batchSize !in 1..4096 -> "batchSize must be in 1..4096"
        !learningRate.isFinite() || learningRate <= 0f ->
            "learningRate must be finite and positive"
        checkpointInterval !in 1..100_000 -> "checkpointInterval must be in 1..100000"
        seed !in 1L..99_999L -> "seed must be in 1..99999"
        tokenizerId.isBlank() -> "tokenizerId must not be blank"
        optimizerId.isBlank() -> "optimizerId must not be blank"
        else -> null
    }

    /** Stable identity passed to the native checkpoint compatibility gate. */
    val compatibilityKey: String
        get() = listOf(
            "L=$layers", "H=$heads", "T=$tokens", "D=$dimension",
            "FFN=$feedForwardDimension", "V=$vocabularySize", "B=$batchSize",
            "LR=${learningRate.toBits()}", "interval=$checkpointInterval",
            "seed=$seed", "tokenizer=$tokenizerId", "tokenizer_hash=${tokenizerHash ?: "legacy"}",
            "optimizer=$optimizerId",
        ).joinToString("|")

    companion object {
        /** Canonical Nicopedia L19 training preset. Do not duplicate these values in UI code. */
        val NICOPEDIA_L19 = TrainingModelConfig(
            layers = 19,
            heads = 2,
            tokens = 32,
            dimension = 32,
            feedForwardDimension = 32,
            vocabularySize = 256,
            batchSize = 8,
            learningRate = 0.003f,
            checkpointInterval = 250,
        )

        fun nicopediaBpeV1024(
            tokenizerHash: String = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH,
        ) = TrainingModelConfig(
            layers = 19,
            heads = 2,
            tokens = 32,
            dimension = 32,
            feedForwardDimension = 32,
            vocabularySize = 1024,
            batchSize = 8,
            learningRate = 0.003f,
            checkpointInterval = 250,
            seed = 1,
            tokenizerId = "byte_bpe",
            tokenizerHash = tokenizerHash,
        )
    }
}

/**
 * A training plan is deliberately separate from the editable model identity.
 * The UI displays this plan; it does not invent a target step count.
 */
data class TrainingPlan(
    val modelConfig: TrainingModelConfig,
    val targetSteps: Int,
    val checkpointFormat: String,
    val checkpointFormatVersion: Int,
) {
    init {
        require(targetSteps > 0) { "targetSteps must be positive" }
        require(checkpointFormat.isNotBlank()) { "checkpointFormat must not be blank" }
        require(checkpointFormatVersion > 0) { "checkpointFormatVersion must be positive" }
    }

    companion object {
        fun forConfig(config: TrainingModelConfig): TrainingPlan {
            require(SupportedTrainingModelPolicy.validationError(config) == null) {
                SupportedTrainingModelPolicy.validationError(config) ?: "unsupported model configuration"
            }
            val checkpoint = CheckpointFormatPolicy.forConfig(config)
            return TrainingPlan(config, 8_000, checkpoint.format, checkpoint.version)
        }

        /** Canonical long-training ceiling used by the existing Nicopedia run. */
        val NICOPEDIA_L19 = TrainingPlan(
            modelConfig = TrainingModelConfig.NICOPEDIA_L19,
            targetSteps = 8_000,
            checkpointFormat = "NPRTCKPTV2",
            checkpointFormatVersion = 2,
        )
    }
}
