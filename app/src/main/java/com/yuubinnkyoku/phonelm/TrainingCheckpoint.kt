package com.yuubinnkyoku.phonelm

data class TrainingCheckpointMetadata(
    val uri: String,
    val completedStep: Int,
    val modelConfig: TrainingModelConfig,
    val format: String,
    val formatVersion: Int,
    val finite: Boolean,
    val createdAtMs: Long,
    val datasetIdentity: String? = null,
) {
    init {
        require(uri.isNotBlank()) { "checkpoint uri must not be blank" }
        require(completedStep > 0) { "checkpoint step must be positive" }
        require(format.isNotBlank()) { "checkpoint format must not be blank" }
        require(formatVersion > 0) { "checkpoint format version must be positive" }
        require(createdAtMs >= 0) { "checkpoint creation time must be non-negative" }
    }
}

/** Opaque request; parameter and Adam payloads remain in the native V2 codec. */
data class TrainingResumeRequest(val checkpoint: TrainingCheckpointMetadata)

interface TrainingCheckpointStore {
    fun list(): List<TrainingCheckpointMetadata>
    fun save(metadata: TrainingCheckpointMetadata)
    fun archive(metadata: TrainingCheckpointMetadata)

    /** Resolve an opaque metadata id to an app-private native checkpoint path. */
    fun resolveNativePath(metadata: TrainingCheckpointMetadata): String? = null

    /** Whether the opaque record has a usable native payload for Resume UI. */
    fun isUsableForResume(metadata: TrainingCheckpointMetadata): Boolean = true

    /** Register a native-created checkpoint without exposing its path to UI state. */
    fun registerNativePath(metadata: TrainingCheckpointMetadata, path: String): Boolean = false

    /** App-private checkpoint payloads available for generation inspection. */
    fun listNativeCheckpointPaths(): List<String> = emptyList()
}

class InMemoryTrainingCheckpointStore : TrainingCheckpointStore {
    private val checkpoints = linkedMapOf<String, TrainingCheckpointMetadata>()
    override fun list(): List<TrainingCheckpointMetadata> = checkpoints.values.toList()
    override fun save(metadata: TrainingCheckpointMetadata) { checkpoints[metadata.uri] = metadata }
    override fun archive(metadata: TrainingCheckpointMetadata) { checkpoints.remove(metadata.uri) }
}

sealed interface TrainingCheckpointSelection {
    data class Selected(val checkpoint: TrainingCheckpointMetadata) : TrainingCheckpointSelection
    data object None : TrainingCheckpointSelection
    data class Incompatible(val reason: String) : TrainingCheckpointSelection
}

object TrainingCheckpointCatalog {
    /** Newest completed step first; creation time and URI make ties deterministic. */
    fun sortedNewestFirst(checkpoints: Iterable<TrainingCheckpointMetadata>): List<TrainingCheckpointMetadata> =
        checkpoints.sortedWith(compareByDescending<TrainingCheckpointMetadata> { it.completedStep }
            .thenByDescending { it.createdAtMs }.thenBy { it.uri })

    /** Fails closed: a present but incompatible checkpoint is never silently skipped. */
    fun latestCompatible(
        checkpoints: Iterable<TrainingCheckpointMetadata>,
        expectedConfig: TrainingModelConfig,
        expectedFormat: String,
        expectedFormatVersion: Int,
        expectedDatasetIdentity: String? = null,
        expectedTotalSteps: Int? = null,
    ): TrainingCheckpointSelection {
        val latest = sortedNewestFirst(checkpoints).firstOrNull() ?: return TrainingCheckpointSelection.None
        if (!latest.finite) return TrainingCheckpointSelection.Incompatible("latest checkpoint is not finite")
        if (latest.modelConfig != expectedConfig) return TrainingCheckpointSelection.Incompatible("latest checkpoint model configuration differs")
        if (latest.format != expectedFormat || latest.formatVersion != expectedFormatVersion) {
            return TrainingCheckpointSelection.Incompatible("latest checkpoint format differs")
        }
        if (expectedTotalSteps != null && latest.completedStep >= expectedTotalSteps) {
            return TrainingCheckpointSelection.Incompatible("latest checkpoint step is outside the resume range")
        }
        if (expectedDatasetIdentity != null && latest.datasetIdentity != expectedDatasetIdentity) {
            return TrainingCheckpointSelection.Incompatible("latest checkpoint dataset identity differs")
        }
        return TrainingCheckpointSelection.Selected(latest)
    }
}
