package com.yuubinnkyoku.phonelm

/** An opaque content URI. This layer deliberately does not depend on android.net.Uri. */
data class TrainingDataset(
    val uri: String,
    val displayName: String? = null,
    /** Stable provider/content digest supplied by the native/import adapter when available. */
    val identity: String? = null,
) {
    init { require(uri.isNotBlank()) { "dataset uri must not be blank" } }
}

interface TrainingDatasetUriStore {
    /** Persists read access and returns the canonical URI to store with a session. */
    fun persistReadAccess(uri: String): Result<TrainingDataset>
    fun releaseReadAccess(uri: String): Result<Unit>

    /** Lightweight permission/provider validation; never reads the whole dataset. */
    fun validateReadAccess(dataset: TrainingDataset): Result<Unit> = Result.success(Unit)
}

/** JVM/test implementation. Production Android uses the SAF adapter. */
class InMemoryTrainingDatasetUriStore : TrainingDatasetUriStore {
    private val selected = linkedSetOf<String>()

    override fun persistReadAccess(uri: String): Result<TrainingDataset> = runCatching {
        require(uri.startsWith("content://")) { "dataset must be a content:// URI" }
        selected += uri
        TrainingDataset(uri)
    }

    override fun releaseReadAccess(uri: String): Result<Unit> = runCatching { selected -= uri }

    override fun validateReadAccess(dataset: TrainingDataset): Result<Unit> = runCatching {
        require(dataset.uri.startsWith("content://")) { "dataset must be a content:// URI" }
        require(dataset.uri in selected || selected.isEmpty()) { "dataset URI was not selected" }
    }
}

object UnavailableTrainingDatasetUriStore : TrainingDatasetUriStore {
    private const val MESSAGE = "Persistent dataset URI access is unavailable on this platform"
    override fun persistReadAccess(uri: String): Result<TrainingDataset> = Result.failure(UnsupportedOperationException(MESSAGE))
    override fun releaseReadAccess(uri: String): Result<Unit> = Result.failure(UnsupportedOperationException(MESSAGE))
    override fun validateReadAccess(dataset: TrainingDataset): Result<Unit> = Result.failure(UnsupportedOperationException(MESSAGE))
}
