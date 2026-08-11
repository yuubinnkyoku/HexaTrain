package com.yuubinnkyoku.phonelm

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.Process
import android.provider.OpenableColumns
import java.util.Locale

/** SAF adapter: validates only provider access and metadata, never the full file on the UI thread. */
class AndroidTrainingDatasetUriStore(private val context: Context) : TrainingDatasetUriStore {
    private val resolver = context.contentResolver

    override fun persistReadAccess(uri: String): Result<TrainingDataset> = runCatching {
        val parsed = Uri.parse(uri)
        require(parsed.scheme == "content") { "dataset must be a content:// URI" }
        if (resolver.persistedUriPermissions.none { it.uri == parsed && it.isReadPermission }) {
            resolver.takePersistableUriPermission(parsed, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        require(resolver.persistedUriPermissions.any { it.uri == parsed && it.isReadPermission }) {
            "persistable read permission is unavailable"
        }
        val displayName = resolver.query(
            parsed,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (!cursor.moveToFirst()) null
            else cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME))
        }
        TrainingDataset(uri, displayName)
    }

    override fun releaseReadAccess(uri: String): Result<Unit> = runCatching {
        val parsed = Uri.parse(uri)
        context.contentResolver.releasePersistableUriPermission(
            parsed,
            android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
        )
    }

    override fun validateReadAccess(dataset: TrainingDataset): Result<Unit> = runCatching {
        val parsed = Uri.parse(dataset.uri)
        require(parsed.scheme == "content") { "dataset must be a content:// URI" }
        val persisted = resolver.persistedUriPermissions.any {
            it.uri == parsed && it.isReadPermission
        }
        require(persisted) { "persisted read permission is missing" }
        resolver.openAssetFileDescriptor(parsed, "r")?.use { descriptor ->
            require(descriptor.length < 0L || descriptor.length > 0L) {
                "dataset is empty"
            }
        } ?: error("dataset provider returned no readable descriptor")
    }
}

class AndroidTrainingSelectionPersistence(context: Context) : TrainingSelectionPersistence {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    override fun loadDataset(): TrainingDataset? {
        val uri = prefs.getString(KEY_URI, null) ?: return null
        return TrainingDataset(uri, prefs.getString(KEY_NAME, null), prefs.getString(KEY_IDENTITY, null))
    }

    override fun saveDataset(dataset: TrainingDataset) {
        prefs.edit()
            .putString(KEY_URI, dataset.uri)
            .putString(KEY_NAME, dataset.displayName)
            .putString(KEY_IDENTITY, dataset.identity)
            .apply()
    }

    private companion object {
        const val PREFS = "phonelm_standalone_training"
        const val KEY_URI = "dataset_uri"
        const val KEY_NAME = "dataset_display_name"
        const val KEY_IDENTITY = "dataset_identity"
    }
}

/** Metadata-only app-private index; raw NPRTCKPTV2 bytes remain native-owned. */
class AndroidTrainingCheckpointStore(context: Context) : TrainingCheckpointStore {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    // Native integration writes opaque NPRTCKPTV2 files below this app-managed
    // directory. The path is never exposed to UI or persisted metadata.
    private val checkpointDirectory = context.getDir("standalone_training_checkpoints", Context.MODE_PRIVATE)

    init {
        check(checkpointDirectory.isDirectory || checkpointDirectory.mkdirs()) {
            "checkpoint directory is unavailable"
        }
    }

    override fun list(): List<TrainingCheckpointMetadata> {
        val entries = prefs.getStringSet(KEY_ENTRIES, emptySet()).orEmpty()
        return entries.mapNotNull { decode(it) }
    }

    override fun save(metadata: TrainingCheckpointMetadata) {
        val next = list().filterNot { it.uri == metadata.uri }.toMutableList().apply { add(metadata) }
        prefs.edit().putStringSet(KEY_ENTRIES, next.map(::encode).toSet()).apply()
    }

    override fun archive(metadata: TrainingCheckpointMetadata) {
        prefs.edit().putStringSet(KEY_ENTRIES, list().filterNot { it.uri == metadata.uri }.map(::encode).toSet()).apply()
    }

    private fun encode(value: TrainingCheckpointMetadata): String = listOf(
        value.uri,
        value.completedStep.toString(),
        value.format,
        value.formatVersion.toString(),
        value.finite.toString(),
        value.createdAtMs.toString(),
        value.modelConfig.compatibilityKey,
        value.datasetIdentity.orEmpty(),
    ).joinToString(DELIMITER) { Uri.encode(it) }

    private fun decode(encoded: String): TrainingCheckpointMetadata {
        return runCatching {
            val fields = encoded.split(DELIMITER).map(Uri::decode)
            require(fields.size >= 8) { "checkpoint metadata is truncated" }
            val config = TrainingModelConfig.NICOPEDIA_L19
            require(fields[6] == config.compatibilityKey) { "checkpoint config differs" }
            TrainingCheckpointMetadata(
                uri = fields[0],
                completedStep = fields[1].toInt(),
                modelConfig = config,
                format = fields[2],
                formatVersion = fields[3].toInt(),
                finite = fields[4].toBooleanStrict(),
                createdAtMs = fields[5].toLong(),
                datasetIdentity = fields[7].ifBlank { null },
            )
        }.getOrElse {
            // Keep malformed entries visible to the fail-closed catalog instead
            // of silently treating them as if no checkpoint existed.
            TrainingCheckpointMetadata(
                uri = "invalid-checkpoint:${encoded.hashCode()}",
                completedStep = Int.MAX_VALUE,
                modelConfig = TrainingModelConfig.NICOPEDIA_L19,
                format = "INVALID",
                formatVersion = 1,
                finite = false,
                createdAtMs = Long.MAX_VALUE,
            )
        }
    }

    private companion object {
        const val PREFS = "phonelm_standalone_training_checkpoints"
        const val KEY_ENTRIES = "metadata"
        const val DELIMITER = "|"
    }
}

class AndroidCpuProcessMetricSource : CpuProcessMetricSource {
    override fun read(): CpuProcessMetrics = CpuProcessMetrics(Process.getElapsedCpuTime())
}

/** Reuses the existing user-run notification and dataSync foreground service. */
class AndroidTrainingRunLifecycle(context: Context) : TrainingRunLifecycle {
    private val notifications = LiveUpdateNotificationController(context.applicationContext)

    @Synchronized
    override fun onStarted(totalSteps: Int) {
        notifications.onRunStarted("学習", totalSteps.toLong())
    }

    @Synchronized
    override fun onStateChanged(state: TrainingState) {
        val progress = state.progress ?: return
        val event = RunProgress.Step(
            completed = progress.completedSteps.toLong(),
            total = progress.totalSteps.toLong(),
            loss = progress.loss?.toDouble(),
            phase = state.phase.name,
        )
        notifications.onProgress(event)
    }

    @Synchronized
    override fun onFinished(state: TrainingState) {
        when (state.phase) {
            TrainingPhase.COMPLETED -> notifications.onProgress(
                RunProgress.Completed(state.progress?.loss?.let { "loss ${String.format(Locale.ROOT, "%.6f", it)}" }),
            )
            TrainingPhase.INTERRUPTED -> notifications.onProgress(RunProgress.Cancelled)
            TrainingPhase.ERROR -> notifications.onProgress(RunProgress.Failed(state.message ?: "training failed"))
            else -> Unit
        }
    }
}
