package com.yuubinnkyoku.phonelm

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import android.provider.OpenableColumns
import java.util.Locale

/** SAF adapter: full NPRTBYTEV1 identity validation runs only on the caller's IO executor. */
class AndroidTrainingDatasetUriStore(private val context: Context) : TrainingDatasetUriStore {
    private val resolver = context.contentResolver

    override fun persistReadAccess(uri: String): Result<TrainingDataset> = runCatching {
        val parsed = Uri.parse(uri)
        require(parsed.scheme == "content") { "dataset must be a content:// URI" }
        val alreadyPersisted = resolver.persistedUriPermissions.any { it.uri == parsed && it.isReadPermission }
        var grantedHere = false
        try {
            if (!alreadyPersisted) {
                resolver.takePersistableUriPermission(parsed, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                grantedHere = true
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
            // Selection runs on the Activity's IO executor.  Calculate the
            // content identity now so Resume can be fail-closed after restart;
            // the backend still stages and verifies the bytes again before JNI.
            val inspection = resolver.openInputStream(parsed)?.use {
                NicopediaCacheInspector.inspect(it, TrainingPlan.NICOPEDIA_L19.modelConfig)
            } ?: error("dataset provider returned no readable stream")
            TrainingDataset(uri, displayName, inspection.identity)
        } catch (failure: Throwable) {
            if (grantedHere) {
                runCatching {
                    resolver.releasePersistableUriPermission(
                        parsed,
                        android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                    )
                }
            }
            throw failure
        }
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
        check(prefs.edit()
            .putString(KEY_URI, dataset.uri)
            .putString(KEY_NAME, dataset.displayName)
            .putString(KEY_IDENTITY, dataset.identity)
            .commit()) { "dataset selection could not be persisted" }
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
    internal val nativeRootDirectory: java.io.File get() = checkpointDirectory

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
        val existing = list()
        if (existing.any { it.uri == metadata.uri && it.completedStep == metadata.completedStep }) return
        val next = existing.filterNot { it.uri == metadata.uri }.toMutableList().apply { add(metadata) }
        check(prefs.edit().putStringSet(KEY_ENTRIES, next.map(::encode).toSet()).commit()) {
            "checkpoint metadata could not be persisted"
        }
    }

    override fun archive(metadata: TrainingCheckpointMetadata) {
        check(prefs.edit().putStringSet(KEY_ENTRIES, list().filterNot { it.uri == metadata.uri }.map(::encode).toSet()).commit()) {
            "checkpoint metadata could not be archived"
        }
        val paths = nativePaths().filterKeys { it != metadata.uri }
        check(prefs.edit().putStringSet(KEY_PATHS, paths.map { (uri, path) -> encodePath(uri, path) }.toSet()).commit()) {
            "checkpoint path index could not be archived"
        }
    }

    override fun resolveNativePath(metadata: TrainingCheckpointMetadata): String? =
        nativePaths()[metadata.uri]?.takeIf { path ->
            runCatching {
                val root = checkpointDirectory.canonicalFile
                val candidate = java.io.File(path).canonicalFile
                candidate.path.startsWith(root.path + java.io.File.separator) &&
                    candidate.isFile && candidate.length() > 0L
            }.getOrDefault(false)
        }

    override fun isUsableForResume(metadata: TrainingCheckpointMetadata): Boolean =
        resolveNativePath(metadata) != null

    override fun registerNativePath(metadata: TrainingCheckpointMetadata, path: String): Boolean = runCatching {
        val root = checkpointDirectory.canonicalFile
        val candidate = java.io.File(path).canonicalFile
        require(candidate.path.startsWith(root.path + java.io.File.separator)) { "native checkpoint escaped app root" }
        require(candidate.isFile && candidate.length() > 0L) { "native checkpoint is not durable" }
        val next = nativePaths().toMutableMap().apply { put(metadata.uri, candidate.path) }
        require(prefs.edit().putStringSet(KEY_PATHS, next.map { (uri, value) -> encodePath(uri, value) }.toSet()).commit()) {
            "checkpoint path index could not be persisted"
        }
        true
    }.getOrDefault(false)

    override fun listNativeCheckpointPaths(): List<String> {
        val root = checkpointDirectory.canonicalFile
        return root.walkTopDown().filter { candidate ->
            runCatching {
                val canonical = candidate.canonicalFile
                canonical.isFile && canonical.extension.equals("ckpt", ignoreCase = true) &&
                    canonical.path.startsWith(root.path + java.io.File.separator)
            }.getOrDefault(false)
        }.map { it.canonicalPath }.toList()
    }

    private fun nativePaths(): Map<String, String> = prefs.getStringSet(KEY_PATHS, emptySet()).orEmpty().mapNotNull { encoded ->
        runCatching {
            val separator = encoded.indexOf('=')
            require(separator > 0)
            Uri.decode(encoded.substring(0, separator)) to Uri.decode(encoded.substring(separator + 1))
        }.getOrNull()
    }.toMap()

    private fun encodePath(uri: String, path: String): String = "${Uri.encode(uri)}=${Uri.encode(path)}"

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
        const val KEY_PATHS = "native_paths"
        const val DELIMITER = "|"
    }
}

class AndroidCpuProcessMetricSource : CpuProcessMetricSource {
    private var lastMemoryReadAtMs = -MEMORY_READ_INTERVAL_MS
    private var cachedMemoryBytes: Long? = null

    @Synchronized
    override fun read(): CpuProcessMetrics {
        val now = SystemClock.elapsedRealtime()
        if (now - lastMemoryReadAtMs >= MEMORY_READ_INTERVAL_MS) {
            cachedMemoryBytes = runCatching {
                Debug.getPss().takeIf { it >= 0 }?.toLong()?.times(1024L)
            }.getOrNull()
            lastMemoryReadAtMs = now
        }
        return CpuProcessMetrics(Process.getElapsedCpuTime(), memoryBytes = cachedMemoryBytes)
    }

    private companion object { const val MEMORY_READ_INTERVAL_MS = 1_000L }
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
