package com.yuubinnkyoku.phonelm

import java.io.File
import java.security.MessageDigest

/**
 * Identity recorded by the host after it has verified a research checkpoint.
 *
 * These values are deliberately all derived from bytes/header data, never from
 * an artifact filename.  The device repeats the comparison before publishing.
 */
internal data class GenerationCheckpointImportIdentity(
    val checkpointSha256: String,
    val checkpointSizeBytes: Long,
    val vocabulary: Int,
    val tokens: Int,
    val dimension: Int,
    val feedForwardDimension: Int,
    val layers: Int,
    val heads: Int,
    val step: Int,
    val seed: Long,
    val tokenizerKind: String,
    val tokenizerHash: String?,
    val parameterHash: String,
)

/** App-private locations and the host-verified identity needed to publish an import. */
internal data class GenerationCheckpointImportRequest(
    val importId: String,
    val stagingRoot: File,
    val productionRoot: File,
    val expected: GenerationCheckpointImportIdentity,
)

internal data class GenerationCheckpointImportResult(
    val importId: String,
    val publishedDirectory: File,
    val checkpoint: GenerationCheckpoint,
    val checkpointSha256: String,
    val idempotent: Boolean,
)

/**
 * Publishes an already-transferred checkpoint from a private staging directory.
 * It owns neither TrainingCheckpointStore metadata nor its preference index, so
 * an import can be visible to Generation without becoming a resume candidate.
 */
internal object GenerationCheckpointImport {
    private const val CHECKPOINT_FILE = "model.ckpt"
    private const val BPE_MODEL_FILE = "byte-bpe-v1024.model"
    private val importIdPattern = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

    @Synchronized
    fun publish(request: GenerationCheckpointImportRequest): GenerationCheckpointImportResult {
        validateImportId(request.importId)
        val stagingRoot = canonicalDirectory(request.stagingRoot, "staging root")
        val productionRoot = canonicalDirectory(request.productionRoot, "production root")
        require(stagingRoot != productionRoot) { "staging and production roots must differ" }
        val staging = directChild(stagingRoot, "${request.importId}.tmp")
        try {
            require(staging.isDirectory) { "staging import directory is missing" }
            val verified = verifyDirectory(staging, request.expected)
            // Production is not mutated until staging has passed the complete
            // device-side inspection above.
            val destinationParent = directChild(productionRoot, "imported", create = true)
            val destination = directChild(destinationParent, request.importId, create = false)

            if (destination.exists()) {
                require(destination.isDirectory) { "published import target is not a directory" }
                val existing = verifyDirectory(destination, request.expected)
                require(existing.sha256 == verified.sha256) { "import id already contains different checkpoint bytes" }
                removeStaging(staging, stagingRoot)
                return GenerationCheckpointImportResult(
                    request.importId, destination, existing.checkpoint, existing.sha256, idempotent = true,
                )
            }

            if (!staging.renameTo(destination)) {
                // Another publisher may have won the destination race between
                // the existence check above and rename.  Re-verify that
                // directory instead of overwriting it or treating equal bytes
                // as a conflict.
                if (destination.isDirectory) {
                    val existing = verifyDirectory(destination, request.expected)
                    require(existing.sha256 == verified.sha256) {
                        "import id already contains different checkpoint bytes"
                    }
                    removeStaging(staging, stagingRoot)
                    return GenerationCheckpointImportResult(
                        request.importId, destination, existing.checkpoint, existing.sha256, idempotent = true,
                    )
                }
                error("atomic import publish rename failed")
            }
            return GenerationCheckpointImportResult(
                request.importId, destination, verified.checkpoint, verified.sha256, idempotent = false,
            )
        } catch (error: Throwable) {
            // Never remove a production directory on a failed import.  Staging is
            // disposable only after it has been proven to be its exact direct child.
            removeStaging(staging, stagingRoot)
            throw error
        }
    }

    private fun verifyDirectory(
        directory: File,
        expected: GenerationCheckpointImportIdentity,
    ): VerifiedDirectory {
        val checkpointFile = directChild(directory, CHECKPOINT_FILE)
        require(checkpointFile.isFile) { "staged checkpoint is missing" }
        val sha256 = sha256(checkpointFile)
        require(sha256 == normalizeSha256(expected.checkpointSha256)) { "checkpoint SHA-256 differs" }
        require(checkpointFile.length() == expected.checkpointSizeBytes) { "checkpoint byte size differs" }
        val checkpoint = GenerationCheckpointInspector.inspect(checkpointFile)
        require(checkpoint.formatValid && checkpoint.finite && checkpoint.usable) {
            checkpoint.diagnostic ?: "checkpoint is not generation-compatible"
        }
        require(checkpoint.vocabulary == expected.vocabulary && checkpoint.tokens == expected.tokens &&
            checkpoint.dimension == expected.dimension && checkpoint.feedForwardDimension == expected.feedForwardDimension &&
            checkpoint.layers == expected.layers && checkpoint.heads == expected.heads &&
            checkpoint.step == expected.step && checkpoint.seed == expected.seed &&
            checkpoint.tokenizerKind == expected.tokenizerKind && checkpoint.tokenizerHash == expected.tokenizerHash &&
            checkpoint.parameterHash == expected.parameterHash) { "checkpoint header identity differs" }

        if (checkpoint.vocabulary == 1024) {
            val tokenizer = directChild(directory, BPE_MODEL_FILE)
            require(tokenizer.isFile) { "canonical byte-BPE tokenizer model is missing" }
            require(sha256(tokenizer) == checkpoint.tokenizerHash) { "tokenizer model hash differs" }
            require(directory.listFiles().orEmpty().map { it.name }.toSet() == setOf(CHECKPOINT_FILE, BPE_MODEL_FILE)) {
                "import directory contains unexpected entries"
            }
        } else {
            require(checkpoint.tokenizerHash == null) { "unexpected tokenizer hash for legacy checkpoint" }
            require(directory.listFiles().orEmpty().map { it.name }.toSet() == setOf(CHECKPOINT_FILE)) {
                "import directory contains unexpected entries"
            }
        }
        return VerifiedDirectory(checkpoint, sha256)
    }

    private fun validateImportId(importId: String) {
        require(importId.matches(importIdPattern) && importId != "." && importId != "..") {
            "import id is invalid"
        }
    }

    private fun canonicalDirectory(path: File, description: String): File {
        val canonical = path.canonicalFile
        require(canonical.isDirectory) { "$description is unavailable" }
        return canonical
    }

    private fun directChild(parent: File, name: String, create: Boolean = false): File {
        require(name.isNotBlank() && !name.contains('/') && !name.contains('\\')) { "invalid import path component" }
        val child = File(parent, name).canonicalFile
        require(child.parentFile == parent.canonicalFile) { "import path escaped its root" }
        require(child.name == name) { "import path component resolves through an alias" }
        if (create) require(child.isDirectory || child.mkdirs()) { "import directory is unavailable" }
        return child
    }

    private fun removeStaging(staging: File, stagingRoot: File) {
        if (!staging.exists()) return
        val canonical = runCatching { staging.canonicalFile }.getOrNull() ?: return
        if (canonical.parentFile != stagingRoot.canonicalFile || !canonical.name.endsWith(".tmp")) return
        canonical.walkBottomUp().forEach { entry ->
            require(entry.delete() || !entry.exists()) { "staging cleanup failed" }
        }
    }

    private fun normalizeSha256(value: String): String {
        val normalized = value.lowercase().let { if (it.startsWith("sha256:")) it else "sha256:$it" }
        require(normalized.matches(Regex("sha256:[0-9a-f]{64}"))) { "expected SHA-256 is invalid" }
        return normalized
    }

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        "sha256:" + digest.digest().joinToString("") { "%02x".format(it) }
    }

    private data class VerifiedDirectory(val checkpoint: GenerationCheckpoint, val sha256: String)
}
