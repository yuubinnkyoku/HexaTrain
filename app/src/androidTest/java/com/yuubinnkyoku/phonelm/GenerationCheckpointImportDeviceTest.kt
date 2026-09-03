package com.yuubinnkyoku.phonelm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/**
 * Headless bridge for the host import command.  It deliberately has no Activity
 * dependency: callers invoke this one test method with an `operation` argument.
 */
@RunWith(AndroidJUnit4::class)
class GenerationCheckpointImportDeviceTest {
    @Test
    fun runRequestedImportOperationWithoutActivity() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val args = InstrumentationRegistry.getArguments()
        val importId = requiredImportId(args.getString("importId"))
        val runId = requiredRunId(args.getString("runId"))
        val operation = args.getString("operation") ?: throw IllegalArgumentException("operation is required")
        require(operation == "publish" || operation == "smoke") { "operation must be publish or smoke" }
        val reportDirectory = File(context.filesDir, RESULTS_DIRECTORY)
        val reportFile = File(reportDirectory, "$importId-$runId.txt")

        val report = linkedMapOf<String, String>(
            "operation" to operation,
            "import_id" to importId,
            "run_id" to runId,
            "activity_launched" to "false",
            "operation_result_fresh" to "true",
            // Publication is only true after the directory rename has
            // completed successfully.  A failed publish report must never
            // look like an atomic commit.
            "atomic_publish" to "false",
        )
        try {
            val expected = readExpectedIdentity(args)
            val checkpointStore = AndroidTrainingCheckpointStore(context)
            when (operation) {
                "publish" -> publish(context.filesDir, checkpointStore, importId, expected, report)
                "smoke" -> smoke(checkpointStore, context, importId, expected, report)
            }
            report["status"] = "SUCCESS"
            writeAtomicReport(reportDirectory, reportFile, report)
        } catch (failure: Throwable) {
            report["status"] = "FAILED"
            report["error"] = safeReportValue(failure.message ?: failure.javaClass.simpleName)
            runCatching { writeAtomicReport(reportDirectory, reportFile, report) }
            throw AssertionError("generation import $operation failed: ${failure.message}", failure)
        }
    }

    private fun publish(
        filesDir: File,
        checkpointStore: AndroidTrainingCheckpointStore,
        importId: String,
        expected: GenerationCheckpointImportIdentity,
        report: MutableMap<String, String>,
    ) {
        val result = GenerationCheckpointImport.publish(
            GenerationCheckpointImportRequest(
                importId = importId,
                stagingRoot = File(filesDir, STAGING_DIRECTORY),
                productionRoot = checkpointStore.nativeRootDirectory,
                expected = expected,
            ),
        )
        val listed = AppPrivateGenerationCheckpointRepository(checkpointStore).listCheckpoints()
            .singleOrNull { it.path == File(result.publishedDirectory, CHECKPOINT_FILE).canonicalPath }
            ?: error("published checkpoint is not visible to Production Generation")
        require(listed.usable && listed.compatibility == GenerationCheckpointCompatibility.COMPATIBLE) {
            listed.diagnostic ?: "published checkpoint is not generation-compatible"
        }
        require(listed.parameterHash == expected.parameterHash) { "published parameter hash differs" }
        require(listed.step == expected.step && listed.seed == expected.seed) { "published checkpoint identity differs" }
        addCheckpointReport(report, listed, result.checkpointSha256)
        requireTrainingIsolated(checkpointStore, result.publishedDirectory, report)
        report["published_path"] = result.publishedDirectory.canonicalPath
        report["visible"] = "true"
        report["compatibility"] = "compatible"
        report["atomic_publish"] = "true"
        report["atomic"] = "true"
        report["idempotent"] = result.idempotent.toString()
    }

    private fun smoke(
        checkpointStore: AndroidTrainingCheckpointStore,
        context: android.content.Context,
        importId: String,
        expected: GenerationCheckpointImportIdentity,
        report: MutableMap<String, String>,
    ) {
        val published = File(File(checkpointStore.nativeRootDirectory, "imported"), importId)
        val expectedPath = File(published, CHECKPOINT_FILE).canonicalPath
        val checkpoint = AppPrivateGenerationCheckpointRepository(checkpointStore).listCheckpoints()
            .singleOrNull { it.path == expectedPath }
            ?: error("published checkpoint is not visible to Production Generation")
        require(checkpoint.usable && checkpoint.compatibility == GenerationCheckpointCompatibility.COMPATIBLE) {
            checkpoint.diagnostic ?: "published checkpoint is not generation-compatible"
        }
        require(checkpoint.step == expected.step && checkpoint.seed == expected.seed &&
            checkpoint.parameterHash == expected.parameterHash && checkpoint.fileSizeBytes == expected.checkpointSizeBytes) {
            "published checkpoint identity differs"
        }
        val checkpointSha256 = sha256(File(checkpoint.path))
        require(checkpointSha256 == expected.checkpointSha256) { "published checkpoint SHA-256 differs" }
        val result = NativeHtpGenerationBackend(context).generate(
            GenerationRequest(prompt = "import smoke", mode = GenerationMode.GREEDY, maxNewBytes = 8),
            checkpoint,
        )
        require(result.backend == "HTP") { "Production Generation did not use HTP" }
        require(result.finite) { "Production Generation produced non-finite tensors" }
        require(!result.cpuFallback) { "Production Generation used CPU fallback" }
        require(result.qnnExecuteSuccesses > 0L) { "Production Generation did not complete a QNN execute" }
        require(result.qnnExecuteFailures == 0L) { "Production Generation reported QNN execute failures" }
        require(result.byteCount in 0..8) { "Production Generation exceeded the requested 8-byte limit" }
        addCheckpointReport(report, checkpoint, checkpointSha256)
        requireTrainingIsolated(checkpointStore, published, report)
        report["visible"] = "true"
        report["compatibility"] = "compatible"
        report["atomic"] = "false"
        report["smoke_mode"] = "greedy"
        report["smoke_max_new_bytes"] = "8"
        report["htp"] = "true"
        report["qnn_return_code_success"] = "true"
        report["output_tensors_finite"] = "true"
        report["cpu_fallback"] = "false"
        report["qnn_execute_successes"] = result.qnnExecuteSuccesses.toString()
        report["qnn_execute_failures"] = "0"
        report["generated_byte_count"] = result.byteCount.toString()
    }

    private fun addCheckpointReport(report: MutableMap<String, String>, checkpoint: GenerationCheckpoint, sha256: String) {
        report["device_checkpoint_sha256"] = sha256
        report["device_checkpoint_size_bytes"] = checkpoint.fileSizeBytes.toString()
        report["checkpoint_sha256"] = sha256.removePrefix("sha256:")
        report["checkpoint_size"] = checkpoint.fileSizeBytes.toString()
        report["checkpoint_format"] = checkpoint.format
        report["header_vocabulary"] = checkpoint.vocabulary.toString()
        report["header_tokens"] = checkpoint.tokens.toString()
        report["header_dimension"] = checkpoint.dimension.toString()
        report["header_feedforward"] = checkpoint.feedForwardDimension.toString()
        report["header_layers"] = checkpoint.layers.toString()
        report["header_heads"] = checkpoint.heads.toString()
        report["header_step"] = checkpoint.step.toString()
        report["header_seed"] = checkpoint.seed.toString()
        report["finite"] = checkpoint.finite.toString()
        report["tokenizer_kind"] = checkpoint.tokenizerKind
        report["tokenizer_hash"] = checkpoint.tokenizerHash.orEmpty()
        report["tokenizer_sha256"] = checkpoint.tokenizerHash.orEmpty().removePrefix("sha256:")
        report["parameter_hash"] = checkpoint.parameterHash.orEmpty()
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

    private fun requireTrainingIsolated(
        checkpointStore: AndroidTrainingCheckpointStore,
        publishedDirectory: File,
        report: MutableMap<String, String>,
    ) {
        val publishedPath = publishedDirectory.canonicalPath + File.separator
        val visibleToResume = checkpointStore.list().any { metadata ->
            checkpointStore.resolveNativePath(metadata)?.let { path ->
                runCatching { File(path).canonicalPath.startsWith(publishedPath) }.getOrDefault(false)
            } == true
        }
        require(!visibleToResume) { "imported checkpoint is visible to Training resume" }
        report["training_resume"] = visibleToResume.toString()
    }

    private fun readExpectedIdentity(args: android.os.Bundle): GenerationCheckpointImportIdentity =
        GenerationCheckpointImportIdentity(
            checkpointSha256 = requiredSha256(args.required("expectedCheckpointSha256")),
            checkpointSizeBytes = args.positiveLong("expectedCheckpointSizeBytes"),
            vocabulary = args.positiveInt("expectedV"),
            tokens = args.positiveInt("expectedT"),
            dimension = args.positiveInt("expectedD"),
            feedForwardDimension = args.positiveInt("expectedFfn"),
            layers = args.positiveInt("expectedL"),
            heads = args.positiveInt("expectedH"),
            step = args.nonNegativeInt("expectedStep"),
            seed = args.nonNegativeLong("expectedSeed"),
            tokenizerKind = args.required("expectedTokenizerKind").also {
                require(it == "byte" || it == "byte_bpe") { "expectedTokenizerKind is invalid" }
            },
            tokenizerHash = args.getString("expectedTokenizerHash")?.takeIf { it.isNotBlank() }?.let(::requiredSha256),
            parameterHash = args.required("expectedParameterHash").also {
                require(it.matches(Regex("fnv1a64:[0-9a-f]{16}"))) { "expectedParameterHash is invalid" }
            },
        ).also { identity ->
            require((identity.vocabulary == 1024) == (identity.tokenizerKind == "byte_bpe")) {
                "vocabulary and tokenizer kind are inconsistent"
            }
            require((identity.vocabulary == 1024) == (identity.tokenizerHash != null)) {
                "vocabulary and tokenizer hash are inconsistent"
            }
        }

    private fun android.os.Bundle.required(key: String): String =
        getString(key)?.takeIf { it.isNotBlank() && it.length <= 256 && it.none { c -> c.isWhitespace() } }
            ?: throw IllegalArgumentException("$key is required")

    private fun android.os.Bundle.positiveInt(key: String): Int = required(key).toIntOrNull()
        ?.takeIf { it > 0 } ?: throw IllegalArgumentException("$key must be positive")

    private fun android.os.Bundle.nonNegativeInt(key: String): Int = required(key).toIntOrNull()
        ?.takeIf { it >= 0 } ?: throw IllegalArgumentException("$key must be non-negative")

    private fun android.os.Bundle.positiveLong(key: String): Long = required(key).toLongOrNull()
        ?.takeIf { it > 0L } ?: throw IllegalArgumentException("$key must be positive")

    private fun android.os.Bundle.nonNegativeLong(key: String): Long = required(key).toLongOrNull()
        ?.takeIf { it >= 0L } ?: throw IllegalArgumentException("$key must be non-negative")

    private fun requiredImportId(value: String?): String = value.orEmpty().also {
        require(it.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}"))) { "importId is invalid" }
    }

    private fun requiredRunId(value: String?): String = value.orEmpty().also {
        require(it.matches(Regex("[a-f0-9]{32}"))) { "runId is invalid" }
    }

    private fun requiredSha256(value: String): String {
        val normalized = value.lowercase().let { if (it.startsWith("sha256:")) it else "sha256:$it" }
        require(normalized.matches(Regex("sha256:[0-9a-f]{64}"))) { "SHA-256 is invalid" }
        return normalized
    }

    private fun writeAtomicReport(directory: File, destination: File, values: Map<String, String>) {
        require(directory.isDirectory || directory.mkdirs()) { "result directory is unavailable" }
        val temporary = File(directory, "${destination.name}.tmp")
        FileOutputStream(temporary).use { stream ->
            stream.write(values.entries.joinToString("\n") { (key, value) -> "$key=${safeReportValue(value)}" }
                .plus("\n").toByteArray(StandardCharsets.UTF_8))
            stream.fd.sync()
        }
        Files.move(temporary.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    }

    private fun safeReportValue(value: String): String = value.replace(Regex("[\\r\\n=]"), " ").take(512)

    private companion object {
        const val STAGING_DIRECTORY = "generation-import-staging"
        const val RESULTS_DIRECTORY = "generation-import-results"
        const val CHECKPOINT_FILE = "model.ckpt"
    }
}
