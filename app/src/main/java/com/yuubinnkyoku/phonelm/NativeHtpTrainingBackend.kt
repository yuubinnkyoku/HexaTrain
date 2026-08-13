package com.yuubinnkyoku.phonelm

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FilterInputStream
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID

/**
 * The native Nicopedia loader consumes NPRTBYTEV1, not arbitrary text.  This
 * inspector deliberately validates the wire format while calculating a
 * content identity; it never tokenizes, rewrites, or guesses a filesystem
 * path from the SAF URI.
 */
internal data class NicopediaCacheInspection(
    val context: Int,
    val vocabulary: Int,
    val recordCount: Long,
    val byteCount: Long,
    val sha256: String,
    val nativeFnv1a64: String,
    val trainingOrderHash: String,
) {
    val identity: String
        get() = "NPRTBYTEV1;context=$context;vocab=$vocabulary;records=$recordCount;" +
            "bytes=$byteCount;sha256=$sha256;$nativeFnv1a64;" +
            "order_seed=20260806;training_order=$trainingOrderHash"
}

internal object NicopediaCacheInspector {
    private const val MAGIC = "NPRTBYTEV1\n"
    private const val MAX_RECORDS = 10_000_000L
    internal const val MAX_BYTES = 1_536L * 1024L * 1024L
    // FNV-1a 64-bit offset basis 0xCBF29CE484222325 = 14695981039346656037,
    // stored as signed long: -3750763034362895579.
    private const val FNV_OFFSET = -3750763034362895579L
    private const val FNV_PRIME = 1099511628211L

    fun inspectFile(file: File, expected: TrainingModelConfig): NicopediaCacheInspection =
        file.inputStream().use { inspect(it, expected) }

    fun inspect(input: InputStream, expected: TrainingModelConfig): NicopediaCacheInspection {
        require(expected.tokens == 32 && expected.vocabularySize == 256) {
            "native Nicopedia cache contract only supports T32/V256"
        }
        val digest = MessageDigest.getInstance("SHA-256")
        val counted = DigestCountingInputStream(input, digest)
        DataInputStream(BufferedInputStream(counted, 64 * 1024)).use { data ->
            val magic = ByteArray(MAGIC.length)
            data.readFully(magic)
            require(String(magic, Charsets.US_ASCII) == MAGIC) { "NPRT_CACHE_MAGIC" }
            val context = data.readInt()
            val vocabulary = data.readInt()
            val count = data.readLong()
            require(context in 8..256 && vocabulary == 256 && count in 1..MAX_RECORDS) {
                "NPRT_CACHE_HEADER_INVALID"
            }
            require(context == expected.tokens && vocabulary == expected.vocabularySize) {
                "NPRT_CACHE_CONFIG_MISMATCH"
            }
            val recordBytes = 8L + context + 1L
            val expectedBytes = MAGIC.length.toLong() + 4L + 4L + 8L + count * recordBytes
            require(expectedBytes <= MAX_BYTES) { "NPRT_CACHE_TOO_LARGE" }
            var fnv = FNV_OFFSET
            fun updateLongLittleEndian(value: Long) {
                var v = value
                repeat(8) {
                    fnv = (fnv xor (v and 0xffL)) * FNV_PRIME
                    v = v ushr 8
                }
            }
            fun updateIntLittleEndian(value: Int) = updateLongLittleEndian(value.toLong()).also {
                // The native loader hashes uint32_t using the device's little
                // endian representation. Undo the four extra zero bytes.
                // This helper is replaced below for clarity at call sites.
            }
            // Recompute the u32 fields exactly as native uint32_t bytes.
            fnv = FNV_OFFSET
            var contextValue = context
            repeat(4) {
                fnv = (fnv xor (contextValue.toLong() and 0xffL)) * FNV_PRIME
                contextValue = contextValue ushr 8
            }
            var vocabularyValue = vocabulary
            repeat(4) {
                fnv = (fnv xor (vocabularyValue.toLong() and 0xffL)) * FNV_PRIME
                vocabularyValue = vocabularyValue ushr 8
            }
            updateLongLittleEndian(count)
            val window = ByteArray(context + 1)
            repeat(count.toInt()) {
                val articleHash = data.readLong()
                data.readFully(window)
                updateLongLittleEndian(articleHash)
                window.forEach { byte -> fnv = (fnv xor (byte.toLong() and 0xffL)) * FNV_PRIME }
            }
            require(data.read() < 0) { "NPRT_CACHE_TRAILING_BYTES" }
            require(counted.count == expectedBytes) { "NPRT_CACHE_LENGTH_MISMATCH" }
            val sha = digest.digest().joinToString("") { "%02x".format(it) }
            val fnvText = "fnv1a64:" + java.lang.Long.toUnsignedString(fnv, 16).padStart(16, '0')
            val orderHash = trainingOrderHash(count)
            return NicopediaCacheInspection(context, vocabulary, count, counted.count, sha, fnvText, orderHash)
        }
    }

    /** Mirrors native nprtTrainingOrder/orderHash for the canonical T32 plan. */
    private fun trainingOrderHash(recordCount: Long): String {
        var state = 20_260_806L
        var hash = FNV_OFFSET
        val total = 8_000L * 8L
        repeat(total.toInt()) { index ->
            state = splitMix(state + index.toLong())
            val selected = java.lang.Long.remainderUnsigned(state, recordCount)
            var value = selected
            repeat(8) {
                hash = (hash xor (value and 0xffL)) * FNV_PRIME
                value = value ushr 8
            }
        }
        return "fnv1a64:" + java.lang.Long.toUnsignedString(hash, 16).padStart(16, '0')
    }

    private fun splitMix(input: Long): Long {
        var value = input + -7046029254386353131L
        value = (value xor (value ushr 30)) * -4658895280553007687L
        value = (value xor (value ushr 27)) * -7723592293110705685L
        return value xor (value ushr 31)
    }

    private class DigestCountingInputStream(input: InputStream, private val digest: MessageDigest) :
        FilterInputStream(input) {
        var count: Long = 0
            private set

        override fun read(): Int {
            val value = super.read()
            if (value >= 0) {
                digest.update(value.toByte())
                count++
            }
            return value
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val read = super.read(buffer, offset, length)
            if (read > 0) {
                digest.update(buffer, offset, read)
                count += read
            }
            return read
        }
    }
}

internal data class StagedNicopediaDataset(
    val directory: File,
    val cacheFile: File,
    val inspection: NicopediaCacheInspection,
)

/** Background-only SAF importer used by the JNI backend. */
internal class AndroidNicopediaCacheStager(
    private val resolver: ContentResolver,
    private val appRoot: File,
) {
    fun stage(
        dataset: TrainingDataset,
        destination: File,
        expected: TrainingModelConfig,
    ): Result<StagedNicopediaDataset> = runCatching {
        require(dataset.uri.startsWith("content://")) { "dataset must be a content:// URI" }
        require(!dataset.identity.isNullOrBlank()) {
            "dataset content identity is unavailable; reselect the SAF document"
        }
        val root = appRoot.canonicalFile
        val directory = destination.canonicalFile
        require(directory.path.startsWith(root.path + File.separator)) { "staging directory escaped app storage" }
        require(directory.mkdirs() || directory.isDirectory) { "cannot create staging directory" }
        val temporary = File(directory, ".train_pilot.bin.${UUID.randomUUID()}.part")
        try {
            resolver.openInputStream(Uri.parse(dataset.uri))?.use { input ->
                temporary.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var copied = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        copied += read
                        require(copied <= NicopediaCacheInspector.MAX_BYTES) {
                            "dataset exceeds the bounded staging limit"
                        }
                        output.write(buffer, 0, read)
                    }
                    output.fd.sync()
                }
            } ?: error("dataset provider returned no readable stream")
            val inspection = NicopediaCacheInspector.inspectFile(temporary, expected)
            if (dataset.identity != inspection.identity) {
                error("dataset content changed since selection")
            }
            val target = File(directory, "train_pilot.bin")
            moveAtomically(temporary, target)
            val verified = NicopediaCacheInspector.inspectFile(target, expected)
            require(verified.identity == inspection.identity) { "staged dataset verification changed" }
            StagedNicopediaDataset(directory, target, verified)
        } finally {
            temporary.delete()
        }
    }

    private fun moveAtomically(source: File, target: File) {
        runCatching {
            Files.move(
                source.toPath(), target.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        }.getOrElse {
            require(source.renameTo(target)) { "dataset staging rename failed" }
        }
    }
}

internal data class NativeTrainingFields(val values: Map<String, String>) {
    fun string(key: String): String? = values[key]
    fun int(key: String): Int? = values[key]?.toIntOrNull()
    fun long(key: String): Long? = values[key]?.toLongOrNull()
    fun double(key: String): Double? = values[key]?.toDoubleOrNull()
    fun bool(key: String): Boolean? = values[key]?.let {
        when (it.lowercase()) {
            "true", "1", "yes" -> true
            "false", "0", "no" -> false
            else -> null
        }
    }
}

internal object NativeTrainingFieldsParser {
    fun parse(text: String): NativeTrainingFields {
        val values = linkedMapOf<String, String>()
        text.lineSequence().forEach { line ->
            val separator = line.indexOf('=')
            if (separator <= 0) return@forEach
            val key = line.substring(0, separator).trim()
            val value = line.substring(separator + 1).trim()
            val previous = values.putIfAbsent(key, value)
            require(previous == null || previous == value) { "duplicate native field: $key" }
        }
        return NativeTrainingFields(values)
    }
}

/** Maps native progress phases; host-only scratch replay is intentionally ignored. */
internal fun classifyNativeTrainingPhase(nativePhase: String): TrainingPhase? = when (nativePhase) {
    "training" -> TrainingPhase.TRAINING
    "saving_checkpoint" -> TrainingPhase.SAVING_CHECKPOINT
    "cpu_replay" -> null
    else -> error("unknown native phase")
}

/** Builds an observation ratio only when both QNN execute timings were actually reported. */
internal fun observedHtpActivityWindow(
    observationWallUs: Double?,
    fusedExecuteUs: Double?,
    adamExecuteUs: Double?,
    fusedExecuteCount: Long?,
    adamExecuteCount: Long?,
): HtpActivityWindow? {
    val wallUs = observationWallUs?.takeIf { it.isFinite() && it > 0.0 } ?: return null
    val fusedUs = fusedExecuteUs?.takeIf { it.isFinite() && it >= 0.0 } ?: return null
    val adamUs = adamExecuteUs?.takeIf { it.isFinite() && it >= 0.0 } ?: return null
    val executeCount = if (fusedExecuteCount != null && adamExecuteCount != null) {
        fusedExecuteCount + adamExecuteCount
    } else null
    return HtpActivityWindow(
        startedAtMs = 0L,
        endedAtMs = kotlin.math.ceil(wallUs / 1000.0).toLong().coerceAtLeast(1L),
        executeDurationMs = (fusedUs + adamUs) / 1000.0,
        executeCount = executeCount,
    )
}

/** Production adapter over the existing synchronous JNI mode-100 entrypoint. */
class NativeHtpTrainingBackend(
    private val context: Context,
    private val checkpointStore: TrainingCheckpointStore,
) : TrainingBackend {
    private val lock = Any()
    private var running = false
    private var pendingStop = false
    private var nativeCallEntered = false
    private var runArmed = false
    private val checkpointRoot = context.getDir("standalone_training_checkpoints", Context.MODE_PRIVATE)
    private val stager = AndroidNicopediaCacheStager(context.contentResolver, checkpointRoot)

    override val supportsPause: Boolean get() = false
    override val supportsResume: Boolean get() = false

    override fun prepareForRun() {
        synchronized(lock) { runArmed = true }
    }

    override fun cancelPreparedRun() {
        synchronized(lock) {
            runArmed = false
            pendingStop = false
        }
    }

    override fun resolveTotalSteps(config: TrainingModelConfig, dataset: TrainingDataset): Int? =
        TrainingPlan.NICOPEDIA_L19.targetSteps.takeIf { config == TrainingPlan.NICOPEDIA_L19.modelConfig }

    override fun run(
        request: TrainingRequest,
        onProgress: (TrainingBackendProgress) -> Unit,
    ): TrainingBackendResult {
        val runId = UUID.randomUUID().toString()
        val arbiterOwner = "standalone:$runId"
        if (!NativeRunArbiter.tryAcquire(arbiterOwner)) {
            return TrainingBackendResult.Failed("another native run is already active")
        }
        var cancelBeforeStart = false
        synchronized(lock) {
            if (running) {
                NativeRunArbiter.release(arbiterOwner)
                return TrainingBackendResult.Failed("native HTP training is already running")
            }
            running = true
            runArmed = false
            if (pendingStop) {
                pendingStop = false
                running = false
                cancelBeforeStart = true
            }
        }
        if (cancelBeforeStart) {
            runCatching { NativeBridge.nativeClearStop() }
            NativeRunArbiter.release(arbiterOwner)
            return TrainingBackendResult.Cancelled(
                TrainingBackendProgress(0, request.totalSteps),
            )
        }
        return try {
            if (!BuildConfig.PHONELM_QNN_ENABLED) {
                return TrainingBackendResult.Failed("QNN HTP is disabled in this APK; CPU fallback is not permitted")
            }
            require(request.modelConfig == TrainingPlan.NICOPEDIA_L19.modelConfig) {
                "standalone Nicopedia native mode only supports the canonical L19/T32 preset"
            }
            require(request.totalSteps == TrainingPlan.NICOPEDIA_L19.targetSteps) {
                "standalone Nicopedia native mode requires the canonical 8000-step target"
            }
            request.validationError()?.let { return TrainingBackendResult.Failed(it) }
            runCatching { QnnEnvironment.prepare(context) }
                .getOrElse { return TrainingBackendResult.Failed("HTP environment preparation failed: ${it.message}") }
            val resumePath = request.resumeFrom?.let { checkpointStore.resolveNativePath(it) }
            if (request.resumeFrom != null && resumePath == null) {
                return TrainingBackendResult.Failed("resume checkpoint is not available in app storage")
            }
            val runDirectory = (resumePath?.let(::File)?.parentFile ?: File(checkpointRoot, runId)).canonicalFile
            val staged = stager.stage(request.dataset, runDirectory, request.modelConfig)
                .getOrElse { return TrainingBackendResult.Failed("dataset import/validation failed: ${it.message}") }
            val qnnStatus = runCatching { NativeBridge.nativeGetQnnStatus() }
                .getOrElse { return TrainingBackendResult.Failed("QNN capability query failed: ${it.message}") }
            val capability = runCatching { NativeTrainingFieldsParser.parse(qnnStatus) }
                .getOrElse { return TrainingBackendResult.Failed("QNN capability response is malformed") }
            if (capability.string("qnn_enabled") != "true" ||
                capability.string("qnn_sdk_detected") != "true" ||
                capability.string("qnn_implementation_ready") != "true" ||
                capability.string("qnn_status") != "QAIRT_ADAPTER_READY_REQUIRES_DEVICE_EXECUTION"
            ) {
                return TrainingBackendResult.Failed("QNN HTP capability is unavailable; no CPU fallback was selected")
            }
            var callbackFailure: String? = null
            val callback = ProgressCallback { message ->
                try {
                    parseProgress(message, request, runId, staged, onProgress)
                } catch (failure: Throwable) {
                    callbackFailure = "native progress was malformed: " +
                        (failure.message ?: failure.javaClass.simpleName)
                    // Make the native loop leave its next safe boundary; the
                    // terminal result is still converted to FAILED below so
                    // malformed telemetry can never look like success.
                    runCatching { NativeBridge.nativeRequestStop() }
                }
            }
            val cancelledBeforeJni = synchronized(lock) {
                if (pendingStop) {
                    pendingStop = false
                    running = false
                    true
                } else {
                    nativeCallEntered = true
                    false
                }
            }
            if (cancelledBeforeJni) {
                runCatching { NativeBridge.nativeClearStop() }
                return TrainingBackendResult.Cancelled(
                    TrainingBackendProgress(0, request.totalSteps),
                )
            }
            val report = NativeBridge.nativeRunExecutionMode(
                /* QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA */ 100,
                request.modelConfig.batchSize,
                request.modelConfig.dimension,
                request.modelConfig.feedForwardDimension,
                request.modelConfig.vocabularySize,
                request.totalSteps,
                0,
                request.modelConfig.learningRate,
                request.modelConfig.seed,
                request.modelConfig.tokens,
                request.modelConfig.layers,
                request.modelConfig.heads,
                request.modelConfig.seed.toInt(),
                false,
                1,
                0,
                0,
                0,
                false,
                runDirectory.absolutePath,
                request.resumeFrom?.completedStep ?: 0,
                request.modelConfig.checkpointInterval,
                callback,
            )
            // The native guard clears its entered bit before returning. Close
            // the Kotlin window as well, then clear a stop that raced the
            // return; otherwise nativeRequestStop() can leave a process-global
            // pending flag that cancels the next unrelated run.
            synchronized(lock) { nativeCallEntered = false }
            runCatching { NativeBridge.nativeClearStop() }
            callbackFailure?.let { return TrainingBackendResult.Failed(it) }
            parseTerminal(report, request, runId, staged)
        } catch (error: Throwable) {
            TrainingBackendResult.Failed("native HTP training failed: ${error.message ?: error.javaClass.simpleName}")
        } finally {
            synchronized(lock) {
                running = false
                nativeCallEntered = false
                pendingStop = false
                runArmed = false
            }
            NativeRunArbiter.release(arbiterOwner)
        }
    }

    override fun requestStop() {
        var callNative = false
        synchronized(lock) {
            // TrainingSession only calls this for an accepted active run.  A
            // close may race the worker before it enters JNI, so retain the
            // request for the next run() entry instead of losing it.
            if (!running) {
                if (runArmed) pendingStop = true
                return
            }
            pendingStop = true
            callNative = nativeCallEntered
        }
        if (callNative) runCatching { NativeBridge.nativeRequestStop() }
    }

    private fun parseProgress(
        text: String,
        request: TrainingRequest,
        runId: String,
        staged: StagedNicopediaDataset,
        onProgress: (TrainingBackendProgress) -> Unit,
    ) {
        val fields = NativeTrainingFieldsParser.parse(text)
        val nativePhase = fields.string("phase") ?: error("missing native phase")
        // The native Nicopedia run performs a mandatory host-side scratch
        // replay after the HTP trajectory. It is a diagnostic comparison,
        // not a second training phase and must never be presented as HTP
        // timing or treated as a fallback. Ignore its throttled progress;
        // the terminal report remains the authority for completion/evidence.
        val phase = classifyNativeTrainingPhase(nativePhase) ?: return
        val step = fields.int("step") ?: error("missing native step")
        val total = fields.int("steps") ?: error("missing native target step")
        val loss = optionalDouble(fields, "loss", allowNegative = true)?.toFloat()
        val timingWeight = optionalLong(fields, "timing_sample_steps") ?: 1L
        require(timingWeight > 0L) { "native timing sample weight is not positive" }
        val fusedUs = optionalDouble(fields, "fused_forward_backward_qnn_us")
        val adamUs = optionalDouble(fields, "adam_qnn_us")
        val hostUs = optionalDouble(fields, "host_overhead_us")
        val fusedCount = optionalLong(fields, "fused_qnn_execute_count")
        val adamCount = optionalLong(fields, "adam_qnn_execute_count")
        require(fusedCount == null || fusedCount >= 0L) { "native fused execute count is negative" }
        require(adamCount == null || adamCount >= 0L) { "native Adam execute count is negative" }
        val checkpointIoUs = optionalDouble(fields, "checkpoint_io_us")
        val sample = TrainingTimingSample(
            fusedForwardBackward = fusedCount?.let { count -> fusedUs?.let {
                PhaseTiming(TimingBackend.HTP, it / 1000.0 / timingWeight, count / timingWeight)
            }
            },
            adam = adamCount?.let { count -> adamUs?.let {
                PhaseTiming(TimingBackend.HTP, it / 1000.0 / timingWeight, count / timingWeight)
            }
            },
            host = hostUs?.let { PhaseTiming(TimingBackend.CPU, hostMs = it / 1000.0 / timingWeight) },
        ).takeIf { it.entries().isNotEmpty() }
        val checkpointStep = fields.int("checkpoint_step")
        val checkpoint = checkpointStep?.let {
            checkpointMetadata(runId, request, staged, it)
        }
        if (phase == TrainingPhase.SAVING_CHECKPOINT && checkpoint == null) {
            error("native checkpoint event has no verified NPRTCKPTV2 metadata")
        }
        val evidence = runtimeEvidence(fields)
        onProgress(
            TrainingBackendProgress(
                completedSteps = step,
                totalSteps = total,
                loss = loss,
                htpActivity = observedHtpActivityWindow(
                    observationWallUs = fields.double("observation_wall_us"),
                    fusedExecuteUs = fusedUs,
                    adamExecuteUs = adamUs,
                    fusedExecuteCount = fusedCount,
                    adamExecuteCount = adamCount,
                ),
                checkpoint = checkpoint,
                phase = phase,
                currentStepMs = optionalDouble(fields, "step_wall_us")?.div(1000.0)?.toLong(),
                checkpointIoMs = checkpointIoUs?.div(1000.0)?.toLong(),
                timingSample = sample,
                timingSampleWeight = timingWeight,
                runtimeEvidence = evidence,
            ),
        )
    }

    private fun parseTerminal(
        report: String,
        request: TrainingRequest,
        runId: String,
        staged: StagedNicopediaDataset,
    ): TrainingBackendResult {
        val fields = NativeTrainingFieldsParser.parse(report)
        val status = fields.string("status") ?: "FAILED"
        if (status == "SUCCESS") {
            val required = mapOf(
                "backend" to "HTP",
                "api_trace_backend_requested" to "HTP",
                "api_trace_graph_execute_failure_count" to "0",
                "api_trace_last_qnn_result" to "0",
                "api_trace_effective_result" to "0",
                "api_trace_runtime_backend_build_id" to BuildConfig.QAIRT_BUILD_ID,
                "api_trace_fallback_attempted" to "false",
                "api_trace_fallback_succeeded" to "false",
            )
            required.forEach { (key, expected) ->
                val actual = fields.string(key)
                val normalized = if (key == "api_trace_runtime_backend_build_id") {
                    actual?.removePrefix("v")
                } else actual
                if (normalized != expected) {
                    return TrainingBackendResult.Failed("native QNN evidence is incomplete or mismatched: $key")
                }
            }
        }
        val cacheHash = fields.string("cache_content_hash")
        val orderHash = fields.string("training_order_hash")
        val completed = fields.int("completed_steps") ?: 0
        if (completed !in 0..request.totalSteps) {
            return TrainingBackendResult.Failed("native completed step is outside the requested range")
        }
        val nativeCheckpointWritten = fields.bool("checkpoint_written") == true
        if ((status == "SUCCESS" || nativeCheckpointWritten) &&
            (cacheHash == null || orderHash == null)
        ) {
            return TrainingBackendResult.Failed("native dataset/order identity evidence is missing")
        }
        if (cacheHash != null && cacheHash != staged.inspection.nativeFnv1a64) {
            return TrainingBackendResult.Failed("native cache identity differs from the staged SAF dataset")
        }
        if (orderHash != null && orderHash != staged.inspection.trainingOrderHash) {
            return TrainingBackendResult.Failed("native training order identity differs from the staged dataset")
        }
        val finalLoss = fields.double("last_loss")?.toFloat()
        val evidence = runtimeEvidence(fields)
        val checkpoint = if (nativeCheckpointWritten && completed > 0) {
            checkpointMetadata(runId, request, staged, completed)
        } else null
        if (nativeCheckpointWritten && checkpoint == null) {
            return TrainingBackendResult.Failed("native checkpoint was written but could not be indexed safely")
        }
        if (status == "SUCCESS" && !nativeCheckpointWritten) {
            return TrainingBackendResult.Failed("native training completed without a verified NPRTCKPTV2 checkpoint")
        }
        val finalProgress = TrainingBackendProgress(
            completedSteps = completed,
            totalSteps = request.totalSteps,
            loss = finalLoss,
            checkpoint = checkpoint,
            currentStepMs = optionalDouble(fields, "training_step_ms")?.toLong(),
            htpExecuteCount = optionalLong(fields, "qnn_execute_count") ?: optionalLong(fields, "graph_execute_count"),
            runtimeEvidence = evidence,
        )
        return when (status) {
            "SUCCESS" -> TrainingBackendResult.Completed(finalProgress, runtimeEvidence = evidence)
            "CANCELLED" -> TrainingBackendResult.Cancelled(finalProgress, runtimeEvidence = evidence)
            else -> TrainingBackendResult.Failed(
                fields.string("error") ?: fields.string("failure_classification") ?: "native training returned $status",
                runtimeEvidence = evidence,
            )
        }
    }

    private fun runtimeEvidence(fields: NativeTrainingFields): TrainingRuntimeEvidence? {
        val qnn = optionalBoolean(fields, "qnn_return_code_success")
        val finite = optionalBoolean(fields, "output_tensors_finite")
        val fallback = optionalBoolean(fields, "cpu_fallback")
        if (qnn == null && finite == null && fallback == null) return null
        return TrainingRuntimeEvidence(
            qnnReturnCodeSuccess = qnn,
            outputTensorsFinite = finite,
            cpuFallback = fallback,
            // Missing backend identity is deliberately not promoted to HTP;
            // progress may still be displayed, but all HTP timing/evidence
            // gates must fail closed until native says backend=HTP explicitly.
            backend = fields.string("backend"),
            error = fields.string("qnn_error") ?: fields.string("error"),
        )
    }

    private fun optionalDouble(fields: NativeTrainingFields, key: String, allowNegative: Boolean = false): Double? {
        val raw = fields.string(key) ?: return null
        val value = raw.toDoubleOrNull() ?: error("native field $key is not numeric")
        require(value.isFinite() && (allowNegative || value >= 0.0)) {
            "native field $key is not finite/non-negative"
        }
        return value
    }

    private fun optionalLong(fields: NativeTrainingFields, key: String): Long? {
        val raw = fields.string(key) ?: return null
        return raw.toLongOrNull() ?: error("native field $key is not an integer")
    }

    private fun optionalBoolean(fields: NativeTrainingFields, key: String): Boolean? {
        val raw = fields.string(key) ?: return null
        return when (raw.lowercase()) {
            "true", "1", "yes" -> true
            "false", "0", "no" -> false
            else -> error("native field $key is not boolean")
        }
    }

    private fun checkpointMetadata(
        runId: String,
        request: TrainingRequest,
        staged: StagedNicopediaDataset,
        step: Int,
    ): TrainingCheckpointMetadata? {
        if (step <= 0) return null
        val path = File(staged.directory, "htp-seed${request.modelConfig.seed}-l${request.modelConfig.layers}-step$step.ckpt")
        if (!path.isFile || path.length() <= 0L) return null
        val metadata = TrainingCheckpointMetadata(
            uri = "native-checkpoint:$runId:$step",
            completedStep = step,
            modelConfig = request.modelConfig,
            format = TrainingPlan.NICOPEDIA_L19.checkpointFormat,
            formatVersion = TrainingPlan.NICOPEDIA_L19.checkpointFormatVersion,
            finite = true,
            createdAtMs = System.currentTimeMillis(),
            datasetIdentity = staged.inspection.identity,
        )
        if (!checkpointStore.registerNativePath(metadata, path.absolutePath)) return null
        checkpointStore.save(metadata)
        return metadata
    }
}
