package com.yuubinnkyoku.phonelm

import android.content.Context
import java.io.Closeable
import java.io.File
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

enum class GenerationMode { GREEDY, SAMPLE }

data class GenerationRequest(
    val prompt: String,
    val mode: GenerationMode = GenerationMode.GREEDY,
    val maxNewBytes: Int = 64,
    val temperature: Float = 0.8f,
    val topK: Int = 32,
    val samplingSeed: Long = 1L,
) {
    val promptBytes: ByteArray get() = prompt.toByteArray(Charsets.UTF_8)

    fun validationError(): String? = when {
        prompt.isEmpty() -> "Prompt is required"
        maxNewBytes !in 1..1024 -> "Max new bytes must be in 1..1024"
        mode == GenerationMode.SAMPLE && (!temperature.isFinite() || temperature <= 0f) ->
            "Temperature must be finite and positive"
        mode == GenerationMode.SAMPLE && topK !in 1..256 -> "TopK must be in 1..256"
        mode == GenerationMode.SAMPLE && samplingSeed < 0L -> "SamplingSeed must be non-negative"
        else -> null
    }
}

data class GenerationResult(
    val generatedBytes: ByteArray,
    val displayText: String,
    val byteCount: Int,
    val elapsedMs: Long,
    val backend: String,
    val checkpointParameterHash: String,
    val cpuFallback: Boolean,
    val finite: Boolean,
    val generationGate: Boolean,
    val debugReport: String,
)

sealed interface GenerationState {
    data object Idle : GenerationState
    data object Running : GenerationState
    data class Success(val result: GenerationResult) : GenerationState
    data class Failed(val message: String, val debugReport: String? = null) : GenerationState
}

sealed interface GenerationCheckpointStatus {
    data class Available(val checkpoint: TrainingCheckpointMetadata) : GenerationCheckpointStatus
    data class Unavailable(val message: String = "No trained checkpoint") : GenerationCheckpointStatus
}

data class GenerationUiState(
    val prompt: String = "",
    val mode: GenerationMode = GenerationMode.GREEDY,
    val temperatureText: String = "0.8",
    val topKText: String = "32",
    val samplingSeedText: String = "1",
    val maxNewBytesText: String = "64",
    val checkpoint: GenerationCheckpointStatus = GenerationCheckpointStatus.Unavailable(),
    val execution: GenerationState = GenerationState.Idle,
) {
    fun requestOrNull(): GenerationRequest? {
        val request = GenerationRequest(
            prompt = prompt,
            mode = mode,
            maxNewBytes = maxNewBytesText.toIntOrNull() ?: return null,
            temperature = temperatureText.toFloatOrNull() ?: return null,
            topK = topKText.toIntOrNull() ?: return null,
            samplingSeed = samplingSeedText.toLongOrNull() ?: return null,
        )
        return request.takeIf { it.validationError() == null }
    }

    val canGenerate: Boolean
        get() = checkpoint is GenerationCheckpointStatus.Available &&
            execution !is GenerationState.Running && requestOrNull() != null
}

interface GenerationBackend {
    fun checkpointStatus(): GenerationCheckpointStatus
    fun generate(request: GenerationRequest, checkpoint: TrainingCheckpointMetadata): GenerationResult
}

private class GenerationBackendException(message: String, val report: String) : IllegalStateException(message)

class GenerationSession(
    private val backend: GenerationBackend,
    private val executor: Executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-generation").apply { isDaemon = true }
    },
) : Closeable {
    private val lock = Any()
    private val listeners = CopyOnWriteArrayList<(GenerationUiState) -> Unit>()
    private var state = GenerationUiState(checkpoint = backend.checkpointStatus())

    fun snapshot(): GenerationUiState = synchronized(lock) { state }

    fun subscribe(listener: (GenerationUiState) -> Unit): AutoCloseable {
        listeners += listener
        listener(snapshot())
        return AutoCloseable { listeners.remove(listener) }
    }

    fun updatePrompt(value: String) = update { copy(prompt = value) }
    fun updateMode(value: GenerationMode) = update { copy(mode = value) }
    fun updateTemperature(value: String) = update { copy(temperatureText = value) }
    fun updateTopK(value: String) = update { copy(topKText = value) }
    fun updateSamplingSeed(value: String) = update { copy(samplingSeedText = value) }
    fun updateMaxNewBytes(value: String) = update { copy(maxNewBytesText = value) }

    fun refreshCheckpoint() = update { copy(checkpoint = backend.checkpointStatus()) }

    fun generate(trainingActive: Boolean = false): Boolean {
        val accepted = synchronized(lock) {
            val refreshed = state.copy(checkpoint = backend.checkpointStatus())
            val request = refreshed.requestOrNull()
            val checkpoint = (refreshed.checkpoint as? GenerationCheckpointStatus.Available)?.checkpoint
            if (trainingActive || request == null || checkpoint == null || refreshed.execution is GenerationState.Running) {
                state = refreshed
                null
            } else {
                state = refreshed.copy(execution = GenerationState.Running)
                request to checkpoint
            }
        }
        publish()
        if (accepted == null) return false
        executor.execute {
            val terminal = runCatching { backend.generate(accepted.first, accepted.second) }
                .fold(
                    onSuccess = { GenerationState.Success(it) },
                    onFailure = {
                        GenerationState.Failed(
                            it.message ?: it.javaClass.simpleName,
                            (it as? GenerationBackendException)?.report,
                        )
                    },
                )
            update { copy(execution = terminal, checkpoint = backend.checkpointStatus()) }
        }
        return true
    }

    private fun update(transform: GenerationUiState.() -> GenerationUiState) {
        synchronized(lock) { state = state.transform() }
        publish()
    }

    private fun publish() {
        val snapshot = snapshot()
        listeners.forEach { listener -> runCatching { listener(snapshot) } }
    }

    override fun close() {
        (executor as? ExecutorService)?.shutdown()
        listeners.clear()
    }
}

class NativeHtpGenerationBackend(
    context: Context,
    private val checkpointStore: TrainingCheckpointStore = AndroidTrainingCheckpointStore(context.applicationContext),
) : GenerationBackend {
    private val appContext = context.applicationContext
    private val promptDirectory = File(appContext.cacheDir, "generation-prompts")

    override fun checkpointStatus(): GenerationCheckpointStatus {
        val plan = TrainingPlan.NICOPEDIA_L19
        val selection = TrainingCheckpointCatalog.latestCompatible(
            checkpointStore.list(), plan.modelConfig, plan.checkpointFormat, plan.checkpointFormatVersion,
        )
        val selected = (selection as? TrainingCheckpointSelection.Selected)?.checkpoint
        if (selected == null) {
            val detail = (selection as? TrainingCheckpointSelection.Incompatible)?.reason
            return GenerationCheckpointStatus.Unavailable(
                if (detail == null) "No trained checkpoint" else "No trained checkpoint: $detail",
            )
        }
        if (checkpointStore.resolveNativePath(selected) == null) {
            return GenerationCheckpointStatus.Unavailable("No trained checkpoint: checkpoint payload is unavailable")
        }
        return GenerationCheckpointStatus.Available(selected)
    }

    override fun generate(request: GenerationRequest, checkpoint: TrainingCheckpointMetadata): GenerationResult {
        request.validationError()?.let { error(it) }
        if (!BuildConfig.PHONELM_QNN_ENABLED) error("QNN HTP is disabled in this APK; CPU fallback is not permitted")
        val plan = TrainingPlan.NICOPEDIA_L19
        require(checkpoint.modelConfig == plan.modelConfig && checkpoint.format == plan.checkpointFormat &&
            checkpoint.formatVersion == plan.checkpointFormatVersion && checkpoint.finite) {
            "checkpoint identity is incompatible with the production model"
        }
        val checkpointPath = checkpointStore.resolveNativePath(checkpoint)
            ?: error("No trained checkpoint: checkpoint payload is unavailable")
        check(promptDirectory.isDirectory || promptDirectory.mkdirs()) { "prompt staging directory is unavailable" }
        val promptFile = File(promptDirectory, "prompt-${UUID.randomUUID()}.bin")
        val owner = "generation:${UUID.randomUUID()}"
        if (!NativeRunArbiter.tryAcquire(owner)) error("Training or another native run is already active")
        try {
            QnnEnvironment.prepare(appContext)
            promptFile.outputStream().use { it.write(request.promptBytes) }
            val config = plan.modelConfig
            val report = NativeBridge.nativeRunNicopediaGenerate(
                checkpointPath = checkpointPath,
                promptPath = promptFile.absolutePath,
                seed = config.seed,
                layers = config.layers,
                tokens = config.tokens,
                dimension = config.dimension,
                feedForwardDimension = config.feedForwardDimension,
                maxNewBytes = request.maxNewBytes,
                generateMode = if (request.mode == GenerationMode.GREEDY) "greedy" else "sample",
                temperature = request.temperature,
                topK = request.topK,
                samplingSeed = request.samplingSeed,
                gatePolicy = "htp-smoke",
                htpGraphPrecisionMode = 0,
                htpGraphPrecisionCompensation = 0,
                htpGraphWeightsPacking = 0,
                htpGraphAdvancedActivationFusion = 0,
                htpContextGraphSplitting = 0,
                htpNativeTensorFp16 = false,
            )
            return try {
                parseReport(report)
            } catch (error: GenerationBackendException) {
                throw error
            } catch (error: Throwable) {
                throw GenerationBackendException(
                    error.message ?: "native generation response is invalid",
                    report,
                )
            }
        } finally {
            runCatching { promptFile.delete() }
            NativeRunArbiter.release(owner)
        }
    }

    internal fun parseReport(report: String): GenerationResult {
        val fields = NativeTrainingFieldsParser.parse(report)
        fun requireField(key: String, expected: String) {
            require(fields.string(key)?.removePrefix("v") == expected.removePrefix("v")) {
                "native generation evidence mismatch: $key"
            }
        }
        if (fields.string("status") != "SUCCESS") {
            val reason = fields.string("error") ?: fields.string("failure_classification") ?: "native generation failed"
            throw GenerationBackendException(reason, report)
        }
        requireField("checkpoint_format", "NPRTCKPTV2")
        requireField("checkpoint_header_tokens", "32")
        requireField("checkpoint_header_dimension", "32")
        requireField("checkpoint_header_feedforward", "32")
        requireField("checkpoint_header_layers", "19")
        requireField("checkpoint_header_heads", "2")
        requireField("checkpoint_finite", "true")
        requireField("qnn_return_code_success", "true")
        requireField("output_tensors_finite", "true")
        requireField("cpu_fallback", "false")
        requireField("generation_gate", "true")
        requireField("generation_health", "true")
        requireField("nan_detected", "false")
        requireField("inf_detected", "false")
        val bytes = decodeHex(fields.string("generated_hex") ?: error("generated bytes are missing"))
        val byteCount = fields.int("generated_byte_count") ?: error("generated byte count is missing")
        require(byteCount == bytes.size) { "generated byte count differs from payload" }
        val elapsedSeconds = fields.double("generation_total_seconds")
            ?.takeIf { it.isFinite() && it >= 0.0 } ?: error("generation elapsed time is invalid")
        val displayBytes = NativeBridge.nativeSafeUtf8Display(bytes)
        return GenerationResult(
            generatedBytes = bytes,
            displayText = String(displayBytes, Charsets.UTF_8),
            byteCount = byteCount,
            elapsedMs = (elapsedSeconds * 1000.0).toLong(),
            backend = "HTP",
            checkpointParameterHash = fields.string("checkpoint_parameter_hash")
                ?: error("checkpoint parameter hash is missing"),
            cpuFallback = false,
            finite = true,
            generationGate = true,
            debugReport = report,
        )
    }

    private fun decodeHex(hex: String): ByteArray {
        require(hex.length % 2 == 0) { "generated hex length is odd" }
        return ByteArray(hex.length / 2) { index ->
            hex.substring(index * 2, index * 2 + 2).toIntOrNull(16)?.toByte()
                ?: error("generated hex is malformed")
        }
    }
}

object GenerationSessionRegistry {
    private val sessions = java.util.WeakHashMap<Any, GenerationSession>()

    @Synchronized
    fun get(applicationContext: Any): GenerationSession {
        require(applicationContext is Context) { "Android context is required" }
        val appContext = applicationContext.applicationContext
        return sessions.getOrPut(appContext) { GenerationSession(NativeHtpGenerationBackend(appContext)) }
    }
}
