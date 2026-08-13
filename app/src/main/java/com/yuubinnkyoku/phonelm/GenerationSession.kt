package com.yuubinnkyoku.phonelm

import android.content.Context
import java.io.Closeable
import java.io.File
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

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
    val qnnExecuteAttempts: Long = 0,
    val qnnExecuteSuccesses: Long = 0,
    val qnnExecuteFailures: Long = 0,
)

enum class GenerationPhase {
    PREPARING,
    CHECKPOINT_VALIDATION,
    HTP_INITIALIZATION,
    GRAPH_PREPARATION,
    GENERATING,
    COMPLETED,
    FAILED,
}

data class GenerationProgress(
    val phase: GenerationPhase,
    val generatedBytes: Int = 0,
    val maxNewBytes: Int,
    val elapsedMs: Long = 0,
    val qnnExecuteAttempts: Long = 0,
    val qnnExecuteSuccesses: Long = 0,
    val qnnExecuteFailures: Long = 0,
    val cpuFallback: Boolean = false,
    val finite: Boolean = true,
    val displayText: String = "",
    val rawBytes: ByteArray = byteArrayOf(),
)

sealed interface GenerationState {
    data object Idle : GenerationState
    data class Running(val progress: GenerationProgress) : GenerationState
    data class Success(val result: GenerationResult) : GenerationState
    data class Failed(val message: String, val debugReport: String? = null) : GenerationState
}

data class GenerationUiState(
    val prompt: String = "",
    val mode: GenerationMode = GenerationMode.GREEDY,
    val temperatureText: String = "0.8",
    val topKText: String = "32",
    val samplingSeedText: String = "1",
    val maxNewBytesText: String = "64",
    val checkpoints: List<GenerationCheckpoint> = emptyList(),
    val selectedCheckpointPath: String? = null,
    val requiredCheckpointHash: String? = null,
    val checkpointLoading: Boolean = false,
    val checkpointMessage: String? = "No usable checkpoint",
    val checkpointWarning: String? = null,
    val execution: GenerationState = GenerationState.Idle,
    val history: GenerationHistoryUiState = GenerationHistoryUiState(),
) {
    val selectedCheckpoint: GenerationCheckpoint?
        get() = checkpoints.firstOrNull { it.path == selectedCheckpointPath }

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
        get() = selectedCheckpoint?.usable == true &&
            execution !is GenerationState.Running && requestOrNull() != null
}

interface GenerationBackend {
    fun generate(
        request: GenerationRequest,
        checkpoint: GenerationCheckpoint,
        onProgress: (GenerationProgress) -> Unit = {},
    ): GenerationResult
}

private class GenerationBackendException(message: String, val report: String) : IllegalStateException(message)

class GenerationSession(
    private val backend: GenerationBackend,
    private val checkpointRepository: GenerationCheckpointRepository,
    private val historyRepository: GenerationHistoryRepository = EmptyGenerationHistoryRepository,
    private val losslessByteDisplay: LosslessByteDisplay = LosslessByteDisplay { bytes ->
        String(bytes, Charsets.UTF_8)
    },
    private val executor: Executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-generation").apply { isDaemon = true }
    },
) : Closeable {
    private val lock = Any()
    private val listeners = CopyOnWriteArrayList<(GenerationUiState) -> Unit>()
    private var checkpointRefreshInFlight = false
    private var state = GenerationUiState()

    init {
        refreshCheckpoints()
        refreshHistory()
    }

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

    fun selectHistory(id: String): Boolean {
        val selected = synchronized(lock) {
            if (state.history.items.none { it.record.id == id }) false
            else {
                state = state.copy(history = state.history.copy(selectedId = id))
                true
            }
        }
        if (selected) publish()
        return selected
    }

    fun closeHistoryDetail() = update { copy(history = history.copy(selectedId = null)) }

    fun deleteHistory(id: String) {
        executor.execute {
            val result = runCatching { historyRepository.delete(id) }
            updateHistory(result.exceptionOrNull()?.message)
        }
    }

    fun clearHistory() {
        executor.execute {
            val result = runCatching { historyRepository.clear() }
            updateHistory(result.exceptionOrNull()?.message)
        }
    }

    fun useHistoryAgain(id: String): Boolean {
        val restored = synchronized(lock) {
            val record = state.history.items.firstOrNull { it.record.id == id }?.record
                ?: return@synchronized false
            val originalCheckpoint = state.checkpoints.firstOrNull {
                it.usable && it.parameterHash == record.checkpointParameterHash
            }
            state = state.copy(
                prompt = String(record.promptBytes, Charsets.UTF_8),
                mode = record.mode,
                temperatureText = record.temperature.toString(),
                topKText = record.topK.toString(),
                samplingSeedText = record.samplingSeed.toString(),
                maxNewBytesText = record.maxNewBytes.toString(),
                selectedCheckpointPath = originalCheckpoint?.path,
                requiredCheckpointHash = record.checkpointParameterHash,
                checkpointMessage = null,
                checkpointWarning = if (originalCheckpoint == null) {
                    "Original checkpoint is no longer available"
                } else null,
                history = state.history.copy(selectedId = null),
            )
            true
        }
        if (restored) publish()
        return restored
    }

    fun selectCheckpoint(path: String): Boolean {
        val accepted = synchronized(lock) {
            val candidate = state.checkpoints.firstOrNull { it.path == path && it.usable }
                ?: return@synchronized false
            state = state.copy(
                selectedCheckpointPath = candidate.path,
                requiredCheckpointHash = null,
                checkpointMessage = null,
                checkpointWarning = null,
            )
            true
        }
        if (accepted) publish()
        return accepted
    }

    fun refreshCheckpoints() {
        val shouldStart = synchronized(lock) {
            if (checkpointRefreshInFlight || state.execution is GenerationState.Running) false
            else {
                checkpointRefreshInFlight = true
                state = state.copy(checkpointLoading = true, checkpointMessage = "Checking checkpoints…")
                true
            }
        }
        if (!shouldStart) return
        publish()
        executor.execute {
            val listed = runCatching { checkpointRepository.listCheckpoints() }.getOrDefault(emptyList())
            synchronized(lock) {
                checkpointRefreshInFlight = false
                state = state.withCheckpointCandidates(listed)
            }
            publish()
        }
    }

    fun refreshHistory() {
        update { copy(history = history.copy(loading = true, message = null)) }
        executor.execute { updateHistory(null) }
    }

    fun generate(trainingActive: Boolean = false): Boolean {
        val accepted = synchronized(lock) {
            val request = state.requestOrNull()
            val checkpoint = state.selectedCheckpoint
            if (trainingActive || request == null || checkpoint?.usable != true || state.execution is GenerationState.Running) {
                null
            } else {
                state = state.copy(
                    execution = GenerationState.Running(
                        GenerationProgress(GenerationPhase.PREPARING, maxNewBytes = request.maxNewBytes),
                    ),
                )
                AcceptedGenerationRun(
                    id = UUID.randomUUID().toString(),
                    createdAtMs = System.currentTimeMillis(),
                    request = request,
                    checkpoint = checkpoint,
                )
            }
        }
        publish()
        if (accepted == null) return false
        executor.execute {
            val terminal = runCatching {
                update {
                    copy(
                        execution = GenerationState.Running(
                            GenerationProgress(
                                GenerationPhase.CHECKPOINT_VALIDATION,
                                maxNewBytes = accepted.request.maxNewBytes,
                            ),
                        ),
                    )
                }
                val validated = checkpointRepository.validateCheckpoint(accepted.checkpoint).getOrThrow()
                backend.generate(accepted.request, validated) { progress ->
                    update {
                        if (execution is GenerationState.Running) {
                            copy(execution = GenerationState.Running(progress))
                        } else this
                    }
                }
            }
                .fold(
                    onSuccess = { GenerationState.Success(it) },
                    onFailure = {
                        GenerationState.Failed(
                            it.message ?: it.javaClass.simpleName,
                            (it as? GenerationBackendException)?.report,
                        )
                    },
                )
            val progress = (snapshot().execution as? GenerationState.Running)?.progress
            val historyRecord = terminalHistoryRecord(accepted, terminal, progress)
            val historyFailure = runCatching { historyRepository.insert(historyRecord) }.exceptionOrNull()?.message
            val listed = runCatching { checkpointRepository.listCheckpoints() }.getOrDefault(emptyList())
            val historyItems = loadHistoryItems()
            update {
                withCheckpointCandidates(listed).copy(
                    execution = terminal,
                    history = history.copy(items = historyItems, loading = false, message = historyFailure),
                )
            }
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

    private fun updateHistory(message: String?) {
        val loaded = runCatching { loadHistoryItems() }
        update {
            val items = loaded.getOrDefault(history.items)
            copy(
                history = history.copy(
                    items = items,
                    selectedId = history.selectedId?.takeIf { id -> items.any { it.record.id == id } },
                    loading = false,
                    message = message ?: loaded.exceptionOrNull()?.message,
                ),
            )
        }
    }

    private fun loadHistoryItems(): List<GenerationHistoryItem> = historyRepository.listNewestFirst().map { record ->
        GenerationHistoryItem(
            record = record,
            promptText = losslessByteDisplay.display(record.promptBytes),
            outputText = losslessByteDisplay.display(record.generatedBytes),
        )
    }

    private fun terminalHistoryRecord(
        run: AcceptedGenerationRun,
        terminal: GenerationState,
        progress: GenerationProgress?,
    ): GenerationHistoryRecord {
        val result = (terminal as? GenerationState.Success)?.result
        val failure = terminal as? GenerationState.Failed
        val failureFields = failure?.debugReport?.let { NativeTrainingFieldsParser.parse(it) }
        val checkpoint = run.checkpoint
        return GenerationHistoryRecord(
            id = run.id,
            createdAtMs = run.createdAtMs,
            promptBytes = run.request.promptBytes,
            mode = run.request.mode,
            temperature = run.request.temperature,
            topK = run.request.topK,
            samplingSeed = run.request.samplingSeed,
            maxNewBytes = run.request.maxNewBytes,
            checkpointStep = checkpoint.step,
            checkpointParameterHash = checkpoint.parameterHash!!,
            vocabulary = 256,
            tokens = checkpoint.tokens,
            dimension = checkpoint.dimension,
            feedForwardDimension = checkpoint.feedForwardDimension,
            layers = checkpoint.layers,
            heads = checkpoint.heads,
            generatedBytes = result?.generatedBytes ?: progress?.rawBytes ?: byteArrayOf(),
            elapsedMs = result?.elapsedMs ?: progress?.elapsedMs ?: 0,
            backend = result?.backend ?: "HTP",
            qnnExecuteAttempts = result?.qnnExecuteAttempts
                ?: failureFields?.long("api_trace_graph_execute_attempt_count")
                ?: progress?.qnnExecuteAttempts ?: 0,
            qnnExecuteSuccesses = result?.qnnExecuteSuccesses
                ?: failureFields?.long("api_trace_graph_execute_success_count")
                ?: progress?.qnnExecuteSuccesses ?: 0,
            qnnExecuteFailures = result?.qnnExecuteFailures
                ?: failureFields?.long("api_trace_graph_execute_failure_count")
                ?: progress?.qnnExecuteFailures ?: 0,
            cpuFallback = result?.cpuFallback ?: failureFields?.bool("cpu_fallback")
                ?: progress?.cpuFallback ?: false,
            finite = result?.finite ?: failureFields?.bool("output_tensors_finite")
                ?: failureFields?.bool("htp_native_application_tensors_finite")
                ?: progress?.finite?.takeIf {
                progress.phase == GenerationPhase.GENERATING
            } ?: false,
            status = if (result != null) GenerationHistoryStatus.SUCCESS else GenerationHistoryStatus.FAILED,
            failureMessage = failure?.message,
        )
    }

    private fun GenerationUiState.withCheckpointCandidates(candidates: List<GenerationCheckpoint>): GenerationUiState {
        val sorted = GenerationCheckpointSelection.sorted(candidates)
        val required = requiredCheckpointHash?.let { hash ->
            sorted.firstOrNull { it.usable && it.parameterHash == hash }
        }
        val preserved = selectedCheckpointPath?.let { selected ->
            sorted.firstOrNull { it.path == selected && it.usable }
        }
        val selected = if (requiredCheckpointHash != null) required else preserved ?: checkpointRepository.defaultCheckpoint(sorted)
        val newerUnusable = selected?.let { chosen -> sorted.any { !it.usable && it.step > chosen.step } } == true
        val originalMissing = requiredCheckpointHash != null && required == null
        return copy(
            checkpoints = sorted,
            selectedCheckpointPath = selected?.path,
            checkpointLoading = false,
            checkpointMessage = if (selected == null && !originalMissing) "No usable checkpoint" else null,
            checkpointWarning = when {
                originalMissing -> "Original checkpoint is no longer available"
                newerUnusable -> "Newer unusable checkpoint exists"
                else -> null
            },
        )
    }

    override fun close() {
        (executor as? ExecutorService)?.shutdown()
        listeners.clear()
    }

    private data class AcceptedGenerationRun(
        val id: String,
        val createdAtMs: Long,
        val request: GenerationRequest,
        val checkpoint: GenerationCheckpoint,
    )
}

class NativeHtpGenerationBackend(
    context: Context,
) : GenerationBackend {
    private val appContext = context.applicationContext
    private val promptDirectory = File(appContext.cacheDir, "generation-prompts")

    override fun generate(
        request: GenerationRequest,
        checkpoint: GenerationCheckpoint,
        onProgress: (GenerationProgress) -> Unit,
    ): GenerationResult {
        request.validationError()?.let { error(it) }
        if (!BuildConfig.PHONELM_QNN_ENABLED) error("QNN HTP is disabled in this APK; CPU fallback is not permitted")
        val plan = TrainingPlan.NICOPEDIA_L19
        require(checkpoint.usable && checkpoint.tokens == plan.modelConfig.tokens &&
            checkpoint.dimension == plan.modelConfig.dimension &&
            checkpoint.feedForwardDimension == plan.modelConfig.feedForwardDimension &&
            checkpoint.layers == plan.modelConfig.layers && checkpoint.heads == plan.modelConfig.heads &&
            checkpoint.seed == plan.modelConfig.seed) {
            "checkpoint identity is incompatible with the production model"
        }
        check(promptDirectory.isDirectory || promptDirectory.mkdirs()) { "prompt staging directory is unavailable" }
        val promptFile = File(promptDirectory, "prompt-${UUID.randomUUID()}.bin")
        val owner = "generation:${UUID.randomUUID()}"
        if (!NativeRunArbiter.tryAcquire(owner)) error("Training or another native run is already active")
        val progressPoller = Executors.newSingleThreadScheduledExecutor { runnable ->
            Thread(runnable, "PhoneLM-generation-progress").apply { isDaemon = true }
        }
        try {
            onProgress(GenerationProgress(GenerationPhase.HTP_INITIALIZATION, maxNewBytes = request.maxNewBytes))
            QnnEnvironment.prepare(appContext)
            promptFile.outputStream().use { it.write(request.promptBytes) }
            progressPoller.scheduleAtFixedRate(
                {
                    runCatching { parseProgress(NativeBridge.nativeGetNicopediaGenerationProgress()) }
                        .getOrNull()?.let(onProgress)
                },
                PROGRESS_POLL_MS,
                PROGRESS_POLL_MS,
                TimeUnit.MILLISECONDS,
            )
            val config = plan.modelConfig
            val report = NativeBridge.nativeRunNicopediaGenerate(
                checkpointPath = checkpoint.path,
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
                parseReport(report, checkpoint.parameterHash!!)
            } catch (error: GenerationBackendException) {
                throw error
            } catch (error: Throwable) {
                throw GenerationBackendException(
                    error.message ?: "native generation response is invalid",
                    report,
                )
            }
        } finally {
            progressPoller.shutdownNow()
            runCatching { promptFile.delete() }
            NativeRunArbiter.release(owner)
        }
    }

    internal fun parseProgress(report: String): GenerationProgress {
        val fields = NativeTrainingFieldsParser.parse(report)
        val phase = when (fields.string("phase")) {
            "preparing" -> GenerationPhase.PREPARING
            "checkpoint_validation" -> GenerationPhase.CHECKPOINT_VALIDATION
            "htp_initialization" -> GenerationPhase.HTP_INITIALIZATION
            "graph_preparation" -> GenerationPhase.GRAPH_PREPARATION
            "generating" -> GenerationPhase.GENERATING
            "completed" -> GenerationPhase.COMPLETED
            "failed" -> GenerationPhase.FAILED
            else -> error("native generation progress phase is invalid")
        }
        val generated = fields.int("generated_bytes")?.coerceAtLeast(0) ?: 0
        val max = fields.int("max_new_bytes")?.takeIf { it > 0 } ?: 1
        val raw = decodeHex(fields.string("generated_hex").orEmpty())
        require(raw.size == generated) { "live generated byte count differs from payload" }
        val display = if (raw.isEmpty()) "" else String(NativeBridge.nativeSafeUtf8Display(raw), Charsets.UTF_8)
        return GenerationProgress(
            phase = phase,
            generatedBytes = generated,
            maxNewBytes = max,
            elapsedMs = fields.long("elapsed_ms")?.coerceAtLeast(0) ?: 0,
            qnnExecuteAttempts = fields.long("qnn_execute_attempts")?.coerceAtLeast(0) ?: 0,
            qnnExecuteSuccesses = fields.long("qnn_execute_successes")?.coerceAtLeast(0) ?: 0,
            qnnExecuteFailures = fields.long("qnn_execute_failures")?.coerceAtLeast(0) ?: 0,
            cpuFallback = fields.string("cpu_fallback") == "true",
            finite = fields.string("finite") != "false",
            displayText = display,
            rawBytes = raw,
        )
    }

    internal fun parseReport(report: String, expectedCheckpointHash: String): GenerationResult {
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
        requireField("checkpoint_parameter_hash", expectedCheckpointHash)
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
            qnnExecuteAttempts = fields.long("api_trace_graph_execute_attempt_count")?.coerceAtLeast(0) ?: 0,
            qnnExecuteSuccesses = fields.long("api_trace_graph_execute_success_count")?.coerceAtLeast(0) ?: 0,
            qnnExecuteFailures = fields.long("api_trace_graph_execute_failure_count")?.coerceAtLeast(0) ?: 0,
        )
    }

    private fun decodeHex(hex: String): ByteArray {
        require(hex.length % 2 == 0) { "generated hex length is odd" }
        return ByteArray(hex.length / 2) { index ->
            hex.substring(index * 2, index * 2 + 2).toIntOrNull(16)?.toByte()
                ?: error("generated hex is malformed")
        }
    }

    private companion object {
        const val PROGRESS_POLL_MS = 150L
    }
}

object GenerationSessionRegistry {
    private val sessions = java.util.WeakHashMap<Any, GenerationSession>()

    @Synchronized
    fun get(applicationContext: Any): GenerationSession {
        require(applicationContext is Context) { "Android context is required" }
        val appContext = applicationContext.applicationContext
        return sessions.getOrPut(appContext) {
            val checkpointStore = AndroidTrainingCheckpointStore(appContext)
            GenerationSession(
                NativeHtpGenerationBackend(appContext),
                AppPrivateGenerationCheckpointRepository(checkpointStore),
                FileGenerationHistoryRepository(File(appContext.filesDir, "generation-history-v1.bin")),
                LosslessByteDisplay { bytes ->
                    if (bytes.isEmpty()) "" else String(NativeBridge.nativeSafeUtf8Display(bytes), Charsets.UTF_8)
                },
            )
        }
    }
}

private object EmptyGenerationHistoryRepository : GenerationHistoryRepository {
    override fun insert(record: GenerationHistoryRecord) = false
    override fun listNewestFirst() = emptyList<GenerationHistoryRecord>()
    override fun delete(id: String) = false
    override fun clear() = Unit
}
