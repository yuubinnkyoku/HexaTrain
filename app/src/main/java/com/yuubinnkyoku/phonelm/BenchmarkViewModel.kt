package com.yuubinnkyoku.phonelm

import android.os.Handler
import android.os.Looper
import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class BenchmarkUiState(
    val running: Boolean = false,
    val output: String = "",
    val lastResult: BenchmarkResult? = null,
    val lastReport: String? = null,
    val configurationSummary: String = "",
    val qnnStatus: String = "QNN status loading...",
)

internal const val MAX_UI_OUTPUT_CHARS = 32 * 1024
private const val TRUNCATED_OUTPUT_MARKER = "[earlier output truncated]\n"

fun interface UiDispatcher {
    fun dispatch(block: () -> Unit)
}

interface BenchmarkEngine {
    fun environmentReport(): String
    fun run(config: BenchmarkConfig, progress: (String) -> Unit): String
    fun requestStop()

    /** Arms cancellation for the accepted-but-not-yet-entered JNI window. */
    fun prepareRun() = Unit

    /** Clears the arm when the worker was rejected or could not be queued. */
    fun cancelPreparedRun() = Unit

    fun qnnStatus(): String = "qnn_status=BLOCKED_BY_QAIRT_SDK_NOT_INSTALLED"

    fun runMode(
        mode: ExecutionMode,
        config: BenchmarkConfig,
        progress: (String) -> Unit,
    ): String = run(config, progress)
}

object NativeBenchmarkEngine : BenchmarkEngine {
    private const val OWNER = "benchmark"
    private val stopLock = Any()
    private var prepared = false
    private var pendingStop = false

    override fun environmentReport(): String = NativeBridge.nativeGetEnvironmentInfo()

    override fun run(config: BenchmarkConfig, progress: (String) -> Unit): String =
        withOwnership {
            NativeBridge.nativeRunBenchmark(
                backend = config.backend.nativeCode,
                batchSize = config.batchSize,
                dimension = config.dimension,
                steps = config.steps,
                warmupSteps = config.warmupSteps,
                learningRate = config.learningRate,
                seed = config.seed,
                progressCallback = ProgressCallback(progress),
            )
        }

    override fun requestStop() {
        val callNative = synchronized(stopLock) {
            when {
                NativeRunArbiter.isOwner(OWNER) -> true
                prepared -> {
                    pendingStop = true
                    false
                }
                else -> false
            }
        }
        if (callNative) NativeBridge.nativeRequestStop()
    }

    override fun prepareRun() {
        synchronized(stopLock) {
            prepared = true
            pendingStop = false
        }
    }

    override fun cancelPreparedRun() {
        synchronized(stopLock) {
            prepared = false
            pendingStop = false
        }
    }

    override fun qnnStatus(): String = NativeBridge.nativeGetQnnStatus()

    override fun runMode(
        mode: ExecutionMode,
        config: BenchmarkConfig,
        progress: (String) -> Unit,
    ): String = withOwnership {
        NativeBridge.nativeRunExecutionMode(
            executionMode = mode.nativeCode,
            batchSize = config.batchSize,
            dimension = config.dimension,
            hiddenDimension = config.hiddenDimension,
            outputDimension = config.outputDimension,
            steps = config.steps,
            warmupSteps = config.warmupSteps,
            learningRate = config.learningRate,
            seed = config.seed,
            sampleCount = config.sampleCount,
            epochs = config.epochs,
            measuredSteps = config.measuredSteps,
            correctnessInterval = config.correctnessInterval,
            benchmarkMode = config.benchmarkMode,
            seedSelectionMode = config.seedSelectionMode.nativeCode,
            trainingStabilityMode = config.trainingStabilityMode.nativeCode,
            depthPairInitMode = config.depthPairInitMode.nativeCode,
            checkpointSelectionMode = config.checkpointSelectionMode.nativeCode,
            diagnosticTrajectory = config.diagnosticTrajectory,
            diagnosticCheckpointDir = config.diagnosticCheckpointDir,
            diagnosticResumeStep = config.diagnosticResumeStep,
            diagnosticCheckpointInterval = config.diagnosticCheckpointInterval,
            progressCallback = ProgressCallback(progress),
        )
    }

    private fun withOwnership(block: () -> String): String {
        synchronized(stopLock) {
            if (pendingStop) {
                pendingStop = false
                prepared = false
                return "RESULT\nstatus=CANCELLED\nerror=run cancelled before JNI entry\n"
            }
            prepared = false
            if (!NativeRunArbiter.tryAcquire(OWNER)) {
                return "RESULT\nstatus=FAILED\nerror=another native run is already active\n"
            }
        }
        return try {
            block()
        } finally {
            NativeRunArbiter.release(OWNER)
        }
    }
}

class BenchmarkViewModel(
    private val engine: BenchmarkEngine = NativeBenchmarkEngine,
    private val runNotifications: RunNotificationSink = NoOpRunNotificationSink,
    private val worker: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-benchmark").apply { isDaemon = true }
    },
    private val uiDispatcher: UiDispatcher = AndroidUiDispatcher(),
) : Closeable {
    private val lock = Any()
    private val output = StringBuilder()
    private var running = false
    private var closed = false
    private var lastResult: BenchmarkResult? = null
    private var lastReport: String? = null
    private var configurationSummary = ""
    private var qnnStatus = "QNN status loading..."
    private var publishScheduled = false
    private var publishDirty = false

    @Volatile
    private var listener: ((BenchmarkUiState) -> Unit)? = null

    fun setListener(listener: ((BenchmarkUiState) -> Unit)?) {
        this.listener = listener
        if (listener != null) requestPublish()
    }

    fun loadEnvironment() {
        synchronized(lock) {
            if (closed) return
        }
        worker.execute {
            try {
                append(engine.environmentReport())
                val status = engine.qnnStatus()
                synchronized(lock) { qnnStatus = status }
                append(status)
            } catch (error: Throwable) {
                append("environment_status=FAILED\nerror=${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun start(config: BenchmarkConfig): Boolean {
        return startMode(ExecutionMode.fromBackend(config.backend), config)
    }

    fun startMode(mode: ExecutionMode, config: BenchmarkConfig): Boolean {
        config.validationError()?.let { error ->
            append("status=REJECTED\nerror=$error")
            return false
        }

        var startupFailure: Throwable? = null
        synchronized(lock) {
            if (closed || running) return false
            running = true
            lastResult = null
            lastReport = null
            configurationSummary =
                if (mode == ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC) {
                    "generic_configuration=B${config.batchSize} " +
                        "T${config.sampleCount} V${config.outputDimension} " +
                        "D${config.dimension} FFN${config.hiddenDimension} " +
                        "L${config.epochs} H${config.measuredSteps}" +
                        " seed_mode=${config.seedSelectionMode.name}" +
                        (if (config.seedSelectionMode == SeedSelectionMode.EXACT_SEED)
                            " exact_seed=${config.seed}" else "") +
                        " stability_mode=${config.trainingStabilityMode.name}" +
                        " checkpoint_selection=${config.checkpointSelectionMode.name}" +
                        (if (config.depthPairInitMode != DepthPairInitMode.LEGACY)
                            " pair_init=${config.depthPairInitMode.name}" else "")
                } else {
                    ""
                }
            output.appendLine()
            output.appendLine("RUN")
            output.appendLine("execution_mode=${mode.name}")
            output.appendLine("backend_requested=${config.backend.name}")
            if (configurationSummary.isNotEmpty()) output.appendLine(configurationSummary)
            // Arm the native adapter while the accepted/running state is
            // still protected by the same lock. A lifecycle close or stop
            // cannot otherwise slip between this state transition and the
            // worker's JNI entry and lose its cancellation request.
            try {
                engine.prepareRun()
                // Publish the foreground-run ownership before releasing the
                // same lock. A close cannot otherwise send its terminal event
                // before this start notification and leave an orphaned service.
                runNotifications.onRunStarted(runKind(mode), config.steps.toLong())
            } catch (error: Throwable) {
                startupFailure = error
                engine.cancelPreparedRun()
                running = false
                output.appendLine(
                    "status=FAILED\nerror=benchmark startup could not be accepted: " +
                        (error.message ?: error.javaClass.simpleName),
                )
            }
        }
        if (startupFailure != null) {
            runNotifications.onProgress(RunProgress.Cancelled)
            requestPublish()
            return false
        }
        requestPublish()

        try {
            worker.execute {
            var lastProgress = ""
            var terminalProgressSent = false
            val accepted = synchronized(lock) { !closed && running }
            if (!accepted) {
                engine.cancelPreparedRun()
                synchronized(lock) { running = false }
                runNotifications.onProgress(RunProgress.Cancelled)
                requestPublish()
                return@execute
            }
            fun forwardProgress(message: String) {
                NativeProgressParser.parse(message)?.let { progress ->
                    val terminal = progress is RunProgress.Completed ||
                        progress is RunProgress.Failed || progress is RunProgress.Cancelled
                    if (!terminal || !terminalProgressSent) {
                        runNotifications.onProgress(progress)
                        if (terminal) terminalProgressSent = true
                    }
                }
            }
            try {
                runNotifications.onProgress(RunProgress.PhaseChanged(runKind(mode)))
                val report = engine.runMode(mode, config) { message ->
                    lastProgress = message
                    append(message)
                    forwardProgress(message)
                }
                if (report != lastProgress) {
                    append(report)
                }
                synchronized(lock) {
                    lastResult = BenchmarkResult.parse(report)
                    lastReport = report
                }
                forwardProgress(report)
            } catch (error: Throwable) {
                runNotifications.onProgress(RunProgress.Failed(error.message ?: error.javaClass.simpleName))
                append(
                    "RESULT\nbackend_requested=${config.backend.name}\n" +
                        "backend_actual=UNINITIALIZED\nstatus=FAILED\n" +
                        "execution_mode=${mode.name}\n" +
                        "error=${error.message ?: error.javaClass.simpleName}",
                )
            } finally {
                synchronized(lock) {
                    running = false
                }
                requestPublish()
            }
            }
        } catch (error: Throwable) {
            // RejectedExecutionException can race close() after the
            // foreground notification was accepted. Compensate with a
            // terminal event so the dataSync service cannot remain orphaned.
            engine.cancelPreparedRun()
            synchronized(lock) { running = false }
            runNotifications.onProgress(RunProgress.Cancelled)
            append("status=FAILED\nerror=benchmark worker could not be queued: ${error.message ?: error.javaClass.simpleName}")
            requestPublish()
        }
        return true
    }

    fun requestStop(): Boolean {
        val shouldStop = synchronized(lock) { running && !closed }
        if (!shouldStop) return false
        engine.requestStop()
        append("stop_requested=true\nstop_semantics=after_current_step")
        return true
    }

    fun snapshot(): BenchmarkUiState = synchronized(lock) { currentStateLocked() }

    private fun append(message: String) {
        synchronized(lock) {
            if (output.isNotEmpty() && output.last() != '\n') output.appendLine()
            output.appendLine(message)
            trimOutputLocked()
        }
        requestPublish()
    }

    private fun trimOutputLocked() {
        if (output.length <= MAX_UI_OUTPUT_CHARS) return
        val retainedCharacters = MAX_UI_OUTPUT_CHARS - TRUNCATED_OUTPUT_MARKER.length
        output.delete(0, output.length - retainedCharacters)
        output.insert(0, TRUNCATED_OUTPUT_MARKER)
    }

    private fun currentStateLocked() = BenchmarkUiState(
        running = running,
        output = output.toString(),
        lastResult = lastResult,
        lastReport = lastReport,
        configurationSummary = configurationSummary,
        qnnStatus = qnnStatus,
    )

    private fun requestPublish() {
        val schedule = synchronized(lock) {
            publishDirty = true
            if (publishScheduled) {
                false
            } else {
                publishScheduled = true
                true
            }
        }
        if (schedule) uiDispatcher.dispatch(::drainPublish)
    }

    private fun drainPublish() {
        val state = synchronized(lock) {
            publishDirty = false
            currentStateLocked()
        }
        listener?.invoke(state)
        val reschedule = synchronized(lock) {
            if (publishDirty) {
                true
            } else {
                publishScheduled = false
                false
            }
        }
        if (reschedule) uiDispatcher.dispatch(::drainPublish)
    }

    override fun close() {
        val stop = synchronized(lock) {
            if (closed) return
            closed = true
            running
        }
        if (stop) engine.requestStop()
        if (stop) runNotifications.onProgress(RunProgress.Cancelled)
        worker.shutdown()
    }

    private fun runKind(mode: ExecutionMode) = when {
        mode.name.contains("INFERENCE") || mode.name.contains("FORWARD") -> "推論"
        mode.name.contains("BENCHMARK") -> "ベンチマーク"
        else -> "学習"
    }

    private class AndroidUiDispatcher : UiDispatcher {
        private val handler = Handler(Looper.getMainLooper())
        override fun dispatch(block: () -> Unit) {
            handler.post(block)
        }
    }
}
