package com.yuubinnkyoku.phonelm

import android.content.Context
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Instrumentation entry point: never launches MainActivity. */
@RunWith(AndroidJUnit4::class)
class HeadlessDeviceTestRunner {
    @Test
    fun runRequestedSuiteWithoutActivity() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        val suite = arguments.getString("suite") ?: "device-probe"
        val testMode = arguments.getString("testMode") ?: "BACKGROUND_CORRECTNESS"
        val liveUpdateNotification = arguments.getString("liveUpdateNotification")?.let {
            it.toBooleanStrictOrNull() ?: throw IllegalArgumentException(
                "liveUpdateNotification must be true or false",
            )
        } ?: false
        val nicopediaSuite = suite in NICOPEDIA_SUITES
        require(testMode == "BACKGROUND_CORRECTNESS" || testMode == "EXCLUSIVE_BENCHMARK") {
            "Unknown headless test mode"
        }
        require(suite != "nicopedia-parity" || testMode == "BACKGROUND_CORRECTNESS") {
            "nicopedia-parity is restricted to BACKGROUND_CORRECTNESS"
        }
        require(!nicopediaSuite || testMode == "BACKGROUND_CORRECTNESS") {
            "$suite is restricted to BACKGROUND_CORRECTNESS"
        }
        require(!nicopediaSuite || !liveUpdateNotification) {
            "$suite does not allow live update notifications"
        }
        require(suite.length <= 64) { "suite is too long" }
        val rawRunId = arguments.getString("runId") ?: "local"
        require(rawRunId.length in 1..64 && rawRunId.matches(Regex("[A-Za-z0-9._-]+"))) {
            "runId must match [A-Za-z0-9._-]+ and be at most 64 characters"
        }
        val runId = rawRunId
        val state = HeadlessTestState.forContext(context)
        val acquired = state.acquire()
        if (acquired.lease == null) {
            // Do not replace the live runner's status record just to report contention.
            throw AssertionError("ALREADY_RUNNING existing_status=${acquired.existingStatus}")
        }
        acquired.lease.use {
            HeadlessActivityCounters.reset()
            val started = System.currentTimeMillis()
            var reportPath = ""
            var phase = "initializing"
            var test = "environment"
            val currentPhase = AtomicReference(phase)
            val currentTest = AtomicReference(test)
            state.write(HeadlessStatus(runId, suite, "STARTING", phase, test, 0, 2, startTime = started))
            val contender = state.acquire()
            check(contender.lease == null && contender.existingStatus?.contains("\"run_id\":\"$runId\"") == true) {
                contender.lease?.close()
                "single-flight lock admitted a second runner"
            }
            val singleFlightResult = "ALREADY_RUNNING"
            val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PhoneLM:headless-$runId")
            try {
                wakeLock.acquire(4 * 60 * 60 * 1000L)
                QnnEnvironment.prepare(context)
                phase = "native"
                test = suite
                currentPhase.set(phase)
                currentTest.set(test)
                state.write(HeadlessStatus(runId, suite, "RUNNING", phase, test, 1, 2, startTime = started))
                val mode = if (suite == "nicopedia-parity" || nicopediaSuite) null else modeFor(suite)
                val notification = if (liveUpdateNotification) LiveUpdateNotificationController(context) else null
                notification?.onRunStarted("QNN数値検証", 1)
                notification?.onProgress(RunProgress.PhaseChanged(suite))
                val heartbeatRunning = AtomicBoolean(true)
                val heartbeat = Thread({
                    while (heartbeatRunning.get()) {
                        try {
                            Thread.sleep(30_000L)
                            if (heartbeatRunning.get()) {
                                state.write(HeadlessStatus(runId, suite, "RUNNING", currentPhase.get(), currentTest.get(), 1, 2,
                                    startTime = started, lastHeartbeat = System.currentTimeMillis()))
                            }
                        } catch (_: InterruptedException) {
                            break
                        }
                    }
                }, "PhoneLM-headless-heartbeat").apply { isDaemon = true; start() }
                val lastProgressStatus = AtomicLong(started)
                val progressCallback = ProgressCallback { message ->
                    val event = NativeProgressParser.parse(message)
                    if (event != null) {
                        when (event) {
                            is RunProgress.PhaseChanged -> {
                                phase = event.phase
                                test = event.phase
                                currentPhase.set(event.phase)
                                currentTest.set(event.phase)
                            }
                            is RunProgress.Step -> {
                                event.phase?.let {
                                    phase = it
                                    currentPhase.set(it)
                                }
                                val stepLabel = event.phase ?: "step-${event.completed}"
                                test = stepLabel
                                currentTest.set(stepLabel)
                            }
                            is RunProgress.Started -> {
                                test = event.kind
                                currentTest.set(event.kind)
                            }
                            is RunProgress.Completed -> {
                                test = "complete"
                                currentTest.set(test)
                            }
                            is RunProgress.Failed -> {
                                test = "failed"
                                currentTest.set(test)
                            }
                            RunProgress.Cancelled -> {
                                test = "cancelled"
                                currentTest.set(test)
                            }
                        }
                        notification?.onProgress(event)
                        val now = System.currentTimeMillis()
                        val terminal = event is RunProgress.Completed ||
                            event is RunProgress.Failed || event is RunProgress.Cancelled
                        val previous = lastProgressStatus.get()
                        if (terminal || now - previous >= PROGRESS_STATUS_INTERVAL_MS) {
                            if (terminal || lastProgressStatus.compareAndSet(previous, now)) {
                                state.write(HeadlessStatus(runId, suite, "RUNNING",
                                    currentPhase.get(), currentTest.get(), 1, 2,
                                    startTime = started, lastHeartbeat = now))
                            }
                        }
                    }
                }
                val report = try {
                    when {
                        suite == "nicopedia-parity" ->
                            runNicopediaParity(context, arguments, runId)
                        suite == "nicopedia-long-training" ->
                            runNicopediaLongTraining(context, arguments, runId, progressCallback)
                        suite == "nicopedia-eval" ->
                            runNicopediaEvaluate(context, arguments, runId)
                        suite == "nicopedia-generate" ->
                            runNicopediaGenerate(context, arguments, runId)
                        else ->
                            NativeBridge.nativeRunExecutionMode(
                                executionMode = requireNotNull(mode).nativeCode, batchSize = 2, dimension = 4, hiddenDimension = 5,
                                outputDimension = 3, steps = if (suite == "qnn-reproducibility") 2 else 1,
                                warmupSteps = 0, learningRate = 0.1f, seed = 20_260_710L, sampleCount = 2,
                                epochs = 0, measuredSteps = 0, correctnessInterval = 1, benchmarkMode = false,
                                seedSelectionMode = 0,
                                trainingStabilityMode = 0,
                                depthPairInitMode = 0,
                                checkpointSelectionMode = 0,
                                diagnosticTrajectory = false,
                                diagnosticCheckpointDir = null,
                                diagnosticResumeStep = 0,
                                diagnosticCheckpointInterval = 250,
                                progressCallback = progressCallback,
                            )
                    }
                } finally {
                    heartbeatRunning.set(false)
                    heartbeat.interrupt()
                    heartbeat.join(5_000L)
                }
                notification?.onProgress(RunProgress.Completed(null))
                reportPath = state.writeReport(runId, report)
                val success = isSuccessfulSuiteResult(suite, report, splitForSuite(suite, arguments)) ||
                    (suite == "generation-diagnostics" &&
                        Regex("(?m)^status=PARTIAL_SUCCESS$").containsMatchIn(report))
                val countersOk = HeadlessActivityCounters.create.get() == 0 &&
                    HeadlessActivityCounters.resume.get() == 0 && HeadlessActivityCounters.becameTop.get() == 0 &&
                    HeadlessActivityCounters.focusTakeover.get() == 0
                // Nicopedia native reports already carry the authoritative
                // fallback flag.  Do not append a second false line that
                // could mask a native fallback in a key/value consumer.
                val fallbackAnnotation = if (nicopediaSuite) "" else "\ncpu_fallback=false"
                val appended = report.trimEnd() + "\nactivity_create_count=${HeadlessActivityCounters.create.get()}" +
                    "\nactivity_resume_count=${HeadlessActivityCounters.resume.get()}" +
                    "\nphonelm_became_top_activity_count=${HeadlessActivityCounters.becameTop.get()}" +
                    "\nfocus_takeover_count=${HeadlessActivityCounters.focusTakeover.get()}" +
                    "\nsingle_flight_result=$singleFlightResult" +
                    "\nheadless_test_mode=$testMode" +
                    "\nbackend_requested=HTP" +
                    "\nlive_update_notification_enabled=$liveUpdateNotification" +
                    fallbackAnnotation + "\n"
                reportPath = state.writeReport(runId, appended)
                state.write(HeadlessStatus(runId, suite, if (success && countersOk) "PASSED" else "FAILED", "complete", test, 2, 2,
                    result = if (success) "SUCCESS" else "NATIVE_FAILED", failureCode = if (countersOk) "" else "ACTIVITY_LAUNCHED", reportRelativePath = reportPath, startTime = started))
                assertTrue("native result failed", success)
                assertTrue("PhoneLM Activity was launched", countersOk)
            } catch (error: Throwable) {
                state.write(HeadlessStatus(runId, suite, "FAILED", currentPhase.get(), currentTest.get(), 0, 2, result = "FAILED",
                    failureCode = error.javaClass.simpleName, reportRelativePath = reportPath, startTime = started))
                throw error
            } finally {
                if (wakeLock.isHeld) wakeLock.release()
            }
        }
    }

    private data class NicopediaArguments(
        val seed: Long,
        val layers: Int,
        val heads: Int,
        val steps: Int,
        val batchSize: Int,
        val resumeStep: Int,
        val checkpointInterval: Int,
        val validationChunks: Int,
        val developmentChunks: Int,
        val checkpointStep: Int,
        val generateMode: String,
        val maxNewBytes: Int,
        val temperature: Float,
        val topK: Int,
        val samplingSeed: Long,
        val gatePolicy: String,
    )

    private fun parseNicopediaArguments(
        arguments: android.os.Bundle,
        suite: String,
    ): NicopediaArguments {
        // Instrumentation extras are strings.  Reject non-string values and
        // malformed numeric text instead of silently falling back to a
        // different trajectory.
        val seed = longArgument(arguments, "seed", 1L, 1L..99_999L)
        val layers = intArgument(arguments, "layers", 19, 1..100)
        val heads = intArgument(arguments, "heads", 2, 1..32)
        val steps = intArgument(arguments, "steps", 1_000, 1..100_000)
        val batchSize = intArgument(arguments, "batchSize", 8, 1..4_096)
        val resumeStep = intArgument(arguments, "resumeStep", 0, 0..100_000)
        val checkpointInterval = intArgument(arguments, "checkpointInterval", 250, 1..10_000)
        val validationChunks = intArgument(arguments, "validationChunks", 8_192, 1..1_000_000)
        val developmentChunks = intArgument(arguments, "developmentChunks", 16_384, 1..1_000_000)
        val checkpointStep = intArgument(arguments, "checkpointStep", 1_000, 1..100_000)
        val generateMode = stringArgument(arguments, "generateMode", "greedy")
        require(generateMode == "greedy" || generateMode == "sample") {
            "generateMode must be greedy or sample"
        }
        val maxNewBytes = intArgument(arguments, "maxNewBytes", 64, 1..1_024)
        val temperature = floatArgument(arguments, "temperature", 0.6f, 0.0001f..100f)
        val topK = intArgument(arguments, "topK", 16, 1..256)
        val samplingSeed = longArgument(arguments, "samplingSeed", 42L, 0L..Long.MAX_VALUE)
        val gatePolicy = stringArgument(arguments, "gatePolicy", "legacy")
        require(gatePolicy == "legacy" || gatePolicy == "candidate" || gatePolicy == "htp-native") {
            "gatePolicy must be legacy, candidate, or htp-native"
        }

        require(layers == 19) { "$suite requires layers=19" }
        require(heads == 2) { "$suite requires heads=2" }
        if (suite == "nicopedia-long-training") {
            require(batchSize == 8) { "nicopedia-long-training requires batchSize=8" }
            require(steps in 1..8_000) { "nicopedia-long-training hard ceiling is step 8000" }
            require(resumeStep < steps) { "resumeStep must be smaller than steps" }
        }
        if (suite == "nicopedia-eval" || suite == "nicopedia-generate") {
            require(checkpointStep >= 1) { "$suite requires checkpointStep >= 1" }
        }
        return NicopediaArguments(
            seed = seed,
            layers = layers,
            heads = heads,
            steps = steps,
            batchSize = batchSize,
            resumeStep = resumeStep,
            checkpointInterval = checkpointInterval,
            validationChunks = validationChunks,
            developmentChunks = developmentChunks,
            checkpointStep = checkpointStep,
            generateMode = generateMode,
            maxNewBytes = maxNewBytes,
            temperature = temperature,
            topK = topK,
            samplingSeed = samplingSeed,
            gatePolicy = gatePolicy,
        )
    }

    private fun stringArgument(arguments: android.os.Bundle, name: String, default: String): String {
        if (!arguments.containsKey(name)) return default
        val value = arguments.get(name)
        require(value is String) { "$name must be a string" }
        require(value.isNotEmpty() && value.trim() == value) { "$name must not be empty or padded" }
        return value
    }

    private fun intArgument(arguments: android.os.Bundle, name: String, default: Int, range: IntRange): Int {
        if (!arguments.containsKey(name)) return default
        val raw = arguments.get(name)
        require(raw is String) { "$name must be a decimal string" }
        require(raw.matches(DECIMAL_INTEGER)) { "$name must be a decimal integer" }
        val value = raw.toLongOrNull() ?: throw IllegalArgumentException("$name is out of range")
        require(value in range.first.toLong()..range.last.toLong()) { "$name must be in ${range.first}..${range.last}" }
        return value.toInt()
    }

    private fun longArgument(arguments: android.os.Bundle, name: String, default: Long, range: LongRange): Long {
        if (!arguments.containsKey(name)) return default
        val raw = arguments.get(name)
        require(raw is String) { "$name must be a decimal string" }
        require(raw.matches(DECIMAL_INTEGER)) { "$name must be a decimal integer" }
        val value = raw.toLongOrNull() ?: throw IllegalArgumentException("$name is out of range")
        require(value in range) { "$name must be in ${range.first}..${range.last}" }
        return value
    }

    private fun floatArgument(arguments: android.os.Bundle, name: String, default: Float, range: ClosedFloatingPointRange<Float>): Float {
        if (!arguments.containsKey(name)) return default
        val raw = arguments.get(name)
        require(raw is String) { "$name must be a finite decimal" }
        require(raw.matches(DECIMAL_FLOAT)) { "$name must be a finite decimal" }
        val value = raw.toFloatOrNull() ?: throw IllegalArgumentException("$name is out of range")
        require(value.isFinite() && value in range) { "$name must be in ${range.start}..${range.endInclusive}" }
        return value
    }

    private fun nicopediaInputDirectory(context: Context, runId: String): File {
        val filesRoot = context.filesDir.canonicalFile
        val directory = File(filesRoot, "headless-input/$runId").canonicalFile
        require(directory.path.startsWith(filesRoot.path + File.separator)) {
            "headless input directory escaped app files"
        }
        require(directory.isDirectory) { "headless input directory is unavailable" }
        return directory
    }

    private fun requiredInputFile(directory: File, name: String): File {
        val file = File(directory, name).canonicalFile
        require(file.path.startsWith(directory.path + File.separator)) {
            "headless input file escaped run directory"
        }
        require(file.isFile) { "headless input file is unavailable: $name" }
        return file
    }

    private fun runNicopediaLongTraining(
        context: Context,
        arguments: android.os.Bundle,
        runId: String,
        progressCallback: ProgressCallback,
    ): String {
        val config = parseNicopediaArguments(arguments, "nicopedia-long-training")
        val directory = nicopediaInputDirectory(context, runId)
        requiredInputFile(directory, "train_pilot.bin")
        if (config.resumeStep > 0) {
            requiredInputFile(directory,
                "htp-seed${config.seed}-l${config.layers}-step${config.resumeStep}.ckpt")
        }
        return NativeBridge.nativeRunExecutionMode(
            executionMode = ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA.nativeCode,
            batchSize = config.batchSize,
            dimension = 16,
            hiddenDimension = 32,
            outputDimension = 256,
            steps = config.steps,
            warmupSteps = 0,
            learningRate = 0.003f,
            seed = config.seed,
            sampleCount = 32,
            epochs = config.layers,
            measuredSteps = config.heads,
            correctnessInterval = config.seed.toInt(),
            benchmarkMode = false,
            seedSelectionMode = 1,
            trainingStabilityMode = 0,
            depthPairInitMode = 0,
            checkpointSelectionMode = 0,
            diagnosticTrajectory = false,
            diagnosticCheckpointDir = directory.absolutePath,
            diagnosticResumeStep = config.resumeStep,
            diagnosticCheckpointInterval = config.checkpointInterval,
            progressCallback = progressCallback,
        )
    }

    private fun runNicopediaEvaluate(
        context: Context,
        arguments: android.os.Bundle,
        runId: String,
    ): String {
        val config = parseNicopediaArguments(arguments, "nicopedia-eval")
        val directory = nicopediaInputDirectory(context, runId)
        requiredInputFile(directory,
            "htp-seed${config.seed}-l${config.layers}-step${config.checkpointStep}.ckpt")
        requiredInputFile(directory, "validation.bin")
        requiredInputFile(directory, "development.bin")
        return NativeBridge.nativeRunNicopediaEvaluate(
            checkpointDir = directory.absolutePath,
            seed = config.seed,
            layers = config.layers,
            heads = config.heads,
            checkpointStep = config.checkpointStep,
            validationChunks = config.validationChunks,
            developmentChunks = config.developmentChunks,
        )
    }

    private fun runNicopediaGenerate(
        context: Context,
        arguments: android.os.Bundle,
        runId: String,
    ): String {
        val config = parseNicopediaArguments(arguments, "nicopedia-generate")
        val directory = nicopediaInputDirectory(context, runId)
        val checkpoint = requiredInputFile(directory,
            "htp-seed${config.seed}-l${config.layers}-step${config.checkpointStep}.ckpt")
        val prompt = requiredInputFile(directory, "prompt.bin")
        return NativeBridge.nativeRunNicopediaGenerate(
            checkpointPath = checkpoint.absolutePath,
            promptPath = prompt.absolutePath,
            seed = config.seed,
            layers = config.layers,
            maxNewBytes = config.maxNewBytes,
            generateMode = config.generateMode,
            temperature = config.temperature,
            topK = config.topK,
            samplingSeed = config.samplingSeed,
            gatePolicy = config.gatePolicy,
            htpGraphPrecisionMode = 0,
            htpGraphPrecisionCompensation = 0,
            htpGraphWeightsPacking = 0,
            htpGraphAdvancedActivationFusion = 0,
            htpContextGraphSplitting = 0,
            htpNativeTensorFp16 = false,
        )
    }

    private fun modeFor(suite: String): ExecutionMode = when (suite) {
        "device-probe" -> ExecutionMode.QNN_HTP_DEVICE_PROBE
        "qnn-forward" -> ExecutionMode.QNN_HTP_FORWARD
        "linear" -> ExecutionMode.QNN_HTP_LINEAR_TRAINING
        "mlp-split" -> ExecutionMode.QNN_HTP_MLP_HTP_LINEAR_BACKWARD
        "mlp-fused" -> ExecutionMode.QNN_HTP_MLP_FUSED_BACKWARD
        "mlp-full-step" -> ExecutionMode.QNN_HTP_MLP_FULL_STEP
        "transformer-forward" -> ExecutionMode.QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK
        "softmax-backward" -> ExecutionMode.QNN_HTP_SOFTMAX_BACKWARD_CHECK
        "attention-backward" -> ExecutionMode.QNN_HTP_ATTENTION_BACKWARD_CHECK
        "layernorm-backward" -> ExecutionMode.QNN_HTP_LAYER_NORM_BACKWARD_CHECK
        "transformer-mse" -> ExecutionMode.QNN_HTP_TINY_TRANSFORMER_TRAINING_STEP
        "tiny-lm-ce" -> ExecutionMode.QNN_HTP_CROSS_ENTROPY_CHECK
        "sgd-one-step" -> ExecutionMode.QNN_HTP_SGD_CHECK
        "momentum-one-step" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_STEP
        "adam-one-step" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP
        "tiny-lm-stability" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_1
        "phase01-adam" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_2
        "generation-diagnostics" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_INFERENCE
        "api-trace" -> ExecutionMode.QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE
        "callback-bound" -> ExecutionMode.QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE
        "qnn-reproducibility" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_REPRODUCIBILITY
        "qnn-graph-bisection" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_BISECTION
        "qnn-graph-bisection-prelude" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_BISECTION_PRELUDE
        "qnn-graph-full-isolated" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_FULL_ISOLATED
        "qnn-graph-dinput-isolated" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_DINPUT_ISOLATED
        "qnn-graph-dembedding-isolated" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_DEMBEDDING_ISOLATED
        "qnn-graph-order-full-dinput-dembedding" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_FULL_DINPUT_DEMBEDDING
        "qnn-graph-order-full-dembedding-dinput" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_FULL_DEMBEDDING_DINPUT
        "qnn-graph-order-dinput-full-dembedding" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DINPUT_FULL_DEMBEDDING
        "qnn-graph-order-dinput-dembedding-full" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DINPUT_DEMBEDDING_FULL
        "qnn-graph-order-dembedding-full-dinput" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DEMBEDDING_FULL_DINPUT
        "qnn-graph-order-dembedding-dinput-full" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DEMBEDDING_DINPUT_FULL
        "qnn-graph-order-full-full-full" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_FULL_FULL_FULL
        "qnn-graph-order-dinput-dinput-dinput" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DINPUT_DINPUT_DINPUT
        "qnn-graph-order-dembedding-dembedding-dembedding" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DEMBEDDING_DEMBEDDING_DEMBEDDING
        "nicopedia-parity" -> throw IllegalStateException("nicopedia-parity does not use ExecutionMode")
        "qnn-tap-backward-regions" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_TAP_BACKWARD_REGIONS
        "qnn-tap-layernorm1" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_TAP_LAYERNORM1
        "qnn-tap-dscores-only" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_TAP_DSCORES_ONLY
        "qnn-tap-dprob-dscores" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_TAP_DPROB_DSCORES
        "qnn-adam-diagnostic" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP
        "qnn-adam-late-baseline" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_LATE_NONFINITE_BASELINE
        "qnn-adam-late-diagnostic" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_LATE_NONFINITE_DIAGNOSTIC
        "post-fix-end-to-end" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_POST_FIX_END_TO_END
        "scale-sequence-16-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_SEQUENCE_16_SMOKE
        "scale-sequence-32-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_SEQUENCE_32_SMOKE
        "scale-dimension-32-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_DIMENSION_32_SMOKE
        "scale-layers-2-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_LAYERS_2_SMOKE
        "scale-heads-2-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_HEADS_2_SMOKE
        "scale-formal" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_FORMAL
        "scale-l2h1-t16d16-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T16D16_SMOKE
        "scale-l2h1-t32d32-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T32D32_SMOKE
        "scale-l1h2-t16d16-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T16D16_SMOKE
        "scale-l1h2-t32d32-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T32D32_SMOKE
        "scale-l2h2-t16d16-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T16D16_SMOKE
        "scale-l2h2-t32d32-smoke" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_SMOKE
        "scale-l2h1-formal" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_FORMAL
        "scale-l1h2-formal" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_FORMAL
        "scale-l2h2-t32d32-formal" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_FORMAL
        "scale-l2h2-t32d32-diagnostic" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_DIAGNOSTIC
        else -> throw IllegalArgumentException("Unknown headless suite: $suite")
    }

    private fun runNicopediaParity(
        context: Context,
        arguments: android.os.Bundle,
        runId: String,
    ): String {
        val split = arguments.getString("htpContextGraphSplitting")?.toIntOrNull()
            ?: throw IllegalArgumentException("htpContextGraphSplitting is required for nicopedia-parity")
        require(split in 0..2) { "htpContextGraphSplitting must be 0, 1, or 2" }
        val inputDirectory = File(context.filesDir, "headless-input/$runId")
        val checkpoint = File(inputDirectory, "htp-seed1-l19-step1000.ckpt")
        val prompt = File(inputDirectory, "prompt.bin")
        require(checkpoint.isFile) { "nicopedia parity checkpoint is unavailable" }
        require(prompt.isFile) { "nicopedia parity prompt is unavailable" }
        return NativeBridge.nativeRunNicopediaGenerate(
            checkpointPath = checkpoint.absolutePath,
            promptPath = prompt.absolutePath,
            seed = 1L,
            layers = 19,
            // If the unchanged legacy gate unexpectedly passes, this is the
            // preregistered fixed Greedy generation (64 bytes). On the known
            // reject path no generation executes and the count remains zero.
            maxNewBytes = 64,
            generateMode = "greedy",
            temperature = 1.0f,
            topK = 1,
            samplingSeed = 0L,
            gatePolicy = "legacy",
            htpGraphPrecisionMode = 0,
            htpGraphPrecisionCompensation = 0,
            htpGraphWeightsPacking = 0,
            htpGraphAdvancedActivationFusion = 0,
            htpContextGraphSplitting = split,
            htpNativeTensorFp16 = false,
        )
    }

    private fun splitForSuite(suite: String, arguments: android.os.Bundle): Int? =
        if (suite == "nicopedia-parity") arguments.getString("htpContextGraphSplitting")?.toIntOrNull() else null

    private fun isSuccessfulSuiteResult(suite: String, report: String, requestedSplit: Int?): Boolean {
        if (suite != "nicopedia-parity") {
            val statusOk = Regex("(?m)^status=SUCCESS$").containsMatchIn(report)
            if (suite !in NICOPEDIA_SUITES) return statusOk
            val prefixOk = when (suite) {
                "nicopedia-long-training" -> report.startsWith("NICOPEDIA_HTP\n")
                "nicopedia-eval" -> report.startsWith("NICOPEDIA_HTP_EVAL\n")
                "nicopedia-generate" -> report.startsWith("NICOPEDIA_HTP_GENERATION\n")
                else -> false
            }
            val fallbackOk = Regex("(?m)^cpu_fallback=false$").containsMatchIn(report)
            val finiteOk = !Regex("(?m)^(?:nan_detected|inf_detected)=true$").containsMatchIn(report)
            return statusOk && prefixOk && fallbackOk && finiteOk
        }
        val values = report.lineSequence()
            .mapNotNull { line -> line.split('=', limit = 2).takeIf { it.size == 2 } }
            .associate { it[0] to it[1] }
        fun required(name: String): String = values[name]
            ?: throw AssertionError("nicopedia parity report is missing $name")
        val expectedRuntimeRequest = when (requestedSplit) {
            0 -> "unset"
            1 -> "false"
            2 -> "true"
            else -> throw AssertionError("nicopedia parity requested split is invalid")
        }
        val contextConfigRejected = requestedSplit != 0 &&
            required("status") == "FAILED" &&
            values["error"] == "context_create: contextCreate=5010" &&
            values["failed_api"] == "context_create"
        if (contextConfigRejected) {
            assertTrue("context rejection did not carry the requested split",
                required("htp_context_graph_splitting_runtime") == expectedRuntimeRequest)
            assertTrue("context rejection did not deliver one non-null config",
                required("context_create_config_pointer_null") == "false" &&
                    required("context_create_config_count") == "1")
            assertTrue("context rejection did not return QNN_CONTEXT_ERROR_INVALID_CONFIG",
                required("context_create_result") == "5010" &&
                    required("api_trace_context_create_result") == "5010")
            assertTrue("context rejection unexpectedly returned a context handle",
                required("context_handle_null") == "true" &&
                    required("api_trace_context_handle_nonnull") == "false")
            assertTrue("context rejection unexpectedly attempted graph execution",
                required("api_trace_graph_execute_attempt_count") == "0" &&
                    required("api_trace_graph_execute_success_count") == "0" &&
                    required("api_trace_graph_execute_failure_count") == "0")
            assertTrue("context rejection initialized the CPU backend",
                required("api_trace_cpu_backend_initialized") == "false")
            assertTrue("context rejection attempted CPU fallback",
                required("api_trace_fallback_attempted") == "false" &&
                    required("api_trace_fallback_succeeded") == "false" &&
                    required("cpu_fallback") == "false")
            return true
        }
        val rejected = required("status") == "FAILED" &&
            required("failure_classification") == "PARITY_GATE_REJECTED"
        val generated = required("generated_byte_count").toIntOrNull()
            ?: throw AssertionError("nicopedia parity generated_byte_count is invalid")
        assertTrue("nicopedia parity must keep the legacy gate", required("gate_policy") == "legacy")
        assertTrue("nicopedia parity checkpoint must be finite", required("checkpoint_finite") == "true")
        assertTrue("nicopedia parity checkpoint identity is invalid",
            required("checkpoint_parameter_hash").matches(Regex("fnv1a64:[0-9a-f]{16}")))
        assertTrue("nicopedia parity checkpoint elements are invalid",
            (required("checkpoint_parameter_elements").toLongOrNull() ?: 0L) > 0L)
        assertTrue("nicopedia parity checkpoint size is invalid",
            (required("checkpoint_file_bytes").toLongOrNull() ?: 0L) > 0L)
        assertTrue("nicopedia parity report is incomplete", required("parity_prefix_count") == "20")
        assertTrue("nicopedia parity report is missing parity_13", values.containsKey("parity_13_logits_max_abs_error"))
        for (index in 0 until 20) {
            assertTrue("nicopedia parity prefix $index is non-finite",
                required("parity_${index}_finite") == "true")
        }
        assertTrue("nicopedia AR report is incomplete", required("ar_steps") == "8")
        for (index in 0 until 8) {
            assertTrue("nicopedia AR step $index is non-finite",
                required("ar_step_${index}_finite") == "true")
        }
        assertTrue("nicopedia parity reported NaN", required("nan_detected") == "false")
        assertTrue("nicopedia parity reported Inf", required("inf_detected") == "false")
        assertTrue("nicopedia parity initialized the CPU backend",
            required("api_trace_cpu_backend_initialized") == "false")
        assertTrue("nicopedia parity attempted CPU fallback",
            required("api_trace_fallback_attempted") == "false")
        assertTrue("nicopedia parity completed CPU fallback",
            required("api_trace_fallback_succeeded") == "false")
        assertTrue("nicopedia parity had a QNN execute failure",
            required("api_trace_graph_execute_failure_count") == "0")
        assertTrue("nicopedia parity did not complete a QNN execute",
            (required("api_trace_graph_execute_success_count").toLongOrNull() ?: 0L) > 0L)
        assertTrue("nicopedia parity last QNN result was nonzero",
            required("api_trace_last_qnn_result") == "0")
        assertTrue("nicopedia parity execution fingerprint is invalid",
            required("htp_execution_fingerprint_sha256").matches(Regex("[0-9a-f]{64}")))
        assertTrue("nicopedia parity report used the requested split",
            required("htp_context_graph_splitting") == requestedSplit?.toString())
        assertTrue("nicopedia parity runtime split request is invalid",
            required("htp_context_graph_splitting_runtime") == expectedRuntimeRequest)
        val expectedDelivery = if (requestedSplit == 0) {
            "unset_nullptr"
        } else {
            "passed_to_qnn_context_create"
        }
        assertTrue("nicopedia parity context config delivery is invalid",
            required("htp_context_graph_splitting_delivery") == expectedDelivery)
        if (rejected) {
            assertTrue("legacy parity gate must remain closed", required("parity_gate") == "false")
            assertTrue("candidate parity gate must remain closed", required("parity_gate_candidate") == "false")
            assertTrue("generation ran despite a rejected legacy parity gate", generated == 0)
            assertTrue("generation gate must remain closed after legacy parity rejection",
                required("generation_gate") == "false")
        } else {
            assertTrue("successful parity did not pass the legacy gate",
                required("parity_gate") == "true")
            assertTrue("successful parity did not open the generation gate",
                required("generation_gate") == "true")
            assertTrue("successful parity did not complete the fixed Greedy generation",
                generated == 64)
        }
        return rejected || required("status") == "SUCCESS"
    }

    private companion object {
        val NICOPEDIA_SUITES = setOf(
            "nicopedia-long-training",
            "nicopedia-eval",
            "nicopedia-generate",
        )
        const val PROGRESS_STATUS_INTERVAL_MS = 1_000L
        val DECIMAL_INTEGER = Regex("[+-]?(?:0|[1-9][0-9]*)")
        val DECIMAL_FLOAT = Regex(
            "[+-]?(?:(?:[0-9]+(?:\\.[0-9]*)?)|(?:\\.[0-9]+))(?:[eE][+-]?[0-9]+)?",
        )
    }
}
