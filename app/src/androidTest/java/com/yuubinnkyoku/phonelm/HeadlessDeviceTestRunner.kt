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
        val liveUpdateNotification = arguments.getString("liveUpdateNotification")?.toBooleanStrictOrNull() ?: false
        require(testMode == "BACKGROUND_CORRECTNESS" || testMode == "EXCLUSIVE_BENCHMARK") {
            "Unknown headless test mode"
        }
        require(suite != "nicopedia-parity" || testMode == "BACKGROUND_CORRECTNESS") {
            "nicopedia-parity is restricted to BACKGROUND_CORRECTNESS"
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
                state.write(HeadlessStatus(runId, suite, "RUNNING", phase, test, 1, 2, startTime = started))
                val mode = if (suite == "nicopedia-parity") null else modeFor(suite)
                val notification = if (liveUpdateNotification) LiveUpdateNotificationController(context) else null
                notification?.onRunStarted("QNN数値検証", 1)
                notification?.onProgress(RunProgress.PhaseChanged(suite))
                val heartbeatRunning = AtomicBoolean(true)
                val heartbeat = Thread({
                    while (heartbeatRunning.get()) {
                        try {
                            Thread.sleep(30_000L)
                            if (heartbeatRunning.get()) {
                                state.write(HeadlessStatus(runId, suite, "RUNNING", phase, test, 1, 2,
                                    startTime = started, lastHeartbeat = System.currentTimeMillis()))
                            }
                        } catch (_: InterruptedException) {
                            break
                        }
                    }
                }, "PhoneLM-headless-heartbeat").apply { isDaemon = true; start() }
                val report = try {
                    if (suite == "nicopedia-parity") {
                        runNicopediaParity(context, arguments, runId)
                    } else {
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
                            progressCallback = ProgressCallback { message ->
                                NativeProgressParser.parse(message)?.let { notification?.onProgress(it) }
                            },
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
                val appended = report.trimEnd() + "\nactivity_create_count=${HeadlessActivityCounters.create.get()}" +
                    "\nactivity_resume_count=${HeadlessActivityCounters.resume.get()}" +
                    "\nphonelm_became_top_activity_count=${HeadlessActivityCounters.becameTop.get()}" +
                    "\nfocus_takeover_count=${HeadlessActivityCounters.focusTakeover.get()}" +
                    "\nsingle_flight_result=$singleFlightResult" +
                    "\nheadless_test_mode=$testMode" +
                    "\ncompile_time_qairt_build_id=${BuildConfig.QAIRT_BUILD_ID}" +
                    "\nbackend_requested=HTP" +
                    "\nlive_update_notification_enabled=$liveUpdateNotification" +
                    "\ncpu_fallback=false\n"
                reportPath = state.writeReport(runId, appended)
                state.write(HeadlessStatus(runId, suite, if (success && countersOk) "PASSED" else "FAILED", "complete", test, 2, 2,
                    result = if (success) "SUCCESS" else "NATIVE_FAILED", failureCode = if (countersOk) "" else "ACTIVITY_LAUNCHED", reportRelativePath = reportPath, startTime = started))
                assertTrue("native result failed", success)
                assertTrue("PhoneLM Activity was launched", countersOk)
            } catch (error: Throwable) {
                state.write(HeadlessStatus(runId, suite, "FAILED", phase, test, 0, 2, result = "FAILED",
                    failureCode = error.javaClass.simpleName, reportRelativePath = reportPath, startTime = started))
                throw error
            } finally {
                if (wakeLock.isHeld) wakeLock.release()
            }
        }
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
            return Regex("(?m)^status=SUCCESS$").containsMatchIn(report)
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
}
