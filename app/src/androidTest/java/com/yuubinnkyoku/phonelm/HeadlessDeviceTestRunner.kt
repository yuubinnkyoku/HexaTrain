package com.yuubinnkyoku.phonelm

import android.content.Context
import android.os.PowerManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
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
        require(testMode == "BACKGROUND_CORRECTNESS" || testMode == "EXCLUSIVE_BENCHMARK") {
            "Unknown headless test mode"
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
                val mode = modeFor(suite)
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
                    NativeBridge.nativeRunExecutionMode(
                        executionMode = mode.nativeCode, batchSize = 2, dimension = 4, hiddenDimension = 5,
                        outputDimension = 3, steps = if (suite == "qnn-reproducibility") 2 else 1,
                        warmupSteps = 0, learningRate = 0.1f, seed = 20_260_710L, sampleCount = 2,
                        epochs = 0, measuredSteps = 0, correctnessInterval = 1, benchmarkMode = false,
                        progressCallback = ProgressCallback { },
                    )
                } finally {
                    heartbeatRunning.set(false)
                    heartbeat.interrupt()
                    heartbeat.join(5_000L)
                }
                reportPath = state.writeReport(runId, report)
                val success = Regex("(?m)^status=SUCCESS$").containsMatchIn(report) ||
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
        "qnn-adam-diagnostic" -> ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP
        else -> throw IllegalArgumentException("Unknown headless suite: $suite")
    }
}
