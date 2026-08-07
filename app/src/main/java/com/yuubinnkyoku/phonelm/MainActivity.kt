package com.yuubinnkyoku.phonelm

import android.app.Activity
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.content.pm.ApplicationInfo
import android.util.Log
import java.io.File
import android.widget.Button
import android.widget.EditText
import android.widget.ArrayAdapter
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import java.util.Locale

class MainActivity : Activity() {
    private lateinit var viewModel: BenchmarkViewModel
    private lateinit var batchSizeInput: EditText
    private lateinit var dimensionInput: EditText
    private lateinit var stepsInput: EditText
    private lateinit var warmupStepsInput: EditText
    private lateinit var learningRateInput: EditText
    private lateinit var cpuButton: Button
    private lateinit var openClButton: Button
    private lateinit var vulkanButton: Button
    private lateinit var stopButton: Button
    private lateinit var resultText: TextView
    private lateinit var resultScrollView: ScrollView
    private lateinit var qnnStatusText: TextView
    private lateinit var configurationSummaryText: TextView
    private lateinit var executionModeSpinner: Spinner
    private lateinit var runSelectedModeButton: Button
    private lateinit var qnnForwardButton: Button
    private lateinit var qnnForwardDwButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        QnnEnvironment.prepare(this)

        batchSizeInput = findViewById(R.id.batchSizeInput)
        dimensionInput = findViewById(R.id.dimensionInput)
        stepsInput = findViewById(R.id.stepsInput)
        warmupStepsInput = findViewById(R.id.warmupStepsInput)
        learningRateInput = findViewById(R.id.learningRateInput)
        cpuButton = findViewById(R.id.cpuButton)
        openClButton = findViewById(R.id.openClButton)
        vulkanButton = findViewById(R.id.vulkanButton)
        stopButton = findViewById(R.id.stopButton)
        resultText = findViewById(R.id.resultText)
        resultScrollView = findViewById(R.id.resultScrollView)
        qnnStatusText = findViewById(R.id.qnnStatusText)
        configurationSummaryText = findViewById(R.id.configurationSummaryText)
        executionModeSpinner = findViewById(R.id.executionModeSpinner)
        runSelectedModeButton = findViewById(R.id.runSelectedModeButton)
        qnnForwardButton = findViewById(R.id.qnnForwardButton)
        qnnForwardDwButton = findViewById(R.id.qnnForwardDwButton)

        executionModeSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            ExecutionMode.values().map(ExecutionMode::name),
        )

        viewModel = BenchmarkViewModel(runNotifications = LiveUpdateNotificationController(applicationContext))
        viewModel.setListener(::render)

        findViewById<Button>(R.id.smallPresetButton).setOnClickListener {
            applyPreset(BenchmarkConfig.small())
        }
        findViewById<Button>(R.id.benchmarkPresetButton).setOnClickListener {
            applyPreset(BenchmarkConfig.benchmark())
        }
        findViewById<Button>(R.id.qnnPresetButton).setOnClickListener {
            applyPreset(
                BenchmarkConfig(
                    backend = Backend.CPU,
                    batchSize = 2,
                    dimension = 4,
                    steps = 20,
                    warmupSteps = 0,
                ),
            )
        }
        cpuButton.setOnClickListener { start(Backend.CPU) }
        openClButton.setOnClickListener { start(Backend.OPENCL) }
        vulkanButton.setOnClickListener { start(Backend.VULKAN) }
        stopButton.setOnClickListener {
            if (!viewModel.requestStop()) toast("No benchmark is running")
        }
        runSelectedModeButton.setOnClickListener {
            startMode(ExecutionMode.values()[executionModeSpinner.selectedItemPosition])
        }
        qnnForwardButton.setOnClickListener { startMode(ExecutionMode.QNN_HTP_FORWARD) }
        qnnForwardDwButton.setOnClickListener { startMode(ExecutionMode.QNN_HTP_FORWARD_DW) }

        applyPreset(BenchmarkConfig.small())
        runDebugIntentIfRequested()
        viewModel.loadEnvironment()
    }

    private fun runDebugIntentIfRequested() {
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val requested = intent.getStringExtra("phonelm.mode") ?: return
        if (requested == "QNN_HTP_FIRST_NONFINITE_REPLAY") {
            val repeatCount = intent.getIntExtra("phonelm.replay_count", 3)
            val tapSet = intent.getIntExtra("phonelm.tap_set", 0)
            val checkpoint = File(filesDir, "first-nonfinite-checkpoint.bin")
            Log.i("PhoneLMDeviceTest", "DEVICE_TEST_START mode=$requested")
            Thread({
                val report = if (checkpoint.isFile) {
                    NativeBridge.nativeReplayFirstNonfiniteCheckpoint(
                        payload = checkpoint.readBytes(),
                        repeatCount = repeatCount,
                        tapSet = tapSet,
                    )
                } else {
                    "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n" +
                        "failure_classification=CHECKPOINT_UNAVAILABLE\n" +
                        "error=app_private_checkpoint_file_missing\n"
                }
                openFileOutput("device-test-result.txt", MODE_PRIVATE).bufferedWriter().use {
                    it.write(report)
                }
                Log.i("PhoneLMDeviceTest", "DEVICE_TEST_COMPLETE mode=$requested")
            }, "PhoneLM-first-nonfinite-replay").start()
            return
        }
        val mode = runCatching { ExecutionMode.valueOf(requested) }.getOrNull() ?: return
        val batchSize = intent.getIntExtra("phonelm.batch_size", 2)
        val dimension = intent.getIntExtra("phonelm.dimension", 4)
        val hiddenDimension = intent.getIntExtra("phonelm.hidden_dimension", dimension)
        val outputDimension = intent.getIntExtra("phonelm.output_dimension", maxOf(1, dimension / 2))
        val steps = intent.getIntExtra("phonelm.steps", 20)
        val warmupSteps = intent.getIntExtra("phonelm.warmup_steps", 0)
        val learningRate = intent.getStringExtra("phonelm.learning_rate")?.toFloatOrNull() ?: 0.1f
        val seed = intent.getStringExtra("phonelm.seed")?.toLongOrNull() ?: 20_260_710L
        val sampleCount = intent.getIntExtra("phonelm.sample_count", 512)
        val epochs = intent.getIntExtra("phonelm.epochs", 0)
        val measuredSteps = intent.getIntExtra("phonelm.measured_steps", 0)
        val correctnessInterval = intent.getIntExtra("phonelm.correctness_interval", 0)
        val benchmarkMode = intent.getBooleanExtra("phonelm.benchmark_mode", false)
        val seedSelectionName = intent.getStringExtra("phonelm.seed_selection_mode")
        val seedSelectionMode = seedSelectionName?.let {
            runCatching { SeedSelectionMode.valueOf(it) }.getOrNull()
        } ?: SeedSelectionMode.COUNT_FROM_ONE
        val stabilityName = intent.getStringExtra("phonelm.training_stability_mode")
        val trainingStabilityMode = stabilityName?.let {
            runCatching { TrainingStabilityMode.valueOf(it) }.getOrNull()
        } ?: TrainingStabilityMode.LEGACY
        val pairInitName = intent.getStringExtra("phonelm.depth_pair_init_mode")
        val depthPairInitMode = pairInitName?.let {
            runCatching { DepthPairInitMode.valueOf(it) }.getOrNull()
        } ?: DepthPairInitMode.LEGACY
        val checkpointSelectionName = intent.getStringExtra("phonelm.checkpoint_selection_mode")
        val checkpointSelectionMode = checkpointSelectionName?.let {
            runCatching { CheckpointSelectionMode.valueOf(it) }.getOrNull()
        } ?: CheckpointSelectionMode.FINAL_STEP
        val diagnosticTrajectory = intent.getBooleanExtra("phonelm.diagnostic_trajectory", false)
        // The NICOPEDIA mode reads the private tokenized pilot cache pushed by
        // the host runner into files/nicopedia-cache/.  The path is resolved
        // through the same fail-closed app-files check as checkpoints.
        val requestedDir = when (mode) {
            ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA ->
                intent.getStringExtra("phonelm.checkpoint_dump_dir") ?: "nicopedia-cache"
            else -> intent.getStringExtra("phonelm.checkpoint_dump_dir")
        }
        val checkpointDir = requestedDir?.let { dir ->
            // Fail closed: private checkpoints never leave the app files dir.
            filesDir.toPath().resolve("checkpoints").resolve(dir)
                .normalize().takeIf { it.startsWith(filesDir.toPath()) }?.toFile()
        }
        if (checkpointDir != null) checkpointDir.mkdirs()
        val config = BenchmarkConfig(backend = Backend.CPU, batchSize = batchSize, dimension = dimension,
            steps = steps, warmupSteps = warmupSteps, learningRate = learningRate, seed = seed,
            sampleCount = sampleCount, epochs = epochs, measuredSteps = measuredSteps,
            correctnessInterval = correctnessInterval, benchmarkMode = benchmarkMode,
            seedSelectionMode = seedSelectionMode,
            trainingStabilityMode = trainingStabilityMode,
            depthPairInitMode = depthPairInitMode,
            checkpointSelectionMode = checkpointSelectionMode,
            diagnosticTrajectory = diagnosticTrajectory,
            diagnosticCheckpointDir = checkpointDir?.absolutePath,
            hiddenDimension = hiddenDimension, outputDimension = outputDimension)
        applyPreset(config)
        Log.i("PhoneLMDeviceTest", "DEVICE_TEST_START mode=$requested")
        val validationError = when {
            seedSelectionName != null && seedSelectionMode.name != seedSelectionName ->
                "unknown phonelm.seed_selection_mode: $seedSelectionName"
            stabilityName != null && trainingStabilityMode.name != stabilityName ->
                "unknown phonelm.training_stability_mode: $stabilityName"
            pairInitName != null && depthPairInitMode.name != pairInitName ->
                "unknown phonelm.depth_pair_init_mode: $pairInitName"
            checkpointSelectionName != null &&
                checkpointSelectionMode.name != checkpointSelectionName ->
                "unknown phonelm.checkpoint_selection_mode: $checkpointSelectionName"
            checkpointSelectionMode == CheckpointSelectionMode.BEST_VALIDATION_V1 &&
                mode != ExecutionMode.QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC ->
                "BEST_VALIDATION_V1 is only supported by QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC"
            intent.getStringExtra("phonelm.checkpoint_dump_dir") != null && checkpointDir == null ->
                "phonelm.checkpoint_dump_dir must resolve under the app files dir"
            else -> config.validationError()
        }
        if (validationError != null) {
            Log.e("PhoneLMDeviceTest", "DEVICE_TEST_REJECTED error=$validationError")
            return
        }
        if (intent.getBooleanExtra("phonelm.live_update", false)) {
            Log.i("PhoneLMDeviceTest", "DEVICE_TEST_LIVE_UPDATE mode=$requested")
            if (!viewModel.startMode(mode, config)) {
                Log.e("PhoneLMDeviceTest", "DEVICE_TEST_REJECTED error=run already active")
            }
            return
        }
        Thread({
            val report = NativeBridge.nativeRunExecutionMode(
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
                progressCallback = ProgressCallback { },
            )
            openFileOutput("device-test-result.txt", MODE_PRIVATE).bufferedWriter().use {
                it.write(report)
            }
            Log.i("PhoneLMDeviceTest", "DEVICE_TEST_DONE\n$report")
            runOnUiThread {
                resultText.text = report
            }
        }, "PhoneLM-device-test").start()
    }

    private fun applyPreset(config: BenchmarkConfig) {
        batchSizeInput.setText(String.format(Locale.ROOT, "%d", config.batchSize))
        dimensionInput.setText(String.format(Locale.ROOT, "%d", config.dimension))
        stepsInput.setText(String.format(Locale.ROOT, "%d", config.steps))
        warmupStepsInput.setText(String.format(Locale.ROOT, "%d", config.warmupSteps))
        learningRateInput.setText(String.format(Locale.ROOT, "%.3g", config.learningRate))
    }

    private fun start(backend: Backend) {
        requestNotificationPermissionForUserRun()
        val config = readConfig(backend) ?: return
        val error = config.validationError()
        if (error != null) {
            toast(error)
            return
        }
        if (!viewModel.start(config)) {
            toast("A benchmark is already running")
        }
    }

    private fun startMode(mode: ExecutionMode) {
        requestNotificationPermissionForUserRun()
        val backend = when (mode) {
            ExecutionMode.MNN_OPENCL -> Backend.OPENCL
            ExecutionMode.MNN_VULKAN -> Backend.VULKAN
            else -> Backend.CPU
        }
        val config = readConfig(backend) ?: return
        val error = config.validationError()
        if (error != null) {
            toast(error)
            return
        }
        if (!viewModel.startMode(mode, config)) {
            toast("A benchmark is already running")
        }
    }

    private fun readConfig(backend: Backend): BenchmarkConfig? {
        val batchSize = batchSizeInput.text.toString().toIntOrNull()
        val dimension = dimensionInput.text.toString().toIntOrNull()
        val steps = stepsInput.text.toString().toIntOrNull()
        val warmup = warmupStepsInput.text.toString().toIntOrNull()
        val learningRate = learningRateInput.text.toString().toFloatOrNull()
        if (batchSize == null || dimension == null || steps == null || warmup == null ||
            learningRate == null
        ) {
            toast("All configuration fields must contain valid numbers")
            return null
        }
        return BenchmarkConfig(
            backend = backend,
            batchSize = batchSize,
            dimension = dimension,
            steps = steps,
            warmupSteps = warmup,
            learningRate = learningRate,
        )
    }

    private fun render(state: BenchmarkUiState) {
        cpuButton.isEnabled = !state.running
        openClButton.isEnabled = !state.running
        vulkanButton.isEnabled = !state.running
        stopButton.isEnabled = state.running
        executionModeSpinner.isEnabled = !state.running
        runSelectedModeButton.isEnabled = !state.running
        qnnForwardButton.isEnabled = !state.running
        qnnForwardDwButton.isEnabled = !state.running
        qnnStatusText.text = state.qnnStatus
        configurationSummaryText.text = state.configurationSummary
        resultText.text = state.output
        resultScrollView.post { resultScrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        if (!state.running && state.lastResult != null &&
            applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
        ) {
            openFileOutput("device-test-result.txt", MODE_PRIVATE).bufferedWriter().use {
                it.write(state.lastReport ?: state.output)
            }
            Log.i("PhoneLMDeviceTest", "DEVICE_TEST_DONE\n${state.lastReport ?: state.output}")
        }
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun requestNotificationPermissionForUserRun() {
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
        }
    }

    override fun onDestroy() {
        viewModel.setListener(null)
        if (isFinishing) viewModel.close()
        super.onDestroy()
    }

    private companion object { const val REQUEST_NOTIFICATIONS = 91 }
}
