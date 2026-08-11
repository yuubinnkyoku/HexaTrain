package com.yuubinnkyoku.phonelm

fun interface ProgressCallback {
    fun onNativeProgress(message: String)
}

object NativeBridge {
    init {
        // libMNN.so is monolithic: Express, MNN-Train, OpenCL, and Vulkan are
        // deliberately linked together so registration objects cannot be lost.
        System.loadLibrary("MNN")
        System.loadLibrary("phonelm_native")
    }

    external fun nativeGetEnvironmentInfo(): String

    external fun nativeGetQnnStatus(): String

    external fun nativeRunBenchmark(
        backend: Int,
        batchSize: Int,
        dimension: Int,
        steps: Int,
        warmupSteps: Int,
        learningRate: Float,
        seed: Long,
        progressCallback: ProgressCallback,
    ): String

    external fun nativeRequestStop()

    external fun nativeRunExecutionMode(
        executionMode: Int,
        batchSize: Int,
        dimension: Int,
        hiddenDimension: Int,
        outputDimension: Int,
        steps: Int,
        warmupSteps: Int,
        learningRate: Float,
        seed: Long,
        sampleCount: Int,
        epochs: Int,
        measuredSteps: Int,
        correctnessInterval: Int,
        benchmarkMode: Boolean,
        seedSelectionMode: Int,
        trainingStabilityMode: Int,
        depthPairInitMode: Int,
        checkpointSelectionMode: Int,
        diagnosticTrajectory: Boolean,
        diagnosticCheckpointDir: String?,
        diagnosticResumeStep: Int,
        diagnosticCheckpointInterval: Int,
        progressCallback: ProgressCallback,
    ): String

    external fun nativeReplayFirstNonfiniteCheckpoint(
        payload: ByteArray,
        repeatCount: Int,
        tapSet: Int,
    ): String

    external fun nativeRunNicopediaGenerate(
        checkpointPath: String,
        promptPath: String,
        seed: Long,
        layers: Int,
        tokens: Int,
        maxNewBytes: Int,
        generateMode: String,
        temperature: Float,
        topK: Int,
        samplingSeed: Long,
        gatePolicy: String,
        htpGraphPrecisionMode: Int,
        htpGraphPrecisionCompensation: Int,
        htpGraphWeightsPacking: Int,
        htpGraphAdvancedActivationFusion: Int,
        htpContextGraphSplitting: Int,
        htpNativeTensorFp16: Boolean,
    ): String

    external fun nativeRunNicopediaDivergenceLocalization(
        checkpointPath: String,
        seed: Long,
        layers: Int,
        tapScope: String,
        diagnosticLayerIndex: Int,
    ): String

    external fun nativeRunNicopediaEvaluate(
        checkpointDir: String,
        seed: Long,
        layers: Int,
        heads: Int,
        tokens: Int,
        checkpointStep: Int,
        validationChunks: Int,
        developmentChunks: Int,
    ): String
}
