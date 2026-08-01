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
        diagnosticTrajectory: Boolean,
        diagnosticCheckpointDir: String?,
        progressCallback: ProgressCallback,
    ): String

    external fun nativeReplayFirstNonfiniteCheckpoint(
        payload: ByteArray,
        repeatCount: Int,
        tapSet: Int,
    ): String
}
