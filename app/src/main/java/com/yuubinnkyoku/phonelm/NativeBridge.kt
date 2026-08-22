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

    /** Clears a stop queued before a JNI run was actually entered. */
    external fun nativeClearStop()

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
        heads: Int,
        tokens: Int,
        vocabulary: Int,
        dimension: Int,
        feedForwardDimension: Int,
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

    /** Thread-safe observation snapshot; the terminal generation report remains authoritative. */
    external fun nativeGetNicopediaGenerationProgress(): String

    /**
     * Prepares (or reuses) the process-local generation engine for the exact
     * checkpoint identity and graph options. Returns a nonzero handle on
     * success, 0 on failure (no partial engine is retained). The handle must
     * be released with [nativeReleaseNicopediaGeneration]; use-after-release
     * and double release fail closed.
     */
    external fun nativePrepareNicopediaGeneration(
        checkpointPath: String,
        checkpointFileBytes: Long,
        checkpointModifiedMs: Long,
        vocabulary: Int,
        tokens: Int,
        dimension: Int,
        feedForwardDimension: Int,
        layers: Int,
        heads: Int,
        seed: Long,
        checkpointStep: Int,
        tokenizerKind: String,
        tokenizerHash: String?,
        parameterHash: String,
        htpGraphPrecisionMode: Int,
        htpGraphPrecisionCompensation: Int,
        htpGraphWeightsPacking: Int,
        htpGraphAdvancedActivationFusion: Int,
        htpContextGraphSplitting: Int,
        htpNativeTensorFp16: Boolean,
        errorOut: StringBuilder?,
    ): Long

    /**
     * Runs one generation on the prepared engine. Returns the private
     * NICOPEDIA_HTP_GENERATION KEY=VALUE report with
     * prepared_graph_reused=true. Any QNN/finite/identity failure poisons
     * the engine.
     */
    external fun nativeRunPreparedNicopediaGeneration(
        handle: Long,
        promptPath: String,
        maxNewBytes: Int,
        generateMode: String,
        temperature: Float,
        topK: Int,
        samplingSeed: Long,
    ): String

    /** Releases the prepared engine. Safe with 0; release exactly once per handle. */
    external fun nativeReleaseNicopediaGeneration(handle: Long)

    /** Applies the shared native lossless UTF-8 display contract and returns valid UTF-8 bytes. */
    external fun nativeSafeUtf8Display(bytes: ByteArray): ByteArray

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
        vocabulary: Int,
        dimension: Int,
        feedForwardDimension: Int,
        checkpointStep: Int,
        validationChunks: Int,
        developmentChunks: Int,
    ): String

    /** Research-only bounded D/FFN graph probe; never used by production UI. */
    external fun nativeRunNicopediaOneUpdateProbe(
        cacheDir: String,
        seed: Long,
        layers: Int,
        heads: Int,
        tokens: Int,
        dimension: Int,
        feedForwardDimension: Int,
        batchSize: Int,
        learningRate: Float,
        progressCallback: ProgressCallback,
    ): String
}
