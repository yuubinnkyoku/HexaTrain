package com.yuubinnkyoku.phonelm

enum class Backend(val nativeCode: Int) {
    CPU(0),
    OPENCL(1),
    VULKAN(2),
}

enum class ExecutionMode(val nativeCode: Int) {
    CPU_REFERENCE(0),
    MNN_CPU(1),
    MNN_OPENCL(2),
    MNN_VULKAN(3),
    QNN_CPU_FORWARD(4),
    QNN_HTP_FORWARD(5),
    QNN_HTP_FORWARD_CPU_BACKWARD(6),
    QNN_HTP_FORWARD_DW(7),
    QNN_HTP_FORWARD_DW_DX(8),
    QNN_HTP_FULL_STEP(9),
    QNN_HTP_DEVICE_PROBE(10),
    QNN_CPU_LINEAR_TRAINING(11),
    QNN_HTP_LINEAR_TRAINING(12),
    QNN_LINEAR_GRADIENT_CHECK(13),
    QNN_CPU_MULTIBATCH_TRAINING(14),
    QNN_HTP_MULTIBATCH_TRAINING(15),
    QNN_CPU_TRAINING_BENCHMARK(16),
    QNN_HTP_TRAINING_BENCHMARK(17),
    QNN_HTP_DW_CHECK(18),
    QNN_HTP_FORWARD_HTP_DW_TRAINING(19),
    QNN_HTP_FORWARD_HTP_DW_BENCHMARK(20),
    QNN_HTP_DX_CHECK(21),
    QNN_CPU_MLP_TRAINING(22),
    QNN_HTP_MLP_CPU_BACKWARD(23),
    QNN_HTP_MLP_HTP_LINEAR_BACKWARD(24),
    QNN_HTP_MLP_BENCHMARK(25),
    QNN_MLP_GRADIENT_CHECK(26),
    QNN_HTP_RELU_BACKWARD_CHECK(27),
    QNN_HTP_MLP_FUSED_BACKWARD(28),
    QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK(29),
    QNN_HTP_MSE_CHECK(30),
    QNN_HTP_SGD_CHECK(31),
    QNN_HTP_MLP_FULL_STEP(32),
    QNN_HTP_MLP_FULL_STEP_BENCHMARK(33),
    QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE(34),
    QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE(35),
    QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE(36),
    QNN_HTP_LAYER_NORM_CHECK(37),
    QNN_HTP_SOFTMAX_CHECK(38),
    QNN_HTP_ATTENTION_FORWARD_CHECK(39),
    QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK(40),
    QNN_HTP_SOFTMAX_BACKWARD_CHECK(41),
    QNN_HTP_ATTENTION_BACKWARD_CHECK(42),
    QNN_HTP_LAYER_NORM_BACKWARD_CHECK(43),
    QNN_HTP_TINY_TRANSFORMER_TRAINING_STEP(44),
    QNN_HTP_TINY_TRANSFORMER_TRAINING_MULTI_STEP(45),
    QNN_HTP_CROSS_ENTROPY_CHECK(46),
    QNN_HTP_TINY_LANGUAGE_MODEL_STEP(47),
    QNN_HTP_TINY_LANGUAGE_MODEL_MULTI_STEP(48),
    QNN_HTP_TINY_LANGUAGE_MODEL_INFERENCE(49),
    QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_1(50),
    QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_2(51),
    QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_3(52),
    QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_STEP(53),
    QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_1(54),
    QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_2(55),
    QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_INFERENCE(56),
    QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP(57),
    QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_1(58),
    QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_2(59),
    QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_INFERENCE(60),
    ;

    companion object {
        fun fromBackend(backend: Backend) = when (backend) {
            Backend.CPU -> MNN_CPU
            Backend.OPENCL -> MNN_OPENCL
            Backend.VULKAN -> MNN_VULKAN
        }
    }
}

data class BenchmarkConfig(
    val backend: Backend,
    val batchSize: Int,
    val dimension: Int,
    val steps: Int,
    val warmupSteps: Int,
    val learningRate: Float = 0.1f,
    val seed: Long = 20_260_710L,
    val sampleCount: Int = 512,
    val epochs: Int = 0,
    val measuredSteps: Int = 0,
    val correctnessInterval: Int = 0,
    val benchmarkMode: Boolean = false,
    val hiddenDimension: Int = dimension,
    val outputDimension: Int = maxOf(1, dimension / 2),
) {
    fun validationError(): String? {
        if (batchSize !in 1..4096) return "batchSize must be in 1..4096"
        if (dimension !in 1..4096) return "dimension must be in 1..4096"
        if (hiddenDimension !in 1..4096) return "hiddenDimension must be in 1..4096"
        if (outputDimension !in 1..4096) return "outputDimension must be in 1..4096"
        if (steps !in 1..100_000) return "steps must be in 1..100000"
        if (warmupSteps !in 0..10_000) return "warmupSteps must be in 0..10000"
        if (!learningRate.isFinite() || learningRate <= 0f || learningRate > 10f) {
            return "learningRate must be finite and in (0, 10]"
        }
        if (sampleCount !in 1..1_000_000) return "sampleCount must be in 1..1000000"
        if (epochs !in 0..100_000) return "epochs must be in 0..100000"
        if (measuredSteps !in 0..100_000) return "measuredSteps must be in 0..100000"
        if (correctnessInterval !in 0..100_000) return "correctnessInterval must be in 0..100000"

        val matrixElements = dimension.toLong() * dimension.toLong()
        val batchElements = batchSize.toLong() * dimension.toLong()
        val estimatedBytes = (matrixElements * 3L + batchElements * 6L) * 12L * Float.SIZE_BYTES
        if (estimatedBytes > MAX_ESTIMATED_BYTES) {
            return "estimated working set exceeds 1536 MiB safety limit"
        }
        return null
    }

    companion object {
        private const val MAX_ESTIMATED_BYTES = 1536L * 1024L * 1024L

        fun small(backend: Backend = Backend.CPU) = BenchmarkConfig(
            backend = backend,
            batchSize = 8,
            dimension = 128,
            steps = 100,
            warmupSteps = 0,
        )

        fun benchmark(backend: Backend = Backend.CPU) = BenchmarkConfig(
            backend = backend,
            batchSize = 32,
            dimension = 512,
            steps = 200,
            warmupSteps = 20,
        )
    }
}
