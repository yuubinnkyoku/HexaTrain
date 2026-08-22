#pragma once

#include "qnn_backend_info.h"

#include <cstdint>
#include <string>
#include <vector>

namespace phonelm::qnn {

enum class QnnBackendKind {
    CPU,
    HTP,
};

// Selects the terminal node for diagnostic language-model training graphs.
// FULL preserves the normal forward/backward/SGD graph.  FORWARD_ONLY is the
// generation-only generalized graph: forward nodes and logits APP_READ only,
// no target/loss/backward tensors or nodes.  It is rejected by the fixed L1
// builder and only served by the generalized N-layer/H-head builder.
enum class TinyTransformerTrainingVariant {
    FULL,
    STOP_AFTER_DINPUT,
    STOP_AFTER_DEMBEDDING,
    FORWARD_ONLY,
};

// Selects graph-preserving APP_READ taps for language-model backward
// diagnostics. NONE preserves the established graph/output contract.
enum class TinyTransformerTrainingTapSet {
    NONE,
    DSCORES_ONLY,
    DPROB_DSCORES,
    BACKWARD_REGIONS,
    LAYERNORM1,
    // Generalized graph scopes add selected APP_READ observability outputs;
    // NONE retains the established graph and output ABI exactly.
    COARSE_LAYER_BOUNDARIES,
    LAYER_SUBBLOCKS,
    LAYER_OPS,
    // Final-layer LayerNorm2 scopes intentionally expose at most two adjacent
    // values.  Keeping these graphs small avoids changing graph finalization
    // merely by making every LayerNorm intermediate an APP_READ tensor.
    LN2_CENTER_SCALE,
    LN2_SQUARE,
    LN2_REDUCTION,
    LN2_INVERSE,
    LN2_NORMALIZED,
    LN2_AFFINE,
    // Exposes forward/backward intermediates in producer order. Existing
    // regular APP_READ outputs are not rebound as taps.
    ALL_INTERNAL,
    NICOPEDIA_FINE,
};

const char* backendKindName(QnnBackendKind kind);

struct RuntimeMetrics {
    std::uint64_t graphCreateCount = 0;
    std::uint64_t graphFinalizeCount = 0;
    std::uint64_t graphExecuteCount = 0;
    std::uint64_t runtimeWeightUpdateCount = 0;
    double backendCreateUs = 0.0;
    double deviceCreateUs = 0.0;
    double contextCreateUs = 0.0;
    double graphCreateUs = 0.0;
    double graphFinalizeUs = 0.0;
    std::vector<double> executeUs;
    std::vector<double> weightUpdateUs;
    std::vector<double> weightBufferCopyUs;
    std::vector<double> inputBindUs;
    std::vector<double> outputBindUs;
    std::uint64_t dWeightGraphCreateCount = 0;
    std::uint64_t dWeightGraphFinalizeCount = 0;
    std::uint64_t dWeightGraphExecuteCount = 0;
    std::uint64_t xInputUpdateCount = 0;
    std::uint64_t dPredictionInputUpdateCount = 0;
    double dWeightGraphCreateUs = 0.0;
    double dWeightGraphFinalizeUs = 0.0;
    std::vector<double> dWeightExecuteUs;
    std::vector<double> dWeightXBindUs;
    std::vector<double> dPredictionBindUs;
    std::vector<double> dWeightOutputBindUs;};

struct ApiTrace {
    std::string backendRequested = "UNAVAILABLE";
    std::string backendLibrary = "UNAVAILABLE";
    int backendLibraryLoadResult = -1;
    bool providerSymbolResolved = false;
    std::uint32_t providerCount = 0;
    int selectedProviderIndex = -1;
    std::string selectedCoreApiVersion = "UNAVAILABLE";
    std::string selectedBackendApiVersion = "UNAVAILABLE";
    std::string runtimeBackendBuildId = "UNAVAILABLE";
    std::string backendCreateSymbolLibrary = "UNAVAILABLE";
    std::string deviceCreateSymbolLibrary = "UNAVAILABLE";
    std::string contextCreateSymbolLibrary = "UNAVAILABLE";
    std::string graphCreateSymbolLibrary = "UNAVAILABLE";
    std::string graphFinalizeSymbolLibrary = "UNAVAILABLE";
    std::string graphExecuteSymbolLibrary = "UNAVAILABLE";
    bool backendCreateCalled = false;
    bool deviceCreateCalled = false;
    bool contextCreateCalled = false;
    bool fullStepGraphCreateCalled = false;
    bool fullStepGraphFinalizeCalled = false;
    int backendCreateResult = -1;
    int deviceCreateResult = -1;
    int contextCreateResult = -1;
    int fullStepGraphCreateResult = -1;
    int fullStepGraphFinalizeResult = -1;
    bool backendHandleNonnull = false;
    bool deviceHandleNonnull = false;
    bool contextHandleNonnull = false;
    bool fullStepGraphHandleNonnull = false;
    std::uint64_t graphExecuteAttemptCount = 0;
    std::uint64_t graphExecuteSuccessCount = 0;
    std::uint64_t graphExecuteFailureCount = 0;
    std::int64_t graphExecuteFirstFailureCall = -1;
    std::int64_t graphExecuteFirstCallIndex = -1;
    std::int64_t graphExecuteLastCallIndex = -1;
    int graphExecuteFirstResult = -1;
    int graphExecuteLastResult = -1;
    bool failureInjectionEnabled = false;
    std::string failureInjectionPoint = "NONE";
    std::int64_t failureInjectionCall = -1;
    int lastQnnResult = -1;
    int effectiveResult = -1;
    bool cpuBackendInitialized = false;
    bool fallbackAttempted = false;
    bool fallbackSucceeded = false;
};

struct RuntimeOptions {
    bool captureQnnCallback = true;
    int qnnLogLevel = 4;
    std::int64_t failGraphExecuteAt = -1;
    bool failGraphFinalize = false;
    // s=1 avoids application-created range amplification in
    // square(s*(x-mean)). The LayerNorm transform remains
    // rsqrt(s^2*(variance+epsilon))*s = rsqrt(variance+epsilon).
    float tinyTransformerCenteredScale = 1.0f;
    std::uint32_t diagnosticLayerIndex = 0xffffffffu;
    // HTP graph precision control (QAIRT 2.48.40.260702 QnnHtpGraph
    // QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION). 0 = unset (backend default,
    // preserves established behavior), 1 = QNN_PRECISION_FLOAT16,
    // 2 = QNN_PRECISION_FLOAT32.  Numeric-path-only switch: no graph math,
    // checkpoint, or gate change.
    std::uint32_t htpGraphPrecisionMode = 0;
    // QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION (bool). 0 = unset,
    // 1 = false, 2 = true.
    std::uint32_t htpGraphPrecisionCompensation = 0;
    // QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING (bool). 0 = unset,
    // 1 = false, 2 = true.  Weight packing can round weights to a
    // lower-precision storage; disabling it is a numeric-path probe.
    std::uint32_t htpGraphWeightsPacking = 0;
    // QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION (bool).
    // 0 = unset, 1 = false, 2 = true.  Fusion can change intermediate
    // rounding; disabling it is a numeric-path probe.
    std::uint32_t htpGraphAdvancedActivationFusion = 0;
    // QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED (bool).
    // 0 = unset (passes nullptr to QnnContext_create exactly as the
    // established path), 1 = false, 2 = true. Context-only private
    // diagnostic: no graph math, checkpoint, or gate change.
    std::uint32_t htpContextGraphSplitting = 0;
    // Diagnostic: declare NATIVE (internal) float tensors as FP16 instead of
    // FP32.  APP_READ/APP_WRITE/STATIC tensors stay FP32 because the host
    // buffers are FP32.  Used to probe whether the HTP backend already
    // executes FP32-declared graphs in FP16.  Default false = established
    // behavior.  Private diagnostics only; no gate/math/checkpoint change.
    bool htpNativeTensorFp16 = false;
};

struct LayerNormBackwardOutputs {
    std::vector<float> output;
    std::vector<float> normalized;
    std::vector<float> dInput;
    std::vector<float> dGamma;
    std::vector<float> dBeta;
};

struct AttentionBackwardOutputs {
    std::vector<float> probabilities;
    std::vector<float> dScores;
    std::vector<float> dQuery;
    std::vector<float> dKey;
    std::vector<float> dValue;
};

// A layer owns every trainable tensor that is local to one Transformer block.
// The first layer remains flattened in TinyTransformerParameters for source and
// ABI compatibility with the established single-layer QNN graph.
struct TinyTransformerLayerParameters {
    std::vector<float> gamma1, beta1, wq, wk, wv, wo;
    std::vector<float> gamma2, beta2, w1, w2;
};

struct TinyTransformerParameters : TinyTransformerLayerParameters {
    // Compatibility storage for layer 0.  Additional layers are independent
    // allocations in layers (where layers[0] is logical layer 1).
    std::vector<float> tokenEmbedding, outputProjection;
    std::vector<TinyTransformerLayerParameters> layers;
};

struct TinyTransformerTrainingTapOutput {
    std::string name;
    std::vector<float> values;
};

struct TinyTransformerTrainingOutputs {
    float loss = 0.0f;
    std::vector<float> output;
    std::vector<float> dOutput;
    std::vector<float> embeddedInput, logits, probabilities, dLogits;
    std::vector<float> dEmbeddedInput;
    // Ordered layer inputs: element 0 is the gradient after positional input
    // construction and is also exposed through dEmbeddedInput for compatibility.
    std::vector<std::vector<float>> layerInputGradients;
    TinyTransformerParameters gradients;
    TinyTransformerParameters next;
    float tapPoison = 0.0f;
    std::vector<TinyTransformerTrainingTapOutput> taps;
};
struct CrossEntropyGradientOutputs {
    std::vector<float> probabilities;
    std::vector<float> dLogits;
};
struct MomentumOptimizerOutputs {
    std::vector<float> velocityNext, weightNext;
};
struct AdamOptimizerOutputs {
    std::vector<float> firstMomentNext, secondMomentNext;
    std::vector<float> firstMomentHat, secondMomentHat, secondRoot;
    std::vector<float> denominator, dividedUpdate, normalizedUpdate;
    std::vector<float> scaledUpdate, weightNext;
};
struct MlpFullStepOutputs {
    float loss = 0.0f;
    std::vector<float> w1Next;
    std::vector<float> w2Next;
    std::vector<float> prediction;
    std::vector<float> hidden;
    std::vector<float> error;
    std::vector<float> dPrediction;
    std::vector<float> dW2;
    std::vector<float> dHidden;
    std::vector<std::uint8_t> mask;
    std::vector<float> dZ1;
    std::vector<float> dW1;
};
class Runtime {
public:
    Runtime();
    ~Runtime();
    const BackendInfo& info() const;
    const std::string& diagnostics() const;
    const ApiTrace& apiTrace() const;
    std::uint32_t tinyTransformerTrainingSourceTensorCreateSuccessCount() const;
    std::uint32_t tinyTransformerTrainingSourceGraphAddNodeSuccessCount() const;
    std::uint32_t tinyTransformerTrainingLastInputTensorCount() const;
    std::uint32_t tinyTransformerTrainingLastOutputTensorCount() const;
    bool tinyTransformerTrainingLastLearningRateBytesUnchanged() const;
    void setOptions(const RuntimeOptions& options);
    std::string apiTraceSummary() const;
    std::string qnnCallbackCaptureSummary() const;
    void recordGraphExecuteResult(int qnnResult, int effectiveResult, bool success);
    bool initialize(QnnBackendKind requestedBackend, std::string& error);
    bool recreateContext(std::string& error);
    bool prepareMatMul(uint32_t m, uint32_t k, uint32_t n, bool transposeInput0,
                       std::string& error);
    bool executeMatMul(const std::vector<float>& a, const std::vector<float>& b,
                       std::vector<float>& output, std::string& error);
    bool setInitialWeight(const std::vector<float>& weight, std::string& error);
    bool updateWeight(const std::vector<float>& weight, std::string& error);
    bool executePrepared(const std::vector<float>& input, std::vector<float>& output,
                         std::string& error);
    bool prepareDWeightMatMul(uint32_t batchSize, uint32_t inputDimension,
                              uint32_t outputDimension, std::string& error);
    bool executeDWeight(const std::vector<float>& input,
                        const std::vector<float>& dPrediction,
                        std::vector<float>& dWeight, std::string& error);
    bool prepareInputGradientMatMul(uint32_t batchSize, uint32_t inputDimension,
                                    uint32_t outputDimension, std::string& error);
    bool executeInputGradient(const std::vector<float>& dPrediction,
                              const std::vector<float>& weight,
                              std::vector<float>& dInput, std::string& error);
    bool prepareMlp(uint32_t batchSize, uint32_t inputDimension,
                    uint32_t hiddenDimension, uint32_t outputDimension,
                    std::string& error, bool prepareSplitBackward = true);
    bool prepareReluBackward(uint32_t batchSize, uint32_t hiddenDimension,
                             std::string& error);
    bool executeReluBackward(const std::vector<float>& activation,
                             const std::vector<float>& dHidden,
                             std::vector<std::uint8_t>& mask,
                             std::vector<float>& dZ1, std::string& error);
    bool prepareMlpFusedBackward(bool diagnosticOutputs, std::string& error);
    bool executeMlpFusedBackward(const std::vector<float>& input,
                                 const std::vector<float>& hidden,
                                 const std::vector<float>& dPrediction,
                                 std::vector<float>& dW2,
                                 std::vector<float>& dHidden,
                                 std::vector<std::uint8_t>& mask,
                                 std::vector<float>& dZ1,
                                 std::vector<float>& dW1, std::string& error);
    bool setMlpWeights(const std::vector<float>& w1, const std::vector<float>& w2,
                       std::string& error);
    bool executeMlpForward(const std::vector<float>& input, std::vector<float>& hidden,
                           std::vector<float>& prediction, std::string& error);
    bool executeMlpSecondBackward(const std::vector<float>& hidden,
                                  const std::vector<float>& dPrediction,
                                  std::vector<float>& dW2, std::vector<float>& dHidden,
                                  std::string& error);
    bool executeMlpFirstBackward(const std::vector<float>& input,
                                 const std::vector<float>& dZ1,
                                 std::vector<float>& dW1, std::string& error);
    bool prepareTrainingOpsMicro(uint32_t batchSize, uint32_t outputDimension,
                                 uint32_t weightRows, uint32_t weightColumns,
                                 std::string& error);
    bool executeTrainingOpsMicro(const std::vector<float>& prediction,
                                 const std::vector<float>& target,
                                 const std::vector<float>& weight,
                                 const std::vector<float>& dWeight,
                                 float learningRate, float& loss,
                                 std::vector<float>& dPrediction,
                                 std::vector<float>& weightNext,
                                 std::string& error);
    bool prepareLayerNorm(uint32_t batch, uint32_t tokens, uint32_t dimension,
                          float epsilon, std::string& error);
    bool executeLayerNorm(const std::vector<float>& input,
                          std::vector<float>& output, std::string& error);
    // Diagnostic single-node graph used to minimize a proven operation-path
    // boundary without retaining checkpoint data.
    bool prepareElementwiseSquare(uint32_t elements, std::string& error);
    bool executeElementwiseSquare(const std::vector<float>& input,
                                  std::vector<float>& output,
                                  std::string& error);
    bool prepareLayerNormBackward(uint32_t rows, uint32_t dimension,
                                  float epsilon, std::string& error);
    bool executeLayerNormBackward(const std::vector<float>& input,
                                  const std::vector<float>& upstream,
                                  const std::vector<float>& gamma,
                                  const std::vector<float>& beta,
                                  LayerNormBackwardOutputs& outputs,
                                  std::string& error);
    bool prepareSoftmax(uint32_t rows, uint32_t columns, std::string& error);
    bool executeSoftmax(const std::vector<float>& input,
                        std::vector<float>& output, std::string& error);
    bool prepareSoftmaxBackward(uint32_t rows, uint32_t columns,
                                std::string& error);
    bool executeSoftmaxBackward(const std::vector<float>& probabilities,
                                const std::vector<float>& upstream,
                                std::vector<float>& inputGradient,
                                std::string& error);
    bool prepareMomentumOptimizer(uint32_t elements, std::string& error);
    bool executeMomentumOptimizer(const std::vector<float>& current,
                                  const std::vector<float>& gradient,
                                  const std::vector<float>& velocity,
                                  float learningRate, float momentum,
                                  MomentumOptimizerOutputs& outputs,
                                  std::string& error);
    bool prepareAdamOptimizer(uint32_t elements, std::string& error);
    bool executeAdamOptimizer(const std::vector<float>& current,
                              const std::vector<float>& gradient,
                              const std::vector<float>& firstMoment,
                              const std::vector<float>& secondMoment,
                              float learningRate, float gradientScale,
                              float firstCorrection,
                              float secondCorrection,
                              AdamOptimizerOutputs& outputs,
                              std::string& error);
    bool prepareCrossEntropyGradient(uint32_t rows, uint32_t columns,
                                     std::string& error);
    bool executeCrossEntropyGradient(const std::vector<float>& logits,
                                     const std::vector<float>& targetOneHot,
                                     CrossEntropyGradientOutputs& outputs,
                                     std::string& error);
    bool prepareAttention(uint32_t tokens, uint32_t headDimension, std::string& error);
    bool executeAttention(const std::vector<float>& query,
                          const std::vector<float>& key,
                          const std::vector<float>& value,
                          const std::vector<float>& causalMask,
                          std::vector<float>& output,
                          std::vector<float>& probabilities,
                          std::string& error);
    bool prepareAttentionBackward(uint32_t tokens, uint32_t headDimension,
                                  std::string& error);
    bool executeAttentionBackward(const std::vector<float>& query,
                                  const std::vector<float>& key,
                                  const std::vector<float>& value,
                                  const std::vector<float>& upstream,
                                  const std::vector<float>& causalMask,
                                  AttentionBackwardOutputs& outputs,
                                  std::string& error);
    bool prepareTinyTransformer(uint32_t tokens, uint32_t dimension,
                                uint32_t feedForwardDimension, float epsilon,
                                std::string& error);
    bool executeTinyTransformer(const std::vector<float>& input,
                                std::vector<float>& output, std::string& error);
    bool prepareTinyTransformerTraining(uint32_t tokens, uint32_t dimension,
                                        uint32_t feedForwardDimension,
                                        float epsilon, bool diagnosticOutputs,
                                        std::string& error,
                                        uint32_t vocabularySize = 0,
                                        TinyTransformerTrainingVariant variant =
                                            TinyTransformerTrainingVariant::FULL,
                                        TinyTransformerTrainingTapSet tapSet =
                                            TinyTransformerTrainingTapSet::NONE,
                                        uint32_t numLayers = 1,
                                        uint32_t numHeads = 1);
    bool executeTinyTransformerTraining(
        const std::vector<float>& input, const std::vector<float>& target,
        const TinyTransformerParameters& current, float learningRate,
        TinyTransformerTrainingOutputs& outputs, std::string& error);
    // Forward-only generation execute: binds the token one-hot and every
    // parameter APP_WRITE tensor, reads back logits only.  The target input
    // and every backward/gradient output of the FULL graph do not exist in
    // this graph; learningRate is rejected (must be 0) exactly like the
    // training execute.
    bool executeTinyTransformerForwardOnly(
        const std::vector<float>& input,
        const TinyTransformerParameters& current,
        TinyTransformerTrainingOutputs& outputs, std::string& error);
    bool prepareMlpFullStep(uint32_t batchSize, uint32_t inputDimension,
                            uint32_t hiddenDimension, uint32_t outputDimension,
                            bool diagnosticOutputs, std::string& error);
    bool executeMlpFullStep(const std::vector<float>& input,
                            const std::vector<float>& target,
                            const std::vector<float>& w1Current,
                            const std::vector<float>& w2Current,
                            float learningRate, MlpFullStepOutputs& outputs,
                            std::string& error);
    const RuntimeMetrics& metrics() const;

private:
    bool prepareTinyTransformerTrainingGeneralized(
        uint32_t tokens, uint32_t dimension, uint32_t feedForwardDimension,
        float epsilon, bool diagnosticOutputs, std::string& error,
        uint32_t vocabularySize, TinyTransformerTrainingVariant variant,
        TinyTransformerTrainingTapSet tapSet, uint32_t numLayers,
        uint32_t numHeads);
    bool executeTinyTransformerTrainingGeneralized(
        const std::vector<float>& input, const std::vector<float>& target,
        const TinyTransformerParameters& current, float learningRate,
        TinyTransformerTrainingOutputs& outputs, std::string& error);
    BackendInfo info_;
    std::string diagnostics_;
    RuntimeMetrics metrics_;
    ApiTrace apiTrace_;
    RuntimeOptions options_;
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace phonelm::qnn
