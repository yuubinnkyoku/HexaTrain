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
};

struct AttentionBackwardOutputs {
    std::vector<float> probabilities;
    std::vector<float> dScores;
    std::vector<float> dQuery;
    std::vector<float> dKey;
    std::vector<float> dValue;
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
    void setOptions(const RuntimeOptions& options);
    std::string apiTraceSummary() const;
    std::string qnnCallbackCaptureSummary() const;
    void recordGraphExecuteResult(int qnnResult, int effectiveResult, bool success);
    bool initialize(QnnBackendKind requestedBackend, std::string& error);
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
    bool prepareSoftmax(uint32_t rows, uint32_t columns, std::string& error);
    bool executeSoftmax(const std::vector<float>& input,
                        std::vector<float>& output, std::string& error);
    bool prepareSoftmaxBackward(uint32_t rows, uint32_t columns,
                                std::string& error);
    bool executeSoftmaxBackward(const std::vector<float>& probabilities,
                                const std::vector<float>& upstream,
                                std::vector<float>& inputGradient,
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
    BackendInfo info_;
    std::string diagnostics_;
    RuntimeMetrics metrics_;
    ApiTrace apiTrace_;
    RuntimeOptions options_;
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace phonelm::qnn
