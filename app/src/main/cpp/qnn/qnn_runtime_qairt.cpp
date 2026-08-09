#include "qnn_runtime.h"
#include "qnn_graph_shape_validator.h"
#include "../transformer_resource_estimator.h"

#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpGraph.h>
#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <list>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>
#include <unordered_set>

namespace phonelm::qnn {
namespace {
using Clock = std::chrono::steady_clock;
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
double elapsedUs(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
constexpr std::size_t kCallbackMessageLimitBytes = 1024;
constexpr std::size_t kCallbackLimitMessages = 64;
constexpr std::size_t kCallbackLimitBytes = 32 * 1024;
struct CapturedQnnMessage { int level; std::uint64_t timestamp; std::string message; };
struct QnnCallbackCapture {
    std::mutex mutex;
    std::vector<CapturedQnnMessage> messages;
    std::size_t savedBytes = 0;
    std::atomic<std::uint64_t> droppedMessages{0};
    std::atomic_bool enabled{true};
    std::atomic_bool saturated{false};
} gQnnCallbackCapture;
void resetQnnCallbackCapture(bool enabled) {
    std::lock_guard<std::mutex> lock(gQnnCallbackCapture.mutex);
    gQnnCallbackCapture.messages.clear();
    gQnnCallbackCapture.savedBytes = 0;
    gQnnCallbackCapture.droppedMessages.store(0, std::memory_order_relaxed);
    gQnnCallbackCapture.saturated.store(false, std::memory_order_relaxed);
    gQnnCallbackCapture.enabled.store(enabled, std::memory_order_release);
}
void qnnLogCallback(const char* format, QnnLog_Level_t level,
                    std::uint64_t timestamp, va_list arguments) {
    if (!gQnnCallbackCapture.enabled.load(std::memory_order_acquire)) return;
    if (gQnnCallbackCapture.saturated.load(std::memory_order_relaxed)) {
        gQnnCallbackCapture.droppedMessages.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(gQnnCallbackCapture.mutex);
    if (!gQnnCallbackCapture.enabled.load(std::memory_order_relaxed)) return;
    if (gQnnCallbackCapture.messages.size() >= kCallbackLimitMessages ||
        gQnnCallbackCapture.savedBytes >= kCallbackLimitBytes) {
        gQnnCallbackCapture.saturated.store(true, std::memory_order_relaxed);
        gQnnCallbackCapture.droppedMessages.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::array<char, kCallbackMessageLimitBytes + 1> buffer{};
    va_list captureArgs;
    va_copy(captureArgs, arguments);
    const int required = std::vsnprintf(buffer.data(), buffer.size(), format, captureArgs);
    va_end(captureArgs);
    if (required < 0 || static_cast<std::size_t>(required) > kCallbackMessageLimitBytes ||
        gQnnCallbackCapture.savedBytes + static_cast<std::size_t>(required) > kCallbackLimitBytes) {
        gQnnCallbackCapture.droppedMessages.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::string message(buffer.data(), static_cast<std::size_t>(required));
    std::replace(message.begin(), message.end(), '\r', ' ');
    std::replace(message.begin(), message.end(), '\n', ' ');
    gQnnCallbackCapture.savedBytes += message.size();
    gQnnCallbackCapture.messages.push_back({static_cast<int>(level), timestamp, std::move(message)});
}
std::string functionLibraryBasename(const void* function) {
    if (!function) return "UNAVAILABLE";
    Dl_info info{};
    if (dladdr(function, &info) == 0 || !info.dli_fname) return "UNAVAILABLE";
    const char* slash = std::strrchr(info.dli_fname, '/');
    return slash ? slash + 1 : info.dli_fname;
}
const char* boolText(bool value) { return value ? "true" : "false"; }

// Holds the official HTP custom config and its QNN wrapper through each
// QnnContext_create call.  Keeping unset as nullptr preserves the established
// context creation ABI exactly rather than passing an empty config array.
struct HtpContextConfigs {
    QnnHtpContext_CustomConfig_t graphSplitting = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
    std::array<const QnnContext_Config_t*, 2> configPtrs{};
    bool failClosedInvalid = false;
    const char* requested = "unset";

    const QnnContext_Config_t** array() {
        if (configPtrs[0] == nullptr) return nullptr;
        // The builder returns by value. Rebind every self-reference on the
        // final object before passing it to QNN; named-return-value elision is
        // not a lifetime guarantee.
        contextConfig.customConfig = &graphSplitting;
        configPtrs[0] = &contextConfig;
        return configPtrs.data();
    }

    const char* delivery() const {
        return configPtrs[0] == nullptr ? "unset_nullptr"
                                        : "passed_to_qnn_context_create";
    }
};

HtpContextConfigs buildHtpContextConfigs(const RuntimeOptions& options) {
    HtpContextConfigs configs;
    if (options.htpContextGraphSplitting == 0) return configs;
    if (options.htpContextGraphSplitting > 2) {
        configs.failClosedInvalid = true;
        return configs;
    }
    const bool enabled = options.htpContextGraphSplitting == 2;
    configs.graphSplitting.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED;
    configs.graphSplitting.graphSplittingEnabled = enabled;
    configs.contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    configs.contextConfig.customConfig = &configs.graphSplitting;
    configs.configPtrs[0] = &configs.contextConfig;
    configs.configPtrs[1] = nullptr;
    configs.requested = enabled ? "true" : "false";
    return configs;
}
const char* logLevelText(int level) {
    switch (level) {
        case 1: return "ERROR";
        case 2: return "WARN";
        case 3: return "INFO";
        case 4: return "VERBOSE";
        default: return "UNKNOWN";
    }
}
}

struct Runtime::Impl {
    void* transportLibrary = nullptr;
    void* library = nullptr;
    QNN_INTERFACE_VER_TYPE api = QNN_INTERFACE_VER_TYPE_INIT;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    std::list<std::string> matmulGraphNames;
    uint32_t matmulGraphSerial = 0;
    Qnn_GraphHandle_t graph = nullptr;
    Qnn_GraphHandle_t dWeightGraph = nullptr;
    Qnn_Tensor_t dWeightInputs[2] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT};
    Qnn_Tensor_t dWeightOutput = QNN_TENSOR_INIT;
    uint32_t dWeightADims[2]{}, dWeightBDims[2]{}, dWeightODims[2]{};
    uint32_t dWeightInputElements = 0;
    uint32_t dPredictionElements = 0;
    uint32_t dWeightOutputElements = 0;
    Qnn_Tensor_t inputs[2] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT};
    Qnn_Tensor_t output = QNN_TENSOR_INIT;
    uint32_t adims[2]{}, bdims[2]{}, odims[2]{};
    uint32_t inputElements = 0;
    uint32_t weightElements = 0;
    uint32_t outputElements = 0;
    std::vector<float> weight;
    struct Slot { Qnn_GraphHandle_t graph=nullptr; Qnn_Tensor_t in[2]={QNN_TENSOR_INIT,QNN_TENSOR_INIT}; Qnn_Tensor_t out=QNN_TENSOR_INIT; uint32_t d0[2]{},d1[2]{},doo[2]{}; uint32_t n0=0,n1=0,no=0; } dx,dw2,dh,dw1;
    Qnn_GraphHandle_t mlpForward=nullptr;
    Qnn_Tensor_t mlpInputs[3]={QNN_TENSOR_INIT,QNN_TENSOR_INIT,QNN_TENSOR_INIT};
    Qnn_Tensor_t mlpZ=QNN_TENSOR_INIT, mlpH=QNN_TENSOR_INIT, mlpP=QNN_TENSOR_INIT;
    uint32_t xDims[2]{},w1Dims[2]{},w2Dims[2]{},zDims[2]{},pDims[2]{};
    uint32_t batch=0,inDim=0,hidDim=0,outDim=0;
    std::vector<float> mlpW1,mlpW2;
    struct ReluBackward {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t inputs[3] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT, QNN_TENSOR_INIT};
        Qnn_Tensor_t mask = QNN_TENSOR_INIT;
        Qnn_Tensor_t dz = QNN_TENSOR_INIT;
        uint32_t dims[2]{};
        uint32_t elements = 0;
        std::vector<float> zero;
    } reluBackward;
    struct FusedBackward {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t inputs[5] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT,
                                  QNN_TENSOR_INIT, QNN_TENSOR_INIT,
                                  QNN_TENSOR_INIT};
        Qnn_Tensor_t dw2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t dh = QNN_TENSOR_INIT;
        Qnn_Tensor_t mask = QNN_TENSOR_INIT;
        Qnn_Tensor_t dz = QNN_TENSOR_INIT;
        Qnn_Tensor_t dw1 = QNN_TENSOR_INIT;
        uint32_t dpDims[2]{}, dw2Dims[2]{}, dw1Dims[2]{};
        std::vector<float> zero;
        bool diagnosticOutputs = false;
    } fusedBackward;
    struct TrainingOpsMicro {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t inputs[6] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT, QNN_TENSOR_INIT,
                                  QNN_TENSOR_INIT, QNN_TENSOR_INIT, QNN_TENSOR_INIT};
        Qnn_Tensor_t error = QNN_TENSOR_INIT;
        Qnn_Tensor_t squared = QNN_TENSOR_INIT;
        Qnn_Tensor_t loss = QNN_TENSOR_INIT;
        Qnn_Tensor_t dPrediction = QNN_TENSOR_INIT;
        Qnn_Tensor_t scaledWeight = QNN_TENSOR_INIT;
        Qnn_Tensor_t weightNext = QNN_TENSOR_INIT;
        Qnn_Tensor_t axes = QNN_TENSOR_INIT;
        uint32_t predictionDims[2]{}, weightDims[2]{}, scalarDims[2]{1, 1};
        uint32_t axesDims[1]{2}, lossDims[2]{1, 1};
        uint32_t axesData[2]{0, 1};
        float gradScale = 0.0f;
    } trainingOpsMicro;
    struct MlpFullStep {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t inputs[5] = {QNN_TENSOR_INIT, QNN_TENSOR_INIT, QNN_TENSOR_INIT,
                                  QNN_TENSOR_INIT, QNN_TENSOR_INIT};
        Qnn_Tensor_t z1 = QNN_TENSOR_INIT, hidden = QNN_TENSOR_INIT;
        Qnn_Tensor_t prediction = QNN_TENSOR_INIT, error = QNN_TENSOR_INIT;
        Qnn_Tensor_t squared = QNN_TENSOR_INIT, loss = QNN_TENSOR_INIT;
        Qnn_Tensor_t dPrediction = QNN_TENSOR_INIT, dW2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t dHidden = QNN_TENSOR_INIT, mask = QNN_TENSOR_INIT;
        Qnn_Tensor_t dZ1 = QNN_TENSOR_INIT, dW1 = QNN_TENSOR_INIT;
        Qnn_Tensor_t scaledDW1 = QNN_TENSOR_INIT, scaledDW2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t w1Next = QNN_TENSOR_INIT, w2Next = QNN_TENSOR_INIT;
        Qnn_Tensor_t axes = QNN_TENSOR_INIT, gradScale = QNN_TENSOR_INIT;
        Qnn_Tensor_t zero = QNN_TENSOR_INIT;
        uint32_t xDims[2]{}, yDims[2]{}, w1Dims[2]{}, w2Dims[2]{};
        uint32_t hiddenDims[2]{}, lossDims[2]{1, 1}, scalarDims[2]{1, 1};
        uint32_t axesDims[1]{2}, axesData[2]{0, 1};
        float gradScaleData = 0.0f;
        std::vector<float> zeroData;
        bool diagnosticOutputs = false;
    } mlpFullStep;
    struct UnaryTransformerGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t input = QNN_TENSOR_INIT;
        Qnn_Tensor_t output = QNN_TENSOR_INIT;
        Qnn_Tensor_t gamma = QNN_TENSOR_INIT;
        Qnn_Tensor_t beta = QNN_TENSOR_INIT;
        Qnn_Tensor_t axes = QNN_TENSOR_INIT;
        uint32_t dims[3]{};
        uint32_t normDims[1]{};
        uint32_t axesDims[1]{1};
        uint32_t axesData[1]{};
        uint32_t rank = 0;
        uint32_t elements = 0;
        float epsilon = 1.0e-5f;
        std::vector<float> gammaData;
        std::vector<float> betaData;
    } layerNorm, softmax;
    struct ElementwiseSquareGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t input = QNN_TENSOR_INIT;
        Qnn_Tensor_t output = QNN_TENSOR_INIT;
        uint32_t dims[1]{};
        uint32_t elements = 0;
    } elementwiseSquare;
    struct LayerNormBackwardGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t input = QNN_TENSOR_INIT, upstream = QNN_TENSOR_INIT;
        Qnn_Tensor_t gamma = QNN_TENSOR_INIT, beta = QNN_TENSOR_INIT;
        Qnn_Tensor_t mean = QNN_TENSOR_INIT, centered = QNN_TENSOR_INIT;
        Qnn_Tensor_t centeredScaled = QNN_TENSOR_INIT;
        Qnn_Tensor_t squared = QNN_TENSOR_INIT, variance = QNN_TENSOR_INIT;
        Qnn_Tensor_t varianceEpsilon = QNN_TENSOR_INIT;
        Qnn_Tensor_t inverseStdScaled = QNN_TENSOR_INIT;
        Qnn_Tensor_t inverseStd = QNN_TENSOR_INIT;
        Qnn_Tensor_t normalized = QNN_TENSOR_INIT;
        Qnn_Tensor_t normalizedGamma = QNN_TENSOR_INIT;
        Qnn_Tensor_t output = QNN_TENSOR_INIT;
        Qnn_Tensor_t dNormalized = QNN_TENSOR_INIT;
        Qnn_Tensor_t sumDNormalized = QNN_TENSOR_INIT;
        Qnn_Tensor_t dNormalizedTimesXhat = QNN_TENSOR_INIT;
        Qnn_Tensor_t sumDNormalizedTimesXhat = QNN_TENSOR_INIT;
        Qnn_Tensor_t dimensionTimesDNormalized = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstDifference = QNN_TENSOR_INIT;
        Qnn_Tensor_t xhatTimesSum = QNN_TENSOR_INIT;
        Qnn_Tensor_t bracket = QNN_TENSOR_INIT;
        Qnn_Tensor_t inverseStdOverDimension = QNN_TENSOR_INIT;
        Qnn_Tensor_t dInput = QNN_TENSOR_INIT;
        Qnn_Tensor_t upstreamTimesXhat = QNN_TENSOR_INIT;
        Qnn_Tensor_t dGamma = QNN_TENSOR_INIT, dBeta = QNN_TENSOR_INIT;
        Qnn_Tensor_t lastAxis = QNN_TENSOR_INIT, rowAxis = QNN_TENSOR_INIT;
        Qnn_Tensor_t epsilon = QNN_TENSOR_INIT, dimensionScalar = QNN_TENSOR_INIT;
        Qnn_Tensor_t inverseDimension = QNN_TENSOR_INIT;
        Qnn_Tensor_t centeredScale = QNN_TENSOR_INIT;
        uint32_t dims[2]{}, reducedDims[2]{}, parameterDims[1]{};
        uint32_t scalarDims[2]{1, 1}, axisDims[1]{1};
        uint32_t lastAxisData[1]{1}, rowAxisData[1]{0};
        uint32_t rows = 0, dimension = 0, elements = 0;
        float epsilonData = 1.0e-5f, dimensionData = 1.0f;
        float centeredScaleData = 64.0f;
        float inverseDimensionData = 1.0f;
    } layerNormBackward;
    struct SoftmaxBackwardGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t probabilities = QNN_TENSOR_INIT;
        Qnn_Tensor_t upstream = QNN_TENSOR_INIT;
        Qnn_Tensor_t product = QNN_TENSOR_INIT;
        Qnn_Tensor_t dot = QNN_TENSOR_INIT;
        Qnn_Tensor_t centered = QNN_TENSOR_INIT;
        Qnn_Tensor_t inputGradient = QNN_TENSOR_INIT;
        Qnn_Tensor_t axes = QNN_TENSOR_INIT;
        uint32_t dims[2]{}, reducedDims[2]{}, axesDims[1]{1};
        uint32_t axesData[1]{1};
        uint32_t elements = 0;
    } softmaxBackward;
    struct AttentionGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t query = QNN_TENSOR_INIT, key = QNN_TENSOR_INIT;
        Qnn_Tensor_t value = QNN_TENSOR_INIT, mask = QNN_TENSOR_INIT;
        Qnn_Tensor_t scores = QNN_TENSOR_INIT, scaled = QNN_TENSOR_INIT;
        Qnn_Tensor_t masked = QNN_TENSOR_INIT, probabilities = QNN_TENSOR_INIT;
        Qnn_Tensor_t output = QNN_TENSOR_INIT, scale = QNN_TENSOR_INIT;
        uint32_t activationDims[2]{}, scoreDims[2]{}, scalarDims[2]{1, 1};
        uint32_t tokens = 0, headDimension = 0;
        float scaleData = 1.0f;
    } attention;
    struct AttentionBackwardGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t query = QNN_TENSOR_INIT, key = QNN_TENSOR_INIT;
        Qnn_Tensor_t value = QNN_TENSOR_INIT, upstream = QNN_TENSOR_INIT;
        Qnn_Tensor_t causalMask = QNN_TENSOR_INIT;
        Qnn_Tensor_t scores = QNN_TENSOR_INIT, scaledScores = QNN_TENSOR_INIT;
        Qnn_Tensor_t maskedScores = QNN_TENSOR_INIT;
        Qnn_Tensor_t probabilities = QNN_TENSOR_INIT;
        Qnn_Tensor_t dProbabilities = QNN_TENSOR_INIT;
        Qnn_Tensor_t dValue = QNN_TENSOR_INIT;
        Qnn_Tensor_t softmaxProduct = QNN_TENSOR_INIT;
        Qnn_Tensor_t softmaxDot = QNN_TENSOR_INIT;
        Qnn_Tensor_t softmaxCentered = QNN_TENSOR_INIT;
        Qnn_Tensor_t dScores = QNN_TENSOR_INIT;
        Qnn_Tensor_t dQueryRaw = QNN_TENSOR_INIT;
        Qnn_Tensor_t dKeyRaw = QNN_TENSOR_INIT;
        Qnn_Tensor_t dQuery = QNN_TENSOR_INIT, dKey = QNN_TENSOR_INIT;
        Qnn_Tensor_t scale = QNN_TENSOR_INIT, axes = QNN_TENSOR_INIT;
        uint32_t activationDims[2]{}, scoreDims[2]{}, reducedDims[2]{};
        uint32_t scalarDims[2]{1, 1}, axesDims[1]{1}, axesData[1]{1};
        uint32_t tokens = 0, headDimension = 0;
        float scaleData = 1.0f;
    } attentionBackward;
    struct MomentumOptimizerGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t current = QNN_TENSOR_INIT, gradient = QNN_TENSOR_INIT;
        Qnn_Tensor_t velocity = QNN_TENSOR_INIT, momentum = QNN_TENSOR_INIT;
        Qnn_Tensor_t learningRate = QNN_TENSOR_INIT;
        Qnn_Tensor_t scaledVelocity = QNN_TENSOR_INIT;
        Qnn_Tensor_t velocityNext = QNN_TENSOR_INIT;
        Qnn_Tensor_t scaledUpdate = QNN_TENSOR_INIT;
        Qnn_Tensor_t weightNext = QNN_TENSOR_INIT;
        uint32_t dims[2]{}, scalarDims[2]{1, 1};
        uint32_t elements = 0;
    } momentumOptimizer;
    struct AdamOptimizerGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t current = QNN_TENSOR_INIT, gradient = QNN_TENSOR_INIT;
        Qnn_Tensor_t gradientScale = QNN_TENSOR_INIT;
        Qnn_Tensor_t gradientClipped = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstMoment = QNN_TENSOR_INIT, secondMoment = QNN_TENSOR_INIT;
        Qnn_Tensor_t learningRate = QNN_TENSOR_INIT;
        Qnn_Tensor_t beta1 = QNN_TENSOR_INIT, beta2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t oneMinusBeta1 = QNN_TENSOR_INIT, oneMinusBeta2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t inverseEpsilon = QNN_TENSOR_INIT, one = QNN_TENSOR_INIT;
        Qnn_Tensor_t zero = QNN_TENSOR_INIT, nonzeroMask = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstCorrection = QNN_TENSOR_INIT, secondCorrection = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstScaled = QNN_TENSOR_INIT, gradientFirstScaled = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstMomentNext = QNN_TENSOR_INIT;
        Qnn_Tensor_t gradientSquared = QNN_TENSOR_INIT;
        Qnn_Tensor_t secondScaled = QNN_TENSOR_INIT, gradientSecondScaled = QNN_TENSOR_INIT;
        Qnn_Tensor_t secondMomentNext = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstMomentHat = QNN_TENSOR_INIT, secondMomentHat = QNN_TENSOR_INIT;
        Qnn_Tensor_t secondRoot = QNN_TENSOR_INIT;
        Qnn_Tensor_t firstOverEpsilon = QNN_TENSOR_INIT;
        Qnn_Tensor_t rootOverEpsilon = QNN_TENSOR_INIT;
        Qnn_Tensor_t denominator = QNN_TENSOR_INIT;
        Qnn_Tensor_t safeDenominator = QNN_TENSOR_INIT;
        Qnn_Tensor_t divided = QNN_TENSOR_INIT, normalized = QNN_TENSOR_INIT;
        Qnn_Tensor_t scaledUpdate = QNN_TENSOR_INIT;
        Qnn_Tensor_t weightNext = QNN_TENSOR_INIT;
        uint32_t dims[2]{}, scalarDims[2]{1, 1};
        uint32_t elements = 0;
    } adamOptimizer;
    struct CrossEntropyGradientGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t logits = QNN_TENSOR_INIT, target = QNN_TENSOR_INIT;
        Qnn_Tensor_t probabilities = QNN_TENSOR_INIT;
        Qnn_Tensor_t difference = QNN_TENSOR_INIT, dLogits = QNN_TENSOR_INIT;
        Qnn_Tensor_t scale = QNN_TENSOR_INIT;
        uint32_t dims[2]{}, scalarDims[2]{1, 1};
        uint32_t rows = 0, columns = 0;
        float scaleData = 1.0f;
    } crossEntropyGradient;
    struct TinyTransformerTrainingGraph {
        enum TensorIndex {
            X, TARGET, G1, B1, WQ, WK, WV, WO, G2, B2, W1, W2, LR,
            MASK, SCALE, LAST_AXIS, ROW_AXIS, ALL_AXES, EPS_SCALED,
            CENTER_SCALE, DIMENSION, INV_DIMENSION, GRAD_SCALE, ZERO_FF,
            MEAN1, CENTERED1, CENTERED_S1, SQUARE1, VAR1, VAR_EPS1,
            INV_STD_S1, INV_STD1, XHAT1, XHAT_G1, LN1,
            Q, K, V, SCORES, SCALED_SCORES, MASKED_SCORES, PROBABILITIES,
            CONTEXT, PROJECTED, RESIDUAL1,
            MEAN2, CENTERED2, CENTERED_S2, SQUARE2, VAR2, VAR_EPS2,
            INV_STD_S2, INV_STD2, XHAT2, XHAT_G2, LN2,
            FF1, RELU, FF2, OUTPUT, ERROR, SQUARED_ERROR, LOSS, DOUTPUT,
             DW2, DRELU, RELU_MASK, DFF1, DW1, DLN2,
             DXHAT2, SUM_DXHAT2, DXHAT_XHAT2, DY_XHAT2, SUM_DXHAT_XHAT2,
            D_TIMES_DXHAT2, FIRST_DIFF2, XHAT_TIMES_SUM2, BRACKET2,
            INV_STD_OVER_D2, DRESIDUAL1_LN, DG2, DB2, DRESIDUAL1,
            DWO, DCONTEXT, DPROBABILITIES, DV, SOFTMAX_PRODUCT,
            SOFTMAX_DOT, SOFTMAX_CENTERED, DSCORES, DQ_RAW, DK_RAW, DQ, DK,
            DWQ, DWK, DWV, DLN1_Q, DLN1_K, DLN1_V, DLN1_QK, DLN1,
             DXHAT1, SUM_DXHAT1, DXHAT_XHAT1, DY_XHAT1, SUM_DXHAT_XHAT1,
            D_TIMES_DXHAT1, FIRST_DIFF1, XHAT_TIMES_SUM1, BRACKET1,
            INV_STD_OVER_D1, DG1, DB1,
            S_DG1, N_G1, S_DB1, N_B1, S_DWQ, N_WQ, S_DWK, N_WK,
            S_DWV, N_WV, S_DWO, N_WO, S_DG2, N_G2, S_DB2, N_B2,
            S_DW1, N_W1, S_DW2, N_W2,
            ONE_HOT, EMBEDDING, POSITION, EMBEDDED_TOKEN, OUTPUT_PROJECTION,
            LOGITS, LM_PROBABILITIES, LM_DIFFERENCE, DLOGITS,
            DOUTPUT_PROJECTION,
            DINPUT_NORM, DINPUT, DEMBEDDING, S_DEMBEDDING, N_EMBEDDING,
            S_DOUTPUT_PROJECTION, N_OUTPUT_PROJECTION, TENSOR_COUNT
        };
        Qnn_GraphHandle_t graph = nullptr;
        std::array<Qnn_Tensor_t, TENSOR_COUNT> tensors{};
        std::array<std::string, TENSOR_COUNT> names{};
        uint32_t activationDims[2]{}, scoreDims[2]{}, ffDims[2]{};
        uint32_t ddDims[2]{}, dfDims[2]{}, fdDims[2]{};
        uint32_t normDims[1]{}, rowDims[2]{}, scalarDims[2]{1, 1};
        uint32_t oneHotDims[2]{}, embeddingDims[2]{}, logitsDims[2]{};
        uint32_t outputProjectionDims[2]{};
        uint32_t axisOneDims[1]{1}, axisTwoDims[1]{2};
        uint32_t lastAxisData[1]{1}, rowAxisData[1]{0}, allAxesData[2]{0, 1};
        uint32_t tokens = 0, dimension = 0, feedForwardDimension = 0;
        // Kept with the graph so execute-time buffer validation cannot silently
        // bind a multi-layer parameter object to the established L1/H1 graph.
        // The graph builder owns the eventual dynamic layer/head registry.
        uint32_t numLayers = 1, numHeads = 1, headDimension = 0;
        uint32_t vocabularySize = 0;
        uint32_t sourceTensorCreateSuccessCount = 0;
        uint32_t sourceGraphAddNodeSuccessCount = 0;
        uint32_t lastInputTensorCount = 0, lastOutputTensorCount = 0;
        bool lastLearningRateBytesUnchanged = true;
        bool languageModel = false;
        bool diagnosticOutputs = true;
        TinyTransformerTrainingVariant variant =
            TinyTransformerTrainingVariant::FULL;
        TinyTransformerTrainingTapSet tapSet =
            TinyTransformerTrainingTapSet::NONE;
        float scaleData = 1.0f, epsilonScaledData = 1.0e-5f;
        // The HTP ElementWiseMultiply kernel overflows near the FP16 square
        // limit even though these tensors are declared FP32.  A scale of 64
        // made ordinary late-training centered values around +/-4 become
        // +/-256 and produced Inf in tt_ln2_square.  Eight retains the
        // low-variance precision aid while leaving an 8x measured headroom.
        float centeredScaleData = 8.0f, dimensionData = 1.0f;
        float inverseDimensionData = 1.0f, gradientScaleData = 1.0f;
        std::vector<float> maskData, zeroFfData, positionData;
    } tinyTransformerTraining;

    // Non-baseline Transformer training graphs deliberately do not reuse the
    // fixed L1/H1 tensor enum above.  Each tensor owns its dimension storage
    // and receives one registry entry, so a layer-local activation, gradient,
    // or optimizer buffer cannot alias the same-named item in another layer.
    // The builder lives in qnn_runtime_transformer_training_generalized.inc.
    struct GeneralizedTinyTransformerTrainingGraph {
        enum class State : std::uint8_t { INACTIVE, PREPARING, ACTIVE, FAILED };
        struct HeadRegistry {
            std::uint32_t selector = 0;
            std::uint32_t query = 0, key = 0, value = 0;
            std::uint32_t scores = 0, scaled = 0, masked = 0,
                          probabilities = 0;
            std::uint32_t context = 0, contextScatter = 0;
            std::uint32_t dContext = 0, dProbabilities = 0, dValue = 0;
            std::uint32_t softmaxProduct = 0, softmaxDot = 0,
                          softmaxCentered = 0, dScores = 0;
            std::uint32_t dQueryRaw = 0, dKeyRaw = 0, dQuery = 0,
                          dKey = 0;
            std::uint32_t dQueryScatter = 0, dKeyScatter = 0,
                          dValueScatter = 0;
            std::size_t selectorOffset = 0;
        };
        struct LayerRegistry {
            std::uint32_t input = 0, output = 0;
            std::uint32_t gamma1 = 0, beta1 = 0, wq = 0, wk = 0, wv = 0,
                          wo = 0, gamma2 = 0, beta2 = 0, w1 = 0, w2 = 0;
            std::vector<std::uint32_t> activations;
            std::vector<std::uint32_t> backward;
            std::vector<std::uint32_t> gradients;
            std::vector<HeadRegistry> heads;
            // H>1 accumulates selector/scatter results here.  The final
            // element of each chain is the established full-width tensor.
            std::vector<std::uint32_t> headContextAccumulators;
            std::vector<std::uint32_t> headDqAccumulators;
            std::vector<std::uint32_t> headDkAccumulators;
            std::vector<std::uint32_t> headDvAccumulators;
            std::vector<std::uint32_t> scaledGradients;
            std::vector<std::uint32_t> nextParameters;
        };
        Qnn_GraphHandle_t graph = nullptr;
        std::uint32_t tokenOneHot = 0, tokenEmbedding = 0, target = 0,
                      outputProjection = 0;
        std::vector<Qnn_Tensor_t> tensors;
        std::vector<std::string> names;
        std::vector<std::vector<uint32_t>> dimensions;
        std::vector<LayerRegistry> layers;
        std::vector<std::uint32_t> appWriteRegistry;
        std::vector<std::uint32_t> appReadRegistry;
        // Appended after the established output ABI for scoped diagnostics.
        std::vector<std::uint32_t> tapRegistry;
        std::vector<std::uint32_t> parameterRegistry;
        std::vector<std::uint32_t> gradientRegistry;
        std::vector<std::uint32_t> nextParameterRegistry;
        phonelm::transformer::ResourceEstimate resourceEstimate;
        std::vector<float> maskData, zeroFfData, positionData, selectorData;
        float attentionScale = 1.0f, centeredScale = 8.0f,
              epsilonScaled = 1.0e-5f, gradientScale = 1.0f,
              dimensionValue = 1.0f, inverseDimensionValue = 1.0f;
        std::uint32_t lastAxisData[1]{1}, rowAxisData[1]{0};
        std::uint32_t tokens = 0, dimension = 0, feedForwardDimension = 0;
        std::uint32_t vocabularySize = 0, numLayers = 0, numHeads = 0;
        std::uint32_t tensorCreateSuccessCount = 0, graphAddNodeSuccessCount = 0;
        std::uint32_t lastInputTensorCount = 0, lastOutputTensorCount = 0;
        bool lastLearningRateBytesUnchanged = true;
        State state = State::INACTIVE;
        bool active = false, languageModel = false, diagnosticOutputs = false;
        TinyTransformerTrainingTapSet tapSet =
            TinyTransformerTrainingTapSet::NONE;
        std::uint32_t diagnosticLayerIndex = std::numeric_limits<std::uint32_t>::max();
        bool executeDiagnosticsEmitted = false;
    } generalizedTinyTransformerTraining;

    struct TinyTransformerGraph {
        Qnn_GraphHandle_t graph = nullptr;
        Qnn_Tensor_t input = QNN_TENSOR_INIT, output = QNN_TENSOR_INIT;
        Qnn_Tensor_t ln1 = QNN_TENSOR_INIT, q = QNN_TENSOR_INIT, k = QNN_TENSOR_INIT, v = QNN_TENSOR_INIT;
        Qnn_Tensor_t scores = QNN_TENSOR_INIT, scaled = QNN_TENSOR_INIT, masked = QNN_TENSOR_INIT;
        Qnn_Tensor_t probabilities = QNN_TENSOR_INIT, context = QNN_TENSOR_INIT;
        Qnn_Tensor_t projected = QNN_TENSOR_INIT, residual1 = QNN_TENSOR_INIT, ln2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t ff1 = QNN_TENSOR_INIT, relu = QNN_TENSOR_INIT, ff2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t gamma1 = QNN_TENSOR_INIT, beta1 = QNN_TENSOR_INIT;
        Qnn_Tensor_t gamma2 = QNN_TENSOR_INIT, beta2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t wq = QNN_TENSOR_INIT, wk = QNN_TENSOR_INIT, wv = QNN_TENSOR_INIT;
        Qnn_Tensor_t wo = QNN_TENSOR_INIT, w1 = QNN_TENSOR_INIT, w2 = QNN_TENSOR_INIT;
        Qnn_Tensor_t mask = QNN_TENSOR_INIT, scale = QNN_TENSOR_INIT, axes = QNN_TENSOR_INIT;
        uint32_t activationDims[2]{}, scoreDims[2]{}, ffDims[2]{};
        uint32_t weightDdDims[2]{}, weightDfDims[2]{}, weightFdDims[2]{};
        uint32_t normDims[1]{}, scalarDims[2]{1, 1}, axesDims[1]{1}, axesData[1]{1};
        uint32_t tokens = 0, dimension = 0, feedForwardDimension = 0;
        float epsilon = 1.0e-5f, scaleData = 1.0f;
        std::vector<float> gammaData, betaData, wqData, wkData, wvData, woData;
        std::vector<float> w1Data, w2Data, maskData;
    } tinyTransformer;
};

const char* backendKindName(QnnBackendKind kind) {
    return kind == QnnBackendKind::CPU ? "CPU" : "HTP";
}

Runtime::Runtime() : info_(queryBackendInfo()) {}
Runtime::~Runtime() {
    if (!impl_) return;
    if (impl_->context) impl_->api.contextFree(impl_->context, nullptr);
    if (impl_->device && impl_->api.deviceFree) impl_->api.deviceFree(impl_->device);
    if (impl_->backend) impl_->api.backendFree(impl_->backend);
    if (impl_->log && impl_->api.logFree) impl_->api.logFree(impl_->log);
    if (impl_->library) dlclose(impl_->library);
    if (impl_->transportLibrary) dlclose(impl_->transportLibrary);
    delete impl_;
}
const BackendInfo& Runtime::info() const { return info_; }
const std::string& Runtime::diagnostics() const { return diagnostics_; }
const RuntimeMetrics& Runtime::metrics() const { return metrics_; }
const ApiTrace& Runtime::apiTrace() const { return apiTrace_; }
std::uint32_t Runtime::tinyTransformerTrainingSourceTensorCreateSuccessCount() const {
    return impl_ ? impl_->tinyTransformerTraining.sourceTensorCreateSuccessCount : 0;
}
std::uint32_t Runtime::tinyTransformerTrainingSourceGraphAddNodeSuccessCount() const {
    return impl_ ? impl_->tinyTransformerTraining.sourceGraphAddNodeSuccessCount : 0;
}
std::uint32_t Runtime::tinyTransformerTrainingLastInputTensorCount() const {
    if (!impl_) return 0;
    const auto &generalized = impl_->generalizedTinyTransformerTraining;
    return generalized.state == Impl::GeneralizedTinyTransformerTrainingGraph::State::ACTIVE
        ? generalized.lastInputTensorCount
        : impl_->tinyTransformerTraining.lastInputTensorCount;
}
std::uint32_t Runtime::tinyTransformerTrainingLastOutputTensorCount() const {
    if (!impl_) return 0;
    const auto &generalized = impl_->generalizedTinyTransformerTraining;
    return generalized.state == Impl::GeneralizedTinyTransformerTrainingGraph::State::ACTIVE
        ? generalized.lastOutputTensorCount
        : impl_->tinyTransformerTraining.lastOutputTensorCount;
}
bool Runtime::tinyTransformerTrainingLastLearningRateBytesUnchanged() const {
    if (!impl_) return false;
    const auto &generalized = impl_->generalizedTinyTransformerTraining;
    return generalized.state == Impl::GeneralizedTinyTransformerTrainingGraph::State::ACTIVE
        ? generalized.lastLearningRateBytesUnchanged
        : impl_->tinyTransformerTraining.lastLearningRateBytesUnchanged;
}

void Runtime::setOptions(const RuntimeOptions& options) { options_ = options; }

void Runtime::recordGraphExecuteResult(int qnnResult, int effectiveResult, bool success) {
    const auto call = apiTrace_.graphExecuteAttemptCount++;
    if (call == 0) {
        apiTrace_.graphExecuteFirstCallIndex = 0;
        apiTrace_.graphExecuteFirstResult = effectiveResult;
    }
    apiTrace_.graphExecuteLastCallIndex = static_cast<std::int64_t>(call);
    apiTrace_.graphExecuteLastResult = effectiveResult;
    apiTrace_.lastQnnResult = qnnResult;
    apiTrace_.effectiveResult = effectiveResult;
    if (success) ++apiTrace_.graphExecuteSuccessCount;
    else {
        ++apiTrace_.graphExecuteFailureCount;
        if (apiTrace_.graphExecuteFirstFailureCall < 0)
            apiTrace_.graphExecuteFirstFailureCall = static_cast<std::int64_t>(call);
    }
}

std::string Runtime::apiTraceSummary() const {
    const auto& t = apiTrace_;
    std::ostringstream s;
    s << "api_trace_version=1\n"
      << "api_trace_backend_requested=" << t.backendRequested << '\n'
      << "api_trace_backend_library=" << t.backendLibrary << '\n'
      << "api_trace_backend_library_load_result=" << t.backendLibraryLoadResult << '\n'
      << "api_trace_provider_symbol_resolved=" << boolText(t.providerSymbolResolved) << '\n'
      << "api_trace_provider_count=" << t.providerCount << '\n'
      << "api_trace_selected_provider_index=" << t.selectedProviderIndex << '\n'
      << "api_trace_selected_core_api_version=" << t.selectedCoreApiVersion << '\n'
      << "api_trace_selected_backend_api_version=" << t.selectedBackendApiVersion << '\n'
      << "api_trace_runtime_backend_build_id=" << t.runtimeBackendBuildId << '\n'
      << "api_trace_backend_create_symbol_library=" << t.backendCreateSymbolLibrary << '\n'
      << "api_trace_device_create_symbol_library=" << t.deviceCreateSymbolLibrary << '\n'
      << "api_trace_context_create_symbol_library=" << t.contextCreateSymbolLibrary << '\n'
      << "api_trace_graph_create_symbol_library=" << t.graphCreateSymbolLibrary << '\n'
      << "api_trace_graph_finalize_symbol_library=" << t.graphFinalizeSymbolLibrary << '\n'
      << "api_trace_graph_execute_symbol_library=" << t.graphExecuteSymbolLibrary << '\n'
      << "api_trace_backend_create_called=" << boolText(t.backendCreateCalled) << '\n'
      << "api_trace_backend_create_result=" << t.backendCreateResult << '\n'
      << "api_trace_backend_handle_nonnull=" << boolText(t.backendHandleNonnull) << '\n'
      << "api_trace_device_create_called=" << boolText(t.deviceCreateCalled) << '\n'
      << "api_trace_device_create_result=" << t.deviceCreateResult << '\n'
      << "api_trace_device_handle_nonnull=" << boolText(t.deviceHandleNonnull) << '\n'
      << "api_trace_context_create_called=" << boolText(t.contextCreateCalled) << '\n'
      << "api_trace_context_create_result=" << t.contextCreateResult << '\n'
      << "api_trace_context_handle_nonnull=" << boolText(t.contextHandleNonnull) << '\n'
      << "api_trace_full_step_graph_create_called=" << boolText(t.fullStepGraphCreateCalled) << '\n'
      << "api_trace_full_step_graph_create_result=" << t.fullStepGraphCreateResult << '\n'
      << "api_trace_full_step_graph_handle_nonnull=" << boolText(t.fullStepGraphHandleNonnull) << '\n'
      << "api_trace_full_step_graph_finalize_called=" << boolText(t.fullStepGraphFinalizeCalled) << '\n'
      << "api_trace_full_step_graph_finalize_result=" << t.fullStepGraphFinalizeResult << '\n'
      << "api_trace_graph_execute_attempt_count=" << t.graphExecuteAttemptCount << '\n'
      << "api_trace_graph_execute_success_count=" << t.graphExecuteSuccessCount << '\n'
      << "api_trace_graph_execute_failure_count=" << t.graphExecuteFailureCount << '\n'
      << "api_trace_graph_execute_first_call_index=" << t.graphExecuteFirstCallIndex << '\n'
      << "api_trace_graph_execute_first_result=" << t.graphExecuteFirstResult << '\n'
      << "api_trace_graph_execute_last_call_index=" << t.graphExecuteLastCallIndex << '\n'
      << "api_trace_graph_execute_last_result=" << t.graphExecuteLastResult << '\n'
      << "api_trace_graph_execute_first_failure_call=" << t.graphExecuteFirstFailureCall << '\n'
      << "api_trace_failure_injection_enabled=" << boolText(t.failureInjectionEnabled) << '\n'
      << "api_trace_failure_injection_point=" << t.failureInjectionPoint << '\n'
      << "api_trace_failure_injection_call=" << t.failureInjectionCall << '\n'
      << "api_trace_last_qnn_result=" << t.lastQnnResult << '\n'
      << "api_trace_effective_result=" << t.effectiveResult << '\n'
      << "api_trace_cpu_backend_initialized=" << boolText(t.cpuBackendInitialized) << '\n'
      << "api_trace_fallback_attempted=" << boolText(t.fallbackAttempted) << '\n'
      << "api_trace_fallback_succeeded=" << boolText(t.fallbackSucceeded) << '\n';
    return s.str();
}

std::string Runtime::qnnCallbackCaptureSummary() const {
    std::lock_guard<std::mutex> lock(gQnnCallbackCapture.mutex);
    std::ostringstream s;
    s << "qnn_callback_capture_enabled="
      << boolText(gQnnCallbackCapture.enabled.load(std::memory_order_acquire)) << '\n'
      << "qnn_callback_log_level=" << logLevelText(options_.qnnLogLevel) << '\n'
      << "qnn_callback_saved_message_count=" << gQnnCallbackCapture.messages.size() << '\n'
      << "qnn_callback_dropped_message_count="
      << gQnnCallbackCapture.droppedMessages.load(std::memory_order_relaxed) << '\n'
      << "qnn_callback_saved_bytes=" << gQnnCallbackCapture.savedBytes << '\n'
      << "qnn_callback_limit_messages=" << kCallbackLimitMessages << '\n'
      << "qnn_callback_limit_bytes=" << kCallbackLimitBytes << '\n'
      << "qnn_callback_message_limit_bytes=" << kCallbackMessageLimitBytes << '\n'
      << "qnn_callback_begin\n";
    for (std::size_t i = 0; i < gQnnCallbackCapture.messages.size(); ++i) {
        const auto& message = gQnnCallbackCapture.messages[i];
        s << '[' << i << "] level=" << message.level << " timestamp=" << message.timestamp
          << " message=" << message.message << '\n';
    }
    s << "qnn_callback_end\n";
    return s.str();
}
bool Runtime::initialize(QnnBackendKind kind, std::string& error) {
    impl_ = new Impl;
    apiTrace_ = ApiTrace{};
    apiTrace_.backendRequested = backendKindName(kind);
    apiTrace_.backendLibrary =
        kind == QnnBackendKind::CPU ? "libQnnCpu.so" : "libQnnHtp.so";
    apiTrace_.cpuBackendInitialized = false;
    resetQnnCallbackCapture(options_.captureQnnCallback);
    std::ostringstream d;
    const char* library = kind == QnnBackendKind::CPU ? "libQnnCpu.so" : "libQnnHtp.so";
    d << "requested_backend=" << backendKindName(kind) << '\n'
      << "backend_library=" << library << '\n'
      << "compile_time_sdk_build_id="
      << (QNN_SDK_BUILD_ID[0] == 'v' ? &QNN_SDK_BUILD_ID[1] : QNN_SDK_BUILD_ID) << '\n'
      << "compile_time_qnn_api_version=" << QNN_API_VERSION_MAJOR << '.'
      << QNN_API_VERSION_MINOR << '.' << QNN_API_VERSION_PATCH << '\n'
      << "cpu_fallback=false\n";
    if (kind == QnnBackendKind::HTP) {
        const char* skelDir = std::getenv("PHONELM_QNN_SKEL_DIR");
        const char* expectedHash = std::getenv("PHONELM_QNN_SKEL_EXPECTED_SHA256");
        const char* actualHash = std::getenv("PHONELM_QNN_SKEL_ACTUAL_SHA256");
        const char* skelAction = std::getenv("PHONELM_QNN_SKEL_ACTION");
        const char* adspPath = std::getenv("ADSP_LIBRARY_PATH");
        d << "qnn_skel_dir=" << (skelDir ? skelDir : "UNAVAILABLE") << '\n'
          << "qnn_skel_expected_sha256=" << (expectedHash ? expectedHash : "UNAVAILABLE") << '\n'
          << "qnn_skel_actual_sha256=" << (actualHash ? actualHash : "UNAVAILABLE") << '\n'
          << "qnn_skel_action=" << (skelAction ? skelAction : "UNAVAILABLE") << '\n'
          << "adsp_library_path=" << (adspPath ? adspPath : "UNAVAILABLE") << '\n';
        if (adspPath == nullptr) {
            setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp", 1);
        }
        impl_->transportLibrary = dlopen("libQnnHtpV81Stub.so", RTLD_NOW | RTLD_GLOBAL);
        if (!impl_->transportLibrary) {
            error = std::string("library_load: dlopen(libQnnHtpV81Stub.so): ") + dlerror();
            diagnostics_ = d.str() + "failed_api=library_load\n";
            return false;
        }
    }
    impl_->library = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
    if (!impl_->library) {
        error = std::string("library_load: dlopen(") + library + "): " + dlerror();
        diagnostics_ = d.str() + "failed_api=library_load\n";
        return false;
    }
    apiTrace_.backendLibraryLoadResult = 0;
    d << "library_load_result=0\n";
    auto getProviders = reinterpret_cast<GetProviders>(dlsym(impl_->library, "QnnInterface_getProviders"));
    if (!getProviders) { error = "get_providers: symbol missing"; diagnostics_=d.str()+"failed_api=get_providers\n"; return false; }
    apiTrace_.providerSymbolResolved = true;
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (getProviders(&providers, &count) != QNN_SUCCESS) { error = "get_providers: call failed"; diagnostics_=d.str()+"failed_api=get_providers\n"; return false; }
    apiTrace_.providerCount = count;
    d << "provider_count=" << count << '\n';
    int selected = -1;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& c = providers[i]->apiVersion.coreApiVersion;
        const auto& b = providers[i]->apiVersion.backendApiVersion;
        const bool compatible = c.major == QNN_API_VERSION_MAJOR && c.minor >= QNN_API_VERSION_MINOR;
        d << "provider_" << i << "_core_api_version=" << c.major << '.' << c.minor << '.' << c.patch << '\n'
          << "provider_" << i << "_backend_api_version=" << b.major << '.' << b.minor << '.' << b.patch << '\n'
          << "provider_" << i << "_compatible=" << (compatible ? "true" : "false") << '\n';
        if (selected < 0 && compatible) {
            selected = static_cast<int>(i);
            impl_->api = providers[i]->QNN_INTERFACE_VER_NAME;
            apiTrace_.selectedProviderIndex = selected;
            apiTrace_.selectedCoreApiVersion = std::to_string(c.major) + "." +
                std::to_string(c.minor) + "." + std::to_string(c.patch);
            apiTrace_.selectedBackendApiVersion = std::to_string(b.major) + "." +
                std::to_string(b.minor) + "." + std::to_string(b.patch);
            d << "provider_core_api_version=" << c.major << '.' << c.minor << '.' << c.patch << '\n'
              << "provider_backend_api_version=" << b.major << '.' << b.minor << '.' << b.patch << '\n';
        }
    }
    d << "selected_provider_index=" << selected << '\n';
    if (!impl_->api.backendCreate) { error = "provider_select: compatible provider missing"; diagnostics_=d.str()+"failed_api=provider_select\n"; return false; }
    apiTrace_.backendCreateSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.backendCreate));
    apiTrace_.deviceCreateSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.deviceCreate));
    apiTrace_.contextCreateSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.contextCreate));
    apiTrace_.graphCreateSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.graphCreate));
    apiTrace_.graphFinalizeSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.graphFinalize));
    apiTrace_.graphExecuteSymbolLibrary = functionLibraryBasename(reinterpret_cast<const void*>(impl_->api.graphExecute));
    const char* runtimeBuildId = nullptr;
    if (!impl_->api.backendGetBuildId || impl_->api.backendGetBuildId(&runtimeBuildId) != QNN_SUCCESS || !runtimeBuildId) {
        error = "backend_build_id: QnnBackend_getBuildId failed";
        diagnostics_ = d.str() + "runtime_backend_build_id=UNAVAILABLE\nfailed_api=backend_build_id\n";
        return false;
    }
    apiTrace_.runtimeBackendBuildId = runtimeBuildId;
    d << "runtime_backend_build_id=" << runtimeBuildId << '\n';
    const char* compileBuildId = QNN_SDK_BUILD_ID[0] == 'v' ? &QNN_SDK_BUILD_ID[1] : QNN_SDK_BUILD_ID;
    const char* normalizedRuntimeId = runtimeBuildId[0] == 'v' ? runtimeBuildId + 1 : runtimeBuildId;
    if (std::strcmp(compileBuildId, normalizedRuntimeId) != 0) {
        error = std::string("backend_build_id: compile=") + compileBuildId + ", runtime=" + runtimeBuildId;
        diagnostics_ = d.str() + "backend_build_id_match=false\nfailed_api=backend_build_id\n";
        return false;
    }
    d << "backend_build_id_match=true\n";
    const auto configuredLogLevel = static_cast<QnnLog_Level_t>(options_.qnnLogLevel);
    auto status = impl_->api.logCreate(qnnLogCallback, configuredLogLevel, &impl_->log);
    if (status != QNN_SUCCESS && status != QNN_COMMON_ERROR_NOT_SUPPORTED) {
        error = "log_create: logCreate=" + std::to_string(QNN_GET_ERROR_CODE(status)); diagnostics_=d.str()+"failed_api=log_create\n"; return false;
    }
    d << "log_create_result=" << QNN_GET_ERROR_CODE(status) << '\n';
    auto started = Clock::now();
    apiTrace_.backendCreateCalled = true;
    status = impl_->api.backendCreate(impl_->log, nullptr, &impl_->backend);
    apiTrace_.backendCreateResult = QNN_GET_ERROR_CODE(status);
    apiTrace_.backendHandleNonnull = impl_->backend != nullptr;
    metrics_.backendCreateUs = elapsedUs(started);
    d << "backend_create_result=" << QNN_GET_ERROR_CODE(status) << "\nQnnBackend_create=" << QNN_GET_ERROR_CODE(status) << '\n';
    if (status != QNN_SUCCESS) { error = "backend_create: backendCreate=" + std::to_string(QNN_GET_ERROR_CODE(status)); diagnostics_=d.str()+"failed_api=backend_create\n"; return false; }
    if (kind == QnnBackendKind::CPU) apiTrace_.cpuBackendInitialized = true;
    if (impl_->api.deviceCreate) {
        d << "device_create_called=true\ndevice_create_config_variant=OFFICIAL_SAMPLE_NULL\ndevice_config_pointer_null=true\nconfig_count=0\n";
        started = Clock::now();
        apiTrace_.deviceCreateCalled = true;
        status = impl_->api.deviceCreate(impl_->log, nullptr, &impl_->device);
        apiTrace_.deviceCreateResult = QNN_GET_ERROR_CODE(status);
        apiTrace_.deviceHandleNonnull = impl_->device != nullptr;
        metrics_.deviceCreateUs = elapsedUs(started);
        d << "device_create_result=" << QNN_GET_ERROR_CODE(status) << "\nQnnDevice_create=" << QNN_GET_ERROR_CODE(status)
          << "\ndevice_handle_null=" << (impl_->device ? "false" : "true") << '\n';
        if (status != QNN_SUCCESS && (kind == QnnBackendKind::HTP || status != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE)) {
            error = "device_create: deviceCreate=" + std::to_string(QNN_GET_ERROR_CODE(status)); diagnostics_=d.str()+"context_create_called=false\nfailed_api=device_create\n"; return false;
        }
    }
    auto htpContextConfigs = buildHtpContextConfigs(options_);
    if (htpContextConfigs.failClosedInvalid) {
        error = "context_create: invalid htp context graph splitting option";
        diagnostics_ = d.str() + "htp_context_graph_splitting_runtime=invalid\n"
            "context_create_called=false\nfailed_api=context_create_config\n";
        return false;
    }
    if (kind != QnnBackendKind::HTP && htpContextConfigs.array() != nullptr) {
        error = "context_create: htp context graph splitting requires HTP backend";
        diagnostics_ = d.str() + "htp_context_graph_splitting_runtime=" +
            htpContextConfigs.requested +
            "\ncontext_create_called=false\nfailed_api=context_create_config\n";
        return false;
    }
    d << "htp_context_graph_splitting_runtime=" << htpContextConfigs.requested
      << "\ncontext_create_config_pointer_null="
      << (htpContextConfigs.array() == nullptr ? "true" : "false")
      << "\ncontext_create_config_count="
      << (htpContextConfigs.array() == nullptr ? 0 : 1)
      << "\ncontext_create_called=true\n";
    started = Clock::now();
    apiTrace_.contextCreateCalled = true;
    status = impl_->api.contextCreate(impl_->backend, impl_->device,
                                      htpContextConfigs.array(), &impl_->context);
    apiTrace_.contextCreateResult = QNN_GET_ERROR_CODE(status);
    apiTrace_.contextHandleNonnull = impl_->context != nullptr;
    metrics_.contextCreateUs = elapsedUs(started);
    d << "context_create_result=" << QNN_GET_ERROR_CODE(status) << "\nQnnContext_create=" << QNN_GET_ERROR_CODE(status)
      << "\ncontext_handle_null=" << (impl_->context ? "false" : "true") << '\n';
    if (status != QNN_SUCCESS) { error = "context_create: contextCreate=" + std::to_string(QNN_GET_ERROR_CODE(status)); diagnostics_=d.str()+"failed_api=context_create\n"; return false; }
    diagnostics_ = d.str() + "htp_context_graph_splitting_delivery=" +
        htpContextConfigs.delivery() + "\nfailed_api=none\n";
    return true;
}

static void tensor(Qnn_Tensor_t& t, const char* name, Qnn_TensorType_t type,
                   uint32_t* dims,
                   Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32,
                   uint32_t rank = 2) {
    t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    t.v1.dataType = dataType;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
}

// Builds the HTP graph custom-config array from RuntimeOptions.  Only
// numeric-path switches of the QAIRT 2.48.40.260702 HTP backend are emitted
// (when explicitly requested); none changes graph math.  The caller keeps
// the storage alive across graphCreate.  Returns the config pointer array
// (nullptr when no option is requested) and the number of entries.
struct HtpGraphConfigs {
    QnnHtpGraph_CustomConfig_t precisionConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnHtpGraph_CustomConfig_t compensationConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnHtpGraph_CustomConfig_t weightsPackingConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnHtpGraph_CustomConfig_t fusionConfig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnGraph_Config_t graphConfigs[4] = {};
    QnnGraph_Config_t* graphConfigPtrs[5] = {};
    std::uint32_t count = 0;
    const QnnGraph_Config_t** array = nullptr;
    std::string description;
    bool failClosedInvalid = false;
};

HtpGraphConfigs buildHtpGraphConfigs(const RuntimeOptions& options) {
    HtpGraphConfigs configs;
    const std::uint32_t requested =
        options.htpGraphPrecisionMode + options.htpGraphPrecisionCompensation +
        options.htpGraphWeightsPacking + options.htpGraphAdvancedActivationFusion;
    if (requested == 0) return configs;
    const auto inRange = [](std::uint32_t value) { return value <= 2; };
    if (!inRange(options.htpGraphPrecisionMode) ||
        !inRange(options.htpGraphPrecisionCompensation) ||
        !inRange(options.htpGraphWeightsPacking) ||
        !inRange(options.htpGraphAdvancedActivationFusion)) {
        configs.failClosedInvalid = true;
        return configs;
    }
    const char* parts[4] = {nullptr, nullptr, nullptr, nullptr};
    const auto emit = [&](QnnHtpGraph_CustomConfig_t* custom,
                          QnnHtpGraph_ConfigOption_t option) {
        custom->option = option;
        configs.graphConfigs[configs.count].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        configs.graphConfigs[configs.count].customConfig = custom;
        configs.graphConfigPtrs[configs.count] = &configs.graphConfigs[configs.count];
        ++configs.count;
    };
    if (options.htpGraphPrecisionMode != 0) {
        configs.precisionConfig.precision =
            options.htpGraphPrecisionMode == 2 ? QNN_PRECISION_FLOAT32
                                               : QNN_PRECISION_FLOAT16;
        emit(&configs.precisionConfig, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION);
        parts[0] = options.htpGraphPrecisionMode == 2 ? "float32" : "float16";
    }
    if (options.htpGraphPrecisionCompensation != 0) {
        configs.compensationConfig.precisionCompensation =
            options.htpGraphPrecisionCompensation == 2;
        emit(&configs.compensationConfig,
             QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION);
        parts[1] = options.htpGraphPrecisionCompensation == 2
                       ? "compensation=true"
                       : "compensation=false";
    }
    if (options.htpGraphWeightsPacking != 0) {
        configs.weightsPackingConfig.weightsPacking =
            options.htpGraphWeightsPacking == 2;
        emit(&configs.weightsPackingConfig,
             QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING);
        parts[2] = options.htpGraphWeightsPacking == 2 ? "weights_packing=true"
                                                       : "weights_packing=false";
    }
    if (options.htpGraphAdvancedActivationFusion != 0) {
        configs.fusionConfig.advancedActivationFusion =
            options.htpGraphAdvancedActivationFusion == 2;
        emit(&configs.fusionConfig,
             QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION);
        parts[3] = options.htpGraphAdvancedActivationFusion == 2
                       ? "fusion=true"
                       : "fusion=false";
    }
    configs.graphConfigPtrs[configs.count] = nullptr;
    configs.array = const_cast<const QnnGraph_Config_t**>(configs.graphConfigPtrs);
    std::ostringstream description;
    description << "htp_graph_precision=" << (parts[0] ? parts[0] : "unset")
                << ",htp_graph_precision_compensation="
                << (parts[1] ? parts[1] : "unset")
                << ",htp_graph_weights_packing="
                << (parts[2] ? parts[2] : "unset")
                << ",htp_graph_activation_fusion="
                << (parts[3] ? parts[3] : "unset");
    configs.description = description.str();
    return configs;
}

bool Runtime::recreateContext(std::string& error) {
    if (!impl_ || !impl_->backend || !impl_->device || !impl_->context) {
        error = "context recreation requires initialized backend, device, and context";
        return false;
    }
    // QNN graph/tensor handles belong to a context.  Do not invalidate them
    // before contextFree returns: on a free failure we explicitly fail-close
    // below, while on success no stale handle may survive into the new context.
    auto invalidateContextGraphs = [&] {
        impl_->graph = nullptr;
        impl_->dWeightGraph = nullptr;
        impl_->dWeightInputs[0] = QNN_TENSOR_INIT;
        impl_->dWeightInputs[1] = QNN_TENSOR_INIT;
        impl_->dWeightOutput = QNN_TENSOR_INIT;
        std::fill(std::begin(impl_->dWeightADims), std::end(impl_->dWeightADims), 0);
        std::fill(std::begin(impl_->dWeightBDims), std::end(impl_->dWeightBDims), 0);
        std::fill(std::begin(impl_->dWeightODims), std::end(impl_->dWeightODims), 0);
        impl_->dWeightInputElements = 0;
        impl_->dPredictionElements = 0;
        impl_->dWeightOutputElements = 0;
        impl_->inputs[0] = QNN_TENSOR_INIT;
        impl_->inputs[1] = QNN_TENSOR_INIT;
        impl_->output = QNN_TENSOR_INIT;
        std::fill(std::begin(impl_->adims), std::end(impl_->adims), 0);
        std::fill(std::begin(impl_->bdims), std::end(impl_->bdims), 0);
        std::fill(std::begin(impl_->odims), std::end(impl_->odims), 0);
        impl_->inputElements = 0;
        impl_->weightElements = 0;
        impl_->outputElements = 0;
        impl_->weight.clear();
        impl_->matmulGraphNames.clear();
        impl_->matmulGraphSerial = 0;
        impl_->dx = {};
        impl_->dw2 = {};
        impl_->dh = {};
        impl_->dw1 = {};
        impl_->mlpForward = nullptr;
        impl_->mlpInputs[0] = QNN_TENSOR_INIT;
        impl_->mlpInputs[1] = QNN_TENSOR_INIT;
        impl_->mlpInputs[2] = QNN_TENSOR_INIT;
        impl_->mlpZ = QNN_TENSOR_INIT;
        impl_->mlpH = QNN_TENSOR_INIT;
        impl_->mlpP = QNN_TENSOR_INIT;
        std::fill(std::begin(impl_->xDims), std::end(impl_->xDims), 0);
        std::fill(std::begin(impl_->w1Dims), std::end(impl_->w1Dims), 0);
        std::fill(std::begin(impl_->w2Dims), std::end(impl_->w2Dims), 0);
        std::fill(std::begin(impl_->zDims), std::end(impl_->zDims), 0);
        std::fill(std::begin(impl_->pDims), std::end(impl_->pDims), 0);
        impl_->batch = impl_->inDim = impl_->hidDim = impl_->outDim = 0;
        impl_->mlpW1.clear();
        impl_->mlpW2.clear();
        impl_->reluBackward = {};
        impl_->fusedBackward = {};
        impl_->trainingOpsMicro = {};
        impl_->mlpFullStep = {};
        impl_->layerNorm = {};
        impl_->softmax = {};
        impl_->layerNormBackward = {};
        impl_->softmaxBackward = {};
        impl_->attention = {};
        impl_->attentionBackward = {};
        impl_->momentumOptimizer = {};
        impl_->adamOptimizer = {};
        impl_->crossEntropyGradient = {};
        impl_->tinyTransformerTraining = {};
        impl_->generalizedTinyTransformerTraining = {};
        impl_->tinyTransformer = {};
        apiTrace_.fullStepGraphCreateCalled = false;
        apiTrace_.fullStepGraphFinalizeCalled = false;
        apiTrace_.fullStepGraphCreateResult = -1;
        apiTrace_.fullStepGraphFinalizeResult = -1;
        apiTrace_.fullStepGraphHandleNonnull = false;
    };
    auto status = impl_->api.contextFree(impl_->context, nullptr);
    if (status != QNN_SUCCESS) {
        // The old context is not trustworthy after a failed destruction
        // request.  Deliberately abandon its handles rather than permitting a
        // later execute to use a possibly half-destroyed graph.
        invalidateContextGraphs();
        impl_->context = nullptr;
        error = "contextFree=" + std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }
    invalidateContextGraphs();
    impl_->context = nullptr;
    auto htpContextConfigs = buildHtpContextConfigs(options_);
    if (htpContextConfigs.failClosedInvalid) {
        error = "contextCreate(recreate): invalid htp context graph splitting option";
        return false;
    }
    status = impl_->api.contextCreate(
        impl_->backend, impl_->device, htpContextConfigs.array(), &impl_->context);
    if (status != QNN_SUCCESS) {
        error = "contextCreate(recreate)=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }
    diagnostics_ += "htp_context_graph_splitting_recreate=" +
        std::string(htpContextConfigs.requested) +
        "\nhtp_context_graph_splitting_recreate_config_pointer_null=" +
        (htpContextConfigs.array() == nullptr ? "true" : "false") + "\n";
    return true;
}

bool Runtime::prepareMatMul(uint32_t m, uint32_t k, uint32_t n, bool trans0, std::string& error) {
    if (!impl_ || !impl_->context) { error = "runtime not initialized"; return false; }
    impl_->adims[0] = trans0 ? k : m; impl_->adims[1] = trans0 ? m : k;
    impl_->bdims[0] = k; impl_->bdims[1] = n; impl_->odims[0] = m; impl_->odims[1] = n;
    impl_->inputElements = impl_->adims[0] * impl_->adims[1];
    impl_->weightElements = k * n;
    impl_->outputElements = m * n;
    tensor(impl_->inputs[0], "a", QNN_TENSOR_TYPE_APP_WRITE, impl_->adims);
    tensor(impl_->inputs[1], "b", QNN_TENSOR_TYPE_APP_WRITE, impl_->bdims);
    tensor(impl_->output, "out", QNN_TENSOR_TYPE_APP_READ, impl_->odims);
    auto started = Clock::now();
    impl_->matmulGraphNames.push_back(
        "phonelm_matmul_" + std::to_string(impl_->matmulGraphSerial++));
    const HtpGraphConfigs htpConfigs = buildHtpGraphConfigs(options_);
    if (htpConfigs.failClosedInvalid) {
        error = "graphCreate: invalid htp graph precision option";
        return false;
    }
    auto status = impl_->api.graphCreate(
        impl_->context, impl_->matmulGraphNames.back().c_str(),
        htpConfigs.array, &impl_->graph);
    if (!htpConfigs.description.empty())
        diagnostics_ += htpConfigs.description + "\n";
    metrics_.graphCreateUs = elapsedUs(started);
    ++metrics_.graphCreateCount;
    if (status != QNN_SUCCESS) { error = "graphCreate=" + std::to_string(status); return false; }
    for (auto& input : impl_->inputs) {
        status = impl_->api.tensorCreateGraphTensor(impl_->graph, &input);
        if (status != QNN_SUCCESS) { error = "tensorCreate(input)=" + std::to_string(status); return false; }
    }
    status = impl_->api.tensorCreateGraphTensor(impl_->graph, &impl_->output);
    if (status != QNN_SUCCESS) { error = "tensorCreate(output)=" + std::to_string(status); return false; }
    Qnn_Scalar_t transpose = QNN_SCALAR_INIT;
    transpose.dataType = QNN_DATATYPE_BOOL_8;
    transpose.bool8Value = trans0;
    Qnn_Param_t param = QNN_PARAM_INIT;
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    param.scalarParam = transpose;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "matmul";
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 1; op.v1.params = &param;
    op.v1.numOfInputs = 2; op.v1.inputTensors = impl_->inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &impl_->output;
    status = impl_->api.graphAddNode(impl_->graph, op);
    if (status != QNN_SUCCESS) { error = "graphAddNode=" + std::to_string(status); return false; }
    started = Clock::now();
    status = impl_->api.graphFinalize(impl_->graph, nullptr, nullptr);
    metrics_.graphFinalizeUs = elapsedUs(started);
    ++metrics_.graphFinalizeCount;
    if (status != QNN_SUCCESS) { error = "graphFinalize=" + std::to_string(status); return false; }
    diagnostics_ += "graph_create=success\ngraph_finalize=success\n";
    return true;
}

bool Runtime::prepareDWeightMatMul(uint32_t batchSize, uint32_t inputDimension,
                                    uint32_t outputDimension, std::string& error) {
    if (!impl_ || !impl_->context) { error = "runtime not initialized"; return false; }
    impl_->dWeightADims[0] = batchSize;
    impl_->dWeightADims[1] = inputDimension;
    impl_->dWeightBDims[0] = batchSize;
    impl_->dWeightBDims[1] = outputDimension;
    impl_->dWeightODims[0] = inputDimension;
    impl_->dWeightODims[1] = outputDimension;
    impl_->dWeightInputElements = batchSize * inputDimension;
    impl_->dPredictionElements = batchSize * outputDimension;
    impl_->dWeightOutputElements = inputDimension * outputDimension;
    tensor(impl_->dWeightInputs[0], "dw_x", QNN_TENSOR_TYPE_APP_WRITE,
           impl_->dWeightADims);
    tensor(impl_->dWeightInputs[1], "dw_dp", QNN_TENSOR_TYPE_APP_WRITE,
           impl_->dWeightBDims);
    tensor(impl_->dWeightOutput, "dw_out", QNN_TENSOR_TYPE_APP_READ,
           impl_->dWeightODims);

    auto started = Clock::now();
    const HtpGraphConfigs htpConfigs = buildHtpGraphConfigs(options_);
    if (htpConfigs.failClosedInvalid) {
        error = "graphCreate: invalid htp graph precision option";
        return false;
    }
    auto status = impl_->api.graphCreate(impl_->context, "phonelm_dw_matmul",
                                         htpConfigs.array, &impl_->dWeightGraph);
    if (!htpConfigs.description.empty())
        diagnostics_ += htpConfigs.description + "\n";
    metrics_.dWeightGraphCreateUs = elapsedUs(started);
    ++metrics_.graphCreateCount;
    ++metrics_.dWeightGraphCreateCount;
    if (status != QNN_SUCCESS) {
        error = "dWeightGraphCreate=" + std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }
    for (auto& input : impl_->dWeightInputs) {
        status = impl_->api.tensorCreateGraphTensor(impl_->dWeightGraph, &input);
        if (status != QNN_SUCCESS) {
            error = "dWeightTensorCreate(input)=" +
                    std::to_string(QNN_GET_ERROR_CODE(status));
            return false;
        }
    }
    status = impl_->api.tensorCreateGraphTensor(impl_->dWeightGraph,
                                                 &impl_->dWeightOutput);
    if (status != QNN_SUCCESS) {
        error = "dWeightTensorCreate(output)=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }

    Qnn_Scalar_t transpose = QNN_SCALAR_INIT;
    transpose.dataType = QNN_DATATYPE_BOOL_8;
    transpose.bool8Value = true;
    Qnn_Param_t param = QNN_PARAM_INIT;
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    param.scalarParam = transpose;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "dweight_matmul";
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 1;
    op.v1.params = &param;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = impl_->dWeightInputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &impl_->dWeightOutput;
    status = impl_->api.graphAddNode(impl_->dWeightGraph, op);
    if (status != QNN_SUCCESS) {
        error = "dWeightGraphAddNode=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }

    started = Clock::now();
    status = impl_->api.graphFinalize(impl_->dWeightGraph, nullptr, nullptr);
    metrics_.dWeightGraphFinalizeUs = elapsedUs(started);
    ++metrics_.graphFinalizeCount;
    ++metrics_.dWeightGraphFinalizeCount;
    if (status != QNN_SUCCESS) {
        error = "dWeightGraphFinalize=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }
    diagnostics_ +=
        "dw_graph_create=success\ndw_graph_finalize=success\n"
        "dw_transpose_implementation=QNN_MATMUL_TRANSPOSE_IN0\n";
    return true;
}

bool Runtime::executeDWeight(const std::vector<float>& input,
                             const std::vector<float>& dPrediction,
                             std::vector<float>& dWeight, std::string& error) {
    if (!impl_ || !impl_->dWeightGraph) {
        error = "dW graph not finalized";
        return false;
    }
    if (input.size() != impl_->dWeightInputElements ||
        dPrediction.size() != impl_->dPredictionElements) {
        error = "dW input buffer size mismatch";
        return false;
    }
    dWeight.resize(impl_->dWeightOutputElements);
    auto started = Clock::now();
    impl_->dWeightInputs[0].v1.clientBuf = {
        const_cast<float*>(input.data()),
        static_cast<uint32_t>(input.size() * sizeof(float))};
    metrics_.dWeightXBindUs.push_back(elapsedUs(started));
    ++metrics_.xInputUpdateCount;
    started = Clock::now();
    impl_->dWeightInputs[1].v1.clientBuf = {
        const_cast<float*>(dPrediction.data()),
        static_cast<uint32_t>(dPrediction.size() * sizeof(float))};
    metrics_.dPredictionBindUs.push_back(elapsedUs(started));
    ++metrics_.dPredictionInputUpdateCount;
    started = Clock::now();
    impl_->dWeightOutput.v1.clientBuf = {
        dWeight.data(), static_cast<uint32_t>(dWeight.size() * sizeof(float))};
    metrics_.dWeightOutputBindUs.push_back(elapsedUs(started));
    started = Clock::now();
    const auto status = impl_->api.graphExecute(
        impl_->dWeightGraph, impl_->dWeightInputs, 2, &impl_->dWeightOutput, 1,
        nullptr, nullptr);
    metrics_.dWeightExecuteUs.push_back(elapsedUs(started));
    ++metrics_.graphExecuteCount;
    ++metrics_.dWeightGraphExecuteCount;
    recordGraphExecuteResult(QNN_GET_ERROR_CODE(status), QNN_GET_ERROR_CODE(status),
                             status == QNN_SUCCESS);
    if (status != QNN_SUCCESS) {
        error = "dWeightGraphExecute=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
    }
    return true;
}
bool Runtime::setInitialWeight(const std::vector<float>& weight, std::string& error) {
    if (!impl_ || !impl_->graph) { error = "weight binding requires a finalized graph"; return false; }
    if (weight.size() != impl_->weightElements) { error = "weight buffer size mismatch"; return false; }
    impl_->weight = weight;
    return true;
}

bool Runtime::updateWeight(const std::vector<float>& weight, std::string& error) {
    if (!impl_ || !impl_->graph) { error = "runtime weight update requires a finalized graph"; return false; }
    if (weight.size() != impl_->weightElements) { error = "runtime weight update size mismatch"; return false; }
    auto started = Clock::now();
    std::copy(weight.begin(), weight.end(), impl_->weight.begin());
    metrics_.weightBufferCopyUs.push_back(elapsedUs(started));
    started = Clock::now();
    impl_->inputs[1].v1.clientBuf = {impl_->weight.data(), static_cast<uint32_t>(impl_->weight.size() * sizeof(float))};
    metrics_.weightUpdateUs.push_back(elapsedUs(started));
    ++metrics_.runtimeWeightUpdateCount;
    return true;
}

bool Runtime::executePrepared(const std::vector<float>& input, std::vector<float>& out,
                              std::string& error) {
    if (!impl_ || !impl_->graph) { error = "graph not finalized"; return false; }
    if (input.size() != impl_->inputElements || impl_->weight.size() != impl_->weightElements) {
        error = "prepared input or weight buffer size mismatch";
        return false;
    }
    out.resize(impl_->outputElements);
    auto bindStarted = Clock::now();
    impl_->inputs[0].v1.clientBuf = {const_cast<float*>(input.data()), static_cast<uint32_t>(input.size() * sizeof(float))};
    impl_->inputs[1].v1.clientBuf = {impl_->weight.data(), static_cast<uint32_t>(impl_->weight.size() * sizeof(float))};
    metrics_.inputBindUs.push_back(elapsedUs(bindStarted));
    bindStarted = Clock::now();
    impl_->output.v1.clientBuf = {out.data(), static_cast<uint32_t>(out.size() * sizeof(float))};
    metrics_.outputBindUs.push_back(elapsedUs(bindStarted));
    const auto started = Clock::now();
    const auto status = impl_->api.graphExecute(impl_->graph, impl_->inputs, 2, &impl_->output, 1, nullptr, nullptr);
    metrics_.executeUs.push_back(elapsedUs(started));
    ++metrics_.graphExecuteCount;
    recordGraphExecuteResult(QNN_GET_ERROR_CODE(status), QNN_GET_ERROR_CODE(status),
                             status == QNN_SUCCESS);
    if (status != QNN_SUCCESS) { error = "graphExecute=" + std::to_string(status); return false; }
    return true;
}

bool Runtime::executeMatMul(const std::vector<float>& a, const std::vector<float>& b,
                            std::vector<float>& out, std::string& error) {
    const bool weightOk = impl_ && impl_->weight.empty() ? setInitialWeight(b, error) : updateWeight(b, error);
    return weightOk && executePrepared(a, out, error);
}
#include "qnn_runtime_mlp.inc"
#include "qnn_runtime_full_step.inc"
#include "qnn_runtime_transformer.inc"
#include "qnn_runtime_transformer_training.inc"
#include "qnn_runtime_transformer_training_generalized.inc"
#include "qnn_runtime_transformer_training_generalized_execute.inc"
}
