#include "qnn_runtime.h"

#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpDevice.h>
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
#include <mutex>
#include <sstream>
#include <utility>

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
    d << "context_create_called=true\n";
    started = Clock::now();
    apiTrace_.contextCreateCalled = true;
    status = impl_->api.contextCreate(impl_->backend, impl_->device, nullptr, &impl_->context);
    apiTrace_.contextCreateResult = QNN_GET_ERROR_CODE(status);
    apiTrace_.contextHandleNonnull = impl_->context != nullptr;
    metrics_.contextCreateUs = elapsedUs(started);
    d << "context_create_result=" << QNN_GET_ERROR_CODE(status) << "\nQnnContext_create=" << QNN_GET_ERROR_CODE(status)
      << "\ncontext_handle_null=" << (impl_->context ? "false" : "true") << '\n';
    if (status != QNN_SUCCESS) { error = "context_create: contextCreate=" + std::to_string(QNN_GET_ERROR_CODE(status)); diagnostics_=d.str()+"failed_api=context_create\n"; return false; }
    diagnostics_ = d.str() + "failed_api=none\n";
    return true;
}

static void tensor(Qnn_Tensor_t& t, const char* name, Qnn_TensorType_t type,
                   uint32_t* dims,
                   Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32,
                   uint32_t rank = 2) {
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    t.v1.dataType = dataType;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
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
    auto status = impl_->api.graphCreate(impl_->context, "phonelm_matmul", nullptr, &impl_->graph);
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
    auto status = impl_->api.graphCreate(impl_->context, "phonelm_dw_matmul", nullptr,
                                         &impl_->dWeightGraph);
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
}