#include "qnn_runtime.h"

namespace phonelm::qnn {

const char* backendKindName(QnnBackendKind kind) {
    switch (kind) {
        case QnnBackendKind::CPU: return "CPU";
        case QnnBackendKind::HTP: return "HTP";
    }
    return "UNKNOWN";
}

Runtime::Runtime() : info_(queryBackendInfo()) {}
Runtime::~Runtime() = default;

const BackendInfo& Runtime::info() const {
    return info_;
}
const std::string& Runtime::diagnostics() const { return diagnostics_; }
const RuntimeMetrics& Runtime::metrics() const { return metrics_; }
void Runtime::setOptions(const RuntimeOptions& options) { options_ = options; }
std::string Runtime::apiTraceSummary() const { return "api_trace_version=UNAVAILABLE\n"; }
std::string Runtime::qnnCallbackCaptureSummary() const { return "qnn_callback_capture_enabled=false\n"; }
void Runtime::recordGraphExecuteResult(int, int, bool) {}

bool Runtime::prepareMatMul(uint32_t, uint32_t, uint32_t, bool, std::string& error) {
    error = "QNN_DISABLED: MatMul unavailable"; return false;
}
bool Runtime::executeMatMul(const std::vector<float>&, const std::vector<float>&,
                            std::vector<float>&, std::string& error) {
    error = "QNN_DISABLED: MatMul unavailable"; return false;
}bool Runtime::setInitialWeight(const std::vector<float>&, std::string& error) {
    error = "QNN_DISABLED: weight binding unavailable"; return false;
}
bool Runtime::updateWeight(const std::vector<float>&, std::string& error) {
    error = "QNN_DISABLED: weight update unavailable"; return false;
}
bool Runtime::executePrepared(const std::vector<float>&, std::vector<float>&,
                              std::string& error) {
    error = "QNN_DISABLED: MatMul unavailable"; return false;
}

bool Runtime::prepareDWeightMatMul(uint32_t, uint32_t, uint32_t, std::string& error) {
    error = "QNN_DISABLED: dW MatMul unavailable"; return false;
}
bool Runtime::executeDWeight(const std::vector<float>&, const std::vector<float>&,
                             std::vector<float>&, std::string& error) {
    error = "QNN_DISABLED: dW MatMul unavailable"; return false;
}

bool Runtime::initialize(QnnBackendKind requestedBackend, std::string& error) {
    const std::string backend = backendKindName(requestedBackend);
    if (!info_.sdkDetected) {
        error = "QNN_SDK_NOT_FOUND: cannot initialize " + backend;
        return false;
    }
    if (!info_.qnnBuildEnabled) {
        error = "QNN_DISABLED: cannot initialize " + backend;
        return false;
    }
    error = "NOT_IMPLEMENTED: qnn_runtime_qairt.cpp is unavailable until the installed "
            "SDK headers and official samples are audited";
    return false;
}

bool Runtime::prepareInputGradientMatMul(uint32_t,uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: dX unavailable";return false;}
bool Runtime::executeInputGradient(const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: dX unavailable";return false;}
bool Runtime::prepareMlp(uint32_t,uint32_t,uint32_t,uint32_t,std::string&e,bool){e="QNN_DISABLED: MLP unavailable";return false;}
bool Runtime::setMlpWeights(const std::vector<float>&,const std::vector<float>&,std::string&e){e="QNN_DISABLED: MLP unavailable";return false;}
bool Runtime::executeMlpForward(const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: MLP unavailable";return false;}
bool Runtime::executeMlpSecondBackward(const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: MLP unavailable";return false;}
bool Runtime::executeMlpFirstBackward(const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: MLP unavailable";return false;}
bool Runtime::prepareReluBackward(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: ReLU backward unavailable";return false;}
bool Runtime::executeReluBackward(const std::vector<float>&,const std::vector<float>&,std::vector<std::uint8_t>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: ReLU backward unavailable";return false;}
bool Runtime::prepareMlpFusedBackward(bool,std::string&e){e="QNN_DISABLED: fused backward unavailable";return false;}
bool Runtime::executeMlpFusedBackward(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::vector<std::uint8_t>&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: fused backward unavailable";return false;}
bool Runtime::prepareTrainingOpsMicro(uint32_t,uint32_t,uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: training ops micro unavailable";return false;}
bool Runtime::executeTrainingOpsMicro(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,float,float&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: training ops micro unavailable";return false;}
bool Runtime::prepareLayerNorm(uint32_t,uint32_t,uint32_t,float,std::string&e){e="QNN_DISABLED: LayerNorm unavailable";return false;}
bool Runtime::executeLayerNorm(const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: LayerNorm unavailable";return false;}
bool Runtime::prepareSoftmax(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: Softmax unavailable";return false;}
bool Runtime::executeSoftmax(const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: Softmax unavailable";return false;}
bool Runtime::prepareAttention(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: attention unavailable";return false;}
bool Runtime::executeAttention(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: attention unavailable";return false;}
bool Runtime::prepareTinyTransformer(uint32_t,uint32_t,uint32_t,float,std::string&e){e="QNN_DISABLED: tiny transformer unavailable";return false;}
bool Runtime::executeTinyTransformer(const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: tiny transformer unavailable";return false;}
bool Runtime::prepareMlpFullStep(uint32_t,uint32_t,uint32_t,uint32_t,bool,std::string&e){e="QNN_DISABLED: full step unavailable";return false;}
bool Runtime::executeMlpFullStep(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,float,MlpFullStepOutputs&,std::string&e){e="QNN_DISABLED: full step unavailable";return false;}
}  // namespace phonelm::qnn
