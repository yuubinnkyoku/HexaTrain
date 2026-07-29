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
const ApiTrace& Runtime::apiTrace() const { return apiTrace_; }
std::uint32_t Runtime::tinyTransformerTrainingSourceTensorCreateSuccessCount() const { return 0; }
std::uint32_t Runtime::tinyTransformerTrainingSourceGraphAddNodeSuccessCount() const { return 0; }
std::uint32_t Runtime::tinyTransformerTrainingLastInputTensorCount() const { return 0; }
std::uint32_t Runtime::tinyTransformerTrainingLastOutputTensorCount() const { return 0; }
bool Runtime::tinyTransformerTrainingLastLearningRateBytesUnchanged() const { return false; }
void Runtime::setOptions(const RuntimeOptions& options) { options_ = options; }
std::string Runtime::apiTraceSummary() const { return "api_trace_version=UNAVAILABLE\n"; }
std::string Runtime::qnnCallbackCaptureSummary() const { return "qnn_callback_capture_enabled=false\n"; }
void Runtime::recordGraphExecuteResult(int, int, bool) {}

bool Runtime::prepareMatMul(uint32_t, uint32_t, uint32_t, bool, std::string& error) {
    error = "QNN_DISABLED: MatMul unavailable"; return false;
}
bool Runtime::recreateContext(std::string& error) {
    error = "QNN_DISABLED: context recreation unavailable"; return false;
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
bool Runtime::prepareLayerNormBackward(uint32_t,uint32_t,float,std::string&e){e="QNN_DISABLED: LayerNorm backward unavailable";return false;}
bool Runtime::executeLayerNormBackward(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,LayerNormBackwardOutputs&,std::string&e){e="QNN_DISABLED: LayerNorm backward unavailable";return false;}
bool Runtime::prepareSoftmax(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: Softmax unavailable";return false;}
bool Runtime::executeSoftmax(const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: Softmax unavailable";return false;}
bool Runtime::prepareSoftmaxBackward(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: Softmax backward unavailable";return false;}
bool Runtime::executeSoftmaxBackward(const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: Softmax backward unavailable";return false;}
bool Runtime::prepareMomentumOptimizer(uint32_t,std::string&e){e="QNN_DISABLED: momentum optimizer unavailable";return false;}
bool Runtime::executeMomentumOptimizer(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,float,float,MomentumOptimizerOutputs&,std::string&e){e="QNN_DISABLED: momentum optimizer unavailable";return false;}
bool Runtime::prepareAdamOptimizer(uint32_t,std::string&e){e="QNN_DISABLED: Adam optimizer unavailable";return false;}
bool Runtime::executeAdamOptimizer(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,float,float,float,float,AdamOptimizerOutputs&,std::string&e){e="QNN_DISABLED: Adam optimizer unavailable";return false;}
bool Runtime::prepareCrossEntropyGradient(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: cross entropy gradient unavailable";return false;}
bool Runtime::executeCrossEntropyGradient(const std::vector<float>&,const std::vector<float>&,CrossEntropyGradientOutputs&,std::string&e){e="QNN_DISABLED: cross entropy gradient unavailable";return false;}
bool Runtime::prepareAttention(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: attention unavailable";return false;}
bool Runtime::executeAttention(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: attention unavailable";return false;}
bool Runtime::prepareAttentionBackward(uint32_t,uint32_t,std::string&e){e="QNN_DISABLED: attention backward unavailable";return false;}
bool Runtime::executeAttentionBackward(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,AttentionBackwardOutputs&,std::string&e){e="QNN_DISABLED: attention backward unavailable";return false;}
bool Runtime::prepareTinyTransformer(uint32_t,uint32_t,uint32_t,float,std::string&e){e="QNN_DISABLED: tiny transformer unavailable";return false;}
bool Runtime::executeTinyTransformer(const std::vector<float>&,std::vector<float>&,std::string&e){e="QNN_DISABLED: tiny transformer unavailable";return false;}
bool Runtime::prepareTinyTransformerTraining(uint32_t,uint32_t,uint32_t,float,bool,std::string&e,uint32_t,TinyTransformerTrainingVariant,TinyTransformerTrainingTapSet,uint32_t numLayers,uint32_t numHeads){(void)numLayers;(void)numHeads;e="QNN_DISABLED: tiny transformer training unavailable";return false;}
bool Runtime::executeTinyTransformerTraining(const std::vector<float>&,const std::vector<float>&,const TinyTransformerParameters&,float,TinyTransformerTrainingOutputs&,std::string&e){e="QNN_DISABLED: tiny transformer training unavailable";return false;}
bool Runtime::prepareMlpFullStep(uint32_t,uint32_t,uint32_t,uint32_t,bool,std::string&e){e="QNN_DISABLED: full step unavailable";return false;}
bool Runtime::executeMlpFullStep(const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,float,MlpFullStepOutputs&,std::string&e){e="QNN_DISABLED: full step unavailable";return false;}
}  // namespace phonelm::qnn
