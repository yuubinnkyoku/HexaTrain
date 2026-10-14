// Diagnostic-only package; production graph and optimizer remain separate.
#include <cstring>
#include <QnnOpPackage.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/core/qhpi.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#endif
#include "layernorm_ops.h"
#include "custom_matmul_ops.h"
#include "head_split_op.h"
#include "broadcast_multiply_op.h"

namespace {
constexpr const char* package = "HexaFp32";
const char* names[] = {"Copy", "Add", "Multiply", "Affine", "LnForward", "LnBackward", "LnResidual", "MatMulDContext", "MatMulDWo", "HeadSplit", "BroadcastMultiply", "MatMulDProbabilities", "MatMulDProbabilitiesNoReceipt", "LnForwardNoReceipt", "LnBackwardNoReceipt", "AddNoReceipt", "MatMulDContextNoReceipt", "MatMulDWoNoReceipt", "HeadSplitNoReceipt", "BroadcastMultiplyNoReceipt", "CopyNoReceipt"};
Qnn_ApiVersion_t apiVersion = QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info = {package, names, nullptr, 21, nullptr, 0,
    QNN_SDK_BUILD_ID, &apiVersion, nullptr, nullptr, {0}};
bool initialized = false;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t) { initialized=true; return QNN_SUCCESS; }
Qnn_ErrorHandle_t terminate() { initialized=false; return QNN_SUCCESS; }
Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** p) {
    if (!p || !initialized) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    *p=&info; return QNN_SUCCESS;
}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c) {
    if(c.version!=QNN_OPCONFIG_VERSION_1 || !c.v1.packageName || !c.v1.typeName ||
       std::strcmp(c.v1.packageName,package) || c.v1.numOfParams)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"LnForward"))return c.v1.numOfInputs==4 && c.v1.numOfOutputs==4?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"LnBackward"))return c.v1.numOfInputs==4 && c.v1.numOfOutputs==4?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"LnResidual"))return c.v1.numOfInputs==5 && c.v1.numOfOutputs==4?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"HeadSplit"))return c.v1.numOfInputs==1 && c.v1.numOfOutputs==3?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"BroadcastMultiply"))return c.v1.numOfInputs==4 && c.v1.numOfOutputs==3?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"MatMulDProbabilities"))return c.v1.numOfInputs==4&&c.v1.numOfOutputs==3?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"MatMulDProbabilitiesNoReceipt"))return c.v1.numOfInputs==4&&c.v1.numOfOutputs==2?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const char* freeNames[]={"LnForwardNoReceipt","LnBackwardNoReceipt","AddNoReceipt","MatMulDContextNoReceipt","MatMulDWoNoReceipt","HeadSplitNoReceipt","BroadcastMultiplyNoReceipt","CopyNoReceipt"};
    const unsigned freeInputs[]={4,4,2,2,2,1,4,1},freeOutputs[]={3,3,1,1,1,2,2,1};
    for(unsigned j=0;j<8;++j)if(!std::strcmp(c.v1.typeName,freeNames[j]))return c.v1.numOfInputs==freeInputs[j]&&c.v1.numOfOutputs==freeOutputs[j]?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(c.v1.numOfOutputs!=2)return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if(!std::strcmp(c.v1.typeName,"MatMulDContext")||!std::strcmp(c.v1.typeName,"MatMulDWo"))
        return c.v1.numOfInputs==2?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    for(unsigned k=0;k<4;++k) if(!std::strcmp(c.v1.typeName,names[k]))
        return c.v1.numOfInputs==(k==0?1u:k==3?3u:2u)?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t,QnnLog_Level_t) {return QNN_SUCCESS;}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t) {return QNN_SUCCESS;}
Qnn_ErrorHandle_t logEnd() {return QNN_SUCCESS;}
QHPI_Tensor_Signature_v1 floats[3] = {
    {QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY}};
QHPI_Tensor_Signature_v1 outputs[2] = {
    {QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY}};
// Integer receipt is produced only by the target kernel. It includes input
// words and a per-call counter; the host must validate it independently.
unsigned calls=0;
template<unsigned Kind,bool Observe=true> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** o,
    uint32_t ni,const QHPI_Tensor* const* in) {
    if(no!=(Observe?2u:1u) || ni!=(Kind==0?1u:Kind==3?3u:2u)) return QHPI_ERROR_FATAL;
    auto shape=qhpi_tensor_shape(in[0]); unsigned n=1;
    for(unsigned d=0;d<shape.rank;++d)n*=shape.dims[d];
    if(!n || n>4096)return QHPI_ERROR_FATAL;
    auto count=[](const QHPI_Tensor* t){auto s=qhpi_tensor_shape(t);unsigned total=1;
        for(unsigned d=0;d<s.rank;++d)total*=s.dims[d];return total;};
    const bool observedAdd=Observe&&Kind==1&&count(o[1])==4*n+8;
    if(count(o[0])!=n)return QHPI_ERROR_FATAL;
    if constexpr(Observe)if(count(o[1])!=3*n+8&&!observedAdd)return QHPI_ERROR_FATAL;
    for(unsigned i=0;i<ni;++i)if(count(in[i])!=n || qhpi_tensor_type(in[i])!=QHPI_FLOAT32)return QHPI_ERROR_FATAL;
    if(qhpi_tensor_type(o[0])!=QHPI_FLOAT32)return QHPI_ERROR_FATAL;
    if constexpr(Observe)if(qhpi_tensor_type(o[1])!=QHPI_INT32)return QHPI_ERROR_FATAL;
    const float* x=static_cast<const float*>(qhpi_tensor_raw_data(in[0]));
    float* y=static_cast<float*>(qhpi_tensor_raw_data(o[0]));
    unsigned* receipt=nullptr;if constexpr(Observe)receipt=static_cast<unsigned*>(qhpi_tensor_raw_data(o[1]));
    if(!x || !y || (Observe&&!receipt))return QHPI_ERROR_FATAL;
    const auto started=lncustom::now();
    if constexpr(Observe){
    #ifdef __hexagon__
    receipt[0]=0x48383100u+Kind;
    #else
    receipt[0]=0x43505500u+Kind;
    #endif
    receipt[1]=++calls; receipt[2]=n;receipt[3]=ni;
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<n;++j) {
        receipt[8+i*n+j]=0;
        if(i<ni)std::memcpy(receipt+8+i*n+j,static_cast<const float*>(qhpi_tensor_raw_data(in[i]))+j,4);
    }
    }
    for(unsigned j=0;j<n;++j) {
        if constexpr(Kind==0) std::memcpy(y+j,x+j,4);
        else {
            const float* b=static_cast<const float*>(qhpi_tensor_raw_data(in[1]));
            if constexpr(Kind==1)y[j]=x[j]+b[j];
            if constexpr(Kind==2)y[j]=x[j]*b[j];
            if constexpr(Kind==3) {
                const float* c=static_cast<const float*>(qhpi_tensor_raw_data(in[2]));
                volatile float product=x[j]*b[j]; y[j]=product+c[j];
            }
        }
    }
    if(observedAdd)std::memcpy(receipt+8+3*n,y,4*n);
    if constexpr(Observe){
    #ifdef __hexagon__
    receipt[4]=unsigned(HAP_perf_get_time_us()-started);
    #else
    receipt[4]=0;
    #endif
    receipt[5]=81;receipt[6]=32;receipt[7]=0;}
    return QHPI_SUCCESS;
}
QHPI_Kernel_v1 kernels[4] = {};
QHPI_OpInfo_v1 ops[4] = {};
}
#include "attention_matmul_op.h"
extern "C" Qnn_ErrorHandle_t HexaFp32InterfaceProvider(QnnOpPackage_Interface_t* p) {
    if(!p)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    *p={};p->interfaceVersion={1,4,0};p->v1_4.init=init;p->v1_4.terminate=terminate;
    p->v1_4.getInfo=getInfo;p->v1_4.validateOpConfig=validate;
    p->v1_4.logInitialize=logInit;p->v1_4.logSetLevel=logLevel;p->v1_4.logTerminate=logEnd;
    return QNN_SUCCESS;
}
extern "C" const char* qhpi_init() {
    QHPI_Plugin_Function functions[]={kernel<0>,kernel<1>,kernel<2>,kernel<3>};
    const char* fullNames[]={"HexaFp32::Copy","HexaFp32::Add","HexaFp32::Multiply","HexaFp32::Affine"};
    for(unsigned j=0;j<4;++j) {
        auto& k=kernels[j];k.function_name=fullNames[j];k.function=functions[j];
        k.resources=QHPI_RESOURCE_HVX;k.min_inputs=j==0?1:j==3?3:2;
        k.input_signature=floats;k.min_outputs=2;k.output_signature=outputs;
        ops[j].name=fullNames[j];ops[j].num_kernels=1;ops[j].kernels=&k;
    }
    return qhpi_register_ops_v1(4,ops,package)==QHPI_SUCCESS && lncustom::registerOps() && mmcustom::registerOps() && headcustom::registerOps() && broadcastcustom::registerOps() && attentioncustom::registerOps()?package:nullptr;
}
