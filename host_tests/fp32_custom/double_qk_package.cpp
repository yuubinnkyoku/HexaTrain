#include <cstring>
#include <cmath>
#include <cstdint>
#include <QnnOpPackage.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/core/qhpi.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#endif
#include "double_qk_arithmetic.h"
namespace doubleqkpackage {
constexpr const char* name="HexaDoubleQk";
constexpr unsigned marker=0x48383401u,header=32,receiptWords=6176;
const char* names[]={"ForwardRawQkDouble","ForwardRawQkDoubleNoReceipt"};
const char* full[]={"HexaDoubleQk::ForwardRawQkDouble","HexaDoubleQk::ForwardRawQkDoubleNoReceipt"};
Qnn_ApiVersion_t apiVersion=QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info={name,nullptr,nullptr,2,nullptr,0,QNN_SDK_BUILD_ID,&apiVersion,nullptr,nullptr,{0}};
bool initialized=false;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t){initialized=true;return QNN_SUCCESS;}
Qnn_ErrorHandle_t end(){initialized=false;return QNN_SUCCESS;}
Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** p){if(!p||!initialized)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;info.operationNames=names;*p=&info;return QNN_SUCCESS;}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c){if(c.version!=QNN_OPCONFIG_VERSION_1||!c.v1.packageName||!c.v1.typeName||std::strcmp(c.v1.packageName,name)||c.v1.numOfParams||c.v1.numOfInputs!=4)return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;for(unsigned i=0;i<2;++i)if(!std::strcmp(c.v1.typeName,names[i]))return c.v1.numOfOutputs==3-i?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t,QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logEnd(){return QNN_SUCCESS;}
bool shape(const QHPI_Tensor* t,unsigned rows,unsigned cols,QHPI_Element_Type type=QHPI_FLOAT32){if(!t)return false;auto s=qhpi_tensor_shape(t);return s.rank==4&&s.dims[0]==1&&s.dims[1]==1&&s.dims[2]==rows&&s.dims[3]==cols&&qhpi_tensor_type(t)==type;}
bool overlap(const void* a,unsigned an,const void* b,unsigned bn){auto p=reinterpret_cast<uintptr_t>(a),q=reinterpret_cast<uintptr_t>(b);return p<=q?q-p<an:p-q<bn;}
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
#if !defined(__hexagon__) && !defined(DOUBLE_QK_HOST_CONTRACT_TEST)
 return QHPI_ERROR_FATAL; // Host preparation must never run tensor arithmetic.
#else
 if(ni!=4||no!=(Observe?3u:2u)||!in||!out)return QHPI_ERROR_FATAL;
 const float* x[4];float* y[2];unsigned* rec=nullptr;
 for(unsigned i=0;i<4;++i){if(!shape(in[i],32,32))return QHPI_ERROR_FATAL;x[i]=static_cast<const float*>(qhpi_tensor_raw_data(in[i]));if(!x[i]||reinterpret_cast<uintptr_t>(x[i])%4)return QHPI_ERROR_FATAL;}
 for(unsigned i=0;i<2;++i){if(!shape(out[i],32,32))return QHPI_ERROR_FATAL;y[i]=static_cast<float*>(qhpi_tensor_raw_data(out[i]));if(!y[i]||reinterpret_cast<uintptr_t>(y[i])%4)return QHPI_ERROR_FATAL;for(unsigned j=0;j<4;++j)if(overlap(y[i],4096,x[j],4096))return QHPI_ERROR_FATAL;for(unsigned j=0;j<i;++j)if(overlap(y[i],4096,y[j],4096))return QHPI_ERROR_FATAL;}
 if constexpr(Observe){if(!shape(out[2],1,receiptWords,QHPI_INT32))return QHPI_ERROR_FATAL;rec=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!rec||reinterpret_cast<uintptr_t>(rec)%4)return QHPI_ERROR_FATAL;for(auto p:x)if(overlap(rec,receiptWords*4,p,4096))return QHPI_ERROR_FATAL;for(auto p:y)if(overlap(rec,receiptWords*4,p,4096))return QHPI_ERROR_FATAL;}
 auto now=[](){
#ifdef __hexagon__
 return HAP_perf_get_time_us();
#else
 return uint64_t(0);
#endif
 };auto start=now();
 for(unsigned i=0;i<4;++i){for(unsigned j=0;j<1024;++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(rec+header+i*1024,x[i],4096);}
 auto math=now();doubleqkarith::nt(x[0],x[2],y[0]);doubleqkarith::nt(x[1],x[3],y[1]);auto stop=now();
 for(unsigned i=0;i<2;++i){for(unsigned j=0;j<1024;++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(rec+header+4096+i*1024,y[i],4096);}
 if constexpr(Observe){static unsigned calls=0;for(unsigned i=0;i<header;++i)rec[i]=0;
#ifdef __hexagon__
 rec[0]=marker;
#else
 rec[0]=0x43503401u;
#endif
 rec[1]=++calls;rec[2]=4096;rec[3]=2048;rec[4]=stop-math;rec[5]=81;rec[6]=32;rec[7]=now()-start;for(unsigned i=0;i<4;++i)rec[8+i]=reinterpret_cast<uintptr_t>(x[i])%128;for(unsigned i=0;i<2;++i)rec[12+i]=reinterpret_cast<uintptr_t>(y[i])%128;rec[16]=2;rec[17]=32;rec[18]=0;rec[19]=1;rec[24]=1;}
 return QHPI_SUCCESS;
#endif
}
QHPI_Tensor_Signature_v1 ins[4]{},outs[3]{};QHPI_Kernel_v1 kernels[2]{};QHPI_OpInfo_v1 ops[2]{};
}
extern "C" __attribute__((visibility("default"))) Qnn_ErrorHandle_t HexaDoubleQkInterfaceProvider(QnnOpPackage_Interface_t* p){using namespace doubleqkpackage;if(!p)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;*p={};p->interfaceVersion={1,4,0};p->v1_4.init=init;p->v1_4.terminate=end;p->v1_4.getInfo=getInfo;p->v1_4.validateOpConfig=validate;p->v1_4.logInitialize=logInit;p->v1_4.logSetLevel=logLevel;p->v1_4.logTerminate=logEnd;return QNN_SUCCESS;}
extern "C" __attribute__((visibility("default"))) const char* qhpi_init(){using namespace doubleqkpackage;for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& s:outs)s=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};QHPI_Plugin_Function f[]={kernel<true>,kernel<false>};for(unsigned i=0;i<2;++i){auto& k=kernels[i];k.function_name=full[i];k.function=f[i];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=ins;k.min_outputs=3-i;k.output_signature=outs;ops[i].name=full[i];ops[i].num_kernels=1;ops[i].kernels=&k;}return qhpi_register_ops_v1(2,ops,name)==QHPI_SUCCESS?name:nullptr;}
