#include <cstring>
#include <cmath>
#include <cstdint>
#include <QnnOpPackage.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/core/qhpi.h>
#ifdef __hexagon__
#include <HAP_perf.h>
extern "C" void hexatrain_ordered_hvx_qkv_nn64(const float*,const float*,float*);
#endif
#include "mixed_qkv_arithmetic.h"
namespace mixedpackage {
constexpr const char* name="HexaMixedQkv";
constexpr unsigned marker=0x48383301u,header=32,inputWords=14336,outputWords=6144,receiptWords=20512;
const char* names[]={"MixedPrecisionQKV","MixedPrecisionQKVNoReceipt"};
const char* full[]={"HexaMixedQkv::MixedPrecisionQKV","HexaMixedQkv::MixedPrecisionQKVNoReceipt"};
Qnn_ApiVersion_t apiVersion=QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info={name,nullptr,nullptr,2,nullptr,0,QNN_SDK_BUILD_ID,&apiVersion,nullptr,nullptr,{0}};
bool initialized=false;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t){initialized=true;return QNN_SUCCESS;}
Qnn_ErrorHandle_t end(){initialized=false;return QNN_SUCCESS;}
Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** p){if(!p||!initialized)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;info.operationNames=names;*p=&info;return QNN_SUCCESS;}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c){if(c.version!=QNN_OPCONFIG_VERSION_1||!c.v1.packageName||!c.v1.typeName||std::strcmp(c.v1.packageName,name)||c.v1.numOfParams||c.v1.numOfInputs!=4)return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
 for(unsigned i=0;i<2;++i)if(!std::strcmp(c.v1.typeName,names[i]))return c.v1.numOfOutputs==4-i?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t,QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logEnd(){return QNN_SUCCESS;}
bool shape(const QHPI_Tensor* t,unsigned rows,unsigned cols){if(!t)return false;auto s=qhpi_tensor_shape(t);return s.rank==4&&s.dims[0]==1&&s.dims[1]==1&&s.dims[2]==rows&&s.dims[3]==cols&&qhpi_tensor_type(t)==QHPI_FLOAT32;}
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
#ifndef __hexagon__
 return QHPI_ERROR_FATAL; // Preparation-only Host library. No CPU arithmetic.
#else
 if(ni!=4||no!=(Observe?4u:3u)||!in||!out)return QHPI_ERROR_FATAL;
 const float* x[4];float* y[3];
 for(unsigned i=0;i<4;++i){if(!shape(in[i],i?64:32,64))return QHPI_ERROR_FATAL;x[i]=static_cast<const float*>(qhpi_tensor_raw_data(in[i]));if(!x[i])return QHPI_ERROR_FATAL;}
 for(unsigned i=0;i<3;++i){if(!shape(out[i],32,64))return QHPI_ERROR_FATAL;y[i]=static_cast<float*>(qhpi_tensor_raw_data(out[i]));if(!y[i])return QHPI_ERROR_FATAL;}
 auto overlap=[](const void* a,unsigned an,const void* b,unsigned bn){auto p=reinterpret_cast<uintptr_t>(a),q=reinterpret_cast<uintptr_t>(b);return p<=q?q-p<an:p-q<bn;};
 for(unsigned i=1;i<4;++i)for(unsigned j=0;j<i;++j)if(overlap(x[i],16384,x[j],j?16384:8192))return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<3;++i){for(unsigned j=0;j<4;++j)if(overlap(y[i],8192,x[j],j?16384:8192))return QHPI_ERROR_FATAL;for(unsigned j=0;j<i;++j)if(overlap(y[i],8192,y[j],8192))return QHPI_ERROR_FATAL;}
 unsigned* rec=nullptr;
 if constexpr(Observe){if(!out[3]||qhpi_tensor_type(out[3])!=QHPI_INT32)return QHPI_ERROR_FATAL;auto s=qhpi_tensor_shape(out[3]);if(s.rank!=4||s.dims[0]!=1||s.dims[1]!=1||s.dims[2]!=1||s.dims[3]!=receiptWords)return QHPI_ERROR_FATAL;rec=static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]));if(!rec)return QHPI_ERROR_FATAL;for(unsigned j=0;j<4;++j)if(overlap(rec,receiptWords*4,x[j],j?16384:8192))return QHPI_ERROR_FATAL;for(auto p:y)if(overlap(rec,receiptWords*4,p,8192))return QHPI_ERROR_FATAL;}
 auto now=[](){return HAP_perf_get_time_us();};auto start=now();unsigned at=header;
 for(unsigned i=0;i<4;++i){unsigned n=i?4096:2048;for(unsigned j=0;j<n;++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(rec+at,x[i],n*4);at+=n;}
 auto math=now();mixedarith::project(x[0],x[1],y[0]);mixedarith::project(x[0],x[2],y[1]);auto qkEnd=now();
 // Exactly one unchanged strict IEEE HVX NN64 call, for V only.
 hexatrain_ordered_hvx_qkv_nn64(x[0],x[3],y[2]);auto stop=now();
 for(unsigned i=0;i<3;++i){for(unsigned j=0;j<2048;++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(rec+header+inputWords+i*2048,y[i],8192);}
 if constexpr(Observe){static unsigned calls=0;for(unsigned i=0;i<header;++i)rec[i]=0;rec[0]=marker;rec[1]=++calls;rec[2]=inputWords;rec[3]=outputWords;rec[4]=stop-math;rec[5]=81;rec[6]=32;rec[7]=now()-start;for(unsigned i=0;i<4;++i)rec[8+i]=reinterpret_cast<uintptr_t>(x[i])%128;for(unsigned i=0;i<3;++i)rec[12+i]=reinterpret_cast<uintptr_t>(y[i])%128;rec[15]=128;rec[17]=64;rec[18]=64;rec[19]=3;rec[20]=1;rec[24]=qkEnd-math;rec[25]=stop-qkEnd;rec[26]=1;rec[27]=1;}
 return QHPI_SUCCESS;
#endif
}
QHPI_Tensor_Signature_v1 ins[4]{},outs[4]{};QHPI_Kernel_v1 kernels[2]{};QHPI_OpInfo_v1 ops[2]{};
}
extern "C" __attribute__((visibility("default"))) Qnn_ErrorHandle_t HexaMixedQkvInterfaceProvider(QnnOpPackage_Interface_t* p){using namespace mixedpackage;if(!p)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;*p={};p->interfaceVersion={1,4,0};p->v1_4.init=init;p->v1_4.terminate=end;p->v1_4.getInfo=getInfo;p->v1_4.validateOpConfig=validate;p->v1_4.logInitialize=logInit;p->v1_4.logSetLevel=logLevel;p->v1_4.logTerminate=logEnd;return QNN_SUCCESS;}
extern "C" __attribute__((visibility("default"))) const char* qhpi_init(){using namespace mixedpackage;for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& s:outs)s=ins[0];outs[3]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};QHPI_Plugin_Function f[]={kernel<true>,kernel<false>};for(unsigned i=0;i<2;++i){auto& k=kernels[i];k.function_name=full[i];k.function=f[i];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=ins;k.min_outputs=4-i;k.output_signature=outs;ops[i].name=full[i];ops[i].num_kernels=1;ops[i].kernels=&k;}return qhpi_register_ops_v1(2,ops,name)==QHPI_SUCCESS?name:nullptr;}
