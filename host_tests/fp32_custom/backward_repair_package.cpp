#include <cmath>
#include <cstring>
#include <cstdint>
#include <QnnOpPackage.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/core/qhpi.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#endif
#include "backward_repair_arithmetic.h"
namespace repairpackage {
constexpr const char* name="HexaBackwardRepair";
constexpr unsigned header=32,softWords=5184,valueWords=3104;
const char* names[]={"Softmax","SoftmaxNoReceipt","SoftmaxTap","SoftmaxTapNoReceipt","DValueTN32","DValueTN32NoReceipt"};
const char* full[]={"HexaBackwardRepair::Softmax","HexaBackwardRepair::SoftmaxNoReceipt","HexaBackwardRepair::SoftmaxTap","HexaBackwardRepair::SoftmaxTapNoReceipt","HexaBackwardRepair::DValueTN32","HexaBackwardRepair::DValueTN32NoReceipt"};
const unsigned outputCount[]={2,1,5,4,2,1};
Qnn_ApiVersion_t apiVersion=QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info={name,nullptr,nullptr,6,nullptr,0,QNN_SDK_BUILD_ID,&apiVersion,nullptr,nullptr,{0}};
bool initialized=false;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t){initialized=true;return QNN_SUCCESS;}
Qnn_ErrorHandle_t end(){initialized=false;return QNN_SUCCESS;}
Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** p){if(!p||!initialized)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;info.operationNames=names;*p=&info;return QNN_SUCCESS;}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c){if(c.version!=QNN_OPCONFIG_VERSION_1||!c.v1.packageName||!c.v1.typeName||std::strcmp(c.v1.packageName,name)||c.v1.numOfParams||c.v1.numOfInputs!=2)return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;for(unsigned i=0;i<6;++i)if(!std::strcmp(c.v1.typeName,names[i]))return c.v1.numOfOutputs==outputCount[i]?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t,QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logEnd(){return QNN_SUCCESS;}
bool shape(const QHPI_Tensor* t,unsigned rows,unsigned cols,QHPI_Element_Type type=QHPI_FLOAT32){
 if(!t||qhpi_tensor_type(t)!=type)return false;auto s=qhpi_tensor_shape(t);
 return (s.rank==4&&s.dims[0]==1&&s.dims[1]==1&&s.dims[2]==rows&&s.dims[3]==cols)||
        (s.rank==2&&s.dims[0]==rows&&s.dims[1]==cols);
}
bool overlap(const void* a,unsigned an,const void* b,unsigned bn){auto p=reinterpret_cast<uintptr_t>(a),q=reinterpret_cast<uintptr_t>(b);return p<=q?q-p<an:p-q<bn;}
template<bool Value,bool Tap,bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
#if !defined(__hexagon__) && !defined(BACKWARD_REPAIR_HOST_CONTRACT)
 return QHPI_ERROR_FATAL;
#else
 constexpr unsigned mathOut=Tap?4:1,receiptWords=Value?valueWords:softWords;
 if(ni!=2||no!=mathOut+Observe||!in||!out)return QHPI_ERROR_FATAL;
 const float* x[2];float* y[4]{};unsigned counts[5]={1024,1024,32,1024,0};
 for(unsigned i=0;i<2;++i){if(!shape(in[i],32,32))return QHPI_ERROR_FATAL;x[i]=static_cast<const float*>(qhpi_tensor_raw_data(in[i]));if(!x[i]||reinterpret_cast<uintptr_t>(x[i])%4)return QHPI_ERROR_FATAL;}
 void* buffers[5]{};
 for(unsigned i=0;i<mathOut+Observe;++i){bool receipt=i==mathOut;unsigned n=receipt?receiptWords:counts[i];
  if(!shape(out[i],receipt?1:i==2?32:32,receipt?n:i==2?1:32,receipt?QHPI_INT32:QHPI_FLOAT32))return QHPI_ERROR_FATAL;
  buffers[i]=qhpi_tensor_raw_data(out[i]);if(!buffers[i]||reinterpret_cast<uintptr_t>(buffers[i])%4)return QHPI_ERROR_FATAL;
  for(auto input:x)if(overlap(buffers[i],n*4,input,4096))return QHPI_ERROR_FATAL;
  for(unsigned j=0;j<i;++j)if(overlap(buffers[i],n*4,buffers[j],counts[j]*4))return QHPI_ERROR_FATAL;
  counts[i]=n;if(i<mathOut)y[i]=static_cast<float*>(buffers[i]);
 }
 auto now=[](){
#ifdef __hexagon__
 if constexpr(Observe)return HAP_perf_get_time_us();else return uint64_t(0);
#else
 return uint64_t(0);
#endif
 };
 auto start=now();unsigned* rec=Observe?static_cast<unsigned*>(buffers[mathOut]):nullptr;
 for(unsigned i=0;i<2;++i){for(unsigned j=0;j<1024;++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(rec+header+i*1024,x[i],4096);}
 // Explicit bounded scratch. Tap tensors or receipt slots hold stage results
 // when available; observer-free NoTap uses local FP32 scratch only.
 float product[Tap||Value?1:1024],dot[Tap||Value?1:32],center[Tap||Value?1:1024];
 float* pr=Tap?y[1]:product;float* dt=Tap?y[2]:dot;float* ce=Tap?y[3]:center;
 auto math=now();if constexpr(Value)backwardrepair::dvalue(x[0],x[1],y[0]);else backwardrepair::softmax(x[0],x[1],y[0],pr,dt,ce);auto stop=now();
 for(unsigned i=0;i<mathOut;++i)for(unsigned j=0;j<counts[i];++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;
 if constexpr(Observe){std::memcpy(rec+2080,y[0],4096);if constexpr(!Value){std::memcpy(rec+3104,pr,4096);std::memcpy(rec+4128,dt,128);std::memcpy(rec+4160,ce,4096);}
  for(unsigned i=0;i<header;++i)rec[i]=0;
#ifdef __hexagon__
  rec[0]=0x48385000u+Value;
#else
  rec[0]=0x43505000u+Value;
#endif
  rec[1]=2048;rec[2]=1024;rec[3]=32;rec[4]=unsigned(stop-math);rec[5]=81;rec[6]=Value;rec[7]=unsigned(now()-start);rec[8]=Tap;
 }
 return QHPI_SUCCESS;
#endif
}
QHPI_Tensor_Signature_v1 ins[2]{},outs[6][5]{};QHPI_Kernel_v1 kernels[6]{};QHPI_OpInfo_v1 ops[6]{};
}
extern "C" __attribute__((visibility("default"))) Qnn_ErrorHandle_t HexaBackwardRepairInterfaceProvider(QnnOpPackage_Interface_t* p){using namespace repairpackage;if(!p)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;*p={};p->interfaceVersion={1,4,0};p->v1_4.init=init;p->v1_4.terminate=end;p->v1_4.getInfo=getInfo;p->v1_4.validateOpConfig=validate;p->v1_4.logInitialize=logInit;p->v1_4.logSetLevel=logLevel;p->v1_4.logTerminate=logEnd;return QNN_SUCCESS;}
extern "C" __attribute__((visibility("default"))) const char* qhpi_init(){using namespace repairpackage;
 for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 QHPI_Plugin_Function f[]={kernel<false,false,true>,kernel<false,false,false>,kernel<false,true,true>,kernel<false,true,false>,kernel<true,false,true>,kernel<true,false,false>};
 for(unsigned i=0;i<6;++i){for(auto& s:outs[i])s=ins[0];if(!(i&1))outs[i][outputCount[i]-1]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};auto& k=kernels[i];k.function_name=full[i];k.function=f[i];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=2;k.input_signature=ins;k.min_outputs=outputCount[i];k.output_signature=outs[i];ops[i].name=full[i];ops[i].num_kernels=1;ops[i].kernels=&k;}
 return qhpi_register_ops_v1(6,ops,name)==QHPI_SUCCESS?name:nullptr;
}
