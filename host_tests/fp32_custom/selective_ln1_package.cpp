// Separate diagnostic package. Existing HexaFp32 registry/ABI is untouched.
#include <cstring>
#include <QnnOpPackage.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/core/qhpi.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#endif
// Reuse the unchanged baseline body while keeping its globals out of the old
// package's namespace/symbol ownership when both libraries are loaded.
#define lncustom s2lnbaseline
#include "layernorm_ops.h"
#undef lncustom
namespace lncustom=s2lnbaseline;
#include "selective_ln1_arithmetic.h"
namespace s2package {
constexpr const char* name="HexaSelectiveLn1";
constexpr unsigned receiptWords=6313;
const char* names[]={"Baseline","BaselineNoReceipt","S2","S2NoReceipt"};
const char* full[]={"HexaSelectiveLn1::Baseline","HexaSelectiveLn1::BaselineNoReceipt","HexaSelectiveLn1::S2","HexaSelectiveLn1::S2NoReceipt"};
Qnn_ApiVersion_t apiVersion=QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t info={name,nullptr,nullptr,4,nullptr,0,QNN_SDK_BUILD_ID,&apiVersion,nullptr,nullptr,{0}};
bool initialized=false;
Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t){initialized=true;return QNN_SUCCESS;}
Qnn_ErrorHandle_t end(){initialized=false;return QNN_SUCCESS;}
Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** p){if(!p||!initialized)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;info.operationNames=names;*p=&info;return QNN_SUCCESS;}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t c){if(c.version!=QNN_OPCONFIG_VERSION_1||!c.v1.packageName||!c.v1.typeName||std::strcmp(c.v1.packageName,name)||c.v1.numOfParams||c.v1.numOfInputs!=4)return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
 for(unsigned i=0;i<4;++i)if(!std::strcmp(c.v1.typeName,names[i]))return c.v1.numOfOutputs==(i%2?3u:4u)?QNN_SUCCESS:QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t,QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t logEnd(){return QNN_SUCCESS;}
bool shape(const QHPI_Tensor* t,unsigned rows,unsigned cols){auto s=qhpi_tensor_shape(t);return s.rank==4&&s.dims[0]==1&&s.dims[1]==1&&s.dims[2]==rows&&s.dims[3]==cols&&qhpi_tensor_type(t)==QHPI_FLOAT32;}
template<bool S2,bool Observe> uint32_t kernel(QHPI_RuntimeHandle* rt,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
#if !defined(__hexagon__) && !defined(SELECTIVE_LN_HOST_KERNEL_TEST)
 return QHPI_ERROR_FATAL; // Host package is preparation only, never CPU fallback.
#else
 if(ni!=4||no!=(Observe?4u:3u)||!shape(in[0],32,64)||!shape(in[1],1,64)||!shape(in[2],1,64)||!shape(in[3],1,1)||!shape(out[0],32,64)||!shape(out[1],32,64)||!shape(out[2],32,1))return QHPI_ERROR_FATAL;
 const float *x=lncustom::data(in[0]),*ga=lncustom::data(in[1]),*be=lncustom::data(in[2]),*eps=lncustom::data(in[3]);
 if(!x||!ga||!be||!eps||eps[0]!=1e-5f)return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<ni;++i)for(unsigned j=0;j<lncustom::count(in[i]);++j)if(!std::isfinite(lncustom::data(in[i])[j]))return QHPI_ERROR_FATAL;
 unsigned* receipt=nullptr;static unsigned calls=0;
 if constexpr(Observe){if(lncustom::count(out[3])!=receiptWords||qhpi_tensor_type(out[3])!=QHPI_INT32)return QHPI_ERROR_FATAL;receipt=static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]));if(!receipt)return QHPI_ERROR_FATAL;}
 for(unsigned i=0;i<3;++i)if(!lncustom::data(out[i]))return QHPI_ERROR_FATAL;
 auto start=lncustom::now();unsigned at=8;
 if constexpr(Observe)for(unsigned i=0;i<ni;++i){unsigned n=lncustom::count(in[i]);std::memcpy(receipt+at,lncustom::data(in[i]),n*4);at+=n;}
 auto math=lncustom::now();
 if constexpr(S2)selective::ln<2>(x,ga,be,lncustom::data(out[0]),lncustom::data(out[1]),lncustom::data(out[2]));
 else {if(lncustom::forwardKernel<false>(rt,3,out,4,in)!=QHPI_SUCCESS)return QHPI_ERROR_FATAL;}
 auto stop=lncustom::now();
 for(unsigned i=0;i<3;++i){unsigned n=lncustom::count(out[i]);for(unsigned j=0;j<n;++j)if(!std::isfinite(lncustom::data(out[i])[j]))return QHPI_ERROR_FATAL;if constexpr(Observe){std::memcpy(receipt+at,lncustom::data(out[i]),n*4);at+=n;}}
 if constexpr(Observe){
#ifdef __hexagon__
  receipt[0]=0x48383200u+(S2?2:0);
#else
  receipt[0]=0x43505500u+(S2?2:0);
#endif
  receipt[1]=++calls;receipt[2]=2048;receipt[3]=4;receipt[4]=unsigned(stop-math);receipt[5]=81;receipt[6]=64;receipt[7]=unsigned(lncustom::now()-start);
 }
 return QHPI_SUCCESS;
#endif
}
QHPI_Tensor_Signature_v1 ins[4]{},outs[4]{};
QHPI_Kernel_v1 kernels[4]{};QHPI_OpInfo_v1 ops[4]{};
}
extern "C" Qnn_ErrorHandle_t HexaSelectiveLn1InterfaceProvider(QnnOpPackage_Interface_t* p){using namespace s2package;if(!p)return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;*p={};p->interfaceVersion={1,4,0};p->v1_4.init=init;p->v1_4.terminate=end;p->v1_4.getInfo=getInfo;p->v1_4.validateOpConfig=validate;p->v1_4.logInitialize=logInit;p->v1_4.logSetLevel=logLevel;p->v1_4.logTerminate=logEnd;return QNN_SUCCESS;}
extern "C" const char* qhpi_init(){using namespace s2package;for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& s:outs)s=ins[0];outs[3]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 QHPI_Plugin_Function f[]={kernel<false,true>,kernel<false,false>,kernel<true,true>,kernel<true,false>};for(unsigned i=0;i<4;++i){auto& k=kernels[i];k.function_name=full[i];k.function=f[i];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=ins;k.min_outputs=i%2?3:4;k.output_signature=outs;ops[i].name=full[i];ops[i].num_kernels=1;ops[i].kernels=&k;}return qhpi_register_ops_v1(4,ops,name)==QHPI_SUCCESS?name:nullptr;}
