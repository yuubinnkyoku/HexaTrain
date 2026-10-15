#pragma once
#include "closure_contract.h"
namespace closurecustom {
using namespace closurecontract;
template<Kind K,bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 // One paired invocation per kind per synchronous graph execute. No shared
 // counter among independent Q/K/V or gradient branches. Concurrent same-kind
 // graphs are deliberately outside this diagnostic contract.
 static unsigned calls=0;constexpr auto c=contracts[K];
 if(ni!=c.inputs||no!=c.outputs+unsigned(Observe))return QHPI_ERROR_FATAL;
 const float* x[c.inputs];float* y[c.outputs];unsigned sizes[c.inputs];
 for(unsigned i=0;i<c.inputs;++i){unsigned r=32,n=32;
  if constexpr(K==FFN2){r=i?128:32;n=i?64:128;}
  if constexpr(K==QKVHeads)n=64;
  if(!mmcustom::matrix(in[i],r,n))return QHPI_ERROR_FATAL;
  sizes[i]=r*n;x[i]=lncustom::data(in[i]);if(!x[i])return QHPI_ERROR_FATAL;
 }
 constexpr unsigned yn=K==FFN2?2048:1024;
 for(unsigned i=0;i<c.outputs;++i){if(!mmcustom::matrix(out[i],32,K==FFN2?64:32))return QHPI_ERROR_FATAL;y[i]=lncustom::data(out[i]);if(!y[i])return QHPI_ERROR_FATAL;}
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[c.outputs])!=QHPI_INT32||lncustom::count(out[c.outputs])!=receiptWords(K))return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[c.outputs]));if(!r)return QHPI_ERROR_FATAL;}
 auto start=lncustom::now();unsigned at=8;
 for(unsigned i=0;i<c.inputs;++i){for(unsigned j=0;j<sizes[i];++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+at,x[i],sizes[i]*4);at+=sizes[i];}
 auto math=lncustom::now();
 if constexpr(K==RawDQDK)for(unsigned h=0;h<2;++h){mmcustom::ordered<32,32,32,false,false>(x[h],x[2+h],y[2*h]);mmcustom::ordered<32,32,32,true,false>(x[h],x[4+h],y[2*h+1]);}
 if constexpr(K==FFN2)mmcustom::ordered<32,64,128,false,false>(x[0],x[1],y[0]);
 if constexpr(K==QKVHeads)for(unsigned j=0;j<3;++j)headcustom::copyHeads(x[j],y[j*2],y[j*2+1]);
 auto end=lncustom::now();for(unsigned i=0;i<c.outputs;++i){for(unsigned j=0;j<yn;++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+at+i*yn,y[i],yn*4);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383100u+c.marker;
 #else
 r[0]=0x43505500u+c.marker;
 #endif
 r[1]=++calls;r[2]=c.inputWords;r[3]=c.outputWords;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[6]={},outs[3][7]={};QHPI_Kernel_v1 kernels[6]={};QHPI_OpInfo_v1 ops[6]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& a:outs)for(auto& s:a)s=ins[0];
 static QHPI_Tensor_Signature_v1 freeOut[6];for(auto& s:freeOut)s=ins[0];
 const char* names[]={"HexaFp32::RawDQDK","HexaFp32::RawDQDKNoReceipt","HexaFp32::FFN2","HexaFp32::FFN2NoReceipt","HexaFp32::QKVHeads","HexaFp32::QKVHeadsNoReceipt"};
 QHPI_Plugin_Function fn[]={kernel<RawDQDK,true>,kernel<RawDQDK,false>,kernel<FFN2,true>,kernel<FFN2,false>,kernel<QKVHeads,true>,kernel<QKVHeads,false>};
 for(unsigned j=0;j<6;++j){auto c=contracts[j/2];outs[j/2][c.outputs]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=c.inputs;k.input_signature=ins;k.min_outputs=c.outputs+(j%2?0:1);k.output_signature=j%2?freeOut:outs[j/2];ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
 return qhpi_register_ops_v1(6,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
