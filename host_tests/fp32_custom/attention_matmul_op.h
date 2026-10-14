#pragma once
// Diagnostic G1 geometry only. Both heads share one launch; no batched ABI
// or production arithmetic is changed. Observer-free kernels never write receipts.
namespace attentioncustom {
constexpr unsigned words=8+4096+2048;
template<bool Observe> uint32_t dprob(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned calls=0;
 if(ni!=4||no!=(Observe?3u:2u))return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<4;++i)if(!mmcustom::matrix(in[i],32,32))return QHPI_ERROR_FATAL;
 for(unsigned h=0;h<2;++h)if(!mmcustom::matrix(out[h],32,32))return QHPI_ERROR_FATAL;
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[2])!=QHPI_INT32||lncustom::count(out[2])!=words)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!r)return QHPI_ERROR_FATAL;}
 const float* a[2]={lncustom::data(in[0]),lncustom::data(in[1])};const float* b[2]={lncustom::data(in[2]),lncustom::data(in[3])};float* y[2]={lncustom::data(out[0]),lncustom::data(out[1])};
 for(unsigned h=0;h<2;++h)if(!a[h]||!b[h]||!y[h])return QHPI_ERROR_FATAL;
 auto start=lncustom::now();for(unsigned i=0;i<4;++i){auto p=lncustom::data(in[i]);for(unsigned j=0;j<1024;++j)if(!std::isfinite(p[j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+8+i*1024,p,4096);}
 auto mathStart=lncustom::now();for(unsigned h=0;h<2;++h)mmcustom::ordered<32,32,32,false,true>(a[h],b[h],y[h]);auto mathEnd=lncustom::now();
 for(unsigned h=0;h<2;++h){for(unsigned j=0;j<1024;++j)if(!std::isfinite(y[h][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+4104+h*1024,y[h],4096);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x4838310bu;
 #else
 r[0]=0x4350550bu;
 #endif
 r[1]=++calls;r[2]=4096;r[3]=2048;r[4]=mathEnd-mathStart;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[4]={},outs[4]={};QHPI_Kernel_v1 kernels[10]={};QHPI_OpInfo_v1 ops[10]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& s:outs)s=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 const char* names[]={"HexaFp32::MatMulDProbabilities","HexaFp32::MatMulDProbabilitiesNoReceipt","HexaFp32::LnForwardNoReceipt","HexaFp32::LnBackwardNoReceipt","HexaFp32::AddNoReceipt","HexaFp32::MatMulDContextNoReceipt","HexaFp32::MatMulDWoNoReceipt","HexaFp32::HeadSplitNoReceipt","HexaFp32::BroadcastMultiplyNoReceipt","HexaFp32::CopyNoReceipt"};
 QHPI_Plugin_Function fn[]={dprob<true>,dprob<false>,lncustom::forwardKernel<false>,lncustom::backward<false,false>,kernel<1,false>,mmcustom::kernel<false,false>,mmcustom::kernel<true,false>,headcustom::kernelImpl<false>,broadcastcustom::kernelImpl<false>,kernel<0,false>};
 const unsigned ni[]={4,4,4,4,2,2,2,1,4,1},no[]={3,2,3,3,1,1,1,2,2,1};
 // Only the observed dProb operation has an INT32 output; all free signatures
 // are FP32, including xhat/inverse. Keep their signature storage separate.
 static QHPI_Tensor_Signature_v1 freeOut[3];for(auto& s:freeOut)s=ins[0];
 for(unsigned j=0;j<10;++j){auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=ni[j];k.input_signature=ins;k.min_outputs=no[j];k.output_signature=j==0?outs:freeOut;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
 return qhpi_register_ops_v1(10,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
