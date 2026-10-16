#pragma once
// Diagnostic only. Same-kind graph executions must be synchronous.
// QK reuses the existing NT dot; the specialization owns a separate counter.
namespace qkscalecustom {
constexpr unsigned qkWords=6152,scaleWords=8201;
template<bool Observe> uint32_t scale(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned calls=0;
 if(ni!=5||no!=4+unsigned(Observe)||!mmcustom::matrix(in[4],1,1))return QHPI_ERROR_FATAL;
 const float* x[4];float* y[4];auto s=lncustom::data(in[4]);
 if(!s||!std::isfinite(*s))return QHPI_ERROR_FATAL;
 for(unsigned j=0;j<4;++j){if(!mmcustom::matrix(in[j],32,32)||!mmcustom::matrix(out[j],32,32))return QHPI_ERROR_FATAL;x[j]=lncustom::data(in[j]);y[j]=lncustom::data(out[j]);if(!x[j]||!y[j])return QHPI_ERROR_FATAL;}
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[4])!=QHPI_INT32||!mmcustom::matrix(in[4],1,1)||lncustom::count(out[4])!=scaleWords)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[4]));if(!r)return QHPI_ERROR_FATAL;}
 // No in-place or overlapping outputs/receipt are part of this ABI.
 auto overlap=[](const void* a,unsigned na,const void* b,unsigned nb){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+nb&&y<x+na;};
 for(unsigned j=0;j<4;++j){for(unsigned k=0;k<4;++k){if(overlap(y[j],4096,x[k],4096)||(k!=j&&overlap(y[j],4096,y[k],4096)))return QHPI_ERROR_FATAL;}if(overlap(y[j],4096,s,4))return QHPI_ERROR_FATAL;if constexpr(Observe)if(overlap(r,scaleWords*4,x[j],4096)||overlap(r,scaleWords*4,y[j],4096))return QHPI_ERROR_FATAL;}
 if constexpr(Observe)if(overlap(r,scaleWords*4,s,4))return QHPI_ERROR_FATAL;
 auto start=lncustom::now();for(unsigned j=0;j<4;++j){for(unsigned i=0;i<1024;++i)if(!std::isfinite(x[j][i]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+8+j*1024,x[j],4096);}
 if constexpr(Observe)std::memcpy(r+4104,s,4);
 auto math=lncustom::now();for(unsigned j=0;j<4;++j)for(unsigned i=0;i<1024;++i)y[j][i]=elementProduct(x[j][i],*s);auto end=lncustom::now();
 for(unsigned j=0;j<4;++j){for(unsigned i=0;i<1024;++i)if(!std::isfinite(y[j][i]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+4105+j*1024,y[j],4096);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383117u;
 #else
 r[0]=0x43505517u;
 #endif
 r[1]=++calls;r[2]=4097;r[3]=4096;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[5]={},outs[4][5]={};QHPI_Kernel_v1 kernels[4]={};QHPI_OpInfo_v1 ops[4]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 const char* names[]={"HexaFp32::ForwardQKRaw","HexaFp32::ForwardQKRawNoReceipt","HexaFp32::BackwardScoreScale","HexaFp32::BackwardScoreScaleNoReceipt"};
 QHPI_Plugin_Function fn[]={attentioncustom::dprob<true,0x16>,attentioncustom::dprob<false,0x16>,scale<true>,scale<false>};
 for(unsigned j=0;j<4;++j){unsigned ny=j<2?2:4;for(auto& s:outs[j])s=ins[0];if(j%2==0)outs[j][ny]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=j<2?4:5;k.input_signature=ins;k.min_outputs=ny+(j%2==0);k.output_signature=outs[j];ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
 return qhpi_register_ops_v1(4,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
