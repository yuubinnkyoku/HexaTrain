#pragma once
// Bounded paired forward specialization. Backward ABI/counters stay unchanged.
namespace forwardscale {
constexpr unsigned receiptWords=4105;
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned serial=0;if(ni!=3||no!=2+unsigned(Observe)||!mmcustom::matrix(in[2],1,1))return QHPI_ERROR_FATAL;
 const float* x[2];float* y[2];auto s=lncustom::data(in[2]);if(!s||!std::isfinite(*s))return QHPI_ERROR_FATAL;
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[2])!=QHPI_INT32||lncustom::count(out[2])!=receiptWords)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!r)return QHPI_ERROR_FATAL;}
 for(unsigned h=0;h<2;++h){if(!mmcustom::matrix(in[h],32,32)||!mmcustom::matrix(out[h],32,32))return QHPI_ERROR_FATAL;x[h]=lncustom::data(in[h]);y[h]=lncustom::data(out[h]);if(!x[h]||!y[h])return QHPI_ERROR_FATAL;}
 auto overlap=[](const void* a,unsigned na,const void* b,unsigned nb){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+nb&&y<x+na;};
 for(unsigned h=0;h<2;++h){for(unsigned j=0;j<2;++j)if(overlap(y[h],4096,x[j],4096)||(j!=h&&overlap(y[h],4096,y[j],4096)))return QHPI_ERROR_FATAL;if(overlap(y[h],4096,s,4))return QHPI_ERROR_FATAL;if constexpr(Observe)if(overlap(r,receiptWords*4,x[h],4096)||overlap(r,receiptWords*4,y[h],4096))return QHPI_ERROR_FATAL;}
 if constexpr(Observe)if(overlap(r,receiptWords*4,s,4))return QHPI_ERROR_FATAL;
 auto start=lncustom::now();for(unsigned h=0;h<2;++h){for(unsigned i=0;i<1024;++i)if(!std::isfinite(x[h][i]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+8+h*1024,x[h],4096);}if constexpr(Observe)std::memcpy(r+2056,s,4);
 auto math=lncustom::now();for(unsigned h=0;h<2;++h)for(unsigned i=0;i<1024;++i)y[h][i]=elementProduct(x[h][i],*s);auto end=lncustom::now();
 for(unsigned h=0;h<2;++h){for(unsigned i=0;i<1024;++i)if(!std::isfinite(y[h][i]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+2057+h*1024,y[h],4096);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383118u;
 #else
 r[0]=0x43505518u;
 #endif
 r[1]=++serial;r[2]=2049;r[3]=2048;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[3]={},outs[3]={};QHPI_Kernel_v1 kernels[2]={};QHPI_OpInfo_v1 ops[2]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};const char* names[]={"HexaFp32::ForwardScoreScale","HexaFp32::ForwardScoreScaleNoReceipt"};for(unsigned j=0;j<2;++j){auto& k=kernels[j];k.function_name=names[j];k.function=j?kernel<false>:kernel<true>;k.resources=QHPI_RESOURCE_HVX;k.min_inputs=3;k.input_signature=ins;k.min_outputs=3-j;k.output_signature=outs;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;}
}
