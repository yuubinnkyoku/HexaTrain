#pragma once
// One bounded diagnostic strategy: source beta=1, stable max shift, pinned
// toolchain expf, increasing-column SF32 sum and SF32 divide. No clipping,
// epsilon, renormalization pass or special causal-mask arithmetic.
namespace forwardsoftmax {
constexpr unsigned receiptWords=8328;
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned serial=0;if(ni!=2||no!=2+unsigned(Observe))return QHPI_ERROR_FATAL;const float* x[2];float* y[2];unsigned* r=nullptr;
 for(unsigned h=0;h<2;++h){if(!mmcustom::matrix(in[h],32,32)||!mmcustom::matrix(out[h],32,32))return QHPI_ERROR_FATAL;x[h]=lncustom::data(in[h]);y[h]=lncustom::data(out[h]);if(!x[h]||!y[h])return QHPI_ERROR_FATAL;}
 if constexpr(Observe){if(qhpi_tensor_type(out[2])!=QHPI_INT32||lncustom::count(out[2])!=receiptWords)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!r)return QHPI_ERROR_FATAL;}
 auto overlap=[](const void* a,unsigned na,const void* b,unsigned nb){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+nb&&y<x+na;};
 for(unsigned h=0;h<2;++h){for(unsigned j=0;j<2;++j)if(overlap(y[h],4096,x[j],4096))return QHPI_ERROR_FATAL;if(overlap(y[h],4096,y[1-h],4096))return QHPI_ERROR_FATAL;if constexpr(Observe)if(overlap(r,receiptWords*4,x[h],4096)||overlap(r,receiptWords*4,y[h],4096))return QHPI_ERROR_FATAL;for(unsigned i=0;i<1024;++i)if(!std::isfinite(x[h][i]))return QHPI_ERROR_FATAL;}
 auto start=lncustom::now();if constexpr(Observe)for(unsigned h=0;h<2;++h)std::memcpy(r+8+h*1024,x[h],4096);auto math=lncustom::now();
 for(unsigned h=0;h<2;++h)for(unsigned row=0;row<32;++row){unsigned base=row*32;float mx=x[h][base];for(unsigned c=1;c<32;++c)if(x[h][base+c]>mx)mx=x[h][base+c];float exponent[32];float sum=+0.f;
  if constexpr(Observe)std::memcpy(r+2056+h*32+row,&mx,4);
  for(unsigned c=0;c<32;++c){volatile float shifted=x[h][base+c]-mx;float e=::expf(shifted);if(!std::isfinite(e)||e<0)return QHPI_ERROR_FATAL;exponent[c]=e;volatile float next=sum+e;sum=next;if constexpr(Observe){std::memcpy(r+2120+h*1024+base+c,const_cast<const float*>(&shifted),4);std::memcpy(r+4168+h*1024+base+c,&e,4);}}
  if(!std::isfinite(sum)||sum<=0)return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+6216+h*32+row,&sum,4);
  for(unsigned c=0;c<32;++c){volatile float p=exponent[c]/sum;if(!std::isfinite(p)||p<0)return QHPI_ERROR_FATAL;y[h][base+c]=p;}
 }
 auto end=lncustom::now();if constexpr(Observe){for(unsigned h=0;h<2;++h)std::memcpy(r+6280+h*1024,y[h],4096);
 #ifdef __hexagon__
 r[0]=0x4838311au;
 #else
 r[0]=0x4350551au;
 #endif
 r[1]=++serial;r[2]=2048;r[3]=2048;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[2]={},outs[3]={};QHPI_Kernel_v1 kernels[2]={};QHPI_OpInfo_v1 ops[2]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};const char* names[]={"HexaFp32::ForwardSoftmax","HexaFp32::ForwardSoftmaxNoReceipt"};for(unsigned j=0;j<2;++j){auto& k=kernels[j];k.function_name=names[j];k.function=j?kernel<false>:kernel<true>;k.resources=QHPI_RESOURCE_HVX;k.min_inputs=2;k.input_signature=ins;k.min_outputs=3-j;k.output_signature=outs;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;}
}
