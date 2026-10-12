// Diagnostic QHPI extension; no production LayerNorm or optimizer changes.
#pragma once
#include <cmath>
namespace lncustom {
struct Sum {float value=0, correction=0;void add(float x){float y=x-correction,t=value+y;correction=(t-value)-y;value=t;}};
unsigned count(const QHPI_Tensor* t){auto s=qhpi_tensor_shape(t);unsigned n=1;for(unsigned j=0;j<s.rank;++j)n*=s.dims[j];return n;}
float* data(const QHPI_Tensor* t){return static_cast<float*>(qhpi_tensor_raw_data(t));}
uint64_t now(){
#ifdef __hexagon__
 return HAP_perf_get_time_us();
#else
 return 0;
#endif
}
bool receipt(QHPI_Tensor* output,unsigned kind,unsigned ni,const QHPI_Tensor* const* inputs,unsigned& serial){
 unsigned size=8;for(unsigned i=0;i<ni;++i)size+=count(inputs[i]);
 if(count(output)!=size || qhpi_tensor_type(output)!=QHPI_INT32)return false;
 auto p=static_cast<unsigned*>(qhpi_tensor_raw_data(output));
#ifdef __hexagon__
 p[0]=0x48383100u+kind;
#else
 p[0]=0x43505500u+kind;
#endif
 p[1]=++serial;p[2]=count(inputs[0]);p[3]=ni;p[4]=0;p[5]=81;p[6]=32;p[7]=0;
 unsigned at=8;for(unsigned i=0;i<ni;++i){std::memcpy(p+at,data(inputs[i]),count(inputs[i])*4);at+=count(inputs[i]);}
 return true;
}
unsigned serial=0;
uint32_t forward(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 if(ni!=4||no!=4)return QHPI_ERROR_FATAL;
 const unsigned d=count(in[1]),n=count(in[0]);if(d<8||d>128||!n||n%d)return QHPI_ERROR_FATAL;const unsigned rows=n/d;
 if(rows>32||count(in[2])!=d||count(in[3])!=1 ||
    count(out[0])!=n||count(out[1])!=n||count(out[2])!=rows)return QHPI_ERROR_FATAL;
 for(unsigned j=0;j<ni;++j)if(qhpi_tensor_type(in[j])!=QHPI_FLOAT32)return QHPI_ERROR_FATAL;
 if(!receipt(out[3],4,ni,in,serial))return QHPI_ERROR_FATAL;
 auto started=now();float eps=data(in[3])[0];if(!std::isfinite(eps)||eps<=0)return QHPI_ERROR_FATAL;
 const auto x=data(in[0]),gamma=data(in[1]),beta=data(in[2]);auto y=data(out[0]),h=data(out[1]),inv=data(out[2]);
 float centered[128];
 for(unsigned r=0;r<rows;++r){
  float anchor=x[r*d];Sum mean;for(unsigned j=0;j<d;++j)mean.add(x[r*d+j]-anchor);
  float delta=mean.value/float(d),a=0;
  for(unsigned j=0;j<d;++j){centered[j]=(x[r*d+j]-anchor)-delta;a=std::fmax(a,std::fabs(centered[j]));}
  Sum squares;if(a>0)for(unsigned j=0;j<d;++j){float q=centered[j]/a;squares.add(q*q);}
  float rms=a*std::sqrt(squares.value/float(d));
  float denominator=std::hypot(rms,std::sqrt(eps));
  inv[r]=1.f/denominator;
  for(unsigned j=0;j<d;++j){unsigned i=r*d+j;h[i]=centered[j]*inv[r];y[i]=gamma[j]*h[i]+beta[j];
   if(!std::isfinite(h[i])||!std::isfinite(y[i]))return QHPI_ERROR_FATAL;}
  if(!std::isfinite(inv[r]))return QHPI_ERROR_FATAL;
 }
 static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]))[4]=unsigned(now()-started);
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 inSigs[5]={};
QHPI_Tensor_Signature_v1 outSigs[4]={};
QHPI_Kernel_v1 kernel{};QHPI_OpInfo_v1 op{};
bool registerOps(){
 for(auto& s:inSigs)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 for(unsigned j=0;j<4;++j)outSigs[j]={j==3?QHPI_INT32:QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 kernel.function_name="HexaFp32::LnForward";kernel.function=forward;kernel.resources=QHPI_RESOURCE_HVX;
 kernel.min_inputs=4;kernel.input_signature=inSigs;kernel.min_outputs=4;kernel.output_signature=outSigs;
 op.name="HexaFp32::LnForward";op.num_kernels=1;op.kernels=&kernel;
 return qhpi_register_ops_v1(1,&op,"HexaFp32")==QHPI_SUCCESS;
}
}
