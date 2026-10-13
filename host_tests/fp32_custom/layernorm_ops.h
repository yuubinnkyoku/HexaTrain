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
bool receipt(QHPI_Tensor* output,unsigned kind,unsigned ni,const QHPI_Tensor* const* inputs,unsigned& serial,unsigned extra=0){
 unsigned size=8;for(unsigned i=0;i<ni;++i)size+=count(inputs[i]);
 if(count(output)!=size+extra || qhpi_tensor_type(output)!=QHPI_INT32)return false;
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
 if(!receipt(out[3],4,ni,in,serial,n+rows))return QHPI_ERROR_FATAL;
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
 auto metadata=static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]));unsigned offset=8+n+2*d+1;
 std::memcpy(metadata+offset,h,n*4);std::memcpy(metadata+offset+n,inv,rows*4);metadata[4]=unsigned(now()-started);
 return QHPI_SUCCESS;
}
template<bool Fused> uint32_t backward(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 if(ni!=(Fused?5u:4u)||no!=4)return QHPI_ERROR_FATAL;
 unsigned d=count(in[1]),n=count(in[0]);if(d<8||d>128||!n||n%d)return QHPI_ERROR_FATAL;unsigned rows=n/d;
 if(rows>32||count(in[2])!=n||count(in[3])!=rows||count(out[0])!=n||count(out[1])!=d||count(out[2])!=d)return QHPI_ERROR_FATAL;
 if constexpr(Fused)if(count(in[4])!=n)return QHPI_ERROR_FATAL;
 const bool observedFused=Fused&&count(out[3])==8+2*n+d+rows+3*n;
 if(!receipt(out[3],Fused?6:5,ni,in,serial,n+(observedFused?n:0)))return QHPI_ERROR_FATAL;
 auto started=now();const auto dy=data(in[0]),gamma=data(in[1]),h=data(in[2]),inv=data(in[3]);
 auto dx=data(out[0]),dg=data(out[1]),db=data(out[2]);Sum sumsG[128],sumsB[128];
 auto metadata=static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]));unsigned offset=8;for(unsigned j=0;j<ni;++j)offset+=count(in[j]);
 for(unsigned r=0;r<rows;++r){Sum sum,prod;
  for(unsigned j=0;j<d;++j){unsigned i=r*d+j;float dh=dy[i]*gamma[j];sum.add(dh);prod.add(dh*h[i]);sumsG[j].add(dy[i]*h[i]);sumsB[j].add(dy[i]);}
  for(unsigned j=0;j<d;++j){unsigned i=r*d+j;float value=inv[r]*(dy[i]*gamma[j]-sum.value/float(d)-h[i]*prod.value/float(d));
   std::memcpy(metadata+offset+i,&value,4);if constexpr(Fused)dx[i]=value+data(in[4])[i];else dx[i]=value;
   if(!std::isfinite(dx[i]))return QHPI_ERROR_FATAL;}
 }
 for(unsigned j=0;j<d;++j){dg[j]=sumsG[j].value;db[j]=sumsB[j].value;if(!std::isfinite(dg[j])||!std::isfinite(db[j]))return QHPI_ERROR_FATAL;}
 if(observedFused)std::memcpy(metadata+offset+n,dx,4*n);
 metadata[4]=unsigned(now()-started);return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 inSigs[5]={};
QHPI_Tensor_Signature_v1 outSigs[4]={};
QHPI_Kernel_v1 kernels[3]{};QHPI_OpInfo_v1 ops[3]{};
bool registerOps(){
 for(auto& s:inSigs)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 for(unsigned j=0;j<4;++j)outSigs[j]={j==3?QHPI_INT32:QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 const char* names[]={"HexaFp32::LnForward","HexaFp32::LnBackward","HexaFp32::LnResidual"};
 QHPI_Plugin_Function functions[]={forward,backward<false>,backward<true>};
 for(unsigned j=0;j<3;++j){auto& k=kernels[j];k.function_name=names[j];k.function=functions[j];k.resources=QHPI_RESOURCE_HVX;
  k.min_inputs=j==2?5:4;k.input_signature=inSigs;k.min_outputs=4;k.output_signature=outSigs;
  ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
 return qhpi_register_ops_v1(3,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
