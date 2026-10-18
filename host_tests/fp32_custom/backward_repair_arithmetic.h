#pragma once
// Diagnostic contracts only. No scale. Separate IEEE FP32 rounding at each
// source primitive boundary, including when executed inside one fused op.
namespace backwardrepair {
__attribute__((noinline)) inline void softmax(const float* p,const float* dp,float* y,
 float* product,float* dot,float* centered){
 for(unsigned i=0;i<1024;++i){volatile float v=dp[i]*p[i];product[i]=v;}
 for(unsigned r=0;r<32;++r){float acc=0.f;for(unsigned c=0;c<32;++c){volatile float v=acc+product[r*32+c];acc=v;}dot[r]=acc;}
 for(unsigned i=0;i<1024;++i){volatile float v=dp[i]-dot[i/32];centered[i]=v;}
 for(unsigned i=0;i<1024;++i){volatile float v=p[i]*centered[i];y[i]=v;}
}
// Same ordered product/add contract as existing mmcustom::ordered; TN32 index
// specialization. The existing implementation and its ABI remain unchanged.
__attribute__((noinline)) inline void dvalue(const float* p,const float* dc,float* y){
 for(unsigned i=0;i<32;++i)for(unsigned j=0;j<32;++j){float acc=0.f;
  for(unsigned t=0;t<32;++t){volatile float product=p[t*32+i]*dc[t*32+j];volatile float sum=acc+product;acc=sum;}
  y[i*32+j]=acc;
 }
}
}
