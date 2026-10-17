#pragma once
#include <cmath>
// Research arithmetic only; no QHPI registration or production caller.
// Output tensors remain float, including saved h/inverse.
namespace selective {
struct Sum {float s=0,c=0;void add(float x){float y=x-c,t=s+y;c=(t-s)-y;s=t;}};
template<unsigned B> void ln(const float* x,const float* gamma,const float* beta,float* y,float* h,float* inverse){
 for(unsigned r=0;r<32;++r){double center[64],var=0,rms=0,inv=0;
  if constexpr(B&1){double mean=0;for(unsigned j=0;j<64;++j)mean+=double(x[r*64+j]);mean/=64;
   for(unsigned j=0;j<64;++j){center[j]=double(x[r*64+j])-mean;var+=center[j]*center[j];}var/=64;
   if constexpr(!(B&2)){rms=double(float(std::sqrt(var)));for(auto& v:center)v=double(float(v));}
  }else{float anchor=x[r*64];Sum mean;for(unsigned j=0;j<64;++j)mean.add(x[r*64+j]-anchor);float delta=mean.s/64.f,amax=0;
   for(unsigned j=0;j<64;++j){float v=(x[r*64+j]-anchor)-delta;center[j]=double(v);if(std::fabs(v)>amax)amax=std::fabs(v);}
   Sum sq;if(amax>0)for(double v:center){float q=float(v)/amax;sq.add(q*q);}rms=double(amax*std::sqrt(sq.s/64.f));
  }
  if constexpr(B&2){if constexpr(B&1)inv=1/std::sqrt(var+double(1e-5f));else inv=1/std::hypot(rms,std::sqrt(double(1e-5f)));}
  else inv=double(1.f/std::hypot(float(rms),std::sqrt(1e-5f)));
  inverse[r]=float(inv);
  for(unsigned j=0;j<64;++j){unsigned i=r*64+j;double normalized;
   if constexpr(B&2)normalized=center[j]*inv;else normalized=double(float(center[j])*float(inv));
   h[i]=float(normalized);
   if constexpr(B&4)y[i]=float(double(gamma[j])*normalized+double(beta[j]));
   else {float product=gamma[j]*h[i];y[i]=product+beta[j];}
  }
 }
}
}
