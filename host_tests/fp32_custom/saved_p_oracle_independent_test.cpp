// Hand-solvable asymmetric transpose and exact binary-rational derivative tests.
#include "saved_p_backward_reference_lib.h"
#include <cstdio>
int main() {
  using namespace phonelm::saved_p_backward;
  std::vector<float> p(1024),dp(1024),dc(1024);
  for(unsigned r=0;r<32;++r) {
    p[r*32+(r+1)%32]=1;
    for(unsigned c=0;c<32;++c){dp[r*32+c]=float(r*32+c);dc[r*32+c]=float(r*32+c);}
  }
  auto ds=SoftmaxBackwardDouble(p,dp),dv=DValueDouble(p,dc);
  auto sf=SoftmaxBackwardOrderedFp32(p,dp),vf=DValueOrderedFp32(p,dc);
  for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c) {
    unsigned i=r*32+c;double expected=double(((r+31)%32)*32+c);
    if(ds[i]!=0||sf[i]!=0||dv[i]!=expected||vf[i]!=expected)return 1;
  }
  std::fill(p.begin(),p.end(),1.f/32.f);
  ds=SoftmaxBackwardDouble(p,dp);sf=SoftmaxBackwardOrderedFp32(p,dp);
  for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c) {
    double expected=(double(c)-15.5)/32.;unsigned i=r*32+c;
    if(ds[i]!=expected||sf[i]!=expected)return 2;
  }
  std::puts("saved_p_independent_oracle=PASS asymmetric_P_transpose=1 exact_onehot_derivative=1 exact_uniform_derivative=1 double_and_FP32=1 device_calls=0");
  return 0;
}
