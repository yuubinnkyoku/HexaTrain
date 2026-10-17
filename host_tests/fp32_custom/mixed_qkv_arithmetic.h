#pragma once
// Diagnostic NN64 only. FP32 tensor boundaries, increasing-q binary64 sum.
namespace mixedarith {
__attribute__((noinline)) inline void project(const float* a,const float* b,float* y) {
 for(unsigned r=0;r<32;++r)for(unsigned c=0;c<64;++c){
  double acc=0.0;
  for(unsigned q=0;q<64;++q){double product=double(a[r*64+q])*double(b[q*64+c]);acc=acc+product;}
  y[r*64+c]=float(acc);
 }
}
}
