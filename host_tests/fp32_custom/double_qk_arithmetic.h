#pragma once
// Diagnostic NT32 only: FP32 storage, separate ordered binary64 product/add.
namespace doubleqkarith {
__attribute__((noinline)) inline void nt(const float* q,const float* k,float* out) {
 for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c){
  double acc=0.0;
  for(unsigned j=0;j<32;++j){double product=double(q[r*32+j])*double(k[c*32+j]);acc=acc+product;}
  out[r*32+c]=float(acc);
 }
}
}
