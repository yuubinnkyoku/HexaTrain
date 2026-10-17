// Actual Selective LN1 arithmetic compiled for V81; no QHPI registration.
#include "selective_ln1_arithmetic.h"
#define WRAP(ID,B) extern "C" __attribute__((noinline)) void selective_ln_##ID(const float* x,const float* g,const float* b,float* y,float* h,float* inv){selective::ln<B>(x,g,b,y,h,inv);}
WRAP(S0,0) WRAP(S1,1) WRAP(S2,2) WRAP(S3,4)
WRAP(S4,3) WRAP(S5,5) WRAP(S6,6) WRAP(S7,7)
extern "C" __attribute__((noinline)) void selective_qk_nn64(const float* a,const float* b,float* out){
 for(unsigned r=0;r<32;++r)for(unsigned j=0;j<64;++j){double s=0;for(unsigned q=0;q<64;++q)s+=double(a[r*64+q])*double(b[q*64+j]);out[r*64+j]=float(s);}
}
extern "C" __attribute__((noinline)) void selective_qk_nt32(const float* a,const float* b,float* out){
 for(unsigned r=0;r<32;++r)for(unsigned j=0;j<32;++j){double s=0;for(unsigned q=0;q<32;++q)s+=double(a[r*32+q])*double(b[j*32+q]);out[r*32+j]=float(s);}
}
