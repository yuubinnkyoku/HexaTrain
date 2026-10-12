#include "hvx_rpc/fp32_boundary_fixture.h"
#include "hvx_rpc/fp32_finite.h"
#include <cassert>
#include <cstdio>
#include <vector>
int main(){
  std::vector<float> a(256),b(256);hexatrain_boundary_fill(a.data(),256);hexatrain_boundary_fill(b.data(),256);
  assert(hexatrain_fp32_all_finite(a.data(),256));assert(!memcmp(a.data(),b.data(),1024));
  assert(hexatrain_boundary_bits(0)==0);assert(hexatrain_boundary_bits(1)==0x80000000u);
  assert(hexatrain_boundary_bits(13)==1);assert(a[6]<1024&&a[7]>1024);
  assert(a[4]<.25f&&a[5]>.25f);assert(a[8]<256&&a[9]==256);
  assert(a[10]>512&&a[11]>4096);
  for(unsigned i=0;i<256;++i){uint32_t actual;memcpy(&actual,a.data()+i,4);assert(actual==hexatrain_boundary_bits(i));}
  puts("fp32_boundary_fixture=PASS finite_count=256 zero_signed_subnormal_random=1");
}
