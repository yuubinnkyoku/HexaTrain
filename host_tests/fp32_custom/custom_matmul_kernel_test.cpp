#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include "custom_matmul_fixture.h"
// Host-only mock of opaque QHPI accessors exercises the actual scalar kernel.
struct QHPI_Tensor {QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t,QHPI_OpInfo_v1*,const char*){return QHPI_SUCCESS;}
#include "layernorm_ops.h"
#include "custom_matmul_ops.h"
QHPI_Tensor tensor(unsigned rows,unsigned cols,void* p,unsigned dtype=QHPI_FLOAT32){return {{4,{1,1,rows,cols}},dtype,p};}
int main(){unsigned checked=0;
 for(auto s:mmfixture::shapes){for(auto f:mmfixture::fixtures(s)){auto grad=mmfixture::gradient(f),a=s.left?grad:f.other,b=s.left?f.other:grad;
  std::vector<float> y(s.m*s.n,NAN);std::vector<unsigned> receipt(16+a.size()+b.size()+y.size(),0x7fc0dead);
  auto at=tensor(s.ar,s.ac,a.data()),bt=tensor(s.br,s.bc,b.data()),yt=tensor(s.m,s.n,y.data()),rt=tensor(1,receipt.size(),receipt.data(),QHPI_INT32);
  const QHPI_Tensor* in[]={&at,&bt};QHPI_Tensor* out[]={&yt,&rt};auto kernel=s.ta?mmcustom::kernel<true>:mmcustom::kernel<false>;
  if(kernel(nullptr,2,out,2,in)!=QHPI_SUCCESS)return 1;auto fp=mmfixture::reference(a,b,s,true);for(unsigned i=0;i<y.size();++i){float expected=float(fp[i]);if(std::memcmp(&expected,&y[i],4))return 2;}
  if(std::memcmp(receipt.data()+16,a.data(),4*a.size())||std::memcmp(receipt.data()+16+a.size(),b.data(),4*b.size())||std::memcmp(receipt.data()+16+a.size()+b.size(),y.data(),4*y.size()))return 3;
  at.shape.dims[2]=16;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 4;at.shape.dims[2]=s.ar;
  bt.dtype=QHPI_FLOAT16;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 5;bt.dtype=QHPI_FLOAT32;
  auto old=a[0];a[0]=INFINITY;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 6;a[0]=old;
  std::fill(a.begin(),a.end(),std::numeric_limits<float>::max());std::fill(b.begin(),b.end(),2.f);if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 7;
  ++checked;
 }}std::printf("kernel_host_test,cases=%u,fp32_bitwise=1,shape_dtype_nonfinite_overflow_rejected=1\n",checked);return checked==136?0:8;
}
