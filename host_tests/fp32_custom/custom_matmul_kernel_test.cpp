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
#include "package.cpp"
QHPI_Tensor tensor(unsigned rows,unsigned cols,void* p,unsigned dtype=QHPI_FLOAT32){return {{4,{1,1,rows,cols}},dtype,p};}
int chainTest(){auto fs=mmfixture::fixtures(mmfixture::shapes[0]),ws=mmfixture::fixtures(mmfixture::shapes[1]);
 for(unsigned i=0;i<fs.size();++i){auto& f=fs[i];float eps=1e-5f;std::vector<float> y(2048),h(2048),inv(32),dx(2048),dg(64),db(64),res(2048),fused(2048),fg(64),fb(64);
  std::vector<unsigned> fr(4265),br(6248),ar(8200),fbr(10344),legacyFbr(8296);
  auto xt=tensor(32,64,f.c.x.data()),gt=tensor(1,64,f.c.gamma.data()),bt=tensor(1,64,f.c.beta.data()),et=tensor(1,1,&eps),dyt=tensor(32,64,f.c.dy.data()),rt=tensor(32,64,f.residual.data());
  auto yt=tensor(32,64,y.data()),ht=tensor(32,64,h.data()),it=tensor(32,1,inv.data()),dxt=tensor(32,64,dx.data()),dgt=tensor(1,64,dg.data()),dbt=tensor(1,64,db.data()),rest=tensor(32,64,res.data()),ft=tensor(32,64,fused.data()),fgt=tensor(1,64,fg.data()),fbt=tensor(1,64,fb.data());
  auto frt=tensor(1,fr.size(),fr.data(),QHPI_INT32),brt=tensor(1,br.size(),br.data(),QHPI_INT32),art=tensor(1,ar.size(),ar.data(),QHPI_INT32),fbrt=tensor(1,fbr.size(),fbr.data(),QHPI_INT32),lfrt=tensor(1,legacyFbr.size(),legacyFbr.data(),QHPI_INT32);
  const QHPI_Tensor* fi[]={&xt,&gt,&bt,&et};QHPI_Tensor* fo[]={&yt,&ht,&it,&frt};if(lncustom::forward(nullptr,4,fo,4,fi)!=QHPI_SUCCESS)return 10;
  const QHPI_Tensor* bi[]={&dyt,&gt,&ht,&it,&rt};QHPI_Tensor* bo[]={&dxt,&dgt,&dbt,&brt};if(lncustom::backward<false>(nullptr,4,bo,4,bi)!=QHPI_SUCCESS)return 11;
  const QHPI_Tensor* ai[]={&dxt,&rt};QHPI_Tensor* ao[]={&rest,&art};if(kernel<1>(nullptr,2,ao,2,ai)!=QHPI_SUCCESS)return 12;
  QHPI_Tensor* fbo[]={&ft,&fgt,&fbt,&fbrt};if(lncustom::backward<true>(nullptr,4,fbo,5,bi)!=QHPI_SUCCESS)return 13;
  if(std::memcmp(res.data(),fused.data(),8192)||std::memcmp(ar.data()+6152,res.data(),8192)||std::memcmp(fbr.data()+8296,res.data(),8192)||std::memcmp(fbr.data()+6248,dx.data(),8192))return 14;
  fbo[3]=&lfrt;if(lncustom::backward<true>(nullptr,4,fbo,5,bi)!=QHPI_SUCCESS||std::memcmp(res.data(),fused.data(),8192))return 15;
  for(auto s:mmfixture::shapes){auto other=s.left?f.other:ws[i].other;auto a=s.left?res:other,b=s.left?other:res;auto actual=mmfixture::reference(a,b,s,true),ref=mmfixture::chainReference(s.left?f:ws[i],s);
   if(mmfixture::ln::errors(ref,actual).absolute>.03)return 16;
  }
 }std::printf("chain_host_test,cases=68,observed_add_fused_legacy_receipts=PASS,double_final_gate=PASS\n");return 0;
}
int main(){unsigned checked=0;
 unsigned firstCounter[2]={};
 for(auto s:mmfixture::shapes){for(auto f:mmfixture::fixtures(s)){auto grad=mmfixture::gradient(f),a=s.left?grad:f.other,b=s.left?f.other:grad;
  std::vector<float> y(s.m*s.n,NAN);std::vector<unsigned> receipt(16+a.size()+b.size()+y.size(),0x7fc0dead);
  auto at=tensor(s.ar,s.ac,a.data()),bt=tensor(s.br,s.bc,b.data()),yt=tensor(s.m,s.n,y.data()),rt=tensor(1,receipt.size(),receipt.data(),QHPI_INT32);
  const QHPI_Tensor* in[]={&at,&bt};QHPI_Tensor* out[]={&yt,&rt};auto kernel=s.ta?mmcustom::kernel<true>:mmcustom::kernel<false>;
  if(kernel(nullptr,2,out,2,in)!=QHPI_SUCCESS)return 1;auto fp=mmfixture::reference(a,b,s,true);for(unsigned i=0;i<y.size();++i){float expected=float(fp[i]);if(std::memcmp(&expected,&y[i],4))return 2;}
  if(checked%68==0)firstCounter[s.ta]=receipt[1];
  if(std::memcmp(receipt.data()+16,a.data(),4*a.size())||std::memcmp(receipt.data()+16+a.size(),b.data(),4*b.size())||std::memcmp(receipt.data()+16+a.size()+b.size(),y.data(),4*y.size()))return 3;
  at.shape.dims[2]=16;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 4;at.shape.dims[2]=s.ar;
  bt.dtype=QHPI_FLOAT16;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 5;bt.dtype=QHPI_FLOAT32;
  auto old=a[0];a[0]=INFINITY;if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 6;a[0]=old;
  std::fill(a.begin(),a.end(),std::numeric_limits<float>::max());std::fill(b.begin(),b.end(),2.f);if(kernel(nullptr,2,out,2,in)!=QHPI_ERROR_FATAL)return 7;
  ++checked;
 }}if(firstCounter[0]!=1||firstCounter[1]!=1)return 9;
 std::printf("kernel_host_test,cases=%u,fp32_bitwise=1,shape_dtype_nonfinite_overflow_rejected=1,branch_counters_independent=1\n",checked);return checked==136?chainTest():8;
}
