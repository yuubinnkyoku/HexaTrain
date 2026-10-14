#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include "broadcast_fixture.h"
#include "upstream_reference.h"
struct QHPI_Tensor {QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
unsigned registered=0;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1* ops,const char*){for(unsigned i=0;i<n;++i)if(!std::strcmp(ops[i].name,"HexaFp32::BroadcastMultiply")){if(ops[i].kernels[0].min_inputs!=4||ops[i].kernels[0].min_outputs!=3)return QHPI_ERROR_FATAL;++registered;}return QHPI_SUCCESS;}
#include "package.cpp"
QHPI_Tensor tensor(unsigned r,unsigned c,void* p,unsigned dt=QHPI_FLOAT32){return {{4,{1,1,r,c}},dt,p};}
int main(int argc,char** argv){if(argc!=2)return 1;if(!qhpi_init()||registered!=1)return 2;auto fs=upstreamreference::fullUpstream(one::read(argv[1]));unsigned cases=0;double maxError=0;
 for(auto z:broadcastfixture::matrix(fs)){auto saved=z;std::array<std::vector<unsigned>,3> guarded;for(unsigned h=0;h<3;++h){guarded[h].assign((h<2?1024:4168)+8,0xa55aa55a);std::fill(guarded[h].begin()+4,guarded[h].end()-4,0x7fc0dead);}
  auto a=tensor(32,32,z.a[0].data()),b=tensor(32,32,z.a[1].data()),g=tensor(32,1,z.g[0].data()),h=tensor(32,1,z.g[1].data()),y=tensor(32,32,guarded[0].data()+4),v=tensor(32,32,guarded[1].data()+4),r=tensor(1,4168,guarded[2].data()+4,QHPI_INT32);const QHPI_Tensor* ins[]={&a,&b,&g,&h};QHPI_Tensor* outs[]={&y,&v,&r};
  if(broadcastcustom::kernel(nullptr,3,outs,4,ins)!=QHPI_SUCCESS)return 3;auto receipt=guarded[2].data()+4;if(receipt[0]!=0x4350550a||receipt[2]!=2048||receipt[3]!=64||receipt[5]!=81||receipt[6]!=32)return 4;
  for(unsigned head=0;head<2;++head){auto ref=broadcastfixture::reference(z.a[head],z.g[head],false),fp=broadcastfixture::reference(z.a[head],z.g[head],true);for(unsigned i=0;i<1024;++i){float actual;std::memcpy(&actual,guarded[head].data()+4+i,4);float expected=fp[i];if(!std::isfinite(actual)||std::memcmp(&actual,&expected,4))return 5;maxError=std::max(maxError,std::fabs(actual-ref[i]));}if(std::memcmp(receipt+8+head*1024,z.a[head].data(),4096)||std::memcmp(receipt+2056+head*32,z.g[head].data(),128)||std::memcmp(receipt+2120+head*1024,guarded[head].data()+4,4096))return 6;}
  for(auto& buf:guarded){for(unsigned i=0;i<4;++i)if(buf[i]!=0xa55aa55a||buf[buf.size()-1-i]!=0xa55aa55a)return 7;if(std::find(buf.begin()+4,buf.end()-4,0x7fc0dead)!=buf.end()-4)return 8;}
  if(z.a!=saved.a||z.g!=saved.g)return 9;
  if(broadcastcustom::kernel(nullptr,2,outs,4,ins)!=QHPI_ERROR_FATAL)return 10;g.shape.dims[3]=2;if(broadcastcustom::kernel(nullptr,3,outs,4,ins)!=QHPI_ERROR_FATAL)return 11;g.shape.dims[3]=1;a.dtype=QHPI_FLOAT16;if(broadcastcustom::kernel(nullptr,3,outs,4,ins)!=QHPI_ERROR_FATAL)return 12;a.dtype=QHPI_FLOAT32;r.shape.dims[3]=4167;if(broadcastcustom::kernel(nullptr,3,outs,4,ins)!=QHPI_ERROR_FATAL)return 13;r.shape.dims[3]=4168;
  z.g[1][31]=INFINITY;if(broadcastcustom::kernel(nullptr,3,outs,4,ins)!=QHPI_ERROR_FATAL)return 14;++cases;
 }
 if(maxError>.01)return 15;std::printf("broadcast_host_kernel=PASS cases=%u heads=2 maxAbs=%.12g fp_words_exact=1 registration=1 receipt=1 poison=1 canary=1 mutation=0 negative_contract=1\n",cases,maxError);
}
