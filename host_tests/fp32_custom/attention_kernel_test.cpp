#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include "attention_fixture.h"
#include "upstream_reference.h"
struct QHPI_Tensor{QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
unsigned registered=0;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1* ops,const char*){for(unsigned j=0;j<n;++j)if(!std::strcmp(ops[j].name,"HexaFp32::MatMulDProbabilities")||!std::strcmp(ops[j].name,"HexaFp32::MatMulDProbabilitiesNoReceipt"))++registered;return QHPI_SUCCESS;}
#include "package.cpp"
QHPI_Tensor tensor(unsigned r,unsigned c,void* p,unsigned dt=QHPI_FLOAT32){return {{4,{1,1,r,c}},dt,p};}
int main(int argc,char** argv){if(argc!=2)return 1;if(!qhpi_init()||registered!=2)return 2;auto fs=upstreamreference::fullUpstream(one::read(argv[1]));unsigned count=0;double maxError=0;
 for(auto z:attentionfixture::matrix(fs))for(bool observed:{true,false}){auto saved=z;std::array<std::vector<uint32_t>,3> buf;for(unsigned h=0;h<3;++h){buf[h].assign((h<2?1024:6152)+8,0xa55aa55a);std::fill(buf[h].begin()+4,buf[h].end()-4,0x7fc0dead);}auto a=tensor(32,32,z.a[0].data()),b=tensor(32,32,z.a[1].data()),c=tensor(32,32,z.b[0].data()),d=tensor(32,32,z.b[1].data()),y=tensor(32,32,buf[0].data()+4),v=tensor(32,32,buf[1].data()+4),r=tensor(1,6152,buf[2].data()+4,QHPI_INT32);const QHPI_Tensor* in[]={&a,&b,&c,&d};QHPI_Tensor* out[]={&y,&v,&r};auto fn=observed?attentioncustom::dprob<true>:attentioncustom::dprob<false>;if(fn(nullptr,observed?3:2,out,4,in)!=QHPI_SUCCESS)return 3;
  for(unsigned h=0;h<2;++h){auto ref=one::mm(one::d(z.a[h]),one::d(z.b[h]),32,32,32,false,true);auto fp=one::f(one::mm(one::d(z.a[h]),one::d(z.b[h]),32,32,32,false,true,true));if(std::memcmp(fp.data(),buf[h].data()+4,4096))return 4;for(unsigned j=0;j<1024;++j){float value;std::memcpy(&value,buf[h].data()+4+j,4);maxError=std::max(maxError,std::fabs(value-ref[j]));}}
  if(observed){auto p=buf[2].data()+4;if(p[0]!=0x4350550bu||p[2]!=4096||p[3]!=2048||p[5]!=81)return 5;for(unsigned h=0;h<2;++h)if(std::memcmp(p+8+h*1024,z.a[h].data(),4096)||std::memcmp(p+2056+h*1024,z.b[h].data(),4096)||std::memcmp(p+4104+h*1024,buf[h].data()+4,4096))return 6;}
  else for(unsigned j=4;j<6156;++j)if(buf[2][j]!=0x7fc0dead)return 7;
  for(unsigned h=0;h<(observed?3u:2u);++h){auto& p=buf[h];for(unsigned j=0;j<4;++j)if(p[j]!=0xa55aa55a||p[p.size()-1-j]!=0xa55aa55a)return 8;if(std::find(p.begin()+4,p.end()-4,0x7fc0dead)!=p.end()-4)return 9;}if(z.a!=saved.a||z.b!=saved.b)return 10;
  c.shape.dims[3]=31;if(fn(nullptr,observed?3:2,out,4,in)!=QHPI_ERROR_FATAL)return 11;c.shape.dims[3]=32;z.b[1][997]=INFINITY;if(fn(nullptr,observed?3:2,out,4,in)!=QHPI_ERROR_FATAL)return 12;z=saved;++count;
 }if(maxError>.01)return 13;std::printf("attention_host_kernel=PASS cases=%u maxAbs=%.12g fp_words_exact=1 receipt_and_free=1 guards=1 immutable=1 negative_contract=1\n",count,maxError);return 0;
}
