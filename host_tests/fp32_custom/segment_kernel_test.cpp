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
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1*,const char*){registered+=n;return QHPI_SUCCESS;}
#include "package.cpp"
QHPI_Tensor tensor(unsigned r,unsigned c,void* p,unsigned dt=QHPI_FLOAT32){return {{4,{1,1,r,c}},dt,p};}
int main(int argc,char** argv){if(argc!=2)return 1;if(!qhpi_init()||registered!=51)return 2;auto fs=one::read(argv[1]);unsigned count=0;double maxProduct=0,maxProjection=0;
 for(unsigned projection=0;projection<2;++projection)for(unsigned index=0;index<fs.size()+10;++index)for(bool observed:{true,false}){auto& z=fs[index%fs.size()];unsigned n=projection?2048:1024,ny=projection?3:2,nr=projection?20488:6152;std::array<one::FV,4> x;for(unsigned i=0;i<4;++i){x[i].resize(projection?(i?4096:2048):1024);if(projection)x[i]=i?z.v.at(i==1?"wq":i==2?"wk":"wv"):z.v.at("ln1");else for(unsigned j=0;j<1024;++j)x[i][j]=float(int((j*17+i*31+index*11)%113)-56)*.017f;}if(index>=fs.size()){uint32_t seed=713+index;for(unsigned i=0;i<4;++i)for(unsigned j=0;j<x[i].size();++j){seed=1664525*seed+1013904223;x[i][j]=float(int(seed>>16)-32768)*.00003f*(i+1);if(index%10==0)x[i][j]=.25f*(i+1);if(index%10==1)x[i][j]=64.f+(j%7)*.001f;if(index%10==2)x[i][j]*=1e-9f;if(index%10==3)x[i][j]=(j%2?-1:1)*.125f;}}auto before=x;
 std::array<std::vector<uint32_t>,4> buf;for(unsigned i=0;i<ny+1;++i){buf[i].assign((i==ny?nr:n)+8,0xa55aa55a);std::fill(buf[i].begin()+4,buf[i].end()-4,0x7fc0dead);}QHPI_Tensor ts[4];for(unsigned i=0;i<4;++i)ts[i]=tensor(projection&&i?64:32,projection?64:32,x[i].data());const QHPI_Tensor* in[]={ts,ts+1,ts+2,ts+3};QHPI_Tensor ot[4];for(unsigned i=0;i<ny;++i)ot[i]=tensor(32,projection?64:32,buf[i].data()+4);ot[ny]=tensor(1,nr,buf[ny].data()+4,QHPI_INT32);QHPI_Tensor* out[]={ot,ot+1,ot+2,ot+3};auto fn=projection?(observed?segmentcustom::kernel<true,true>:segmentcustom::kernel<true,false>):observed?segmentcustom::kernel<false,true>:segmentcustom::kernel<false,false>;if(fn(nullptr,ny+observed,out,4,in)!=QHPI_SUCCESS)return 3;
 for(unsigned h=0;h<ny;++h){auto ref=projection?one::mm(one::d(x[0]),one::d(x[h+1]),32,64,64):one::mul(one::d(x[h]),one::d(x[h+2]));auto fp=one::f(projection?one::mm(one::d(x[0]),one::d(x[h+1]),32,64,64,false,false,true):one::mul(one::d(x[h]),one::d(x[h+2]),true));if(std::memcmp(fp.data(),buf[h].data()+4,n*4))return 4;for(unsigned j=0;j<n;++j){float a;std::memcpy(&a,buf[h].data()+4+j,4);(projection?maxProjection:maxProduct)=std::max(projection?maxProjection:maxProduct,std::fabs(a-ref[j]));}}
 if(observed){auto r=buf[ny].data()+4;unsigned at=8;for(auto& a:x){if(std::memcmp(r+at,a.data(),a.size()*4))return 5;at+=a.size();}if(r[0]!=0x43505500u+(projection?13:12)||r[2]!=at-8||r[3]!=ny*n||r[5]!=81)return 6;for(unsigned i=0;i<ny;++i)if(std::memcmp(r+at+i*n,buf[i].data()+4,n*4))return 7;}else for(unsigned j=4;j<nr+4;++j)if(buf[ny][j]!=0x7fc0dead)return 8;
 for(unsigned i=0;i<ny+observed;++i){auto& b=buf[i];for(unsigned j=0;j<4;++j)if(b[j]!=0xa55aa55a||b[b.size()-1-j]!=0xa55aa55a)return 9;if(std::find(b.begin()+4,b.end()-4,0x7fc0dead)!=b.end()-4)return 10;}if(x!=before)return 11;ts[3].shape.dims[3]--;if(fn(nullptr,ny+observed,out,4,in)!=QHPI_ERROR_FATAL)return 12;ts[3].shape.dims[3]++;x[0][0]=INFINITY;if(fn(nullptr,ny+observed,out,4,in)!=QHPI_ERROR_FATAL)return 13;++count;
 }std::printf("segment_host_kernel=PASS cases=%u product_maxAbs=%.12g projection_maxAbs=%.12g fp_exact=1 receipts=1 complete_write=1 guards=1 immutable=1 negative_contract=1 accuracy_claim=separate_probe_fixtures\n",count,maxProduct,maxProjection);return 0;}
