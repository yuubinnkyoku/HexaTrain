#define main inherited_main
#include "segment_kernel_test.cpp"
#undef main
#include <stdexcept>
#define need(x) do{if(!(x))throw std::runtime_error("qk_scale_host line="+std::to_string(__LINE__));}while(false)
int main(int argc,char** argv){need(argc==2);need(qhpi_init()&&registered==51&&info.numOperations==51);
 need(std::string(names[11])=="MatMulDProbabilities"&&std::string(names[39])=="ProjectionQKVHvx"&&std::string(names[41])=="ForwardQKRaw"&&std::string(names[43])=="BackwardScoreScale");
 auto fs=upstreamreference::fullUpstream(one::read(argv[1]));unsigned calls=0;float scale=1.f/std::sqrt(float(64/2));uint32_t bits;std::memcpy(&bits,&scale,4);need(bits==0x3e3504f3u);
 for(bool qk:{true,false})for(unsigned ci=0;ci<fs.size()+10;++ci)for(bool obs:{true,false}){auto ref=one::reference(fs[ci%fs.size()],true);std::vector<one::FV>x(5);for(unsigned j=0;j<4;++j)x[j]=qk?fs[ci%fs.size()].v.at("h"+std::to_string(j%2)+(j<2?"_q":"_k")):one::f(ref.at("h"+std::to_string(j/2)+(j%2?"_dkraw":"_dqraw")));x[4]={scale};
  if(ci>=fs.size())for(unsigned j=0;j<4;++j)for(unsigned i=0;i<1024;++i){x[j][i]=float(int((i*31+j*17+ci)%113)-56)*.01f;if(ci%10==0)x[j][i]=i%2?-0.f:0.f;if(ci%10==1)x[j][i]*=1e-15f;if(ci%10==2)x[j][i]=(i%2?-1:1)*.25f*(j+1);}
  auto before=x;unsigned ny=qk?2:4,nr=qk?6152:8201;std::vector<std::vector<uint32_t>> b(ny+1);for(unsigned j=0;j<=ny;++j){b[j].assign((j==ny?nr:1024)+8,0xa55aa55a);std::fill(b[j].begin()+4,b[j].end()-4,0x7fc0dead);}
  QHPI_Tensor it[5],ot[5];const QHPI_Tensor* in[5];QHPI_Tensor* out[5];for(unsigned j=0;j<5;++j){it[j]=tensor(j==4?1:32,j==4?1:32,x[j].data());in[j]=it+j;}for(unsigned j=0;j<=ny;++j){ot[j]=tensor(j==ny?1:32,j==ny?nr:32,b[j].data()+4,j==ny?QHPI_INT32:QHPI_FLOAT32);out[j]=ot+j;}
  auto fn=qk?(obs?attentioncustom::dprob<true,0x16>:attentioncustom::dprob<false,0x16>):obs?qkscalecustom::scale<true>:qkscalecustom::scale<false>;need(fn(nullptr,ny+obs,out,qk?4:5,in)==QHPI_SUCCESS);
  for(unsigned j=0;j<ny;++j){auto fp=one::f(qk?one::mm(one::d(x[j]),one::d(x[j+2]),32,32,32,false,true,true):one::mul(one::d(x[j]),one::d(x[4]),true));need(!std::memcmp(fp.data(),b[j].data()+4,4096));for(unsigned k=0;k<4;++k)need(b[j][k]==0xa55aa55a&&b[j][b[j].size()-1-k]==0xa55aa55a);need(std::find(b[j].begin()+4,b[j].end()-4,0x7fc0dead)==b[j].end()-4);}
  if(obs){auto r=b[ny].data()+4;need(r[0]==(qk?0x43505516u:0x43505517u));unsigned at=8;for(unsigned j=0;j<(qk?4u:5u);++j){need(!std::memcmp(r+at,x[j].data(),x[j].size()*4));at+=x[j].size();}for(unsigned j=0;j<ny;++j)need(!std::memcmp(r+at+j*1024,b[j].data()+4,4096));need(r[2]==at-8&&r[3]==ny*1024);for(unsigned k=0;k<4;++k)need(b[ny][k]==0xa55aa55a&&b[ny][b[ny].size()-1-k]==0xa55aa55a);need(std::find(b[ny].begin()+4,b[ny].end()-4,0x7fc0dead)==b[ny].end()-4);}
  for(unsigned j=0;j<5;++j)need(!std::memcmp(x[j].data(),before[j].data(),x[j].size()*4));it[0].shape.dims[3]=31;need(fn(nullptr,ny+obs,out,qk?4:5,in)==QHPI_ERROR_FATAL);it[0].shape.dims[3]=32;x[0][0]=INFINITY;need(fn(nullptr,ny+obs,out,qk?4:5,in)==QHPI_ERROR_FATAL);x[0][0]=NAN;need(fn(nullptr,ny+obs,out,qk?4:5,in)==QHPI_ERROR_FATAL);++calls;
 }
 std::printf("qk_scale_host=PASS calls=%u scale_bits=%08x words=1 guards=1 immutable=1 complete_write=1 negative=1 old_registry_prefix=1\n",calls,bits);
}
