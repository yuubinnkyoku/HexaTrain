#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include "selective_ln1_fixture.h"
#include <HTP/core/qhpi.h>
struct QHPI_Tensor {QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
unsigned registered=0;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1*,const char* name){if(std::strcmp(name,"HexaSelectiveLn1"))return QHPI_ERROR_FATAL;registered=n;return QHPI_SUCCESS;}
#define SELECTIVE_LN_HOST_KERNEL_TEST
#include "selective_ln1_package.cpp"
QHPI_Tensor tensor(unsigned r,unsigned c,void* p,unsigned type=QHPI_FLOAT32){return {{4,{1,1,r,c}},type,p};}
template<class T> void write(FILE* f,const T* p,unsigned n){if(std::fwrite(p,sizeof(T),n,f)!=n)throw std::runtime_error("fixture write");}
int main(int argc,char** argv){try{if(argc!=3)return 1;if(!qhpi_init()||registered!=4)throw std::runtime_error("new registration");auto fs=segmentstudy::lnCases(one::read(argv[1]));auto hs=precision::synthetic();fs.insert(fs.end(),hs.begin(),hs.end());if(fs.size()!=60)throw std::runtime_error("coverage");FILE* file=std::fopen(argv[2],"wb");if(!file)return 1;unsigned header[]={s2fixture::magic,s2fixture::version,60};write(file,header,3);
 for(unsigned i=0;i<fs.size();++i){auto& z=fs[i];auto c=one::lncase(z);c.x=z.v.at("layerinput");c.gamma=z.v.at("gamma1");c.beta=z.v.at("beta1");unsigned id=z.index,set=i>=44;write(file,&id,1);write(file,&set,1);for(auto* v:{&c.x,&c.gamma,&c.beta,&c.dy})write(file,v->data(),v->size());
  for(bool selected:{false,true}){float eps=1e-5f;std::vector<float> values[6];for(unsigned k=0;k<6;++k)values[k].assign(s2fixture::sizes[k],NAN);std::vector<unsigned> receipt(s2package::receiptWords);QHPI_Tensor ti[]={tensor(32,64,c.x.data()),tensor(1,64,c.gamma.data()),tensor(1,64,c.beta.data()),tensor(1,1,&eps)};const QHPI_Tensor* in[]={ti,ti+1,ti+2,ti+3};QHPI_Tensor to[]={tensor(32,64,values[0].data()),tensor(32,64,values[1].data()),tensor(32,1,values[2].data()),tensor(1,receipt.size(),receipt.data(),QHPI_INT32)};QHPI_Tensor* out[]={to,to+1,to+2,to+3};auto before=c;
   auto fn=selected?s2package::kernel<true,true>:s2package::kernel<false,true>;if(fn(nullptr,4,out,4,in)!=QHPI_SUCCESS)throw std::runtime_error("Host primitive");
   if(receipt[0]!=(0x43505500u+(selected?2:0)))throw std::runtime_error("Host marker");
   auto bad=to[0].shape;to[0].shape.dims[3]=63;if(fn(nullptr,4,out,4,in)!=QHPI_ERROR_FATAL)throw std::runtime_error("bad output shape accepted");to[0].shape=bad;
   QHPI_Tensor bi[]={tensor(32,64,c.dy.data()),ti[1],to[1],to[2]},bo[]={tensor(32,64,values[3].data()),tensor(1,64,values[4].data()),tensor(1,64,values[5].data())};const QHPI_Tensor* bin[]={bi,bi+1,bi+2,bi+3};QHPI_Tensor* bout[]={bo,bo+1,bo+2};if(lncustom::backward<false,false>(nullptr,3,bout,4,bin)!=QHPI_SUCCESS)throw std::runtime_error("Host backward");
   if(before.x!=c.x||before.gamma!=c.gamma||before.beta!=c.beta||before.dy!=c.dy)throw std::runtime_error("Host mutation");for(auto& v:values){for(float n:v)if(!std::isfinite(n))throw std::runtime_error("Host incomplete/nonfinite");write(file,v.data(),v.size());}
  }
  auto ref=mmfixture::ln::reference(one::d(c.x),one::d(c.dy),one::d(c.gamma),one::d(c.beta),64,double(1e-5f));for(auto* v:{&ref.y,&ref.xhat,&ref.inverse,&ref.dx,&ref.dg,&ref.db})write(file,v->data(),v->size());
 }
 if(std::fclose(file))throw std::runtime_error("fixture close");auto check=s2fixture::load(argv[2]);if(check.size()!=60)throw std::runtime_error("round trip");std::puts("selective_ln1_fixture=PASS cases=60 registration=4 same_binary_baseline=1 source_epsilon=1 negative_shape=1 input_mutation=0 complete_finite=1");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
