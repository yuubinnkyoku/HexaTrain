#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include "selective_ln1_arithmetic.h"
#include <HTP/core/qhpi.h>
// Opaque QHPI Host mocks exercise the unchanged actual LN backward kernel.
struct QHPI_Tensor {QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t,QHPI_OpInfo_v1*,const char*){return QHPI_SUCCESS;}
#include "layernorm_ops.h"
QHPI_Tensor stensor(unsigned n,void* p){return {{4,{1,1,1,n}},QHPI_FLOAT32,p};}
template<unsigned B> void audit(const std::vector<one::Fixture>& fs,const char* set){double ey=0,eh=0,ei=0,edx=0,edg=0,edb=0,replay=0,hreplay=0;unsigned rw=0,hw=0,cases=0;
 for(auto& z:fs){auto c=one::lncase(z);c.x=z.v.at("layerinput");c.gamma=z.v.at("gamma1");c.beta=z.v.at("beta1");auto before=c;std::vector<float> y(2056,123456),h(2056,123456),inv(40,123456);for(unsigned i=4;i<2052;++i)y[i]=h[i]=NAN;for(unsigned i=4;i<36;++i)inv[i]=NAN;
  selective::ln<B>(c.x.data(),c.gamma.data(),c.beta.data(),y.data()+4,h.data()+4,inv.data()+4);
  if(c.x!=before.x||c.gamma!=before.gamma||c.beta!=before.beta||c.dy!=before.dy)throw std::runtime_error("mutation");
  for(auto* v:{&y,&h,&inv}){for(unsigned i=0;i<4;++i)if((*v)[i]!=123456||(*v)[v->size()-1-i]!=123456)throw std::runtime_error("canary");for(unsigned i=4;i<v->size()-4;++i)if(!std::isfinite((*v)[i]))throw std::runtime_error("poison/finite");}
  auto ref=mmfixture::ln::reference(one::d(c.x),one::d(c.dy),one::d(c.gamma),one::d(c.beta),64,double(1e-5f));
  std::vector<float> yy(y.begin()+4,y.end()-4),hh(h.begin()+4,h.end()-4),ii(inv.begin()+4,inv.end()-4),dx(2048,NAN),dg(64,NAN),db(64,NAN);
  QHPI_Tensor ti[]={stensor(2048,c.dy.data()),stensor(64,c.gamma.data()),stensor(2048,hh.data()),stensor(32,ii.data())},to[]={stensor(2048,dx.data()),stensor(64,dg.data()),stensor(64,db.data())};const QHPI_Tensor* inputs[]={ti,ti+1,ti+2,ti+3};QHPI_Tensor* outputs[]={to,to+1,to+2};
  if(lncustom::backward<false,false>(nullptr,3,outputs,4,inputs)!=QHPI_SUCCESS)throw std::runtime_error("actual backward API");
  auto error=[](const auto& a,const auto& b){auto e=mmfixture::ln::errors(a,b);if(e.nonfinite)throw std::runtime_error("nonfinite");return e.absolute;};
  ey=std::max(ey,error(ref.y,yy));eh=std::max(eh,error(ref.xhat,hh));ei=std::max(ei,error(ref.inverse,ii));edx=std::max(edx,error(ref.dx,dx));edg=std::max(edg,error(ref.dg,dg));edb=std::max(edb,error(ref.db,db));
  for(unsigned i=0;i<2048;++i){float product=c.gamma[i%64]*hh[i],value=product+c.beta[i%64];replay=std::max(replay,std::fabs(double(value)-yy[i]));rw+=forwardsoftmaxreference::bits(value)!=forwardsoftmaxreference::bits(yy[i]);}
  // Replace only the inverse by its saved float value, retain each candidate's
  // centered operand contract. This is separate from affine reconstruction.
  for(unsigned r=0;r<32;++r){double mean=0;selective::Sum delta;if constexpr(B&1){for(unsigned j=0;j<64;++j)mean+=c.x[r*64+j];mean/=64;}else for(unsigned j=0;j<64;++j)delta.add(c.x[r*64+j]-c.x[r*64]);
   for(unsigned j=0;j<64;++j){unsigned i=r*64+j;double centered;if constexpr(B&1){centered=double(c.x[i])-mean;if constexpr(!(B&2))centered=float(centered);}else centered=double((c.x[i]-c.x[r*64])-delta.s/64.f);
    float value;if constexpr(B&2)value=float(centered*double(ii[r]));else value=float(centered)*ii[r];hreplay=std::max(hreplay,std::fabs(double(value)-hh[i]));hw+=forwardsoftmaxreference::bits(value)!=forwardsoftmaxreference::bits(hh[i]);}
  }++cases;
 }
 if(edx>.02||edg>.02||edb>.02)throw std::runtime_error("old local LN backward gate");
 std::cout<<"{\"kind\":\"actual_ln_backward\",\"mask\":"<<B<<",\"set\":\""<<set<<"\",\"cases\":"<<cases<<",\"y\":"<<ey<<",\"xhat\":"<<eh<<",\"inverse\":"<<ei<<",\"dx\":"<<edx<<",\"dg\":"<<edg<<",\"db\":"<<edb<<",\"reconstruction_maxAbs\":"<<replay<<",\"reconstruction_word_mismatches\":"<<rw<<",\"saved_inverse_xhat_replay_maxAbs\":"<<hreplay<<",\"saved_inverse_xhat_replay_word_mismatches\":"<<hw<<",\"backward_gate\":true,\"input_mutation\":0,\"canary\":true,\"complete_finite\":true,\"dsp_execution\":false}\n";
}
int main(int argc,char** argv){std::cout<<std::setprecision(17);try{if(argc!=2)return 1;auto fs=segmentstudy::lnCases(one::read(argv[1])),hs=precision::synthetic();
#define AUDIT(B) audit<B>(fs,"frozen");audit<B>(hs,"holdout");
 AUDIT(0) AUDIT(1) AUDIT(2) AUDIT(3) AUDIT(4) AUDIT(5) AUDIT(6) AUDIT(7)
#undef AUDIT
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}}
