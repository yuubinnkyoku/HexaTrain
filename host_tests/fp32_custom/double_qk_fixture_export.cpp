#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include "double_qk_fixture.h"
#include "double_qk_arithmetic.h"
template<class T> void put(FILE* f,const T* p,unsigned n){if(std::fwrite(p,sizeof(T),n,f)!=n)throw std::runtime_error("write");}
std::vector<float> heads(const std::vector<float>& v,unsigned h){return one::f(one::split(one::d(v),h));}
int main(int argc,char** argv){try{if(argc!=4)return 1;precision::checks();auto mixed=mixedfixture::load(argv[2]);auto fs=segmentstudy::lnCases(one::read(argv[1]));auto hs=precision::synthetic();fs.insert(fs.end(),hs.begin(),hs.end());if(fs.size()!=60)throw std::runtime_error("coverage");FILE* f=std::fopen(argv[3],"wb");if(!f)return 1;unsigned header[]={doubleqkfixture::magic,1,76};put(f,header,3);unsigned words=0,different=0;
 for(unsigned ci=0;ci<76;++ci){doubleqkfixture::Case c;c.index=ci;c.set=ci<44?0:ci<60?1:2;
  for(unsigned h=0;h<2;++h){c.heads[h]=heads(mixed[std::min(ci,67u)].expected[1][0],h);c.heads[h+2]=heads(mixed[std::min(ci,67u)].expected[1][1],h);}
  if(ci>=68){unsigned kind=ci-68;uint32_t seed=943401+kind*1009;for(unsigned p=0;p<4;++p)for(unsigned i=0;i<1024;++i){float x=precision::next(seed)*(p+1);if(kind==1)x=(i%2?-1.f:1.f)*(p+1)*1000.f;if(kind==2)x=i%2?-0.f:0.f;if(kind==3)x=p<2?x*1e-38f:(i%33==0?1.f:0.f);if(kind==4)x=p<2?x*1e10f:x*1e-10f;if(kind==5)x=float(int((i*(p+7)+31*p)%109)-54)/17.f;if(kind==6)x=i%33==0?float(p+1):0.f;if(kind==7)x=1.f+float(int(i%3)-1)*0x1p-23f;c.heads[p][i]=x;}}
  for(unsigned h=0;h<2;++h){c.local[h]=one::mm(one::d(c.heads[h]),one::d(c.heads[h+2]),32,32,32,false,true);c.expected[1][h]=one::f(c.local[h]);c.expected[0][h]=one::f(one::mm(one::d(c.heads[h]),one::d(c.heads[h+2]),32,32,32,false,true,true));std::vector<float> out(1024,NAN);doubleqkarith::nt(c.heads[h].data(),c.heads[h+2].data(),out.data());for(unsigned i=0;i<1024;++i){if(forwardsoftmaxreference::bits(out[i])!=forwardsoftmaxreference::bits(c.expected[1][h][i]))throw std::runtime_error("independent NT words");++words;different+=forwardsoftmaxreference::bits(out[i])!=forwardsoftmaxreference::bits(c.expected[0][h][i]);}
   c.scaled[h]=one::f(one::mul(one::d(c.expected[1][h]),{double(probabilitycontract::scale())},true));c.masked[h]=one::f(probabilitycontract::masked(one::d(c.scaled[h]),true));c.localP[h]=one::softmax(one::d(c.masked[h]));c.hostP[h]=one::f(one::softmax(one::d(c.masked[h]),true));
  }
  if(ci<60){auto& z=fs[ci];auto lc=one::lncase(z);lc.x=z.v.at("layerinput");lc.gamma=z.v.at("gamma1");lc.beta=z.v.at("beta1");for(unsigned ref=0;ref<2;++ref){auto ps=precision::ideal(lc,one::d(z.v.at("wq")),one::d(z.v.at("wk")),ref==1);for(unsigned h=0;h<2;++h)c.idealP[ref][h]=ps[h];}}
  else for(auto& ref:c.idealP)for(unsigned h=0;h<2;++h)ref[h]=c.localP[h];
  put(f,&c.index,1);put(f,&c.set,1);for(auto& v:c.heads)put(f,v.data(),1024);for(auto& m:c.expected)for(auto& v:m)put(f,v.data(),1024);for(auto& v:c.local)put(f,v.data(),1024);for(auto* vs:{&c.scaled,&c.masked,&c.hostP})for(auto& v:*vs)put(f,v.data(),1024);for(auto& v:c.localP)put(f,v.data(),1024);for(auto& ref:c.idealP)for(auto& v:ref)put(f,v.data(),1024);
 }
 if(std::fclose(f)||doubleqkfixture::load(argv[3]).size()!=76)throw std::runtime_error("roundtrip");std::printf("double_qk_host=PASS cases=76 core=44 holdout=16 stress=16 words=%u ordered_fp32_different_words=%u independent=1\n",words,different);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
