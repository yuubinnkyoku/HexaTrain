#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include "mixed_qkv_fixture.h"
#include "mixed_qkv_arithmetic.h"
template<class T> void put(FILE* f,const T* p,unsigned n){if(std::fwrite(p,sizeof(T),n,f)!=n)throw std::runtime_error("write");}
unsigned bits(float f){unsigned u;std::memcpy(&u,&f,4);return u;}
int main(int argc,char** argv){try{if(argc!=4)return 1;auto ln=s2fixture::load(argv[2]);auto fs=segmentstudy::lnCases(one::read(argv[1]));auto hs=precision::synthetic();fs.insert(fs.end(),hs.begin(),hs.end());if(fs.size()!=60)throw std::runtime_error("coverage");FILE* f=std::fopen(argv[3],"wb");if(!f)return 1;unsigned h[]={mixedfixture::magic,1,68};put(f,h,3);unsigned words=0,distinct=0;
 for(unsigned i=0;i<68;++i){unsigned source=i<60?i:0,set=i<44?0:i<60?1:2;auto& z=fs[source];std::vector<float> y=ln[source].expected[1][0],w[3];w[0]=z.v.at("wq");w[1]=z.v.at("wk");if(z.v.count("wv"))w[2]=z.v.at("wv");else {uint32_t seed=707003+source*104729;for(unsigned j=0;j<4096;++j)w[2].push_back(precision::next(seed)*.0625f);}
  if(i>=60){uint32_t seed=900001+i*1009;for(unsigned j=0;j<2048;++j){float a=precision::next(seed);switch(i-60){case 0:a=(j%2?-0.f:0.f);break;case 1:a*=1e-20f;break;case 2:a*=1e10f;break;case 3:a=(j%2?-1.f:1.f)*1e4f;break;case 4:a=(j%64<32?1.f:-3.f)*a;break;case 5:a=float(int(j%67)-33)/31.f;break;case 6:a*=1e-38f;break;default:break;}y[j]=a;}for(unsigned p=0;p<3;++p)for(unsigned j=0;j<4096;++j){float b=precision::next(seed)*(p+1)*.1f;if(i==62)b*=1e-10f;if(i==65)b=float(int((j*17+p*29)%127)-63)/97.f;if(i==66)b=(j%65==0?1.f:0.f);w[p][j]=b;}}
  put(f,&i,1);put(f,&set,1);put(f,y.data(),2048);for(auto& v:w)put(f,v.data(),4096);
  std::vector<float> expected[2][3];std::vector<double> refs[2][3];for(unsigned p=0;p<3;++p){refs[0][p]=one::mm(one::d(y),one::d(w[p]),32,64,64);refs[1][p]=i<60?one::mm(ln[i].reference[0],one::d(w[p]),32,64,64):refs[0][p];expected[0][p]=one::f(one::mm(one::d(y),one::d(w[p]),32,64,64,false,false,true));expected[1][p]=p<2?one::f(refs[0][p]):expected[0][p];if(p<2){std::vector<float> actual(2048,NAN);mixedarith::project(y.data(),w[p].data(),actual.data());for(unsigned j=0;j<2048;++j){if(bits(actual[j])!=bits(expected[1][p][j]))throw std::runtime_error("double words");++words;distinct+=bits(expected[0][p][j])!=bits(expected[1][p][j]);}}}
  for(auto& mode:expected)for(auto& v:mode)put(f,v.data(),2048);for(auto& mode:refs)for(auto& v:mode)put(f,v.data(),2048);
 }
 if(std::fclose(f))throw std::runtime_error("close");if(mixedfixture::load(argv[3]).size()!=68)throw std::runtime_error("roundtrip");std::printf("mixed_host=PASS cases=68 core=44 holdout=16 arithmetic_controls=8 qk_words=%u baseline_different_words=%u custom_reference_independent=1 fixture_roundtrip=1\n",words,distinct);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
