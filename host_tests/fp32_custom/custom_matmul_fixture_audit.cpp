#include "custom_matmul_fixture.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <set>
#include <limits>
uint64_t hash(const std::vector<float>& v,uint64_t h=14695981039346656037ull){for(float f:v){uint32_t u;std::memcpy(&u,&f,4);for(unsigned j=0;j<4;++j){h^=(u>>(8*j))&255;h*=1099511628211ull;}}return h;}
// Serialize exact operands and LN inputs before device execution; SHA256 is external.
int main(int argc,char** argv){if(argc!=2)return 1;FILE* file=std::fopen(argv[1],"wb");if(!file)return 2;
 unsigned cases=0;double maxDouble=0,maxChain=0;
 for(auto s:mmfixture::shapes)for(auto f:mmfixture::fixtures(s)){auto gradient=mmfixture::gradient(f);
  auto a=s.left?gradient:f.other,b=s.left?f.other:gradient;
  std::printf("frozenfixture,%s,%s,%u,%u,input_hash=%016llx,ln_hash=%016llx,other_hash=%016llx\n",s.name,f.c.name.c_str(),f.control,f.additional,(unsigned long long)hash(b,hash(a)),(unsigned long long)hash(f.residual,hash(f.c.dy,hash(f.c.beta,hash(f.c.gamma,hash(f.c.x))))),(unsigned long long)hash(f.other));
  for(auto v:{f.c.x,f.c.gamma,f.c.beta,f.c.dy,f.residual,a,b})if(std::fwrite(v.data(),4,v.size(),file)!=v.size())return 3;
  if(f.additional){std::set<std::vector<float>> rows;for(unsigned r=0;r<32;++r)rows.insert({f.c.x.begin()+r*64,f.c.x.begin()+(r+1)*64});if(rows.size()!=32)return 4;}
  auto d=mmfixture::reference(a,b,s,false),fp=mmfixture::reference(a,b,s,true),chain=mmfixture::chainReference(f,s);
  for(unsigned i=0;i<d.size();++i){if(!std::isfinite(d[i])||!std::isfinite(fp[i]))return 5;maxDouble=std::max(maxDouble,std::fabs(d[i]-fp[i]));maxChain=std::max(maxChain,std::fabs(chain[i]-fp[i]));}
  ++cases;
 }
 if(std::fclose(file))return 6;
 std::printf("fixture_audit,cases=%u,fp32_double_max=%.12g,host_chain_double_max=%.12g,independent_rows=32\n",cases,maxDouble,maxChain);
 return cases==136&&maxDouble<=.03&&maxChain<=.03?0:7;
}
