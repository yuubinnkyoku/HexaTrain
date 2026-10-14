#include "upstream_reference.h"
#include <cstdio>
int main(int argc,char** argv){if(argc!=2)return 1;auto fs=upstreamreference::fullUpstream(one::read(argv[1]));double maximum=0;unsigned values=0;
 for(auto& z:fs){auto ref=upstreamreference::forward(z),fp=upstreamreference::forward(z,true);for(auto& p:ref){if(p.second.size()!=fp.at(p.first).size())return 2;for(unsigned i=0;i<p.second.size();++i){if(!std::isfinite(p.second[i])||!std::isfinite(fp.at(p.first)[i]))return 3;++values;}}
  for(auto name:{"x","context","dy"}){auto& v=ref.at("up_"+std::string(name));auto& expected=z.v.at(name);for(unsigned i=0;i<v.size();++i)maximum=std::max(maximum,std::fabs(v[i]-expected[i]));}
  // The split/scatter maps are permutation identities, independent of arithmetic.
  auto q=ref.at("up_q");auto combined=one::add(one::scatter(one::split(q,0),0),one::scatter(one::split(q,1),1));if(combined!=q)return 4;
 }
 if(maximum>.01)return 5;for(auto x:fs.back().v.at("residual"))if(x!=1)return 6;std::printf("upstream_reference_host=PASS cases=%zu values=%u frozen_fp32_maxAbs=%.12g full_upstream=1 split_scatter_identity=1\n",fs.size(),values,maximum);
}
