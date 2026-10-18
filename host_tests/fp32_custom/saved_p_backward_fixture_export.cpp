// Freeze precisely the existing gradient selection; never replace a recovered fixture.
#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include <cstdio>
int main(int argc,char** argv) {
  try {
    if(argc!=3)return 1;
    auto original=one::read(argv[1]);
    auto core=upstreamreference::fullUpstream(original);
    std::vector<std::array<precision::Gradient,2>> gradients;
    for(auto& z:core) {
      auto ref=one::reference(z,true);std::array<precision::Gradient,2> g;
      for(unsigned h=0;h<2;++h){auto n="h"+std::to_string(h)+"_";g[h]={ref.at(n+"dp"),ref.at(n+"dc")};}
      gradients.push_back(g);
    }
    if(gradients.size()!=25)throw std::runtime_error("core gradient count");
    auto hold=precision::syntheticGradients();
    if(hold.size()!=8)throw std::runtime_error("holdout gradient count");
    gradients.insert(gradients.end(),hold.begin(),hold.end());
    // Exclusive creation prevents accidental fixture regeneration over previous words.
    FILE* f=std::fopen(argv[2],"wbx");if(!f)throw std::runtime_error("new fixture exclusive creation");
    unsigned header[]={0x53425001,1,25,8};
    if(std::fwrite(header,4,4,f)!=4)return 2;
    for(auto& g:gradients)for(auto& h:g)for(auto* v:{&h.dp,&h.dc}) {
      auto fp=one::f(*v);if(fp.size()!=1024)throw std::runtime_error("gradient shape");
      for(unsigned i=0;i<fp.size();++i)if(!std::isfinite(fp[i])||double(fp[i])!=v->at(i))
        throw std::runtime_error("gradient must already be a frozen FP32 word");
      if(std::fwrite(fp.data(),4,1024,f)!=1024)return 2;
    }
    if(std::fclose(f))return 2;
    std::puts("saved_p_gradient_freeze=PASS core=25 holdout=8 heads=2 original_selection=1 rescale=0");return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 3;}
}
