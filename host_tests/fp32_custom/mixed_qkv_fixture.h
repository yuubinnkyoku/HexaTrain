#pragma once
#include "selective_ln1_fixture.h"
#include <array>
namespace mixedfixture {
constexpr unsigned magic=0x4d513301;
struct Case {unsigned index=0,set=0;std::vector<float> y,w[3],expected[2][3];std::vector<double> local[3],ideal[3];};
inline std::vector<Case> load(const char* path){FILE* f=std::fopen(path,"rb");if(!f)throw std::runtime_error("mixed fixture open");unsigned h[3];s2fixture::read(f,h,3);if(h[0]!=magic||h[1]!=1||h[2]!=68)throw std::runtime_error("mixed fixture header");std::vector<Case> cs(h[2]);for(auto& c:cs){s2fixture::read(f,&c.index,1);s2fixture::read(f,&c.set,1);c.y.resize(2048);s2fixture::read(f,c.y.data(),2048);for(auto& w:c.w){w.resize(4096);s2fixture::read(f,w.data(),4096);}for(auto& mode:c.expected)for(auto& v:mode){v.resize(2048);s2fixture::read(f,v.data(),2048);}for(auto* refs:{&c.local,&c.ideal})for(auto& v:*refs){v.resize(2048);s2fixture::read(f,v.data(),2048);}}if(std::fgetc(f)!=EOF)throw std::runtime_error("mixed fixture trailing");std::fclose(f);return cs;}
}
