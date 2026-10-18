#pragma once
#include "mixed_qkv_fixture.h"
namespace doubleqkfixture {
constexpr unsigned magic=0x44513401u;
struct Case {unsigned index=0,set=0;std::vector<float> heads[4],expected[2][2],scaled[2],masked[2],hostP[2];std::vector<double> local[2],localP[2],idealP[2][2];};
template<class T> void vectorRead(FILE* f,std::vector<T>& v){v.resize(1024);s2fixture::read(f,v.data(),1024);}
inline std::vector<Case> load(const char* path){FILE* f=std::fopen(path,"rb");if(!f)throw std::runtime_error("double QK fixture open");unsigned h[3];s2fixture::read(f,h,3);if(h[0]!=magic||h[1]!=1||h[2]!=76)throw std::runtime_error("double QK fixture header");std::vector<Case> out(76);for(auto& c:out){s2fixture::read(f,&c.index,1);s2fixture::read(f,&c.set,1);for(auto& v:c.heads)vectorRead(f,v);for(auto& mode:c.expected)for(auto& v:mode)vectorRead(f,v);for(auto& v:c.local)vectorRead(f,v);for(auto* vs:{&c.scaled,&c.masked,&c.hostP})for(auto& v:*vs)vectorRead(f,v);for(auto& v:c.localP)vectorRead(f,v);for(auto& ref:c.idealP)for(auto& v:ref)vectorRead(f,v);}if(std::fgetc(f)!=EOF)throw std::runtime_error("fixture trailing");std::fclose(f);return out;}
}
