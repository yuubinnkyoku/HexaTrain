#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>
#include <stdexcept>
namespace s2fixture {
constexpr unsigned magic=0x53324c4e,version=1;
constexpr unsigned sizes[]={2048,2048,32,2048,64,64};
struct Case {unsigned id=0,set=0;std::vector<float>x,ga,be,dy;std::vector<float> expected[2][6];std::vector<double> reference[6];};
template<class T> void read(FILE* f,T* p,unsigned n){if(std::fread(p,sizeof(T),n,f)!=n)throw std::runtime_error("fixture read");}
inline std::vector<Case> load(const char* path){FILE* f=std::fopen(path,"rb");if(!f)throw std::runtime_error("fixture open");unsigned header[3];read(f,header,3);if(header[0]!=magic||header[1]!=version||header[2]!=60)throw std::runtime_error("fixture header");std::vector<Case> out(header[2]);for(auto& c:out){read(f,&c.id,1);read(f,&c.set,1);for(auto* v:{&c.x,&c.ga,&c.be,&c.dy}){v->resize(v==&c.ga||v==&c.be?64:2048);read(f,v->data(),v->size());}for(auto& model:c.expected)for(unsigned k=0;k<6;++k){model[k].resize(sizes[k]);read(f,model[k].data(),sizes[k]);}for(unsigned k=0;k<6;++k){c.reference[k].resize(sizes[k]);read(f,c.reference[k].data(),sizes[k]);}}if(std::fgetc(f)!=EOF)throw std::runtime_error("fixture trailing data");std::fclose(f);return out;}
}
