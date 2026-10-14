#pragma once
#include "one_layer_reference.h"
#include <array>
namespace broadcastfixture {
struct Case {std::string name;std::array<one::FV,2> a,g;};
inline std::vector<Case> matrix(const std::vector<one::Fixture>& fs){
 std::vector<Case> cases;uint32_t random=0x31415926;
 const char* names[]={"ordinary","low_variance","constant","large_offset_low_variance","mixed_signs","near_zero","cancellation_pairs","magnitude_mixture","nontrivial_coefficients","fixed_seed_random"};
 for(unsigned c=0;c<10;++c){Case z;z.name=names[c];for(unsigned h=0;h<2;++h){z.a[h].resize(1024);z.g[h].resize(32);
  for(unsigned r=0;r<32;++r)z.g[h][r]=(h?-.713f:.319f)+(float(r)-16)*.00317f;
  for(unsigned i=0;i<1024;++i){float s=(i%2?-1.f:1.f),v=(float(i%113)-56)*.0271f;
   switch(c){case 1:v=1.f+float(i%7)*.00003f;break;case 2:v=2.125f;break;case 3:v=1024.f+float(i%7)*.001f;break;case 4:v=s*(3.1f+i*.017f);break;case 5:v=s*1e-20f*(i%11);break;case 6:v=s*(21.234f+float(i/2%13)*.01f);break;case 7:v=s*(i%3==0?10001.3f:i%3==1?1e-15f:.07213f);break;case 8:v=s*(i*.27319f);break;case 9:random=random*1664525u+1013904223u;v=(float(random>>8)/16777216.f-.5f)*100.f;break;default:break;}
   z.a[h][i]=v*(h?1.013f:1.f);
  }}cases.push_back(z);
 }
 for(auto& f:fs){Case z;z.name="previous_gate_"+std::to_string(f.index);auto dc=one::f(one::reference(f,true).at("dc"));for(unsigned h=0;h<2;++h){z.a[h]=one::f(one::split(one::d(dc),h));z.g[h]=f.v.at("h"+std::to_string(h)+"_gate");}cases.push_back(z);}return cases;
}
// Independent reference: iterate linear source, selecting a token coefficient.
inline one::Vec reference(const one::FV& a,const one::FV& g,bool fp){one::Vec out;out.reserve(a.size());unsigned token=0,channel=0;for(float x:a){if(fp){volatile float p=x*g.at(token);out.push_back(p);}else out.push_back(double(x)*double(g.at(token)));if(++channel==32){channel=0;++token;}}return out;}
}
