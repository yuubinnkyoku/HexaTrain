#pragma once
#include "one_layer_reference.h"
#include <array>
namespace attentionfixture {
struct Case{std::string name;std::array<one::FV,2>a,b;};
inline std::vector<Case> matrix(const std::vector<one::Fixture>& fs){std::vector<Case> cases;
 for(auto& z:fs){Case c;c.name="previous_"+std::to_string(z.index);auto r=one::reference(z,true);for(unsigned h=0;h<2;++h){auto p="h"+std::to_string(h)+"_";c.a[h]=one::f(r.at(p+"dc"));c.b[h]=z.v.at(p+"v");}cases.push_back(c);}
 const char* names[]={"ordinary","low_variance","constant","mixed_signs","cancellation","small","large","asymmetric_heads","fixed_random","offset"};
 uint32_t seed=71181;
 for(unsigned kind=0;kind<10;++kind){Case c;c.name=names[kind];for(unsigned h=0;h<2;++h){c.a[h].resize(1024);c.b[h].resize(1024);for(unsigned row=0;row<32;++row)for(unsigned q=0;q<32;++q){unsigned i=row*32+q;seed=seed*1664525u+1013904223u;float v=float(int(seed>>8)%2001-1000)/1000.f;
  c.a[h][i]=v*(h+1)*.5f+row*.007f;c.b[h][i]=float(int((row*17+q*23+h*13)%101)-50)*.01f;
  if(kind==1){c.a[h][i]=.75f+h+v*.001f;c.b[h][i]=.5f+row*.002f+v*.0001f;}
  if(kind==2){c.a[h][i]=.3f+h*.2f;c.b[h][i]=.7f-h*.1f;}
  if(kind==4){c.a[h][i]=(q%2?-1:1)*128.f;c.b[h][i]=.125f+row*.00390625f;}
  if(kind==5){c.a[h][i]*=1e-7f;c.b[h][i]*=1e-4f;}
  if(kind==6)c.a[h][i]*=128;
  if(kind==7){c.a[h][i]=(row*31+q*7+h*29)*.001f;c.b[h][i]=(int(row)*3-int(q)*11+int(h)*17)*.002f;}
  if(kind==9)c.a[h][i]=64.f+h+v*.001f;
 }}cases.push_back(c);}return cases;
}
}
