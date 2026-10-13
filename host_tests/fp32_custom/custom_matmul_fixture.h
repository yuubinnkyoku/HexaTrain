#pragma once
// Frozen diagnostic inputs and independent CPU references. No runtime fallback.
#include "../../app/src/main/cpp/qnn/layernorm_redesign_reference.h"
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
namespace mmfixture {
namespace ln=phonelm::qnn::lnvalidation;
struct Shape {const char* name;unsigned ar,ac,br,bc,m,n,k;bool ta,tb,left;};
inline const Shape shapes[]={{"dcontext",32,64,64,64,32,64,64,false,true,true},
                             {"dwo",32,64,32,64,64,64,32,true,false,false}};
struct Fixture {ln::Case c;std::vector<float> residual,other;unsigned control;bool additional;};
inline ln::Case repeated(ln::Case c){auto x=c.x,dy=c.dy;c.x.resize(2048);c.dy.resize(2048);
 for(unsigned r=0;r<32;++r)for(unsigned j=0;j<64;++j){c.x[r*64+j]=x[(r%2)*64+j];c.dy[r*64+j]=dy[(r%2)*64+j]*(r%3==2?-1.f:1.f);}return c;}
inline std::vector<float> coefficients(const Shape& s,unsigned control){
 std::vector<float> v(s.left?s.br*s.bc:s.ar*s.ac);uint32_t seed=97;
 for(auto& f:v){seed=1664525u*seed+1013904223u;f=float(int(seed>>8)%2001-1000)/1000.f*(s.left?.125f:1.f);}
 if(control==1){std::fill(v.begin(),v.end(),0.f);unsigned rows=s.left?s.br:s.ar,cols=s.left?s.bc:s.ac;for(unsigned r=0;r<rows;++r)v[r*cols+r%cols]=1.f;}
 if(control==2)for(unsigned j=0;j<v.size();++j)v[j]=j%2?.125f:-.125f;
 return v;
}
inline std::vector<Fixture> fixtures(const Shape& s){std::vector<Fixture> out;
 for(auto c:ln::redesignCases(64))for(unsigned control=0;control<3;++control){c=repeated(c);out.push_back({c,c.dy,coefficients(s,control),control,false});}
 for(unsigned recipe=0;recipe<8;++recipe){auto c=repeated(ln::redesignCases(64)[0]);c.name="independent32_"+std::to_string(recipe);uint32_t seed=101+recipe*37;
  for(unsigned r=0;r<32;++r)for(unsigned j=0;j<64;++j){seed=1664525u*seed+1013904223u;float v=float(int(seed>>8)%20001-10000)/10000.f;
   c.x[r*64+j]=v*(.25f+float(r%7)*.25f)+float(r)*.03125f;
   if(recipe==1)c.x[r*64+j]=float(r)*.25f+v*.001f;
   if(recipe==2)c.x[r*64+j]=v*128.f;
   if(recipe==3)c.x[r*64+j]=(j%11==r%11?v:0.f)+float(r)*.03125f;
   if(recipe==4)c.x[r*64+j]=float(int(j%9)-4)*.125f+float(r)*.015625f;
   seed=1664525u*seed+1013904223u;c.dy[r*64+j]=float(int(seed>>8)%2001-1000)/1000.f*(.125f+float(r%5)*.5f);
  }
  std::vector<float> residual(2048);for(unsigned i=0;i<2048;++i)residual[i]=c.dy[i]*float(int(i%7)-3)*.25f;
  auto other=coefficients(s,0);for(unsigned i=0;i<other.size();++i){seed=1664525u*seed+1013904223u;other[i]=float(int(seed>>8)%2001-1000)/1000.f*(s.left?.125f:1.f);
   if(recipe==5)other[i]=(i%2?.125f:-.125f)*(1.f+float((i/64)%3)*.03125f);
   if(recipe==6)other[i]=i%17==0?other[i]:0.f;
   if(recipe==7)other[i]=float(int(i%13)-6)*(s.left?.015625f:.125f);
  }out.push_back({c,residual,other,recipe,true});
 }return out;
}
inline std::vector<float> gradient(const Fixture& f){auto v=ln::anchoredFp32Oracle(f.c,64).dx;for(unsigned i=0;i<v.size();++i)v[i]+=f.residual[i];return v;}
inline std::vector<double> reference(const std::vector<float>& a,const std::vector<float>& b,const Shape& s,bool fp32){
 std::vector<double> out(s.m*s.n);for(unsigned row=0;row<s.m;++row)for(unsigned col=0;col<s.n;++col){double d=0;float f=0;
  for(unsigned q=0;q<s.k;++q){const float x=s.ta?a[q*s.ac+row]:a[row*s.ac+q],y=s.tb?b[col*s.bc+q]:b[q*s.bc+col];
   if(fp32){volatile float p=x*y;volatile float next=f+p;f=next;}else d+=double(x)*double(y);
  }out[row*s.n+col]=fp32?double(f):d;
 }return out;
}
inline std::vector<double> chainReference(const Fixture& f,const Shape& s){auto grad=ln::doubleReference(f.c,64).dx;for(unsigned i=0;i<grad.size();++i)grad[i]+=double(f.residual[i]);
 std::vector<double> result(s.m*s.n);for(unsigned i=0;i<s.m;++i)for(unsigned j=0;j<s.n;++j)for(unsigned q=0;q<s.k;++q){
  double a=s.left?grad[i*s.ac+q]:double(f.other[q*s.ac+i]);double b=s.left?double(f.other[j*s.bc+q]):grad[q*s.bc+j];result[i*s.n+j]+=a*b;
 }return result;
}
}
