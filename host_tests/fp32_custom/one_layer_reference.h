#pragma once
// Diagnostic-only, independent scalar analytic equations for current gated G1.
#include "custom_matmul_fixture.h"
#include <map>
#include <cstdio>
#include <stdexcept>
namespace one {
using Vec=std::vector<double>;
using FV=std::vector<float>;
struct Fixture { unsigned index=0, checkpoint=0; std::map<std::string,FV> v; };
inline Vec d(const FV& v){return {v.begin(),v.end()};}
inline FV f(const Vec& v){return {v.begin(),v.end()};}
inline Vec mm(const Vec& a,const Vec& b,unsigned m,unsigned n,unsigned k,bool ta=false,bool tb=false,bool fp=false){
 Vec y(m*n);for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j){double z=0;float s=0;
  for(unsigned q=0;q<k;++q){double x=a[ta?q*m+i:i*k+q],v=b[tb?j*k+q:q*n+j];
   if(fp){volatile float p=float(x)*float(v);volatile float t=s+p;s=t;}else z+=x*v;
  }y[i*n+j]=fp?double(s):z;
 }return y;
}
inline Vec split(const Vec& a,unsigned head){Vec o(1024);for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c)o[r*32+c]=a[r*64+head*32+c];return o;}
inline Vec scatter(const Vec& a,unsigned head){Vec o(2048);for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c)o[r*64+head*32+c]=a[r*32+c];return o;}
inline Vec add(Vec a,const Vec& b,bool fp=false){for(unsigned i=0;i<a.size();++i)a[i]=fp?double(float(a[i])+float(b[i])):a[i]+b[i];return a;}
inline Vec mul(const Vec& a,const Vec& b,bool fp=false){Vec o(a.size());for(unsigned i=0;i<a.size();++i){double v=b.size()==1?b[0]:b.size()==32?b[i/32]:b[i];o[i]=fp?double(float(a[i])*float(v)):a[i]*v;}return o;}
inline Vec sumRows(const Vec& a,bool fp=false){Vec o(32);for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c)o[r]=fp?double(float(o[r])+float(a[r*32+c])):o[r]+a[r*32+c];return o;}
inline Vec softmax(const Vec& scores,bool fp=false){Vec p(1024);for(unsigned r=0;r<32;++r){double mx=-1e300;for(unsigned c=0;c<32;++c)mx=std::max(mx,scores[r*32+c]);double sum=0;
 for(unsigned c=0;c<32;++c){double x=std::exp(scores[r*32+c]-mx);p[r*32+c]=fp?double(float(x)):x;sum=fp?double(float(sum)+float(p[r*32+c])):sum+p[r*32+c];}
 for(unsigned c=0;c<32;++c)p[r*32+c]=fp?double(float(p[r*32+c])/float(sum)):p[r*32+c]/sum;
 }return p;}
inline std::map<std::string,Vec> attention(const Fixture& z,const Vec& dc,bool fp=false){
 std::map<std::string,Vec> a;Vec dq(2048),dk(2048),dv(2048),dgates(64);
 for(unsigned h=0;h<2;++h){auto prefix="h"+std::to_string(h)+"_";auto q=d(z.v.at(prefix+"q")),k=d(z.v.at(prefix+"k")),v=d(z.v.at(prefix+"v")),p=d(z.v.at(prefix+"p")),gate=d(z.v.at(prefix+"gate")),ctx=d(z.v.at(prefix+"context"));
  auto put=[&](const char* name,Vec value){a[prefix+name]=value;return value;};
  auto selected=put("selected",split(dc,h)),dh=put("dc",mul(selected,gate,fp));
  auto gp=put("gate_product",mul(selected,ctx,fp)),gd=put("gate_dot",sumRows(gp,fp));for(unsigned r=0;r<32;++r)dgates[r*2+h]=gd[r];
  auto dp=put("dp",mm(dh,v,32,32,32,false,true,fp)),vh=put("dv",mm(p,dh,32,32,32,true,false,fp));
  auto prod=put("product",mul(dp,p,fp)),dot=put("dot",sumRows(prod,fp));auto centered=dp;for(unsigned i=0;i<1024;++i)centered[i]=fp?double(float(dp[i])-float(dot[i/32])):dp[i]-dot[i/32];put("centered",centered);
  auto ds=put("ds",mul(p,centered,fp));auto qh=put("dqraw",mm(ds,k,32,32,32,false,false,fp)),kh=put("dkraw",mm(ds,q,32,32,32,true,false,fp));
  qh=put("dq",mul(qh,{double(1.f/std::sqrt(32.f))},fp));kh=put("dk",mul(kh,{double(1.f/std::sqrt(32.f))},fp));dq=add(dq,scatter(qh,h),fp);dk=add(dk,scatter(kh,h),fp);dv=add(dv,scatter(vh,h),fp);
 }
 a["dq"]=dq;a["dk"]=dk;a["dv"]=dv;
 auto gates=d(z.v.at("gates"));Vec derivative(64),dl(64);for(unsigned i=0;i<64;++i){derivative[i]=fp?double(float(gates[i])*(1.f-float(gates[i]))):gates[i]*(1-gates[i]);dl[i]=fp?double(float(dgates[i])*float(derivative[i])):dgates[i]*derivative[i];}
 a["gate_dlogits"]=dl;auto ln1=d(z.v.at("ln1"));a["dwq"]=mm(ln1,dq,64,64,32,true,false,fp);a["dwk"]=mm(ln1,dk,64,64,32,true,false,fp);a["dwv"]=mm(ln1,dv,64,64,32,true,false,fp);a["dwg"]=mm(ln1,dl,64,2,32,true,false,fp);
 auto iq=mm(dq,d(z.v.at("wq")),32,64,64,false,true,fp),ik=mm(dk,d(z.v.at("wk")),32,64,64,false,true,fp),iv=mm(dv,d(z.v.at("wv")),32,64,64,false,true,fp),ig=mm(dl,d(z.v.at("wg")),32,64,2,false,true,fp);
 a["dln1"]=add(add(add(iq,ik,fp),iv,fp),ig,fp);return a;
}
inline mmfixture::ln::Case lncase(const Fixture& z){mmfixture::ln::Case c;c.name="one_"+std::to_string(z.index);c.x=z.v.at("x");c.gamma=z.v.at("gamma");c.beta=z.v.at("beta");c.dy=z.v.at("dy");return c;}
inline std::map<std::string,Vec> reference(const Fixture& z,bool fp=false){auto c=lncase(z);auto dr=mmfixture::ln::doubleReference(c,64);auto fr=mmfixture::ln::anchoredFp32Oracle(c,64);Vec dx=fp?d(fr.dx):dr.dx;auto res=add(dx,d(z.v.at("residual")),fp);
 auto dc=mm(res,d(z.v.at("wo")),32,64,64,false,true,fp),dw=mm(d(z.v.at("context")),res,64,64,32,true,false,fp);auto a=attention(z,dc,fp);
 a["dc"]=dc;a["dw"]=dw;a["residual"]=res;a["dx"]=dx;a["y"]=fp?d(fr.y):dr.y;a["xhat"]=fp?d(fr.xhat):dr.xhat;a["inverse"]=fp?d(fr.inverse):dr.inverse;a["dg"]=fp?d(fr.dg):dr.dg;a["db"]=fp?d(fr.db):dr.db;return a;
}
inline const std::vector<std::pair<std::string,unsigned>>& fields(){static std::vector<std::pair<std::string,unsigned>> x={{"x",2048},{"gamma",64},{"beta",64},{"dy",2048},{"residual",2048},{"wo",4096},{"context",2048},{"ln1",2048},{"layerinput",2048},{"wq",4096},{"wk",4096},{"wv",4096},{"wg",128},{"w1",8192},{"w2",8192},{"gates",64}};
 if(x.size()==16)for(unsigned h=0;h<2;++h)for(auto n:{"q","k","v","p","context"})x.push_back({"h"+std::to_string(h)+"_"+n,1024});
 if(x.size()==26)for(unsigned h=0;h<2;++h)x.push_back({"h"+std::to_string(h)+"_gate",32});return x;}
inline void write(FILE* file,const std::vector<Fixture>& fs){uint32_t magic=0x31474e4f,count=fs.size();std::fwrite(&magic,4,1,file);std::fwrite(&count,4,1,file);for(auto& z:fs){std::fwrite(&z.index,4,1,file);std::fwrite(&z.checkpoint,4,1,file);for(auto& field:fields()){auto& v=z.v.at(field.first);if(v.size()!=field.second)throw std::runtime_error("fixture shape");std::fwrite(v.data(),4,v.size(),file);}}}
inline std::vector<Fixture> read(const char* path){FILE* file=std::fopen(path,"rb");if(!file)throw std::runtime_error("fixture missing");uint32_t magic=0,count=0;if(std::fread(&magic,4,1,file)!=1||std::fread(&count,4,1,file)!=1||magic!=0x31474e4f||count!=24)throw std::runtime_error("fixture header");std::vector<Fixture> fs;
 for(unsigned j=0;j<count;++j){Fixture z;if(std::fread(&z.index,4,1,file)!=1||std::fread(&z.checkpoint,4,1,file)!=1||z.index!=j)throw std::runtime_error("fixture id");for(auto& field:fields()){auto& v=z.v[field.first];v.resize(field.second);if(std::fread(v.data(),4,v.size(),file)!=v.size())throw std::runtime_error("fixture truncated");for(auto x:v)if(!std::isfinite(x))throw std::runtime_error("fixture nonfinite");}fs.push_back(z);}if(std::fgetc(file)!=EOF)throw std::runtime_error("fixture trailing");std::fclose(file);return fs;}
}
