#pragma once
// Diagnostic contract fixed before device probability measurements.
namespace probabilitycontract {
constexpr double scaleGate=.01,maskGate=.001,pGate=1e-6,rowGate=2e-6,dsGate=.002,dvGate=.01;
inline float scale(){return 1.f/std::sqrt(float(64/2));}
inline one::FV mask(){one::FV m(1024,0.f);for(unsigned r=0;r<32;++r)for(unsigned c=r+1;c<32;++c)m[r*32+c]=-10000.f;return m;}
inline one::Vec masked(const one::Vec& s,bool fp=false){return one::add(s,one::d(mask()),fp);}
inline one::Vec backward(const one::Vec& dp,const one::Vec& p,bool fp=false){auto prod=one::mul(dp,p,fp),dot=one::sumRows(prod,fp),center=dp;for(unsigned i=0;i<1024;++i)center[i]=fp?double(float(dp[i])-float(dot[i/32])):dp[i]-dot[i/32];return one::mul(p,center,fp);}
struct Sensitivity {double d=0,l=0,v=0;};
inline Sensitivity sensitivity(const std::vector<one::Fixture>& fs){Sensitivity s;for(auto& z:upstreamreference::fullUpstream(fs)){auto ref=one::reference(z,true);for(unsigned h=0;h<2;++h){auto p="h"+std::to_string(h)+"_";auto dp=ref.at(p+"dp"),dc=ref.at(p+"dc");for(unsigned r=0;r<32;++r){double l=0;for(unsigned c=0;c<32;++c){s.d=std::max(s.d,std::fabs(dp[r*32+c]));l+=std::fabs(dp[r*32+c]);}s.l=std::max(s.l,l);}for(unsigned c=0;c<32;++c){double v=0;for(unsigned r=0;r<32;++r)v+=std::fabs(dc[r*32+c]);s.v=std::max(s.v,v);}}}return s;}
// The uniform box bound is diagnostic, NOT sufficient for certification.
// Certification additionally checks these componentwise bounds on actual P error.
inline std::pair<double,double> impact(const one::Vec& dp,const one::Vec& dc,const one::Vec& p,const one::Vec& actual){double ds=0,dv=0;for(unsigned r=0;r<32;++r){double dot=0,delta=0;for(unsigned c=0;c<32;++c){dot+=dp[r*32+c]*p[r*32+c];delta+=std::fabs(dp[r*32+c]*(actual[r*32+c]-p[r*32+c]));}for(unsigned c=0;c<32;++c){unsigned i=r*32+c;ds=std::max(ds,std::fabs(actual[i]-p[i])*std::fabs(dp[i]-dot)+std::fabs(actual[i])*delta);}}for(unsigned c=0;c<32;++c)for(unsigned k=0;k<32;++k){double sum=0;for(unsigned r=0;r<32;++r)sum+=std::fabs((actual[r*32+c]-p[r*32+c])*dc[r*32+k]);dv=std::max(dv,sum);}return {ds,dv};}
}
