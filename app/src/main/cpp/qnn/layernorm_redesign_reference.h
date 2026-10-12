// Independent diagnostic analysis, never a training execution path.
#pragma once
#include "layernorm_stability_reference.h"
#include <cstdint>
namespace phonelm::qnn::lnvalidation {
inline std::vector<Case> redesignCases(size_t d) {
  auto out=cases(d);for(auto& c:out)c.required=true; // 512/4096 are required now.
  for(uint32_t seed:{1u,7u,31u,97u}) {
    Case c=out.front();c.name="random_seed"+std::to_string(seed);uint32_t state=seed;
    for(size_t i=0;i<c.x.size();++i){state=1664525u*state+1013904223u;
      c.x[i]=float(int(state>>8)%20001-10000)/10000.f;
      if(seed==7)c.x[i]=.25f+c.x[i]*.001f;
      if(seed==31)c.x[i]*=4096.f;
      if(seed==97)c.x[i]=1024.f+c.x[i]*.001f;
    }out.push_back(std::move(c));
  }return out;
}
struct Errors {double absolute=0,rms=0,relative=0,minimum=INFINITY,maximum=-INFINITY;size_t nonfinite=0;};
template<class T,class U> Errors errors(const std::vector<T>& ref,const std::vector<U>& out){
  Errors e;double squares=0;
  if(ref.size()!=out.size()){e.absolute=INFINITY;e.nonfinite=out.size()+1;return e;}
  for(size_t i=0;i<out.size();++i){double a=ref[i],b=out[i];if(!std::isfinite(a)||!std::isfinite(b)){++e.nonfinite;continue;}
    double delta=std::abs(a-b);e.absolute=std::max(e.absolute,delta);squares+=delta*delta;
    e.relative=std::max(e.relative,delta/std::max(.001,std::abs(a)));e.minimum=std::min(e.minimum,b);e.maximum=std::max(e.maximum,b);}
  e.rms=std::sqrt(squares/std::max(size_t(1),out.size()));if(e.nonfinite)e.absolute=INFINITY;return e;
}
inline std::vector<double> replayDx(const Case& c,size_t d,const std::vector<double>& xhat,const std::vector<double>& inv){
  std::vector<double> dx(c.x.size());for(size_t r=0;r<inv.size();++r){double sum=0,prod=0;
    for(size_t j=0;j<d;++j){size_t i=r*d+j;double dh=double(c.dy[i])*c.gamma[j];sum+=dh;prod+=dh*xhat[i];}
    for(size_t j=0;j<d;++j){size_t i=r*d+j;dx[i]=inv[r]*(double(c.dy[i])*c.gamma[j]-sum/d-xhat[i]*prod/d);}}
  return dx;
}
inline std::vector<std::vector<double>> stages(const Case& c,size_t d){
  auto ref=doubleReference(c,d);size_t rows=c.x.size()/d;
  std::vector<std::vector<double>> t(21);for(size_t k=0;k<t.size();++k)t[k].resize((k==0||k==4||k==5||k==6||k==7||k==8||k==9||k==10||k==12||k==15||k==20)?rows:c.x.size());
  for(size_t r=0;r<rows;++r){double a=0;for(size_t j=0;j<d;++j)a=std::max(a,std::abs(double(c.x[r*d+j])-ref.mean[r]));
    double s=a>255.875?.5:1,ep=1e-5*s*s;t[0][r]=ref.mean[r];t[4][r]=ref.variance[r]*s*s;t[5][r]=t[4][r]+ep;t[6][r]=ref.inverse[r]/s;t[7][r]=ref.inverse[r];t[8][r]=a;t[9][r]=s;t[10][r]=ep;t[20][r]=ref.inverse[r]/d;
    double sum=0,prod=0;for(size_t j=0;j<d;++j){size_t i=r*d+j;t[1][i]=double(c.x[i])-ref.mean[r];t[2][i]=t[1][i]*s;t[3][i]=t[2][i]*t[2][i];t[11][i]=double(c.dy[i])*c.gamma[j];t[13][i]=t[11][i]*ref.xhat[i];t[14][i]=double(c.dy[i])*ref.xhat[i];sum+=t[11][i];prod+=t[13][i];}
    t[12][r]=sum;t[15][r]=prod;
    for(size_t j=0;j<d;++j){size_t i=r*d+j;t[16][i]=d*t[11][i];t[17][i]=t[16][i]-sum;t[18][i]=ref.xhat[i]*prod;t[19][i]=t[17][i]-t[18][i];}
  }return t;
}
// Nearest IEEE binary16 lattice, diagnostic only; not an HTP precision assumption.
inline double halfLattice(double x){if(!std::isfinite(x)||std::abs(x)>65504)return std::copysign(INFINITY,x);
  int exponent=0;std::frexp(std::abs(x),&exponent);double step=std::ldexp(1.,std::max(-24,exponent-11));return std::nearbyint(x/step)*step;}
inline double latticeLowerBound(const std::vector<double>& x){double e=0;for(double v:x)e=std::max(e,std::abs(v-halfLattice(v)));return e;}
// Design oracle for a future FP32 DSP kernel, NOT wired to Runtime/training.
// Anchor-relative centering preserves representable small differences at large
// offsets. hypot evaluates the standard variance+epsilon denominator without
// squaring the large centered magnitude. Kahan sums are genuinely float here.
inline Result<float> anchoredFp32Oracle(const Case& c,size_t d){
  struct Sum{float sum=0,correction=0;void add(float x){float y=x-correction,t=sum+y;correction=(t-sum)-y;sum=t;}};
  Result<float> o;size_t rows=c.x.size()/d;o.mean.resize(rows);o.variance.resize(rows);o.inverse.resize(rows);
  o.xhat.resize(c.x.size());o.y.resize(c.x.size());o.dx.resize(c.x.size());o.dg.assign(d,0);o.db.assign(d,0);
  std::vector<float> centered(d),normalized(d);std::vector<Sum> dg(d),db(d);
  for(size_t r=0;r<rows;++r){float anchor=c.x[r*d];Sum mean;for(size_t j=0;j<d;++j)mean.add(c.x[r*d+j]-anchor);
    float delta=mean.sum/float(d),amax=0;o.mean[r]=anchor+delta;
    for(size_t j=0;j<d;++j){centered[j]=(c.x[r*d+j]-anchor)-delta;amax=std::max(amax,std::abs(centered[j]));}
    Sum squares;if(amax>0)for(float v:centered){float q=v/amax;squares.add(q*q);}
    float rms=amax*std::sqrt(squares.sum/float(d));float denominator=std::hypot(rms,std::sqrt(1e-5f));
    float inv=1.f/denominator;o.inverse[r]=inv;o.variance[r]=rms*rms; // diagnostic var can overflow outside this tested domain.
    Sum sum,prod;for(size_t j=0;j<d;++j){size_t i=r*d+j;float h=centered[j]*inv;o.xhat[i]=h;o.y[i]=c.gamma[j]*h+c.beta[j];float dh=c.dy[i]*c.gamma[j];sum.add(dh);prod.add(dh*h);dg[j].add(c.dy[i]*h);db[j].add(c.dy[i]);}
    for(size_t j=0;j<d;++j){size_t i=r*d+j;o.dx[i]=inv*(c.dy[i]*c.gamma[j]-sum.sum/float(d)-o.xhat[i]*prod.sum/float(d));}
  }for(size_t j=0;j<d;++j){o.dg[j]=dg[j].sum;o.db[j]=db[j].sum;}return o;
}
} // namespace phonelm::qnn::lnvalidation
