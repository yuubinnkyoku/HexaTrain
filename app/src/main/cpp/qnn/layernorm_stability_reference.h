// SPDX-License-Identifier: Apache-2.0
// Independent two-pass reference; never used to execute a training step.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
namespace phonelm::qnn::lnvalidation {
template<class T> struct Result {
  std::vector<T> mean, variance, inverse, xhat, y, dx, dg, db;
};
template<class T> Result<T> reference(const std::vector<T>& x, const std::vector<T>& dy,
    const std::vector<T>& gamma, const std::vector<T>& beta, size_t d, T epsilon, bool scaled=false, bool dynamic=false) {
  Result<T> o; const size_t rows=x.size()/d;
  o.mean.resize(rows); o.variance.resize(rows); o.inverse.resize(rows);
  o.xhat.resize(x.size()); o.y.resize(x.size()); o.dx.resize(x.size()); o.dg.assign(d,0); o.db.assign(d,0);
  for(size_t r=0;r<rows;++r){
    T mean=0; for(size_t j=0;j<d;++j) mean+=x[r*d+j]; mean/=T(d);
    T vmax=0; for(size_t j=0;j<d;++j) vmax=std::max(vmax,std::abs(x[r*d+j]-mean));
    T s=dynamic?T(128)/std::max(T(128),vmax):(scaled && vmax>T(255.875)?T(.5):T(1)), variance=0;
    for(size_t j=0;j<d;++j){T c=s*(x[r*d+j]-mean);variance+=c*c;} variance/=T(d);
    T inv=s/std::sqrt(variance+epsilon*s*s), sum=0, prod=0;
    o.mean[r]=mean;o.variance[r]=variance/(s*s);o.inverse[r]=inv;
    for(size_t j=0;j<d;++j){size_t i=r*d+j;T h=(x[i]-mean)*inv;
      o.xhat[i]=h;o.y[i]=h*gamma[j]+beta[j];T dh=dy[i]*gamma[j];sum+=dh;prod+=dh*h;
      o.dg[j]+=dy[i]*h;o.db[j]+=dy[i];}
    for(size_t j=0;j<d;++j){size_t i=r*d+j;o.dx[i]=inv*(dy[i]*gamma[j]-sum/T(d)-o.xhat[i]*prod/T(d));}
  } return o;
}
struct Case { std::string name; std::vector<float> x,dy,gamma,beta; bool required=true; };
inline std::vector<Case> cases(size_t d) {
  struct Pattern { const char* name; float amplitude, mean; bool required; };
  const Pattern patterns[]{
    {"ordinary",1,0,true},{"small_centered",.01f,0,true},
    {"low_variance",.001f,.25f,true},{"epsilon_dominated",.00001f,.25f,true},
    {"constant",0,.25f,true},{"large_mean_low_variance",.001f,1024,true},
    {"boundary_below",255.75f,0,true},{"boundary_equal",255.875f,0,true},
    {"boundary_above",256,0,true},{"known_failure",269.75f,0,true},
    {"large_signed",400,0,true},{"half_scale_limit",512,0,false},
    {"very_large_signed",4096,0,false},{"large_mean",100,8192,true}};
  std::vector<Case> out;
  for(const auto& p:patterns){Case c;c.name=p.name;c.required=p.required;
    c.x.resize(2*d);c.dy.resize(2*d);c.gamma.resize(d);c.beta.resize(d);
    for(size_t j=0;j<d;++j){c.gamma[j]=.75f+float(j%5)*.125f;c.beta[j]=float(int(j%7)-3)*.03125f;}
    for(size_t i=0;i<2*d;++i){size_t j=i%d;
      // The second row stays ordinary: selected and unselected paths coexist.
      float a=i<d?p.amplitude:1.f, m=i<d?p.mean:0.f;
      c.x[i]=m+((j%2)?a:-a);
      c.dy[i]=float(int(i%7)-3)*.125f;}
    out.push_back(std::move(c));
  }
  for(float a:{269.75f,400.f}){
    Case c=out.front();c.name=a<300?"sparse_known_failure":"sparse_large";
    std::fill(c.x.begin(),c.x.begin()+d,0.f);c.x[0]=-a;c.x[1]=a;
    out.push_back(std::move(c));
  }
  return out;
}
template<class T> std::vector<T> cast(const std::vector<float>& v){return {v.begin(),v.end()};}
inline Result<double> doubleReference(const Case& c,size_t d,bool scaled=false){
 return reference(cast<double>(c.x),cast<double>(c.dy),cast<double>(c.gamma),cast<double>(c.beta),d,1e-5,scaled);
}
inline double finiteDifferenceError(const Case& c,size_t d,double h,bool scaled,bool dynamic=false){
 auto x=cast<double>(c.x),dy=cast<double>(c.dy),g=cast<double>(c.gamma),b=cast<double>(c.beta);
 auto analytic=reference(x,dy,g,b,d,1e-5,scaled,dynamic);double maxError=0;
 auto loss=[&](){auto o=reference(x,dy,g,b,d,1e-5,scaled,dynamic);double l=0;for(size_t i=0;i<x.size();++i)l+=o.y[i]*dy[i];return l;};
 for(int variable=0;variable<3;++variable){auto& v=variable==0?x:variable==1?g:b;
  const auto& a=variable==0?analytic.dx:variable==1?analytic.dg:analytic.db;
  for(size_t i=0;i<v.size();++i){double old=v[i];v[i]=old+h;double plus=loss();v[i]=old-h;double minus=loss();v[i]=old;
    maxError=std::max(maxError,std::abs((plus-minus)/(2*h)-a[i]));}}
 return maxError;
}
template<class T,class U> double error(const std::vector<T>& a,const std::vector<U>& b){
 if(a.size()!=b.size()) return INFINITY;
 double e=0;
 for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))return INFINITY;e=std::max(e,std::abs(double(a[i])-double(b[i])));}return e;
}
} // namespace phonelm::qnn::lnvalidation
