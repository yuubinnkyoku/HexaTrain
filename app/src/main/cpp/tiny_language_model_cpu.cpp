// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "tiny_language_model_cpu.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace phonelm::tiny_lm { namespace {
using P=qnn::TinyTransformerParameters;
using LP=qnn::TinyTransformerLayerParameters;
struct N{std::vector<float>xhat,inv,out,centered,square,variance_eps;};
struct C{N n1,n2;std::vector<float>et,x,q,k,v,ap,ctx,r1,f1,relu,out,logits,prob;};
std::vector<float> mm(const std::vector<float>&a,const std::vector<float>&b,uint32_t r,uint32_t k,uint32_t c){std::vector<float>o(size_t(r)*c);for(uint32_t i=0;i<r;++i)for(uint32_t j=0;j<c;++j){double s=0;for(uint32_t z=0;z<k;++z)s+=double(a[size_t(i)*k+z])*b[size_t(z)*c+j];o[size_t(i)*c+j]=float(s);}return o;}
std::vector<float> atb(const std::vector<float>&a,const std::vector<float>&b,uint32_t r,uint32_t ka,uint32_t cb){std::vector<float>o(size_t(ka)*cb);for(uint32_t i=0;i<ka;++i)for(uint32_t j=0;j<cb;++j){double s=0;for(uint32_t z=0;z<r;++z)s+=double(a[size_t(z)*ka+i])*b[size_t(z)*cb+j];o[size_t(i)*cb+j]=float(s);}return o;}
std::vector<float> abt(const std::vector<float>&a,const std::vector<float>&b,uint32_t ra,uint32_t rb,uint32_t k){std::vector<float>o(size_t(ra)*rb);for(uint32_t i=0;i<ra;++i)for(uint32_t j=0;j<rb;++j){double s=0;for(uint32_t z=0;z<k;++z)s+=double(a[size_t(i)*k+z])*b[size_t(j)*k+z];o[size_t(i)*rb+j]=float(s);}return o;}
void add(std::vector<float>&a,const std::vector<float>&b){for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
void upd(std::vector<float>&n,const std::vector<float>&w,const std::vector<float>&d,float lr);
LP& layer(P& p,uint32_t i){
  if(i==0) return static_cast<LP&>(p);
  return p.layers.at(i-1);
}
const LP& layer(const P& p,uint32_t i){ return layer(const_cast<P&>(p),i); }
void updateLayer(LP& n,const LP&w,const LP&d,float lr){
  upd(n.gamma1,w.gamma1,d.gamma1,lr); upd(n.beta1,w.beta1,d.beta1,lr);
  upd(n.wq,w.wq,d.wq,lr); upd(n.wk,w.wk,d.wk,lr); upd(n.wv,w.wv,d.wv,lr); upd(n.wo,w.wo,d.wo,lr);
  upd(n.gamma2,w.gamma2,d.gamma2,lr); upd(n.beta2,w.beta2,d.beta2,lr); upd(n.w1,w.w1,d.w1,lr); upd(n.w2,w.w2,d.w2,lr);
}
bool exact(const std::vector<float>& v,size_t n){return v.size()==n;}
bool validLayerShape(const LP& p,const Config& c){const size_t d=c.dimension,f=c.feedForwardDimension;return exact(p.gamma1,d)&&exact(p.beta1,d)&&exact(p.gamma2,d)&&exact(p.beta2,d)&&exact(p.wq,d*d)&&exact(p.wk,d*d)&&exact(p.wv,d*d)&&exact(p.wo,d*d)&&exact(p.w1,d*f)&&exact(p.w2,d*f);}
void requireGeneralParameterShape(const Config& c,const P& p){
  if(p.layers.size()!=size_t(c.numLayers-1)||!exact(p.tokenEmbedding,size_t(c.vocabularySize)*c.dimension)||!exact(p.outputProjection,size_t(c.dimension)*c.vocabularySize)||!validLayerShape(layer(p,0),c))throw std::invalid_argument("INVALID_TINY_LM_PARAMETER_SCHEMA");
  for(uint32_t i=1;i<c.numLayers;++i)if(!validLayerShape(layer(p,i),c))throw std::invalid_argument("INVALID_TINY_LM_PARAMETER_SCHEMA");
}
N nf(const Config&c,const std::vector<float>&x,const std::vector<float>&g,const std::vector<float>&b){N n;n.xhat.resize(x.size());n.inv.resize(c.tokens);n.out.resize(x.size());n.centered.resize(x.size());n.square.resize(x.size());n.variance_eps.resize(c.tokens);for(uint32_t r=0;r<c.tokens;++r){double m=0,v=0;for(uint32_t d=0;d<c.dimension;++d)m+=x[size_t(r)*c.dimension+d];m/=c.dimension;for(uint32_t d=0;d<c.dimension;++d){double z=x[size_t(r)*c.dimension+d]-m;v+=z*z;n.centered[size_t(r)*c.dimension+d]=float(z);}v/=c.dimension;n.variance_eps[r]=float(v+c.epsilon);n.inv[r]=float(1/std::sqrt(v+c.epsilon));for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;n.square[i]=n.centered[i]*n.centered[i];n.xhat[i]=(x[i]-float(m))*n.inv[r];n.out[i]=n.xhat[i]*g[d]+b[d];}}return n;}
void nb(const Config&c,const std::vector<float>&dy,const N&n,const std::vector<float>&g,std::vector<float>&dx,std::vector<float>&dg,std::vector<float>&db){dx.resize(dy.size());dg.assign(c.dimension,0);db.assign(c.dimension,0);for(uint32_t r=0;r<c.tokens;++r){double s=0,sx=0;for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;double z=dy[i]*g[d];s+=z;sx+=z*n.xhat[i];dg[d]+=dy[i]*n.xhat[i];db[d]+=dy[i];}for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;double z=dy[i]*g[d];dx[i]=float(n.inv[r]/c.dimension*(c.dimension*z-s-n.xhat[i]*sx));}}}
C fw(const Config&c,const std::vector<float>&oh,const P&w){C z;z.et=mm(oh,w.tokenEmbedding,c.tokens,c.vocabularySize,c.dimension);z.x=z.et;add(z.x,fixedPosition(c));z.n1=nf(c,z.x,w.gamma1,w.beta1);z.q=mm(z.n1.out,w.wq,c.tokens,c.dimension,c.dimension);z.k=mm(z.n1.out,w.wk,c.tokens,c.dimension,c.dimension);z.v=mm(z.n1.out,w.wv,c.tokens,c.dimension,c.dimension);auto sc=abt(z.q,z.k,c.tokens,c.tokens,c.dimension);z.ap.assign(size_t(c.tokens)*c.tokens,0);float scale=1/std::sqrt(float(c.dimension));for(uint32_t r=0;r<c.tokens;++r){float mx=-std::numeric_limits<float>::infinity();for(uint32_t j=0;j<=r;++j)mx=std::max(mx,sc[size_t(r)*c.tokens+j]*scale);double s=0;for(uint32_t j=0;j<=r;++j){float e=std::exp(sc[size_t(r)*c.tokens+j]*scale-mx);z.ap[size_t(r)*c.tokens+j]=e;s+=e;}for(uint32_t j=0;j<=r;++j)z.ap[size_t(r)*c.tokens+j]/=float(s);}z.ctx=mm(z.ap,z.v,c.tokens,c.tokens,c.dimension);z.r1=z.x;add(z.r1,mm(z.ctx,w.wo,c.tokens,c.dimension,c.dimension));z.n2=nf(c,z.r1,w.gamma2,w.beta2);z.f1=mm(z.n2.out,w.w1,c.tokens,c.dimension,c.feedForwardDimension);z.relu=z.f1;for(float&v:z.relu)v=std::max(0.f,v);z.out=z.r1;add(z.out,mm(z.relu,w.w2,c.tokens,c.feedForwardDimension,c.dimension));z.logits=mm(z.out,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);z.prob.resize(z.logits.size());for(uint32_t r=0;r<c.tokens;++r){size_t base=size_t(r)*c.vocabularySize;float mx=*std::max_element(z.logits.begin()+base,z.logits.begin()+base+c.vocabularySize);double s=0;for(uint32_t j=0;j<c.vocabularySize;++j){float e=std::exp(z.logits[base+j]-mx);z.prob[base+j]=e;s+=e;}for(uint32_t j=0;j<c.vocabularySize;++j)z.prob[base+j]/=float(s);}return z;}
struct GL { N n1,n2; std::vector<float> x,q,k,v,prob,ctx,r1,f1,relu,out; };
struct GF { std::vector<float> embedded, logits, prob; std::vector<GL> layers; };
GF generalForward(const Config& c,const std::vector<float>&oh,const P&w){
  GF g; g.embedded=mm(oh,w.tokenEmbedding,c.tokens,c.vocabularySize,c.dimension); std::vector<float>x=g.embedded; add(x,fixedPosition(c)); g.embedded=x;
  const uint32_t dh=c.dimension/c.numHeads;
  for(uint32_t li=0;li<c.numLayers;++li){ const LP& p=layer(w,li); GL z; z.x=x; z.n1=nf(c,x,p.gamma1,p.beta1); z.q=mm(z.n1.out,p.wq,c.tokens,c.dimension,c.dimension); z.k=mm(z.n1.out,p.wk,c.tokens,c.dimension,c.dimension); z.v=mm(z.n1.out,p.wv,c.tokens,c.dimension,c.dimension); z.prob.assign(size_t(c.numHeads)*c.tokens*c.tokens,0); z.ctx.assign(size_t(c.tokens)*c.dimension,0);
    const float scale=1/std::sqrt(float(dh));
    if(c.numHeads==1){auto scores=abt(z.q,z.k,c.tokens,c.tokens,c.dimension);for(uint32_t r=0;r<c.tokens;++r){float mx=-std::numeric_limits<float>::infinity();for(uint32_t j=0;j<=r;++j)mx=std::max(mx,scores[size_t(r)*c.tokens+j]*scale);double sum=0;for(uint32_t j=0;j<=r;++j){float e=std::exp(scores[size_t(r)*c.tokens+j]*scale-mx);z.prob[size_t(r)*c.tokens+j]=e;sum+=e;}for(uint32_t j=0;j<=r;++j)z.prob[size_t(r)*c.tokens+j]/=float(sum);}z.ctx=mm(z.prob,z.v,c.tokens,c.tokens,c.dimension);}else for(uint32_t h=0;h<c.numHeads;++h)for(uint32_t r=0;r<c.tokens;++r){ size_t base=(size_t(h)*c.tokens+r)*c.tokens; float mx=-std::numeric_limits<float>::infinity(); for(uint32_t j=0;j<=r;++j){ double s=0;for(uint32_t d=0;d<dh;++d)s+=double(z.q[size_t(r)*c.dimension+h*dh+d])*z.k[size_t(j)*c.dimension+h*dh+d]; mx=std::max(mx,float(s)*scale); } double sum=0;for(uint32_t j=0;j<=r;++j){double s=0;for(uint32_t d=0;d<dh;++d)s+=double(z.q[size_t(r)*c.dimension+h*dh+d])*z.k[size_t(j)*c.dimension+h*dh+d]; float e=std::exp(float(s)*scale-mx);z.prob[base+j]=e;sum+=e;}for(uint32_t j=0;j<=r;++j){float a=z.prob[base+j]/float(sum);z.prob[base+j]=a;for(uint32_t d=0;d<dh;++d)z.ctx[size_t(r)*c.dimension+h*dh+d]+=a*z.v[size_t(j)*c.dimension+h*dh+d];}}
    z.r1=x;add(z.r1,mm(z.ctx,p.wo,c.tokens,c.dimension,c.dimension)); z.n2=nf(c,z.r1,p.gamma2,p.beta2);z.f1=mm(z.n2.out,p.w1,c.tokens,c.dimension,c.feedForwardDimension);z.relu=z.f1;for(float&v:z.relu)v=std::max(0.f,v);z.out=z.r1;add(z.out,mm(z.relu,p.w2,c.tokens,c.feedForwardDimension,c.dimension));x=z.out;g.layers.push_back(std::move(z)); }
  g.logits=mm(x,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);g.prob.resize(g.logits.size());for(uint32_t r=0;r<c.tokens;++r){size_t b=size_t(r)*c.vocabularySize;float mx=*std::max_element(g.logits.begin()+b,g.logits.begin()+b+c.vocabularySize);double s=0;for(uint32_t j=0;j<c.vocabularySize;++j){float e=std::exp(g.logits[b+j]-mx);g.prob[b+j]=e;s+=e;}for(uint32_t j=0;j<c.vocabularySize;++j)g.prob[b+j]/=float(s);}return g;
}
void upd(std::vector<float>&n,const std::vector<float>&w,const std::vector<float>&d,float lr){n.resize(w.size());for(size_t i=0;i<w.size();++i)n[i]=w[i]-lr*d[i];}
using M=std::vector<float>P::*;std::vector<std::pair<const char*,M>> fields(){return{{"token_embedding",&P::tokenEmbedding},{"output_projection",&P::outputProjection},{"wq",&P::wq},{"wk",&P::wk},{"wv",&P::wv},{"wo",&P::wo},{"norm1_gamma",&P::gamma1},{"norm1_beta",&P::beta1},{"norm2_gamma",&P::gamma2},{"norm2_beta",&P::beta2},{"ffn_w1",&P::w1},{"ffn_w2",&P::w2}};}
}
bool validateConfig(const Config& c,std::string* error){
  const auto estimate=resourceEstimate(c);
  if(!estimate.ok){if(error)*error=estimate.failureClassification+": "+estimate.detail;return false;}
  return true;
}
transformer::ResourceEstimate resourceEstimate(const Config& c){
  return transformer::estimateTrainingResources(c.tokens,c.vocabularySize,c.dimension,
      c.feedForwardDimension,c.numLayers,c.numHeads);
}
uint32_t headDim(const Config& c){std::string e;if(!validateConfig(c,&e))throw std::invalid_argument(e);return c.dimension/c.numHeads;}
std::vector<ParameterInfo> parameterRegistry(const P&p){std::vector<ParameterInfo>r;auto addLayer=[&](uint32_t i,const LP&l){std::ostringstream index;index<<std::setw(3)<<std::setfill('0')<<i;const std::string pre="layer_"+index.str()+".";for(auto e:{std::pair<const char*,const std::vector<float>*>{"norm1_gamma",&l.gamma1},{"norm1_beta",&l.beta1},{"wq",&l.wq},{"wk",&l.wk},{"wv",&l.wv},{"wo",&l.wo},{"norm2_gamma",&l.gamma2},{"norm2_beta",&l.beta2},{"ffn_w1",&l.w1},{"ffn_w2",&l.w2}})r.push_back({pre+e.first,e.second});};r.push_back({"token_embedding",&p.tokenEmbedding});addLayer(0,layer(p,0));for(uint32_t i=1;i<=p.layers.size();++i)addLayer(i,p.layers[i-1]);r.push_back({"output_projection",&p.outputProjection});return r;}
size_t parameterElementCount(const P&p){size_t n=0;for(const auto&e:parameterRegistry(p))n+=e.values->size();return n;}
bool storageRangesHaveNoAliases(const std::vector<ParameterInfo>&r){for(size_t i=0;i<r.size();++i)for(size_t j=0;j<i;++j){if(!r[i].values||!r[j].values)return false;if(r[i].values->empty()||r[j].values->empty())continue;const auto ai=reinterpret_cast<std::uintptr_t>(r[i].values->data());const auto ae=ai+r[i].values->size()*sizeof(float);const auto bi=reinterpret_cast<std::uintptr_t>(r[j].values->data());const auto be=bi+r[j].values->size()*sizeof(float);if(ai<be&&bi<ae)return false;}return true;}
bool parameterStorageHasNoAliases(const P&p){return storageRangesHaveNoAliases(parameterRegistry(p));}
std::vector<float> oneHot(const std::vector<uint32_t>&t,uint32_t v){std::vector<float>o(size_t(t.size())*v);for(size_t i=0;i<t.size();++i){if(t[i]>=v)throw std::invalid_argument("token");o[i*v+t[i]]=1;}return o;}
std::vector<float> fixedPosition(const Config&c){std::vector<float>p(size_t(c.tokens)*c.dimension);for(uint32_t t=0;t<c.tokens;++t)for(uint32_t d=0;d<c.dimension;++d){float f=std::pow(10000.f,-float(d&~1u)/float(c.dimension));p[size_t(t)*c.dimension+d]=((d&1u)?std::cos(float(t)*f):std::sin(float(t)*f))*.05f;}return p;}
P initialParameters(const Config&c,uint32_t seed){std::string error;if(!validateConfig(c,&error))throw std::invalid_argument(error);P p;p.gamma1.assign(c.dimension,1);p.beta1.assign(c.dimension,0);p.gamma2.assign(c.dimension,1);p.beta2.assign(c.dimension,0);auto fill=[&](std::vector<float>&v,size_t n,uint32_t phase,float scale){v.resize(n);uint32_t s=seed*747796405u+phase*2891336453u;for(float&x:v){s=s*1664525u+1013904223u;x=(float(int((s>>8)&65535u))/32767.5f-1)*scale;}};fill(p.tokenEmbedding,size_t(c.vocabularySize)*c.dimension,1,.18f);fill(p.wq,size_t(c.dimension)*c.dimension,2,.12f);fill(p.wk,size_t(c.dimension)*c.dimension,3,.12f);fill(p.wv,size_t(c.dimension)*c.dimension,4,.12f);fill(p.wo,size_t(c.dimension)*c.dimension,5,.12f);fill(p.w1,size_t(c.dimension)*c.feedForwardDimension,6,.1f);fill(p.w2,size_t(c.feedForwardDimension)*c.dimension,7,.1f);fill(p.outputProjection,size_t(c.dimension)*c.vocabularySize,8,.16f);
  p.layers.resize(c.numLayers-1);for(uint32_t i=1;i<c.numLayers;++i){LP&l=p.layers[i-1];l.gamma1.assign(c.dimension,1);l.beta1.assign(c.dimension,0);l.gamma2.assign(c.dimension,1);l.beta2.assign(c.dimension,0);uint32_t base=8+i*10;fill(l.wq,size_t(c.dimension)*c.dimension,base+1,.12f);fill(l.wk,size_t(c.dimension)*c.dimension,base+2,.12f);fill(l.wv,size_t(c.dimension)*c.dimension,base+3,.12f);fill(l.wo,size_t(c.dimension)*c.dimension,base+4,.12f);fill(l.w1,size_t(c.dimension)*c.feedForwardDimension,base+5,.1f);fill(l.w2,size_t(c.feedForwardDimension)*c.dimension,base+6,.1f);}return p;}
StepResult generalForwardBackward(const Config&,const std::vector<float>&,const std::vector<float>&,const P&,float);
StepResult forwardBackward(const Config&c,const std::vector<float>&oh,const std::vector<float>&target,const P&w,float lr){std::string error;if(!validateConfig(c,&error))throw std::invalid_argument(error);if(c.numLayers!=1||c.numHeads!=1)return generalForwardBackward(c,oh,target,w,lr);if(oh.size()!=size_t(c.tokens)*c.vocabularySize||target.size()!=oh.size())throw std::invalid_argument("shape");StepResult r;auto z=fw(c,oh,w);r.embeddedInput=z.x;r.transformerOutput=z.out;r.logits=z.logits;r.probabilities=z.prob;r.dLogits.resize(z.logits.size());double loss=0;uint32_t correct=0;for(uint32_t row=0;row<c.tokens;++row){size_t base=size_t(row)*c.vocabularySize;float mx=z.logits[base];uint32_t pred=0,truth=0;for(uint32_t j=0;j<c.vocabularySize;++j){if(z.logits[base+j]>mx){mx=z.logits[base+j];pred=j;}if(target[base+j]>.5f)truth=j;r.dLogits[base+j]=(z.prob[base+j]-target[base+j])/float(c.tokens);}double s=0;for(uint32_t j=0;j<c.vocabularySize;++j)s+=std::exp(double(z.logits[base+j]-mx));loss+=mx+std::log(s)-z.logits[base+truth];correct+=pred==truth;}r.loss=float(loss/c.tokens);r.accuracy=float(correct)/c.tokens;auto&d=r.gradients;d.outputProjection=atb(z.out,r.dLogits,c.tokens,c.dimension,c.vocabularySize);auto dout=abt(r.dLogits,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);d.w2=atb(z.relu,dout,c.tokens,c.feedForwardDimension,c.dimension);auto drelu=abt(dout,w.w2,c.tokens,c.feedForwardDimension,c.dimension);std::vector<float>df1(drelu.size());for(size_t i=0;i<df1.size();++i)df1[i]=z.f1[i]>0?drelu[i]:0;d.w1=atb(z.n2.out,df1,c.tokens,c.dimension,c.feedForwardDimension);auto dn2=abt(df1,w.w1,c.tokens,c.dimension,c.feedForwardDimension);std::vector<float>drn;nb(c,dn2,z.n2,w.gamma2,drn,d.gamma2,d.beta2);auto dr=dout;add(dr,drn);d.wo=atb(z.ctx,dr,c.tokens,c.dimension,c.dimension);auto dc=abt(dr,w.wo,c.tokens,c.dimension,c.dimension);auto dap=abt(dc,z.v,c.tokens,c.tokens,c.dimension);auto dv=atb(z.ap,dc,c.tokens,c.tokens,c.dimension);std::vector<float>ds(size_t(c.tokens)*c.tokens);for(uint32_t row=0;row<c.tokens;++row){double dot=0;for(uint32_t j=0;j<c.tokens;++j)dot+=dap[size_t(row)*c.tokens+j]*z.ap[size_t(row)*c.tokens+j];for(uint32_t j=0;j<c.tokens;++j)ds[size_t(row)*c.tokens+j]=z.ap[size_t(row)*c.tokens+j]*(dap[size_t(row)*c.tokens+j]-float(dot));}float sc=1/std::sqrt(float(c.dimension));auto dq=mm(ds,z.k,c.tokens,c.tokens,c.dimension);auto dk=atb(ds,z.q,c.tokens,c.tokens,c.dimension);for(float&v:dq)v*=sc;for(float&v:dk)v*=sc;d.wq=atb(z.n1.out,dq,c.tokens,c.dimension,c.dimension);d.wk=atb(z.n1.out,dk,c.tokens,c.dimension,c.dimension);d.wv=atb(z.n1.out,dv,c.tokens,c.dimension,c.dimension);auto dn1=abt(dq,w.wq,c.tokens,c.dimension,c.dimension);add(dn1,abt(dk,w.wk,c.tokens,c.dimension,c.dimension));add(dn1,abt(dv,w.wv,c.tokens,c.dimension,c.dimension));std::vector<float>dxn;nb(c,dn1,z.n1,w.gamma1,dxn,d.gamma1,d.beta1);r.dEmbeddedInput=dr;add(r.dEmbeddedInput,dxn);d.tokenEmbedding=atb(oh,r.dEmbeddedInput,c.tokens,c.vocabularySize,c.dimension);upd(r.next.tokenEmbedding,w.tokenEmbedding,d.tokenEmbedding,lr);upd(r.next.outputProjection,w.outputProjection,d.outputProjection,lr);upd(r.next.gamma1,w.gamma1,d.gamma1,lr);upd(r.next.beta1,w.beta1,d.beta1,lr);upd(r.next.wq,w.wq,d.wq,lr);upd(r.next.wk,w.wk,d.wk,lr);upd(r.next.wv,w.wv,d.wv,lr);upd(r.next.wo,w.wo,d.wo,lr);upd(r.next.gamma2,w.gamma2,d.gamma2,lr);upd(r.next.beta2,w.beta2,d.beta2,lr);upd(r.next.w1,w.w1,d.w1,lr);upd(r.next.w2,w.w2,d.w2,lr);return r;}
StepResult generalForwardBackward(const Config&c,const std::vector<float>&oh,const std::vector<float>&target,const P&w,float lr){
  if(oh.size()!=size_t(c.tokens)*c.vocabularySize||target.size()!=oh.size())throw std::invalid_argument("INVALID_TINY_LM_INPUT_SHAPE");
  requireGeneralParameterShape(c,w);
  StepResult r; auto g=generalForward(c,oh,w);r.embeddedInput=g.embedded;r.transformerOutput=g.layers.back().out;r.logits=g.logits;r.probabilities=g.prob;r.dLogits.resize(g.logits.size());r.layerInputGradients.resize(c.numLayers);r.attentionHeadProbabilities.reserve(size_t(c.numLayers)*c.numHeads);const size_t headProbabilityElements=size_t(c.tokens)*c.tokens;for(uint32_t li=0;li<c.numLayers;++li)for(uint32_t h=0;h<c.numHeads;++h){const auto begin=g.layers[li].prob.begin()+size_t(h)*headProbabilityElements;r.attentionHeadProbabilities.emplace_back(begin,begin+headProbabilityElements);}double loss=0;uint32_t correct=0;
  for(uint32_t row=0;row<c.tokens;++row){size_t b=size_t(row)*c.vocabularySize;float mx=g.logits[b];uint32_t pred=0,truth=0;for(uint32_t j=0;j<c.vocabularySize;++j){if(g.logits[b+j]>mx){mx=g.logits[b+j];pred=j;}if(target[b+j]>.5f)truth=j;r.dLogits[b+j]=(g.prob[b+j]-target[b+j])/float(c.tokens);}double s=0;for(uint32_t j=0;j<c.vocabularySize;++j)s+=std::exp(double(g.logits[b+j]-mx));loss+=mx+std::log(s)-g.logits[b+truth];correct+=pred==truth;}r.loss=float(loss/c.tokens);r.accuracy=float(correct)/c.tokens;r.gradients.layers.resize(c.numLayers-1);r.next.layers.resize(c.numLayers-1);
  r.gradients.outputProjection=atb(g.layers.back().out,r.dLogits,c.tokens,c.dimension,c.vocabularySize);std::vector<float> dout=abt(r.dLogits,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);const uint32_t dh=c.dimension/c.numHeads;
  for(uint32_t li=c.numLayers;li-- > 0;){const GL&z=g.layers[li];const LP&p=layer(w,li);LP&d=layer(r.gradients,li);LP&n=layer(r.next,li);d.w2=atb(z.relu,dout,c.tokens,c.feedForwardDimension,c.dimension);auto drelu=abt(dout,p.w2,c.tokens,c.feedForwardDimension,c.dimension);std::vector<float>df1(drelu.size());for(size_t i=0;i<df1.size();++i)df1[i]=z.f1[i]>0?drelu[i]:0;d.w1=atb(z.n2.out,df1,c.tokens,c.dimension,c.feedForwardDimension);auto dn2=abt(df1,p.w1,c.tokens,c.dimension,c.feedForwardDimension);std::vector<float> drn;nb(c,dn2,z.n2,p.gamma2,drn,d.gamma2,d.beta2);std::vector<float> dr=dout;add(dr,drn);d.wo=atb(z.ctx,dr,c.tokens,c.dimension,c.dimension);auto dc=abt(dr,p.wo,c.tokens,c.dimension,c.dimension);std::vector<float>dq(size_t(c.tokens)*c.dimension),dk(dq.size()),dv(dq.size());
    if(c.numHeads==1){auto dap=abt(dc,z.v,c.tokens,c.tokens,c.dimension);dv=atb(z.prob,dc,c.tokens,c.tokens,c.dimension);std::vector<float>ds(size_t(c.tokens)*c.tokens);for(uint32_t rr=0;rr<c.tokens;++rr){double dot=0;for(uint32_t j=0;j<c.tokens;++j)dot+=dap[size_t(rr)*c.tokens+j]*z.prob[size_t(rr)*c.tokens+j];for(uint32_t j=0;j<c.tokens;++j)ds[size_t(rr)*c.tokens+j]=z.prob[size_t(rr)*c.tokens+j]*(dap[size_t(rr)*c.tokens+j]-float(dot));}dq=mm(ds,z.k,c.tokens,c.tokens,c.dimension);dk=atb(ds,z.q,c.tokens,c.tokens,c.dimension);const float scale=1/std::sqrt(float(dh));for(float& value:dq)value*=scale;for(float& value:dk)value*=scale;}else for(uint32_t h=0;h<c.numHeads;++h){std::vector<float> dap(size_t(c.tokens)*c.tokens),ds(dap.size());for(uint32_t rr=0;rr<c.tokens;++rr)for(uint32_t j=0;j<c.tokens;++j)for(uint32_t dd=0;dd<dh;++dd)dap[size_t(rr)*c.tokens+j]+=dc[size_t(rr)*c.dimension+h*dh+dd]*z.v[size_t(j)*c.dimension+h*dh+dd];for(uint32_t rr=0;rr<c.tokens;++rr){double dot=0;size_t pb=(size_t(h)*c.tokens+rr)*c.tokens;for(uint32_t j=0;j<c.tokens;++j)dot+=dap[size_t(rr)*c.tokens+j]*z.prob[pb+j];for(uint32_t j=0;j<c.tokens;++j){ds[size_t(rr)*c.tokens+j]=z.prob[pb+j]*(dap[size_t(rr)*c.tokens+j]-float(dot));for(uint32_t dd=0;dd<dh;++dd)dv[size_t(j)*c.dimension+h*dh+dd]+=z.prob[pb+j]*dc[size_t(rr)*c.dimension+h*dh+dd];}}for(uint32_t rr=0;rr<c.tokens;++rr)for(uint32_t j=0;j<c.tokens;++j)for(uint32_t dd=0;dd<dh;++dd){dq[size_t(rr)*c.dimension+h*dh+dd]+=ds[size_t(rr)*c.tokens+j]*z.k[size_t(j)*c.dimension+h*dh+dd]/std::sqrt(float(dh));dk[size_t(j)*c.dimension+h*dh+dd]+=ds[size_t(rr)*c.tokens+j]*z.q[size_t(rr)*c.dimension+h*dh+dd]/std::sqrt(float(dh));}}
    d.wq=atb(z.n1.out,dq,c.tokens,c.dimension,c.dimension);d.wk=atb(z.n1.out,dk,c.tokens,c.dimension,c.dimension);d.wv=atb(z.n1.out,dv,c.tokens,c.dimension,c.dimension);auto dn1=abt(dq,p.wq,c.tokens,c.dimension,c.dimension);add(dn1,abt(dk,p.wk,c.tokens,c.dimension,c.dimension));add(dn1,abt(dv,p.wv,c.tokens,c.dimension,c.dimension));std::vector<float>dxn;nb(c,dn1,z.n1,p.gamma1,dxn,d.gamma1,d.beta1);dout=dr;add(dout,dxn);r.layerInputGradients[li]=dout;updateLayer(n,p,d,lr);}
  r.dEmbeddedInput=dout;r.gradients.tokenEmbedding=atb(oh,dout,c.tokens,c.vocabularySize,c.dimension);upd(r.next.tokenEmbedding,w.tokenEmbedding,r.gradients.tokenEmbedding,lr);upd(r.next.outputProjection,w.outputProjection,r.gradients.outputProjection,lr);return r;
}
StepResult forwardBackwardGeneralized(const Config& c,const std::vector<float>& input,const std::vector<float>& target,const P& parameters,float learningRate){std::string error;if(!validateConfig(c,&error))throw std::invalid_argument(error);return generalForwardBackward(c,input,target,parameters,learningRate);}
GeneralizedCpuTrace forwardTraceGeneralized(const Config& c, const std::vector<float>& oh, const P& w) {
  std::string error;
  if (!validateConfig(c, &error)) throw std::invalid_argument(error);
  if (oh.size() != size_t(c.tokens) * c.vocabularySize)
    throw std::invalid_argument("INVALID_TINY_LM_INPUT_SHAPE");
  requireGeneralParameterShape(c, w);
  auto g = generalForward(c, oh, w);
  GeneralizedCpuTrace trace;
  trace.embeddedInput = std::move(g.embedded);
  trace.logits = std::move(g.logits);
  trace.probabilities = std::move(g.prob);
  trace.layers.reserve(g.layers.size());
  for (auto& l : g.layers) {
    GeneralizedCpuTrace::Layer layerTrace;
    layerTrace.input = std::move(l.x);
    layerTrace.ln1 = std::move(l.n1.out);
    layerTrace.ln1Centered = std::move(l.n1.centered);
    layerTrace.ln1Square = std::move(l.n1.square);
    layerTrace.ln1VarianceEps = std::move(l.n1.variance_eps);
    layerTrace.ln1Inv = std::move(l.n1.inv);
    layerTrace.q = std::move(l.q);
    layerTrace.k = std::move(l.k);
    layerTrace.v = std::move(l.v);
    layerTrace.probabilities = std::move(l.prob);
    layerTrace.context = std::move(l.ctx);
    layerTrace.residual1 = std::move(l.r1);
    layerTrace.ln2 = std::move(l.n2.out);
    layerTrace.ln2Centered = std::move(l.n2.centered);
    layerTrace.ln2Square = std::move(l.n2.square);
    layerTrace.ln2VarianceEps = std::move(l.n2.variance_eps);
    layerTrace.ln2Inv = std::move(l.n2.inv);
    layerTrace.ff1 = std::move(l.f1);
    layerTrace.relu = std::move(l.relu);
    layerTrace.output = std::move(l.out);
    trace.layers.push_back(std::move(layerTrace));
  }
  return trace;
}
MomentumResult momentumUpdate(const P&current,const P&gradient,const P&velocity,float lr,float momentum){MomentumResult result;for(auto[name,member]:fields()){(void)name;const auto&w=current.*member;const auto&g=gradient.*member;const auto&v=velocity.*member;auto&nv=result.velocity.*member;auto&nw=result.next.*member;nv.resize(w.size());nw.resize(w.size());for(size_t i=0;i<w.size();++i){nv[i]=momentum*v[i]+g[i];nw[i]=w[i]-lr*nv[i];}}result.velocity.layers.resize(current.layers.size());result.next.layers.resize(current.layers.size());for(size_t li=0;li<current.layers.size();++li){const LP&w=current.layers[li],&g=gradient.layers[li],&v=velocity.layers[li];LP&nv=result.velocity.layers[li];LP&nw=result.next.layers[li];auto f=[&](const std::vector<float>&a,const std::vector<float>&b,const std::vector<float>&old,std::vector<float>&m,std::vector<float>&out){m.resize(a.size());out.resize(a.size());for(size_t i=0;i<a.size();++i){m[i]=momentum*old[i]+b[i];out[i]=a[i]-lr*m[i];}};f(w.gamma1,g.gamma1,v.gamma1,nv.gamma1,nw.gamma1);f(w.beta1,g.beta1,v.beta1,nv.beta1,nw.beta1);f(w.wq,g.wq,v.wq,nv.wq,nw.wq);f(w.wk,g.wk,v.wk,nv.wk,nw.wk);f(w.wv,g.wv,v.wv,nv.wv,nw.wv);f(w.wo,g.wo,v.wo,nv.wo,nw.wo);f(w.gamma2,g.gamma2,v.gamma2,nv.gamma2,nw.gamma2);f(w.beta2,g.beta2,v.beta2,nv.beta2,nw.beta2);f(w.w1,g.w1,v.w1,nv.w1,nw.w1);f(w.w2,g.w2,v.w2,nv.w2,nw.w2);}return result;}
AdamResult adamUpdate(const P&current,const P&gradient,const P&firstMoment,
                      const P&secondMoment,float lr,float beta1,float beta2,
                      float epsilon,float firstCorrection,float secondCorrection){
  AdamResult result;
  for(auto[name,member]:fields()){
    (void)name;const auto&w=current.*member;const auto&g=gradient.*member;
    const auto&m=firstMoment.*member;const auto&v=secondMoment.*member;
    auto&nm=result.firstMoment.*member;auto&nv=result.secondMoment.*member;
    auto&mh=result.firstMomentHat.*member;auto&vh=result.secondMomentHat.*member;
    auto&nw=result.next.*member;
    nm.resize(w.size());nv.resize(w.size());mh.resize(w.size());vh.resize(w.size());nw.resize(w.size());
    for(size_t i=0;i<w.size();++i){
      nm[i]=beta1*m[i]+(1-beta1)*g[i];
      nv[i]=beta2*v[i]+(1-beta2)*g[i]*g[i];
      mh[i]=nm[i]*firstCorrection;vh[i]=nv[i]*secondCorrection;
      nw[i]=w[i]-lr*mh[i]/(std::sqrt(vh[i])+epsilon);
    }
  }
  result.firstMoment.layers.resize(current.layers.size());result.secondMoment.layers.resize(current.layers.size());result.firstMomentHat.layers.resize(current.layers.size());result.secondMomentHat.layers.resize(current.layers.size());result.next.layers.resize(current.layers.size());
  for(size_t li=0;li<current.layers.size();++li){const LP&w=current.layers[li],&g=gradient.layers[li],&m=firstMoment.layers[li],&v=secondMoment.layers[li];LP&nm=result.firstMoment.layers[li],&nv=result.secondMoment.layers[li],&mh=result.firstMomentHat.layers[li],&vh=result.secondMomentHat.layers[li],&nw=result.next.layers[li];auto f=[&](const std::vector<float>&a,const std::vector<float>&b,const std::vector<float>&om,const std::vector<float>&ov,std::vector<float>&a1,std::vector<float>&a2,std::vector<float>&h1,std::vector<float>&h2,std::vector<float>&out){a1.resize(a.size());a2.resize(a.size());h1.resize(a.size());h2.resize(a.size());out.resize(a.size());for(size_t i=0;i<a.size();++i){a1[i]=beta1*om[i]+(1-beta1)*b[i];a2[i]=beta2*ov[i]+(1-beta2)*b[i]*b[i];h1[i]=a1[i]*firstCorrection;h2[i]=a2[i]*secondCorrection;out[i]=a[i]-lr*h1[i]/(std::sqrt(h2[i])+epsilon);}};f(w.gamma1,g.gamma1,m.gamma1,v.gamma1,nm.gamma1,nv.gamma1,mh.gamma1,vh.gamma1,nw.gamma1);f(w.beta1,g.beta1,m.beta1,v.beta1,nm.beta1,nv.beta1,mh.beta1,vh.beta1,nw.beta1);f(w.wq,g.wq,m.wq,v.wq,nm.wq,nv.wq,mh.wq,vh.wq,nw.wq);f(w.wk,g.wk,m.wk,v.wk,nm.wk,nv.wk,mh.wk,vh.wk,nw.wk);f(w.wv,g.wv,m.wv,v.wv,nm.wv,nv.wv,mh.wv,vh.wv,nw.wv);f(w.wo,g.wo,m.wo,v.wo,nm.wo,nv.wo,mh.wo,vh.wo,nw.wo);f(w.gamma2,g.gamma2,m.gamma2,v.gamma2,nm.gamma2,nv.gamma2,mh.gamma2,vh.gamma2,nw.gamma2);f(w.beta2,g.beta2,m.beta2,v.beta2,nm.beta2,nv.beta2,mh.beta2,vh.beta2,nw.beta2);f(w.w1,g.w1,m.w1,v.w1,nm.w1,nv.w1,mh.w1,vh.w1,nw.w1);f(w.w2,g.w2,m.w2,v.w2,nm.w2,nv.w2,mh.w2,vh.w2,nw.w2);}
  return result;
}
GradientCheckResult gradientCheck(uint32_t seed,float eps){Config c;c.vocabularySize=8;c.tokens=3;c.dimension=4;c.feedForwardDimension=8;auto x=oneHot({0,1,2},8),y=oneHot({1,2,3},8);auto p=initialParameters(c,seed);auto a=forwardBackward(c,x,y,p,0);GradientCheckResult r;std::ostringstream s;s<<std::setprecision(9);for(auto[name,m]:fields()){auto&v=p.*m;const auto&g=a.gradients.*m;size_t ix[3]{0,v.size()/2,v.size()-1};float ma=0,mr=0;for(size_t i:ix){float old=v[i];v[i]=old+eps;float plus=forwardBackward(c,x,y,p,0).loss;v[i]=old-eps;float minus=forwardBackward(c,x,y,p,0).loss;v[i]=old;float n=(plus-minus)/(2*eps),ae=std::abs(g[i]-n),re=ae/std::max(1e-4f,std::abs(g[i])+std::abs(n));ma=std::max(ma,ae);mr=std::max(mr,re);s<<"gradient_check_parameter="<<name<<" index="<<i<<" analytic="<<g[i]<<" numeric="<<n<<" absolute_error="<<ae<<" relative_error="<<re<<'\n';}s<<"gradient_check_parameter_summary="<<name<<" max_absolute_error="<<ma<<" max_relative_error="<<mr<<'\n';r.maximumAbsoluteError=std::max(r.maximumAbsoluteError,ma);r.maximumRelativeError=std::max(r.maximumRelativeError,mr);}r.passed=r.maximumAbsoluteError<=2e-3f;r.report=s.str();return r;}
}
