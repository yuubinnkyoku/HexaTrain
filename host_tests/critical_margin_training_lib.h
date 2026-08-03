// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only margin-aware training support for the L19 critical-margin
// stabilization investigation.
//
// The forward arithmetic below is copied VERBATIM from
// app/src/main/cpp/tiny_language_model_cpu.cpp (generalForward and the
// gradient half of generalForwardBackward). The CPU reference implementation
// is the single source of truth for training arithmetic; this copy exists
// only because the public API cannot inject a modified dLogits into the
// backward pass. Do not refactor these functions. The probe asserts bitwise
// gradient equality against tiny_lm::forwardBackward for the CE dLogits and
// an end-to-end lambda==0 run must reproduce the canonical trajectory
// exactly.
#ifndef CRITICAL_MARGIN_TRAINING_LIB_H
#define CRITICAL_MARGIN_TRAINING_LIB_H

#include "tiny_language_model_cpu.h"
#include "depth_quality_lib.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace phonelm::critical_margin::train {

namespace tiny = phonelm::tiny_lm;
using Config = tiny::Config;
using P = qnn::TinyTransformerParameters;
using LP = qnn::TinyTransformerLayerParameters;
struct N{std::vector<float>xhat,inv,out;};
std::vector<float> mm(const std::vector<float>&a,const std::vector<float>&b,uint32_t r,uint32_t k,uint32_t c){std::vector<float>o(size_t(r)*c);for(uint32_t i=0;i<r;++i)for(uint32_t j=0;j<c;++j){double s=0;for(uint32_t z=0;z<k;++z)s+=double(a[size_t(i)*k+z])*b[size_t(z)*c+j];o[size_t(i)*c+j]=float(s);}return o;}
std::vector<float> atb(const std::vector<float>&a,const std::vector<float>&b,uint32_t r,uint32_t ka,uint32_t cb){std::vector<float>o(size_t(ka)*cb);for(uint32_t i=0;i<ka;++i)for(uint32_t j=0;j<cb;++j){double s=0;for(uint32_t z=0;z<r;++z)s+=double(a[size_t(z)*ka+i])*b[size_t(z)*cb+j];o[size_t(i)*cb+j]=float(s);}return o;}
std::vector<float> abt(const std::vector<float>&a,const std::vector<float>&b,uint32_t ra,uint32_t rb,uint32_t k){std::vector<float>o(size_t(ra)*rb);for(uint32_t i=0;i<ra;++i)for(uint32_t j=0;j<rb;++j){double s=0;for(uint32_t z=0;z<k;++z)s+=double(a[size_t(i)*k+z])*b[size_t(j)*k+z];o[size_t(i)*rb+j]=float(s);}return o;}
void add(std::vector<float>&a,const std::vector<float>&b){for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
LP& layer(P& p,uint32_t i){
  if(i==0) return static_cast<LP&>(p);
  return p.layers.at(i-1);
}
const LP& layer(const P& p,uint32_t i){ return layer(const_cast<P&>(p),i); }
bool exact(const std::vector<float>& v,size_t n){return v.size()==n;}
bool validLayerShape(const LP& p,const Config& c){const size_t d=c.dimension,f=c.feedForwardDimension;return exact(p.gamma1,d)&&exact(p.beta1,d)&&exact(p.gamma2,d)&&exact(p.beta2,d)&&exact(p.wq,d*d)&&exact(p.wk,d*d)&&exact(p.wv,d*d)&&exact(p.wo,d*d)&&exact(p.w1,d*f)&&exact(p.w2,d*f);}
void requireGeneralParameterShape(const Config& c,const P& p){
  if(p.layers.size()!=size_t(c.numLayers-1)||!exact(p.tokenEmbedding,size_t(c.vocabularySize)*c.dimension)||!exact(p.outputProjection,size_t(c.dimension)*c.vocabularySize)||!validLayerShape(layer(p,0),c))throw std::invalid_argument("INVALID_TINY_LM_PARAMETER_SCHEMA");
  for(uint32_t i=1;i<c.numLayers;++i)if(!validLayerShape(layer(p,i),c))throw std::invalid_argument("INVALID_TINY_LM_PARAMETER_SCHEMA");
}
N nf(const Config&c,const std::vector<float>&x,const std::vector<float>&g,const std::vector<float>&b){N n;n.xhat.resize(x.size());n.inv.resize(c.tokens);n.out.resize(x.size());for(uint32_t r=0;r<c.tokens;++r){double m=0,v=0;for(uint32_t d=0;d<c.dimension;++d)m+=x[size_t(r)*c.dimension+d];m/=c.dimension;for(uint32_t d=0;d<c.dimension;++d){double z=x[size_t(r)*c.dimension+d]-m;v+=z*z;}v/=c.dimension;n.inv[r]=float(1/std::sqrt(v+c.epsilon));for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;n.xhat[i]=(x[i]-float(m))*n.inv[r];n.out[i]=n.xhat[i]*g[d]+b[d];}}return n;}
void nb(const Config&c,const std::vector<float>&dy,const N&n,const std::vector<float>&g,std::vector<float>&dx,std::vector<float>&dg,std::vector<float>&db){dx.resize(dy.size());dg.assign(c.dimension,0);db.assign(c.dimension,0);for(uint32_t r=0;r<c.tokens;++r){double s=0,sx=0;for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;double z=dy[i]*g[d];s+=z;sx+=z*n.xhat[i];dg[d]+=dy[i]*n.xhat[i];db[d]+=dy[i];}for(uint32_t d=0;d<c.dimension;++d){size_t i=size_t(r)*c.dimension+d;double z=dy[i]*g[d];dx[i]=float(n.inv[r]/c.dimension*(c.dimension*z-s-n.xhat[i]*sx));}}}
struct GL { N n1,n2; std::vector<float> x,q,k,v,prob,ctx,r1,f1,relu,out; };
struct GF { std::vector<float> embedded, logits, prob; std::vector<GL> layers; };
std::vector<float> fixedPositionCpu(const Config&c){std::vector<float>p(size_t(c.tokens)*c.dimension);for(uint32_t t=0;t<c.tokens;++t)for(uint32_t d=0;d<c.dimension;++d){float f=std::pow(10000.f,-float(d&~1u)/float(c.dimension));p[size_t(t)*c.dimension+d]=((d&1u)?std::cos(float(t)*f):std::sin(float(t)*f))*.05f;}return p;}
GF generalForward(const Config& c,const std::vector<float>&oh,const P&w){
  GF g; g.embedded=mm(oh,w.tokenEmbedding,c.tokens,c.vocabularySize,c.dimension); std::vector<float>x=g.embedded; add(x,fixedPositionCpu(c)); g.embedded=x;
  const uint32_t dh=c.dimension/c.numHeads;
  for(uint32_t li=0;li<c.numLayers;++li){ const LP& p=layer(w,li); GL z; z.x=x; z.n1=nf(c,x,p.gamma1,p.beta1); z.q=mm(z.n1.out,p.wq,c.tokens,c.dimension,c.dimension); z.k=mm(z.n1.out,p.wk,c.tokens,c.dimension,c.dimension); z.v=mm(z.n1.out,p.wv,c.tokens,c.dimension,c.dimension); z.prob.assign(size_t(c.numHeads)*c.tokens*c.tokens,0); z.ctx.assign(size_t(c.tokens)*c.dimension,0);
    const float scale=1/std::sqrt(float(dh));
    if(c.numHeads==1){auto scores=abt(z.q,z.k,c.tokens,c.tokens,c.dimension);for(uint32_t r=0;r<c.tokens;++r){float mx=-std::numeric_limits<float>::infinity();for(uint32_t j=0;j<=r;++j)mx=std::max(mx,scores[size_t(r)*c.tokens+j]*scale);double sum=0;for(uint32_t j=0;j<=r;++j){float e=std::exp(scores[size_t(r)*c.tokens+j]*scale-mx);z.prob[size_t(r)*c.tokens+j]=e;sum+=e;}for(uint32_t j=0;j<=r;++j)z.prob[size_t(r)*c.tokens+j]/=float(sum);}z.ctx=mm(z.prob,z.v,c.tokens,c.tokens,c.dimension);}else for(uint32_t h=0;h<c.numHeads;++h)for(uint32_t r=0;r<c.tokens;++r){ size_t base=(size_t(h)*c.tokens+r)*c.tokens; float mx=-std::numeric_limits<float>::infinity(); for(uint32_t j=0;j<=r;++j){ double s=0;for(uint32_t d=0;d<dh;++d)s+=double(z.q[size_t(r)*c.dimension+h*dh+d])*z.k[size_t(j)*c.dimension+h*dh+d]; mx=std::max(mx,float(s)*scale); } double sum=0;for(uint32_t j=0;j<=r;++j){double s=0;for(uint32_t d=0;d<dh;++d)s+=double(z.q[size_t(r)*c.dimension+h*dh+d])*z.k[size_t(j)*c.dimension+h*dh+d]; float e=std::exp(float(s)*scale-mx);z.prob[base+j]=e;sum+=e;}for(uint32_t j=0;j<=r;++j){float a=z.prob[base+j]/float(sum);z.prob[base+j]=a;for(uint32_t d=0;d<dh;++d)z.ctx[size_t(r)*c.dimension+h*dh+d]+=a*z.v[size_t(j)*c.dimension+h*dh+d];}}
    z.r1=x;add(z.r1,mm(z.ctx,p.wo,c.tokens,c.dimension,c.dimension)); z.n2=nf(c,z.r1,p.gamma2,p.beta2);z.f1=mm(z.n2.out,p.w1,c.tokens,c.dimension,c.feedForwardDimension);z.relu=z.f1;for(float&v:z.relu)v=std::max(0.f,v);z.out=z.r1;add(z.out,mm(z.relu,p.w2,c.tokens,c.feedForwardDimension,c.dimension));x=z.out;g.layers.push_back(std::move(z)); }
  g.logits=mm(x,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);g.prob.resize(g.logits.size());for(uint32_t r=0;r<c.tokens;++r){size_t b=size_t(r)*c.vocabularySize;float mx=*std::max_element(g.logits.begin()+b,g.logits.begin()+b+c.vocabularySize);double s=0;for(uint32_t j=0;j<c.vocabularySize;++j){float e=std::exp(g.logits[b+j]-mx);g.prob[b+j]=e;s+=e;}for(uint32_t j=0;j<c.vocabularySize;++j)g.prob[b+j]/=float(s);}return g;
}

// Gradient-only backward: identical arithmetic to the gradient half of
// generalForwardBackward, without the SGD update (Adam is applied by the
// caller). dLogits is caller-controlled so margin-aware training can inject
// a modified dLogits.
P generalBackwardGradients(const Config& c, const GF& g,
                           const std::vector<float>& oh, const P& w,
                           const std::vector<float>& dLogits) {
  P out;
  out.layers.resize(c.numLayers - 1);
    out.outputProjection=atb(g.layers.back().out,dLogits,c.tokens,c.dimension,c.vocabularySize);std::vector<float> dout=abt(dLogits,w.outputProjection,c.tokens,c.dimension,c.vocabularySize);const uint32_t dh=c.dimension/c.numHeads;
    for(uint32_t li=c.numLayers;li-- > 0;){const GL&z=g.layers[li];const LP&p=layer(w,li);LP&d=layer(out,li);d.w2=atb(z.relu,dout,c.tokens,c.feedForwardDimension,c.dimension);auto drelu=abt(dout,p.w2,c.tokens,c.feedForwardDimension,c.dimension);std::vector<float>df1(drelu.size());for(size_t i=0;i<df1.size();++i)df1[i]=z.f1[i]>0?drelu[i]:0;d.w1=atb(z.n2.out,df1,c.tokens,c.dimension,c.feedForwardDimension);auto dn2=abt(df1,p.w1,c.tokens,c.dimension,c.feedForwardDimension);std::vector<float> drn;nb(c,dn2,z.n2,p.gamma2,drn,d.gamma2,d.beta2);std::vector<float> dr=dout;add(dr,drn);d.wo=atb(z.ctx,dr,c.tokens,c.dimension,c.dimension);auto dc=abt(dr,p.wo,c.tokens,c.dimension,c.dimension);std::vector<float>dq(size_t(c.tokens)*c.dimension),dk(dq.size()),dv(dq.size());
      if(c.numHeads==1){auto dap=abt(dc,z.v,c.tokens,c.tokens,c.dimension);dv=atb(z.prob,dc,c.tokens,c.tokens,c.dimension);std::vector<float>ds(size_t(c.tokens)*c.tokens);for(uint32_t rr=0;rr<c.tokens;++rr){double dot=0;for(uint32_t j=0;j<c.tokens;++j)dot+=dap[size_t(rr)*c.tokens+j]*z.prob[size_t(rr)*c.tokens+j];for(uint32_t j=0;j<c.tokens;++j)ds[size_t(rr)*c.tokens+j]=z.prob[size_t(rr)*c.tokens+j]*(dap[size_t(rr)*c.tokens+j]-float(dot));}dq=mm(ds,z.k,c.tokens,c.tokens,c.dimension);dk=atb(ds,z.q,c.tokens,c.tokens,c.dimension);const float scale=1/std::sqrt(float(dh));for(float& value:dq)value*=scale;for(float& value:dk)value*=scale;}else for(uint32_t h=0;h<c.numHeads;++h){std::vector<float> dap(size_t(c.tokens)*c.tokens),ds(dap.size());for(uint32_t rr=0;rr<c.tokens;++rr)for(uint32_t j=0;j<c.tokens;++j)for(uint32_t dd=0;dd<dh;++dd)dap[size_t(rr)*c.tokens+j]+=dc[size_t(rr)*c.dimension+h*dh+dd]*z.v[size_t(j)*c.dimension+h*dh+dd];for(uint32_t rr=0;rr<c.tokens;++rr){double dot=0;size_t pb=(size_t(h)*c.tokens+rr)*c.tokens;for(uint32_t j=0;j<c.tokens;++j)dot+=dap[size_t(rr)*c.tokens+j]*z.prob[pb+j];for(uint32_t j=0;j<c.tokens;++j){ds[size_t(rr)*c.tokens+j]=z.prob[pb+j]*(dap[size_t(rr)*c.tokens+j]-float(dot));for(uint32_t dd=0;dd<dh;++dd)dv[size_t(j)*c.dimension+h*dh+dd]+=z.prob[pb+j]*dc[size_t(rr)*c.dimension+h*dh+dd];}}for(uint32_t rr=0;rr<c.tokens;++rr)for(uint32_t j=0;j<c.tokens;++j)for(uint32_t dd=0;dd<dh;++dd){dq[size_t(rr)*c.dimension+h*dh+dd]+=ds[size_t(rr)*c.tokens+j]*z.k[size_t(j)*c.dimension+h*dh+dd]/std::sqrt(float(dh));dk[size_t(j)*c.dimension+h*dh+dd]+=ds[size_t(rr)*c.tokens+j]*z.q[size_t(rr)*c.dimension+h*dh+dd]/std::sqrt(float(dh));}}
      d.wq=atb(z.n1.out,dq,c.tokens,c.dimension,c.dimension);d.wk=atb(z.n1.out,dk,c.tokens,c.dimension,c.dimension);d.wv=atb(z.n1.out,dv,c.tokens,c.dimension,c.dimension);auto dn1=abt(dq,p.wq,c.tokens,c.dimension,c.dimension);add(dn1,abt(dk,p.wk,c.tokens,c.dimension,c.dimension));add(dn1,abt(dv,p.wv,c.tokens,c.dimension,c.dimension));std::vector<float>dxn;nb(c,dn1,z.n1,p.gamma1,dxn,d.gamma1,d.beta1);dout=dr;add(dout,dxn);}
    out.tokenEmbedding=atb(oh,dout,c.tokens,c.vocabularySize,c.dimension);return out;
}


// ---------------------------------------------------------------------------
// Margin-aware loss (preregistered families)
// ---------------------------------------------------------------------------

struct MarginLossSpec {
  enum class Family { PairwiseMarginCe, SequenceWorstMarginCe };
  Family family = Family::PairwiseMarginCe;
  float delta = 0.5f;   // pairwise margin floor (logit units)
  float tau = 0.5f;     // sequence-worst temperature
  float lambda = 1.0f;  // margin term weight
};

struct MarginStepSummary {
  double nll = 0.0;         // canonical CE loss (mean over tokens)
  double marginTerm = 0.0;  // margin penalty
  double total = 0.0;       // nll + marginTerm
  float accuracy = 0.0f;
  double meanMargin = 0.0;
  double criticalShare = 0.0;  // fraction of tokens with margin < 0
  double gradientNorm = 0.0;
};

struct MarginForwardResult {
  MarginStepSummary summary;
  P gradients;
  GF forward;
  std::vector<float> dLogits;
};

// Forward plus a backward pass over the augmented dLogits (CE dLogits +
// margin term). Arithmetic is identical to tiny_lm::forwardBackward except
// that dLogits is modified before the backward, so the returned gradients
// include the margin penalty.
//
// PAIRWISE_MARGIN_CE: margin_r = logit[truth] minus the best competing
//   logit; penalty = lambda * mean_r max(0, delta - margin_r); gradient
//   pushes the truth logit up and the best competitor down by
//   lambda/tokens per deficient token.
// SEQUENCE_WORST_MARGIN_CE: penalty = lambda * tau * logsumexp_r(-margin_r/tau)
//   divided by tokens (the effective mean-per-token objective, matching the
//   applied gradient scale); gradient weights rows by softmax(-margin/tau),
//   emphasizing the worst margins; the tau=0 limit is max_r(-margin_r).
MarginForwardResult marginForwardBackward(const Config& c,
    const std::vector<float>& oh, const std::vector<float>& target,
    const P& w, const MarginLossSpec& spec);

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

struct MarginTrainingRun {
  MarginLossSpec spec;
  std::uint32_t seed = 0;
  int steps = 0;
  P finalParameters;    // checkpoint at `steps` (pre-update), like FormRun
  P stabilityParameters;  // checkpoint at steps-16 (pre-update)
  std::vector<MarginStepSummary> stepSummaries;  // one per step, 1..steps
  bool finite = true;
  int lastParityStep = 0;  // last step with bitwise CE-gradient parity PASS
};

// Called with the pre-update parameters at every requested evaluation step.
using MarginCheckpointCallback =
    void (*)(int step, const qnn::TinyTransformerParameters& params,
             void* context);

// Replicates the runFormalCpu LEGACY loop (Adam, stability learning rate,
// batch pattern (step-1)%4, applyInitStability(LEGACY)) with the
// margin-augmented backward in place of the canonical forwardBackward.
MarginTrainingRun runMarginTraining(const tiny::Config& config,
    std::uint32_t seed, const MarginLossSpec& spec, int steps,
    const std::vector<int>& evaluationSteps, int parityCheckEvery,
    MarginCheckpointCallback onCheckpoint, void* context);

// ---------------------------------------------------------------------------
// Inline definitions
// ---------------------------------------------------------------------------

inline MarginForwardResult marginForwardBackward(const Config& c,
    const std::vector<float>& oh, const std::vector<float>& target,
    const P& w, const MarginLossSpec& spec) {
  std::string error;
  if (!tiny::validateConfig(c, &error))
    throw std::invalid_argument("invalid tiny LM config: " + error);
  if (oh.size() != size_t(c.tokens) * c.vocabularySize ||
      target.size() != oh.size())
    throw std::invalid_argument("INVALID_TINY_LM_INPUT_SHAPE");
  requireGeneralParameterShape(c, w);
  MarginForwardResult result;
  result.forward = generalForward(c, oh, w);
  const GF& g = result.forward;
  result.dLogits.resize(g.logits.size());
  std::vector<std::uint32_t> truthRows(c.tokens), bestRows(c.tokens);
  std::vector<float> margins(c.tokens);
  double loss = 0.0;
  std::uint32_t correct = 0;
  for (std::uint32_t row = 0; row < c.tokens; ++row) {
    const std::size_t b = std::size_t(row) * c.vocabularySize;
    float mx = g.logits[b];
    std::uint32_t pred = 0, truth = 0;
    for (std::uint32_t j = 0; j < c.vocabularySize; ++j) {
      if (g.logits[b + j] > mx) { mx = g.logits[b + j]; pred = j; }
      if (target[b + j] > .5f) truth = j;
      result.dLogits[b + j] = (g.prob[b + j] - target[b + j]) /
                              float(c.tokens);
    }
    double s = 0.0;
    for (std::uint32_t j = 0; j < c.vocabularySize; ++j)
      s += std::exp(double(g.logits[b + j] - mx));
    loss += mx + std::log(s) - g.logits[b + truth];
    correct += pred == truth;
    truthRows[row] = truth;
    float best = -std::numeric_limits<float>::infinity();
    std::uint32_t bestIdx = 0;
    for (std::uint32_t j = 0; j < c.vocabularySize; ++j)
      if (j != truth && g.logits[b + j] > best) {
        best = g.logits[b + j];
        bestIdx = j;
      }
    margins[row] = g.logits[b + truth] - best;
    bestRows[row] = bestIdx;
  }
  MarginStepSummary& summary = result.summary;
  const double tokens = double(c.tokens);
  summary.nll = loss / tokens;
  summary.accuracy = float(correct) / float(c.tokens);
  double marginSum = 0.0, critical = 0.0;
  for (std::uint32_t row = 0; row < c.tokens; ++row) {
    marginSum += margins[row];
    if (margins[row] < 0.0f) critical += 1.0;
  }
  summary.meanMargin = marginSum / tokens;
  summary.criticalShare = critical / tokens;
  if (spec.lambda != 0.0f) {
    if (spec.family == MarginLossSpec::Family::PairwiseMarginCe) {
      double term = 0.0;
      for (std::uint32_t row = 0; row < c.tokens; ++row) {
        const double shortfall = double(spec.delta) - double(margins[row]);
        if (shortfall <= 0.0) continue;
        const std::size_t b = std::size_t(row) * c.vocabularySize;
        result.dLogits[b + truthRows[row]] -= spec.lambda / float(c.tokens);
        result.dLogits[b + bestRows[row]] += spec.lambda / float(c.tokens);
        term += shortfall;
      }
      summary.marginTerm = double(spec.lambda) * term / tokens;
    } else {
      if (!(spec.tau > 0.0f))
        throw std::invalid_argument("INVALID_MARGIN_TAU");
      double zmax = -std::numeric_limits<double>::infinity();
      for (std::uint32_t row = 0; row < c.tokens; ++row)
        zmax = std::max(zmax, -double(margins[row]) / double(spec.tau));
      double sum = 0.0;
      std::vector<double> weights(c.tokens);
      for (std::uint32_t row = 0; row < c.tokens; ++row) {
        weights[row] = std::exp(-double(margins[row]) / double(spec.tau) -
                                zmax);
        sum += weights[row];
      }
      for (std::uint32_t row = 0; row < c.tokens; ++row) {
        const double p = weights[row] / sum;
        const std::size_t b = std::size_t(row) * c.vocabularySize;
        result.dLogits[b + truthRows[row]] -=
            spec.lambda * float(p) / float(c.tokens);
        result.dLogits[b + bestRows[row]] +=
            spec.lambda * float(p) / float(c.tokens);
      }
      // Reported term = effective objective (mean per token), so the reported
      // total loss is exactly the minimized function: d/dtheta of
      // lambda*tau*logsumexp/T equals the applied dLogits contributions
      // lambda*p/T (verified by the probe's finite-difference spot check).
      summary.marginTerm = double(spec.lambda) * double(spec.tau) *
                           (zmax + std::log(sum)) / tokens;
    }
  }
  summary.total = summary.nll + summary.marginTerm;
  result.gradients =
      generalBackwardGradients(c, g, oh, w, result.dLogits);
  double gradSq = 0.0;
  for (const auto& e : tiny::parameterRegistry(result.gradients))
    for (const float gv : *e.values) gradSq += double(gv) * double(gv);
  summary.gradientNorm = std::sqrt(gradSq);
  return result;
}

inline MarginTrainingRun runMarginTraining(const tiny::Config& config,
    std::uint32_t seed, const MarginLossSpec& spec, int steps,
    const std::vector<int>& evaluationSteps, int parityCheckEvery,
    MarginCheckpointCallback onCheckpoint, void* context) {
  if (steps < 1) throw std::invalid_argument("INVALID_TRAINING_STEPS");
  P params = tiny::initialParameters(config, seed);
  params = depth_quality::applyInitStability(
      config, std::move(params), depth_quality::StabilityMode::LEGACY);
  P m, v;
  const auto zeroMoments = [](P& target, const P& like) {
    target = like;
    for (const auto& e : tiny::parameterRegistry(target))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
  };
  zeroMoments(m, params);
  zeroMoments(v, params);
  MarginTrainingRun run;
  run.spec = spec;
  run.seed = seed;
  run.steps = steps;
  const int stabilityStep = steps >= 32 ? steps - 16 : 0;
  if (onCheckpoint &&
      std::find(evaluationSteps.begin(), evaluationSteps.end(), 0) !=
          evaluationSteps.end()) {
    // Step-0 checkpoint is the init-stabilized parameters (pre-update).
    onCheckpoint(0, params, context);
  }
  for (int step = 1; step <= steps; ++step) {
    const std::uint32_t pattern = std::uint32_t((step - 1) % 4);
    const auto batch = depth_quality::formalBatch(config, pattern, 0);
    const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
    const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
    const float lr = phonelm::stabilityLearningRate(
        std::uint32_t(depth_quality::StabilityMode::LEGACY), 0.003f,
        std::uint32_t(step), std::uint32_t(steps));
    const auto result =
        marginForwardBackward(config, batch.first, batch.second, params, spec);
    if (parityCheckEvery > 0 && step % parityCheckEvery == 0) {
      // Bitwise check of the copied backward machinery against
      // tiny_lm::forwardBackward using the CE dLogits (the canonical API
      // cannot accept our augmented dLogits, so rebuild them from the same
      // forward pass; the margin term is verified separately in the probe
      // self-test and is linear in dLogits).
      const auto canonical =
          tiny::forwardBackward(config, batch.first, batch.second, params,
                                0.0f);
      std::vector<float> ceDLogits(result.dLogits.size());
      for (std::uint32_t row = 0; row < config.tokens; ++row) {
        const std::size_t b = std::size_t(row) * config.vocabularySize;
        for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
          ceDLogits[b + j] =
              (result.forward.prob[b + j] - batch.second[b + j]) /
              float(config.tokens);
      }
      const auto mineCe = generalBackwardGradients(config, result.forward,
                                                   batch.first, params,
                                                   ceDLogits);
      bool equal = true;
      const auto compare = [&](const std::vector<float>& a,
                               const std::vector<float>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
          if (a[i] != b[i]) return false;
        return true;
      };
      equal = compare(mineCe.tokenEmbedding,
                      canonical.gradients.tokenEmbedding) &&
              compare(mineCe.outputProjection,
                      canonical.gradients.outputProjection) &&
              mineCe.layers.size() == canonical.gradients.layers.size();
      for (std::size_t li = 0; equal && li < mineCe.layers.size(); ++li) {
        const auto& a = mineCe.layers[li];
        const auto& b = canonical.gradients.layers[li];
        equal = compare(a.gamma1, b.gamma1) && compare(a.beta1, b.beta1) &&
                compare(a.gamma2, b.gamma2) && compare(a.beta2, b.beta2) &&
                compare(a.wq, b.wq) && compare(a.wk, b.wk) &&
                compare(a.wv, b.wv) && compare(a.wo, b.wo) &&
                compare(a.w1, b.w1) && compare(a.w2, b.w2);
      }
      if (!equal)
        throw std::runtime_error("MARGIN_BACKWARD_PARITY_MISMATCH at step " +
                                 std::to_string(step));
      run.lastParityStep = step;
    }
    const auto update = tiny::adamUpdate(params, result.gradients, m, v, lr,
                                         .9f, .999f, 1e-8f, c1, c2);
    run.stepSummaries.push_back(result.summary);
    const bool atStability = step == stabilityStep;
    const bool atFinal = step == steps;
    const bool atEvaluation =
        std::find(evaluationSteps.begin(), evaluationSteps.end(), step) !=
        evaluationSteps.end();
    // Canonical checkpoint semantics: checkpoints[step] holds the
    // post-update parameters (runFormalCpu records after the update).
    params = update.next;
    m = update.firstMoment;
    v = update.secondMoment;
    if (atStability) run.stabilityParameters = params;
    if (atFinal) run.finalParameters = params;
    if ((atEvaluation || atStability || atFinal) && onCheckpoint)
      onCheckpoint(step, params, context);
  }
  for (const auto& s : run.stepSummaries)
    if (!std::isfinite(s.total) || !std::isfinite(s.meanMargin) ||
        !std::isfinite(s.gradientNorm)) {
      run.finite = false;
      break;
    }
  return run;
}

}  // namespace phonelm::critical_margin::train

#endif  // CRITICAL_MARGIN_TRAINING_LIB_H
