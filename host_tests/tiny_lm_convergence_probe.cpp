// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "tiny_language_model_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using Params = phonelm::qnn::TinyTransformerParameters;
using Step = phonelm::tiny_lm::StepResult;
using Member = std::vector<float> Params::*;
struct Field { const char* name; Member member; };
const std::vector<Field>& fields() {
  static const std::vector<Field> value{
      {"token_embedding", &Params::tokenEmbedding},
      {"wq", &Params::wq}, {"wk", &Params::wk}, {"wv", &Params::wv},
      {"wo", &Params::wo}, {"norm1_gamma", &Params::gamma1},
      {"norm1_beta", &Params::beta1}, {"norm2_gamma", &Params::gamma2},
      {"norm2_beta", &Params::beta2}, {"ffn_w1", &Params::w1},
      {"ffn_w2", &Params::w2},
      {"output_projection", &Params::outputProjection}};
  return value;
}
const std::vector<std::vector<uint32_t>>& patterns() {
  static const std::vector<std::vector<uint32_t>> value{
      {0,1,2,3}, {4,5,6,7}, {8,9}, {10,11,12}};
  return value;
}
struct Batch { std::vector<float> input, target; };
Batch batch(const phonelm::tiny_lm::Config& c, uint32_t pattern, uint32_t phase) {
  const auto& p = patterns().at(pattern);
  std::vector<uint32_t> input(c.tokens), target(c.tokens);
  for (uint32_t i = 0; i < c.tokens; ++i) {
    input[i] = p[(i + phase) % p.size()];
    target[i] = p[(i + phase + 1) % p.size()];
  }
  return {phonelm::tiny_lm::oneHot(input, c.vocabularySize),
          phonelm::tiny_lm::oneHot(target, c.vocabularySize)};
}
void scaleInitialization(Params& p, double scale) {
  for (Member member : {&Params::tokenEmbedding, &Params::wq, &Params::wk,
                        &Params::wv, &Params::wo, &Params::w1, &Params::w2,
                        &Params::outputProjection})
    for (float& value : p.*member) value = float(value * scale);
}
double l2(const std::vector<float>& values) {
  double sum = 0; for (float v : values) sum += double(v) * v; return std::sqrt(sum);
}
double maxAbs(const std::vector<float>& values) {
  double result = 0; for (float v : values) result = std::max(result, std::abs(double(v))); return result;
}
struct Quality {
  double loss=0, accuracy=0, meanCorrectProbability=0, medianCorrectProbability=0;
  double entropy=0, meanMargin=0, minimumMargin=0;
};
Quality quality(const phonelm::tiny_lm::Config& c, const Params& p, uint32_t phase) {
  Quality q; std::vector<double> correct; bool first = true; uint32_t correctCount = 0;
  for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern) {
    const auto b = batch(c, pattern, phase);
    const auto step = phonelm::tiny_lm::forwardBackward(c, b.input, b.target, p, 0);
    q.loss += step.loss;
    for (uint32_t row = 0; row < c.tokens; ++row) {
      const size_t base = size_t(row) * c.vocabularySize;
      uint32_t truth = 0, prediction = 0; float bestOther = -INFINITY;
      for (uint32_t column = 0; column < c.vocabularySize; ++column) {
        if (b.target[base + column] > .5f) truth = column;
        if (step.logits[base + column] > step.logits[base + prediction]) prediction = column;
      }
      for (uint32_t column = 0; column < c.vocabularySize; ++column) {
        const double probability = std::max(double(step.probabilities[base + column]), 1e-30);
        q.entropy -= probability * std::log(probability);
        if (column != truth) bestOther = std::max(bestOther, step.logits[base + column]);
      }
      const double probability = step.probabilities[base + truth];
      const double margin = step.logits[base + truth] - bestOther;
      correct.push_back(probability); q.meanCorrectProbability += probability;
      q.meanMargin += margin; if (first || margin < q.minimumMargin) q.minimumMargin = margin;
      first = false; correctCount += prediction == truth;
    }
  }
  const double rows = double(patterns().size() * c.tokens);
  q.loss /= patterns().size(); q.accuracy = correctCount / rows;
  q.meanCorrectProbability /= rows; q.entropy /= rows; q.meanMargin /= rows;
  std::sort(correct.begin(), correct.end());
  q.medianCorrectProbability = .5 * (correct[(correct.size()-1)/2] + correct[correct.size()/2]);
  return q;
}
std::vector<uint32_t> orderFor(uint32_t seed, uint32_t epoch, bool shuffle) {
  std::vector<uint32_t> order{0,1,2,3}; if (!shuffle) return order;
  uint32_t state = seed * 747796405u + epoch * 2891336453u + 0x9e3779b9u;
  for (size_t i = order.size() - 1; i > 0; --i) {
    state = state * 1664525u + 1013904223u;
    std::swap(order[i], order[state % (i + 1)]);
  }
  return order;
}
struct Norms { double gradient=0, update=0, parameter=0, ratio=0; };
Norms norms(const Params& current, const Step& step) {
  double g2=0,u2=0,p2=0;
  for (const auto& field : fields()) {
    const auto& p=current.*field.member; const auto& g=step.gradients.*field.member;
    const auto& n=step.next.*field.member;
    for(size_t i=0;i<p.size();++i){g2+=double(g[i])*g[i];double u=double(n[i])-p[i];u2+=u*u;p2+=double(p[i])*p[i];}
  }
  Norms result{std::sqrt(g2),std::sqrt(u2),std::sqrt(p2),0};
  result.ratio=result.parameter?result.update/result.parameter:0; return result;
}
void printDiagnostics(const std::string& id,uint32_t seed,int stepIndex,const std::string& split,
                      const Quality& q,const Params& current,const Step* step) {
  std::cout<<"diagnostic="<<id<<","<<seed<<","<<stepIndex<<","<<split
           <<","<<q.loss<<","<<q.accuracy<<","<<q.meanCorrectProbability
           <<","<<q.medianCorrectProbability<<","<<q.entropy<<","<<q.meanMargin
           <<","<<q.minimumMargin;
  if(step){const auto n=norms(current,*step);std::cout<<","<<n.gradient<<","<<n.update<<","<<n.parameter<<","<<n.ratio;}
  else std::cout<<",0,0,0,0";
  std::cout<<'\n';
  if(!step)return;
  for(const auto& field:fields()){
    const auto& p=current.*field.member;const auto& g=step->gradients.*field.member;const auto& next=step->next.*field.member;
    std::vector<float> update(p.size());for(size_t i=0;i<p.size();++i)update[i]=next[i]-p[i];
    std::cout<<"parameter_diagnostic="<<id<<","<<seed<<","<<stepIndex<<","<<field.name
             <<","<<l2(g)<<","<<l2(update)<<","<<l2(p)<<","<<maxAbs(g)<<","<<maxAbs(update)<<'\n';
  }
}
struct SeedResult { Quality initialTrain,initialEval,finalTrain,finalEval; Params parameters; bool finite=true; };
SeedResult train(const std::string& id,uint32_t seed,double lr,int steps,double initScale,
                 bool shuffle,bool diagnostics,double momentum,bool adam) {
  phonelm::tiny_lm::Config c; auto p=phonelm::tiny_lm::initialParameters(c,seed);scaleInitialization(p,initScale);
  Params velocity=p;for(const auto&field:fields())std::fill((velocity.*field.member).begin(),(velocity.*field.member).end(),0.0f);
  Params secondMoment=velocity;
  SeedResult result;result.initialTrain=quality(c,p,0);result.initialEval=quality(c,p,1);
  const std::vector<int> checkpoints{0,1,2,5,10,20,50,100,200,320,640,1000};
  if(diagnostics)printDiagnostics(id,seed,0,"train",result.initialTrain,p,nullptr);
  if(diagnostics)printDiagnostics(id,seed,0,"evaluation",result.initialEval,p,nullptr);
  for(int index=1;index<=steps;++index){
    const uint32_t epoch=uint32_t((index-1)/4);const auto order=orderFor(seed,epoch,shuffle);
    const auto b=batch(c,order[size_t(index-1)%4],0);auto step=phonelm::tiny_lm::forwardBackward(c,b.input,b.target,p,(momentum>0||adam)?0.0f:float(lr));
    if(momentum>0){auto update=phonelm::tiny_lm::momentumUpdate(p,step.gradients,velocity,float(lr),float(momentum));velocity=std::move(update.velocity);step.next=std::move(update.next);}
    if(adam){
      const float correction1=float(1.0/(1.0-std::pow(0.9,index)));
      const float correction2=float(1.0/(1.0-std::pow(0.999,index)));
      auto update=phonelm::tiny_lm::adamUpdate(p,step.gradients,velocity,secondMoment,
          float(lr),.9f,.999f,1e-8f,correction1,correction2);
      velocity=std::move(update.firstMoment);secondMoment=std::move(update.secondMoment);
      step.next=std::move(update.next);
    }
    bool checkpoint=std::find(checkpoints.begin(),checkpoints.end(),index)!=checkpoints.end();
    if(diagnostics&&checkpoint){printDiagnostics(id,seed,index,"train",quality(c,step.next,0),p,&step);printDiagnostics(id,seed,index,"evaluation",quality(c,step.next,1),p,&step);}
    p=step.next;
    for(const auto& field:fields())for(float value:p.*field.member)result.finite=result.finite&&std::isfinite(value);
    if(!result.finite)break;
  }
  result.finalTrain=quality(c,p,0);result.finalEval=quality(c,p,1);result.parameters=std::move(p);return result;
}
bool sameParameters(const Params&a,const Params&b){for(const auto&f:fields())if(a.*f.member!=b.*f.member)return false;return true;}
}
int main(int argc,char**argv){
  if(argc<7||argc>9){std::cerr<<"usage: probe id lr steps init_scale fixed|shuffle diagnostics0|1 [momentum] [adam0|1]\n";return 2;}
  const std::string id=argv[1];const double lr=std::stod(argv[2]);const int steps=std::stoi(argv[3]);const double scale=std::stod(argv[4]);const bool shuffle=std::string(argv[5])=="shuffle";const bool diagnostics=std::stoi(argv[6])!=0;const double momentum=argc>=8?std::stod(argv[7]):0;const bool adam=argc==9&&std::stoi(argv[8])!=0;
  std::cout<<std::setprecision(10)<<"configuration_id="<<id<<"\nlearning_rate="<<lr<<"\nsteps="<<steps<<"\ninitialization_scale="<<scale<<"\nsampling="<<(shuffle?"shuffle":"fixed")<<"\noptimizer="<<(adam?"ADAM":(momentum>0?"MOMENTUM_SGD":"SGD"))<<"\nmomentum="<<momentum<<"\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8\ngradient_clipping=disabled\n";
  std::vector<double> reductions;int accuracy75=0;bool allLoss=true,allAccuracy=true,finite=true;
  for(uint32_t seed=1;seed<=5;++seed){const auto r=train(id,seed,lr,steps,scale,shuffle,diagnostics,momentum,adam);const double reduction=100*(r.initialEval.loss-r.finalEval.loss)/r.initialEval.loss;reductions.push_back(reduction);accuracy75+=r.finalEval.accuracy>=.75;allLoss=allLoss&&r.finalEval.loss<r.initialEval.loss;allAccuracy=allAccuracy&&r.finalEval.accuracy>r.initialEval.accuracy;finite=finite&&r.finite;
    std::cout<<"seed_"<<seed<<"_initial_train_loss="<<r.initialTrain.loss<<"\nseed_"<<seed<<"_final_train_loss="<<r.finalTrain.loss<<"\nseed_"<<seed<<"_initial_eval_loss="<<r.initialEval.loss<<"\nseed_"<<seed<<"_final_eval_loss="<<r.finalEval.loss<<"\nseed_"<<seed<<"_initial_train_accuracy="<<r.initialTrain.accuracy<<"\nseed_"<<seed<<"_final_train_accuracy="<<r.finalTrain.accuracy<<"\nseed_"<<seed<<"_initial_eval_accuracy="<<r.initialEval.accuracy<<"\nseed_"<<seed<<"_final_eval_accuracy="<<r.finalEval.accuracy<<"\nseed_"<<seed<<"_loss_reduction="<<reduction<<"\nseed_"<<seed<<"_correct_probability="<<r.finalEval.meanCorrectProbability<<"\nseed_"<<seed<<"_entropy="<<r.finalEval.entropy<<"\nseed_"<<seed<<"_mean_margin="<<r.finalEval.meanMargin<<"\nseed_"<<seed<<"_minimum_margin="<<r.finalEval.minimumMargin<<"\n";
  }
  std::sort(reductions.begin(),reductions.end());const auto replayA=train(id,1,lr,steps,scale,shuffle,false,momentum,adam),replayB=train(id,1,lr,steps,scale,shuffle,false,momentum,adam);
  std::cout<<"median_loss_reduction="<<reductions[2]<<"\nminimum_loss_reduction="<<reductions.front()<<"\naccuracy_75_seed_count="<<accuracy75<<"\nall_seeds_loss_decreased="<<(allLoss?"true":"false")<<"\nall_seeds_accuracy_increased="<<(allAccuracy?"true":"false")<<"\nnan_inf_count="<<(finite?0:1)<<"\ndeterministic_replay="<<(sameParameters(replayA.parameters,replayB.parameters)?"true":"false")<<"\n";
  return finite?0:1;
}
