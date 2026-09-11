// SPDX-License-Identifier: Apache-2.0
#include "nicopedia_muon_optimizer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace phonelm;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
bool close(const std::vector<float>& a, const std::vector<float>& b,
           double tolerance = 2e-5) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(double(a[i]) - b[i]) > tolerance) return false;
  return true;
}
bool contains(const std::string& value, const std::string& needle) {
  return value.find(needle) != std::string::npos;
}
std::vector<float> transpose(const std::vector<float>& x, unsigned r, unsigned c) {
  std::vector<float> y(x.size());
  for (unsigned i=0;i<r;++i) for(unsigned j=0;j<c;++j) y[std::size_t(j)*r+i]=x[std::size_t(i)*c+j];
  return y;
}
void matrixCases() {
  for (const auto shape : {std::pair<unsigned,unsigned>{2,2},{2,3},{3,2},{64,64},{128,64},{64,128}}) {
    std::vector<float> input(std::size_t(shape.first)*shape.second);
    for (std::size_t i=0;i<input.size();++i) input[i]=float((int(i%17)-8)*0.013);
    std::vector<float> a,b;
    std::string error;
    require(nicopedia_muon::zeropowerNewtonSchulzFp32(input,shape.first,shape.second,5,&a,nullptr,&error), error.c_str());
    require(nicopedia_muon::zeropowerNewtonSchulzFp32(input,shape.first,shape.second,5,&b,nullptr,&error), error.c_str());
    require(a==b,"NS repeated result not deterministic");
    require(std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}),"NS nonfinite");
    const auto inputT=transpose(input,shape.first,shape.second);
    std::vector<float> outputT;
    require(nicopedia_muon::zeropowerNewtonSchulzFp32(inputT,shape.second,shape.first,5,&outputT,nullptr,&error),error.c_str());
    require(close(a,transpose(outputT,shape.second,shape.first)),"NS transpose mismatch");
  }
}
void zeroAndTiny() {
  for(float value : {0.0f,1e-30f}) {
    std::vector<float> input(6,value), output;
    std::string error;
    require(nicopedia_muon::zeropowerNewtonSchulzFp32(input,2,3,5,&output,nullptr,&error),error.c_str());
    require(std::all_of(output.begin(),output.end(),[](float v){return std::isfinite(v);}),"zero/tiny nonfinite");
    if(value==0) require(std::all_of(output.begin(),output.end(),[](float v){return v==0;}),"zero not exact");
  }
}
void classificationAndStep() {
  tiny_lm::Config config{1024,32,64,128,1e-5f,19,2};
  auto parameters=tiny_lm::initialParameters(config,1);
  auto gradients=parameters;
  auto zero=parameters;
  for(const auto& entry:tiny_lm::parameterRegistry(gradients))
    std::fill(const_cast<std::vector<float>*>(entry.values)->begin(),const_cast<std::vector<float>*>(entry.values)->end(),1e-3f);
  for(const auto& entry:tiny_lm::parameterRegistry(zero))
    std::fill(const_cast<std::vector<float>*>(entry.values)->begin(),const_cast<std::vector<float>*>(entry.values)->end(),0.0f);
  std::string error;
  require(tiny_lm::validateParameterRegistry(parameters,&error),error.c_str());
  tiny_lm::ParameterPartition split;
  require(tiny_lm::splitParameterRegistry(parameters,&split,&error),error.c_str());
  require(split.muon.size()==114,"Muon matrix count");
  require(split.auxiliaryAdam.size()==78,"aux registry count");
  std::uint64_t muonElements=0,auxElements=0;
  for(const auto& e:split.muon)muonElements+=e.values->size();
  for(const auto& e:split.auxiliaryAdam)auxElements+=e.values->size();
  require(muonElements==622592,"Muon element count");
  require(auxElements==135936,"aux element count");
  nicopedia_muon::Config updateConfig;
  auto first=nicopedia_muon::update(parameters,gradients,zero,zero,zero,updateConfig);
  require(first.error.empty(),first.error.c_str());
  require(first.muonMatrixCount==114 && first.muonParameterCount==622592 && first.auxiliaryAdamParameterCount==135936,"update counts");
  // Fixed step-1 Adam values for the first norm gain.  This anchors the
  // mixed update independently of the loss trajectory: m=.1*g, v=.001*g^2,
  // followed by the beta corrections at optimizer step one.
  const double expectedM = 0.1 * 1.0e-3;
  const double expectedV = 0.001 * 1.0e-6;
  const double expectedAdamStep = (expectedM * 10.0) /
      (std::sqrt(expectedV * 1000.0) + 1.0e-8);
  require(std::abs(double(first.auxiliaryAdamM.gamma1.front()) - expectedM) <
              1.0e-9,
          "fixed Aux Adam m mismatch");
  require(std::abs(double(first.auxiliaryAdamV.gamma1.front()) - expectedV) <
              1.0e-12,
          "fixed Aux Adam v mismatch");
  require(std::abs(double(first.parameters.gamma1.front()) -
                   (1.0 - double(updateConfig.auxiliaryAdamLearningRate) *
                                expectedAdamStep)) < 1.0e-7,
          "fixed Aux Adam parameter mismatch");
  auto duplicate=nicopedia_muon::update(parameters,gradients,zero,zero,zero,updateConfig);
  require(duplicate.error.empty() && tiny_lm::parameterRegistry(first.parameters).size()==tiny_lm::parameterRegistry(duplicate.parameters).size(),"duplicate update failed");
  for(std::size_t i=0;i<tiny_lm::parameterRegistry(first.parameters).size();++i)
    require(*tiny_lm::parameterRegistry(first.parameters)[i].values==*tiny_lm::parameterRegistry(duplicate.parameters)[i].values,"mixed update nondeterministic");
  auto auxiliaryOnly = nicopedia_muon::updateAuxiliaryAdamOnly(
      parameters, gradients, first.muonMomentum, zero, zero, updateConfig);
  require(auxiliaryOnly.error.empty(), auxiliaryOnly.error.c_str());
  const auto mixedRegistry = tiny_lm::parameterRegistry(first.parameters);
  const auto auxiliaryRegistry = tiny_lm::parameterRegistry(auxiliaryOnly.parameters);
  const auto parameterRegistry = tiny_lm::parameterRegistry(parameters);
  for (std::size_t i = 0; i < mixedRegistry.size(); ++i) {
    if (mixedRegistry[i].role == tiny_lm::ParameterRole::AUX_ADAM)
      require(*mixedRegistry[i].values == *auxiliaryRegistry[i].values,
              "Aux Adam-only result changed");
    else
      require(*parameterRegistry[i].values == *auxiliaryRegistry[i].values,
              "Aux Adam-only changed Muon parameter");
  }
  auto zeroStep=nicopedia_muon::update(parameters,zero,zero,zero,zero,updateConfig);
  require(zeroStep.error.empty(),zeroStep.error.c_str());
  for(std::size_t i=0;i<tiny_lm::parameterRegistry(parameters).size();++i)
    require(*tiny_lm::parameterRegistry(parameters)[i].values==*tiny_lm::parameterRegistry(zeroStep.parameters)[i].values,"fresh zero gradient changed parameter");
}

void rejectsCorruptOptimizerState() {
  tiny_lm::Config config{1024,32,64,128,1e-5f,19,2};
  const auto parameters = tiny_lm::initialParameters(config, 1);
  const auto gradients = parameters;
  auto zero = parameters;
  for (const auto& entry : tiny_lm::parameterRegistry(zero))
    std::fill(const_cast<std::vector<float>*>(entry.values)->begin(),
              const_cast<std::vector<float>*>(entry.values)->end(), 0.0f);
  nicopedia_muon::Config updateConfig;
  std::string error;

  // The public validator must reject a name mutation even when the backing
  // storage and shape remain valid.  Production update() uses this same
  // exact-identity check before indexing any state registry.
  auto expected = tiny_lm::parameterRegistry(parameters);
  auto corruptName = tiny_lm::parameterRegistry(zero);
  corruptName.front().name = "corrupt_token_embedding";
  require(!nicopedia_muon::validateOptimizerStateRegistry(
              expected, corruptName, "MOMENTUM", &error) &&
              contains(error, "MUON_MOMENTUM_NAME_MISMATCH"),
          "corrupt optimizer-state name accepted");

  // A shortened matrix must be rejected before the update loop can index it.
  auto badMomentum = zero;
  badMomentum.wq.pop_back();
  auto rejected = nicopedia_muon::update(
      parameters, gradients, badMomentum, zero, zero, updateConfig);
  require(contains(rejected.error, "MUON_MOMENTUM_REGISTRY_INVALID") &&
              rejected.health.momentumFinite,
          "corrupt momentum shape accepted or health changed");

  // Auxiliary m/v shape and finite checks are independent of the Muon
  // momentum registry and must fail closed as well.
  auto badMShape = zero;
  badMShape.tokenEmbedding.pop_back();
  rejected = nicopedia_muon::update(parameters, gradients, zero, badMShape,
                                    zero, updateConfig);
  require(contains(rejected.error, "MUON_AUX_ADAM_M_REGISTRY_INVALID"),
          "corrupt auxiliary m shape accepted");

  auto badMomentumFinite = zero;
  badMomentumFinite.wq.front() = std::numeric_limits<float>::quiet_NaN();
  rejected = nicopedia_muon::update(parameters, gradients,
                                    badMomentumFinite, zero, zero,
                                    updateConfig);
  require(contains(rejected.error, "MUON_MOMENTUM_NONFINITE") &&
              !rejected.health.momentumFinite,
          "nonfinite momentum accepted");

  auto badMFinite = zero;
  badMFinite.tokenEmbedding.front() = std::numeric_limits<float>::infinity();
  rejected = nicopedia_muon::update(parameters, gradients, zero, badMFinite,
                                    zero, updateConfig);
  require(contains(rejected.error, "MUON_AUX_ADAM_M_NONFINITE") &&
              !rejected.health.updateFinite,
          "nonfinite auxiliary m accepted");

  auto badVFinite = zero;
  badVFinite.outputProjection.front() =
      std::numeric_limits<float>::quiet_NaN();
  rejected = nicopedia_muon::update(parameters, gradients, zero, zero,
                                    badVFinite, updateConfig);
  require(contains(rejected.error, "MUON_AUX_ADAM_V_NONFINITE") &&
              !rejected.health.updateFinite,
          "nonfinite auxiliary v accepted");
}
}
int main(){try{matrixCases();zeroAndTiny();classificationAndStep();rejectsCorruptOptimizerState();std::cout<<"nicopedia_muon_optimizer_test=PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
