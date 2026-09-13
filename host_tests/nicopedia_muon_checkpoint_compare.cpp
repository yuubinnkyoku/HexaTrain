// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only NPRTCKPTV4 comparator used by CPU-vs-HVX Muon trajectory and
// resume parity gates. Usage:
//   nicopedia_muon_checkpoint_compare <left.ckpt> <right.ckpt> [label]
// Prints identity fields plus per-tensor-group maxAbs / relativeL2 / cosine /
// finite metrics for parameters, Muon momentum, Aux Adam M, and Aux Adam V.

#include "nicopedia_muon_checkpoint.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace mc = phonelm::nicopedia_muon_checkpoint;

struct Difference {
  double maxAbs = 0.0;
  double relativeL2 = 0.0;
  double cosine = 1.0;
  bool finite = true;
  std::size_t elements = 0;
};

bool readAll(const std::string& path, std::vector<std::uint8_t>* bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  bytes->assign(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
  return !bytes->empty();
}

Difference difference(const std::vector<float>& a, const std::vector<float>& b) {
  Difference d;
  d.elements = a.size();
  if (a.size() != b.size()) {
    d.finite = false;
    d.maxAbs = std::numeric_limits<double>::infinity();
    d.relativeL2 = std::numeric_limits<double>::infinity();
    d.cosine = 0.0;
    return d;
  }
  double sumAbs = 0.0, sumA2 = 0.0, sumB2 = 0.0, sumAB = 0.0, maxAbs = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) d.finite = false;
    const double diff = double(a[i]) - double(b[i]);
    const double absDiff = std::abs(diff);
    if (absDiff > maxAbs) maxAbs = absDiff;
    sumAbs += diff * diff;
    sumA2 += double(a[i]) * double(a[i]);
    sumB2 += double(b[i]) * double(b[i]);
    sumAB += double(a[i]) * double(b[i]);
  }
  d.maxAbs = maxAbs;
  const double denomA = std::sqrt(sumA2);
  const double denomB = std::sqrt(sumB2);
  d.relativeL2 = denomA > 0.0 ? std::sqrt(sumAbs) / denomA
                              : (denomB > 0.0 ? std::sqrt(sumAbs) / denomB
                                              : 0.0);
  const double denom = denomA * denomB;
  d.cosine = denom > 0.0 ? sumAB / denom : 1.0;
  return d;
}

void printDifference(const char* group, const Difference& d) {
  std::cout << group << "_elements=" << d.elements << "\n"
            << group << "_max_abs=" << std::setprecision(10) << d.maxAbs << "\n"
            << group << "_relative_l2=" << d.relativeL2 << "\n"
            << group << "_cosine=" << d.cosine << "\n"
            << group << "_finite=" << (d.finite ? "true" : "false") << "\n";
}

bool sameIdentity(const mc::Checkpoint& left, const mc::Checkpoint& right,
                  std::string* error) {
  if (left.identity.seed != right.identity.seed) {
    *error = "IDENTITY_SEED_MISMATCH";
    return false;
  }
  if (left.identity.globalStep != right.identity.globalStep) {
    *error = "IDENTITY_STEP_MISMATCH";
    return false;
  }
  if (left.identity.tokenizerHash != right.identity.tokenizerHash) {
    *error = "IDENTITY_TOKENIZER_MISMATCH";
    return false;
  }
  if (left.identity.dataCursor.datasetHash !=
      right.identity.dataCursor.datasetHash) {
    *error = "IDENTITY_DATASET_MISMATCH";
    return false;
  }
  if (left.identity.dataCursor.recordIndex !=
      right.identity.dataCursor.recordIndex) {
    *error = "IDENTITY_RECORD_INDEX_MISMATCH";
    return false;
  }
  if (left.identity.dataCursor.orderSeed != right.identity.dataCursor.orderSeed) {
    *error = "IDENTITY_ORDER_SEED_MISMATCH";
    return false;
  }
  if (left.parameters.size() != right.parameters.size()) {
    *error = "REGISTRY_COUNT_MISMATCH";
    return false;
  }
  for (std::size_t i = 0; i < left.parameters.size(); ++i) {
    if (left.parameters[i].name != right.parameters[i].name ||
        left.parameters[i].role != right.parameters[i].role ||
        left.parameters[i].shape.rows != right.parameters[i].shape.rows ||
        left.parameters[i].shape.columns != right.parameters[i].shape.columns) {
      *error = "REGISTRY_ENTRY_MISMATCH:" + left.parameters[i].name;
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <left.ckpt> <right.ckpt> [label]\n";
    return 2;
  }
  const std::string leftPath = argv[1];
  const std::string rightPath = argv[2];
  const std::string label = argc >= 4 ? argv[3] : "compare";

  std::vector<std::uint8_t> leftBytes, rightBytes;
  if (!readAll(leftPath, &leftBytes) || !readAll(rightPath, &rightBytes)) {
    std::cout << "status=FAILED\nerror=CHECKPOINT_READ\n";
    return 1;
  }
  mc::Checkpoint left, right;
  std::string error;
  if (!mc::decodeCheckpoint(leftBytes, &left, &error)) {
    std::cout << "status=FAILED\nerror=LEFT_DECODE:" << error << "\n";
    return 1;
  }
  if (!mc::decodeCheckpoint(rightBytes, &right, &error)) {
    std::cout << "status=FAILED\nerror=RIGHT_DECODE:" << error << "\n";
    return 1;
  }
  if (!sameIdentity(left, right, &error)) {
    std::cout << "status=FAILED\nerror=" << error << "\n";
    return 1;
  }

  Difference params, momentum, adamM, adamV;
  bool finite = true;
  for (std::size_t i = 0; i < left.parameters.size(); ++i) {
    const auto& a = left.parameters[i];
    const auto& b = right.parameters[i];
    const Difference dp = difference(a.values, b.values);
    params.maxAbs = std::max(params.maxAbs, dp.maxAbs);
    params.relativeL2 = std::max(params.relativeL2, dp.relativeL2);
    params.cosine = std::min(params.cosine, dp.cosine);
    params.finite = params.finite && dp.finite;
    params.elements += dp.elements;
    if (a.role == mc::ParameterRole::MUON) {
      const Difference dm = difference(a.momentum, b.momentum);
      momentum.maxAbs = std::max(momentum.maxAbs, dm.maxAbs);
      momentum.relativeL2 = std::max(momentum.relativeL2, dm.relativeL2);
      momentum.cosine = std::min(momentum.cosine, dm.cosine);
      momentum.finite = momentum.finite && dm.finite;
      momentum.elements += dm.elements;
    } else {
      const Difference dM = difference(a.adamM, b.adamM);
      const Difference dV = difference(a.adamV, b.adamV);
      adamM.maxAbs = std::max(adamM.maxAbs, dM.maxAbs);
      adamM.relativeL2 = std::max(adamM.relativeL2, dM.relativeL2);
      adamM.cosine = std::min(adamM.cosine, dM.cosine);
      adamM.finite = adamM.finite && dM.finite;
      adamM.elements += dM.elements;
      adamV.maxAbs = std::max(adamV.maxAbs, dV.maxAbs);
      adamV.relativeL2 = std::max(adamV.relativeL2, dV.relativeL2);
      adamV.cosine = std::min(adamV.cosine, dV.cosine);
      adamV.finite = adamV.finite && dV.finite;
      adamV.elements += dV.elements;
    }
  }
  finite = params.finite && momentum.finite && adamM.finite && adamV.finite;

  std::cout << "NPRTCKPTV4_COMPARE\nstatus=" << (finite ? "SUCCESS" : "FAILED")
            << "\nlabel=" << label << "\nleft_path=" << leftPath
            << "\nright_path=" << rightPath << "\nseed=" << left.identity.seed
            << "\nstep=" << left.identity.globalStep
            << "\noptimizer=" << left.optimizerIdentity
            << "\nmuon_lr=" << left.hyperparameters.muonLearningRate
            << "\naux_adam_lr=" << left.hyperparameters.auxAdamLearningRate
            << "\n";
  printDifference("parameters", params);
  printDifference("muon_momentum", momentum);
  printDifference("aux_adam_m", adamM);
  printDifference("aux_adam_v", adamV);
  return finite ? 0 : 1;
}
