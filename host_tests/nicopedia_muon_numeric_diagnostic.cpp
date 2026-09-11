// SPDX-License-Identifier: Apache-2.0
// Diagnostic-only Newton-Schulz parity root-cause probe (host side).
// The CPU oracle (nicopedia_muon::zeropowerNewtonSchulzFp32) is used as-is;
// this tool only adds diagnostic float-accumulation / half-input / chunked
// references and compares them against the frozen single-matrix input.
#include "nicopedia_muon_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kGateMaxAbs = 2.0e-3;
constexpr double kGateRelL2 = 1.0e-3;
constexpr double kGateCosine = 0.99999;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t fnv1a64(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string fnvHex(std::uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return std::string(buffer);
}

std::uint16_t f32ToF16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exp = int32_t((bits >> 23) & 0xffu) - 112;
  const std::uint32_t mant = bits & 0x7fffffu;
  if (exp >= 31) {
    if (mant == 0) return std::uint16_t(sign | 0x7c00u);
    return std::uint16_t(sign | 0x7c00u | (mant >> 13) | (mant ? 1u : 0u));
  }
  if (exp <= 0) {
    if (exp < -10) return std::uint16_t(sign);
    const std::uint32_t shifted = (mant | 0x800000u) >> (1 - exp);
    const std::uint32_t rounded = (shifted + 0xfffu + ((shifted >> 13) & 1u)) >> 13;
    return std::uint16_t(sign | rounded);
  }
  const std::uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
  if (rounded & 0x800000u) return std::uint16_t(sign | ((exp + 1) << 10));
  return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (rounded >> 13));
}

float f16BitsToF32(std::uint16_t value) {
  std::uint32_t sign = (std::uint32_t(value & 0x8000u)) << 16;
  std::uint32_t exp = (value >> 10) & 0x1fu;
  std::uint32_t mant = value & 0x3ffu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ffu;
      bits = sign | ((std::uint32_t(exp + 112)) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float quantizeHalf(float value) { return f16BitsToF32(f32ToF16Bits(value)); }

std::int64_t ulpDistance(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b)) return INT64_MAX;
  std::uint32_t ua = 0, ub = 0;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  const std::int32_t oa =
      std::int32_t(ua ^ ((ua >> 31) ? 0xffffffffu : 0x80000000u));
  const std::int32_t ob =
      std::int32_t(ub ^ ((ub >> 31) ? 0xffffffffu : 0x80000000u));
  return std::llabs(std::int64_t(oa) - std::int64_t(ob));
}

struct Metrics {
  double maxAbs = 0.0, meanAbs = 0.0, rmsErr = 0.0, relL2 = 0.0;
  double cosine = 1.0, meanSigned = 0.0, maxUlp = 0.0;
  std::size_t total = 0, finiteCount = 0;
  std::vector<long long> histogram;
};

Metrics compare(const std::vector<float>& expected,
                const std::vector<float>& actual, bool ulp = true) {
  Metrics m;
  static const double kEdges[] = {1e-7, 1e-6, 1e-5, 1e-4,
                                  5e-4, 1e-3, 1e-2, 1e-1};
  m.histogram.assign(sizeof(kEdges) / sizeof(kEdges[0]) + 1, 0);
  if (expected.size() != actual.size() || expected.empty()) {
    m.maxAbs = m.relL2 = std::numeric_limits<double>::infinity();
    m.cosine = -1.0;
    return m;
  }
  m.total = expected.size();
  double absolute = 0.0, squared = 0.0, signedSum = 0.0;
  double expectedSquared = 0.0, actualSquared = 0.0, dot = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double e = expected[i], a = actual[i], d = a - e;
    if (std::isfinite(expected[i]) && std::isfinite(actual[i]))
      ++m.finiteCount;
    m.maxAbs = std::max(m.maxAbs, std::abs(d));
    absolute += std::abs(d);
    signedSum += d;
    squared += d * d;
    expectedSquared += e * e;
    actualSquared += a * a;
    dot += e * a;
    if (ulp) m.maxUlp = std::max(m.maxUlp, double(ulpDistance(expected[i], actual[i])));
    const double ad = std::abs(d);
    std::size_t bin = 0;
    while (bin < m.histogram.size() - 1 && ad >= kEdges[bin]) ++bin;
    m.histogram[bin]++;
  }
  m.meanAbs = absolute / m.total;
  m.rmsErr = std::sqrt(squared / m.total);
  m.meanSigned = signedSum / m.total;
  m.relL2 = std::sqrt(squared) / std::max(std::sqrt(expectedSquared), 1.0e-30);
  const double normProduct = std::sqrt(expectedSquared * actualSquared);
  m.cosine = normProduct > 0.0 ? dot / normProduct : (squared == 0.0 ? 1.0 : 0.0);
  return m;
}

struct Stats {
  double min = 0.0, max = 0.0, rms = 0.0, frob = 0.0;
};

Stats stats(const std::vector<float>& v) {
  Stats s;
  if (v.empty()) return s;
  s.min = v[0];
  s.max = v[0];
  double sum = 0.0;
  for (float x : v) {
    s.min = std::min(s.min, double(x));
    s.max = std::max(s.max, double(x));
    sum += double(x) * x;
  }
  s.frob = std::sqrt(sum);
  s.rms = std::sqrt(sum / v.size());
  return s;
}

double orthDeviation(const std::vector<float>& x, std::uint32_t rows,
                     std::uint32_t cols) {
  double sum = 0.0;
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double dot = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k)
        dot += double(x[std::size_t(i) * cols + k]) *
               x[std::size_t(j) * cols + k];
      const double target = (i == j) ? 1.0 : 0.0;
      sum += (dot - target) * (dot - target);
    }
  return std::sqrt(sum);
}

std::vector<float> makeValues(std::size_t count, float scale, int phase) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = scale *
        (std::sin(float(index + 1) * (0.013f + phase * 0.001f)) +
         float(int(index % 17) - 8) * 0.03125f);
  return result;
}

std::vector<float> buildNesterov(const std::vector<float>& gradient,
                                 const std::vector<float>& prior) {
  std::vector<float> nesterov(gradient.size());
  for (std::size_t i = 0; i < nesterov.size(); ++i) {
    const float nextMomentum = 0.95f * prior[i] + 0.05f * gradient[i];
    nesterov[i] = 0.05f * gradient[i] + 0.95f * nextMomentum;
  }
  return nesterov;
}

std::vector<float> normalizeDirect(const std::vector<float>& nesterov) {
  double normSquared = 0.0;
  for (float v : nesterov) normSquared += double(v) * v;
  const double denominator = std::sqrt(normSquared) + 1.0e-7;
  std::vector<float> x(nesterov.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = float(double(nesterov[i]) / denominator);
  return x;
}

struct Stages {
  std::vector<float> a, a2, b, bx, x1;
};

void matXXTDouble(const std::vector<float>& x, std::uint32_t rows,
                  std::uint32_t cols, std::vector<float>* a) {
  a->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double sum = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k)
        sum += double(x[std::size_t(i) * cols + k]) *
               x[std::size_t(j) * cols + k];
      (*a)[std::size_t(i) * rows + j] = float(sum);
    }
}

void matSqDouble(const std::vector<float>& a, std::uint32_t rows,
                 std::vector<float>* a2) {
  a2->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double sum = 0.0;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += double(a[std::size_t(i) * rows + k]) *
               a[std::size_t(k) * rows + j];
      (*a2)[std::size_t(i) * rows + j] = float(sum);
    }
}

void matXXTFloat(const std::vector<float>& x, std::uint32_t rows,
                 std::uint32_t cols, std::vector<float>* a) {
  a->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < cols; ++k)
        sum += x[std::size_t(i) * cols + k] * x[std::size_t(j) * cols + k];
      (*a)[std::size_t(i) * rows + j] = sum;
    }
}

void matSqFloat(const std::vector<float>& a, std::uint32_t rows,
                std::vector<float>* a2) {
  a2->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += a[std::size_t(i) * rows + k] * a[std::size_t(k) * rows + j];
      (*a2)[std::size_t(i) * rows + j] = sum;
    }
}

void matXXTChunked(const std::vector<float>& x, std::uint32_t rows,
                   std::uint32_t cols, std::uint32_t chunk,
                   bool finalDouble, std::vector<float>* a) {
  a->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double totalD = 0.0;
      float totalF = 0.0f;
      for (std::uint32_t base = 0; base < cols; base += chunk) {
        const std::uint32_t width = std::min(chunk, cols - base);
        float part = 0.0f;
        for (std::uint32_t t = 0; t < width; ++t) {
          const std::uint32_t k = base + t;
          part += x[std::size_t(i) * cols + k] * x[std::size_t(j) * cols + k];
        }
        if (finalDouble)
          totalD += double(part);
        else
          totalF += part;
      }
      (*a)[std::size_t(i) * rows + j] =
          finalDouble ? float(totalD) : totalF;
    }
}

void matSqChunked(const std::vector<float>& a, std::uint32_t rows,
                  std::uint32_t chunk, bool finalDouble,
                  std::vector<float>* a2) {
  a2->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double totalD = 0.0;
      float totalF = 0.0f;
      for (std::uint32_t base = 0; base < rows; base += chunk) {
        const std::uint32_t width = std::min(chunk, rows - base);
        float part = 0.0f;
        for (std::uint32_t t = 0; t < width; ++t) {
          const std::uint32_t k = base + t;
          part += a[std::size_t(i) * rows + k] * a[std::size_t(k) * rows + j];
        }
        if (finalDouble)
          totalD += double(part);
        else
          totalF += part;
      }
      (*a2)[std::size_t(i) * rows + j] =
          finalDouble ? float(totalD) : totalF;
    }
}

void matBXChunked(const std::vector<float>& b, const std::vector<float>& x,
                  std::uint32_t rows, std::uint32_t cols, std::uint32_t chunk,
                  bool finalDouble, std::vector<float>* bx) {
  bx->assign(std::size_t(rows) * cols, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < cols; ++j) {
      double totalD = 0.0;
      float totalF = 0.0f;
      for (std::uint32_t base = 0; base < rows; base += chunk) {
        const std::uint32_t width = std::min(chunk, rows - base);
        float part = 0.0f;
        for (std::uint32_t t = 0; t < width; ++t) {
          const std::uint32_t k = base + t;
          part += b[std::size_t(i) * rows + k] * x[std::size_t(k) * cols + j];
        }
        if (finalDouble)
          totalD += double(part);
        else
          totalF += part;
      }
      (*bx)[std::size_t(i) * cols + j] =
          finalDouble ? float(totalD) : totalF;
    }
}

Stages stepDouble(const std::vector<float>& x, std::uint32_t rows,
                  std::uint32_t cols) {
  using phonelm::nicopedia_muon::kNsA;
  using phonelm::nicopedia_muon::kNsB;
  using phonelm::nicopedia_muon::kNsC;
  Stages s;
  matXXTDouble(x, rows, cols, &s.a);
  matSqDouble(s.a, rows, &s.a2);
  s.b.assign(s.a.size(), 0.0f);
  for (std::size_t i = 0; i < s.b.size(); ++i)
    s.b[i] = float(double(kNsB) * s.a[i] + double(kNsC) * s.a2[i]);
  s.bx.assign(x.size(), 0.0f);
  s.x1.assign(x.size(), 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < cols; ++j) {
      double bx = 0.0;
      for (std::uint32_t k = 0; k < rows; ++k) {
        const double b = double(kNsB) * s.a[std::size_t(i) * rows + k] +
                         double(kNsC) * s.a2[std::size_t(i) * rows + k];
        bx += b * x[std::size_t(k) * cols + j];
      }
      s.bx[std::size_t(i) * cols + j] = float(bx);
      s.x1[std::size_t(i) * cols + j] =
          float(double(kNsA) * x[std::size_t(i) * cols + j] + bx);
    }
  return s;
}

Stages stepFloat(const std::vector<float>& x, std::uint32_t rows,
                 std::uint32_t cols) {
  using phonelm::nicopedia_muon::kNsA;
  using phonelm::nicopedia_muon::kNsB;
  using phonelm::nicopedia_muon::kNsC;
  Stages s;
  matXXTFloat(x, rows, cols, &s.a);
  matSqFloat(s.a, rows, &s.a2);
  s.b.assign(s.a.size(), 0.0f);
  for (std::size_t i = 0; i < s.b.size(); ++i) {
    const float t1 = kNsB * s.a[i];
    const float t2 = kNsC * s.a2[i];
    s.b[i] = t1 + t2;
  }
  s.bx.assign(x.size(), 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < cols; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += s.b[std::size_t(i) * rows + k] * x[std::size_t(k) * cols + j];
      s.bx[std::size_t(i) * cols + j] = sum;
    }
  s.x1.assign(x.size(), 0.0f);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float ax = kNsA * x[i];
    s.x1[i] = ax + s.bx[i];
  }
  return s;
}

Stages stepChunkedFloat(const std::vector<float>& x, std::uint32_t rows,
                        std::uint32_t cols, std::uint32_t chunk) {
  using phonelm::nicopedia_muon::kNsA;
  using phonelm::nicopedia_muon::kNsB;
  using phonelm::nicopedia_muon::kNsC;
  Stages s;
  matXXTChunked(x, rows, cols, chunk, false, &s.a);
  matSqChunked(s.a, rows, chunk, false, &s.a2);
  s.b.assign(s.a.size(), 0.0f);
  for (std::size_t i = 0; i < s.b.size(); ++i) {
    const float t1 = kNsB * s.a[i];
    const float t2 = kNsC * s.a2[i];
    s.b[i] = t1 + t2;
  }
  matBXChunked(s.b, x, rows, cols, chunk, false, &s.bx);
  s.x1.assign(x.size(), 0.0f);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float ax = kNsA * x[i];
    s.x1[i] = ax + s.bx[i];
  }
  return s;
}

std::string metricsJson(const Metrics& m) {
  std::ostringstream out;
  out << std::setprecision(10) << std::showpoint;
  out << "{\"max_abs\":" << m.maxAbs << ",\"mean_abs\":" << m.meanAbs
      << ",\"rms_error\":" << m.rmsErr << ",\"relative_l2\":" << m.relL2
      << ",\"cosine\":" << m.cosine << ",\"mean_signed_error\":"
      << m.meanSigned << ",\"max_ulp\":" << m.maxUlp << ",\"finite\":\""
      << m.finiteCount << "/" << m.total << "\"}";
  return out.str();
}

std::string statsJson(const Stats& s) {
  std::ostringstream out;
  out << std::setprecision(10) << std::showpoint;
  out << "{\"min\":" << s.min << ",\"max\":" << s.max << ",\"rms\":" << s.rms
      << ",\"frob_norm\":" << s.frob << "}";
  return out.str();
}

bool gatePass(const Metrics& m) {
  return std::isfinite(m.maxAbs) && m.maxAbs <= kGateMaxAbs &&
         m.relL2 <= kGateRelL2 && m.cosine >= kGateCosine;
}

void writeFile(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary);
  require(bool(out), "cannot open output file");
  out << content;
  out.flush();
  require(bool(out), "cannot write output file");
}

void writeBin(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  require(bool(out), "cannot open bin output");
  out.write(reinterpret_cast<const char*>(v.data()),
            std::streamsize(v.size() * sizeof(float)));
  out.flush();
  require(bool(out), "cannot write bin output");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: nicopedia_muon_numeric_diagnostic <outDir>");
    const std::string outDir = argv[1];

    const std::uint32_t kRows = 64, kCols = 64;
    const auto gradient = makeValues(64 * 64, 0.001f, 2);
    const auto prior = makeValues(64 * 64, 0.0002f, 3);
    const auto nesterov = buildNesterov(gradient, prior);
    const auto x0 = normalizeDirect(nesterov);
    require(std::all_of(x0.begin(), x0.end(),
                        [](float v) { return std::isfinite(v); }),
            "frozen input nonfinite");

    const Stats x0Stats = stats(x0);
    const std::uint64_t x0Fnv =
        fnv1a64(x0.data(), x0.size() * sizeof(float));
    writeBin(outDir + "/input_f32le.bin", x0);

    const auto rectGradient = makeValues(64 * 128, 0.001f, 5);
    const auto rectPrior = makeValues(64 * 128, 0.0002f, 6);
    const auto rectNesterov = buildNesterov(rectGradient, rectPrior);
    const auto x0r = normalizeDirect(rectNesterov);
    const Stats x0rStats = stats(x0r);
    const std::uint64_t x0rFnv =
        fnv1a64(x0r.data(), x0r.size() * sizeof(float));
    writeBin(outDir + "/input_rect_f32le.bin", x0r);

    {
      std::ostringstream json;
      json << std::setprecision(10) << std::showpoint;
      json << "{\"schema\":\"phonelm.htp_muon.numeric_input.v1\","
           << "\"generator\":\"makeValues(0.02/0.001/0.0002,phases "
              "1/2/3)+validation-style nesterov+direct normalize\","
           << "\"shape\":[64,64],\"elements\":" << x0.size()
           << ",\"sha256_f32le\":\"PENDING_RUNNER_HASH\","
           << "\"fnv1a64\":\"" << fnvHex(x0Fnv) << "\","
           << "\"identity\":" << statsJson(x0Stats)
           << ",\"rect_secondary\":{\"shape\":[64,128],\"elements\":"
           << x0r.size() << ",\"sha256_f32le\":\"PENDING_RUNNER_HASH\","
           << "\"fnv1a64\":\"" << fnvHex(x0rFnv) << "\",\"identity\":"
           << statsJson(x0rStats) << "}}";
      writeFile(outDir + "/input.json", json.str());
    }

    std::string error;
    std::vector<float> oracle1, oracle5;
    require(phonelm::nicopedia_muon::zeropowerNewtonSchulzFp32(
                nesterov, kRows, kCols, 1, &oracle1, nullptr, &error),
            error.c_str());
    require(phonelm::nicopedia_muon::zeropowerNewtonSchulzFp32(
                nesterov, kRows, kCols, 5, &oracle5, nullptr, &error),
            error.c_str());
    std::vector<float> oracleRect1;
    require(phonelm::nicopedia_muon::zeropowerNewtonSchulzFp32(
                rectNesterov, 64, 128, 1, &oracleRect1, nullptr, &error),
            error.c_str());

    const Stages d1 = stepDouble(x0, kRows, kCols);
    const Stages f1 = stepFloat(x0, kRows, kCols);
    const Metrics chainVsOracle1 = compare(oracle1, d1.x1);
    std::vector<float> xd = x0;
    for (int i = 0; i < 5; ++i) xd = stepDouble(xd, kRows, kCols).x1;
    const Metrics chainVsOracle5 = compare(oracle5, xd);

    const Metrics mX0 = compare(x0, x0, false);
    const Metrics mA = compare(d1.a, f1.a);
    const Metrics mA2 = compare(d1.a2, f1.a2);
    const Metrics mB = compare(d1.b, f1.b);
    const Metrics mBX = compare(d1.bx, f1.bx);
    const Metrics mX1 = compare(d1.x1, f1.x1);

    Stages aAbl = stepDouble(x0, kRows, kCols);
    {
      std::vector<float> af;
      matXXTFloat(x0, kRows, kCols, &af);
      Stages rest;
      matSqDouble(af, kRows, &rest.a2);
      rest.bx.assign(x0.size(), 0.0f);
      rest.x1.assign(x0.size(), 0.0f);
      using phonelm::nicopedia_muon::kNsA;
      using phonelm::nicopedia_muon::kNsB;
      using phonelm::nicopedia_muon::kNsC;
      for (std::uint32_t i = 0; i < kRows; ++i)
        for (std::uint32_t j = 0; j < kCols; ++j) {
          double bx = 0.0;
          for (std::uint32_t k = 0; k < kRows; ++k) {
            const double b = double(kNsB) * af[std::size_t(i) * kRows + k] +
                             double(kNsC) * rest.a2[std::size_t(i) * kRows + k];
            bx += b * x0[std::size_t(k) * kCols + j];
          }
          rest.x1[std::size_t(i) * kCols + j] =
              float(double(kNsA) * x0[std::size_t(i) * kCols + j] + bx);
        }
      aAbl = rest;
    }
    const Metrics ablA = compare(d1.x1, aAbl.x1);

    Stages bAbl = stepDouble(x0, kRows, kCols);
    {
      using phonelm::nicopedia_muon::kNsA;
      using phonelm::nicopedia_muon::kNsB;
      using phonelm::nicopedia_muon::kNsC;
      std::vector<float> bf(bAbl.b.size());
      for (std::size_t i = 0; i < bf.size(); ++i) {
        const float t1 = kNsB * bAbl.a[i];
        const float t2 = kNsC * bAbl.a2[i];
        bf[i] = t1 + t2;
      }
      std::vector<float> x1(x0.size());
      for (std::uint32_t i = 0; i < kRows; ++i)
        for (std::uint32_t j = 0; j < kCols; ++j) {
          double bx = 0.0;
          for (std::uint32_t k = 0; k < kRows; ++k)
            bx += double(bf[std::size_t(i) * kRows + k]) *
                  x0[std::size_t(k) * kCols + j];
          x1[std::size_t(i) * kCols + j] =
              float(double(kNsA) * x0[std::size_t(i) * kCols + j] + bx);
        }
      bAbl.x1 = x1;
    }
    const Metrics ablB = compare(d1.x1, bAbl.x1);

    {
      std::ostringstream json;
      json << std::setprecision(10) << std::showpoint;
      json << "{\"schema\":\"phonelm.htp_muon.stage_parity.v1\","
           << "\"gate\":{\"max_abs_lte\":" << kGateMaxAbs
           << ",\"relative_l2_lte\":" << kGateRelL2 << ",\"cosine_gte\":"
           << kGateCosine << "},"
           << "\"chain_vs_oracle\":{\"iter1\":" << metricsJson(chainVsOracle1)
           << ",\"iter5\":" << metricsJson(chainVsOracle5) << "},"
           << "\"float_vs_double\":{\"X0\":" << metricsJson(mX0)
           << ",\"A\":" << metricsJson(mA) << ",\"A2\":" << metricsJson(mA2)
           << ",\"B\":" << metricsJson(mB) << ",\"BX\":" << metricsJson(mBX)
           << ",\"X1\":" << metricsJson(mX1) << "},"
           << "\"first_stage_exceeding_gate\":\""
           << (!gatePass(mA)
                   ? "A"
                   : (!gatePass(mA2)
                          ? "A2"
                          : (!gatePass(mB)
                                 ? "B"
                                 : (!gatePass(mBX) ? "BX"
                                                  : (!gatePass(mX1) ? "X1"
                                                                    : "NONE")))))
           << "\","
           << "\"ablation_x1\":{\"A_float_only\":" << metricsJson(ablA)
           << ",\"B_float_only\":" << metricsJson(ablB) << "},"
           << "\"x1_histogram_counts\":[";
      for (std::size_t i = 0; i < mX1.histogram.size(); ++i)
        json << (i ? "," : "") << mX1.histogram[i];
      json << "],\"x1_histogram_edges\":[1e-7,1e-6,1e-5,1e-4,5e-4,1e-3,1e-2,"
              "1e-1]}";
      writeFile(outDir + "/stage-parity.json", json.str());
    }

    {
      std::ostringstream json;
      json << std::setprecision(10) << std::showpoint;
      json << "{\"schema\":\"phonelm.htp_muon.matmul_precision.v1\",\"probes\":[";
      const std::uint32_t kSizes[] = {16, 32, 64, 128};
      bool first = true;
      for (int c = 0; c < 5; ++c) {
        for (std::uint32_t k : kSizes) {
          std::vector<float> x(std::size_t(k) * k);
          for (std::uint32_t i = 0; i < k; ++i)
            for (std::uint32_t j = 0; j < k; ++j) {
              const std::size_t idx = std::size_t(i) * k + j;
              const float s =
                  std::sin(float(idx + 1) * (0.013f + c * 0.002f));
              if (c == 0)
                x[idx] = 0.02f + 0.01f * s;
              else if (c == 1)
                x[idx] = ((i + j) % 2 ? 1.0f : -1.0f) *
                         (0.05f + 0.01f * s);
              else if (c == 2)
                x[idx] = ((i * 7 + j * 13) % 2 ? 1.0f : -1.0f) *
                         std::pow(10.0f, -4.0f + 6.0f * float(idx) /
                                                 float(k * k - 1));
              else if (c == 3) {
                const float raw =
                    (s + float(int(idx % 17) - 8) * 0.03125f) * 0.001f;
                x[idx] = raw;
              } else {
                x[idx] = (i == j ? 1.0f : 0.0f) + 1.0e-3f * s;
              }
            }
          std::vector<float> xn = x;
          if (c == 3) xn = normalizeDirect(x);
          std::vector<float> cd, cf, xh(xn.size());
          for (std::size_t i = 0; i < xh.size(); ++i)
            xh[i] = quantizeHalf(xn[i]);
          std::vector<float> ch;
          matXXTDouble(xn, k, k, &cd);
          matXXTFloat(xn, k, k, &cf);
          matXXTDouble(xh, k, k, &ch);
          const Metrics fvd = compare(cd, cf, false);
          const Metrics hvd = compare(cd, ch, false);
          static const char* kNames[] = {"positive_small", "cancellation",
                                         "wide_range", "muon_range",
                                         "identity_like"};
          json << (first ? "" : ",") << "{\"case\":\"" << kNames[c]
               << "\",\"k\":" << k
               << ",\"float_vs_double\":" << metricsJson(fvd)
               << ",\"halfinput_vs_double\":" << metricsJson(hvd) << "}";
          first = false;
        }
      }
      json << "],\"multiply\":[";
      {
        const std::vector<std::pair<std::string, std::vector<float>>> vecs = {
            {"x0_row", std::vector<float>(x0.begin(), x0.begin() + 64)},
            {"a_row", std::vector<float>(d1.a.begin(), d1.a.begin() + 64)},
        };
        bool mfirst = true;
        for (const auto& named : vecs) {
          const auto& v = named.second;
          std::vector<float> fm(v.size()), dm(v.size());
          double maxUlp = 0.0;
          for (std::size_t i = 0; i < v.size(); ++i) {
            fm[i] = v[i] * v[i];
            dm[i] = float(double(v[i]) * v[i]);
            maxUlp = std::max(maxUlp, double(ulpDistance(dm[i], fm[i])));
          }
          const Metrics mm = compare(dm, fm, false);
          json << (mfirst ? "" : ",") << "{\"vector\":\"" << named.first
               << "\",\"float_mul_vs_double_round\":"
               << metricsJson(mm) << ",\"max_ulp\":" << maxUlp << "}";
          mfirst = false;
        }
      }
      json << "],\"norm_prescale\":";
      {
        double normSq = 0.0, scaledSq = 0.0;
        for (float v : nesterov) {
          normSq += double(v) * v;
          const double s = double(v) * 1024.0;
          scaledSq += s * s;
        }
        std::vector<float> ref(x0.size()), pre(x0.size());
        const double dDen = std::sqrt(normSq) + 1.0e-7;
        const double pDen = std::sqrt(scaledSq) + 1024.0 * 1.0e-7;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          ref[i] = float(double(nesterov[i]) / dDen);
          pre[i] = float((double(nesterov[i]) * 1024.0) / pDen);
        }
        float fSum = 0.0f;
        for (float v : nesterov) fSum += v * v;
        std::vector<float> fdir(x0.size());
        const float fDen = std::sqrt(fSum) + 1.0e-7f;
        for (std::size_t i = 0; i < fdir.size(); ++i)
          fdir[i] = nesterov[i] / fDen;
        std::vector<float> halfSq(x0.size());
        for (std::size_t i = 0; i < halfSq.size(); ++i) {
          const float h = quantizeHalf(nesterov[i]);
          const float sq = quantizeHalf(h * h);
          halfSq[i] = sq;
        }
        double hSum = 0.0;
        for (float v : halfSq) hSum += double(v);
        std::vector<float> hnorm(x0.size());
        const double hDen = std::sqrt(hSum) + 1.0e-7;
        for (std::size_t i = 0; i < hnorm.size(); ++i)
          hnorm[i] = float(double(quantizeHalf(nesterov[i])) / hDen);
        json << "{\"prescaled_double_vs_direct\":" << metricsJson(compare(ref, pre, false))
             << ",\"float_direct_vs_double\":" << metricsJson(compare(ref, fdir, false))
             << ",\"half_square_vs_double\":" << metricsJson(compare(ref, hnorm, false))
             << "}";
      }
      json << "}";
      writeFile(outDir + "/matmul-precision.json", json.str());
    }

    {
      std::ostringstream json;
      json << std::setprecision(10) << std::showpoint;
      json << "{\"schema\":\"phonelm.htp_muon.chunked_matmul.v1\",\"A\":{";
      const std::uint32_t kChunks[] = {64, 32, 16, 8};
      bool first = true;
      std::string best;
      double bestRel = std::numeric_limits<double>::infinity();
      for (std::uint32_t chunk : kChunks) {
        std::vector<float> ac;
        if (chunk >= 64)
          matXXTFloat(x0, kRows, kCols, &ac);
        else
          matXXTChunked(x0, kRows, kCols, chunk, false, &ac);
        const Metrics m = compare(d1.a, ac, false);
        json << (first ? "" : ",") << "\"chunk_" << chunk << "\":"
             << metricsJson(m);
        first = false;
        if (chunk != 64 && m.relL2 < bestRel) {
          bestRel = m.relL2;
          best = "chunk_" + std::to_string(chunk);
        }
      }
      json << "},\"iter1_x1\":{";
      first = true;
      std::string bestX1;
      double bestX1Rel = std::numeric_limits<double>::infinity();
      bool anyPass = false;
      for (std::uint32_t chunk : kChunks) {
        Stages s = (chunk >= 64) ? stepFloat(x0, kRows, kCols)
                                 : stepChunkedFloat(x0, kRows, kCols, chunk);
        const Metrics m = compare(d1.x1, s.x1, false);
        const bool pass = gatePass(m);
        anyPass = anyPass || (pass && chunk != 64);
        json << (first ? "" : ",") << "\"chunk_" << chunk
             << "\":{\"metrics\":" << metricsJson(m) << ",\"gate_pass\":"
             << (pass ? "true" : "false") << "}";
        first = false;
        if (chunk != 64 && m.relL2 < bestX1Rel) {
          bestX1Rel = m.relL2;
          bestX1 = "chunk_" + std::to_string(chunk);
        }
      }
      json << "},\"best_A\":\"" << best << "\",\"best_X1\":\"" << bestX1
           << "\",\"any_chunked_iter1_gate_pass\":" << (anyPass ? "true" : "false")
           << "}";
      writeFile(outDir + "/chunked-matmul.json", json.str());
    }

    {
      std::ostringstream json;
      json << std::setprecision(10) << std::showpoint;
      json << "{\"schema\":\"phonelm.htp_muon.ns_iterations.v1\",\"square\":[";
      std::vector<float> xd = x0, xf = x0;
      for (int iter = 0; iter <= 5; ++iter) {
        Stages sd, sf;
        if (iter > 0) {
          sd = stepDouble(xd, kRows, kCols);
          sf = stepFloat(xf, kRows, kCols);
          xd = sd.x1;
          xf = sf.x1;
        }
        const Metrics xm = (iter == 0) ? compare(x0, x0, false)
                                       : compare(xd, xf, false);
        json << (iter ? "," : "") << "{\"iter\":" << iter
             << ",\"float_vs_double\":" << metricsJson(xm)
             << ",\"frob_double\":" << stats(xd).frob
             << ",\"frob_float\":" << stats(xf).frob
             << ",\"orth_double\":" << orthDeviation(xd, kRows, kCols)
             << ",\"orth_float\":" << orthDeviation(xf, kRows, kCols);
        if (iter > 0) {
          json << ",\"mag_double\":{\"A\":" << statsJson(stats(sd.a))
               << ",\"A2\":" << statsJson(stats(sd.a2))
               << ",\"B\":" << statsJson(stats(sd.b))
               << ",\"BX\":" << statsJson(stats(sd.bx))
               << ",\"X\":" << statsJson(stats(sd.x1)) << "}"
               << ",\"mag_float\":{\"A\":" << statsJson(stats(sf.a))
               << ",\"A2\":" << statsJson(stats(sf.a2))
               << ",\"B\":" << statsJson(stats(sf.b))
               << ",\"BX\":" << statsJson(stats(sf.bx))
               << ",\"X\":" << statsJson(stats(sf.x1)) << "}";
        }
        json << "}";
      }
      json << "],\"rect_secondary\":[";
      std::vector<float> rd = x0r, rf = x0r;
      for (int iter = 0; iter <= 5; ++iter) {
        if (iter > 0) {
          rd = stepDouble(rd, 64, 128).x1;
          rf = stepFloat(rf, 64, 128).x1;
        }
        const Metrics xm = (iter == 0) ? compare(x0r, x0r, false)
                                       : compare(rd, rf, false);
        json << (iter ? "," : "") << "{\"iter\":" << iter
             << ",\"float_vs_double\":" << metricsJson(xm)
             << ",\"orth_double\":" << orthDeviation(rd, 64, 128)
             << ",\"orth_float\":" << orthDeviation(rf, 64, 128) << "}";
      }
      json << "]}";
      writeFile(outDir + "/ns-iteration-error.json", json.str());
    }

    require(chainVsOracle1.relL2 <= 1.0e-6, "double chain diverges from oracle");
    require(chainVsOracle5.relL2 <= 1.0e-6, "double chain diverges at iter5");
    require(mX1.finiteCount == mX1.total, "nonfinite diagnostic");
    std::cout << "nicopedia_muon_numeric_diagnostic=PASS\n";
    std::cout << "input_fnv=" << fnvHex(x0Fnv) << "\n";
    std::cout << "chain_vs_oracle_iter1_relL2=" << std::setprecision(10)
              << chainVsOracle1.relL2 << "\n";
    std::cout << "float_vs_double_X1_relL2=" << mX1.relL2 << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
