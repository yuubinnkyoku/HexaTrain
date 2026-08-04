// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only output-projection information audit for the L19 transformer
// (OUTPUT_PROJECTION_AUDIT_V1 protocol, fixed before results).
//
// Goal: determine whether the attention output projection actually loses
// linear next-token information (rank deficiency) or merely makes it
// numerically/optimizationally harder to read (bad conditioning / probe
// standardization / initialization).
//
// The math follows the actual code orientation:
//   y = x W
//   x : 1 x dim concatenated head context (CTX_CONCAT)
//   W : dim x dim Attention output projection (p.wo, row-major)
//   y : 1 x dim ATT_UPDATE
// The context probe is logits = z_x A + b with A class-major, i.e.
// logit[c] = b[c] + sum_d z_x[d] * A[c*dim+d]. The transport to the
// projection side solves C = B_raw * W^T with C = A * D_x^-1, then
// re-standardizes on the y side.
//
// No device, QAIRT, ADB, QNN graph, or Android involvement. Production code
// is not modified.
#ifndef OUTPUT_PROJECTION_AUDIT_LIB_H
#define OUTPUT_PROJECTION_AUDIT_LIB_H

#include "attention_internal_diagnosis_lib.h"
#include "readout_probe_lib.h"
#include "critical_margin_training_lib.h"
#include "intra_block_readability_lib.h"
#include "depth_quality_lib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::output_projection_audit {

namespace aid = phonelm::attention_internal;
namespace rp = phonelm::readout_probe;
namespace train = phonelm::critical_margin::train;
namespace tiny = phonelm::tiny_lm;
namespace dq = phonelm::depth_quality;
namespace ibr = phonelm::intra_block_readability;

inline constexpr const char* kProtocolId = "OUTPUT_PROJECTION_AUDIT_V1";
inline constexpr double kDoubleEps = 2.220446049250313e-16;
inline constexpr double kFloatEps = 1.1920928955078125e-7;
inline constexpr double kTransportLogitTol = 1.0e-5;
inline constexpr int kJacobiSweeps = 24;
inline constexpr double kJacobiStopOff = 1e-24;
inline constexpr double kJacobiSkip = 1e-18;
inline constexpr double kPinvRcond = 1.0e-15;  // not used; explicit tol preferred
inline constexpr std::uint32_t kTokens = 8;
inline constexpr std::uint32_t kClasses = 32;

// ---------------------------------------------------------------------------
// Small dense matrix helpers (templated so we can run the same code in float
// and double for the float/double comparison).
// ---------------------------------------------------------------------------
template <typename T>
using Mat = std::vector<std::vector<T>>;

template <typename T>
inline Mat<T> zerosMat(int rows, int cols) {
  return Mat<T>(static_cast<std::size_t>(rows),
                std::vector<T>(static_cast<std::size_t>(cols), T(0)));
}

template <typename T>
inline Mat<T> identityMat(int n) {
  Mat<T> m = zerosMat<T>(n, n);
  for (int i = 0; i < n; ++i) m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = T(1);
  return m;
}

template <typename T>
inline Mat<T> transposeMat(const Mat<T>& a) {
  const int r = static_cast<int>(a.size());
  const int c = static_cast<int>(a[0].size());
  Mat<T> t = zerosMat<T>(c, r);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j)
      t[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  return t;
}

template <typename T>
inline Mat<T> mulMat(const Mat<T>& a, const Mat<T>& b) {
  const int r = static_cast<int>(a.size());
  const int k = static_cast<int>(a[0].size());
  const int c = static_cast<int>(b[0].size());
  if (static_cast<int>(b.size()) != k) throw std::invalid_argument("MATMUL_DIM");
  Mat<T> o = zerosMat<T>(r, c);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) {
      T s = T(0);
      for (int z = 0; z < k; ++z)
        s += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(z)] *
             b[static_cast<std::size_t>(z)][static_cast<std::size_t>(j)];
      o[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s;
    }
  return o;
}

template <typename T>
inline T frobeniusNormSq(const Mat<T>& a) {
  T s = T(0);
  for (const auto& row : a)
    for (T x : row) s += x * x;
  return s;
}

template <typename T>
inline T frobeniusNorm(const Mat<T>& a) {
  return std::sqrt(frobeniusNormSq(a));
}

// ---------------------------------------------------------------------------
// Deterministic symmetric eigendecomposition by cyclic Jacobi.
// Returns eigenvalues descending and eigenvectors as columns.
// ---------------------------------------------------------------------------
template <typename T>
struct SymEig {
  std::vector<T> values;                  // descending
  Mat<T> vectors;                         // n x n, column j = eigenvector
};

template <typename T>
inline SymEig<T> symmetricEigen(const Mat<T>& matrix) {
  const int n = static_cast<int>(matrix.size());
  Mat<T> a = matrix;
  Mat<T> v = zerosMat<T>(n, n);
  for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = T(1);
  for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
    T off = T(0);
    for (int p = 0; p < n - 1; ++p)
      for (int q = p + 1; q < n; ++q)
        off += a[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] *
               a[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)];
    if (off < T(kJacobiStopOff)) break;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const T apq = a[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)];
        if (std::abs(apq) < T(kJacobiSkip)) continue;
        const T theta = (a[static_cast<std::size_t>(q)][static_cast<std::size_t>(q)] -
                         a[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)]) /
                        (T(2) * apq);
        const T t = std::copysign(T(1), theta) /
                    (std::abs(theta) + std::sqrt(theta * theta + T(1)));
        const T c = T(1) / std::sqrt(t * t + T(1));
        const T s = t * c;
        for (int k = 0; k < n; ++k) {
          if (k == p || k == q) continue;
          const T akp = a[static_cast<std::size_t>(k)][static_cast<std::size_t>(p)];
          const T akq = a[static_cast<std::size_t>(k)][static_cast<std::size_t>(q)];
          a[static_cast<std::size_t>(k)][static_cast<std::size_t>(p)] = c * akp - s * akq;
          a[static_cast<std::size_t>(k)][static_cast<std::size_t>(q)] = s * akp + c * akq;
          a[static_cast<std::size_t>(p)][static_cast<std::size_t>(k)] = a[static_cast<std::size_t>(k)][static_cast<std::size_t>(p)];
          a[static_cast<std::size_t>(q)][static_cast<std::size_t>(k)] = a[static_cast<std::size_t>(k)][static_cast<std::size_t>(q)];
        }
        const T app = a[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)];
        const T aqq = a[static_cast<std::size_t>(q)][static_cast<std::size_t>(q)];
        a[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)] =
            c * c * app - T(2) * c * s * apq + s * s * aqq;
        a[static_cast<std::size_t>(q)][static_cast<std::size_t>(q)] =
            s * s * app + T(2) * c * s * apq + c * c * aqq;
        a[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] = T(0);
        a[static_cast<std::size_t>(q)][static_cast<std::size_t>(p)] = T(0);
        for (int k = 0; k < n; ++k) {
          const T vkp = v[static_cast<std::size_t>(k)][static_cast<std::size_t>(p)];
          const T vkq = v[static_cast<std::size_t>(k)][static_cast<std::size_t>(q)];
          v[static_cast<std::size_t>(k)][static_cast<std::size_t>(p)] = c * vkp - s * vkq;
          v[static_cast<std::size_t>(k)][static_cast<std::size_t>(q)] = s * vkp + c * vkq;
        }
      }
    }
  }
  std::vector<T> raw(n);
  for (int i = 0; i < n; ++i) {
    T val = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    // Only suppress tiny negative round-off; preserve small positive values
    // so the caller's absolute tolerance decides rank.
    if (val < T(0) && std::abs(val) < T(1e-30)) val = T(0);
    raw[static_cast<std::size_t>(i)] = val;
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    return raw[static_cast<std::size_t>(x)] > raw[static_cast<std::size_t>(y)];
  });
  SymEig<T> result;
  result.values.resize(static_cast<std::size_t>(n));
  result.vectors = zerosMat<T>(n, n);
  for (int j = 0; j < n; ++j) {
    const int src = order[static_cast<std::size_t>(j)];
    result.values[static_cast<std::size_t>(j)] = raw[static_cast<std::size_t>(src)];
    for (int k = 0; k < n; ++k)
      result.vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
          v[static_cast<std::size_t>(k)][static_cast<std::size_t>(src)];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Thin SVD W = U S V^T via eig(W^T W). U columns recovered as W*v/sigma.
// For square W, this is the full SVD.
// ---------------------------------------------------------------------------
template <typename T>
struct Svd {
  Mat<T> u;            // n x n, columns are left singular vectors
  std::vector<T> s;    // descending singular values
  Mat<T> vt;           // n x n, rows are right singular vectors (V^T)
};

template <typename T>
inline Svd<T> computeSvd(const Mat<T>& w) {
  const int n = static_cast<int>(w.size());
  const Mat<T> wt = transposeMat(w);
  const Mat<T> g = mulMat(wt, w);
  const SymEig<T> eig = symmetricEigen(g);
  Svd<T> result;
  result.s.resize(static_cast<std::size_t>(n));
  result.vt = zerosMat<T>(n, n);
  result.u = zerosMat<T>(n, n);
  for (int j = 0; j < n; ++j) {
    const T lam = eig.values[static_cast<std::size_t>(j)];
    const T sigma = lam > T(0) ? std::sqrt(lam) : T(0);
    result.s[static_cast<std::size_t>(j)] = sigma;
    // column j of V is eig.vectors[*][j]
    for (int i = 0; i < n; ++i)
      result.vt[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
          eig.vectors[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    if (sigma > T(0)) {
      for (int i = 0; i < n; ++i) {
        T acc = T(0);
        for (int k = 0; k < n; ++k)
          acc += w[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                 eig.vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
        result.u[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = acc / sigma;
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Moore-Penrose pseudoinverse W^+ = V S^+ U^T.
// tol is absolute on singular values.
// ---------------------------------------------------------------------------
template <typename T>
inline Mat<T> pseudoInverse(const Mat<T>& w, T tol) {
  const Svd<T> svd = computeSvd(w);
  const int n = static_cast<int>(svd.s.size());
  Mat<T> pinv = zerosMat<T>(n, n);
  for (int k = 0; k < n; ++k) {
    const T sigma = svd.s[static_cast<std::size_t>(k)];
    if (sigma <= tol) continue;
    const T inv = T(1) / sigma;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        pinv[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
            inv * svd.vt[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] *
            svd.u[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
  }
  return pinv;
}

// ---------------------------------------------------------------------------
// Matrix decomposition statistics.
// ---------------------------------------------------------------------------
struct MatrixStats {
  int dim = 0;
  std::vector<double> singularValues;   // descending
  double sigmaMax = 0.0;
  double sigmaMin = 0.0;
  int mathRank = 0;
  int floatRank = 0;
  double conditionDouble = 0.0;
  double conditionFloat = 0.0;
  double effectiveRank = 0.0;             // exp(-sum p log p), p = sigma/sum
  double participationRatio = 0.0;        // (sum sigma^2)^2 / sum sigma^4
  double frobeniusNorm = 0.0;
  double spectralNorm = 0.0;
  int determinantSign = 0;
  double logAbsDeterminant = 0.0;
  double doubleTol = 0.0;
  double floatTol = 0.0;
  bool ambiguousRank = false;
};

inline MatrixStats analyzeMatrix(const Mat<double>& w) {
  MatrixStats stats;
  stats.dim = static_cast<int>(w.size());
  const Svd<double> svd = computeSvd(w);
  stats.singularValues = svd.s;
  stats.sigmaMax = svd.s.empty() ? 0.0 : svd.s.front();
  stats.sigmaMin = 0.0;
  for (int i = static_cast<int>(svd.s.size()) - 1; i >= 0; --i)
    if (svd.s[static_cast<std::size_t>(i)] > 0.0) {
      stats.sigmaMin = svd.s[static_cast<std::size_t>(i)];
      break;
    }
  stats.doubleTol = static_cast<double>(stats.dim) * kDoubleEps * stats.sigmaMax;
  stats.floatTol = static_cast<double>(stats.dim) * kFloatEps * stats.sigmaMax;
  for (double s : svd.s) {
    if (s > stats.doubleTol) ++stats.mathRank;
    if (s > stats.floatTol) ++stats.floatRank;
  }
  stats.ambiguousRank = stats.sigmaMin > 0.0 && stats.sigmaMin < 10.0 * stats.doubleTol;
  stats.conditionDouble = stats.sigmaMin > 0.0 ? stats.sigmaMax / stats.sigmaMin
                                               : std::numeric_limits<double>::infinity();
  stats.conditionFloat = stats.sigmaMin > 0.0 ? stats.sigmaMax / std::max(stats.sigmaMin, stats.floatTol)
                                              : std::numeric_limits<double>::infinity();
  double sum = 0.0, sumSq = 0.0, sumFourth = 0.0, ent = 0.0;
  for (double s : svd.s) {
    sum += s;
    sumSq += s * s;
    sumFourth += s * s * s * s;
  }
  if (sum > 0.0) {
    for (double s : svd.s) {
      const double p = s / sum;
      if (p > 0.0) ent -= p * std::log(p);
    }
  }
  stats.effectiveRank = std::exp(ent);
  stats.participationRatio = sumFourth > 0.0 ? sumSq * sumSq / sumFourth : 0.0;
  stats.frobeniusNorm = std::sqrt(sumSq);
  stats.spectralNorm = stats.sigmaMax;
  double logDet = 0.0;
  int sign = 1;
  for (double s : svd.s) {
    if (s > 0.0) logDet += std::log(s);
    if (s < 0.0) sign = -sign;  // should not happen for singular values
  }
  // Determinant sign from original matrix (U*V^T det sign) equals sign of det(W).
  // For square W, det(W) = product(s) * det(U)*det(V). Singular values are non-negative,
  // so det sign = det(U)*det(V). We approximate by checking if W is orientation-preserving
  // via the sign of the product of the diagonal of R from a tiny QR, but for our square
  // 16x16 W we can just compute det(W) directly with double LU for the sign.
  // Fallback: report sign based on singular-value product parity (1 for full rank if sigma>0).
  stats.determinantSign = sign;
  stats.logAbsDeterminant = logDet;
  return stats;
}

// Determinant sign via recursive cofactor is too slow for 16x16; use LU.
inline double luDeterminantSign(const Mat<double>& a) {
  const int n = static_cast<int>(a.size());
  Mat<double> m = a;
  int swaps = 0;
  for (int i = 0; i < n; ++i) {
    int piv = i;
    double best = std::abs(m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)]);
    for (int r = i + 1; r < n; ++r) {
      const double v = std::abs(m[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)]);
      if (v > best) { best = v; piv = r; }
    }
    if (piv != i) {
      std::swap(m[static_cast<std::size_t>(piv)], m[static_cast<std::size_t>(i)]);
      ++swaps;
    }
    const double diag = m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    if (std::abs(diag) < 1e-18) return 0.0;
    for (int r = i + 1; r < n; ++r) {
      const double f = m[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)] / diag;
      for (int c = i; c < n; ++c)
        m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] -= f * m[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
    }
  }
  double det = (swaps % 2 == 0) ? 1.0 : -1.0;
  for (int i = 0; i < n; ++i) det *= m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
  if (det > 0.0) return 1.0;
  if (det < 0.0) return -1.0;
  return 0.0;
}

inline MatrixStats analyzeMatrixWithDetSign(const Mat<double>& w) {
  MatrixStats stats = analyzeMatrix(w);
  stats.determinantSign = static_cast<int>(luDeterminantSign(w));
  return stats;
}

// ---------------------------------------------------------------------------
// Convert the flat float p.wo into a dim x dim matrix in the actual code
// orientation: y = x W, W[i][j] = p.wo[i*dim + j].
// ---------------------------------------------------------------------------
inline Mat<double> outputProjectionMatrix(const std::vector<float>& wo, int dim) {
  Mat<double> w = zerosMat<double>(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          static_cast<double>(wo[static_cast<std::size_t>(i) * dim + j]);
  return w;
}

// ---------------------------------------------------------------------------
// Transport a context probe to the projection-side representation.
//
// Context:  logit = b + z_x A, z_x = (x - mu_x) D_x^-1, A class-major.
// Raw:      logit = b_raw + x C, C = A D_x^-1, b_raw = b - mu_x C^T.
// Projection: y = x W.
// Want:     logit = b_yraw + y B_raw, with C = B_raw W^T.
// Solve:    B_raw = C (W^T)^+ = C (W^+)^T.
// Then:    B = B_raw D_y, b_y = b_raw + B_raw mu_y^T.
// ---------------------------------------------------------------------------
struct TransportResult {
  rp::Probe probe;              // standardized projection-side probe
  bool finite = true;
  Mat<double> c;                // classes x dim raw context coefficients
  Mat<double> bRaw;             // classes x 1 raw bias (as vector)
  Mat<double> bRawProjection;   // classes x 1 raw bias on projection side
  Mat<double> bProjection;      // classes x 1 standardized bias
  Mat<double> wPinv;            // dim x dim pseudoinverse of W
  double nullspaceFraction = 0.0;
  Mat<double> cLost;            // classes x dim lost components
  Mat<double> cRecoverable;     // classes x dim recoverable components
};

inline TransportResult transportProbe(const rp::Probe& ctxProbe,
                                      const rp::ZStats& ctxStats,
                                      const rp::ZStats& projStats,
                                      const Mat<double>& w,
                                      double pinvTol) {
  TransportResult result;
  const int classes = ctxProbe.classes;
  const int dim = ctxProbe.dim;
  if (dim != ctxStats.dim || dim != projStats.dim)
    throw std::invalid_argument("TRANSPORT_DIM_MISMATCH");

  // C = A * D_x^-1
  result.c = zerosMat<double>(classes, dim);
  for (int c = 0; c < classes; ++c)
    for (int d = 0; d < dim; ++d) {
      const double std = ctxStats.std[static_cast<std::size_t>(d)];
      result.c[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
          std > rp::kProbeStdFloor
              ? ctxProbe.w[static_cast<std::size_t>(c) * dim + d] / std
              : 0.0;
    }

  // b_raw = b - mu_x C^T  => b_raw[c] = b[c] - sum_d mu_x[d] * C[c,d]
  result.bRaw = zerosMat<double>(classes, 1);
  for (int c = 0; c < classes; ++c) {
    double bias = ctxProbe.b[static_cast<std::size_t>(c)];
    for (int d = 0; d < dim; ++d)
      bias -= ctxStats.mean[static_cast<std::size_t>(d)] *
              result.c[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)];
    result.bRaw[static_cast<std::size_t>(c)][0] = bias;
  }

  // W^+
  result.wPinv = pseudoInverse(w, pinvTol);

  // B_raw = C * (W^+)^T
  const Mat<double> wpinvT = transposeMat(result.wPinv);
  const Mat<double> bRawMat = mulMat(result.c, wpinvT);  // classes x dim

  // Projection-side standardized probe: B = B_raw * D_y
  rp::Probe projProbe = rp::zeroProbe(dim);
  projProbe.classes = classes;
  for (int c = 0; c < classes; ++c)
    for (int d = 0; d < dim; ++d) {
      const double std = projStats.std[static_cast<std::size_t>(d)];
      projProbe.w[static_cast<std::size_t>(c) * dim + d] =
          std > rp::kProbeStdFloor
              ? bRawMat[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] * std
              : 0.0;
    }

  // b_yraw = b_raw; b_y = b_raw + B_raw * mu_y^T
  result.bRawProjection = result.bRaw;
  result.bProjection = zerosMat<double>(classes, 1);
  for (int c = 0; c < classes; ++c) {
    double bias = result.bRaw[static_cast<std::size_t>(c)][0];
    for (int d = 0; d < dim; ++d)
      bias += bRawMat[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] *
              projStats.mean[static_cast<std::size_t>(d)];
    result.bProjection[static_cast<std::size_t>(c)][0] = bias;
    projProbe.b[static_cast<std::size_t>(c)] = bias;
  }

  // Null-space component: C_recoverable = C * W * W^+
  const Mat<double> p = mulMat(w, result.wPinv);
  result.cRecoverable = mulMat(result.c, p);
  result.cLost = zerosMat<double>(classes, dim);
  for (int c = 0; c < classes; ++c)
    for (int d = 0; d < dim; ++d)
      result.cLost[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
          result.c[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] -
          result.cRecoverable[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)];
  const double cNormSq = frobeniusNormSq(result.c);
  const double lostNormSq = frobeniusNormSq(result.cLost);
  result.nullspaceFraction = cNormSq > 0.0 ? std::sqrt(lostNormSq / cNormSq) : 0.0;

  // Finite check
  for (double x : projProbe.w)
    if (!std::isfinite(x)) result.finite = false;
  for (double x : projProbe.b)
    if (!std::isfinite(x)) result.finite = false;
  for (const auto& row : result.cLost)
    for (double x : row)
      if (!std::isfinite(x)) result.finite = false;

  result.probe = std::move(projProbe);
  return result;
}

// ---------------------------------------------------------------------------
// Train a projection-side probe starting from a supplied probe (warm start).
// Identical protocol to rp::trainProbe except the initial probe is fixed.
// ---------------------------------------------------------------------------
inline rp::ProbeTrainResult trainProbeFromInit(
    const rp::LayerSet& set, int rep, const rp::ZStats& stats,
    const rp::Probe& initProbe,
    const std::vector<rp::ProbeRow>& trainRows, std::size_t trainBegin,
    std::size_t trainEnd, const std::vector<rp::ProbeRow>& calRows,
    std::size_t calBegin, std::size_t calEnd) {
  rp::ProbeTrainResult result;
  result.probe = initProbe;
  const std::size_t wn = result.probe.w.size();
  std::vector<double> m(wn + rp::kClasses, 0.0), v(wn + rp::kClasses, 0.0);
  std::vector<double> gradW(wn, 0.0), gradB(rp::kClasses, 0.0);
  std::vector<double> logits(rp::kClasses);
  const auto& f = set.features.at(static_cast<std::size_t>(rep));

  const auto evaluate = [&](const std::vector<rp::ProbeRow>& rows,
                            std::size_t begin, std::size_t end) {
    return rp::evaluateProbeRows(result.probe, stats, set, rep, rows, begin, end);
  };

  {
    rp::ProbeGridPoint point;
    point.step = 0;
    const auto t = evaluate(trainRows, trainBegin, trainEnd);
    const auto c = evaluate(calRows, calBegin, calEnd);
    point.trainCe = t.ce;
    point.calCe = c.ce;
    point.trainExact = t.exact;
    point.calExact = c.exact;
    if (!t.finite || !c.finite) result.finite = false;
    result.grid.push_back(point);
  }

  const double lr = rp::kProbeLr;
  for (int step = 1; step <= rp::kProbeSteps; ++step) {
    std::fill(gradW.begin(), gradW.end(), 0.0);
    std::fill(gradB.begin(), gradB.end(), 0.0);
    const double invN = 1.0 / static_cast<double>(trainEnd - trainBegin);
    for (std::size_t r = trainBegin; r < trainEnd; ++r) {
      rp::probeForward(result.probe, stats, f.data() + r * set.dim, logits.data());
      for (const double l : logits)
        if (!std::isfinite(l)) {
          result.finite = false;
          result.nonfiniteStep = step;
          result.nonfiniteWhat = "train_logits";
          return result;
        }
      const auto p = rp::softmaxRow(logits.data(), result.probe.classes);
      for (int c = 0; c < result.probe.classes; ++c) {
        const double delta =
            p[static_cast<std::size_t>(c)] -
            (c == static_cast<int>(rp::rowsTruth(trainRows, r)) ? 1.0 : 0.0);
        gradB[static_cast<std::size_t>(c)] += delta * invN;
        for (int d = 0; d < result.probe.dim; ++d)
          gradW[static_cast<std::size_t>(c) * result.probe.dim + d] +=
              delta * rp::zScore(stats, f.data() + r * set.dim, d) * invN;
      }
    }
    if (!rp::adamStepProbe(result.probe, m, v, gradW, gradB, step, lr)) {
      result.finite = false;
      result.nonfiniteStep = step;
      result.nonfiniteWhat = "adam_update";
      return result;
    }
    if (step % rp::kProbeGridStep == 0 || step == rp::kProbeSteps) {
      rp::ProbeGridPoint point;
      point.step = step;
      const auto t = evaluate(trainRows, trainBegin, trainEnd);
      const auto c = evaluate(calRows, calBegin, calEnd);
      point.trainCe = t.ce;
      point.calCe = c.ce;
      point.trainExact = t.exact;
      point.calExact = c.exact;
      if (!t.finite || !c.finite) {
        result.finite = false;
        result.nonfiniteStep = step;
        result.nonfiniteWhat = "grid_eval";
        return result;
      }
      result.grid.push_back(point);
    }
  }

  const rp::ProbeGridPoint* best = nullptr;
  for (const auto& point : result.grid) {
    if (!best) { best = &point; continue; }
    const double delta = point.calCe - best->calCe;
    bool better = false;
    if (delta < -rp::kTieTolerance) better = true;
    else if (std::abs(delta) <= rp::kTieTolerance) {
      if (point.calExact > best->calExact) better = true;
      else if (point.calExact == best->calExact && point.step < best->step)
        better = true;
    }
    if (better) best = &point;
  }
  if (!best) throw std::runtime_error("PROBE_EMPTY_GRID");
  result.selectedStep = best->step;
  result.trainCe = best->trainCe;
  result.trainExact = best->trainExact;
  result.calCe = best->calCe;
  result.calExact = best->calExact;

  if (best->step != rp::kProbeSteps) {
    rp::Probe replay = initProbe;
    std::vector<double> rm(wn + rp::kClasses, 0.0), rv(wn + rp::kClasses, 0.0);
    for (int step = 1; step <= best->step; ++step) {
      std::fill(gradW.begin(), gradW.end(), 0.0);
      std::fill(gradB.begin(), gradB.end(), 0.0);
      const double invN = 1.0 / static_cast<double>(trainEnd - trainBegin);
      for (std::size_t r = trainBegin; r < trainEnd; ++r) {
        rp::probeForward(replay, stats, f.data() + r * set.dim, logits.data());
        const auto p = rp::softmaxRow(logits.data(), replay.classes);
        for (int c = 0; c < replay.classes; ++c) {
          const double delta =
              p[static_cast<std::size_t>(c)] -
              (c == static_cast<int>(rp::rowsTruth(trainRows, r)) ? 1.0 : 0.0);
          gradB[static_cast<std::size_t>(c)] += delta * invN;
          for (int d = 0; d < replay.dim; ++d)
            gradW[static_cast<std::size_t>(c) * replay.dim + d] +=
                delta * rp::zScore(stats, f.data() + r * set.dim, d) * invN;
        }
      }
      if (!rp::adamStepProbe(replay, rm, rv, gradW, gradB, step, lr)) {
        result.finite = false;
        result.nonfiniteStep = step;
        result.nonfiniteWhat = "replay_adam";
        return result;
      }
    }
    result.probe = std::move(replay);
  }

  {
    const auto t = evaluate(trainRows, trainBegin, trainEnd);
    const auto c = evaluate(calRows, calBegin, calEnd);
    result.trainCe = t.ce;
    result.trainExact = t.exact;
    result.calCe = c.ce;
    result.calExact = c.exact;
    if (!t.finite || !c.finite) {
      result.finite = false;
      result.nonfiniteStep = best->step;
      result.nonfiniteWhat = "selected_eval";
    }
    double maxAbs = 0.0;
    for (double x : result.probe.w) maxAbs = std::max(maxAbs, std::abs(x));
    for (double x : result.probe.b) maxAbs = std::max(maxAbs, std::abs(x));
    result.maxLogitAbs = maxAbs;
  }
  if (!result.grid.empty()) {
    const auto& last = result.grid.back();
    result.trainCe2000 = last.trainCe;
    result.trainExact2000 = last.trainExact;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Compare logits of two probes on two different taps for the same rows.
// ---------------------------------------------------------------------------
struct LogitParity {
  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  double rmsDiff = 0.0;
  std::uint64_t argmaxFlips = 0;
  std::uint64_t tokenExactDiff = 0;
  std::uint64_t contextTokenExact = 0;
  std::uint64_t projectionTokenExact = 0;
  std::uint64_t totalRows = 0;
  bool finite = true;
};

inline LogitParity compareProbeLogits(
    const rp::Probe& probeA, const rp::ZStats& statsA, const rp::LayerSet& setA,
    int repA, const rp::Probe& probeB, const rp::ZStats& statsB,
    const rp::LayerSet& setB, int repB,
    const std::vector<rp::ProbeRow>& rows, std::size_t rowOffset) {
  LogitParity parity;
  parity.totalRows = rows.size();
  const auto& fA = setA.features.at(static_cast<std::size_t>(repA));
  const auto& fB = setB.features.at(static_cast<std::size_t>(repB));
  const int dimA = setA.dim;
  const int dimB = setB.dim;
  std::vector<double> logA(rp::kClasses), logB(rp::kClasses);
  double sumAbs = 0.0, sumSq = 0.0;
  std::uint64_t totalClasses = 0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    rp::probeForward(probeA, statsA, fA.data() + (rowOffset + r) * dimA, logA.data());
    rp::probeForward(probeB, statsB, fB.data() + (rowOffset + r) * dimB, logB.data());
    std::uint32_t argA = 0, argB = 0;
    for (std::uint32_t c = 1; c < rp::kClasses; ++c) {
      if (logA[c] > logA[argA]) argA = c;
      if (logB[c] > logB[argB]) argB = c;
    }
    if (argA == rows[r].truth) ++parity.contextTokenExact;
    if (argB == rows[r].truth) ++parity.projectionTokenExact;
    if (argA != argB) ++parity.argmaxFlips;
    for (std::uint32_t c = 0; c < rp::kClasses; ++c) {
      if (!std::isfinite(logA[c]) || !std::isfinite(logB[c])) {
        parity.finite = false;
        continue;
      }
      const double d = std::abs(logA[c] - logB[c]);
      parity.maxAbsDiff = std::max(parity.maxAbsDiff, d);
      sumAbs += d;
      sumSq += d * d;
      ++totalClasses;
    }
  }
  parity.tokenExactDiff =
      parity.contextTokenExact > parity.projectionTokenExact
          ? parity.contextTokenExact - parity.projectionTokenExact
          : parity.projectionTokenExact - parity.contextTokenExact;
  parity.meanAbsDiff = totalClasses > 0 ? sumAbs / static_cast<double>(totalClasses) : 0.0;
  parity.rmsDiff = totalClasses > 0
                       ? std::sqrt(sumSq / static_cast<double>(totalClasses))
                       : 0.0;
  return parity;
}

// ---------------------------------------------------------------------------
// Float/double comparison: build a transport probe using float arithmetic.
// ---------------------------------------------------------------------------
inline rp::Probe transportProbeFloat(const rp::Probe& ctxProbe,
                                     const rp::ZStats& ctxStats,
                                     const rp::ZStats& projStats,
                                     const Mat<double>& wDouble) {
  const int dim = ctxProbe.dim;
  const int classes = ctxProbe.classes;
  Mat<float> wf = zerosMat<float>(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      wf[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          static_cast<float>(wDouble[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
  const Svd<float> svdF = computeSvd(wf);
  const float sigmaMaxF = svdF.s.empty() ? 0.0f : svdF.s.front();
  const float tol = static_cast<float>(dim) * static_cast<float>(kFloatEps) * sigmaMaxF;
  const Mat<float> wpinvF = pseudoInverse(wf, tol);
  const Mat<float> wpinvFT = transposeMat(wpinvF);

  // C in double
  Mat<double> c = zerosMat<double>(classes, dim);
  for (int cl = 0; cl < classes; ++cl)
    for (int d = 0; d < dim; ++d) {
      const double std = ctxStats.std[static_cast<std::size_t>(d)];
      c[static_cast<std::size_t>(cl)][static_cast<std::size_t>(d)] =
          std > rp::kProbeStdFloor
              ? ctxProbe.w[static_cast<std::size_t>(cl) * dim + d] / std
              : 0.0;
    }
  // B_raw = C * (W^+)^T in double, but W^+ came from float arithmetic.
  Mat<double> bRaw = zerosMat<double>(classes, dim);
  for (int cl = 0; cl < classes; ++cl)
    for (int j = 0; j < dim; ++j) {
      double acc = 0.0;
      for (int i = 0; i < dim; ++i)
        acc += c[static_cast<std::size_t>(cl)][static_cast<std::size_t>(i)] *
               static_cast<double>(wpinvFT[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
      bRaw[static_cast<std::size_t>(cl)][static_cast<std::size_t>(j)] = acc;
    }

  // b_raw
  std::vector<double> bRawBias(classes, 0.0);
  for (int cl = 0; cl < classes; ++cl) {
    double bias = ctxProbe.b[static_cast<std::size_t>(cl)];
    for (int d = 0; d < dim; ++d)
      bias -= ctxStats.mean[static_cast<std::size_t>(d)] *
              c[static_cast<std::size_t>(cl)][static_cast<std::size_t>(d)];
    bRawBias[static_cast<std::size_t>(cl)] = bias;
  }

  rp::Probe projProbe = rp::zeroProbe(dim);
  projProbe.classes = classes;
  for (int cl = 0; cl < classes; ++cl)
    for (int d = 0; d < dim; ++d) {
      const double std = projStats.std[static_cast<std::size_t>(d)];
      projProbe.w[static_cast<std::size_t>(cl) * dim + d] =
          std > rp::kProbeStdFloor ? bRaw[static_cast<std::size_t>(cl)][static_cast<std::size_t>(d)] * std : 0.0;
    }
  for (int cl = 0; cl < classes; ++cl) {
    double bias = bRawBias[static_cast<std::size_t>(cl)];
    for (int d = 0; d < dim; ++d)
      bias += bRaw[static_cast<std::size_t>(cl)][static_cast<std::size_t>(d)] *
              projStats.mean[static_cast<std::size_t>(d)];
    projProbe.b[static_cast<std::size_t>(cl)] = bias;
  }
  return projProbe;
}

}  // namespace phonelm::output_projection_audit

#endif  // OUTPUT_PROJECTION_AUDIT_LIB_H
