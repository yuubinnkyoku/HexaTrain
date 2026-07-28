#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace phonelm::qnn::shape {

enum class Op { ReduceSum, ReduceMean, MatMul, Transpose, ElementWiseBinary,
                Softmax, Reshape, Concat, Split, LayerNorm };

struct Tensor {
  std::string name;
  std::vector<std::size_t> shape;
  std::string dtype = "FLOAT_32";
};

struct Node {
  std::string name;
  Op op;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<std::size_t> axes;
  bool keepDims = false;
  bool transposeInput0 = false;
  bool transposeInput1 = false;
  std::vector<std::size_t> permutation;
  std::vector<std::size_t> reshape;
  std::size_t concatAxis = 0;
  std::vector<std::size_t> splitSizes;
  std::size_t softmaxAxis = 0;
};

struct Result {
  bool ok = false;
  std::string error;
  std::vector<std::vector<std::size_t>> inferredOutputs;
};

Result validate(const Node& node);
std::string formatShape(const std::vector<std::size_t>& shape);

}  // namespace phonelm::qnn::shape
