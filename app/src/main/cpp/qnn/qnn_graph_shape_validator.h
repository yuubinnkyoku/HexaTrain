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

// SDK-independent contract for a complete Transformer stack.  The runtime
// builder supplies its declared intermediate shapes here before creating QNN
// tensors, so invalid layer/head layouts fail closed before graph creation.
struct TransformerTopologyConfig {
  std::size_t tokens = 0;
  std::size_t embeddingDim = 0;
  std::size_t feedForwardDim = 0;
  std::size_t numLayers = 0;
  std::size_t numHeads = 0;
  std::size_t vocabularySize = 0;
  std::size_t parameterElements = 0;
  std::size_t optimizerElements = 0;
};

struct TransformerLayerTopology {
  std::size_t layerIndex = 0;
  std::size_t parameterLayerIndex = 0;
  std::size_t gradientLayerIndex = 0;
  std::vector<std::size_t> input;
  std::vector<std::size_t> output;
  std::vector<std::size_t> query;
  std::vector<std::size_t> key;
  std::vector<std::size_t> value;
  std::vector<std::size_t> queryHeads;
  std::vector<std::size_t> keyHeads;
  std::vector<std::size_t> valueHeads;
  std::vector<std::size_t> scores;
  std::vector<std::size_t> probabilities;
  std::vector<std::size_t> context;
  std::vector<std::size_t> concat;
  std::vector<std::size_t> inputGradient;
  std::vector<std::size_t> norm1Gamma, norm1Beta, norm2Gamma, norm2Beta;
  std::vector<std::size_t> wq, wk, wv, wo, ffnW1, ffnW2;
  std::vector<std::size_t> dNorm1Gamma, dNorm1Beta, dNorm2Gamma, dNorm2Beta;
  std::vector<std::size_t> dWq, dWk, dWv, dWo, dFfnW1, dFfnW2;
  std::vector<std::size_t> norm1Mean, norm1Variance, norm2Mean, norm2Variance;
  std::vector<std::size_t> residualAfterAttention, residualAfterFfn;
  std::size_t parameterElements = 0;
  std::size_t optimizerElements = 0;
};

Result validate(const Node& node);
Result validateTransformerTopology(
    const TransformerTopologyConfig& config,
    const std::vector<TransformerLayerTopology>& layers);
std::string formatShape(const std::vector<std::size_t>& shape);

}  // namespace phonelm::qnn::shape
