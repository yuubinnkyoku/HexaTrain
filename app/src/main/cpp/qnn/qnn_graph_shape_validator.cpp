#include "qnn_graph_shape_validator.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

namespace phonelm::qnn::shape {
namespace {
bool count(const std::vector<std::size_t>& shape, std::size_t* out) {
  if (shape.empty()) return false;
  std::size_t value = 1;
  for (const auto dim : shape) {
    if (dim == 0 || value > std::numeric_limits<std::size_t>::max() / dim) return false;
    value *= dim;
  }
  *out = value;
  return true;
}
bool addChecked(std::size_t a, std::size_t b, std::size_t* out) {
  if (a > std::numeric_limits<std::size_t>::max() - b) return false;
  *out = a + b;
  return true;
}
bool multiplyChecked(std::size_t a, std::size_t b, std::size_t* out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  *out = a * b;
  return true;
}
bool sameDtype(const Node& node) {
  return std::all_of(node.inputs.begin(), node.inputs.end(), [&](const Tensor& t) {
    return t.dtype == node.outputs.front().dtype;
  });
}
bool axesValid(const std::vector<std::size_t>& axes, std::size_t rank) {
  std::vector<std::size_t> sorted = axes;
  std::sort(sorted.begin(), sorted.end());
  return !sorted.empty() && sorted.back() < rank &&
         std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}
bool broadcast(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b,
               std::vector<std::size_t>* result) {
  const auto rank = std::max(a.size(), b.size()); result->assign(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    const auto ad = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const auto bd = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (ad != bd && ad != 1 && bd != 1) return false;
    (*result)[i] = std::max(ad, bd);
  }
  return true;
}
Result fail(const Node& node, const std::string& detail,
            const std::vector<std::vector<std::size_t>>& expected = {}) {
  std::ostringstream text;
  text << "shape validator: node=" << node.name << " op=" << static_cast<int>(node.op)
       << ' ' << detail;
  if (!expected.empty()) text << " expected=" << formatShape(expected.front());
  if (!node.outputs.empty()) text << " declared=" << formatShape(node.outputs.front().shape);
  if (!node.inputs.empty()) text << " input=" << formatShape(node.inputs.front().shape);
  if (!node.axes.empty()) text << " axes=" << formatShape(node.axes) << " keep_dims=" << (node.keepDims ? "true" : "false");
  return {false, text.str(), expected};
}
}

std::string formatShape(const std::vector<std::size_t>& shape) {
  std::ostringstream text; text << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) { if (i) text << ','; text << shape[i]; }
  return text << ']', text.str();
}

Result validate(const Node& node) {
  if (node.name.empty() || node.inputs.empty() || node.outputs.empty()) return fail(node, "missing node name, input, or output");
  for (const auto& t : node.inputs) { std::size_t ignored; if (!count(t.shape, &ignored)) return fail(node, "invalid input element count"); }
  for (const auto& t : node.outputs) { std::size_t ignored; if (!count(t.shape, &ignored)) return fail(node, "invalid declared output element count"); }
  if (!sameDtype(node)) return fail(node, "dtype mismatch");
  std::vector<std::vector<std::size_t>> expected;
  const auto& input = node.inputs.front().shape;
  switch (node.op) {
    case Op::ReduceSum: case Op::ReduceMean: {
      if (node.inputs.size() != 1 || node.outputs.size() != 1 || !axesValid(node.axes, input.size())) return fail(node, "invalid reduction inputs or axes");
      auto out = input;
      if (node.keepDims) for (auto axis : node.axes) out[axis] = 1;
      else {
        auto axes = node.axes;
        std::sort(axes.rbegin(), axes.rend());
        for (auto axis : axes) out.erase(out.begin() + static_cast<std::ptrdiff_t>(axis));
      }
      expected.push_back(std::move(out)); break;
    }
    case Op::MatMul: {
      if (node.inputs.size() != 2 || node.outputs.size() != 1 || input.size() < 2 || node.inputs[1].shape.size() < 2) return fail(node, "MatMul requires two rank >= 2 inputs");
      auto a = input, b = node.inputs[1].shape; if (node.transposeInput0) std::swap(a[a.size()-1], a[a.size()-2]); if (node.transposeInput1) std::swap(b[b.size()-1], b[b.size()-2]);
      if (a.back() != b[b.size()-2]) return fail(node, "MatMul inner dimensions incompatible");
      const std::vector<std::size_t> aBatch(a.begin(), a.end() - 2);
      const std::vector<std::size_t> bBatch(b.begin(), b.end() - 2);
      std::vector<std::size_t> batch; if (!broadcast(aBatch, bBatch, &batch)) return fail(node, "MatMul batch dimensions incompatible");
      batch.push_back(a[a.size()-2]); batch.push_back(b.back()); expected.push_back(std::move(batch)); break;
    }
    case Op::Transpose: {
      if (node.inputs.size() != 1 || node.outputs.size() != 1 || node.permutation.size() != input.size()) return fail(node, "invalid transpose permutation");
      auto permutation = node.permutation; std::sort(permutation.begin(), permutation.end()); for (std::size_t i=0;i<permutation.size();++i) if (permutation[i] != i) return fail(node, "invalid transpose permutation");
      std::vector<std::size_t> out; for (auto axis : node.permutation) out.push_back(input[axis]); expected.push_back(std::move(out)); break;
    }
    case Op::ElementWiseBinary: {
      if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return fail(node, "binary op requires two inputs");
      }
      std::vector<std::size_t> out;
      if (!broadcast(input, node.inputs[1].shape, &out)) {
        return fail(node, "broadcast incompatible");
      }
      expected.push_back(std::move(out));
      break;
    }
    case Op::Softmax: if (node.inputs.size()!=1 || node.outputs.size()!=1 || node.softmaxAxis >= input.size()) return fail(node, "invalid Softmax axis"); expected.push_back(input); break;
    case Op::Reshape: { if (node.inputs.size()!=1 || node.outputs.size()!=1) return fail(node, "Reshape requires one input and output"); std::size_t inCount, outCount; if (!count(input,&inCount) || !count(node.reshape,&outCount) || inCount != outCount) return fail(node, "Reshape element count mismatch"); expected.push_back(node.reshape); break; }
    case Op::Concat: { if (node.inputs.size()<2 || node.outputs.size()!=1 || node.concatAxis >= input.size()) return fail(node, "invalid Concat inputs or axis"); auto out=input; out[node.concatAxis]=0; for(const auto& t:node.inputs){if(t.shape.size()!=input.size())return fail(node,"Concat rank mismatch"); for(std::size_t i=0;i<input.size();++i)if(i!=node.concatAxis&&t.shape[i]!=input[i])return fail(node,"Concat non-axis mismatch"); out[node.concatAxis]+=t.shape[node.concatAxis];} expected.push_back(std::move(out)); break; }
    case Op::Split: { if (node.inputs.size()!=1 || node.outputs.size()!=node.splitSizes.size() || node.concatAxis>=input.size()) return fail(node,"invalid Split inputs, outputs, or axis"); std::size_t total=0; for(auto size:node.splitSizes) total+=size; if(total!=input[node.concatAxis]) return fail(node,"Split sizes do not cover axis"); for(auto size:node.splitSizes){auto out=input;out[node.concatAxis]=size;expected.push_back(std::move(out));} break; }
    case Op::LayerNorm: { if ((node.inputs.size()!=1 && node.inputs.size()!=3) || node.outputs.size()!=1 || !axesValid(node.axes,input.size())) return fail(node,"invalid LayerNorm inputs or axes"); if(node.inputs.size()==3 && (node.inputs[1].shape.size()!=node.axes.size() || node.inputs[2].shape!=node.inputs[1].shape)) return fail(node,"LayerNorm gamma/beta mismatch"); expected.push_back(input); break; }
  }
  if (expected.size() != node.outputs.size()) return fail(node, "output count mismatch", expected);
  for (std::size_t i=0;i<expected.size();++i) if (expected[i] != node.outputs[i].shape) return fail(node, "declared output shape mismatch", expected);
  return {true, {}, expected};
}

Result validateTransformerTopology(const TransformerTopologyConfig& config,
                                   const std::vector<TransformerLayerTopology>& layers) {
  Node contract;
  contract.name = "transformer_topology";
  contract.op = Op::Reshape;
  const auto topologyFail = [&](const std::string& detail) {
    return fail(contract, detail);
  };
  if (config.tokens == 0 || config.embeddingDim == 0 || config.feedForwardDim == 0 ||
      config.numLayers == 0 || config.numHeads == 0 || config.vocabularySize == 0) {
    return topologyFail("tokens, embeddingDim, feedForwardDim, numLayers, numHeads, and vocabularySize must be >= 1");
  }
  if (config.embeddingDim % config.numHeads != 0) {
    return topologyFail("embeddingDim must be divisible by numHeads");
  }
  std::size_t ignored = 0;
  if (!count({config.tokens, config.embeddingDim}, &ignored) ||
      !count({config.tokens, config.feedForwardDim}, &ignored) ||
      !count({config.numHeads, config.tokens, config.tokens}, &ignored)) {
    return topologyFail("derived element count overflows size_t");
  }
  std::size_t d2 = 0, dff = 0, fourD = 0, optimizerElements = 0, layerElements = 0, allLayerElements = 0;
  std::size_t globals = 0, expectedTotalParameters = 0;
  if (!multiplyChecked(config.embeddingDim, config.embeddingDim, &d2) ||
      !multiplyChecked(config.embeddingDim, config.feedForwardDim, &dff) ||
      !multiplyChecked(4, d2, &layerElements) || !multiplyChecked(4, config.embeddingDim, &fourD) ||
      !addChecked(layerElements, fourD, &layerElements) ||
      !multiplyChecked(2, dff, &dff) || !addChecked(layerElements, dff, &layerElements) ||
      !multiplyChecked(layerElements, 2, &optimizerElements) ||
      !multiplyChecked(layerElements, config.numLayers, &allLayerElements)) {
    return topologyFail("layer parameter element count overflows size_t");
  }
  if (!multiplyChecked(2, config.vocabularySize, &globals) ||
      !multiplyChecked(globals, config.embeddingDim, &globals) ||
      !addChecked(allLayerElements, globals, &expectedTotalParameters)) {
    return topologyFail("global parameter element count overflows size_t");
  }
  std::size_t expectedTotalOptimizer = 0;
  if (!multiplyChecked(expectedTotalParameters, 2, &expectedTotalOptimizer)) {
    return topologyFail("global optimizer element count overflows size_t");
  }
  if (config.parameterElements != expectedTotalParameters) {
    return topologyFail("total parameter element count does not match layer/global schema");
  }
  if (config.optimizerElements != expectedTotalOptimizer) {
    return topologyFail("total optimizer m/v element count does not match parameter schema");
  }
  if (layers.size() != config.numLayers) return topologyFail("layer count does not match numLayers");
  const std::size_t headDim = config.embeddingDim / config.numHeads;
  const std::vector<std::size_t> tokenDim{config.tokens, config.embeddingDim};
  const std::vector<std::size_t> rowDim{config.tokens, 1};
  const std::vector<std::size_t> dim{config.embeddingDim};
  const std::vector<std::size_t> square{config.embeddingDim, config.embeddingDim};
  const std::vector<std::size_t> w1{config.embeddingDim, config.feedForwardDim};
  const std::vector<std::size_t> w2{config.feedForwardDim, config.embeddingDim};
  const std::vector<std::size_t> headTensor{config.numHeads, config.tokens, headDim};
  const std::vector<std::size_t> attention{config.numHeads, config.tokens, config.tokens};
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& layer = layers[i];
    const auto failLayer = [&](const std::string& detail) {
      return topologyFail("layer=" + std::to_string(i) + " " + detail);
    };
    if (layer.layerIndex != i) return failLayer("layer index is not deterministic/contiguous");
    if (layer.parameterLayerIndex != i) return failLayer("parameter binding targets another layer");
    if (layer.gradientLayerIndex != i) return failLayer("gradient binding targets another layer");
    if (layer.input != tokenDim) return failLayer("input shape must be [T,D]");
    if (i > 0 && layer.input != layers[i - 1].output) return failLayer("input does not chain from previous layer output");
    if (layer.output != tokenDim || layer.inputGradient != tokenDim) return failLayer("output/input gradient shape must be [T,D]");
    if (layer.query != tokenDim || layer.key != tokenDim || layer.value != tokenDim) return failLayer("Q/K/V shape must be [T,D]");
    if (layer.queryHeads != headTensor || layer.keyHeads != headTensor || layer.valueHeads != headTensor) return failLayer("Q/K/V head reshape shape must be [H,T,Dh]");
    if (layer.scores != attention || layer.probabilities != attention) return failLayer("scores/probabilities shape must be [H,T,T]");
    if (layer.context != headTensor) return failLayer("context shape must be [H,T,Dh]");
    if (layer.concat != tokenDim) return failLayer("head concat shape must be [T,D]");
    if (layer.residualAfterAttention != tokenDim || layer.residualAfterFfn != tokenDim) return failLayer("residual add shape must be [T,D]");
    if (layer.norm1Mean != rowDim || layer.norm1Variance != rowDim ||
        layer.norm2Mean != rowDim || layer.norm2Variance != rowDim) return failLayer("LayerNorm reductions must be [T,1]");
    if (layer.norm1Gamma != dim || layer.norm1Beta != dim || layer.norm2Gamma != dim || layer.norm2Beta != dim ||
        layer.wq != square || layer.wk != square || layer.wv != square || layer.wo != square ||
        layer.ffnW1 != w1 || layer.ffnW2 != w2) return failLayer("parameter shapes do not match Transformer layer schema");
    if (layer.dNorm1Gamma != dim || layer.dNorm1Beta != dim || layer.dNorm2Gamma != dim || layer.dNorm2Beta != dim ||
        layer.dWq != square || layer.dWk != square || layer.dWv != square || layer.dWo != square ||
        layer.dFfnW1 != w1 || layer.dFfnW2 != w2) return failLayer("backward parameter gradient shapes do not match parameter schema");
    if (layer.parameterElements != layerElements) return failLayer("parameter element count does not match layer schema");
    if (layer.optimizerElements != optimizerElements) return failLayer("optimizer m/v element count does not match parameter schema");
  }
  std::vector<std::size_t> totals{layerElements, expectedTotalParameters};
  return {true, {}, {tokenDim, totals}};
}
}  // namespace phonelm::qnn::shape
