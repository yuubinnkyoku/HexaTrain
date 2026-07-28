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
}  // namespace phonelm::qnn::shape
