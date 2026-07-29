// Execution companion for the generalized L1/L2 H1/H2 builder.  The registries established by
// the builder are the binding ABI; do not substitute tensor-name lookup here.
bool Runtime::executeTinyTransformerTrainingGeneralized(
    const std::vector<float> &input, const std::vector<float> &target,
    const TinyTransformerParameters &current, float learningRate,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using GG = Impl::GeneralizedTinyTransformerTrainingGraph;
  auto &g = impl_->generalizedTinyTransformerTraining;
  const size_t expectedParameterCount = 2 + size_t(10) * g.numLayers;
  const size_t expectedAppWriteCount = 4 + size_t(10) * g.numLayers;
  const size_t expectedAppReadCount = 5 + expectedParameterCount + g.numLayers +
      (g.numHeads > 1 ? size_t(g.numLayers) * g.numHeads : 0);
  if (g.state != GG::State::ACTIVE || !g.active || !g.graph ||
      !g.languageModel || g.numLayers == 0 || g.numLayers > 2 ||
      g.numHeads == 0 || g.numHeads > 2 || g.dimension % g.numHeads != 0 ||
      g.parameterRegistry.size() != expectedParameterCount ||
      g.gradientRegistry.size() != expectedParameterCount ||
      g.appWriteRegistry.size() != expectedAppWriteCount ||
      g.appReadRegistry.size() != expectedAppReadCount) {
    error = "generalized tiny training execution state is invalid";
    return false;
  }
  const size_t tokenVocabulary = size_t(g.tokens) * g.vocabularySize;
  const size_t dimensionSquared = size_t(g.dimension) * g.dimension;
  const size_t dimensionFf = size_t(g.dimension) * g.feedForwardDimension;
  const size_t ffDimension = size_t(g.feedForwardDimension) * g.dimension;
  const size_t embedding = size_t(g.vocabularySize) * g.dimension;
  const size_t projection = size_t(g.dimension) * g.vocabularySize;
  const auto finite = [](const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  const auto validLayer = [&](const TinyTransformerLayerParameters &layer) {
    return layer.wq.size() == dimensionSquared &&
           layer.wk.size() == dimensionSquared &&
           layer.wv.size() == dimensionSquared &&
           layer.wo.size() == dimensionSquared &&
           layer.gamma1.size() == g.dimension && layer.beta1.size() == g.dimension &&
           layer.gamma2.size() == g.dimension && layer.beta2.size() == g.dimension &&
           layer.w1.size() == dimensionFf && layer.w2.size() == ffDimension &&
           finite(layer.wq) && finite(layer.wk) && finite(layer.wv) &&
           finite(layer.wo) && finite(layer.gamma1) && finite(layer.beta1) &&
           finite(layer.gamma2) && finite(layer.beta2) && finite(layer.w1) &&
           finite(layer.w2);
  };
  if (input.size() != tokenVocabulary || target.size() != tokenVocabulary ||
      current.layers.size() != g.numLayers - 1 || current.tokenEmbedding.size() != embedding ||
      current.outputProjection.size() != projection || !finite(input) ||
      !finite(target) || !finite(current.tokenEmbedding) ||
      !finite(current.outputProjection) || !validLayer(current) ||
      !std::isfinite(learningRate) ||
      !std::all_of(current.layers.begin(), current.layers.end(), validLayer)) {
    error = "generalized tiny training buffer schema";
    return false;
  }
  // This graph stops at gradients.  Parameter updates are exclusively the
  // separately prepared QNN Adam graph, so accepting a nonzero SGD rate here
  // would silently claim an update that this graph never computes.
  if (learningRate != 0.0f) {
    error = "generalized tiny training requires learningRate=0; use QNN Adam";
    return false;
  }
  const auto elements = [&](uint32_t id, size_t &count) {
    if (id >= g.dimensions.size()) return false;
    count = 1;
    for (uint32_t extent : g.dimensions[id]) {
      if (!extent || count > std::numeric_limits<size_t>::max() / extent)
        return false;
      count *= extent;
    }
    return count <= std::numeric_limits<uint32_t>::max() / sizeof(float);
  };
  std::vector<const std::vector<float> *> appWriteValues;
  appWriteValues.reserve(g.appWriteRegistry.size());
  appWriteValues.insert(appWriteValues.end(), {&input, &target, &current.tokenEmbedding});
  std::vector<const TinyTransformerLayerParameters *> parameterLayers;
  parameterLayers.reserve(g.numLayers);
  parameterLayers.push_back(&current);
  for (const auto &layer : current.layers) parameterLayers.push_back(&layer);
  for (const TinyTransformerLayerParameters *layer : parameterLayers) {
    appWriteValues.insert(appWriteValues.end(),
                          {&layer->wq, &layer->wk, &layer->wv, &layer->wo,
                           &layer->gamma1, &layer->beta1, &layer->gamma2,
                           &layer->beta2, &layer->w1, &layer->w2});
  }
  appWriteValues.push_back(&current.outputProjection);
  if (appWriteValues.size() != g.appWriteRegistry.size()) {
    error = "generalized tiny training APP_WRITE registry count";
    return false;
  }
  std::vector<Qnn_Tensor_t> in;
  std::vector<std::vector<float>> appWriteSnapshots;
  in.reserve(g.appWriteRegistry.size());
  appWriteSnapshots.reserve(g.appWriteRegistry.size());
  for (size_t index = 0; index < g.appWriteRegistry.size(); ++index) {
    const uint32_t id = g.appWriteRegistry[index];
    size_t count = 0;
    if (id >= g.tensors.size() || g.tensors[id].v1.type != QNN_TENSOR_TYPE_APP_WRITE ||
        !elements(id, count) || appWriteValues[index]->size() != count) {
      error = "generalized tiny training APP_WRITE registry schema";
      return false;
    }
    appWriteSnapshots.push_back(*appWriteValues[index]);
    in.push_back(g.tensors[id]);
    in.back().v1.clientBuf = {const_cast<float *>(appWriteValues[index]->data()),
                              uint32_t(count * sizeof(float))};
  }
  // Keep client output memory finite: some HTP drivers reject a NaN-filled
  // APP_READ buffer before the graph gets a chance to overwrite it.  We still
  // check this sentinel and non-finite values independently after success.
  constexpr float poison = 1.1415926f;
  std::vector<std::vector<float>> appReadBuffers(g.appReadRegistry.size());
  std::vector<Qnn_Tensor_t> out;
  out.reserve(g.appReadRegistry.size());
  for (size_t index = 0; index < g.appReadRegistry.size(); ++index) {
    const uint32_t id = g.appReadRegistry[index];
    size_t count = 0;
    if (id >= g.tensors.size() || g.tensors[id].v1.type != QNN_TENSOR_TYPE_APP_READ ||
        !elements(id, count)) {
      error = "generalized tiny training APP_READ registry schema";
      return false;
    }
    appReadBuffers[index].assign(count, poison);
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {appReadBuffers[index].data(),
                               uint32_t(count * sizeof(float))};
  }
  g.lastInputTensorCount = uint32_t(in.size());
  g.lastOutputTensorCount = uint32_t(out.size());
  if (!g.executeDiagnosticsEmitted) {
    diagnostics_ += "generalized_tiny_training_execute_app_write_count=" +
                    std::to_string(in.size()) + '\n' +
                    "generalized_tiny_training_execute_app_read_count=" +
                    std::to_string(out.size()) + '\n';
    g.executeDiagnosticsEmitted = true;
  }
  std::uint32_t learningRateBitsBefore = 0;
  std::memcpy(&learningRateBitsBefore, &learningRate, sizeof(learningRateBitsBefore));
  const auto started = Clock::now();
  const auto status = impl_->api.graphExecute(g.graph, in.data(), uint32_t(in.size()),
                                               out.data(), uint32_t(out.size()),
                                               nullptr, nullptr);
  metrics_.executeUs.push_back(elapsedUs(started));
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status), QNN_GET_ERROR_CODE(status),
                           status == QNN_SUCCESS);
  std::uint32_t learningRateBitsAfter = 0;
  std::memcpy(&learningRateBitsAfter, &learningRate, sizeof(learningRateBitsAfter));
  g.lastLearningRateBytesUnchanged = learningRateBitsBefore == learningRateBitsAfter;
  const bool inputsUnchanged = std::equal(
      appWriteSnapshots.begin(), appWriteSnapshots.end(), appWriteValues.begin(),
      [](const std::vector<float> &before, const std::vector<float> *after) {
        return before == *after;
      });
  if (!inputsUnchanged) {
    error = "generalized tiny training APP_WRITE mutation";
    return false;
  }
  if (status != QNN_SUCCESS) {
    error = "generalized tiny training graphExecute=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  size_t poisonResidual = 0, nonFiniteResidual = 0;
  for (const auto &values : appReadBuffers) {
    poisonResidual += std::count(values.begin(), values.end(), poison);
    nonFiniteResidual += std::count_if(values.begin(), values.end(),
                                       [](float value) { return !std::isfinite(value); });
  }
  if (poisonResidual != 0 || nonFiniteResidual != 0) {
    error = "generalized tiny training APP_READ residual poison=" +
            std::to_string(poisonResidual) + ", nonfinite=" +
            std::to_string(nonFiniteResidual);
    return false;
  }
  // APP_READ registry: logits, final-layer diagnostic values, canonical
  // gradients, layer-input gradients, then H>1 attention-probability taps.
  // The graph has no SGD nodes; QNN Adam consumes gradients separately, so
  // next is empty.
  outputs.taps.clear();
  outputs.next = {};
  outputs.logits = appReadBuffers[0];
  outputs.output = appReadBuffers[1];
  outputs.probabilities = appReadBuffers[2];
  outputs.dLogits = appReadBuffers[3];
  outputs.dOutput = appReadBuffers[4];
  outputs.gradients = current;
  outputs.gradients.tokenEmbedding = appReadBuffers[5];
  size_t gradient = 6;
  std::vector<TinyTransformerLayerParameters *> gradientLayers;
  gradientLayers.reserve(g.numLayers);
  gradientLayers.push_back(&outputs.gradients);
  for (auto &layer : outputs.gradients.layers) gradientLayers.push_back(&layer);
  for (TinyTransformerLayerParameters *layer : gradientLayers) {
    layer->wq = appReadBuffers[gradient++]; layer->wk = appReadBuffers[gradient++];
    layer->wv = appReadBuffers[gradient++]; layer->wo = appReadBuffers[gradient++];
    layer->gamma1 = appReadBuffers[gradient++]; layer->beta1 = appReadBuffers[gradient++];
    layer->gamma2 = appReadBuffers[gradient++]; layer->beta2 = appReadBuffers[gradient++];
    layer->w1 = appReadBuffers[gradient++]; layer->w2 = appReadBuffers[gradient++];
  }
  outputs.gradients.outputProjection = appReadBuffers[gradient++];
  outputs.layerInputGradients.clear();
  outputs.layerInputGradients.reserve(g.numLayers);
  for (uint32_t layer = 0; layer < g.numLayers; ++layer)
    outputs.layerInputGradients.push_back(appReadBuffers[gradient++]);
  if (g.numHeads > 1) {
    for (uint32_t layer = 0; layer < g.numLayers; ++layer) {
      for (uint32_t head = 0; head < g.numHeads; ++head) {
        const uint32_t id = g.layers[layer].heads[head].probabilities;
        outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
      }
    }
  }
  if (gradient != appReadBuffers.size()) {
    error = "generalized tiny training APP_READ result mapping";
    return false;
  }
  outputs.dEmbeddedInput = outputs.layerInputGradients.front();
  outputs.embeddedInput.clear();  // Not an APP_READ terminal in the L2/H1 ABI.
  double loss = 0.0;
  for (uint32_t row = 0; row < g.tokens; ++row) {
    const float *logits = outputs.logits.data() + size_t(row) * g.vocabularySize;
    const float maximum = *std::max_element(logits, logits + g.vocabularySize);
    double sum = 0.0;
    uint32_t targetIndex = 0;
    for (uint32_t column = 0; column < g.vocabularySize; ++column) {
      sum += std::exp(double(logits[column] - maximum));
      if (target[size_t(row) * g.vocabularySize + column] > 0.5f) targetIndex = column;
    }
    loss += double(maximum) + std::log(sum) - logits[targetIndex];
  }
  outputs.loss = float(loss / g.tokens);
  return true;
}
