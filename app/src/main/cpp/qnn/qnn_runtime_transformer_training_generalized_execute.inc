// Execution companion for the generic N-layer/H-head builder. The registries established by
// the builder are the binding ABI; do not substitute tensor-name lookup here.
bool Runtime::executeTinyTransformerTrainingGeneralized(
    const std::vector<float> &input, const std::vector<float> &target,
    const TinyTransformerParameters &current, float learningRate,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using GG = Impl::GeneralizedTinyTransformerTrainingGraph;
  auto &g = impl_->generalizedTinyTransformerTraining;
  if (g.forwardOnly) {
    // The FORWARD_ONLY graph has no target input and no gradient outputs;
    // routing it through the FULL execute would fail closed below anyway.
    error = "generalized tiny training forward-only graph requires "
            "executeTinyTransformerForwardOnly";
    return false;
  }
  const auto checkedMultiply = [](size_t a, size_t b, size_t *out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
  };
  const auto checkedAdd = [](size_t a, size_t b, size_t *out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
  };
  size_t layerParameters = 0, expectedParameterCount = 0;
  size_t expectedAppWriteCount = 0, expectedAppReadCount = 0;
  size_t headTapCount = 0;
  if (!checkedMultiply(10, g.numLayers, &layerParameters) ||
      !checkedAdd(2, layerParameters, &expectedParameterCount) ||
      !checkedAdd(4, layerParameters, &expectedAppWriteCount) ||
      (g.numHeads > 1 &&
       !checkedMultiply(g.numLayers, g.numHeads, &headTapCount)) ||
      !checkedAdd(5, expectedParameterCount, &expectedAppReadCount) ||
      !checkedAdd(expectedAppReadCount, g.numLayers, &expectedAppReadCount) ||
      !checkedAdd(expectedAppReadCount, headTapCount, &expectedAppReadCount) ||
      !checkedAdd(expectedAppReadCount, g.tapRegistry.size(),
                  &expectedAppReadCount)) {
    error = "APP_RESOURCE_ESTIMATOR: execution registry count overflow";
    return false;
  }
  if (g.state != GG::State::ACTIVE || !g.active || !g.graph ||
      !g.languageModel || g.numLayers == 0 || g.numHeads == 0 ||
      g.dimension % g.numHeads != 0 || !g.resourceEstimate.ok ||
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
                          {&layer->gamma1, &layer->beta1,
                           &layer->wq, &layer->wk, &layer->wv, &layer->wo,
                           &layer->gamma2,
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
    diagnostics_ += "generalized_tiny_training_input_immutable=false\n";
    error = "generalized tiny training APP_WRITE mutation";
    return false;
  }
  if (status != QNN_SUCCESS) {
    diagnostics_ += "generalized_tiny_training_qnn_return=" +
                    std::to_string(QNN_GET_ERROR_CODE(status)) + '\n';
    error = "generalized tiny training graphExecute=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  // Materialize every application-visible result before finite validation.
  // Diagnostic replay must retain the values produced by a successful QNN
  // call even when one of those values is non-finite.
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
    layer->gamma1 = appReadBuffers[gradient++];
    layer->beta1 = appReadBuffers[gradient++];
    layer->wq = appReadBuffers[gradient++];
    layer->wk = appReadBuffers[gradient++];
    layer->wv = appReadBuffers[gradient++];
    layer->wo = appReadBuffers[gradient++];
    layer->gamma2 = appReadBuffers[gradient++];
    layer->beta2 = appReadBuffers[gradient++];
    layer->w1 = appReadBuffers[gradient++];
    layer->w2 = appReadBuffers[gradient++];
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
  for (uint32_t id : g.tapRegistry)
    outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
  if (gradient != appReadBuffers.size()) {
    error = "generalized tiny training APP_READ result mapping";
    return false;
  }
  outputs.dEmbeddedInput = outputs.layerInputGradients.front();
  outputs.embeddedInput.clear();  // Not an APP_READ terminal in the L2/H1 ABI.
  size_t poisonResidual = 0, nonFiniteResidual = 0;
  size_t firstBadRegistryIndex = appReadBuffers.size();
  size_t firstBadElement = 0;
  size_t firstBadNanCount = 0, firstBadInfCount = 0;
  for (size_t registryIndex = 0; registryIndex < appReadBuffers.size();
       ++registryIndex) {
    const auto &values = appReadBuffers[registryIndex];
    poisonResidual += std::count(values.begin(), values.end(), poison);
    const auto firstBad = std::find_if(
        values.begin(), values.end(),
        [](float value) { return !std::isfinite(value); });
    const size_t badCount =
        std::count_if(values.begin(), values.end(),
                      [](float value) { return !std::isfinite(value); });
    nonFiniteResidual += badCount;
    if (badCount && firstBadRegistryIndex == appReadBuffers.size()) {
      firstBadRegistryIndex = registryIndex;
      firstBadElement = size_t(firstBad - values.begin());
      firstBadNanCount =
          std::count_if(values.begin(), values.end(),
                        [](float value) { return std::isnan(value); });
      firstBadInfCount = badCount - firstBadNanCount;
    }
  }
  if (poisonResidual != 0 || nonFiniteResidual != 0) {
    diagnostics_ += std::string("generalized_tiny_training_qnn_return=0\n") +
                    "generalized_tiny_training_output_poison_count=" +
                    std::to_string(poisonResidual) + '\n' +
                    "generalized_tiny_training_output_nonfinite_count=" +
                    std::to_string(nonFiniteResidual) + '\n' +
                    "generalized_tiny_training_outputs_all_written=" +
                    (poisonResidual == 0 ? "true\n" : "false\n");
    error = "generalized tiny training APP_READ residual poison=" +
            std::to_string(poisonResidual) + ", nonfinite=" +
            std::to_string(nonFiniteResidual);
    if (firstBadRegistryIndex < appReadBuffers.size()) {
      const uint32_t id = g.appReadRegistry[firstBadRegistryIndex];
      error += ", first_bad_tensor=" + g.names[id] +
               ", first_bad_registry_index=" +
               std::to_string(firstBadRegistryIndex) +
               ", first_bad_element=" + std::to_string(firstBadElement) +
               ", first_bad_nan_count=" +
               std::to_string(firstBadNanCount) +
               ", first_bad_inf_count=" +
               std::to_string(firstBadInfCount);
    }
    return false;
  }
  if (!g.executeDiagnosticsEmitted) {
    // Emit the successful execution schema once per Runtime instance.  The
    // failure paths above intentionally retain per-call details so a bad QNN
    // return or non-finite tensor is never hidden by spam suppression.
    diagnostics_ += "generalized_tiny_training_execute_app_write_count=" +
                    std::to_string(in.size()) + '\n' +
                    "generalized_tiny_training_execute_app_read_count=" +
                    std::to_string(out.size()) + '\n' +
                    "generalized_tiny_training_input_immutable=true\n"
                    "generalized_tiny_training_qnn_return=0\n"
                    "generalized_tiny_training_output_poison_count=0\n"
                    "generalized_tiny_training_output_nonfinite_count=0\n"
                    "generalized_tiny_training_outputs_all_written=true\n";
    g.executeDiagnosticsEmitted = true;
  }
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

// Forward-only generation execute for the FORWARD_ONLY generalized graph.
// The graph has no target APP_WRITE and no gradient/loss APP_READ tensors:
// inputs are the token one-hot plus every parameter, and the only
// application-visible output is logits [T,V].  Everything else (poison
// sentinel, QNN return-code check, finite/poison validation, immutability
// check) mirrors executeTinyTransformerTrainingGeneralized fail-closed.
bool Runtime::executeTinyTransformerForwardOnly(
    const std::vector<float> &input, const TinyTransformerParameters &current,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using GG = Impl::GeneralizedTinyTransformerTrainingGraph;
  auto &g = impl_->generalizedTinyTransformerTraining;
  if (!g.forwardOnly || !g.gradientRegistry.empty()) {
    error = "generalized tiny training forward-only state is invalid";
    return false;
  }
  const auto checkedMultiply = [](size_t a, size_t b, size_t *out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
  };
  const auto checkedAdd = [](size_t a, size_t b, size_t *out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
  };
  size_t layerParameters = 0, expectedParameterCount = 0;
  size_t expectedAppWriteCount = 0, expectedAppReadCount = 0, headTapCount = 0;
  if (!checkedMultiply(10, g.numLayers, &layerParameters) ||
      !checkedAdd(2, layerParameters, &expectedParameterCount) ||
      // one-hot + embedding + parameters + output projection (no target).
      !checkedAdd(3, layerParameters, &expectedAppWriteCount) ||
      (g.numHeads > 1 &&
       !checkedMultiply(g.numLayers, g.numHeads, &headTapCount)) ||
      // logits + per-head probability taps + scoped taps.
      !checkedAdd(1, headTapCount, &expectedAppReadCount)) {
    error = "APP_RESOURCE_ESTIMATOR: forward-only registry count overflow";
    return false;
  }
  if (!checkedAdd(expectedAppReadCount, g.tapRegistry.size(),
                  &expectedAppReadCount)) {
    error = "APP_RESOURCE_ESTIMATOR: forward-only registry count overflow";
    return false;
  }
  if (g.state != GG::State::ACTIVE || !g.active || !g.graph ||
      !g.languageModel || g.numLayers == 0 || g.numHeads == 0 ||
      g.dimension % g.numHeads != 0 || !g.resourceEstimate.ok ||
      g.parameterRegistry.size() != expectedParameterCount ||
      g.appWriteRegistry.size() != expectedAppWriteCount ||
      g.appReadRegistry.size() != expectedAppReadCount) {
    error = "generalized tiny training forward-only execution state is invalid";
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
  if (input.size() != tokenVocabulary ||
      current.layers.size() != g.numLayers - 1 ||
      current.tokenEmbedding.size() != embedding ||
      current.outputProjection.size() != projection || !finite(input) ||
      !finite(current.tokenEmbedding) ||
      !finite(current.outputProjection) || !validLayer(current) ||
      !std::all_of(current.layers.begin(), current.layers.end(), validLayer)) {
    error = "generalized tiny training forward-only buffer schema";
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
  appWriteValues.push_back(&input);
  appWriteValues.push_back(&current.tokenEmbedding);
  std::vector<const TinyTransformerLayerParameters *> parameterLayers;
  parameterLayers.reserve(g.numLayers);
  parameterLayers.push_back(&current);
  for (const auto &layer : current.layers) parameterLayers.push_back(&layer);
  for (const TinyTransformerLayerParameters *layer : parameterLayers) {
    appWriteValues.insert(appWriteValues.end(),
                          {&layer->gamma1, &layer->beta1,
                           &layer->wq, &layer->wk, &layer->wv, &layer->wo,
                           &layer->gamma2,
                           &layer->beta2, &layer->w1, &layer->w2});
  }
  appWriteValues.push_back(&current.outputProjection);
  if (appWriteValues.size() != g.appWriteRegistry.size()) {
    error = "generalized tiny training forward-only APP_WRITE registry count";
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
      error = "generalized tiny training forward-only APP_WRITE registry schema";
      return false;
    }
    appWriteSnapshots.push_back(*appWriteValues[index]);
    in.push_back(g.tensors[id]);
    in.back().v1.clientBuf = {const_cast<float *>(appWriteValues[index]->data()),
                              uint32_t(count * sizeof(float))};
  }
  constexpr float poison = 1.1415926f;
  std::vector<std::vector<float>> appReadBuffers(g.appReadRegistry.size());
  std::vector<Qnn_Tensor_t> out;
  out.reserve(g.appReadRegistry.size());
  for (size_t index = 0; index < g.appReadRegistry.size(); ++index) {
    const uint32_t id = g.appReadRegistry[index];
    size_t count = 0;
    if (id >= g.tensors.size() || g.tensors[id].v1.type != QNN_TENSOR_TYPE_APP_READ ||
        !elements(id, count)) {
      error = "generalized tiny training forward-only APP_READ registry schema";
      return false;
    }
    appReadBuffers[index].assign(count, poison);
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {appReadBuffers[index].data(),
                               uint32_t(count * sizeof(float))};
  }
  g.lastInputTensorCount = uint32_t(in.size());
  g.lastOutputTensorCount = uint32_t(out.size());
  // Bind manifest (metadata only, never tensor content): emitted once per
  // Runtime before the first execute so a QNN graphExecute failure can be
  // attributed to an exact binding mismatch.  Every entry records registry
  // position, identity, shape, and client-buffer validity; duplicates and
  // stale/foreign tensors are counted explicitly.
  if (!g.bindManifestEmitted) {
    std::ostringstream manifest;
    manifest << "bind_manifest_version=1\n"
             << "bind_manifest_forward_only=" << (g.forwardOnly ? "true" : "false")
             << "\nbind_manifest_input_count=" << in.size()
             << "\nbind_manifest_output_count=" << out.size() << '\n';
    const auto emitEntry = [&](const char *direction, size_t index,
                               const Qnn_Tensor_t &bound) {
      const uint32_t id = index < in.size() && direction[0] == 'I'
          ? g.appWriteRegistry[index]
          : g.appReadRegistry[index];
      const bool idValid = id < g.tensors.size();
      size_t elementCount = 0;
      const bool elementsOk = idValid && elements(id, elementCount);
      uint32_t rank = bound.v1.rank;
      manifest << "bind_" << direction << "_" << index << "_registry_id="
               << (idValid ? id : 0xffffffffu)
               << "\nbind_" << direction << "_" << index << "_name="
               << (idValid ? g.names[id] : std::string("<stale>"))
               << "\nbind_" << direction << "_" << index << "_type="
               << static_cast<int>(bound.v1.type)
               << "\nbind_" << direction << "_" << index << "_datatype="
               << static_cast<int>(bound.v1.dataType)
               << "\nbind_" << direction << "_" << index << "_rank="
               << rank
               << "\nbind_" << direction << "_" << index << "_dims=";
      for (uint32_t d = 0; d < rank && bound.v1.dimensions != nullptr; ++d)
        manifest << (d ? "," : "") << bound.v1.dimensions[d];
      manifest << "\nbind_" << direction << "_" << index
               << "_elements_ok=" << (elementsOk ? "true" : "false")
               << "\nbind_" << direction << "_" << index << "_element_count="
               << elementCount
               << "\nbind_" << direction << "_" << index
               << "_expected_bytes="
               << (elementsOk ? uint64_t(elementCount) * sizeof(float) : 0)
               << "\nbind_" << direction << "_" << index << "_buf_nonnull="
               << (bound.v1.clientBuf.data != nullptr ? "true" : "false")
               << "\nbind_" << direction << "_" << index << "_buf_bytes="
               << bound.v1.clientBuf.dataSize
               << "\nbind_" << direction << "_" << index << "_registered_copy="
               << (idValid &&
                           std::memcmp(&g.tensors[id].v1, &bound.v1,
                                       sizeof(Qnn_TensorV1_t)) == 0
                       ? "true"
                       : "false")
               << '\n';
    };
    std::unordered_set<uint32_t> seenInput, seenOutput;
    size_t duplicateInputs = 0, duplicateOutputs = 0;
    for (size_t i = 0; i < in.size(); ++i) {
      if (!seenInput.insert(g.appWriteRegistry[i]).second) ++duplicateInputs;
      emitEntry("in", i, in[i]);
    }
    for (size_t i = 0; i < out.size(); ++i) {
      if (!seenOutput.insert(g.appReadRegistry[i]).second) ++duplicateOutputs;
      emitEntry("out", i, out[i]);
    }
    manifest << "bind_duplicate_input_ids=" << duplicateInputs
             << "\nbind_duplicate_output_ids=" << duplicateOutputs << '\n';
    diagnostics_ += manifest.str();
    g.bindManifestEmitted = true;
  }
  const auto started = Clock::now();
  const auto status = impl_->api.graphExecute(g.graph, in.data(), uint32_t(in.size()),
                                               out.data(), uint32_t(out.size()),
                                               nullptr, nullptr);
  metrics_.executeUs.push_back(elapsedUs(started));
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status), QNN_GET_ERROR_CODE(status),
                           status == QNN_SUCCESS);
  const bool inputsUnchanged = std::equal(
      appWriteSnapshots.begin(), appWriteSnapshots.end(), appWriteValues.begin(),
      [](const std::vector<float> &before, const std::vector<float> *after) {
        return before == *after;
      });
  if (!inputsUnchanged) {
    diagnostics_ += "generalized_tiny_training_input_immutable=false\n";
    error = "generalized tiny training forward-only APP_WRITE mutation";
    return false;
  }
  if (status != QNN_SUCCESS) {
    diagnostics_ += "generalized_tiny_training_qnn_return=" +
                    std::to_string(QNN_GET_ERROR_CODE(status)) + '\n';
    error = "generalized tiny training forward-only graphExecute=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  outputs.taps.clear();
  outputs.next = {};
  outputs.logits = appReadBuffers[0];
  size_t gradient = 1;
  if (g.numHeads > 1) {
    for (uint32_t layer = 0; layer < g.numLayers; ++layer) {
      for (uint32_t head = 0; head < g.numHeads; ++head) {
        const uint32_t id = g.layers[layer].heads[head].probabilities;
        outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
      }
    }
  }
  for (uint32_t id : g.tapRegistry)
    outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
  if (gradient != appReadBuffers.size()) {
    error = "generalized tiny training forward-only APP_READ result mapping";
    return false;
  }
  outputs.output.clear();
  outputs.probabilities.clear();
  outputs.dLogits.clear();
  outputs.dOutput.clear();
  outputs.dEmbeddedInput.clear();
  outputs.embeddedInput.clear();
  outputs.layerInputGradients.clear();
  outputs.gradients = {};
  outputs.next = {};
  size_t poisonResidual = 0, nonFiniteResidual = 0;
  size_t firstBadRegistryIndex = appReadBuffers.size();
  size_t firstBadElement = 0;
  size_t firstBadNanCount = 0, firstBadInfCount = 0;
  for (size_t registryIndex = 0; registryIndex < appReadBuffers.size();
       ++registryIndex) {
    const auto &values = appReadBuffers[registryIndex];
    poisonResidual += std::count(values.begin(), values.end(), poison);
    const auto firstBad = std::find_if(
        values.begin(), values.end(),
        [](float value) { return !std::isfinite(value); });
    const size_t badCount =
        std::count_if(values.begin(), values.end(),
                      [](float value) { return !std::isfinite(value); });
    nonFiniteResidual += badCount;
    if (badCount && firstBadRegistryIndex == appReadBuffers.size()) {
      firstBadRegistryIndex = registryIndex;
      firstBadElement = size_t(firstBad - values.begin());
      firstBadNanCount =
          std::count_if(values.begin(), values.end(),
                        [](float value) { return std::isnan(value); });
      firstBadInfCount = badCount - firstBadNanCount;
    }
  }
  if (poisonResidual != 0 || nonFiniteResidual != 0) {
    diagnostics_ += std::string("generalized_tiny_training_qnn_return=0\n") +
                    "generalized_tiny_training_output_poison_count=" +
                    std::to_string(poisonResidual) + '\n' +
                    "generalized_tiny_training_output_nonfinite_count=" +
                    std::to_string(nonFiniteResidual) + '\n' +
                    "generalized_tiny_training_outputs_all_written=" +
                    (poisonResidual == 0 ? "true\n" : "false\n");
    error = "generalized tiny training forward-only APP_READ residual poison=" +
            std::to_string(poisonResidual) + ", nonfinite=" +
            std::to_string(nonFiniteResidual);
    if (firstBadRegistryIndex < appReadBuffers.size()) {
      const uint32_t id = g.appReadRegistry[firstBadRegistryIndex];
      error += ", first_bad_tensor=" + g.names[id] +
               ", first_bad_registry_index=" +
               std::to_string(firstBadRegistryIndex) +
               ", first_bad_element=" + std::to_string(firstBadElement) +
               ", first_bad_nan_count=" +
               std::to_string(firstBadNanCount) +
               ", first_bad_inf_count=" +
               std::to_string(firstBadInfCount);
    }
    return false;
  }
  if (!g.executeDiagnosticsEmitted) {
    diagnostics_ += "generalized_tiny_training_execute_app_write_count=" +
                    std::to_string(in.size()) + '\n' +
                    "generalized_tiny_training_execute_app_read_count=" +
                    std::to_string(out.size()) + '\n' +
                    "generalized_tiny_training_input_immutable=true\n"
                    "generalized_tiny_training_qnn_return=0\n"
                    "generalized_tiny_training_output_poison_count=0\n"
                    "generalized_tiny_training_output_nonfinite_count=0\n"
                    "generalized_tiny_training_outputs_all_written=true\n";
    g.executeDiagnosticsEmitted = true;
  }
  // No host-side loss: the FORWARD_ONLY graph has no target to score against.
  outputs.loss = 0.0f;
  return true;
}
