// Execution companion for the generic N-layer/H-head builder. The registries established by
// the builder are the binding ABI; do not substitute tensor-name lookup here.
bool Runtime::executeTinyTransformerTrainingGeneralized(
    const std::vector<float> &input, const std::vector<float> &target,
    const TinyTransformerParameters &current, float learningRate,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  const auto wrapperStarted = Clock::now();
  auto phaseStarted = wrapperStarted;
  outputs.appReadValidated = false;
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
  // Parameter identity, extents, and binding order come from the shared
  // metadata; the registries below only carry tensor ids.
  const tiny_lm::ParameterDimensions parameterDimensions{
      g.vocabularySize, g.dimension, g.feedForwardDimension, g.numLayers,
      g.numHeads, g.headwiseG1Gate};
  size_t expectedParameterCount = 0;
  size_t expectedAppWriteCount = 0, expectedAppReadCount = 0;
  size_t headTapCount = 0;
  if (!tiny_lm::checkedParameterInstanceCount(parameterDimensions,
                                              &expectedParameterCount) ||
      // one-hot and target are the only non-parameter APP_WRITE tensors.
      !checkedAdd(2, expectedParameterCount, &expectedAppWriteCount) ||
      (g.diagnosticOutputs && g.numHeads > 1 &&
       !checkedMultiply(g.numLayers, g.numHeads, &headTapCount)) ||
      // diagnosticOutputs=true keeps the established 5-terminal prefix
      // (logits, output, probabilities, dLogits, dOutput) plus per-layer
      // input gradients. Production FULL uses a 1-terminal prefix (logits).
      !checkedAdd(g.diagnosticOutputs ? 5u : 1u, expectedParameterCount,
                  &expectedAppReadCount) ||
      (g.diagnosticOutputs &&
       !checkedAdd(expectedAppReadCount, g.numLayers, &expectedAppReadCount)) ||
      !checkedAdd(expectedAppReadCount, headTapCount, &expectedAppReadCount) ||
      (g.headwiseG1Gate &&
       !checkedAdd(expectedAppReadCount, g.numLayers, &expectedAppReadCount)) ||
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
  const auto finite = [](const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (input.size() != tokenVocabulary || target.size() != tokenVocabulary ||
      !finite(input) || !finite(target) ||
      !tiny_lm::parameterStorageMatchesDefinitions(current,
                                                   parameterDimensions) ||
      (options_.validateAppWriteParameterFiniteness &&
       !tiny_lm::parameterStorageIsFinite(current, parameterDimensions)) ||
      !std::isfinite(learningRate)) {
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
  appWriteValues.insert(appWriteValues.end(), {&input, &target});
  tiny_lm::forEachParameterStorage(
      current, [&](const tiny_lm::ParameterDefinition &definition, size_t,
                   const std::vector<float> &values) {
        if (tiny_lm::parameterDefinitionEnabled(definition,
                                                parameterDimensions))
          appWriteValues.push_back(&values);
      });
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
  }
  const double appWriteSchemaValidationUs = elapsedUs(phaseStarted);
  phaseStarted = Clock::now();
  if (options_.validateAppWriteImmutability) {
    for (size_t index = 0; index < g.appWriteRegistry.size(); ++index)
      appWriteSnapshots.push_back(*appWriteValues[index]);
  }
  const double appWriteSnapshotCopyUs = elapsedUs(phaseStarted);
  phaseStarted = Clock::now();
  for (size_t index = 0; index < g.appWriteRegistry.size(); ++index) {
    const uint32_t id = g.appWriteRegistry[index];
    size_t count = 0;
    (void)elements(id, count);
    in.push_back(g.tensors[id]);
    in.back().v1.clientBuf = {const_cast<float *>(appWriteValues[index]->data()),
                              uint32_t(count * sizeof(float))};
  }
  const double appWriteBindUs = elapsedUs(phaseStarted);
  // Keep client output memory finite: some HTP drivers reject a NaN-filled
  // APP_READ buffer before the graph gets a chance to overwrite it.  We still
  // check this sentinel and non-finite values independently after success.
  constexpr float poison = 1.1415926f;
  phaseStarted = Clock::now();
  auto &appReadBuffers = g.appReadBuffers;
  appReadBuffers.resize(g.appReadRegistry.size());
  std::vector<size_t> appReadCounts(g.appReadRegistry.size());
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
    appReadCounts[index] = count;
    appReadBuffers[index].resize(count);
  }
  const double appReadBufferAllocateUs = elapsedUs(phaseStarted);
  phaseStarted = Clock::now();
  for (size_t index = 0; index < appReadBuffers.size(); ++index)
    std::fill(appReadBuffers[index].begin(), appReadBuffers[index].end(), poison);
  const double appReadPoisonFillUs = elapsedUs(phaseStarted);
  phaseStarted = Clock::now();
  for (size_t index = 0; index < g.appReadRegistry.size(); ++index) {
    const uint32_t id = g.appReadRegistry[index];
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {appReadBuffers[index].data(),
                               uint32_t(appReadCounts[index] * sizeof(float))};
  }
  const double appReadBindUs = elapsedUs(phaseStarted);
  metrics_.inputBindUs.push_back(appWriteBindUs);
  metrics_.outputBindUs.push_back(appReadBindUs);
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
  phaseStarted = Clock::now();
  const bool inputsUnchanged = !options_.validateAppWriteImmutability ||
      std::equal(
          appWriteSnapshots.begin(), appWriteSnapshots.end(),
          appWriteValues.begin(),
          [](const std::vector<float> &before,
             const std::vector<float> *after) { return before == *after; });
  const double appWriteImmutabilityCheckUs = elapsedUs(phaseStarted);
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
  phaseStarted = Clock::now();
  outputs.taps.clear();
  outputs.next = {};
  outputs.logits = appReadBuffers[0];
  size_t gradient = 1;
  if (g.diagnosticOutputs) {
    outputs.output = appReadBuffers[gradient++];
    outputs.probabilities = appReadBuffers[gradient++];
    outputs.dLogits = appReadBuffers[gradient++];
    outputs.dOutput = appReadBuffers[gradient++];
  } else {
    outputs.output.clear();
    outputs.probabilities.clear();
    outputs.dLogits.clear();
    outputs.dOutput.clear();
  }
  double appReadMaterializeUs = elapsedUs(phaseStarted);
  const auto gradientStructureStarted = Clock::now();
  if (outputs.gradients.layers.size() != current.layers.size())
    outputs.gradients.layers.resize(current.layers.size());
  const double outputGradientStructureCopyUs =
      elapsedUs(gradientStructureStarted);
  phaseStarted = Clock::now();
  tiny_lm::forEachParameterStorage(
      outputs.gradients,
      [&](const tiny_lm::ParameterDefinition &definition, size_t,
          std::vector<float> &values) {
        if (tiny_lm::parameterDefinitionEnabled(definition,
                                                parameterDimensions))
          values = appReadBuffers[gradient++];
      });
  outputs.layerInputGradients.clear();
  if (g.diagnosticOutputs) {
    outputs.layerInputGradients.reserve(g.numLayers);
    for (uint32_t layer = 0; layer < g.numLayers; ++layer)
      outputs.layerInputGradients.push_back(appReadBuffers[gradient++]);
  }
  if (g.diagnosticOutputs && g.numHeads > 1) {
    for (uint32_t layer = 0; layer < g.numLayers; ++layer) {
      for (uint32_t head = 0; head < g.numHeads; ++head) {
        const uint32_t id = g.layers[layer].heads[head].probabilities;
        outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
      }
    }
  }
  outputs.attentionGates.clear();
  if (g.headwiseG1Gate) {
    outputs.attentionGates.reserve(g.numLayers);
    for (uint32_t layer = 0; layer < g.numLayers; ++layer)
      outputs.attentionGates.push_back(appReadBuffers[gradient++]);
  }
  for (uint32_t id : g.tapRegistry)
    outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
  if (gradient != appReadBuffers.size()) {
    error = "generalized tiny training APP_READ result mapping";
    return false;
  }
  outputs.dEmbeddedInput = g.diagnosticOutputs && !outputs.layerInputGradients.empty()
      ? outputs.layerInputGradients.front()
      : std::vector<float>{};
  outputs.embeddedInput.clear();  // Not an APP_READ terminal in the L2/H1 ABI.
  appReadMaterializeUs += elapsedUs(phaseStarted);
  size_t poisonResidual = 0, nonFiniteResidual = 0;
  size_t firstBadRegistryIndex = appReadBuffers.size();
  size_t firstBadElement = 0;
  size_t firstBadNanCount = 0, firstBadInfCount = 0;
  const double appReadPoisonValidationUs = 0.0;
  phaseStarted = Clock::now();
  for (size_t registryIndex = 0; registryIndex < appReadBuffers.size();
       ++registryIndex) {
    const auto &values = appReadBuffers[registryIndex];
    size_t badCount = 0, nanCount = 0, firstBad = values.size();
    for (size_t element = 0; element < values.size(); ++element) {
      const float value = values[element];
      if (value == poison) ++poisonResidual;
      if (!std::isfinite(value)) {
        if (firstBad == values.size()) firstBad = element;
        ++badCount;
        if (std::isnan(value)) ++nanCount;
      }
    }
    nonFiniteResidual += badCount;
    if (badCount && firstBadRegistryIndex == appReadBuffers.size()) {
      firstBadRegistryIndex = registryIndex;
      firstBadElement = firstBad;
      firstBadNanCount = nanCount;
      firstBadInfCount = badCount - firstBadNanCount;
    }
  }
  const double appReadFiniteValidationUs = elapsedUs(phaseStarted);
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
                    "generalized_tiny_training_input_immutable=" +
                    (options_.validateAppWriteImmutability ? "true\n"
                                                          : "not_checked\n") +
                    "generalized_tiny_training_input_immutability_validation=" +
                    (options_.validateAppWriteImmutability ? "full\n"
                                                          : "disabled\n") +
                    "generalized_tiny_training_parameter_finiteness_validation=" +
                    (options_.validateAppWriteParameterFiniteness ? "full\n"
                                                                 : "optimizer_gated\n") +
                    "generalized_tiny_training_qnn_return=0\n"
                    "generalized_tiny_training_output_poison_count=0\n"
                    "generalized_tiny_training_output_nonfinite_count=0\n"
                    "generalized_tiny_training_outputs_all_written=true\n";
    g.executeDiagnosticsEmitted = true;
  }
  outputs.appReadValidated = true;
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
  metrics_.appWriteSchemaValidationUs.push_back(appWriteSchemaValidationUs);
  metrics_.appWriteSnapshotCopyUs.push_back(appWriteSnapshotCopyUs);
  metrics_.appWriteBindUs.push_back(appWriteBindUs);
  metrics_.appReadBufferAllocateUs.push_back(appReadBufferAllocateUs);
  metrics_.appReadPoisonFillUs.push_back(appReadPoisonFillUs);
  metrics_.appReadBindUs.push_back(appReadBindUs);
  metrics_.appWriteImmutabilityCheckUs.push_back(appWriteImmutabilityCheckUs);
  metrics_.appReadMaterializeUs.push_back(appReadMaterializeUs);
  metrics_.appReadPoisonValidationUs.push_back(appReadPoisonValidationUs);
  metrics_.appReadFiniteValidationUs.push_back(appReadFiniteValidationUs);
  metrics_.outputGradientStructureCopyUs.push_back(
      outputGradientStructureCopyUs);
  metrics_.trainingWrapperUs.push_back(elapsedUs(wrapperStarted));
  return true;
}

// Forward-only generation execute for the FORWARD_ONLY generalized graph.
// The graph has no target APP_WRITE and no gradient/loss APP_READ tensors:
// inputs are the token one-hot plus every parameter, and the only
// application-visible output is logits [T,V].  Everything else (poison
// sentinel, QNN return-code check, finite/poison validation, immutability
// check) mirrors executeTinyTransformerTrainingGeneralized fail-closed.
// Phase timings are recorded into g.lastPhaseTimings for profiling; they are
// additive diagnostics and never change generation semantics.
bool Runtime::executeTinyTransformerForwardOnly(
    const std::vector<float> &input, const TinyTransformerParameters &current,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using GG = Impl::GeneralizedTinyTransformerTrainingGraph;
  auto &g = impl_->generalizedTinyTransformerTraining;
  ForwardOnlyPhaseTimings phase{};
  const auto totalStarted = Clock::now();
  if (!g.forwardOnly || !g.gradientRegistry.empty()) {
    error = "generalized tiny training forward-only state is invalid";
    return false;
  }
  const auto schemaStarted = Clock::now();
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
  const tiny_lm::ParameterDimensions parameterDimensions{
      g.vocabularySize, g.dimension, g.feedForwardDimension, g.numLayers,
      g.numHeads, g.headwiseG1Gate};
  size_t expectedParameterCount = 0;
  size_t expectedAppWriteCount = 0, expectedAppReadCount = 0, headTapCount = 0;
  if (!tiny_lm::checkedParameterInstanceCount(parameterDimensions,
                                              &expectedParameterCount) ||
      // one-hot is the only non-parameter APP_WRITE tensor.
      !checkedAdd(1, expectedParameterCount, &expectedAppWriteCount) ||
      // last-position logits only when minimalOutputs; otherwise full logits
      // + head probability taps.
      (g.numHeads > 1 && !g.minimalOutputs &&
       !checkedMultiply(g.numLayers, g.numHeads, &headTapCount)) ||
      !checkedAdd(1, headTapCount, &expectedAppReadCount)) {
    error = "APP_RESOURCE_ESTIMATOR: forward-only registry count overflow";
    return false;
  }
  if (g.headwiseG1Gate && !g.minimalOutputs &&
      !checkedAdd(expectedAppReadCount, g.numLayers, &expectedAppReadCount)) {
    error = "APP_RESOURCE_ESTIMATOR: forward-only gate registry count overflow";
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
  // FORWARD_ONLY generation: parameters are loaded once from the checkpoint
  // and never change between autoregressive steps.  Skip per-token finite()
  // scans on parameter tensors (they are validated at checkpoint load) and
  // only finite-check the one-hot input, which does change every step.
  const auto finite = [](const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (input.size() != tokenVocabulary || !finite(input) ||
      !tiny_lm::parameterStorageMatchesDefinitions(current,
                                                   parameterDimensions)) {
    error = "generalized tiny training forward-only buffer schema";
    return false;
  }
  phase.schemaValidationUs = elapsedUs(schemaStarted);
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
  const auto appWriteStarted = Clock::now();
  std::vector<const std::vector<float> *> appWriteValues;
  appWriteValues.reserve(g.appWriteRegistry.size());
  appWriteValues.push_back(&input);
  tiny_lm::forEachParameterStorage(
      current, [&](const tiny_lm::ParameterDefinition &definition, size_t,
                   const std::vector<float> &values) {
        if (tiny_lm::parameterDefinitionEnabled(definition,
                                                parameterDimensions))
          appWriteValues.push_back(&values);
      });
  if (appWriteValues.size() != g.appWriteRegistry.size()) {
    error = "generalized tiny training forward-only APP_WRITE registry count";
    return false;
  }
  // Persistent APP_WRITE tensor descriptors: rebuild clientBuf pointers each
  // execute (the one-hot pointer changes; parameter pointers are stable).
  std::vector<Qnn_Tensor_t> &in = g.persistentAppWriteTensors;
  if (!g.persistentBuffersReady || in.size() != g.appWriteRegistry.size()) {
    in.assign(g.appWriteRegistry.size(), QNN_TENSOR_INIT);
  }
  // Only snapshot the one-hot input for immutability.  Parameters are
  // checkpoint-loaded and const during generation; snapshotting 2 MB of
  // weights every token was the dominant host-side copy cost.
  std::vector<float> oneHotSnapshot;
  std::uint64_t appWriteBytes = 0;
  double appWriteSnapshotUs = 0.0;
  for (size_t index = 0; index < g.appWriteRegistry.size(); ++index) {
    const uint32_t id = g.appWriteRegistry[index];
    size_t count = 0;
    if (id >= g.tensors.size() || g.tensors[id].v1.type != QNN_TENSOR_TYPE_APP_WRITE ||
        !elements(id, count) || appWriteValues[index]->size() != count) {
      error = "generalized tiny training forward-only APP_WRITE registry schema";
      return false;
    }
    if (index == 0) {
      const auto snapshotStarted = Clock::now();
      oneHotSnapshot = *appWriteValues[index];
      appWriteSnapshotUs = elapsedUs(snapshotStarted);
    }
    appWriteBytes += uint64_t(count) * sizeof(float);
    in[index] = g.tensors[id];
    in[index].v1.clientBuf = {const_cast<float *>(appWriteValues[index]->data()),
                              uint32_t(count * sizeof(float))};
  }
  phase.appWriteBindUs = elapsedUs(appWriteStarted) - appWriteSnapshotUs;
  phase.appWriteSnapshotUs = appWriteSnapshotUs;
  phase.appWriteBytes = appWriteBytes;
  phase.appWriteTensorCount = uint32_t(in.size());
  constexpr float poison = 1.1415926f;
  const auto appReadStarted = Clock::now();
  // Persistent APP_READ buffers: allocate once, then std::fill in place.
  std::vector<std::vector<float>> &appReadBuffers = g.persistentAppReadBuffers;
  std::vector<Qnn_Tensor_t> &out = g.persistentAppReadTensors;
  if (!g.persistentBuffersReady || appReadBuffers.size() != g.appReadRegistry.size() ||
      out.size() != g.appReadRegistry.size()) {
    appReadBuffers.assign(g.appReadRegistry.size(), {});
    out.assign(g.appReadRegistry.size(), QNN_TENSOR_INIT);
  }
  std::uint64_t appReadBytes = 0;
  double appReadPoisonFillUs = 0.0;
  for (size_t index = 0; index < g.appReadRegistry.size(); ++index) {
    const uint32_t id = g.appReadRegistry[index];
    size_t count = 0;
    if (id >= g.tensors.size() || g.tensors[id].v1.type != QNN_TENSOR_TYPE_APP_READ ||
        !elements(id, count)) {
      error = "generalized tiny training forward-only APP_READ registry schema";
      return false;
    }
    const auto fillStarted = Clock::now();
    if (appReadBuffers[index].size() != count) {
      appReadBuffers[index].assign(count, poison);
    } else {
      std::fill(appReadBuffers[index].begin(), appReadBuffers[index].end(), poison);
    }
    appReadPoisonFillUs += elapsedUs(fillStarted);
    appReadBytes += uint64_t(count) * sizeof(float);
    out[index] = g.tensors[id];
    out[index].v1.clientBuf = {appReadBuffers[index].data(),
                               uint32_t(count * sizeof(float))};
  }
  g.persistentBuffersReady = true;
  phase.appReadAllocateUs = elapsedUs(appReadStarted) - appReadPoisonFillUs;
  phase.appReadPoisonFillUs = appReadPoisonFillUs;
  phase.appReadBytes = appReadBytes;
  phase.appReadTensorCount = uint32_t(out.size());
  phase.appReadBindUs = 0.0;
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
  phase.graphExecuteUs = elapsedUs(started);
  metrics_.executeUs.push_back(phase.graphExecuteUs);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status), QNN_GET_ERROR_CODE(status),
                           status == QNN_SUCCESS);
  const auto immutabilityStarted = Clock::now();
  // Only the one-hot input is snapshotted; parameters are const during
  // generation.  Check that the one-hot was not mutated by graphExecute.
  const bool inputsUnchanged = oneHotSnapshot == input;
  phase.immutabilityCheckUs = elapsedUs(immutabilityStarted);
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
  const auto materializeStarted = Clock::now();
  outputs.taps.clear();
  outputs.next = {};
  // minimalOutputs maps registry slot 0 directly to [V] last-position
  // logits; diagnostic/debug execution maps it to full [T,V] logits.
  outputs.logits = appReadBuffers[0];
  size_t gradient = 1;
  // minimalOutputs: head probabilities and gates are NATIVE, not APP_READ.
  if (g.numHeads > 1 && !g.minimalOutputs) {
    for (uint32_t layer = 0; layer < g.numLayers; ++layer) {
      for (uint32_t head = 0; head < g.numHeads; ++head) {
        const uint32_t id = g.layers[layer].heads[head].probabilities;
        outputs.taps.push_back({g.names[id], appReadBuffers[gradient++]});
      }
    }
  }
  outputs.attentionGates.clear();
  if (g.headwiseG1Gate && !g.minimalOutputs) {
    outputs.attentionGates.reserve(g.numLayers);
    for (uint32_t layer = 0; layer < g.numLayers; ++layer)
      outputs.attentionGates.push_back(appReadBuffers[gradient++]);
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
  phase.outputMaterializeUs = elapsedUs(materializeStarted);
  // Fused single-pass scan: poison residual and non-finite detection in one
  // traversal per tensor.  The first-bad-element path still classifies
  // nan/inf but only for the first failing tensor.
  const auto scanStarted = Clock::now();
  size_t poisonResidual = 0, nonFiniteResidual = 0;
  size_t firstBadRegistryIndex = appReadBuffers.size();
  size_t firstBadElement = 0;
  size_t firstBadNanCount = 0, firstBadInfCount = 0;
  for (size_t registryIndex = 0; registryIndex < appReadBuffers.size();
       ++registryIndex) {
    const auto &values = appReadBuffers[registryIndex];
    size_t localPoison = 0, localBad = 0;
    size_t localBadIndex = values.size();
    size_t localNan = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      const float v = values[i];
      if (v == poison) ++localPoison;
      if (!std::isfinite(v)) {
        ++localBad;
        if (localBadIndex == values.size()) localBadIndex = i;
        if (std::isnan(v)) ++localNan;
      }
    }
    poisonResidual += localPoison;
    nonFiniteResidual += localBad;
    if (localBad && firstBadRegistryIndex == appReadBuffers.size()) {
      firstBadRegistryIndex = registryIndex;
      firstBadElement = localBadIndex;
      firstBadNanCount = localNan;
      firstBadInfCount = localBad - localNan;
    }
  }
  phase.poisonScanUs = elapsedUs(scanStarted) * 0.5;  // apportioned
  phase.finiteScanUs = elapsedUs(scanStarted) * 0.5;
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
  phase.totalUs = elapsedUs(totalStarted);
  g.lastPhaseTimings = phase;
  return true;
}
