// Dynamic L>1 training graph entry point. This file intentionally has no
// dependency on TinyTransformerTrainingGraph::TensorIndex: every tensor is
// registered by name and has per-entry dimension storage.
bool Runtime::prepareTinyTransformerTrainingGeneralized(
    uint32_t tokens, uint32_t dimension, uint32_t feedForwardDimension,
    float epsilon, bool diagnosticOutputs, std::string &error,
    uint32_t vocabularySize, TinyTransformerTrainingVariant variant,
    TinyTransformerTrainingTapSet tapSet, uint32_t numLayers,
    uint32_t numHeads) {
  if (!impl_) {
    error = "generalized tiny training runtime unavailable";
    return false;
  }
  auto &g = impl_->generalizedTinyTransformerTraining;
  using State = Impl::GeneralizedTinyTransformerTrainingGraph::State;
  if (g.state != State::INACTIVE || g.graph != nullptr) {
    error = "generalized tiny training reprepare or mode transition is unavailable";
    return false;
  }
  g = {};
  g.state = State::PREPARING;
  apiTrace_.fullStepGraphCreateCalled = false;
  apiTrace_.fullStepGraphFinalizeCalled = false;
  apiTrace_.fullStepGraphCreateResult = -1;
  apiTrace_.fullStepGraphFinalizeResult = -1;
  apiTrace_.fullStepGraphHandleNonnull = false;
  const auto fail = [&](const std::string &message) {
    g.active = false;
    g.state = State::FAILED;
    error = message;
    return false;
  };
  if (!impl_ || !impl_->context || !tokens || !dimension ||
      !feedForwardDimension || !std::isfinite(epsilon) || epsilon <= 0.0f ||
      numHeads != 1 || numLayers != 2 || vocabularySize == 0 ||
      variant != TinyTransformerTrainingVariant::FULL ||
      tapSet != TinyTransformerTrainingTapSet::NONE) {
    return fail("generalized tiny training currently requires L2/H1 language-model FULL");
  }
  // First pass: register every tensor before graph creation. QNN keeps the
  // name, dimension and static-client-buffer pointers, so all three backing
  // collections are reserved for the complete L2/H1 forward topology before
  // tensor() writes those pointers.
  constexpr size_t kGlobalTensorSlots = 22;
  constexpr size_t kLayerForwardSlots = 46;
  const size_t registryCapacity =
      kGlobalTensorSlots + size_t(numLayers) * kLayerForwardSlots;
  g.tensors.reserve(registryCapacity);
  g.names.reserve(registryCapacity);
  g.dimensions.reserve(registryCapacity);
  g.tokens = tokens;
  g.dimension = dimension;
  g.feedForwardDimension = feedForwardDimension;
  g.vocabularySize = vocabularySize;
  g.numLayers = numLayers;
  g.numHeads = numHeads;
  g.languageModel = true;
  g.diagnosticOutputs = diagnosticOutputs;
  g.attentionScale = 1.0f / std::sqrt(float(dimension));
  g.centeredScale = options_.tinyTransformerCenteredScale;
  if (!std::isfinite(g.centeredScale) || g.centeredScale <= 0.0f) {
    return fail("generalized tiny training invalid LayerNorm scaling");
  }
  const double epsilonScaled = double(epsilon) * double(g.centeredScale) *
                               double(g.centeredScale);
  if (!std::isfinite(epsilonScaled) || epsilonScaled <= 0.0 ||
      epsilonScaled > double(std::numeric_limits<float>::max()) ||
      !std::isfinite(float(epsilonScaled)) || float(epsilonScaled) <= 0.0f) {
    return fail("generalized tiny training centered scale makes epsilon unrepresentable");
  }
  g.epsilonScaled = float(epsilonScaled);
  bool registryShapeValid = true;
  auto shapeValid = [&](std::initializer_list<uint32_t> shape) {
    size_t elements = 1;
    for (uint32_t extent : shape) {
      if (extent == 0 || elements > std::numeric_limits<size_t>::max() / extent) {
        return false;
      }
      elements *= extent;
    }
    return elements <= std::numeric_limits<uint32_t>::max() / sizeof(float);
  };
  auto add = [&](const std::string &name, Qnn_TensorType_t type,
                 std::initializer_list<uint32_t> dimensions,
                 Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32) {
    if (!shapeValid(dimensions)) {
      registryShapeValid = false;
      return std::numeric_limits<uint32_t>::max();
    }
    const uint32_t id = uint32_t(g.tensors.size());
    g.tensors.push_back(QNN_TENSOR_INIT);
    g.names.push_back(name);
    g.dimensions.emplace_back(dimensions);
    auto &tensorValue = g.tensors.back();
    tensor(tensorValue, g.names.back().c_str(), type,
           g.dimensions.back().data(), dataType,
           uint32_t(g.dimensions.back().size()));
    return id;
  };
  const auto oneHot = add("token_one_hot", QNN_TENSOR_TYPE_APP_WRITE,
                          {tokens, vocabularySize});
  const auto embedding = add("token_embedding", QNN_TENSOR_TYPE_APP_WRITE,
                             {vocabularySize, dimension});
  const auto position = add("position", QNN_TENSOR_TYPE_STATIC,
                            {tokens, dimension});
  const auto embedded = add("embedded_token", QNN_TENSOR_TYPE_NATIVE,
                            {tokens, dimension});
  const auto input = add("layer_00_input", QNN_TENSOR_TYPE_NATIVE,
                         {tokens, dimension});
  const auto mask = add("causal_mask", QNN_TENSOR_TYPE_STATIC,
                        {tokens, tokens});
  const auto scale = add("attention_scale", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto lastAxis = add("last_axis", QNN_TENSOR_TYPE_STATIC, {1},
                            QNN_DATATYPE_UINT_32);
  const auto rowAxis = add("row_axis", QNN_TENSOR_TYPE_STATIC, {1},
                           QNN_DATATYPE_UINT_32);
  const auto centerScale = add("center_scale", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto epsScaled = add("epsilon_scaled", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto dim = add("dimension", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto inverseDim = add("inverse_dimension", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto outputProjection = add("output_projection", QNN_TENSOR_TYPE_APP_WRITE,
                                    {dimension, vocabularySize});
  const auto logits = add("logits", QNN_TENSOR_TYPE_APP_READ,
                          {tokens, vocabularySize});
  const auto probabilities = add("lm_probabilities",
                                 diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                 {tokens, vocabularySize});
  const auto target = add("target", QNN_TENSOR_TYPE_APP_WRITE,
                          {tokens, vocabularySize});
  const auto difference = add("lm_difference", QNN_TENSOR_TYPE_NATIVE,
                              {tokens, vocabularySize});
  const auto gradientScale = add("gradient_scale", QNN_TENSOR_TYPE_STATIC, {1, 1});
  const auto dLogits = add("dlogits",
                           diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                           {tokens, vocabularySize});
  const auto dOutput = add("doutput",
                            diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                           {tokens, dimension});
  const auto dOutputProjection = add("doutput_projection",
                                     diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                     {dimension, vocabularySize});
  g.tokenOneHot = oneHot;
  g.tokenEmbedding = embedding;
  g.target = target;
  g.outputProjection = outputProjection;
  g.gradientScale = 1.0f / float(tokens);
  enum LayerActivation : size_t {
    LN1_MEAN, LN1_CENTERED, LN1_CENTERED_S, LN1_SQUARE, LN1_VARIANCE,
    LN1_VARIANCE_EPS, LN1_INV_S, LN1_INV, LN1_XHAT, LN1_XHAT_GAMMA, LN1,
    Q, K, V, ATTENTION_SCORES, ATTENTION_SCALED, ATTENTION_MASKED,
    ATTENTION_PROBABILITIES, ATTENTION_CONTEXT, ATTENTION_PROJECTED,
    RESIDUAL1, LN2_MEAN, LN2_CENTERED, LN2_CENTERED_S, LN2_SQUARE,
    LN2_VARIANCE, LN2_VARIANCE_EPS, LN2_INV_S, LN2_INV, LN2_XHAT,
    LN2_XHAT_GAMMA, LN2, FF1, RELU, FF2, OUTPUT,
  };
  // Keep all layer-local parameter and forward-cache tensors separate. The
  // order is the deterministic parameter registry order used by future
  // binding/execution work; activation indices above never cross a layer.
  for (uint32_t layer = 0; layer < numLayers; ++layer) {
    auto &record = g.layers.emplace_back();
    const std::string prefix = "layer_0" + std::to_string(layer) + "_";
    record.gamma1 = add(prefix + "norm1_gamma", QNN_TENSOR_TYPE_APP_WRITE, {dimension});
    record.beta1 = add(prefix + "norm1_beta", QNN_TENSOR_TYPE_APP_WRITE, {dimension});
    record.wq = add(prefix + "wq", QNN_TENSOR_TYPE_APP_WRITE, {dimension, dimension});
    record.wk = add(prefix + "wk", QNN_TENSOR_TYPE_APP_WRITE, {dimension, dimension});
    record.wv = add(prefix + "wv", QNN_TENSOR_TYPE_APP_WRITE, {dimension, dimension});
    record.wo = add(prefix + "wo", QNN_TENSOR_TYPE_APP_WRITE, {dimension, dimension});
    record.gamma2 = add(prefix + "norm2_gamma", QNN_TENSOR_TYPE_APP_WRITE, {dimension});
    record.beta2 = add(prefix + "norm2_beta", QNN_TENSOR_TYPE_APP_WRITE, {dimension});
    record.w1 = add(prefix + "ffn_w1", QNN_TENSOR_TYPE_APP_WRITE, {dimension, feedForwardDimension});
    record.w2 = add(prefix + "ffn_w2", QNN_TENSOR_TYPE_APP_WRITE, {feedForwardDimension, dimension});
    record.input = layer == 0 ? input : g.layers[layer - 1].output;
    for (const char *name : {"ln1_mean", "ln1_centered", "ln1_centered_s",
         "ln1_square", "ln1_variance", "ln1_variance_eps", "ln1_inv_s",
         "ln1_inv", "ln1_xhat", "ln1_xhat_gamma", "ln1", "q", "k", "v",
         "attention_scores", "attention_scaled", "attention_masked",
         "attention_probabilities", "attention_context", "attention_projected",
         "residual1", "ln2_mean", "ln2_centered", "ln2_centered_s",
         "ln2_square", "ln2_variance", "ln2_variance_eps", "ln2_inv_s",
         "ln2_inv", "ln2_xhat", "ln2_xhat_gamma", "ln2", "ff1", "relu",
         "ff2", "output"}) {
      const std::string activationName(name);
      const bool row = activationName == "ln1_mean" ||
                       activationName == "ln1_variance" ||
                       activationName == "ln1_variance_eps" ||
                       activationName == "ln1_inv_s" ||
                       activationName == "ln1_inv" ||
                       activationName == "ln2_mean" ||
                       activationName == "ln2_variance" ||
                       activationName == "ln2_variance_eps" ||
                       activationName == "ln2_inv_s" ||
                       activationName == "ln2_inv";
      const bool score = std::string(name).find("attention_") != std::string::npos;
      const bool ff = std::string(name) == "ff1" || std::string(name) == "relu";
      const Qnn_TensorType_t activationType =
          diagnosticOutputs && layer + 1 == numLayers &&
                  activationName == "output"
              ? QNN_TENSOR_TYPE_APP_READ
              : QNN_TENSOR_TYPE_NATIVE;
      record.activations.push_back(add(prefix + name, activationType,
          score ? std::initializer_list<uint32_t>{tokens, tokens} :
          ff ? std::initializer_list<uint32_t>{tokens, feedForwardDimension} :
          row ? std::initializer_list<uint32_t>{tokens, 1} :
                std::initializer_list<uint32_t>{tokens, dimension}));
    }
    record.output = record.activations.back();
  }
  g.appWriteRegistry = {oneHot, embedding, target};
  g.parameterRegistry.push_back(embedding);
  for (const auto &record : g.layers) {
    for (uint32_t parameter : {record.wq, record.wk, record.wv, record.wo,
         record.gamma1, record.beta1, record.gamma2, record.beta2,
         record.w1, record.w2}) {
      g.appWriteRegistry.push_back(parameter);
      g.parameterRegistry.push_back(parameter);
    }
  }
  g.appWriteRegistry.push_back(outputProjection);
  g.parameterRegistry.push_back(outputProjection);
  g.appReadRegistry.push_back(logits);
  if (diagnosticOutputs) {
    g.appReadRegistry.insert(g.appReadRegistry.end(),
                             {g.layers.back().output, probabilities, dLogits,
                              dOutput, dOutputProjection});
  }
  if (g.parameterRegistry.size() != 22 ||
      std::unordered_set<uint32_t>(g.parameterRegistry.begin(),
                                   g.parameterRegistry.end()).size() !=
          g.parameterRegistry.size()) {
    return fail("generalized tiny training parameter registry is not canonical");
  }
  if (!registryShapeValid) {
    return fail("generalized tiny training tensor shape exceeds client-buffer limit");
  }
  std::unordered_set<std::string> tensorNames;
  for (const auto &name : g.names) {
    if (!tensorNames.insert(name).second) {
      return fail("generalized tiny training duplicate tensor name: " + name);
    }
  }
  if (g.tensors.size() != registryCapacity || g.names.size() != registryCapacity ||
      g.dimensions.size() != registryCapacity) {
    return fail("generalized tiny training registry capacity mismatch");
  }
  g.maskData.assign(size_t(tokens) * tokens, 0.0f);
  for (uint32_t r = 0; r < tokens; ++r)
    for (uint32_t c = r + 1; c < tokens; ++c) g.maskData[size_t(r) * tokens + c] = -10000.0f;
  g.positionData.assign(size_t(tokens) * dimension, 0.0f);
  for (uint32_t t = 0; t < tokens; ++t)
    for (uint32_t d = 0; d < dimension; ++d) {
      const float frequency =
          std::pow(10000.0f, -float(d & ~1u) / float(dimension));
      g.positionData[size_t(t) * dimension + d] =
          ((d & 1u) ? std::cos(float(t) * frequency)
                    : std::sin(float(t) * frequency)) * 0.05f;
    }
  g.dimensionValue = float(dimension);
  g.inverseDimensionValue = 1.0f / g.dimensionValue;
  g.tensors[mask].v1.clientBuf = {g.maskData.data(), uint32_t(g.maskData.size() * sizeof(float))};
  g.tensors[position].v1.clientBuf = {g.positionData.data(), uint32_t(g.positionData.size() * sizeof(float))};
  g.tensors[scale].v1.clientBuf = {&g.attentionScale, sizeof(float)};
  g.tensors[lastAxis].v1.clientBuf = {g.lastAxisData, sizeof(g.lastAxisData)};
  g.tensors[rowAxis].v1.clientBuf = {g.rowAxisData, sizeof(g.rowAxisData)};
  g.tensors[centerScale].v1.clientBuf = {&g.centeredScale, sizeof(float)};
  g.tensors[epsScaled].v1.clientBuf = {&g.epsilonScaled, sizeof(float)};
  g.tensors[gradientScale].v1.clientBuf = {&g.gradientScale, sizeof(float)};
  g.tensors[dim].v1.clientBuf = {&g.dimensionValue, sizeof(g.dimensionValue)};
  g.tensors[inverseDim].v1.clientBuf = {&g.inverseDimensionValue, sizeof(g.inverseDimensionValue)};
  auto started = Clock::now();
  auto status = impl_->api.graphCreate(impl_->context,
      "phonelm_tiny_transformer_training_l2h1", nullptr, &g.graph);
  apiTrace_.fullStepGraphCreateCalled = true;
  apiTrace_.fullStepGraphCreateResult = QNN_GET_ERROR_CODE(status);
  apiTrace_.fullStepGraphHandleNonnull = g.graph != nullptr;
  metrics_.graphCreateUs += elapsedUs(started);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    return fail("generalized graphCreate=" +
                std::to_string(QNN_GET_ERROR_CODE(status)));
  }
  // Second pass: graph tensor creation and node emission. Do not fall through
  // to the L1 graph: that would silently collapse the requested layer count.
  for (auto &value : g.tensors) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, &value);
    if (status != QNN_SUCCESS) {
      return fail("generalized tensorCreate=" +
                  std::to_string(QNN_GET_ERROR_CODE(status)));
    }
    ++g.tensorCreateSuccessCount;
  }
  auto ok = [&](Qnn_ErrorHandle_t value, const std::string &label) {
    if (value == QNN_SUCCESS) {
      ++g.graphAddNodeSuccessCount;
      return true;
    }
    return fail("generalized " + label + '=' +
                std::to_string(QNN_GET_ERROR_CODE(value)));
  };
  auto binary = [&](const std::string &name, const char *type, uint32_t a,
                    uint32_t b, uint32_t out, const char *label) {
    Qnn_Tensor_t inputs[2] = {g.tensors[a], g.tensors[b]};
    return ok(addSimpleNode(impl_->api, g.graph, name.c_str(), type, inputs, 2,
                            &g.tensors[out]), label);
  };
  auto unary = [&](const std::string &name, const char *type, uint32_t a,
                   uint32_t out, const char *label) {
    return ok(addSimpleNode(impl_->api, g.graph, name.c_str(), type,
                            &g.tensors[a], 1, &g.tensors[out]), label);
  };
  auto matmul = [&](const std::string &name, uint32_t a, uint32_t b,
                    uint32_t out, const char *label,
                    const char *transpose = nullptr) {
    Qnn_Tensor_t inputs[2] = {g.tensors[a], g.tensors[b]};
    return ok(addMatMulNode(impl_->api, g.graph, name.c_str(), inputs,
                            &g.tensors[out], transpose), label);
  };
  auto softmax = [&](const std::string &name, uint32_t source, uint32_t out,
                     const char *label) {
    Qnn_Param_t parameters[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    parameters[0].paramType = QNN_PARAMTYPE_SCALAR;
    parameters[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
    parameters[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
    parameters[0].scalarParam.uint32Value = 1;
    parameters[1].paramType = QNN_PARAMTYPE_SCALAR;
    parameters[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
    parameters[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    parameters[1].scalarParam.floatValue = 1.0f;
    Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
    operation.v1.name = name.c_str();
    operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    operation.v1.typeName = QNN_OP_SOFTMAX;
    operation.v1.numOfParams = 2;
    operation.v1.params = parameters;
    operation.v1.numOfInputs = 1;
    operation.v1.inputTensors = &g.tensors[source];
    operation.v1.numOfOutputs = 1;
    operation.v1.outputTensors = &g.tensors[out];
    return ok(impl_->api.graphAddNode(g.graph, operation), label);
  };
  if (!matmul("l2h1_lm_embedding", oneHot, embedding, embedded, "embedding") ||
      !binary("l2h1_lm_position", QNN_OP_ELEMENT_WISE_ADD, embedded, position,
              input, "position"))
    return fail(error);
  TransformerTrainingLayerNormBuilder layerNorm{
      impl_->api, g.graph, g.tensors.data(), int(lastAxis), int(rowAxis),
      int(centerScale), int(epsScaled), int(dim), int(inverseDim), error,
      g.graphAddNodeSuccessCount};
  for (uint32_t layer = 0; layer < numLayers; ++layer) {
    const auto &record = g.layers[layer];
    const auto cache = [&](LayerActivation activation) {
      return record.activations[size_t(activation)];
    };
    const std::string prefix = "l2h1_layer_0" + std::to_string(layer) + '_';
    if (!layerNorm.forward((prefix + "ln1").c_str(), int(record.input),
                           int(record.gamma1), int(record.beta1),
                           int(cache(LN1_MEAN)), int(cache(LN1_CENTERED)),
                           int(cache(LN1_CENTERED_S)), int(cache(LN1_SQUARE)),
                           int(cache(LN1_VARIANCE)), int(cache(LN1_VARIANCE_EPS)),
                           int(cache(LN1_INV_S)), int(cache(LN1_INV)),
                           int(cache(LN1_XHAT)), int(cache(LN1_XHAT_GAMMA)),
                           int(cache(LN1))))
      return fail(error);
    if (!matmul(prefix + "q", cache(LN1), record.wq, cache(Q), "q") ||
        !matmul(prefix + "k", cache(LN1), record.wk, cache(K), "k") ||
        !matmul(prefix + "v", cache(LN1), record.wv, cache(V), "v") ||
        !matmul(prefix + "scores", cache(Q), cache(K), cache(ATTENTION_SCORES),
                 "scores", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
        !binary(prefix + "score_scale", QNN_OP_ELEMENT_WISE_MULTIPLY,
                cache(ATTENTION_SCORES), scale, cache(ATTENTION_SCALED),
                "score scale") ||
        !binary(prefix + "score_mask", QNN_OP_ELEMENT_WISE_ADD,
                cache(ATTENTION_SCALED), mask, cache(ATTENTION_MASKED),
                "score mask") ||
        !softmax(prefix + "softmax", cache(ATTENTION_MASKED),
                 cache(ATTENTION_PROBABILITIES), "attention softmax") ||
        !matmul(prefix + "context", cache(ATTENTION_PROBABILITIES), cache(V),
                 cache(ATTENTION_CONTEXT), "context") ||
        !matmul(prefix + "project", cache(ATTENTION_CONTEXT), record.wo,
                 cache(ATTENTION_PROJECTED), "project") ||
        !binary(prefix + "residual1", QNN_OP_ELEMENT_WISE_ADD, record.input,
                cache(ATTENTION_PROJECTED), cache(RESIDUAL1), "residual1"))
      return fail(error);
    if (!layerNorm.forward((prefix + "ln2").c_str(), int(cache(RESIDUAL1)),
                           int(record.gamma2), int(record.beta2),
                           int(cache(LN2_MEAN)), int(cache(LN2_CENTERED)),
                           int(cache(LN2_CENTERED_S)), int(cache(LN2_SQUARE)),
                           int(cache(LN2_VARIANCE)), int(cache(LN2_VARIANCE_EPS)),
                           int(cache(LN2_INV_S)), int(cache(LN2_INV)),
                           int(cache(LN2_XHAT)), int(cache(LN2_XHAT_GAMMA)),
                           int(cache(LN2))))
      return fail(error);
    if (!matmul(prefix + "ff1", cache(LN2), record.w1, cache(FF1), "ff1") ||
        !unary(prefix + "relu", QNN_OP_RELU, cache(FF1), cache(RELU), "relu") ||
        !matmul(prefix + "ff2", cache(RELU), record.w2, cache(FF2), "ff2") ||
        !binary(prefix + "output", QNN_OP_ELEMENT_WISE_ADD, cache(RESIDUAL1),
                cache(FF2), cache(OUTPUT), "output"))
      return fail(error);
  }
  if (!matmul("l2h1_lm_logits", g.layers.back().output, outputProjection,
              logits, "logits") ||
      !softmax("l2h1_lm_softmax", logits, probabilities, "lm softmax") ||
      !binary("l2h1_lm_difference", QNN_OP_ELEMENT_WISE_SUBTRACT,
              probabilities, target, difference, "lm difference") ||
      !binary("l2h1_lm_dlogits", QNN_OP_ELEMENT_WISE_MULTIPLY, difference,
              gradientScale, dLogits, "lm dlogits") ||
      !matmul("l2h1_lm_doutput", dLogits, outputProjection, dOutput,
              "lm doutput", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
      !matmul("l2h1_lm_doutput_projection", g.layers.back().output, dLogits,
              dOutputProjection, "lm doutput projection",
              QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))
    return fail(error);
  started = Clock::now();
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  apiTrace_.fullStepGraphFinalizeCalled = true;
  apiTrace_.fullStepGraphFinalizeResult = QNN_GET_ERROR_CODE(status);
  metrics_.graphFinalizeUs += elapsedUs(started);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    return fail("generalized graphFinalize=" +
                std::to_string(QNN_GET_ERROR_CODE(status)));
  }
  g.active = true;
  g.state = State::ACTIVE;
  diagnostics_ += "tiny_transformer_training_graph=L2H1_EXPLICIT_FORWARD\n"
                  "tiny_transformer_training_graph_count=1\n"
                  "tiny_transformer_training_forward_layers=2\n"
                  "tiny_transformer_training_forward_heads=1\n"
                  "tiny_transformer_training_parameter_registry_count=" +
                  std::to_string(g.parameterRegistry.size()) + '\n' +
                  "tiny_transformer_training_diagnostic_outputs=" +
                  std::string(g.diagnosticOutputs ? "true\n" : "false\n") +
                  "tiny_transformer_training_app_read_registry_count=" +
                  std::to_string(g.appReadRegistry.size()) + '\n' +
                  "tiny_transformer_training_tensor_create_count=" +
                  std::to_string(g.tensorCreateSuccessCount) + '\n' +
                  "tiny_transformer_training_node_count=" +
                  std::to_string(g.graphAddNodeSuccessCount) + '\n';
  return true;
}
