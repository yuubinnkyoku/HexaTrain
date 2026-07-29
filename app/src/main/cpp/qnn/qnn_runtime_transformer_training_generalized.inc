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
      numLayers == 0 || numLayers > 2 || numHeads == 0 || numHeads > 2 ||
      dimension % numHeads != 0 || vocabularySize == 0 ||
      !diagnosticOutputs ||
      variant != TinyTransformerTrainingVariant::FULL ||
      tapSet != TinyTransformerTrainingTapSet::NONE) {
    return fail("generalized tiny training requires diagnostic L1/L2 H1/H2 language-model FULL");
  }
  // First pass: register every tensor before graph creation. QNN keeps the
  // name, dimension and static-client-buffer pointers, so all three backing
  // collections are reserved for the complete L1/L2 H1/H2 topology before
  // tensor() writes those pointers.
  constexpr size_t kGlobalTensorSlots = 24;
  constexpr size_t kLayerForwardBackwardSlots = 100;
  constexpr size_t kHeadForwardBackwardSlots = 24;
  const size_t kHeadAccumulatorSlots = numHeads > 1 ? 4 * (numHeads - 1) : 0;
  const size_t kLayerSlots = kLayerForwardBackwardSlots +
      (numHeads > 1 ? size_t(numHeads) * kHeadForwardBackwardSlots +
                       kHeadAccumulatorSlots
                    : 0);
  const size_t registryCapacity =
      kGlobalTensorSlots + size_t(numLayers) * kLayerSlots;
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
  g.attentionScale = 1.0f / std::sqrt(float(dimension / numHeads));
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
  const auto zeroFf = add("zero_ff", QNN_TENSOR_TYPE_STATIC,
                          {tokens, feedForwardDimension});
  const auto dLogits = add("dlogits",
                           diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                           {tokens, vocabularySize});
  const auto dOutput = add("doutput",
                            diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                           {tokens, dimension});
  const auto dOutputProjection = add("doutput_projection",
                                     QNN_TENSOR_TYPE_APP_READ,
                                     {dimension, vocabularySize});
  const auto dEmbedding = add("dembedding", QNN_TENSOR_TYPE_APP_READ,
                              {vocabularySize, dimension});
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
  enum LayerGradient : size_t {
    DWQ, DWK, DWV, DWO, DG1, DB1, DG2, DB2, DW1, DW2,
  };
  enum LayerBackward : size_t {
    DRELU, RELU_MASK, DFF1, DLN2, DXHAT2, SUM_DXHAT2,
    DXHAT_XHAT2, DY_XHAT2, SUM_DXHAT_XHAT2, D_TIMES_DXHAT2,
    FIRST_DIFF2, XHAT_TIMES_SUM2, BRACKET2, INV_STD_OVER_D2,
    DRESIDUAL1_LN, DRESIDUAL1, DCONTEXT, DPROBABILITIES, DV,
    SOFTMAX_PRODUCT, SOFTMAX_DOT, SOFTMAX_CENTERED, DSCORES,
    DQ_RAW, DK_RAW, DQ, DK, DLN1_Q, DLN1_K, DLN1_V, DLN1_QK,
    DLN1, DXHAT1, SUM_DXHAT1, DXHAT_XHAT1, DY_XHAT1,
    SUM_DXHAT_XHAT1, D_TIMES_DXHAT1, FIRST_DIFF1,
    XHAT_TIMES_SUM1, BRACKET1, INV_STD_OVER_D1, DINPUT_NORM, DINPUT,
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
      // Only the score-space tensors are [T,T].  Context and projected are
      // attention-prefixed too, but remain model-space [T,D].
      const bool score = activationName == "attention_scores" ||
                         activationName == "attention_scaled" ||
                         activationName == "attention_masked" ||
                         activationName == "attention_probabilities";
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
    for (const char *name : {"dwq", "dwk", "dwv", "dwo", "dg1", "db1",
         "dg2", "db2", "dw1", "dw2"}) {
      const std::string gradientName(name);
      const bool norm = gradientName == "dg1" || gradientName == "db1" ||
                        gradientName == "dg2" || gradientName == "db2";
      const bool ff1Gradient = gradientName == "dw1";
      const bool ff2Gradient = gradientName == "dw2";
      record.gradients.push_back(add(prefix + name, QNN_TENSOR_TYPE_APP_READ,
          norm ? std::initializer_list<uint32_t>{dimension} :
          ff1Gradient ? std::initializer_list<uint32_t>{dimension, feedForwardDimension} :
          ff2Gradient ? std::initializer_list<uint32_t>{feedForwardDimension, dimension} :
                        std::initializer_list<uint32_t>{dimension, dimension}));
    }
    for (const char *name : {"drelu", "relu_mask", "dff1", "dln2",
         "dxhat2", "sum_dxhat2", "dxhat_xhat2", "dy_xhat2",
         "sum_dxhat_xhat2", "d_times_dxhat2", "first_diff2",
         "xhat_times_sum2", "bracket2", "inv_std_over_d2",
         "dresidual1_ln", "dresidual1", "dcontext", "dprobabilities",
         "dv", "softmax_product", "softmax_dot", "softmax_centered",
         "dscores", "dq_raw", "dk_raw", "dq", "dk", "dln1_q",
         "dln1_k", "dln1_v", "dln1_qk", "dln1", "dxhat1",
         "sum_dxhat1", "dxhat_xhat1", "dy_xhat1",
         "sum_dxhat_xhat1", "d_times_dxhat1", "first_diff1",
         "xhat_times_sum1", "bracket1", "inv_std_over_d1",
         "dinput_norm", "dinput"}) {
      const std::string backwardName(name);
      const bool score = backwardName == "dprobabilities" ||
                         backwardName == "softmax_product" ||
                         backwardName == "softmax_centered" ||
                         backwardName == "dscores";
      const bool row = backwardName == "sum_dxhat2" ||
                       backwardName == "sum_dxhat_xhat2" ||
                       backwardName == "inv_std_over_d2" ||
                       backwardName == "softmax_dot" ||
                       backwardName == "sum_dxhat1" ||
                       backwardName == "sum_dxhat_xhat1" ||
                       backwardName == "inv_std_over_d1";
      const bool ff = backwardName == "drelu" || backwardName == "relu_mask" ||
                      backwardName == "dff1";
      const Qnn_TensorType_t backwardType =
          diagnosticOutputs && backwardName == "dinput"
              ? QNN_TENSOR_TYPE_APP_READ
              : QNN_TENSOR_TYPE_NATIVE;
      record.backward.push_back(add(
          prefix + name, backwardType,
          score ? std::initializer_list<uint32_t>{tokens, tokens} :
          ff ? std::initializer_list<uint32_t>{tokens, feedForwardDimension} :
          row ? std::initializer_list<uint32_t>{tokens, 1} :
                std::initializer_list<uint32_t>{tokens, dimension},
          backwardName == "relu_mask" ? QNN_DATATYPE_BOOL_8 : QNN_DATATYPE_FLOAT_32));
    }
    if (numHeads > 1) {
      const uint32_t headDimension = dimension / numHeads;
      for (uint32_t head = 0; head < numHeads; ++head) {
        auto &headRecord = record.heads.emplace_back();
        const std::string headPrefix = prefix + "head_" +
            (head < 10 ? "0" : "") + std::to_string(head) + '_';
        headRecord.selector = add(headPrefix + "selector", QNN_TENSOR_TYPE_STATIC,
                                  {dimension, headDimension});
        headRecord.query = add(headPrefix + "query", QNN_TENSOR_TYPE_NATIVE,
                               {tokens, headDimension});
        headRecord.key = add(headPrefix + "key", QNN_TENSOR_TYPE_NATIVE,
                             {tokens, headDimension});
        headRecord.value = add(headPrefix + "value", QNN_TENSOR_TYPE_NATIVE,
                               {tokens, headDimension});
        headRecord.scores = add(headPrefix + "scores", QNN_TENSOR_TYPE_NATIVE,
                                {tokens, tokens});
        headRecord.scaled = add(headPrefix + "scaled", QNN_TENSOR_TYPE_NATIVE,
                                {tokens, tokens});
        headRecord.masked = add(headPrefix + "masked", QNN_TENSOR_TYPE_NATIVE,
                                {tokens, tokens});
        headRecord.probabilities = add(headPrefix + "probabilities",
            QNN_TENSOR_TYPE_APP_READ, {tokens, tokens});
        headRecord.context = add(headPrefix + "context", QNN_TENSOR_TYPE_NATIVE,
                                 {tokens, headDimension});
        headRecord.contextScatter = add(headPrefix + "context_scatter",
            QNN_TENSOR_TYPE_NATIVE, {tokens, dimension});
        headRecord.dContext = add(headPrefix + "dcontext", QNN_TENSOR_TYPE_NATIVE,
                                  {tokens, headDimension});
        headRecord.dProbabilities = add(headPrefix + "dprobabilities",
            QNN_TENSOR_TYPE_NATIVE, {tokens, tokens});
        headRecord.dValue = add(headPrefix + "dvalue", QNN_TENSOR_TYPE_NATIVE,
                                {tokens, headDimension});
        headRecord.softmaxProduct = add(headPrefix + "softmax_product",
            QNN_TENSOR_TYPE_NATIVE, {tokens, tokens});
        headRecord.softmaxDot = add(headPrefix + "softmax_dot",
            QNN_TENSOR_TYPE_NATIVE, {tokens, 1});
        headRecord.softmaxCentered = add(headPrefix + "softmax_centered",
            QNN_TENSOR_TYPE_NATIVE, {tokens, tokens});
        headRecord.dScores = add(headPrefix + "dscores", QNN_TENSOR_TYPE_NATIVE,
                                 {tokens, tokens});
        headRecord.dQueryRaw = add(headPrefix + "dquery_raw",
            QNN_TENSOR_TYPE_NATIVE, {tokens, headDimension});
        headRecord.dKeyRaw = add(headPrefix + "dkey_raw", QNN_TENSOR_TYPE_NATIVE,
                                 {tokens, headDimension});
        headRecord.dQuery = add(headPrefix + "dquery", QNN_TENSOR_TYPE_NATIVE,
                                {tokens, headDimension});
        headRecord.dKey = add(headPrefix + "dkey", QNN_TENSOR_TYPE_NATIVE,
                              {tokens, headDimension});
        headRecord.dQueryScatter = add(headPrefix + "dquery_scatter",
            QNN_TENSOR_TYPE_NATIVE, {tokens, dimension});
        headRecord.dKeyScatter = add(headPrefix + "dkey_scatter",
            QNN_TENSOR_TYPE_NATIVE, {tokens, dimension});
        headRecord.dValueScatter = add(headPrefix + "dvalue_scatter",
            QNN_TENSOR_TYPE_NATIVE, {tokens, dimension});
      }
      for (uint32_t accumulator = 0; accumulator + 1 < numHeads; ++accumulator) {
        const std::string accumulatorSuffix = std::string("head_accumulator_") +
            (accumulator < 10 ? "0" : "") + std::to_string(accumulator);
        record.headContextAccumulators.push_back(add(prefix + accumulatorSuffix +
            "_context", QNN_TENSOR_TYPE_NATIVE, {tokens, dimension}));
        record.headDqAccumulators.push_back(add(prefix + accumulatorSuffix +
            "_dq", QNN_TENSOR_TYPE_NATIVE, {tokens, dimension}));
        record.headDkAccumulators.push_back(add(prefix + accumulatorSuffix +
            "_dk", QNN_TENSOR_TYPE_NATIVE, {tokens, dimension}));
        record.headDvAccumulators.push_back(add(prefix + accumulatorSuffix +
            "_dv", QNN_TENSOR_TYPE_NATIVE, {tokens, dimension}));
      }
    }
  }
  // Binding order is an execution ABI: inputs first, then canonical flattened
  // parameters.  Keep target ahead of token_embedding so no name lookup or
  // incidental tensor-creation order can change it.
  g.appWriteRegistry = {oneHot, target, embedding};
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
  g.gradientRegistry.push_back(dEmbedding);
  g.appReadRegistry.push_back(logits);
  if (diagnosticOutputs) {
    g.appReadRegistry.insert(g.appReadRegistry.end(),
                             {g.layers.back().output, probabilities, dLogits,
                              dOutput});
  }
  for (const auto &record : g.layers) {
    for (uint32_t gradient : record.gradients)
      g.gradientRegistry.push_back(gradient);
  }
  g.gradientRegistry.push_back(dOutputProjection);
  const size_t expectedParameterCount = 2 + size_t(10) * numLayers;
  if (g.parameterRegistry.size() != expectedParameterCount ||
      std::unordered_set<uint32_t>(g.parameterRegistry.begin(),
                                   g.parameterRegistry.end()).size() !=
          g.parameterRegistry.size()) {
    return fail("generalized tiny training parameter registry is not canonical");
  }
  if (g.gradientRegistry.size() != g.parameterRegistry.size() ||
      std::unordered_set<uint32_t>(g.gradientRegistry.begin(),
                                   g.gradientRegistry.end()).size() !=
          g.gradientRegistry.size()) {
    return fail("generalized tiny training gradient registry is not canonical");
  }
  for (size_t index = 0; index < g.parameterRegistry.size(); ++index) {
    if (g.dimensions[g.parameterRegistry[index]] !=
        g.dimensions[g.gradientRegistry[index]]) {
      return fail("generalized tiny training gradient shape does not match parameter");
    }
  }
  g.appReadRegistry.insert(g.appReadRegistry.end(), g.gradientRegistry.begin(),
                           g.gradientRegistry.end());
  if (diagnosticOutputs) {
    for (const auto &record : g.layers)
      g.appReadRegistry.push_back(record.backward[DINPUT]);
  }
  // H=1 deliberately keeps the established L2/H1 29-buffer ABI.  H=2 adds
  // head probabilities only after the canonical diagnostics, gradients and
  // layer-input gradients, in layer-major/head-major order for CPU/HTP taps.
  if (numHeads > 1) {
    for (const auto &record : g.layers)
      for (const auto &head : record.heads)
        g.appReadRegistry.push_back(head.probabilities);
  }
  if (std::unordered_set<uint32_t>(g.appReadRegistry.begin(),
                                   g.appReadRegistry.end()).size() !=
      g.appReadRegistry.size()) {
    return fail("generalized tiny training APP_READ registry contains duplicates");
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
  g.zeroFfData.assign(size_t(tokens) * feedForwardDimension, 0.0f);
  const uint32_t headDimension = dimension / numHeads;
  const size_t selectorElements = numHeads > 1
      ? size_t(numLayers) * numHeads * dimension * headDimension
      : 0;
  if (selectorElements > std::numeric_limits<uint32_t>::max() / sizeof(float)) {
    return fail("generalized tiny training selector bytes exceed client-buffer limit");
  }
  g.selectorData.assign(selectorElements, 0.0f);
  size_t selectorOffset = 0;
  if (numHeads > 1) {
    for (auto &record : g.layers) {
      for (uint32_t head = 0; head < numHeads; ++head) {
        auto &headRecord = record.heads[head];
        headRecord.selectorOffset = selectorOffset;
        for (uint32_t d = 0; d < headDimension; ++d)
          g.selectorData[selectorOffset + size_t(head * headDimension + d) *
                                             headDimension + d] = 1.0f;
        selectorOffset += size_t(dimension) * headDimension;
      }
    }
  }
  if (selectorOffset != g.selectorData.size()) {
    return fail("generalized tiny training selector registry size mismatch");
  }
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
  g.tensors[zeroFf].v1.clientBuf = {
      g.zeroFfData.data(), uint32_t(g.zeroFfData.size() * sizeof(float))};
  for (const auto &record : g.layers) {
    for (const auto &head : record.heads) {
      const size_t selectorBytes = size_t(dimension) * headDimension * sizeof(float);
      if (head.selectorOffset > g.selectorData.size() ||
          selectorBytes > (g.selectorData.size() - head.selectorOffset) * sizeof(float)) {
        return fail("generalized tiny training selector client-buffer range");
      }
      g.tensors[head.selector].v1.clientBuf = {
          g.selectorData.data() + head.selectorOffset, uint32_t(selectorBytes)};
    }
  }
  g.tensors[dim].v1.clientBuf = {&g.dimensionValue, sizeof(g.dimensionValue)};
  g.tensors[inverseDim].v1.clientBuf = {&g.inverseDimensionValue, sizeof(g.inverseDimensionValue)};
  const std::string topologyName = numHeads == 1
      ? "l2h1"
      : "l" + std::to_string(numLayers) + "h" + std::to_string(numHeads);
  auto started = Clock::now();
  auto status = impl_->api.graphCreate(impl_->context,
      ("phonelm_tiny_transformer_training_" + topologyName).c_str(), nullptr, &g.graph);
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
  auto select = [&](const std::string &name, uint32_t condition, uint32_t yes,
                    uint32_t no, uint32_t out, const char *label) {
    Qnn_Tensor_t inputs[3] = {g.tensors[condition], g.tensors[yes],
                              g.tensors[no]};
    return ok(addSimpleNode(impl_->api, g.graph, name.c_str(),
                            QNN_OP_ELEMENT_WISE_SELECT, inputs, 3,
                            &g.tensors[out]), label);
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
  auto reduceSumLast = [&](const std::string &name, uint32_t source,
                           uint32_t out, const char *label) {
    Qnn_Param_t parameters[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    parameters[0].paramType = QNN_PARAMTYPE_TENSOR;
    parameters[0].name = QNN_OP_REDUCE_SUM_PARAM_AXES;
    parameters[0].tensorParam = g.tensors[lastAxis];
    parameters[1].paramType = QNN_PARAMTYPE_SCALAR;
    parameters[1].name = QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS;
    parameters[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    parameters[1].scalarParam.bool8Value = true;
    Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
    operation.v1.name = name.c_str();
    operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    operation.v1.typeName = QNN_OP_REDUCE_SUM;
    operation.v1.numOfParams = 2;
    operation.v1.params = parameters;
    operation.v1.numOfInputs = 1;
    operation.v1.inputTensors = &g.tensors[source];
    operation.v1.numOfOutputs = 1;
    operation.v1.outputTensors = &g.tensors[out];
    return ok(impl_->api.graphAddNode(g.graph, operation), label);
  };
  if (!matmul(topologyName + "_lm_embedding", oneHot, embedding, embedded, "embedding") ||
      !binary(topologyName + "_lm_position", QNN_OP_ELEMENT_WISE_ADD, embedded, position,
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
    const std::string prefix = topologyName + "_layer_0" + std::to_string(layer) + '_';
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
        !matmul(prefix + "v", cache(LN1), record.wv, cache(V), "v"))
      return fail(error);
    if (numHeads == 1) {
      if (!matmul(prefix + "scores", cache(Q), cache(K), cache(ATTENTION_SCORES),
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
                   cache(ATTENTION_CONTEXT), "context"))
        return fail(error);
    } else {
      for (uint32_t head = 0; head < numHeads; ++head) {
        const auto &headRecord = record.heads[head];
        const std::string headPrefix = prefix + "head_" +
            (head < 10 ? "0" : "") + std::to_string(head) + '_';
        if (!matmul(headPrefix + "query", cache(Q), headRecord.selector,
                     headRecord.query, "head query") ||
            !matmul(headPrefix + "key", cache(K), headRecord.selector,
                     headRecord.key, "head key") ||
            !matmul(headPrefix + "value", cache(V), headRecord.selector,
                     headRecord.value, "head value") ||
            !matmul(headPrefix + "scores", headRecord.query, headRecord.key,
                     headRecord.scores, "head scores",
                     QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
            !binary(headPrefix + "score_scale", QNN_OP_ELEMENT_WISE_MULTIPLY,
                    headRecord.scores, scale, headRecord.scaled, "head score scale") ||
            !binary(headPrefix + "score_mask", QNN_OP_ELEMENT_WISE_ADD,
                    headRecord.scaled, mask, headRecord.masked, "head score mask") ||
            !softmax(headPrefix + "softmax", headRecord.masked,
                     headRecord.probabilities, "head attention softmax") ||
            !matmul(headPrefix + "context", headRecord.probabilities,
                     headRecord.value, headRecord.context, "head context") ||
            !matmul(headPrefix + "context_scatter", headRecord.context,
                     headRecord.selector, headRecord.contextScatter,
                     "head context scatter", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))
          return fail(error);
      }
      uint32_t accumulated = record.heads.front().contextScatter;
      for (uint32_t head = 1; head < numHeads; ++head) {
        const uint32_t out = head + 1 == numHeads ? cache(ATTENTION_CONTEXT)
            : record.headContextAccumulators[head - 1];
        if (!binary(prefix + "head_context_sum_" + std::to_string(head),
                    QNN_OP_ELEMENT_WISE_ADD, accumulated,
                    record.heads[head].contextScatter, out, "head context sum"))
          return fail(error);
        accumulated = out;
      }
    }
    if (!matmul(prefix + "project", cache(ATTENTION_CONTEXT), record.wo,
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
  if (!matmul(topologyName + "_lm_logits", g.layers.back().output, outputProjection,
              logits, "logits") ||
       !softmax(topologyName + "_lm_softmax", logits, probabilities, "lm softmax") ||
       !binary(topologyName + "_lm_difference", QNN_OP_ELEMENT_WISE_SUBTRACT,
              probabilities, target, difference, "lm difference") ||
       !binary(topologyName + "_lm_dlogits", QNN_OP_ELEMENT_WISE_MULTIPLY, difference,
              gradientScale, dLogits, "lm dlogits") ||
       !matmul(topologyName + "_lm_doutput", dLogits, outputProjection, dOutput,
              "lm doutput", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
       !matmul(topologyName + "_lm_doutput_projection", g.layers.back().output, dLogits,
              dOutputProjection, "lm doutput projection",
              QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))
    return fail(error);
  for (int layer = int(numLayers) - 1; layer >= 0; --layer) {
    const auto &record = g.layers[size_t(layer)];
    const auto cache = [&](LayerActivation value) {
      return record.activations[size_t(value)];
    };
    const auto gradient = [&](LayerGradient value) {
      return record.gradients[size_t(value)];
    };
    const auto backward = [&](LayerBackward value) {
      return record.backward[size_t(value)];
    };
    const uint32_t upstream = layer + 1 == int(numLayers)
                                  ? dOutput
                                  : g.layers[size_t(layer + 1)].backward[DINPUT];
    const std::string prefix =
        topologyName + "_layer_0" + std::to_string(layer) + "_backward_";
    if (!matmul(prefix + "dw2", cache(RELU), upstream, gradient(DW2), "dw2",
                 QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "drelu", upstream, record.w2, backward(DRELU),
                 "drelu", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
        !binary(prefix + "relu_mask", QNN_OP_ELEMENT_WISE_GREATER, cache(FF1),
                zeroFf, backward(RELU_MASK), "relu mask") ||
        !select(prefix + "relu_select", backward(RELU_MASK), backward(DRELU),
                zeroFf, backward(DFF1), "relu select") ||
        !matmul(prefix + "dw1", cache(LN2), backward(DFF1), gradient(DW1),
                 "dw1", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "dln2", backward(DFF1), record.w1, backward(DLN2),
                 "dln2", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))
      return fail(error);
    if (!layerNorm.backward((prefix + "ln2").c_str(), int(backward(DLN2)),
                            int(record.gamma2), int(cache(LN2_XHAT)),
                            int(cache(LN2_INV)), int(backward(DXHAT2)),
                            int(backward(SUM_DXHAT2)),
                            int(backward(DXHAT_XHAT2)), int(backward(DY_XHAT2)),
                            int(backward(SUM_DXHAT_XHAT2)),
                            int(backward(D_TIMES_DXHAT2)),
                            int(backward(FIRST_DIFF2)),
                            int(backward(XHAT_TIMES_SUM2)), int(backward(BRACKET2)),
                            int(backward(INV_STD_OVER_D2)),
                            int(backward(DRESIDUAL1_LN)), int(gradient(DG2)),
                            int(gradient(DB2))))
      return fail(error);
    if (!binary(prefix + "dresidual1", QNN_OP_ELEMENT_WISE_ADD, upstream,
                backward(DRESIDUAL1_LN), backward(DRESIDUAL1), "dresidual1") ||
        !matmul(prefix + "dwo", cache(ATTENTION_CONTEXT), backward(DRESIDUAL1),
                 gradient(DWO), "dwo", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "dcontext", backward(DRESIDUAL1), record.wo,
                 backward(DCONTEXT), "dcontext", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))
      return fail(error);
    if (numHeads == 1) {
      if (!matmul(prefix + "dprobabilities", backward(DCONTEXT), cache(V),
                   backward(DPROBABILITIES), "dprobabilities",
                   QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
          !matmul(prefix + "dv", cache(ATTENTION_PROBABILITIES), backward(DCONTEXT),
                   backward(DV), "dv", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
          !binary(prefix + "softmax_product", QNN_OP_ELEMENT_WISE_MULTIPLY,
                  backward(DPROBABILITIES), cache(ATTENTION_PROBABILITIES),
                  backward(SOFTMAX_PRODUCT), "softmax product") ||
          !reduceSumLast(prefix + "softmax_dot", backward(SOFTMAX_PRODUCT),
                         backward(SOFTMAX_DOT), "softmax dot") ||
          !binary(prefix + "softmax_centered", QNN_OP_ELEMENT_WISE_SUBTRACT,
                  backward(DPROBABILITIES), backward(SOFTMAX_DOT),
                  backward(SOFTMAX_CENTERED), "softmax centered") ||
          !binary(prefix + "dscores", QNN_OP_ELEMENT_WISE_MULTIPLY,
                  cache(ATTENTION_PROBABILITIES), backward(SOFTMAX_CENTERED),
                  backward(DSCORES), "dscores") ||
          !matmul(prefix + "dq_raw", backward(DSCORES), cache(K),
                   backward(DQ_RAW), "dq raw") ||
          !matmul(prefix + "dk_raw", backward(DSCORES), cache(Q),
                   backward(DK_RAW), "dk raw", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
          !binary(prefix + "dq", QNN_OP_ELEMENT_WISE_MULTIPLY, backward(DQ_RAW),
                  scale, backward(DQ), "dq") ||
          !binary(prefix + "dk", QNN_OP_ELEMENT_WISE_MULTIPLY, backward(DK_RAW),
                  scale, backward(DK), "dk"))
        return fail(error);
    } else {
      for (uint32_t head = 0; head < numHeads; ++head) {
        const auto &headRecord = record.heads[head];
        const std::string headPrefix = prefix + "head_" +
            (head < 10 ? "0" : "") + std::to_string(head) + '_';
        if (!matmul(headPrefix + "dcontext", backward(DCONTEXT),
                     headRecord.selector, headRecord.dContext, "head dcontext") ||
            !matmul(headPrefix + "dprobabilities", headRecord.dContext,
                     headRecord.value, headRecord.dProbabilities,
                     "head dprobabilities", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
            !matmul(headPrefix + "dvalue", headRecord.probabilities,
                     headRecord.dContext, headRecord.dValue, "head dvalue",
                     QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
            !binary(headPrefix + "softmax_product", QNN_OP_ELEMENT_WISE_MULTIPLY,
                    headRecord.dProbabilities, headRecord.probabilities,
                    headRecord.softmaxProduct, "head softmax product") ||
            !reduceSumLast(headPrefix + "softmax_dot", headRecord.softmaxProduct,
                           headRecord.softmaxDot, "head softmax dot") ||
            !binary(headPrefix + "softmax_centered", QNN_OP_ELEMENT_WISE_SUBTRACT,
                    headRecord.dProbabilities, headRecord.softmaxDot,
                    headRecord.softmaxCentered, "head softmax centered") ||
            !binary(headPrefix + "dscores", QNN_OP_ELEMENT_WISE_MULTIPLY,
                    headRecord.probabilities, headRecord.softmaxCentered,
                    headRecord.dScores, "head dscores") ||
            !matmul(headPrefix + "dquery_raw", headRecord.dScores, headRecord.key,
                     headRecord.dQueryRaw, "head dquery raw") ||
            !matmul(headPrefix + "dkey_raw", headRecord.dScores, headRecord.query,
                     headRecord.dKeyRaw, "head dkey raw",
                     QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
            !binary(headPrefix + "dquery", QNN_OP_ELEMENT_WISE_MULTIPLY,
                    headRecord.dQueryRaw, scale, headRecord.dQuery, "head dquery") ||
            !binary(headPrefix + "dkey", QNN_OP_ELEMENT_WISE_MULTIPLY,
                    headRecord.dKeyRaw, scale, headRecord.dKey, "head dkey") ||
            !matmul(headPrefix + "dquery_scatter", headRecord.dQuery,
                     headRecord.selector, headRecord.dQueryScatter,
                     "head dquery scatter", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
            !matmul(headPrefix + "dkey_scatter", headRecord.dKey,
                     headRecord.selector, headRecord.dKeyScatter,
                     "head dkey scatter", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
            !matmul(headPrefix + "dvalue_scatter", headRecord.dValue,
                     headRecord.selector, headRecord.dValueScatter,
                     "head dvalue scatter", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))
          return fail(error);
      }
      struct HeadGradientChain {
        const char *name;
        const std::vector<uint32_t> *accumulators;
        uint32_t target;
        uint32_t Impl::GeneralizedTinyTransformerTrainingGraph::HeadRegistry::*member;
      };
      const std::array<HeadGradientChain, 3> chains{{
          {"dq", &record.headDqAccumulators, backward(DQ),
           &Impl::GeneralizedTinyTransformerTrainingGraph::HeadRegistry::dQueryScatter},
          {"dk", &record.headDkAccumulators, backward(DK),
           &Impl::GeneralizedTinyTransformerTrainingGraph::HeadRegistry::dKeyScatter},
          {"dv", &record.headDvAccumulators, backward(DV),
           &Impl::GeneralizedTinyTransformerTrainingGraph::HeadRegistry::dValueScatter},
      }};
      for (const auto &chain : chains) {
        uint32_t accumulated = record.heads.front().*chain.member;
        for (uint32_t head = 1; head < numHeads; ++head) {
          const uint32_t out = head + 1 == numHeads ? chain.target
              : (*chain.accumulators)[head - 1];
          if (!binary(prefix + "head_" + chain.name + "_sum_" +
                      std::to_string(head), QNN_OP_ELEMENT_WISE_ADD, accumulated,
                      record.heads[head].*chain.member, out, "head gradient sum"))
            return fail(error);
          accumulated = out;
        }
      }
    }
    if (!matmul(prefix + "dwq", cache(LN1), backward(DQ), gradient(DWQ),
                 "dwq", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "dwk", cache(LN1), backward(DK), gradient(DWK),
                 "dwk", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "dwv", cache(LN1), backward(DV), gradient(DWV),
                 "dwv", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0) ||
        !matmul(prefix + "dln1_q", backward(DQ), record.wq, backward(DLN1_Q),
                 "dln1 q", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
        !matmul(prefix + "dln1_k", backward(DK), record.wk, backward(DLN1_K),
                 "dln1 k", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
        !matmul(prefix + "dln1_v", backward(DV), record.wv, backward(DLN1_V),
                 "dln1 v", QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1) ||
        !binary(prefix + "dln1_qk", QNN_OP_ELEMENT_WISE_ADD, backward(DLN1_Q),
                backward(DLN1_K), backward(DLN1_QK), "dln1 qk") ||
        !binary(prefix + "dln1", QNN_OP_ELEMENT_WISE_ADD, backward(DLN1_QK),
                backward(DLN1_V), backward(DLN1), "dln1"))
      return fail(error);
    if (!layerNorm.backward((prefix + "ln1").c_str(), int(backward(DLN1)),
                            int(record.gamma1), int(cache(LN1_XHAT)),
                            int(cache(LN1_INV)), int(backward(DXHAT1)),
                            int(backward(SUM_DXHAT1)),
                            int(backward(DXHAT_XHAT1)), int(backward(DY_XHAT1)),
                            int(backward(SUM_DXHAT_XHAT1)),
                            int(backward(D_TIMES_DXHAT1)),
                            int(backward(FIRST_DIFF1)),
                            int(backward(XHAT_TIMES_SUM1)), int(backward(BRACKET1)),
                            int(backward(INV_STD_OVER_D1)), int(backward(DINPUT_NORM)),
                            int(gradient(DG1)), int(gradient(DB1))))
      return fail(error);
    if (!binary(prefix + "dinput", QNN_OP_ELEMENT_WISE_ADD,
                backward(DRESIDUAL1), backward(DINPUT_NORM), backward(DINPUT),
                "dinput"))
      return fail(error);
  }
  if (!matmul(topologyName + "_lm_dembedding", oneHot,
              g.layers.front().backward[DINPUT], dEmbedding, "dembedding",
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
  const std::string graphLabel = "L" + std::to_string(numLayers) + "H" +
      std::to_string(numHeads) + "_EXPLICIT_FORWARD_BACKWARD";
  diagnostics_ += "tiny_transformer_training_graph=" + graphLabel + '\n' +
                  "tiny_transformer_training_graph_count=1\n"
                  "tiny_transformer_training_forward_layers=" +
                  std::to_string(numLayers) + '\n' +
                  "tiny_transformer_training_forward_heads=" +
                  std::to_string(numHeads) + '\n' +
                  "tiny_transformer_training_parameter_registry_count=" +
                  std::to_string(g.parameterRegistry.size()) + '\n' +
                  "tiny_transformer_training_gradient_registry_count=" +
                  std::to_string(g.gradientRegistry.size()) + '\n' +
                  "tiny_transformer_training_diagnostic_outputs=" +
                  std::string(g.diagnosticOutputs ? "true\n" : "false\n") +
                  "tiny_transformer_training_app_read_registry_count=" +
                  std::to_string(g.appReadRegistry.size()) + '\n' +
                  "tiny_transformer_training_sgd_next=false\n"
                  "tiny_transformer_training_learning_rate=REQUIRES_ZERO_QNN_ADAM_SEPARATE\n" +
                  "tiny_transformer_training_tensor_create_count=" +
                  std::to_string(g.tensorCreateSuccessCount) + '\n' +
                  "tiny_transformer_training_node_count=" +
                  std::to_string(g.graphAddNodeSuccessCount) + '\n';
  return true;
}
