// The fixed L1/H1 graph and the dynamic graph use the same LayerNorm
// subgraphs.  Keep this builder independent from the fixed tensor enum: a
// generalized layer supplies its own tensor IDs, which prevents a later layer
// from accidentally reusing layer zero temporaries.
struct TransformerTrainingLayerNormBuilder {
  QNN_INTERFACE_VER_TYPE &api;
  Qnn_GraphHandle_t graph;
  Qnn_Tensor_t *tensors;
  int lastAxis;
  int rowAxis;
  int centeredScale;
  int epsilonScaled;
  int dimension;
  int inverseDimension;
  std::string &error;
  uint32_t &nodeCount;

  bool ok(Qnn_ErrorHandle_t status, const char *label) {
    if (status == QNN_SUCCESS) {
      ++nodeCount;
      return true;
    }
    error = std::string("tiny training ") + label + '=' +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  bool binary(const std::string &name, const char *type, int a, int b, int out,
              const char *label) {
    Qnn_Tensor_t input[2] = {tensors[a], tensors[b]};
    return ok(addSimpleNode(api, graph, name.c_str(), type, input, 2,
                            &tensors[out]), label);
  }
  bool unary(const std::string &name, const char *type, int a, int out,
             const char *label) {
    return ok(addSimpleNode(api, graph, name.c_str(), type, &tensors[a], 1,
                            &tensors[out]), label);
  }
  bool reduce(const std::string &name, const char *type, const char *axesName,
              const char *keepName, int input, int axes, bool keep, int out,
              const char *label) {
    Qnn_Param_t parameters[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    parameters[0].paramType = QNN_PARAMTYPE_TENSOR;
    parameters[0].name = axesName;
    parameters[0].tensorParam = tensors[axes];
    parameters[1].paramType = QNN_PARAMTYPE_SCALAR;
    parameters[1].name = keepName;
    parameters[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    parameters[1].scalarParam.bool8Value = keep;
    Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
    operation.v1.name = name.c_str();
    operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    operation.v1.typeName = type;
    operation.v1.numOfParams = 2;
    operation.v1.params = parameters;
    operation.v1.numOfInputs = 1;
    operation.v1.inputTensors = &tensors[input];
    operation.v1.numOfOutputs = 1;
    operation.v1.outputTensors = &tensors[out];
    return ok(api.graphAddNode(graph, operation), label);
  }
  bool forward(const char *prefix, int x, int gamma, int beta, int mean,
               int centered, int centeredScaled, int square, int variance,
               int varianceEpsilon, int inverseScaled, int inverse, int xhat,
               int xhatGamma, int output) {
    const std::string p(prefix);
    return reduce(p + "_mean", QNN_OP_REDUCE_MEAN,
                  QNN_OP_REDUCE_MEAN_PARAM_AXES,
                  QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, x, lastAxis, true, mean,
                  "ln mean") &&
           binary(p + "_center", QNN_OP_ELEMENT_WISE_SUBTRACT, x, mean,
                  centered, "ln center") &&
           binary(p + "_center_scale", QNN_OP_ELEMENT_WISE_MULTIPLY, centered,
                  centeredScale, centeredScaled, "ln center scale") &&
           binary(p + "_square", QNN_OP_ELEMENT_WISE_MULTIPLY,
                  centeredScaled, centeredScaled, square, "ln square") &&
           reduce(p + "_var", QNN_OP_REDUCE_MEAN,
                  QNN_OP_REDUCE_MEAN_PARAM_AXES,
                  QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, square, lastAxis, true,
                  variance, "ln var") &&
           binary(p + "_eps", QNN_OP_ELEMENT_WISE_ADD, variance,
                  epsilonScaled, varianceEpsilon, "ln eps") &&
           unary(p + "_rsqrt", QNN_OP_ELEMENT_WISE_RSQRT, varianceEpsilon,
                 inverseScaled, "ln rsqrt") &&
           binary(p + "_unscale", QNN_OP_ELEMENT_WISE_MULTIPLY,
                  inverseScaled, centeredScale, inverse, "ln unscale") &&
           binary(p + "_xhat", QNN_OP_ELEMENT_WISE_MULTIPLY, centered,
                  inverse, xhat, "ln xhat") &&
           binary(p + "_gamma", QNN_OP_ELEMENT_WISE_MULTIPLY, xhat, gamma,
                  xhatGamma, "ln gamma") &&
           binary(p + "_beta", QNN_OP_ELEMENT_WISE_ADD, xhatGamma, beta,
                  output, "ln beta");
  }
  bool backward(const char *prefix, int dOutput, int gamma, int xhat,
                int inverseStd, int dXhat, int sumDXhat, int dXhatXhat,
                int dYXhat, int sumDXhatXhat, int dimensionTimesDXhat,
                int firstDifference, int xhatTimesSum, int bracket,
                int inverseStdOverDimension, int dInput, int dGamma,
                int dBeta) {
    const std::string p(prefix);
    return binary(p + "_dxhat", QNN_OP_ELEMENT_WISE_MULTIPLY, dOutput,
                  gamma, dXhat, "lnb dxhat") &&
           reduce(p + "_sum", QNN_OP_REDUCE_SUM,
                  QNN_OP_REDUCE_SUM_PARAM_AXES,
                  QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dXhat, lastAxis, true,
                  sumDXhat, "lnb sum") &&
           binary(p + "_prod", QNN_OP_ELEMENT_WISE_MULTIPLY, dXhat, xhat,
                  dXhatXhat, "lnb prod") &&
           reduce(p + "_sumprod", QNN_OP_REDUCE_SUM,
                  QNN_OP_REDUCE_SUM_PARAM_AXES,
                  QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dXhatXhat, lastAxis,
                  true, sumDXhatXhat, "lnb sumprod") &&
           binary(p + "_ddx", QNN_OP_ELEMENT_WISE_MULTIPLY, dXhat,
                  dimension, dimensionTimesDXhat, "lnb ddx") &&
           binary(p + "_first", QNN_OP_ELEMENT_WISE_SUBTRACT,
                  dimensionTimesDXhat, sumDXhat, firstDifference,
                  "lnb first") &&
           binary(p + "_xsum", QNN_OP_ELEMENT_WISE_MULTIPLY, xhat,
                  sumDXhatXhat, xhatTimesSum, "lnb xsum") &&
           binary(p + "_bracket", QNN_OP_ELEMENT_WISE_SUBTRACT,
                  firstDifference, xhatTimesSum, bracket, "lnb bracket") &&
           binary(p + "_invd", QNN_OP_ELEMENT_WISE_MULTIPLY, inverseStd,
                  inverseDimension, inverseStdOverDimension, "lnb invd") &&
           binary(p + "_dx", QNN_OP_ELEMENT_WISE_MULTIPLY, bracket,
                  inverseStdOverDimension, dInput, "lnb dx") &&
           binary(p + "_dyx", QNN_OP_ELEMENT_WISE_MULTIPLY, dOutput, xhat,
                  dYXhat, "lnb dyx") &&
           reduce(p + "_dg", QNN_OP_REDUCE_SUM,
                  QNN_OP_REDUCE_SUM_PARAM_AXES,
                  QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dYXhat,
                  /* row axis is always tensor slot supplied by caller */
                  rowAxis, false, dGamma, "lnb dg") &&
           reduce(p + "_db", QNN_OP_REDUCE_SUM,
                  QNN_OP_REDUCE_SUM_PARAM_AXES,
                  QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dOutput, rowAxis, false,
                  dBeta, "lnb db");
  }
};

bool Runtime::prepareTinyTransformerTraining(
    uint32_t tokens, uint32_t dimension, uint32_t feedForwardDimension,
    float epsilon, bool diagnosticOutputs, std::string &error,
    uint32_t vocabularySize, TinyTransformerTrainingVariant variant,
    TinyTransformerTrainingTapSet tapSet, uint32_t numLayers,
    uint32_t numHeads) {
  if (!impl_ || !impl_->context || !tokens || !dimension ||
      !feedForwardDimension || !numLayers || !numHeads ||
      dimension % numHeads != 0 || !std::isfinite(epsilon) || epsilon <= 0) {
    error = "tiny training invalid runtime or shape";
    return false;
  }
  using G = Impl::TinyTransformerTrainingGraph;
  auto &g = impl_->tinyTransformerTraining;
  const auto generalizedState = impl_->generalizedTinyTransformerTraining.state;
  if (generalizedState != Impl::GeneralizedTinyTransformerTrainingGraph::State::INACTIVE ||
      g.graph != nullptr) {
    error = "tiny training reprepare or L1/L2 mode transition is unavailable";
    return false;
  }
  // Hand all non-baseline requests to the stateful builder before touching
  // fixed-L1 state. A generalized prepare failure must therefore leave a
  // FAILED state that executeTinyTransformerTraining cannot reinterpret as L1.
  if (numLayers != 1 || numHeads != 1) {
    return prepareTinyTransformerTrainingGeneralized(
        tokens, dimension, feedForwardDimension, epsilon, diagnosticOutputs,
        error, vocabularySize, variant, tapSet, numLayers, numHeads);
  }
  if (!std::isfinite(options_.tinyTransformerCenteredScale) ||
      options_.tinyTransformerCenteredScale <= 0) {
    error = "tiny training invalid centered scale";
    return false;
  }
  g.centeredScaleData = options_.tinyTransformerCenteredScale;
  const double epsilonScaled =
      double(epsilon) * double(g.centeredScaleData) *
      double(g.centeredScaleData);
  if (!std::isfinite(epsilonScaled) || epsilonScaled <= 0.0 ||
      epsilonScaled > double(std::numeric_limits<float>::max()) ||
      !std::isfinite(float(epsilonScaled)) || float(epsilonScaled) <= 0.0f) {
    error = "tiny training centered scale makes epsilon unrepresentable";
    return false;
  }
  g.tokens = tokens;
  g.dimension = dimension;
  g.feedForwardDimension = feedForwardDimension;
  g.numLayers = numLayers;
  g.numHeads = numHeads;
  g.headDimension = dimension / numHeads;
  g.diagnosticOutputs = diagnosticOutputs;
  g.vocabularySize = vocabularySize;
  g.sourceTensorCreateSuccessCount = 0;
  g.sourceGraphAddNodeSuccessCount = 0;
  g.lastInputTensorCount = 0;
  g.lastOutputTensorCount = 0;
  g.lastLearningRateBytesUnchanged = true;
  g.languageModel = vocabularySize != 0;
  g.variant = variant;
  g.tapSet = tapSet;
  switch (variant) {
  case TinyTransformerTrainingVariant::FULL:
  case TinyTransformerTrainingVariant::STOP_AFTER_DINPUT:
  case TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING:
    break;
  default:
    error = "tiny training invalid graph variant";
    return false;
  }
  if (variant != TinyTransformerTrainingVariant::FULL &&
      !g.languageModel) {
    error = "tiny training graph variant requires language model";
    return false;
  }
  switch (tapSet) {
  case TinyTransformerTrainingTapSet::NONE:
  case TinyTransformerTrainingTapSet::DSCORES_ONLY:
  case TinyTransformerTrainingTapSet::DPROB_DSCORES:
  case TinyTransformerTrainingTapSet::BACKWARD_REGIONS:
  case TinyTransformerTrainingTapSet::LAYERNORM1:
  case TinyTransformerTrainingTapSet::ALL_INTERNAL:
    break;
  default:
    error = "tiny training invalid tap set";
    return false;
  }
  if (tapSet != TinyTransformerTrainingTapSet::NONE && !g.languageModel) {
    error = "tiny training tap set requires language model";
    return false;
  }
  g.activationDims[0] = tokens;
  g.oneHotDims[0] = tokens;
  g.oneHotDims[1] = vocabularySize;
  g.embeddingDims[0] = vocabularySize;
  g.embeddingDims[1] = dimension;
  g.logitsDims[0] = tokens;
  g.logitsDims[1] = vocabularySize;
  g.outputProjectionDims[0] = dimension;
  g.outputProjectionDims[1] = vocabularySize;
  g.activationDims[1] = dimension;
  g.scoreDims[0] = tokens;
  g.scoreDims[1] = tokens;
  g.ffDims[0] = tokens;
  g.ffDims[1] = feedForwardDimension;
  g.ddDims[0] = dimension;
  g.ddDims[1] = dimension;
  g.dfDims[0] = dimension;
  g.dfDims[1] = feedForwardDimension;
  g.fdDims[0] = feedForwardDimension;
  g.fdDims[1] = dimension;
  g.normDims[0] = dimension;
  g.rowDims[0] = tokens;
  g.rowDims[1] = 1;
  g.scaleData = 1.0f / std::sqrt(float(dimension));
  g.epsilonScaledData = float(epsilonScaled);
  g.dimensionData = float(dimension);
  g.inverseDimensionData = 1.0f / float(dimension);
  g.gradientScaleData = g.languageModel ? 1.0f / float(tokens) :
                                            2.0f / float(tokens * dimension);
  g.maskData.assign(size_t(tokens) * tokens, 0);
  for (uint32_t r = 0; r < tokens; ++r)
    for (uint32_t c = r + 1; c < tokens; ++c)
      g.maskData[r * tokens + c] = -10000.0f;
  g.zeroFfData.assign(size_t(tokens) * feedForwardDimension, 0);
  g.positionData.assign(size_t(tokens) * dimension, 0.0f);
  if (g.languageModel) {
    for (uint32_t t = 0; t < tokens; ++t)
      for (uint32_t d = 0; d < dimension; ++d) {
        const float frequency = std::pow(10000.0f, -float(d & ~1u) / float(dimension));
        g.positionData[size_t(t) * dimension + d] =
            ((d & 1u) ? std::cos(float(t) * frequency)
                      : std::sin(float(t) * frequency)) * 0.05f;
      }
  }
  for (int i = 0; i < G::TENSOR_COUNT; ++i) {
    Qnn_Tensor_t initialized = QNN_TENSOR_INIT;
    g.tensors[i] = initialized;
    // The baseline is logical layer zero.  Give every layer-local tensor a
    // stable qualified name now so the dynamic registry uses the same naming
    // convention when it adds layer_01 and head-specific tensors.
    g.names[i] = "layer_00_tensor_" + std::to_string(i);
  }
  auto nameLayerParameter = [&](int id, const char *name) {
    g.names[id] = std::string("layer_00_") + name;
  };
  nameLayerParameter(G::G1, "norm1_gamma");
  nameLayerParameter(G::B1, "norm1_beta");
  nameLayerParameter(G::WQ, "wq");
  nameLayerParameter(G::WK, "wk");
  nameLayerParameter(G::WV, "wv");
  nameLayerParameter(G::WO, "wo");
  nameLayerParameter(G::G2, "norm2_gamma");
  nameLayerParameter(G::B2, "norm2_beta");
  nameLayerParameter(G::W1, "ffn_w1");
  nameLayerParameter(G::W2, "ffn_w2");
  g.names[G::ONE_HOT] = "token_one_hot";
  g.names[G::EMBEDDING] = "token_embedding";
  g.names[G::OUTPUT_PROJECTION] = "output_projection";
  std::unordered_set<std::string> declaredTensorNames;
  for (const auto &name : g.names) {
    if (!declaredTensorNames.insert(name).second) {
      error = "tiny training duplicate tensor name: " + name;
      return false;
    }
  }
  auto tapType = [&](int id, Qnn_TensorType_t type) {
    const bool allInternal =
        g.tapSet == TinyTransformerTrainingTapSet::ALL_INTERNAL &&
        (id == G::EMBEDDED_TOKEN || id == G::X ||
         id == G::MEAN1 || id == G::CENTERED1 || id == G::CENTERED_S1 ||
         id == G::SQUARE1 || id == G::VAR1 || id == G::VAR_EPS1 ||
         id == G::INV_STD_S1 || id == G::INV_STD1 || id == G::XHAT1 ||
         id == G::XHAT_G1 || id == G::LN1 || id == G::Q || id == G::K ||
         id == G::V || id == G::SCORES || id == G::SCALED_SCORES ||
         id == G::MASKED_SCORES || id == G::PROBABILITIES ||
         id == G::CONTEXT || id == G::PROJECTED || id == G::RESIDUAL1 ||
         id == G::MEAN2 || id == G::CENTERED2 || id == G::CENTERED_S2 ||
         id == G::SQUARE2 || id == G::VAR2 || id == G::VAR_EPS2 ||
         id == G::INV_STD_S2 || id == G::INV_STD2 || id == G::XHAT2 ||
         id == G::XHAT_G2 || id == G::LN2 || id == G::FF1 || id == G::RELU ||
         id == G::FF2 || id == G::OUTPUT || id == G::LM_PROBABILITIES ||
         id == G::LM_DIFFERENCE || id == G::DLOGITS || id == G::DOUTPUT ||
         id == G::DOUTPUT_PROJECTION || id == G::DW2 || id == G::DRELU ||
         id == G::DFF1 || id == G::DW1 || id == G::DLN2 ||
         id == G::DXHAT2 || id == G::SUM_DXHAT2 ||
         id == G::DXHAT_XHAT2 || id == G::DY_XHAT2 ||
         id == G::SUM_DXHAT_XHAT2 || id == G::D_TIMES_DXHAT2 ||
         id == G::FIRST_DIFF2 || id == G::XHAT_TIMES_SUM2 ||
         id == G::BRACKET2 || id == G::INV_STD_OVER_D2 ||
         id == G::DRESIDUAL1_LN || id == G::DG2 || id == G::DB2 ||
         id == G::DRESIDUAL1 || id == G::DWO || id == G::DCONTEXT ||
         id == G::DPROBABILITIES || id == G::DV ||
         id == G::SOFTMAX_PRODUCT || id == G::SOFTMAX_DOT ||
         id == G::SOFTMAX_CENTERED || id == G::DSCORES ||
         id == G::DQ_RAW || id == G::DK_RAW || id == G::DQ || id == G::DK ||
         id == G::DWQ || id == G::DWK || id == G::DWV ||
         id == G::DLN1_Q || id == G::DLN1_K || id == G::DLN1_V ||
         id == G::DLN1_QK || id == G::DLN1 || id == G::DXHAT1 ||
         id == G::SUM_DXHAT1 || id == G::DXHAT_XHAT1 ||
         id == G::DY_XHAT1 || id == G::SUM_DXHAT_XHAT1 ||
         id == G::D_TIMES_DXHAT1 || id == G::FIRST_DIFF1 ||
         id == G::XHAT_TIMES_SUM1 || id == G::BRACKET1 ||
         id == G::INV_STD_OVER_D1 || id == G::DINPUT_NORM ||
         id == G::DG1 || id == G::DB1 || id == G::DINPUT ||
         (id == G::DEMBEDDING &&
          variant != TinyTransformerTrainingVariant::STOP_AFTER_DINPUT));
    const bool selected =
        allInternal ||
        (g.tapSet == TinyTransformerTrainingTapSet::DSCORES_ONLY &&
         id == G::DSCORES) ||
        (g.tapSet == TinyTransformerTrainingTapSet::DPROB_DSCORES &&
         (id == G::DPROBABILITIES || id == G::DSCORES)) ||
        (g.tapSet == TinyTransformerTrainingTapSet::BACKWARD_REGIONS &&
         (id == G::DRESIDUAL1 || id == G::DSCORES || id == G::DLN1 ||
          id == G::DINPUT_NORM)) ||
        (g.tapSet == TinyTransformerTrainingTapSet::LAYERNORM1 &&
         (id == G::DLN1 || id == G::DXHAT1 || id == G::SUM_DXHAT1 ||
          id == G::DXHAT_XHAT1 || id == G::SUM_DXHAT_XHAT1 ||
          id == G::DINPUT_NORM));
    return selected ? QNN_TENSOR_TYPE_APP_READ : type;
  };
  auto make = [&](int i, Qnn_TensorType_t type, uint32_t *d, uint32_t rank = 2,
                  Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32) {
    tensor(g.tensors[i], g.names[i].c_str(), tapType(i, type), d, dt, rank);
  };
  auto many = [&](std::initializer_list<int> ids, Qnn_TensorType_t type,
                  uint32_t *d, uint32_t rank = 2,
                  Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32) {
    for (int i : ids)
      make(i, type, d, rank, dt);
  };
  if (g.languageModel) {
    make(G::X, diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ :
                                  QNN_TENSOR_TYPE_NATIVE, g.activationDims);
    make(G::TARGET, QNN_TENSOR_TYPE_APP_WRITE, g.logitsDims);
    make(G::ONE_HOT, QNN_TENSOR_TYPE_APP_WRITE, g.oneHotDims);
    make(G::EMBEDDING, QNN_TENSOR_TYPE_APP_WRITE, g.embeddingDims);
    make(G::POSITION, QNN_TENSOR_TYPE_STATIC, g.activationDims);
    make(G::EMBEDDED_TOKEN, QNN_TENSOR_TYPE_NATIVE, g.activationDims);
    make(G::OUTPUT_PROJECTION, QNN_TENSOR_TYPE_APP_WRITE,
         g.outputProjectionDims);
  } else {
    many({G::X, G::TARGET}, QNN_TENSOR_TYPE_APP_WRITE, g.activationDims);
  }
  many({G::G1, G::B1, G::G2, G::B2}, QNN_TENSOR_TYPE_APP_WRITE, g.normDims, 1);
  many({G::WQ, G::WK, G::WV, G::WO}, QNN_TENSOR_TYPE_APP_WRITE, g.ddDims);
  make(G::W1, QNN_TENSOR_TYPE_APP_WRITE, g.dfDims);
  make(G::W2, QNN_TENSOR_TYPE_APP_WRITE, g.fdDims);
  make(G::LR, variant == TinyTransformerTrainingVariant::FULL
                  ? QNN_TENSOR_TYPE_APP_WRITE
                  : QNN_TENSOR_TYPE_NATIVE,
       g.scalarDims);
  make(G::MASK, QNN_TENSOR_TYPE_STATIC, g.scoreDims);
  many({G::SCALE, G::EPS_SCALED, G::CENTER_SCALE, G::DIMENSION,
        G::INV_DIMENSION, G::GRAD_SCALE},
       QNN_TENSOR_TYPE_STATIC, g.scalarDims);
  make(G::LAST_AXIS, QNN_TENSOR_TYPE_STATIC, g.axisOneDims, 1,
       QNN_DATATYPE_UINT_32);
  make(G::ROW_AXIS, QNN_TENSOR_TYPE_STATIC, g.axisOneDims, 1,
       QNN_DATATYPE_UINT_32);
  make(G::ALL_AXES, QNN_TENSOR_TYPE_STATIC, g.axisTwoDims, 1,
       QNN_DATATYPE_UINT_32);
  make(G::ZERO_FF, QNN_TENSOR_TYPE_STATIC, g.ffDims);
  many({G::MEAN1, G::VAR1, G::VAR_EPS1, G::INV_STD_S1, G::INV_STD1, G::MEAN2,
        G::VAR2, G::VAR_EPS2, G::INV_STD_S2, G::INV_STD2, G::SUM_DXHAT2,
        G::SUM_DXHAT_XHAT2, G::INV_STD_OVER_D2, G::SUM_DXHAT1,
        G::SUM_DXHAT_XHAT1, G::INV_STD_OVER_D1, G::SOFTMAX_DOT},
       QNN_TENSOR_TYPE_NATIVE, g.rowDims);
  many({G::SCORES, G::SCALED_SCORES, G::MASKED_SCORES, G::PROBABILITIES,
        G::DPROBABILITIES, G::SOFTMAX_PRODUCT,
        G::SOFTMAX_CENTERED, G::DSCORES},
       QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  many({G::FF1, G::RELU, G::DRELU, G::DFF1}, QNN_TENSOR_TYPE_NATIVE, g.ffDims);
  make(G::RELU_MASK, QNN_TENSOR_TYPE_NATIVE, g.ffDims, 2, QNN_DATATYPE_BOOL_8);
  many({G::CENTERED1,
        G::CENTERED_S1,
        G::SQUARE1,
        G::XHAT1,
        G::XHAT_G1,
        G::LN1,
        G::Q,
        G::K,
        G::V,
        G::CONTEXT,
        G::PROJECTED,
        G::RESIDUAL1,
        G::CENTERED2,
        G::CENTERED_S2,
        G::SQUARE2,
        G::XHAT2,
        G::XHAT_G2,
        G::LN2,
        G::FF2,
        G::ERROR,
        G::SQUARED_ERROR,
         G::DLN2,
         G::DXHAT2,
         G::DXHAT_XHAT2,
         G::DY_XHAT2,
        G::D_TIMES_DXHAT2,
        G::FIRST_DIFF2,
        G::XHAT_TIMES_SUM2,
        G::BRACKET2,
        G::DRESIDUAL1_LN,
        G::DRESIDUAL1,
        G::DCONTEXT,
        G::DV,
        G::DQ_RAW,
        G::DK_RAW,
        G::DQ,
        G::DK,
        G::DLN1_Q,
        G::DLN1_K,
        G::DLN1_V,
        G::DLN1_QK,
         G::DLN1,
         G::DXHAT1,
         G::DXHAT_XHAT1,
         G::DY_XHAT1,
        G::D_TIMES_DXHAT1,
        G::FIRST_DIFF1,
        G::XHAT_TIMES_SUM1,
        G::BRACKET1},
       QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  make(G::DRESIDUAL1, tapType(G::DRESIDUAL1, QNN_TENSOR_TYPE_NATIVE),
       g.activationDims);
  make(G::DPROBABILITIES,
       tapType(G::DPROBABILITIES, QNN_TENSOR_TYPE_NATIVE), g.scoreDims);
  make(G::DSCORES, tapType(G::DSCORES, QNN_TENSOR_TYPE_NATIVE), g.scoreDims);
  make(G::DLN1, tapType(G::DLN1, QNN_TENSOR_TYPE_NATIVE), g.activationDims);
  make(G::DXHAT1, tapType(G::DXHAT1, QNN_TENSOR_TYPE_NATIVE),
       g.activationDims);
  make(G::SUM_DXHAT1, tapType(G::SUM_DXHAT1, QNN_TENSOR_TYPE_NATIVE),
       g.rowDims);
  make(G::DXHAT_XHAT1, tapType(G::DXHAT_XHAT1, QNN_TENSOR_TYPE_NATIVE),
       g.activationDims);
  make(G::SUM_DXHAT_XHAT1,
       tapType(G::SUM_DXHAT_XHAT1, QNN_TENSOR_TYPE_NATIVE), g.rowDims);
  if(g.tensors[G::SOFTMAX_DOT].v1.rank!=2 ||
     g.tensors[G::SOFTMAX_DOT].v1.dimensions[0]!=tokens ||
     g.tensors[G::SOFTMAX_DOT].v1.dimensions[1]!=1) {
    error="tiny training softmax reduction output must be tokens x 1";
    return false;
  }
  {
    const shape::Node softmaxDot{
        "tt_smd", shape::Op::ReduceSum,
        {{"SOFTMAX_PRODUCT", {tokens, tokens}}},
        {{"SOFTMAX_DOT", {g.tensors[G::SOFTMAX_DOT].v1.dimensions[0],
                            g.tensors[G::SOFTMAX_DOT].v1.dimensions[1]}}},
        {1}, true, false, false, {}, {}, 0, {}, 0};
    const auto validation = shape::validate(softmaxDot);
    if (!validation.ok) {
      error = validation.error;
      return false;
    }
  }
  auto diag =
      diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  make(G::OUTPUT, diag, g.activationDims);
  make(G::DOUTPUT, diag, g.activationDims);
  make(G::LOSS, g.languageModel ? QNN_TENSOR_TYPE_NATIVE :
                              QNN_TENSOR_TYPE_APP_READ, g.scalarDims);
  many({G::DG1, G::DB1, G::DG2, G::DB2}, diag, g.normDims, 1);
  many({G::DWQ, G::DWK, G::DWV, G::DWO}, diag, g.ddDims);
  make(G::DW1, diag, g.dfDims);
  make(G::DW2, diag, g.fdDims);
  if (g.languageModel) {
    make(G::LOGITS, QNN_TENSOR_TYPE_APP_READ, g.logitsDims);
    make(G::LM_PROBABILITIES, diag, g.logitsDims);
    make(G::LM_DIFFERENCE, QNN_TENSOR_TYPE_NATIVE, g.logitsDims);
    make(G::DLOGITS, diag, g.logitsDims);
    make(G::DOUTPUT_PROJECTION, diag, g.outputProjectionDims);
    make(G::DINPUT_NORM, tapType(G::DINPUT_NORM, QNN_TENSOR_TYPE_NATIVE),
         g.activationDims);
    make(G::DINPUT,
         (diagnosticOutputs ||
          variant != TinyTransformerTrainingVariant::FULL)
             ? QNN_TENSOR_TYPE_APP_READ
             : QNN_TENSOR_TYPE_NATIVE,
         g.activationDims);
    make(G::DEMBEDDING,
         (variant == TinyTransformerTrainingVariant::FULL
              ? diagnosticOutputs
              : variant == TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING)
             ? QNN_TENSOR_TYPE_APP_READ
             : QNN_TENSOR_TYPE_NATIVE,
         g.embeddingDims);
    make(G::S_DEMBEDDING, QNN_TENSOR_TYPE_NATIVE, g.embeddingDims);
    make(G::N_EMBEDDING,
         variant == TinyTransformerTrainingVariant::FULL
             ? QNN_TENSOR_TYPE_APP_READ
             : QNN_TENSOR_TYPE_NATIVE,
         g.embeddingDims);
    make(G::S_DOUTPUT_PROJECTION, QNN_TENSOR_TYPE_NATIVE,
         g.outputProjectionDims);
    make(G::N_OUTPUT_PROJECTION,
         variant == TinyTransformerTrainingVariant::FULL
             ? QNN_TENSOR_TYPE_APP_READ
             : QNN_TENSOR_TYPE_NATIVE,
         g.outputProjectionDims);
  }
  many({G::S_DG1, G::S_DB1, G::S_DG2, G::S_DB2}, QNN_TENSOR_TYPE_NATIVE,
       g.normDims, 1);
  many({G::N_G1, G::N_B1, G::N_G2, G::N_B2},
       variant == TinyTransformerTrainingVariant::FULL
           ? QNN_TENSOR_TYPE_APP_READ
           : QNN_TENSOR_TYPE_NATIVE,
       g.normDims, 1);
  many({G::S_DWQ, G::S_DWK, G::S_DWV, G::S_DWO}, QNN_TENSOR_TYPE_NATIVE,
       g.ddDims);
  many({G::N_WQ, G::N_WK, G::N_WV, G::N_WO},
       variant == TinyTransformerTrainingVariant::FULL
           ? QNN_TENSOR_TYPE_APP_READ
           : QNN_TENSOR_TYPE_NATIVE,
       g.ddDims);
  make(G::S_DW1, QNN_TENSOR_TYPE_NATIVE, g.dfDims);
  make(G::N_W1,
       variant == TinyTransformerTrainingVariant::FULL
           ? QNN_TENSOR_TYPE_APP_READ
           : QNN_TENSOR_TYPE_NATIVE,
       g.dfDims);
  make(G::S_DW2, QNN_TENSOR_TYPE_NATIVE, g.fdDims);
  make(G::N_W2,
       variant == TinyTransformerTrainingVariant::FULL
           ? QNN_TENSOR_TYPE_APP_READ
           : QNN_TENSOR_TYPE_NATIVE,
       g.fdDims);
  g.tensors[G::MASK].v1.clientBuf = {
      g.maskData.data(), uint32_t(g.maskData.size() * sizeof(float))};
  g.tensors[G::SCALE].v1.clientBuf = {&g.scaleData, sizeof(float)};
  g.tensors[G::LAST_AXIS].v1.clientBuf = {g.lastAxisData,
                                          sizeof(g.lastAxisData)};
  g.tensors[G::ROW_AXIS].v1.clientBuf = {g.rowAxisData, sizeof(g.rowAxisData)};
  g.tensors[G::ALL_AXES].v1.clientBuf = {g.allAxesData, sizeof(g.allAxesData)};
  g.tensors[G::EPS_SCALED].v1.clientBuf = {&g.epsilonScaledData, sizeof(float)};
  g.tensors[G::CENTER_SCALE].v1.clientBuf = {&g.centeredScaleData,
                                             sizeof(float)};
  g.tensors[G::DIMENSION].v1.clientBuf = {&g.dimensionData, sizeof(float)};
  g.tensors[G::INV_DIMENSION].v1.clientBuf = {&g.inverseDimensionData,
                                              sizeof(float)};
  g.tensors[G::GRAD_SCALE].v1.clientBuf = {&g.gradientScaleData, sizeof(float)};
  if (g.languageModel)
    g.tensors[G::POSITION].v1.clientBuf = {
        g.positionData.data(), uint32_t(g.positionData.size() * sizeof(float))};
  g.tensors[G::ZERO_FF].v1.clientBuf = {
      g.zeroFfData.data(), uint32_t(g.zeroFfData.size() * sizeof(float))};
  auto started = Clock::now();
  auto st = impl_->api.graphCreate(
      impl_->context, "phonelm_tiny_transformer_training", nullptr, &g.graph);
  apiTrace_.fullStepGraphCreateCalled = true;
  apiTrace_.fullStepGraphCreateResult = QNN_GET_ERROR_CODE(st);
  apiTrace_.fullStepGraphHandleNonnull = g.graph != nullptr;
  metrics_.graphCreateUs += elapsedUs(started);
  ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) {
    error =
        "tiny training graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  const int tensorLimit = g.languageModel ? G::TENSOR_COUNT : G::ONE_HOT;
  for (int tensorIndex = 0; tensorIndex < tensorLimit; ++tensorIndex) {
    auto &t = g.tensors[tensorIndex];
    if ((st = impl_->api.tensorCreateGraphTensor(g.graph, &t)) != QNN_SUCCESS) {
      error = "tiny training tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
    ++g.sourceTensorCreateSuccessCount;
  }
  auto bin = [&](const char *n, const char *type, int a, int b, int o) {
    Qnn_Tensor_t in[2] = {g.tensors[a], g.tensors[b]};
    return addSimpleNode(impl_->api, g.graph, n, type, in, 2, &g.tensors[o]);
  };
  auto unary = [&](const char *n, const char *type, int a, int o) {
    return addSimpleNode(impl_->api, g.graph, n, type, &g.tensors[a], 1,
                         &g.tensors[o]);
  };
  auto mm = [&](const char *n, int a, int b, int o, const char *tr = nullptr) {
    Qnn_Tensor_t in[2] = {g.tensors[a], g.tensors[b]};
    return addMatMulNode(impl_->api, g.graph, n, in, &g.tensors[o], tr);
  };
  auto reduce = [&](const char *n, const char *type, const char *axesName,
                    const char *keepName, int a, int axes, bool keep, int o) {
    Qnn_Param_t p[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    p[0].paramType = QNN_PARAMTYPE_TENSOR;
    p[0].name = axesName;
    p[0].tensorParam = g.tensors[axes];
    p[1].paramType = QNN_PARAMTYPE_SCALAR;
    p[1].name = keepName;
    p[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p[1].scalarParam.bool8Value = keep;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = n;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = type;
    op.v1.numOfParams = 2;
    op.v1.params = p;
    op.v1.numOfInputs = 1;
    op.v1.inputTensors = &g.tensors[a];
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &g.tensors[o];
    return impl_->api.graphAddNode(g.graph, op);
  };
  auto ok = [&](Qnn_ErrorHandle_t v, const char *label) {
    if (v == QNN_SUCCESS) {
      ++g.sourceGraphAddNodeSuccessCount;
      return true;
    }
    error = std::string("tiny training ") + label + '=' +
            std::to_string(QNN_GET_ERROR_CODE(v));
    return false;
  };
  TransformerTrainingLayerNormBuilder layerNorm{
      impl_->api, g.graph, g.tensors.data(), G::LAST_AXIS, G::ROW_AXIS,
      G::CENTER_SCALE, G::EPS_SCALED, G::DIMENSION, G::INV_DIMENSION, error,
      g.sourceGraphAddNodeSuccessCount};
  if (g.languageModel &&
      (!ok(mm("lm_embedding", G::ONE_HOT, G::EMBEDDING,
              G::EMBEDDED_TOKEN), "embedding") ||
       !ok(bin("lm_position", QNN_OP_ELEMENT_WISE_ADD, G::EMBEDDED_TOKEN,
               G::POSITION, G::X), "position")))
    return false;
  if (!layerNorm.forward("tt_ln1", G::X, G::G1, G::B1, G::MEAN1, G::CENTERED1, G::CENTERED_S1,
           G::SQUARE1, G::VAR1, G::VAR_EPS1, G::INV_STD_S1, G::INV_STD1,
           G::XHAT1, G::XHAT_G1, G::LN1))
    return false;
  if (!ok(mm("tt_q", G::LN1, G::WQ, G::Q), "q") ||
      !ok(mm("tt_k", G::LN1, G::WK, G::K), "k") ||
      !ok(mm("tt_v", G::LN1, G::WV, G::V), "v") ||
      !ok(mm("tt_scores", G::Q, G::K, G::SCORES,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "scores") ||
      !ok(bin("tt_scale_scores", QNN_OP_ELEMENT_WISE_MULTIPLY, G::SCORES,
              G::SCALE, G::SCALED_SCORES),
          "score scale") ||
      !ok(bin("tt_mask_scores", QNN_OP_ELEMENT_WISE_ADD, G::SCALED_SCORES,
              G::MASK, G::MASKED_SCORES),
          "score mask"))
    return false;
  Qnn_Param_t sp[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  sp[0].paramType = QNN_PARAMTYPE_SCALAR;
  sp[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
  sp[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  sp[0].scalarParam.uint32Value = 1;
  sp[1].paramType = QNN_PARAMTYPE_SCALAR;
  sp[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
  sp[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  sp[1].scalarParam.floatValue = 1;
  Qnn_OpConfig_t sm = QNN_OPCONFIG_INIT;
  sm.v1.name = "tt_softmax";
  sm.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  sm.v1.typeName = QNN_OP_SOFTMAX;
  sm.v1.numOfParams = 2;
  sm.v1.params = sp;
  sm.v1.numOfInputs = 1;
  sm.v1.inputTensors = &g.tensors[G::MASKED_SCORES];
  sm.v1.numOfOutputs = 1;
  sm.v1.outputTensors = &g.tensors[G::PROBABILITIES];
  if (!ok(impl_->api.graphAddNode(g.graph, sm), "softmax") ||
      !ok(mm("tt_context", G::PROBABILITIES, G::V, G::CONTEXT), "context") ||
      !ok(mm("tt_project", G::CONTEXT, G::WO, G::PROJECTED), "project") ||
      !ok(bin("tt_res1", QNN_OP_ELEMENT_WISE_ADD, G::X, G::PROJECTED,
              G::RESIDUAL1),
          "res1"))
    return false;
  if (!layerNorm.forward("tt_ln2", G::RESIDUAL1, G::G2, G::B2, G::MEAN2, G::CENTERED2,
           G::CENTERED_S2, G::SQUARE2, G::VAR2, G::VAR_EPS2, G::INV_STD_S2,
           G::INV_STD2, G::XHAT2, G::XHAT_G2, G::LN2))
    return false;
  if (!ok(mm("tt_ff1", G::LN2, G::W1, G::FF1), "ff1") ||
      !ok(unary("tt_relu", QNN_OP_RELU, G::FF1, G::RELU), "relu") ||
      !ok(mm("tt_ff2", G::RELU, G::W2, G::FF2), "ff2") ||
      !ok(bin("tt_out", QNN_OP_ELEMENT_WISE_ADD, G::RESIDUAL1, G::FF2,
              G::OUTPUT), "out"))
    return false;
  if (g.languageModel) {
    if (!ok(mm("lm_logits", G::OUTPUT, G::OUTPUT_PROJECTION, G::LOGITS),
            "logits"))
      return false;
    Qnn_Param_t lmParams[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    lmParams[0].paramType = QNN_PARAMTYPE_SCALAR;
    lmParams[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
    lmParams[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
    lmParams[0].scalarParam.uint32Value = 1;
    lmParams[1].paramType = QNN_PARAMTYPE_SCALAR;
    lmParams[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
    lmParams[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    lmParams[1].scalarParam.floatValue = 1.0f;
    Qnn_OpConfig_t lmSoftmax = QNN_OPCONFIG_INIT;
    lmSoftmax.v1.name = "lm_softmax";
    lmSoftmax.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    lmSoftmax.v1.typeName = QNN_OP_SOFTMAX;
    lmSoftmax.v1.numOfParams = 2;
    lmSoftmax.v1.params = lmParams;
    lmSoftmax.v1.numOfInputs = 1;
    lmSoftmax.v1.inputTensors = &g.tensors[G::LOGITS];
    lmSoftmax.v1.numOfOutputs = 1;
    lmSoftmax.v1.outputTensors = &g.tensors[G::LM_PROBABILITIES];
    if (!ok(impl_->api.graphAddNode(g.graph, lmSoftmax), "lm softmax") ||
        !ok(bin("lm_difference", QNN_OP_ELEMENT_WISE_SUBTRACT,
                G::LM_PROBABILITIES, G::TARGET, G::LM_DIFFERENCE),
            "lm difference") ||
        !ok(bin("lm_dlogits", QNN_OP_ELEMENT_WISE_MULTIPLY,
                G::LM_DIFFERENCE,
                G::GRAD_SCALE, G::DLOGITS), "lm dlogits") ||
        !ok(mm("lm_doutput", G::DLOGITS, G::OUTPUT_PROJECTION, G::DOUTPUT,
               QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1), "lm doutput") ||
        !ok(mm("lm_doutput_projection", G::OUTPUT, G::DLOGITS,
               G::DOUTPUT_PROJECTION, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
            "lm doutput projection"))
      return false;
  } else if (!ok(bin("tt_error", QNN_OP_ELEMENT_WISE_SUBTRACT, G::OUTPUT,
                     G::TARGET, G::ERROR), "error") ||
             !ok(bin("tt_sq_error", QNN_OP_ELEMENT_WISE_MULTIPLY, G::ERROR,
                     G::ERROR, G::SQUARED_ERROR), "sq error") ||
             !ok(reduce("tt_loss", QNN_OP_REDUCE_MEAN,
                        QNN_OP_REDUCE_MEAN_PARAM_AXES,
                        QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, G::SQUARED_ERROR,
                        G::ALL_AXES, true, G::LOSS), "loss") ||
             !ok(bin("tt_dout", QNN_OP_ELEMENT_WISE_MULTIPLY, G::ERROR,
                     G::GRAD_SCALE, G::DOUTPUT), "dout")) {
    return false;
  }
  if (!ok(mm("tt_dw2", G::RELU, G::DOUTPUT, G::DW2,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dw2") ||
      !ok(mm("tt_drelu", G::DOUTPUT, G::W2, G::DRELU,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "drelu") ||
      !ok(bin("tt_relu_mask", QNN_OP_ELEMENT_WISE_GREATER, G::FF1, G::ZERO_FF,
              G::RELU_MASK),
          "relu mask"))
    return false;
  Qnn_Tensor_t sel[3] = {g.tensors[G::RELU_MASK], g.tensors[G::DRELU],
                         g.tensors[G::ZERO_FF]};
  if (!ok(addSimpleNode(impl_->api, g.graph, "tt_relu_bwd",
                        QNN_OP_ELEMENT_WISE_SELECT, sel, 3,
                        &g.tensors[G::DFF1]),
          "relu bwd") ||
      !ok(mm("tt_dw1", G::LN2, G::DFF1, G::DW1,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dw1") ||
      !ok(mm("tt_dln2", G::DFF1, G::W1, G::DLN2,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dln2"))
    return false;
  if (!layerNorm.backward("tt_lnb2", G::DLN2, G::G2, G::XHAT2, G::INV_STD2, G::DXHAT2,
           G::SUM_DXHAT2, G::DXHAT_XHAT2, G::DY_XHAT2,
           G::SUM_DXHAT_XHAT2, G::D_TIMES_DXHAT2, G::FIRST_DIFF2,
           G::XHAT_TIMES_SUM2, G::BRACKET2, G::INV_STD_OVER_D2,
           G::DRESIDUAL1_LN, G::DG2, G::DB2))
    return false;
  if (!ok(bin("tt_dres1", QNN_OP_ELEMENT_WISE_ADD, G::DOUTPUT, G::DRESIDUAL1_LN,
              G::DRESIDUAL1),
          "dres1") ||
      !ok(mm("tt_dwo", G::CONTEXT, G::DRESIDUAL1, G::DWO,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dwo") ||
      !ok(mm("tt_dcontext", G::DRESIDUAL1, G::WO, G::DCONTEXT,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dcontext") ||
      !ok(mm("tt_dp", G::DCONTEXT, G::V, G::DPROBABILITIES,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dp") ||
      !ok(mm("tt_dv", G::PROBABILITIES, G::DCONTEXT, G::DV,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dv") ||
      !ok(bin("tt_smp", QNN_OP_ELEMENT_WISE_MULTIPLY, G::DPROBABILITIES,
              G::PROBABILITIES, G::SOFTMAX_PRODUCT),
          "smp") ||
      !ok(reduce("tt_smd", QNN_OP_REDUCE_SUM, QNN_OP_REDUCE_SUM_PARAM_AXES,
                 QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, G::SOFTMAX_PRODUCT,
                 G::LAST_AXIS, true, G::SOFTMAX_DOT),
          "smd") ||
      !ok(bin("tt_smc", QNN_OP_ELEMENT_WISE_SUBTRACT, G::DPROBABILITIES,
              G::SOFTMAX_DOT, G::SOFTMAX_CENTERED),
          "smc") ||
      !ok(bin("tt_ds", QNN_OP_ELEMENT_WISE_MULTIPLY, G::PROBABILITIES,
              G::SOFTMAX_CENTERED, G::DSCORES),
          "ds") ||
      !ok(mm("tt_dqr", G::DSCORES, G::K, G::DQ_RAW), "dqr") ||
      !ok(mm("tt_dkr", G::DSCORES, G::Q, G::DK_RAW,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dkr") ||
      !ok(bin("tt_dq", QNN_OP_ELEMENT_WISE_MULTIPLY, G::DQ_RAW, G::SCALE,
              G::DQ),
          "dq") ||
      !ok(bin("tt_dk", QNN_OP_ELEMENT_WISE_MULTIPLY, G::DK_RAW, G::SCALE,
              G::DK),
          "dk"))
    return false;
  if (!ok(mm("tt_dwq", G::LN1, G::DQ, G::DWQ,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dwq") ||
      !ok(mm("tt_dwk", G::LN1, G::DK, G::DWK,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dwk") ||
      !ok(mm("tt_dwv", G::LN1, G::DV, G::DWV,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0),
          "dwv") ||
      !ok(mm("tt_dlq", G::DQ, G::WQ, G::DLN1_Q,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dlq") ||
      !ok(mm("tt_dlk", G::DK, G::WK, G::DLN1_K,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dlk") ||
      !ok(mm("tt_dlv", G::DV, G::WV, G::DLN1_V,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1),
          "dlv") ||
      !ok(bin("tt_dlqk", QNN_OP_ELEMENT_WISE_ADD, G::DLN1_Q, G::DLN1_K,
              G::DLN1_QK),
          "dlqk") ||
      !ok(bin("tt_dln1", QNN_OP_ELEMENT_WISE_ADD, G::DLN1_QK, G::DLN1_V,
              G::DLN1),
          "dln1"))
    return false;
  if (!layerNorm.backward("tt_lnb1", G::DLN1, G::G1, G::XHAT1, G::INV_STD1, G::DXHAT1,
           G::SUM_DXHAT1, G::DXHAT_XHAT1, G::DY_XHAT1,
           G::SUM_DXHAT_XHAT1, G::D_TIMES_DXHAT1, G::FIRST_DIFF1,
           G::XHAT_TIMES_SUM1, G::BRACKET1, G::INV_STD_OVER_D1,
           g.languageModel ? G::DINPUT_NORM : G::CENTERED1, G::DG1, G::DB1))
    return false;
  if (g.languageModel &&
      !ok(bin("lm_dinput", QNN_OP_ELEMENT_WISE_ADD, G::DRESIDUAL1,
              G::DINPUT_NORM, G::DINPUT), "lm dinput"))
    return false;
  if (g.languageModel &&
      g.variant != TinyTransformerTrainingVariant::STOP_AFTER_DINPUT &&
      !ok(mm("lm_dembedding", G::ONE_HOT, G::DINPUT, G::DEMBEDDING,
             QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0), "lm dembedding"))
    return false;
  auto upd = [&](const char *n, int cur, int grad, int scaled, int next) {
    std::string s = n;
    return ok(bin((s + "_scale").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, grad,
                  G::LR, scaled),
              "upd scale") &&
           ok(bin((s + "_next").c_str(), QNN_OP_ELEMENT_WISE_SUBTRACT, cur,
                  scaled, next),
              "upd next");
  };
  if (g.variant == TinyTransformerTrainingVariant::FULL &&
      (!upd("tt_g1", G::G1, G::DG1, G::S_DG1, G::N_G1) ||
       !upd("tt_b1", G::B1, G::DB1, G::S_DB1, G::N_B1) ||
       !upd("tt_wq", G::WQ, G::DWQ, G::S_DWQ, G::N_WQ) ||
       !upd("tt_wk", G::WK, G::DWK, G::S_DWK, G::N_WK) ||
       !upd("tt_wv", G::WV, G::DWV, G::S_DWV, G::N_WV) ||
       !upd("tt_wo", G::WO, G::DWO, G::S_DWO, G::N_WO) ||
       !upd("tt_g2", G::G2, G::DG2, G::S_DG2, G::N_G2) ||
       !upd("tt_b2", G::B2, G::DB2, G::S_DB2, G::N_B2) ||
       !upd("tt_w1", G::W1, G::DW1, G::S_DW1, G::N_W1) ||
       !upd("tt_w2", G::W2, G::DW2, G::S_DW2, G::N_W2) ||
       (g.languageModel &&
        (!upd("lm_embedding", G::EMBEDDING, G::DEMBEDDING,
              G::S_DEMBEDDING, G::N_EMBEDDING) ||
         !upd("lm_output_projection", G::OUTPUT_PROJECTION,
              G::DOUTPUT_PROJECTION, G::S_DOUTPUT_PROJECTION,
              G::N_OUTPUT_PROJECTION)))))
    return false;
  started = Clock::now();
  st = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  apiTrace_.fullStepGraphFinalizeCalled = true;
  apiTrace_.fullStepGraphFinalizeResult = QNN_GET_ERROR_CODE(st);
  metrics_.graphFinalizeUs += elapsedUs(started);
  ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    error =
        "tiny training graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  diagnostics_ +=
      "tiny_transformer_training_graph=EXPLICIT_FORWARD_BACKWARD_SGD\ntiny_"
      "transformer_training_graph_count=1\ntiny_transformer_training_execute_"
      "count_per_step=1\n";
  diagnostics_ += g.languageModel
                      ? "tiny_language_model_weight_handoff=CPU_FIXED_BUFFER_"
                        "COPY\n"
                      : "tiny_transformer_training_weight_handoff=CPU_PING_"
                        "PONG_BINDING\n";
  diagnostics_ += "tiny_transformer_training_in_place_alias=false\n";
  switch (g.variant) {
  case TinyTransformerTrainingVariant::FULL:
    diagnostics_ += "tiny_transformer_training_variant=FULL\n"
                    "tiny_transformer_training_node_boundary=SGD_UPDATE\n";
    break;
  case TinyTransformerTrainingVariant::STOP_AFTER_DINPUT:
    diagnostics_ += "tiny_transformer_training_variant=STOP_AFTER_DINPUT\n"
                    "tiny_transformer_training_node_boundary=lm_dinput\n";
    break;
  case TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING:
    diagnostics_ += "tiny_transformer_training_variant=STOP_AFTER_DEMBEDDING\n"
                    "tiny_transformer_training_node_boundary=lm_dembedding\n";
    break;
  }
  return true;
}

bool Runtime::executeTinyTransformerTraining(
    const std::vector<float> &input, const std::vector<float> &target,
    const TinyTransformerParameters &current, float learningRate,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using G = Impl::TinyTransformerTrainingGraph;
  auto &g = impl_->tinyTransformerTraining;
  const auto generalizedState = impl_->generalizedTinyTransformerTraining.state;
  if (generalizedState == Impl::GeneralizedTinyTransformerTrainingGraph::State::ACTIVE)
    return executeTinyTransformerTrainingGeneralized(input, target, current,
                                                     learningRate, outputs, error);
  if (generalizedState != Impl::GeneralizedTinyTransformerTrainingGraph::State::INACTIVE) {
    error = "generalized L2/H1 preparation did not complete";
    return false;
  }
  outputs.layerInputGradients.clear();
  outputs.taps.clear();
  outputs.taps.reserve(g.tapSet == TinyTransformerTrainingTapSet::ALL_INTERNAL
                           ? 100
                           : 6);
  const size_t a = size_t(g.tokens) * g.dimension;
  const size_t dd = size_t(g.dimension) * g.dimension;
  const size_t df = size_t(g.dimension) * g.feedForwardDimension;
  const size_t fd = size_t(g.feedForwardDimension) * g.dimension;
  const size_t n = g.dimension;
  const size_t tv = size_t(g.tokens) * g.vocabularySize;
  const size_t vd = size_t(g.vocabularySize) * g.dimension;
  const size_t dv = size_t(g.dimension) * g.vocabularySize;
  const size_t expectedInput = g.languageModel ? tv : a;
  auto layerBuffersValid = [&](const TinyTransformerLayerParameters &p) {
    return p.gamma1.size() == n && p.beta1.size() == n &&
           p.wq.size() == dd && p.wk.size() == dd && p.wv.size() == dd &&
           p.wo.size() == dd && p.gamma2.size() == n && p.beta2.size() == n &&
           p.w1.size() == df && p.w2.size() == fd;
  };
  const bool additionalLayersValid =
      current.layers.size() == size_t(g.numLayers - 1) &&
      std::all_of(current.layers.begin(), current.layers.end(), layerBuffersValid);
  if (!g.graph || input.size() != expectedInput || target.size() != expectedInput ||
      !std::isfinite(learningRate) || current.gamma1.size() != n ||
      current.beta1.size() != n || current.wq.size() != dd ||
      current.wk.size() != dd || current.wv.size() != dd ||
      current.wo.size() != dd || current.gamma2.size() != n ||
      current.beta2.size() != n || current.w1.size() != df ||
      current.w2.size() != fd || !additionalLayersValid ||
      (g.languageModel && (current.tokenEmbedding.size() != vd ||
                           current.outputProjection.size() != dv))) {
    error = "tiny training buffer size";
    return false;
  }
  std::vector<Qnn_Tensor_t> in;
  in.reserve(g.languageModel ? 15 : 13);
  auto bi = [&](int id, const std::vector<float> &v) {
    in.push_back(g.tensors[id]);
    in.back().v1.clientBuf = {const_cast<float *>(v.data()),
                              uint32_t(v.size() * sizeof(float))};
  };
  bi(g.languageModel ? G::ONE_HOT : G::X, input);
  bi(G::TARGET, target);
  bi(G::G1, current.gamma1);
  bi(G::B1, current.beta1);
  bi(G::WQ, current.wq);
  bi(G::WK, current.wk);
  bi(G::WV, current.wv);
  bi(G::WO, current.wo);
  bi(G::G2, current.gamma2);
  bi(G::B2, current.beta2);
  bi(G::W1, current.w1);
  bi(G::W2, current.w2);
  if (g.languageModel) {
    bi(G::EMBEDDING, current.tokenEmbedding);
    bi(G::OUTPUT_PROJECTION, current.outputProjection);
  }
  if (g.variant == TinyTransformerTrainingVariant::FULL) {
    in.push_back(g.tensors[G::LR]);
    in.back().v1.clientBuf = {&learningRate, sizeof(float)};
  }
  std::uint32_t learningRateBitsBefore = 0;
  std::memcpy(&learningRateBitsBefore, &learningRate, sizeof(learningRateBitsBefore));
  std::vector<Qnn_Tensor_t> out;
  out.reserve((g.languageModel ? 30 : 23) + 6);
  auto bo = [&](int id, std::vector<float> &v, size_t count) {
    v.resize(count);
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {v.data(), uint32_t(count * sizeof(float))};
  };
  if (!g.languageModel) {
    out.push_back(g.tensors[G::LOSS]);
    out.back().v1.clientBuf = {&outputs.loss, sizeof(float)};
  } else {
    bo(G::LOGITS, outputs.logits, tv);
  }
  if (g.variant == TinyTransformerTrainingVariant::FULL) {
    bo(G::N_G1, outputs.next.gamma1, n);
    bo(G::N_B1, outputs.next.beta1, n);
    bo(G::N_WQ, outputs.next.wq, dd);
    bo(G::N_WK, outputs.next.wk, dd);
    bo(G::N_WV, outputs.next.wv, dd);
    bo(G::N_WO, outputs.next.wo, dd);
    bo(G::N_G2, outputs.next.gamma2, n);
    bo(G::N_B2, outputs.next.beta2, n);
    bo(G::N_W1, outputs.next.w1, df);
    bo(G::N_W2, outputs.next.w2, fd);
    if (g.languageModel) {
      bo(G::N_EMBEDDING, outputs.next.tokenEmbedding, vd);
      bo(G::N_OUTPUT_PROJECTION, outputs.next.outputProjection, dv);
    }
  } else {
    outputs.next = {};
  }
  if (g.diagnosticOutputs) {
    bo(G::OUTPUT, outputs.output, a);
    bo(G::DOUTPUT, outputs.dOutput, a);
    bo(G::DG1, outputs.gradients.gamma1, n);
    bo(G::DB1, outputs.gradients.beta1, n);
    bo(G::DWQ, outputs.gradients.wq, dd);
    bo(G::DWK, outputs.gradients.wk, dd);
    bo(G::DWV, outputs.gradients.wv, dd);
    bo(G::DWO, outputs.gradients.wo, dd);
    bo(G::DG2, outputs.gradients.gamma2, n);
    bo(G::DB2, outputs.gradients.beta2, n);
    bo(G::DW1, outputs.gradients.w1, df);
    bo(G::DW2, outputs.gradients.w2, fd);
    if (g.languageModel) {
      bo(G::X, outputs.embeddedInput, a);
      bo(G::LM_PROBABILITIES, outputs.probabilities, tv);
      bo(G::DLOGITS, outputs.dLogits, tv);
      bo(G::DINPUT, outputs.dEmbeddedInput, a);
      if (g.variant != TinyTransformerTrainingVariant::STOP_AFTER_DINPUT)
        bo(G::DEMBEDDING, outputs.gradients.tokenEmbedding, vd);
      bo(G::DOUTPUT_PROJECTION, outputs.gradients.outputProjection, dv);
    }
  } else {
    outputs.output.clear();
    outputs.dOutput.clear();
    outputs.embeddedInput.clear();
    outputs.probabilities.clear();
    outputs.dLogits.clear();
    outputs.dEmbeddedInput.clear();
    outputs.gradients = {};
  }
  if (g.languageModel && !g.diagnosticOutputs &&
      g.variant != TinyTransformerTrainingVariant::FULL) {
    bo(G::DINPUT, outputs.dEmbeddedInput, a);
    if (g.variant == TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING)
      bo(G::DEMBEDDING, outputs.gradients.tokenEmbedding, vd);
  }
  auto tap = [&](int id, const char *name, size_t count) {
    outputs.taps.push_back({name, {}});
    auto &values = outputs.taps.back().values;
    values.assign(count, outputs.tapPoison);
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {values.data(), uint32_t(count * sizeof(float))};
  };
  auto isRegularOutput = [&](int id) {
    if (id == G::LOGITS)
      return true;
    if (!g.diagnosticOutputs)
      return (id == G::DINPUT &&
              g.variant != TinyTransformerTrainingVariant::FULL) ||
             (id == G::DEMBEDDING &&
              g.variant == TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING);
    if (id == G::OUTPUT || id == G::DOUTPUT || id == G::DG1 ||
        id == G::DB1 || id == G::DWQ || id == G::DWK || id == G::DWV ||
        id == G::DWO || id == G::DG2 || id == G::DB2 || id == G::DW1 ||
        id == G::DW2)
      return true;
    return g.languageModel &&
           (id == G::X || id == G::LM_PROBABILITIES || id == G::DLOGITS ||
            id == G::DINPUT || id == G::DOUTPUT_PROJECTION ||
            (id == G::DEMBEDDING &&
             g.variant != TinyTransformerTrainingVariant::STOP_AFTER_DINPUT));
  };
  auto internalTap = [&](int id, const char *name, size_t count) {
    if (!isRegularOutput(id))
      tap(id, name, count);
  };
  switch (g.tapSet) {
  case TinyTransformerTrainingTapSet::NONE:
    break;
  case TinyTransformerTrainingTapSet::DSCORES_ONLY:
    tap(G::DSCORES, "DSCORES", size_t(g.tokens) * g.tokens);
    break;
  case TinyTransformerTrainingTapSet::DPROB_DSCORES:
    tap(G::DPROBABILITIES, "DPROBABILITIES",
        size_t(g.tokens) * g.tokens);
    tap(G::DSCORES, "DSCORES", size_t(g.tokens) * g.tokens);
    break;
  case TinyTransformerTrainingTapSet::BACKWARD_REGIONS:
    tap(G::DRESIDUAL1, "DRESIDUAL1", a);
    tap(G::DSCORES, "DSCORES", size_t(g.tokens) * g.tokens);
    tap(G::DLN1, "DLN1", a);
    tap(G::DINPUT_NORM, "DINPUT_NORM", a);
    break;
  case TinyTransformerTrainingTapSet::LAYERNORM1:
    tap(G::DLN1, "DLN1", a);
    tap(G::DXHAT1, "DXHAT1", a);
    tap(G::SUM_DXHAT1, "SUM_DXHAT1", g.tokens);
    tap(G::DXHAT_XHAT1, "DXHAT_XHAT1", a);
    tap(G::SUM_DXHAT_XHAT1, "SUM_DXHAT_XHAT1", g.tokens);
    tap(G::DINPUT_NORM, "DINPUT_NORM", a);
    break;
  case TinyTransformerTrainingTapSet::ALL_INTERNAL:
    // Order is the graph's producer order. LOGITS is intentionally omitted
    // here because it is always the regular APP_READ output.
    internalTap(G::EMBEDDED_TOKEN, "EMBEDDED_TOKEN", a);
    internalTap(G::X, "INPUT", a);
    internalTap(G::MEAN1, "MEAN1", g.tokens);
    internalTap(G::CENTERED1, "CENTERED1", a);
    internalTap(G::CENTERED_S1, "CENTERED_S1", a);
    internalTap(G::SQUARE1, "SQUARE1", a);
    internalTap(G::VAR1, "VAR1", g.tokens);
    internalTap(G::VAR_EPS1, "VAR_EPS1", g.tokens);
    internalTap(G::INV_STD_S1, "INV_STD_S1", g.tokens);
    internalTap(G::INV_STD1, "INV_STD1", g.tokens);
    internalTap(G::XHAT1, "XHAT1", a);
    internalTap(G::XHAT_G1, "XHAT_G1", a);
    internalTap(G::LN1, "LN1", a);
    internalTap(G::Q, "Q", a);
    internalTap(G::K, "K", a);
    internalTap(G::V, "V", a);
    internalTap(G::SCORES, "SCORES", size_t(g.tokens) * g.tokens);
    internalTap(G::SCALED_SCORES, "SCALED_SCORES",
                size_t(g.tokens) * g.tokens);
    internalTap(G::MASKED_SCORES, "MASKED_SCORES",
                size_t(g.tokens) * g.tokens);
    internalTap(G::PROBABILITIES, "PROBABILITIES",
                size_t(g.tokens) * g.tokens);
    internalTap(G::CONTEXT, "CONTEXT", a);
    internalTap(G::PROJECTED, "PROJECTED", a);
    internalTap(G::RESIDUAL1, "RESIDUAL1", a);
    internalTap(G::MEAN2, "MEAN2", g.tokens);
    internalTap(G::CENTERED2, "CENTERED2", a);
    internalTap(G::CENTERED_S2, "CENTERED_S2", a);
    internalTap(G::SQUARE2, "SQUARE2", a);
    internalTap(G::VAR2, "VAR2", g.tokens);
    internalTap(G::VAR_EPS2, "VAR_EPS2", g.tokens);
    internalTap(G::INV_STD_S2, "INV_STD_S2", g.tokens);
    internalTap(G::INV_STD2, "INV_STD2", g.tokens);
    internalTap(G::XHAT2, "XHAT2", a);
    internalTap(G::XHAT_G2, "XHAT_G2", a);
    internalTap(G::LN2, "LN2", a);
    internalTap(G::FF1, "FF1", size_t(g.tokens) * g.feedForwardDimension);
    internalTap(G::RELU, "RELU", size_t(g.tokens) * g.feedForwardDimension);
    internalTap(G::FF2, "FF2", a);
    internalTap(G::OUTPUT, "OUTPUT", a);
    internalTap(G::LM_PROBABILITIES, "LM_PROBABILITIES", tv);
    internalTap(G::LM_DIFFERENCE, "LM_DIFFERENCE", tv);
    internalTap(G::DLOGITS, "DLOGITS", tv);
    internalTap(G::DOUTPUT, "DOUTPUT", a);
    internalTap(G::DOUTPUT_PROJECTION, "DOUTPUT_PROJECTION", dv);
    internalTap(G::DW2, "DW2", fd);
    internalTap(G::DRELU, "DRELU", size_t(g.tokens) * g.feedForwardDimension);
    internalTap(G::DFF1, "DFF1", size_t(g.tokens) * g.feedForwardDimension);
    internalTap(G::DW1, "DW1", df);
    internalTap(G::DLN2, "DLN2", a);
    internalTap(G::DXHAT2, "DXHAT2", a);
    internalTap(G::SUM_DXHAT2, "SUM_DXHAT2", g.tokens);
    internalTap(G::DXHAT_XHAT2, "DXHAT_XHAT2", a);
    internalTap(G::SUM_DXHAT_XHAT2, "SUM_DXHAT_XHAT2", g.tokens);
    internalTap(G::D_TIMES_DXHAT2, "D_TIMES_DXHAT2", a);
    internalTap(G::FIRST_DIFF2, "FIRST_DIFF2", a);
    internalTap(G::XHAT_TIMES_SUM2, "XHAT_TIMES_SUM2", a);
    internalTap(G::BRACKET2, "BRACKET2", a);
    internalTap(G::INV_STD_OVER_D2, "INV_STD_OVER_D2", g.tokens);
    internalTap(G::DRESIDUAL1_LN, "DRESIDUAL1_LN", a);
    internalTap(G::DY_XHAT2, "DY_XHAT2", a);
    internalTap(G::DG2, "DG2", n);
    internalTap(G::DB2, "DB2", n);
    internalTap(G::DRESIDUAL1, "DRESIDUAL1", a);
    internalTap(G::DWO, "DWO", dd);
    internalTap(G::DCONTEXT, "DCONTEXT", a);
    internalTap(G::DPROBABILITIES, "DPROBABILITIES",
                size_t(g.tokens) * g.tokens);
    internalTap(G::DV, "DV", a);
    internalTap(G::SOFTMAX_PRODUCT, "SOFTMAX_PRODUCT",
                size_t(g.tokens) * g.tokens);
    internalTap(G::SOFTMAX_DOT, "SOFTMAX_DOT", g.tokens);
    internalTap(G::SOFTMAX_CENTERED, "SOFTMAX_CENTERED",
                size_t(g.tokens) * g.tokens);
    internalTap(G::DSCORES, "DSCORES", size_t(g.tokens) * g.tokens);
    internalTap(G::DQ_RAW, "DQ_RAW", a);
    internalTap(G::DK_RAW, "DK_RAW", a);
    internalTap(G::DQ, "DQ", a);
    internalTap(G::DK, "DK", a);
    internalTap(G::DWQ, "DWQ", dd);
    internalTap(G::DWK, "DWK", dd);
    internalTap(G::DWV, "DWV", dd);
    internalTap(G::DLN1_Q, "DLN1_Q", a);
    internalTap(G::DLN1_K, "DLN1_K", a);
    internalTap(G::DLN1_V, "DLN1_V", a);
    internalTap(G::DLN1_QK, "DLN1_QK", a);
    internalTap(G::DLN1, "DLN1", a);
    internalTap(G::DXHAT1, "DXHAT1", a);
    internalTap(G::SUM_DXHAT1, "SUM_DXHAT1", g.tokens);
    internalTap(G::DXHAT_XHAT1, "DXHAT_XHAT1", a);
    internalTap(G::SUM_DXHAT_XHAT1, "SUM_DXHAT_XHAT1", g.tokens);
    internalTap(G::D_TIMES_DXHAT1, "D_TIMES_DXHAT1", a);
    internalTap(G::FIRST_DIFF1, "FIRST_DIFF1", a);
    internalTap(G::XHAT_TIMES_SUM1, "XHAT_TIMES_SUM1", a);
    internalTap(G::BRACKET1, "BRACKET1", a);
    internalTap(G::INV_STD_OVER_D1, "INV_STD_OVER_D1", g.tokens);
    internalTap(G::DINPUT_NORM, "DINPUT_NORM", a);
    internalTap(G::DY_XHAT1, "DY_XHAT1", a);
    internalTap(G::DG1, "DG1", n);
    internalTap(G::DB1, "DB1", n);
    internalTap(G::DINPUT, "DINPUT", a);
    if (g.variant != TinyTransformerTrainingVariant::STOP_AFTER_DINPUT)
      internalTap(G::DEMBEDDING, "DEMBEDDING", vd);
    break;
  }
  g.lastInputTensorCount = static_cast<uint32_t>(in.size());
  g.lastOutputTensorCount = static_cast<uint32_t>(out.size());
  auto started = Clock::now();
  auto st = impl_->api.graphExecute(g.graph, in.data(), uint32_t(in.size()),
                                    out.data(), uint32_t(out.size()), nullptr,
                                    nullptr);
  metrics_.executeUs.push_back(elapsedUs(started));
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st),
                           st == QNN_SUCCESS);
  std::uint32_t learningRateBitsAfter = 0;
  std::memcpy(&learningRateBitsAfter, &learningRate, sizeof(learningRateBitsAfter));
  g.lastLearningRateBytesUnchanged =
      g.variant != TinyTransformerTrainingVariant::FULL ||
      learningRateBitsBefore == learningRateBitsAfter;
  if (st != QNN_SUCCESS) {
    error =
        "tiny training graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  if (g.languageModel) {
    double loss = 0.0;
    for (uint32_t row = 0; row < g.tokens; ++row) {
      const float *z = outputs.logits.data() + size_t(row) * g.vocabularySize;
      const float maximum = *std::max_element(z, z + g.vocabularySize);
      double sum = 0.0;
      uint32_t targetIndex = 0;
      for (uint32_t column = 0; column < g.vocabularySize; ++column) {
        sum += std::exp(double(z[column] - maximum));
        if (target[size_t(row) * g.vocabularySize + column] > 0.5f)
          targetIndex = column;
      }
      loss += double(maximum) + std::log(sum) - z[targetIndex];
    }
    outputs.loss = float(loss / g.tokens);
  }
  return true;
}
