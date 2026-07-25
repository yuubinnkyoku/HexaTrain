bool Runtime::prepareTinyTransformerTraining(
    uint32_t tokens, uint32_t dimension, uint32_t feedForwardDimension,
    float epsilon, bool diagnosticOutputs, std::string &error) {
  if (!impl_ || !impl_->context || !tokens || !dimension ||
      !feedForwardDimension || !std::isfinite(epsilon) || epsilon <= 0) {
    error = "tiny training invalid runtime or shape";
    return false;
  }
  using G = Impl::TinyTransformerTrainingGraph;
  auto &g = impl_->tinyTransformerTraining;
  g.tokens = tokens;
  g.dimension = dimension;
  g.feedForwardDimension = feedForwardDimension;
  g.diagnosticOutputs = diagnosticOutputs;
  g.activationDims[0] = tokens;
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
  g.epsilonScaledData = epsilon * g.centeredScaleData * g.centeredScaleData;
  g.dimensionData = float(dimension);
  g.inverseDimensionData = 1.0f / float(dimension);
  g.gradientScaleData = 2.0f / float(tokens * dimension);
  g.maskData.assign(size_t(tokens) * tokens, 0);
  for (uint32_t r = 0; r < tokens; ++r)
    for (uint32_t c = r + 1; c < tokens; ++c)
      g.maskData[r * tokens + c] = -10000.0f;
  g.zeroFfData.assign(size_t(tokens) * feedForwardDimension, 0);
  for (int i = 0; i < G::TENSOR_COUNT; ++i) {
    Qnn_Tensor_t initialized = QNN_TENSOR_INIT;
    g.tensors[i] = initialized;
    g.names[i] = "tiny_train_" + std::to_string(i);
  }
  auto make = [&](int i, Qnn_TensorType_t type, uint32_t *d, uint32_t rank = 2,
                  Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32) {
    tensor(g.tensors[i], g.names[i].c_str(), type, d, dt, rank);
  };
  auto many = [&](std::initializer_list<int> ids, Qnn_TensorType_t type,
                  uint32_t *d, uint32_t rank = 2,
                  Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32) {
    for (int i : ids)
      make(i, type, d, rank, dt);
  };
  many({G::X, G::TARGET}, QNN_TENSOR_TYPE_APP_WRITE, g.activationDims);
  many({G::G1, G::B1, G::G2, G::B2}, QNN_TENSOR_TYPE_APP_WRITE, g.normDims, 1);
  many({G::WQ, G::WK, G::WV, G::WO}, QNN_TENSOR_TYPE_APP_WRITE, g.ddDims);
  make(G::W1, QNN_TENSOR_TYPE_APP_WRITE, g.dfDims);
  make(G::W2, QNN_TENSOR_TYPE_APP_WRITE, g.fdDims);
  make(G::LR, QNN_TENSOR_TYPE_APP_WRITE, g.scalarDims);
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
        G::SUM_DXHAT_XHAT1, G::INV_STD_OVER_D1},
       QNN_TENSOR_TYPE_NATIVE, g.rowDims);
  many({G::SCORES, G::SCALED_SCORES, G::MASKED_SCORES, G::PROBABILITIES,
        G::DPROBABILITIES, G::SOFTMAX_PRODUCT, G::SOFTMAX_DOT,
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
        G::D_TIMES_DXHAT1,
        G::FIRST_DIFF1,
        G::XHAT_TIMES_SUM1,
        G::BRACKET1},
       QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  auto diag =
      diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  make(G::OUTPUT, diag, g.activationDims);
  make(G::DOUTPUT, diag, g.activationDims);
  make(G::LOSS, QNN_TENSOR_TYPE_APP_READ, g.scalarDims);
  many({G::DG1, G::DB1, G::DG2, G::DB2}, diag, g.normDims, 1);
  many({G::DWQ, G::DWK, G::DWV, G::DWO}, diag, g.ddDims);
  make(G::DW1, diag, g.dfDims);
  make(G::DW2, diag, g.fdDims);
  many({G::S_DG1, G::S_DB1, G::S_DG2, G::S_DB2}, QNN_TENSOR_TYPE_NATIVE,
       g.normDims, 1);
  many({G::N_G1, G::N_B1, G::N_G2, G::N_B2}, QNN_TENSOR_TYPE_APP_READ,
       g.normDims, 1);
  many({G::S_DWQ, G::S_DWK, G::S_DWV, G::S_DWO}, QNN_TENSOR_TYPE_NATIVE,
       g.ddDims);
  many({G::N_WQ, G::N_WK, G::N_WV, G::N_WO}, QNN_TENSOR_TYPE_APP_READ,
       g.ddDims);
  make(G::S_DW1, QNN_TENSOR_TYPE_NATIVE, g.dfDims);
  make(G::N_W1, QNN_TENSOR_TYPE_APP_READ, g.dfDims);
  make(G::S_DW2, QNN_TENSOR_TYPE_NATIVE, g.fdDims);
  make(G::N_W2, QNN_TENSOR_TYPE_APP_READ, g.fdDims);
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
  for (auto &t : g.tensors)
    if ((st = impl_->api.tensorCreateGraphTensor(g.graph, &t)) != QNN_SUCCESS) {
      error = "tiny training tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
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
    if (v == QNN_SUCCESS)
      return true;
    error = std::string("tiny training ") + label + '=' +
            std::to_string(QNN_GET_ERROR_CODE(v));
    return false;
  };
  auto lnf = [&](const char *p, int x, int ga, int be, int mean, int center,
                 int centerS, int square, int var, int varEps, int invS,
                 int inv, int xhat, int xhatG, int out) {
    std::string s = p;
    return ok(reduce((s + "_mean").c_str(), QNN_OP_REDUCE_MEAN,
                     QNN_OP_REDUCE_MEAN_PARAM_AXES,
                     QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, x, G::LAST_AXIS, true,
                     mean),
              "ln mean") &&
           ok(bin((s + "_center").c_str(), QNN_OP_ELEMENT_WISE_SUBTRACT, x,
                  mean, center),
              "ln center") &&
           ok(bin((s + "_center_scale").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY,
                  center, G::CENTER_SCALE, centerS),
              "ln center scale") &&
           ok(bin((s + "_square").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY,
                  centerS, centerS, square),
              "ln square") &&
           ok(reduce((s + "_var").c_str(), QNN_OP_REDUCE_MEAN,
                     QNN_OP_REDUCE_MEAN_PARAM_AXES,
                     QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, square, G::LAST_AXIS,
                     true, var),
              "ln var") &&
           ok(bin((s + "_eps").c_str(), QNN_OP_ELEMENT_WISE_ADD, var,
                  G::EPS_SCALED, varEps),
              "ln eps") &&
           ok(unary((s + "_rsqrt").c_str(), QNN_OP_ELEMENT_WISE_RSQRT, varEps,
                    invS),
              "ln rsqrt") &&
           ok(bin((s + "_unscale").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, invS,
                  G::CENTER_SCALE, inv),
              "ln unscale") &&
           ok(bin((s + "_xhat").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, center,
                  inv, xhat),
              "ln xhat") &&
           ok(bin((s + "_gamma").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, xhat,
                  ga, xhatG),
              "ln gamma") &&
           ok(bin((s + "_beta").c_str(), QNN_OP_ELEMENT_WISE_ADD, xhatG, be,
                  out),
              "ln beta");
  };
  if (!lnf("tt_ln1", G::X, G::G1, G::B1, G::MEAN1, G::CENTERED1, G::CENTERED_S1,
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
  if (!lnf("tt_ln2", G::RESIDUAL1, G::G2, G::B2, G::MEAN2, G::CENTERED2,
           G::CENTERED_S2, G::SQUARE2, G::VAR2, G::VAR_EPS2, G::INV_STD_S2,
           G::INV_STD2, G::XHAT2, G::XHAT_G2, G::LN2))
    return false;
  if (!ok(mm("tt_ff1", G::LN2, G::W1, G::FF1), "ff1") ||
      !ok(unary("tt_relu", QNN_OP_RELU, G::FF1, G::RELU), "relu") ||
      !ok(mm("tt_ff2", G::RELU, G::W2, G::FF2), "ff2") ||
      !ok(bin("tt_out", QNN_OP_ELEMENT_WISE_ADD, G::RESIDUAL1, G::FF2,
              G::OUTPUT),
          "out") ||
      !ok(bin("tt_error", QNN_OP_ELEMENT_WISE_SUBTRACT, G::OUTPUT, G::TARGET,
              G::ERROR),
          "error") ||
      !ok(bin("tt_sq_error", QNN_OP_ELEMENT_WISE_MULTIPLY, G::ERROR, G::ERROR,
              G::SQUARED_ERROR),
          "sq error") ||
      !ok(reduce("tt_loss", QNN_OP_REDUCE_MEAN, QNN_OP_REDUCE_MEAN_PARAM_AXES,
                 QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, G::SQUARED_ERROR,
                 G::ALL_AXES, true, G::LOSS),
          "loss") ||
      !ok(bin("tt_dout", QNN_OP_ELEMENT_WISE_MULTIPLY, G::ERROR, G::GRAD_SCALE,
              G::DOUTPUT),
          "dout"))
    return false;
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
  auto lnb = [&](const char *p, int dy, int ga, int xhat, int inv, int dxhat,
                 int sum, int prod, int sumprod, int ddx, int first, int xsum,
                 int bracket, int invd, int dx, int dg, int db) {
    std::string s = p;
    return ok(bin((s + "_dxhat").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, dy, ga,
                  dxhat),
              "lnb dxhat") &&
           ok(reduce((s + "_sum").c_str(), QNN_OP_REDUCE_SUM,
                     QNN_OP_REDUCE_SUM_PARAM_AXES,
                     QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dxhat, G::LAST_AXIS,
                     true, sum),
              "lnb sum") &&
           ok(bin((s + "_prod").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, dxhat,
                  xhat, prod),
              "lnb prod") &&
           ok(reduce((s + "_sumprod").c_str(), QNN_OP_REDUCE_SUM,
                     QNN_OP_REDUCE_SUM_PARAM_AXES,
                     QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, prod, G::LAST_AXIS,
                     true, sumprod),
              "lnb sumprod") &&
           ok(bin((s + "_ddx").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, dxhat,
                  G::DIMENSION, ddx),
              "lnb ddx") &&
           ok(bin((s + "_first").c_str(), QNN_OP_ELEMENT_WISE_SUBTRACT, ddx,
                  sum, first),
              "lnb first") &&
           ok(bin((s + "_xsum").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, xhat,
                  sumprod, xsum),
              "lnb xsum") &&
           ok(bin((s + "_bracket").c_str(), QNN_OP_ELEMENT_WISE_SUBTRACT, first,
                  xsum, bracket),
              "lnb bracket") &&
           ok(bin((s + "_invd").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, inv,
                  G::INV_DIMENSION, invd),
              "lnb invd") &&
           ok(bin((s + "_dx").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, bracket,
                  invd, dx),
              "lnb dx") &&
           ok(bin((s + "_dyx").c_str(), QNN_OP_ELEMENT_WISE_MULTIPLY, dy, xhat,
                  prod),
              "lnb dyx") &&
           ok(reduce((s + "_dg").c_str(), QNN_OP_REDUCE_SUM,
                     QNN_OP_REDUCE_SUM_PARAM_AXES,
                     QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, prod, G::ROW_AXIS,
                     false, dg),
              "lnb dg") &&
           ok(reduce((s + "_db").c_str(), QNN_OP_REDUCE_SUM,
                     QNN_OP_REDUCE_SUM_PARAM_AXES,
                     QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, dy, G::ROW_AXIS, false,
                     db),
              "lnb db");
  };
  if (!lnb("tt_lnb2", G::DLN2, G::G2, G::XHAT2, G::INV_STD2, G::DXHAT2,
           G::SUM_DXHAT2, G::DXHAT_XHAT2, G::SUM_DXHAT_XHAT2, G::D_TIMES_DXHAT2,
           G::FIRST_DIFF2, G::XHAT_TIMES_SUM2, G::BRACKET2, G::INV_STD_OVER_D2,
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
  if (!lnb("tt_lnb1", G::DLN1, G::G1, G::XHAT1, G::INV_STD1, G::DXHAT1,
           G::SUM_DXHAT1, G::DXHAT_XHAT1, G::SUM_DXHAT_XHAT1, G::D_TIMES_DXHAT1,
           G::FIRST_DIFF1, G::XHAT_TIMES_SUM1, G::BRACKET1, G::INV_STD_OVER_D1,
           G::CENTERED1, G::DG1, G::DB1))
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
  if (!upd("tt_g1", G::G1, G::DG1, G::S_DG1, G::N_G1) ||
      !upd("tt_b1", G::B1, G::DB1, G::S_DB1, G::N_B1) ||
      !upd("tt_wq", G::WQ, G::DWQ, G::S_DWQ, G::N_WQ) ||
      !upd("tt_wk", G::WK, G::DWK, G::S_DWK, G::N_WK) ||
      !upd("tt_wv", G::WV, G::DWV, G::S_DWV, G::N_WV) ||
      !upd("tt_wo", G::WO, G::DWO, G::S_DWO, G::N_WO) ||
      !upd("tt_g2", G::G2, G::DG2, G::S_DG2, G::N_G2) ||
      !upd("tt_b2", G::B2, G::DB2, G::S_DB2, G::N_B2) ||
      !upd("tt_w1", G::W1, G::DW1, G::S_DW1, G::N_W1) ||
      !upd("tt_w2", G::W2, G::DW2, G::S_DW2, G::N_W2))
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
      "count_per_step=1\ntiny_transformer_training_weight_handoff=CPU_PING_"
      "PONG_BINDING\ntiny_transformer_training_in_place_alias=false\n";
  return true;
}

bool Runtime::executeTinyTransformerTraining(
    const std::vector<float> &input, const std::vector<float> &target,
    const TinyTransformerParameters &current, float learningRate,
    TinyTransformerTrainingOutputs &outputs, std::string &error) {
  using G = Impl::TinyTransformerTrainingGraph;
  auto &g = impl_->tinyTransformerTraining;
  size_t a = size_t(g.tokens) * g.dimension,
         dd = size_t(g.dimension) * g.dimension,
         df = size_t(g.dimension) * g.feedForwardDimension,
         fd = size_t(g.feedForwardDimension) * g.dimension, n = g.dimension;
  if (!g.graph || input.size() != a || target.size() != a ||
      !std::isfinite(learningRate) || current.gamma1.size() != n ||
      current.beta1.size() != n || current.wq.size() != dd ||
      current.wk.size() != dd || current.wv.size() != dd ||
      current.wo.size() != dd || current.gamma2.size() != n ||
      current.beta2.size() != n || current.w1.size() != df ||
      current.w2.size() != fd) {
    error = "tiny training buffer size";
    return false;
  }
  std::vector<Qnn_Tensor_t> in;
  in.reserve(13);
  auto bi = [&](int id, const std::vector<float> &v) {
    in.push_back(g.tensors[id]);
    in.back().v1.clientBuf = {const_cast<float *>(v.data()),
                              uint32_t(v.size() * sizeof(float))};
  };
  bi(G::X, input);
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
  in.push_back(g.tensors[G::LR]);
  in.back().v1.clientBuf = {&learningRate, sizeof(float)};
  std::vector<Qnn_Tensor_t> out;
  out.reserve(23);
  auto bo = [&](int id, std::vector<float> &v, size_t count) {
    v.resize(count);
    out.push_back(g.tensors[id]);
    out.back().v1.clientBuf = {v.data(), uint32_t(count * sizeof(float))};
  };
  out.push_back(g.tensors[G::LOSS]);
  out.back().v1.clientBuf = {&outputs.loss, sizeof(float)};
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
  } else {
    outputs.output.clear();
    outputs.dOutput.clear();
    outputs.gradients = {};
  }
  auto started = Clock::now();
  auto st = impl_->api.graphExecute(g.graph, in.data(), uint32_t(in.size()),
                                    out.data(), uint32_t(out.size()), nullptr,
                                    nullptr);
  metrics_.executeUs.push_back(elapsedUs(started));
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st),
                           st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    error =
        "tiny training graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}
