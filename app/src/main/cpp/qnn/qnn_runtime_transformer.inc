// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku

bool Runtime::prepareLayerNorm(uint32_t batch, uint32_t tokens,
                               uint32_t dimension, float epsilon,
                               std::string &error) {
  if (!impl_ || !impl_->context || batch == 0 || tokens == 0 ||
      dimension == 0) {
    error = "LayerNorm invalid runtime or shape";
    return false;
  }
  auto &g = impl_->layerNorm;
  g.rank = 3;
  g.dims[0] = batch;
  g.dims[1] = tokens;
  g.dims[2] = dimension;
  g.elements = batch * tokens * dimension;
  g.normDims[0] = dimension;
  g.axesData[0] = 2;
  g.epsilon = epsilon;
  g.gammaData.assign(dimension, 1.0f);
  g.betaData.assign(dimension, 0.0f);
  tensor(g.input, "transformer_ln_input", QNN_TENSOR_TYPE_APP_WRITE, g.dims,
         QNN_DATATYPE_FLOAT_32, g.rank);
  tensor(g.output, "transformer_ln_output", QNN_TENSOR_TYPE_APP_READ, g.dims,
         QNN_DATATYPE_FLOAT_32, g.rank);
  tensor(g.gamma, "transformer_ln_gamma", QNN_TENSOR_TYPE_STATIC, g.normDims);
  tensor(g.beta, "transformer_ln_beta", QNN_TENSOR_TYPE_STATIC, g.normDims);
  tensor(g.axes, "transformer_ln_axes", QNN_TENSOR_TYPE_STATIC, g.axesDims,
         QNN_DATATYPE_UINT_32, 1);
  g.gamma.v1.clientBuf = {g.gammaData.data(),
                          uint32_t(g.gammaData.size() * sizeof(float))};
  g.beta.v1.clientBuf = {g.betaData.data(),
                         uint32_t(g.betaData.size() * sizeof(float))};
  g.axes.v1.clientBuf = {g.axesData, uint32_t(sizeof(g.axesData))};
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_layer_norm",
                                       nullptr, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error =
        "LayerNorm graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  for (auto *value : {&g.input, &g.output, &g.gamma, &g.beta, &g.axes}) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, value);
    if (status != QNN_SUCCESS) {
      error = "LayerNorm tensor=" + std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  params[0].paramType = QNN_PARAMTYPE_SCALAR;
  params[0].name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
  params[0].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  params[0].scalarParam.floatValue = epsilon;
  params[1].paramType = QNN_PARAMTYPE_TENSOR;
  params[1].name = QNN_OP_LAYER_NORM_PARAM_AXES;
  params[1].tensorParam = g.axes;
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "transformer_layer_norm";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_LAYER_NORM;
  op.v1.numOfParams = 2;
  op.v1.params = params;
  Qnn_Tensor_t inputs[3] = {g.input, g.gamma, g.beta};
  op.v1.numOfInputs = 3;
  op.v1.inputTensors = inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &g.output;
  status = impl_->api.graphAddNode(g.graph, op);
  if (status != QNN_SUCCESS) {
    error =
        "LayerNorm graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error =
        "LayerNorm graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ += "transformer_layer_norm_method=DIRECT_OP\ntransformer_layer_"
                  "norm_finalize=0\n";
  return true;
}

bool Runtime::executeLayerNorm(const std::vector<float> &input,
                               std::vector<float> &output, std::string &error) {
  auto &g = impl_->layerNorm;
  if (!g.graph || input.size() != g.elements) {
    error = "LayerNorm input size";
    return false;
  }
  output.resize(g.elements);
  Qnn_Tensor_t in = g.input, out = g.output;
  in.v1.clientBuf = {const_cast<float *>(input.data()),
                     uint32_t(input.size() * sizeof(float))};
  out.v1.clientBuf = {output.data(), uint32_t(output.size() * sizeof(float))};
  const auto status =
      impl_->api.graphExecute(g.graph, &in, 1, &out, 1, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error =
        "LayerNorm graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  return true;
}

bool Runtime::prepareSoftmax(uint32_t rows, uint32_t columns,
                             std::string &error) {
  if (!impl_ || !impl_->context || rows == 0 || columns == 0) {
    error = "Softmax invalid runtime or shape";
    return false;
  }
  auto &g = impl_->softmax;
  g.rank = 2;
  g.dims[0] = rows;
  g.dims[1] = columns;
  g.elements = rows * columns;
  tensor(g.input, "transformer_softmax_input", QNN_TENSOR_TYPE_APP_WRITE,
         g.dims);
  tensor(g.output, "transformer_softmax_output", QNN_TENSOR_TYPE_APP_READ,
         g.dims);
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_softmax",
                                       nullptr, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error = "Softmax graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  for (auto *value : {&g.input, &g.output}) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, value);
    if (status != QNN_SUCCESS) {
      error = "Softmax tensor=" + std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  params[0].paramType = QNN_PARAMTYPE_SCALAR;
  params[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
  params[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  params[0].scalarParam.uint32Value = 1;
  params[1].paramType = QNN_PARAMTYPE_SCALAR;
  params[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
  params[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  params[1].scalarParam.floatValue = 1.0f;
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "transformer_softmax";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_SOFTMAX;
  op.v1.numOfParams = 2;
  op.v1.params = params;
  op.v1.numOfInputs = 1;
  op.v1.inputTensors = &g.input;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &g.output;
  status = impl_->api.graphAddNode(g.graph, op);
  if (status != QNN_SUCCESS) {
    error =
        "Softmax graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error =
        "Softmax graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ +=
      "transformer_softmax_method=DIRECT_OP\ntransformer_softmax_finalize=0\n";
  return true;
}

bool Runtime::executeSoftmax(const std::vector<float> &input,
                             std::vector<float> &output, std::string &error) {
  auto &g = impl_->softmax;
  if (!g.graph || input.size() != g.elements) {
    error = "Softmax input size";
    return false;
  }
  output.resize(g.elements);
  Qnn_Tensor_t in = g.input, out = g.output;
  in.v1.clientBuf = {const_cast<float *>(input.data()),
                     uint32_t(input.size() * sizeof(float))};
  out.v1.clientBuf = {output.data(), uint32_t(output.size() * sizeof(float))};
  const auto status =
      impl_->api.graphExecute(g.graph, &in, 1, &out, 1, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error =
        "Softmax graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  return true;
}
bool Runtime::prepareSoftmaxBackward(uint32_t rows, uint32_t columns,
                                     std::string &error) {
  if (!impl_ || !impl_->context || rows == 0 || columns == 0) {
    error = "Softmax backward invalid runtime or shape";
    return false;
  }
  auto &g = impl_->softmaxBackward;
  g.dims[0] = rows;
  g.dims[1] = columns;
  g.reducedDims[0] = rows;
  g.reducedDims[1] = 1;
  g.elements = rows * columns;
  tensor(g.probabilities, "softmax_backward_probabilities",
         QNN_TENSOR_TYPE_APP_WRITE, g.dims);
  tensor(g.upstream, "softmax_backward_upstream", QNN_TENSOR_TYPE_APP_WRITE,
         g.dims);
  tensor(g.product, "softmax_backward_product", QNN_TENSOR_TYPE_NATIVE,
         g.dims);
  tensor(g.dot, "softmax_backward_dot", QNN_TENSOR_TYPE_NATIVE,
         g.reducedDims);
  tensor(g.centered, "softmax_backward_centered", QNN_TENSOR_TYPE_NATIVE,
         g.dims);
  tensor(g.inputGradient, "softmax_backward_dx", QNN_TENSOR_TYPE_APP_READ,
         g.dims);
  tensor(g.axes, "softmax_backward_axes", QNN_TENSOR_TYPE_STATIC, g.axesDims,
         QNN_DATATYPE_UINT_32, 1);
  g.axes.v1.clientBuf = {g.axesData, uint32_t(sizeof(g.axesData))};
  auto status = impl_->api.graphCreate(
      impl_->context, "phonelm_softmax_backward", nullptr, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error = "Softmax backward graphCreate=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  for (auto *item : {&g.probabilities, &g.upstream, &g.product, &g.dot,
                     &g.centered, &g.inputGradient, &g.axes}) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, item);
    if (status != QNN_SUCCESS) {
      error = "Softmax backward tensor=" +
              std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  Qnn_Tensor_t multiplyInputs[2] = {g.upstream, g.probabilities};
  if ((status = addSimpleNode(impl_->api, g.graph, "softmax_backward_ds_s",
                              QNN_OP_ELEMENT_WISE_MULTIPLY, multiplyInputs, 2,
                              &g.product)) != QNN_SUCCESS) {
    error = "Softmax backward product=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Param_t reduceParams[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  reduceParams[0].paramType = QNN_PARAMTYPE_TENSOR;
  reduceParams[0].name = QNN_OP_REDUCE_SUM_PARAM_AXES;
  reduceParams[0].tensorParam = g.axes;
  reduceParams[1].paramType = QNN_PARAMTYPE_SCALAR;
  reduceParams[1].name = QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS;
  reduceParams[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  reduceParams[1].scalarParam.bool8Value = true;
  Qnn_OpConfig_t reduce = QNN_OPCONFIG_INIT;
  reduce.v1.name = "softmax_backward_reduce";
  reduce.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  reduce.v1.typeName = QNN_OP_REDUCE_SUM;
  reduce.v1.numOfParams = 2;
  reduce.v1.params = reduceParams;
  reduce.v1.numOfInputs = 1;
  reduce.v1.inputTensors = &g.product;
  reduce.v1.numOfOutputs = 1;
  reduce.v1.outputTensors = &g.dot;
  if ((status = impl_->api.graphAddNode(g.graph, reduce)) != QNN_SUCCESS) {
    error = "Softmax backward reduce=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t subtractInputs[2] = {g.upstream, g.dot};
  if ((status = addSimpleNode(impl_->api, g.graph, "softmax_backward_center",
                              QNN_OP_ELEMENT_WISE_SUBTRACT, subtractInputs, 2,
                              &g.centered)) != QNN_SUCCESS) {
    error = "Softmax backward subtract=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t outputInputs[2] = {g.probabilities, g.centered};
  if ((status = addSimpleNode(impl_->api, g.graph, "softmax_backward_output",
                              QNN_OP_ELEMENT_WISE_MULTIPLY, outputInputs, 2,
                              &g.inputGradient)) != QNN_SUCCESS) {
    error = "Softmax backward output=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error = "Softmax backward graphFinalize=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ += "transformer_softmax_backward_method=PRIMITIVE_COMPOSITION\n"
                  "transformer_softmax_backward_axis=last\n"
                  "transformer_softmax_backward_finalize=0\n";
  return true;
}

bool Runtime::executeSoftmaxBackward(
    const std::vector<float> &probabilities,
    const std::vector<float> &upstream, std::vector<float> &inputGradient,
    std::string &error) {
  auto &g = impl_->softmaxBackward;
  if (!g.graph || probabilities.size() != g.elements ||
      upstream.size() != g.elements) {
    error = "Softmax backward input size";
    return false;
  }
  inputGradient.resize(g.elements);
  Qnn_Tensor_t inputs[2] = {g.probabilities, g.upstream};
  inputs[0].v1.clientBuf = {
      const_cast<float *>(probabilities.data()),
      uint32_t(probabilities.size() * sizeof(float))};
  inputs[1].v1.clientBuf = {const_cast<float *>(upstream.data()),
                            uint32_t(upstream.size() * sizeof(float))};
  Qnn_Tensor_t output = g.inputGradient;
  output.v1.clientBuf = {inputGradient.data(),
                         uint32_t(inputGradient.size() * sizeof(float))};
  const auto status =
      impl_->api.graphExecute(g.graph, inputs, 2, &output, 1, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error = "Softmax backward graphExecute=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  return true;
}
bool Runtime::prepareAttention(uint32_t tokens, uint32_t headDimension,
                               std::string &error) {
  if (!impl_ || !impl_->context || tokens == 0 || headDimension == 0) {
    error = "Attention invalid runtime or shape";
    return false;
  }
  auto &g = impl_->attention;
  g.tokens = tokens;
  g.headDimension = headDimension;
  g.activationDims[0] = tokens;
  g.activationDims[1] = headDimension;
  g.scoreDims[0] = tokens;
  g.scoreDims[1] = tokens;
  g.scaleData = 1.0f / std::sqrt(static_cast<float>(headDimension));
  tensor(g.query, "attention_query", QNN_TENSOR_TYPE_APP_WRITE,
         g.activationDims);
  tensor(g.key, "attention_key", QNN_TENSOR_TYPE_APP_WRITE, g.activationDims);
  tensor(g.value, "attention_value", QNN_TENSOR_TYPE_APP_WRITE,
         g.activationDims);
  tensor(g.mask, "attention_mask", QNN_TENSOR_TYPE_APP_WRITE, g.scoreDims);
  tensor(g.scores, "attention_scores", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.scaled, "attention_scaled", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.masked, "attention_masked", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.probabilities, "attention_probabilities", QNN_TENSOR_TYPE_APP_READ,
         g.scoreDims);
  tensor(g.output, "attention_output", QNN_TENSOR_TYPE_APP_READ,
         g.activationDims);
  tensor(g.scale, "attention_scale", QNN_TENSOR_TYPE_STATIC, g.scalarDims);
  g.scale.v1.clientBuf = {&g.scaleData, uint32_t(sizeof(g.scaleData))};
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_attention",
                                       nullptr, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error =
        "Attention graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  for (auto *value : {&g.query, &g.key, &g.value, &g.mask, &g.scores, &g.scaled,
                      &g.masked, &g.probabilities, &g.output, &g.scale}) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, value);
    if (status != QNN_SUCCESS) {
      error = "Attention tensor=" + std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  Qnn_Tensor_t qk[2] = {g.query, g.key};
  if ((status = addMatMulNode(impl_->api, g.graph, "attention_qk", qk,
                              &g.scores, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1)) !=
      QNN_SUCCESS) {
    error = "Attention QK MatMul=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t scaleInputs[2] = {g.scores, g.scale};
  if ((status = addSimpleNode(impl_->api, g.graph, "attention_scale_scores",
                              QNN_OP_ELEMENT_WISE_MULTIPLY, scaleInputs, 2,
                              &g.scaled)) != QNN_SUCCESS) {
    error = "Attention scale=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t maskInputs[2] = {g.scaled, g.mask};
  if ((status = addSimpleNode(impl_->api, g.graph, "attention_causal_mask",
                              QNN_OP_ELEMENT_WISE_ADD, maskInputs, 2,
                              &g.masked)) != QNN_SUCCESS) {
    error = "Attention mask=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Param_t softmaxParams[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  softmaxParams[0].paramType = QNN_PARAMTYPE_SCALAR;
  softmaxParams[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
  softmaxParams[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  softmaxParams[0].scalarParam.uint32Value = 1;
  softmaxParams[1].paramType = QNN_PARAMTYPE_SCALAR;
  softmaxParams[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
  softmaxParams[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  softmaxParams[1].scalarParam.floatValue = 1.0f;
  Qnn_OpConfig_t softmax = QNN_OPCONFIG_INIT;
  softmax.v1.name = "attention_softmax";
  softmax.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  softmax.v1.typeName = QNN_OP_SOFTMAX;
  softmax.v1.numOfParams = 2;
  softmax.v1.params = softmaxParams;
  softmax.v1.numOfInputs = 1;
  softmax.v1.inputTensors = &g.masked;
  softmax.v1.numOfOutputs = 1;
  softmax.v1.outputTensors = &g.probabilities;
  if ((status = impl_->api.graphAddNode(g.graph, softmax)) != QNN_SUCCESS) {
    error = "Attention Softmax=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t pv[2] = {g.probabilities, g.value};
  if ((status = addMatMulNode(impl_->api, g.graph, "attention_pv", pv,
                              &g.output)) != QNN_SUCCESS) {
    error = "Attention PV MatMul=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error =
        "Attention graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ += "transformer_attention_method=COMPOSED_QNN_OPS\ntransformer_"
                  "attention_finalize=0\n";
  return true;
}

bool Runtime::executeAttention(const std::vector<float> &query,
                               const std::vector<float> &key,
                               const std::vector<float> &value,
                               const std::vector<float> &causalMask,
                               std::vector<float> &output,
                               std::vector<float> &probabilities,
                               std::string &error) {
  auto &g = impl_->attention;
  const size_t activationElements = size_t(g.tokens) * g.headDimension;
  const size_t scoreElements = size_t(g.tokens) * g.tokens;
  if (!g.graph || query.size() != activationElements ||
      key.size() != activationElements || value.size() != activationElements ||
      causalMask.size() != scoreElements) {
    error = "Attention input size";
    return false;
  }
  output.resize(activationElements);
  probabilities.resize(scoreElements);
  Qnn_Tensor_t inputs[4] = {g.query, g.key, g.value, g.mask};
  inputs[0].v1.clientBuf = {const_cast<float *>(query.data()),
                            uint32_t(query.size() * sizeof(float))};
  inputs[1].v1.clientBuf = {const_cast<float *>(key.data()),
                            uint32_t(key.size() * sizeof(float))};
  inputs[2].v1.clientBuf = {const_cast<float *>(value.data()),
                            uint32_t(value.size() * sizeof(float))};
  inputs[3].v1.clientBuf = {const_cast<float *>(causalMask.data()),
                            uint32_t(causalMask.size() * sizeof(float))};
  Qnn_Tensor_t outputs[2] = {g.output, g.probabilities};
  outputs[0].v1.clientBuf = {output.data(),
                             uint32_t(output.size() * sizeof(float))};
  outputs[1].v1.clientBuf = {probabilities.data(),
                             uint32_t(probabilities.size() * sizeof(float))};
  const auto status =
      impl_->api.graphExecute(g.graph, inputs, 4, outputs, 2, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error =
        "Attention graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  return true;
}
bool Runtime::prepareTinyTransformer(uint32_t tokens, uint32_t dimension,
                                     uint32_t feedForwardDimension,
                                     float epsilon, std::string &error) {
  if (!impl_ || !impl_->context || tokens == 0 || dimension == 0 ||
      feedForwardDimension == 0) {
    error = "Tiny transformer invalid runtime or shape";
    return false;
  }
  auto &g = impl_->tinyTransformer;
  g.tokens = tokens;
  g.dimension = dimension;
  g.feedForwardDimension = feedForwardDimension;
  g.epsilon = epsilon;
  g.activationDims[0] = tokens;
  g.activationDims[1] = dimension;
  g.scoreDims[0] = tokens;
  g.scoreDims[1] = tokens;
  g.ffDims[0] = tokens;
  g.ffDims[1] = feedForwardDimension;
  g.weightDdDims[0] = dimension;
  g.weightDdDims[1] = dimension;
  g.weightDfDims[0] = dimension;
  g.weightDfDims[1] = feedForwardDimension;
  g.weightFdDims[0] = feedForwardDimension;
  g.weightFdDims[1] = dimension;
  g.normDims[0] = dimension;
  g.scaleData = 1.0f / std::sqrt(static_cast<float>(dimension));
  g.gammaData.assign(dimension, 1.0f);
  g.betaData.assign(dimension, 0.0f);
  auto fillWeight = [](std::vector<float> &values, size_t count, int phase) {
    values.resize(count);
    for (size_t i = 0; i < count; ++i)
      values[i] = float(int((i * 17 + size_t(phase) * 13) % 29) - 14) * 0.01f;
  };
  fillWeight(g.wqData, size_t(dimension) * dimension, 1);
  fillWeight(g.wkData, size_t(dimension) * dimension, 2);
  fillWeight(g.wvData, size_t(dimension) * dimension, 3);
  fillWeight(g.woData, size_t(dimension) * dimension, 4);
  fillWeight(g.w1Data, size_t(dimension) * feedForwardDimension, 5);
  fillWeight(g.w2Data, size_t(feedForwardDimension) * dimension, 6);
  g.maskData.assign(size_t(tokens) * tokens, 0.0f);
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t column = row + 1; column < tokens; ++column)
      g.maskData[row * tokens + column] = -10000.0f;
  tensor(g.input, "tiny_input", QNN_TENSOR_TYPE_APP_WRITE, g.activationDims);
  tensor(g.output, "tiny_output", QNN_TENSOR_TYPE_APP_READ, g.activationDims);
  tensor(g.ln1, "tiny_ln1_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.q, "tiny_q_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.k, "tiny_k_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.v, "tiny_v_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.context, "tiny_context", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.projected, "tiny_projected", QNN_TENSOR_TYPE_NATIVE,
         g.activationDims);
  tensor(g.residual1, "tiny_residual1_out", QNN_TENSOR_TYPE_NATIVE,
         g.activationDims);
  tensor(g.ln2, "tiny_ln2_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.ff2, "tiny_ff2_out", QNN_TENSOR_TYPE_NATIVE, g.activationDims);
  tensor(g.scores, "tiny_scores", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.scaled, "tiny_scaled", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.masked, "tiny_masked", QNN_TENSOR_TYPE_NATIVE, g.scoreDims);
  tensor(g.probabilities, "tiny_probabilities", QNN_TENSOR_TYPE_NATIVE,
         g.scoreDims);
  tensor(g.ff1, "tiny_ff1_out", QNN_TENSOR_TYPE_NATIVE, g.ffDims);
  tensor(g.relu, "tiny_relu_out", QNN_TENSOR_TYPE_NATIVE, g.ffDims);
  auto makeStatic = [&](Qnn_Tensor_t &item, const char *name, uint32_t *dims,
                        std::vector<float> &data) {
    tensor(item, name, QNN_TENSOR_TYPE_STATIC, dims);
    item.v1.clientBuf = {data.data(), uint32_t(data.size() * sizeof(float))};
  };
  makeStatic(g.gamma1, "tiny_gamma1", g.normDims, g.gammaData);
  makeStatic(g.beta1, "tiny_beta1", g.normDims, g.betaData);
  makeStatic(g.gamma2, "tiny_gamma2", g.normDims, g.gammaData);
  makeStatic(g.beta2, "tiny_beta2", g.normDims, g.betaData);
  makeStatic(g.wq, "tiny_wq", g.weightDdDims, g.wqData);
  makeStatic(g.wk, "tiny_wk", g.weightDdDims, g.wkData);
  makeStatic(g.wv, "tiny_wv", g.weightDdDims, g.wvData);
  makeStatic(g.wo, "tiny_wo", g.weightDdDims, g.woData);
  makeStatic(g.w1, "tiny_w1", g.weightDfDims, g.w1Data);
  makeStatic(g.w2, "tiny_w2", g.weightFdDims, g.w2Data);
  makeStatic(g.mask, "tiny_mask", g.scoreDims, g.maskData);
  tensor(g.scale, "tiny_scale", QNN_TENSOR_TYPE_STATIC, g.scalarDims);
  g.scale.v1.clientBuf = {&g.scaleData, uint32_t(sizeof(g.scaleData))};
  tensor(g.axes, "tiny_ln_axes", QNN_TENSOR_TYPE_STATIC, g.axesDims,
         QNN_DATATYPE_UINT_32, 1);
  g.axes.v1.clientBuf = {g.axesData, uint32_t(sizeof(g.axesData))};
  auto status = impl_->api.graphCreate(
      impl_->context, "phonelm_tiny_transformer", nullptr, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error = "Tiny graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  for (auto *item :
       {&g.input,   &g.output,    &g.ln1,       &g.q,      &g.k,
        &g.v,       &g.scores,    &g.scaled,    &g.masked, &g.probabilities,
        &g.context, &g.projected, &g.residual1, &g.ln2,    &g.ff1,
        &g.relu,    &g.ff2,       &g.gamma1,    &g.beta1,  &g.gamma2,
        &g.beta2,   &g.wq,        &g.wk,        &g.wv,     &g.wo,
        &g.w1,      &g.w2,        &g.mask,      &g.scale,  &g.axes}) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, item);
    if (status != QNN_SUCCESS) {
      error = "Tiny tensor=" + std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  auto addLn = [&](const char *name, Qnn_Tensor_t &input, Qnn_Tensor_t &gamma,
                   Qnn_Tensor_t &beta, Qnn_Tensor_t &output) {
    Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    params[0].paramType = QNN_PARAMTYPE_SCALAR;
    params[0].name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
    params[0].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    params[0].scalarParam.floatValue = epsilon;
    params[1].paramType = QNN_PARAMTYPE_TENSOR;
    params[1].name = QNN_OP_LAYER_NORM_PARAM_AXES;
    params[1].tensorParam = g.axes;
    Qnn_Tensor_t inputs[3] = {input, gamma, beta};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = name;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_LAYER_NORM;
    op.v1.numOfParams = 2;
    op.v1.params = params;
    op.v1.numOfInputs = 3;
    op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &output;
    return impl_->api.graphAddNode(g.graph, op);
  };
  if ((status = addLn("tiny_ln1", g.input, g.gamma1, g.beta1, g.ln1)) !=
      QNN_SUCCESS) {
    error = "Tiny LN1=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t qInputs[2] = {g.ln1, g.wq}, kInputs[2] = {g.ln1, g.wk},
               vInputs[2] = {g.ln1, g.wv};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_q", qInputs, &g.q)) !=
      QNN_SUCCESS) {
    error = "Tiny Q=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_k", kInputs, &g.k)) !=
      QNN_SUCCESS) {
    error = "Tiny K=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_v", vInputs, &g.v)) !=
      QNN_SUCCESS) {
    error = "Tiny V=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t qk[2] = {g.q, g.k};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_qk", qk, &g.scores,
                              QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1)) !=
      QNN_SUCCESS) {
    error = "Tiny QK=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t scaleInputs[2] = {g.scores, g.scale};
  if ((status = addSimpleNode(impl_->api, g.graph, "tiny_scale_scores",
                              QNN_OP_ELEMENT_WISE_MULTIPLY, scaleInputs, 2,
                              &g.scaled)) != QNN_SUCCESS) {
    error = "Tiny scale=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t maskInputs[2] = {g.scaled, g.mask};
  if ((status = addSimpleNode(impl_->api, g.graph, "tiny_mask_scores",
                              QNN_OP_ELEMENT_WISE_ADD, maskInputs, 2,
                              &g.masked)) != QNN_SUCCESS) {
    error = "Tiny mask=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Param_t smp[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  smp[0].paramType = QNN_PARAMTYPE_SCALAR;
  smp[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
  smp[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  smp[0].scalarParam.uint32Value = 1;
  smp[1].paramType = QNN_PARAMTYPE_SCALAR;
  smp[1].name = QNN_OP_SOFTMAX_PARAM_BETA;
  smp[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  smp[1].scalarParam.floatValue = 1.0f;
  Qnn_OpConfig_t sm = QNN_OPCONFIG_INIT;
  sm.v1.name = "tiny_softmax";
  sm.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  sm.v1.typeName = QNN_OP_SOFTMAX;
  sm.v1.numOfParams = 2;
  sm.v1.params = smp;
  sm.v1.numOfInputs = 1;
  sm.v1.inputTensors = &g.masked;
  sm.v1.numOfOutputs = 1;
  sm.v1.outputTensors = &g.probabilities;
  if ((status = impl_->api.graphAddNode(g.graph, sm)) != QNN_SUCCESS) {
    error = "Tiny Softmax=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t pv[2] = {g.probabilities, g.v};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_pv", pv,
                              &g.context)) != QNN_SUCCESS) {
    error = "Tiny PV=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t po[2] = {g.context, g.wo};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_projection", po,
                              &g.projected)) != QNN_SUCCESS) {
    error = "Tiny projection=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t r1[2] = {g.input, g.projected};
  if ((status = addSimpleNode(impl_->api, g.graph, "tiny_residual1",
                              QNN_OP_ELEMENT_WISE_ADD, r1, 2, &g.residual1)) !=
      QNN_SUCCESS) {
    error = "Tiny residual1=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if ((status = addLn("tiny_ln2", g.residual1, g.gamma2, g.beta2, g.ln2)) !=
      QNN_SUCCESS) {
    error = "Tiny LN2=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t f1[2] = {g.ln2, g.w1};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_ff1", f1, &g.ff1)) !=
      QNN_SUCCESS) {
    error = "Tiny FF1=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if ((status = addSimpleNode(impl_->api, g.graph, "tiny_relu", QNN_OP_RELU,
                              &g.ff1, 1, &g.relu)) != QNN_SUCCESS) {
    error = "Tiny ReLU=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t f2[2] = {g.relu, g.w2};
  if ((status = addMatMulNode(impl_->api, g.graph, "tiny_ff2", f2, &g.ff2)) !=
      QNN_SUCCESS) {
    error = "Tiny FF2=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  Qnn_Tensor_t r2[2] = {g.residual1, g.ff2};
  if ((status = addSimpleNode(impl_->api, g.graph, "tiny_residual2",
                              QNN_OP_ELEMENT_WISE_ADD, r2, 2, &g.output)) !=
      QNN_SUCCESS) {
    error = "Tiny residual2=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error = "Tiny graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ +=
      "transformer_tiny_method=FUSED_QNN_GRAPH\ntransformer_tiny_finalize=0\n";
  return true;
}

bool Runtime::executeTinyTransformer(const std::vector<float> &input,
                                     std::vector<float> &output,
                                     std::string &error) {
  auto &g = impl_->tinyTransformer;
  const size_t count = size_t(g.tokens) * g.dimension;
  if (!g.graph || input.size() != count) {
    error = "Tiny transformer input size";
    return false;
  }
  output.resize(count);
  Qnn_Tensor_t in = g.input, out = g.output;
  in.v1.clientBuf = {const_cast<float *>(input.data()),
                     uint32_t(input.size() * sizeof(float))};
  out.v1.clientBuf = {output.data(), uint32_t(output.size() * sizeof(float))};
  const auto status =
      impl_->api.graphExecute(g.graph, &in, 1, &out, 1, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error = "Tiny graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  return true;
}