namespace {
Qnn_ErrorHandle_t addSimpleNode(QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
                                const char* name, const char* type,
                                Qnn_Tensor_t* inputs, uint32_t inputCount,
                                Qnn_Tensor_t* output) {
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = type;
  op.v1.numOfInputs = inputCount;
  op.v1.inputTensors = inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = output;
  return api.graphAddNode(graph, op);
}

Qnn_ErrorHandle_t addMatMulNode(QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
                                const char* name, Qnn_Tensor_t* inputs,
                                Qnn_Tensor_t* output,
                                const char* transposeName = nullptr) {
  Qnn_Scalar_t transpose = QNN_SCALAR_INIT;
  Qnn_Param_t param = QNN_PARAM_INIT;
  uint32_t paramCount = 0;
  if (transposeName) {
    transpose.dataType = QNN_DATATYPE_BOOL_8;
    transpose.bool8Value = true;
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = transposeName;
    param.scalarParam = transpose;
    paramCount = 1;
  }
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_MAT_MUL;
  op.v1.numOfParams = paramCount;
  op.v1.params = paramCount ? &param : nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = output;
  return api.graphAddNode(graph, op);
}

Qnn_ErrorHandle_t addReduceMeanNode(QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
                                    const char* name, Qnn_Tensor_t* input,
                                    Qnn_Tensor_t* axes, Qnn_Tensor_t* output) {
  Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  params[0].paramType = QNN_PARAMTYPE_TENSOR;
  params[0].name = QNN_OP_REDUCE_MEAN_PARAM_AXES;
  params[0].tensorParam = *axes;
  params[1].paramType = QNN_PARAMTYPE_SCALAR;
  params[1].name = QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS;
  params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  params[1].scalarParam.bool8Value = true;
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_REDUCE_MEAN;
  op.v1.numOfParams = 2;
  op.v1.params = params;
  op.v1.numOfInputs = 1;
  op.v1.inputTensors = input;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = output;
  return api.graphAddNode(graph, op);
}
}

bool Runtime::prepareTrainingOpsMicro(uint32_t batchSize,
                                      uint32_t outputDimension,
                                      uint32_t weightRows,
                                      uint32_t weightColumns,
                                      std::string& e) {
  auto* i = impl_;
  if (!i || !i->context || batchSize == 0 || outputDimension == 0 ||
      weightRows == 0 || weightColumns == 0) {
    e = "training ops micro invalid runtime or shape";
    return false;
  }
  auto& m = i->trainingOpsMicro;
  m.predictionDims[0] = batchSize;
  m.predictionDims[1] = outputDimension;
  m.weightDims[0] = weightRows;
  m.weightDims[1] = weightColumns;
  m.gradScale = 2.0f / float(batchSize * outputDimension);
  tensor(m.inputs[0], "micro_p", QNN_TENSOR_TYPE_APP_WRITE, m.predictionDims);
  tensor(m.inputs[1], "micro_y", QNN_TENSOR_TYPE_APP_WRITE, m.predictionDims);
  tensor(m.inputs[2], "micro_w", QNN_TENSOR_TYPE_APP_WRITE, m.weightDims);
  tensor(m.inputs[3], "micro_dw", QNN_TENSOR_TYPE_APP_WRITE, m.weightDims);
  tensor(m.inputs[4], "micro_grad_scale", QNN_TENSOR_TYPE_APP_WRITE, m.scalarDims);
  tensor(m.inputs[5], "micro_lr", QNN_TENSOR_TYPE_APP_WRITE, m.scalarDims);
  tensor(m.error, "micro_error", QNN_TENSOR_TYPE_NATIVE, m.predictionDims);
  tensor(m.squared, "micro_squared", QNN_TENSOR_TYPE_NATIVE, m.predictionDims);
  tensor(m.loss, "micro_loss", QNN_TENSOR_TYPE_APP_READ, m.lossDims);
  tensor(m.dPrediction, "micro_dp", QNN_TENSOR_TYPE_APP_READ, m.predictionDims);
  tensor(m.scaledWeight, "micro_scaled_dw", QNN_TENSOR_TYPE_NATIVE, m.weightDims);
  tensor(m.weightNext, "micro_w_next", QNN_TENSOR_TYPE_APP_READ, m.weightDims);
  tensor(m.axes, "micro_axes", QNN_TENSOR_TYPE_STATIC, m.axesDims,
         QNN_DATATYPE_UINT_32, 1);
  m.axes.v1.clientBuf = {m.axesData, uint32_t(sizeof(m.axesData))};
  auto started = Clock::now();
  auto st = i->api.graphCreate(i->context, "phonelm_training_ops_micro", nullptr,
                               &m.graph);
  metrics_.graphCreateUs += elapsedUs(started);
  ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) {
    e = "training ops micro graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  for (auto& input : m.inputs) {
    if ((st = i->api.tensorCreateGraphTensor(m.graph, &input)) != QNN_SUCCESS) {
      e = "training ops micro input tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  for (auto* t : {&m.error, &m.squared, &m.loss, &m.dPrediction,
                  &m.scaledWeight, &m.weightNext, &m.axes}) {
    if ((st = i->api.tensorCreateGraphTensor(m.graph, t)) != QNN_SUCCESS) {
      e = "training ops micro tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  Qnn_Tensor_t subtractInputs[2] = {m.inputs[0], m.inputs[1]};
  if ((st = addSimpleNode(i->api, m.graph, "micro_subtract", QNN_OP_ELEMENT_WISE_SUBTRACT,
                          subtractInputs, 2, &m.error)) != QNN_SUCCESS) {
    e = "micro Subtract graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  Qnn_Tensor_t squareInputs[2] = {m.error, m.error};
  if ((st = addSimpleNode(i->api, m.graph, "micro_square", QNN_OP_ELEMENT_WISE_MULTIPLY,
                          squareInputs, 2, &m.squared)) != QNN_SUCCESS) {
    e = "micro square graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  if ((st = addReduceMeanNode(i->api, m.graph, "micro_reduce_mean", &m.squared,
                              &m.axes, &m.loss)) != QNN_SUCCESS) {
    e = "micro ReduceMean graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  Qnn_Tensor_t dpInputs[2] = {m.error, m.inputs[4]};
  if ((st = addSimpleNode(i->api, m.graph, "micro_dp", QNN_OP_ELEMENT_WISE_MULTIPLY,
                          dpInputs, 2, &m.dPrediction)) != QNN_SUCCESS) {
    e = "micro dP Multiply graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  Qnn_Tensor_t scaleInputs[2] = {m.inputs[3], m.inputs[5]};
  if ((st = addSimpleNode(i->api, m.graph, "micro_scale_dw", QNN_OP_ELEMENT_WISE_MULTIPLY,
                          scaleInputs, 2, &m.scaledWeight)) != QNN_SUCCESS) {
    e = "micro SGD Multiply graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  Qnn_Tensor_t updateInputs[2] = {m.inputs[2], m.scaledWeight};
  if ((st = addSimpleNode(i->api, m.graph, "micro_update", QNN_OP_ELEMENT_WISE_SUBTRACT,
                          updateInputs, 2, &m.weightNext)) != QNN_SUCCESS) {
    e = "micro SGD Subtract graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  started = Clock::now();
  st = i->api.graphFinalize(m.graph, nullptr, nullptr);
  metrics_.graphFinalizeUs += elapsedUs(started);
  ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    e = "training ops micro graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  diagnostics_ += "training_ops_micro_graph_create=success\n"
                  "training_ops_micro_graph_finalize=success\n"
                  "mse_op_method=ElementWiseSubtract+ElementWiseMultiply+ReduceMean\n"
                  "dp_op_method=ElementWiseMultiply\n"
                  "sgd_op_method=ElementWiseMultiply+ElementWiseSubtract\n"
                  "scalar_broadcast=true\nreduction_axes=0,1\n"
                  "reduce_keep_dims=true\nloss_output_shape=1x1\n";
  return true;
}

bool Runtime::executeTrainingOpsMicro(const std::vector<float>& prediction,
                                      const std::vector<float>& target,
                                      const std::vector<float>& weight,
                                      const std::vector<float>& dWeight,
                                      float learningRate, float& loss,
                                      std::vector<float>& dPrediction,
                                      std::vector<float>& weightNext,
                                      std::string& e) {
  auto* i = impl_;
  if (!i || !i->trainingOpsMicro.graph) { e = "training ops micro not prepared"; return false; }
  auto& m = i->trainingOpsMicro;
  const size_t pn = size_t(m.predictionDims[0]) * m.predictionDims[1];
  const size_t wn = size_t(m.weightDims[0]) * m.weightDims[1];
  if (prediction.size() != pn || target.size() != pn || weight.size() != wn ||
      dWeight.size() != wn) { e = "training ops micro buffer size mismatch"; return false; }
  float gradScale = m.gradScale;
  m.inputs[0].v1.clientBuf = {const_cast<float*>(prediction.data()), uint32_t(pn * sizeof(float))};
  m.inputs[1].v1.clientBuf = {const_cast<float*>(target.data()), uint32_t(pn * sizeof(float))};
  m.inputs[2].v1.clientBuf = {const_cast<float*>(weight.data()), uint32_t(wn * sizeof(float))};
  m.inputs[3].v1.clientBuf = {const_cast<float*>(dWeight.data()), uint32_t(wn * sizeof(float))};
  m.inputs[4].v1.clientBuf = {&gradScale, uint32_t(sizeof(float))};
  m.inputs[5].v1.clientBuf = {&learningRate, uint32_t(sizeof(float))};
  dPrediction.resize(pn); weightNext.resize(wn);
  Qnn_Tensor_t outputs[3] = {m.loss, m.dPrediction, m.weightNext};
  outputs[0].v1.clientBuf = {&loss, uint32_t(sizeof(float))};
  outputs[1].v1.clientBuf = {dPrediction.data(), uint32_t(pn * sizeof(float))};
  outputs[2].v1.clientBuf = {weightNext.data(), uint32_t(wn * sizeof(float))};
  const auto started = Clock::now();
  const auto st = i->api.graphExecute(m.graph, m.inputs, 6, outputs, 3, nullptr, nullptr);
  metrics_.executeUs.push_back(elapsedUs(started));
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st), st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    e = "training ops micro graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
  }
  return true;
}

bool Runtime::prepareMlpFullStep(uint32_t batchSize, uint32_t inputDimension,
                                 uint32_t hiddenDimension,
                                 uint32_t outputDimension,
                                 bool diagnosticOutputs, std::string& e) {
  auto* i = impl_;
  if (!i || !i->context || !batchSize || !inputDimension || !hiddenDimension || !outputDimension) {
    e = "full step invalid runtime or shape"; return false;
  }
  auto& f = i->mlpFullStep;
  f.diagnosticOutputs = diagnosticOutputs;
  f.xDims[0] = batchSize; f.xDims[1] = inputDimension;
  f.yDims[0] = batchSize; f.yDims[1] = outputDimension;
  f.w1Dims[0] = inputDimension; f.w1Dims[1] = hiddenDimension;
  f.w2Dims[0] = hiddenDimension; f.w2Dims[1] = outputDimension;
  f.hiddenDims[0] = batchSize; f.hiddenDims[1] = hiddenDimension;
  f.gradScaleData = 2.0f / float(batchSize * outputDimension);
  f.zeroData.assign(size_t(batchSize) * hiddenDimension, 0.0f);
  tensor(f.inputs[0], "full_x", QNN_TENSOR_TYPE_APP_WRITE, f.xDims);
  tensor(f.inputs[1], "full_y", QNN_TENSOR_TYPE_APP_WRITE, f.yDims);
  tensor(f.inputs[2], "full_w1_current", QNN_TENSOR_TYPE_APP_WRITE, f.w1Dims);
  tensor(f.inputs[3], "full_w2_current", QNN_TENSOR_TYPE_APP_WRITE, f.w2Dims);
  tensor(f.inputs[4], "full_learning_rate", QNN_TENSOR_TYPE_APP_WRITE, f.scalarDims);
  tensor(f.z1, "full_z1", QNN_TENSOR_TYPE_NATIVE, f.hiddenDims);
  const auto diagnosticType = diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  tensor(f.hidden, "full_h", diagnosticType, f.hiddenDims);
  tensor(f.prediction, "full_prediction", diagnosticType, f.yDims);
  tensor(f.error, "full_error", diagnosticType, f.yDims);
  tensor(f.squared, "full_squared_error", QNN_TENSOR_TYPE_NATIVE, f.yDims);
  tensor(f.loss, "full_loss", QNN_TENSOR_TYPE_APP_READ, f.lossDims);
  tensor(f.dPrediction, "full_dp", diagnosticType, f.yDims);
  tensor(f.dW2, "full_dw2", diagnosticType, f.w2Dims);
  tensor(f.dHidden, "full_dh", diagnosticType, f.hiddenDims);
  tensor(f.mask, "full_mask", diagnosticType, f.hiddenDims, QNN_DATATYPE_BOOL_8);
  tensor(f.dZ1, "full_dz1", diagnosticType, f.hiddenDims);
  tensor(f.dW1, "full_dw1", diagnosticType, f.w1Dims);
  tensor(f.scaledDW1, "full_scaled_dw1", QNN_TENSOR_TYPE_NATIVE, f.w1Dims);
  tensor(f.scaledDW2, "full_scaled_dw2", QNN_TENSOR_TYPE_NATIVE, f.w2Dims);
  tensor(f.w1Next, "full_w1_next", QNN_TENSOR_TYPE_APP_READ, f.w1Dims);
  tensor(f.w2Next, "full_w2_next", QNN_TENSOR_TYPE_APP_READ, f.w2Dims);
  tensor(f.axes, "full_axes", QNN_TENSOR_TYPE_STATIC, f.axesDims, QNN_DATATYPE_UINT_32, 1);
  f.axes.v1.clientBuf = {f.axesData, uint32_t(sizeof(f.axesData))};
  tensor(f.gradScale, "full_grad_scale", QNN_TENSOR_TYPE_STATIC, f.scalarDims);
  f.gradScale.v1.clientBuf = {&f.gradScaleData, uint32_t(sizeof(float))};
  tensor(f.zero, "full_zero", QNN_TENSOR_TYPE_STATIC, f.hiddenDims);
  f.zero.v1.clientBuf = {f.zeroData.data(), uint32_t(f.zeroData.size() * sizeof(float))};
  auto started = Clock::now();
  auto st = i->api.graphCreate(i->context, "phonelm_mlp_training_step", nullptr, &f.graph);
  apiTrace_.fullStepGraphCreateCalled = true;
  apiTrace_.fullStepGraphCreateResult = QNN_GET_ERROR_CODE(st);
  apiTrace_.fullStepGraphHandleNonnull = f.graph != nullptr;
  metrics_.graphCreateUs += elapsedUs(started); ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) { e = "full step graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  for (auto& input : f.inputs) {
    if ((st = i->api.tensorCreateGraphTensor(f.graph, &input)) != QNN_SUCCESS) {
      e = "full step input tensor=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
    }
  }
  for (auto* t : {&f.z1, &f.hidden, &f.prediction, &f.error, &f.squared, &f.loss,
                  &f.dPrediction, &f.dW2, &f.dHidden, &f.mask, &f.dZ1, &f.dW1,
                  &f.scaledDW1, &f.scaledDW2, &f.w1Next, &f.w2Next, &f.axes,
                  &f.gradScale, &f.zero}) {
    if ((st = i->api.tensorCreateGraphTensor(f.graph, t)) != QNN_SUCCESS) {
      e = "full step tensor=" + std::to_string(QNN_GET_ERROR_CODE(st)); return false;
    }
  }
  Qnn_Tensor_t zInputs[2] = {f.inputs[0], f.inputs[2]};
  if ((st = addMatMulNode(i->api, f.graph, "full_z1_mm", zInputs, &f.z1)) != QNN_SUCCESS) { e="full Z1 MatMul graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  if ((st = addSimpleNode(i->api, f.graph, "full_relu", QNN_OP_RELU, &f.z1, 1, &f.hidden)) != QNN_SUCCESS) { e="full ReLU graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t pInputs[2] = {f.hidden, f.inputs[3]};
  if ((st = addMatMulNode(i->api, f.graph, "full_prediction_mm", pInputs, &f.prediction)) != QNN_SUCCESS) { e="full P MatMul graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t errorInputs[2] = {f.prediction, f.inputs[1]};
  if ((st = addSimpleNode(i->api, f.graph, "full_error", QNN_OP_ELEMENT_WISE_SUBTRACT, errorInputs, 2, &f.error)) != QNN_SUCCESS) { e="full error Subtract graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t squareInputs[2] = {f.error, f.error};
  if ((st = addSimpleNode(i->api, f.graph, "full_square", QNN_OP_ELEMENT_WISE_MULTIPLY, squareInputs, 2, &f.squared)) != QNN_SUCCESS) { e="full square graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  if ((st = addReduceMeanNode(i->api, f.graph, "full_loss_reduce", &f.squared, &f.axes, &f.loss)) != QNN_SUCCESS) { e="full ReduceMean graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t dpInputs[2] = {f.error, f.gradScale};
  if ((st = addSimpleNode(i->api, f.graph, "full_dp", QNN_OP_ELEMENT_WISE_MULTIPLY, dpInputs, 2, &f.dPrediction)) != QNN_SUCCESS) { e="full dP Multiply graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t dw2Inputs[2] = {f.hidden, f.dPrediction};
  if ((st = addMatMulNode(i->api, f.graph, "full_dw2_mm", dw2Inputs, &f.dW2, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0)) != QNN_SUCCESS) { e="full dW2 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t dhInputs[2] = {f.dPrediction, f.inputs[3]};
  if ((st = addMatMulNode(i->api, f.graph, "full_dh_mm", dhInputs, &f.dHidden, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1)) != QNN_SUCCESS) { e="full dH graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t greaterInputs[2] = {f.hidden, f.zero};
  if ((st = addSimpleNode(i->api, f.graph, "full_relu_greater", QNN_OP_ELEMENT_WISE_GREATER, greaterInputs, 2, &f.mask)) != QNN_SUCCESS) { e="full Greater graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t selectInputs[3] = {f.mask, f.dHidden, f.zero};
  if ((st = addSimpleNode(i->api, f.graph, "full_relu_select", QNN_OP_ELEMENT_WISE_SELECT, selectInputs, 3, &f.dZ1)) != QNN_SUCCESS) { e="full Select graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t dw1Inputs[2] = {f.inputs[0], f.dZ1};
  if ((st = addMatMulNode(i->api, f.graph, "full_dw1_mm", dw1Inputs, &f.dW1, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0)) != QNN_SUCCESS) { e="full dW1 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t scaled1Inputs[2] = {f.dW1, f.inputs[4]};
  if ((st = addSimpleNode(i->api, f.graph, "full_scale_dw1", QNN_OP_ELEMENT_WISE_MULTIPLY, scaled1Inputs, 2, &f.scaledDW1)) != QNN_SUCCESS) { e="full scale dW1 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t update1Inputs[2] = {f.inputs[2], f.scaledDW1};
  if ((st = addSimpleNode(i->api, f.graph, "full_update_w1", QNN_OP_ELEMENT_WISE_SUBTRACT, update1Inputs, 2, &f.w1Next)) != QNN_SUCCESS) { e="full update W1 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t scaled2Inputs[2] = {f.dW2, f.inputs[4]};
  if ((st = addSimpleNode(i->api, f.graph, "full_scale_dw2", QNN_OP_ELEMENT_WISE_MULTIPLY, scaled2Inputs, 2, &f.scaledDW2)) != QNN_SUCCESS) { e="full scale dW2 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  Qnn_Tensor_t update2Inputs[2] = {f.inputs[3], f.scaledDW2};
  if ((st = addSimpleNode(i->api, f.graph, "full_update_w2", QNN_OP_ELEMENT_WISE_SUBTRACT, update2Inputs, 2, &f.w2Next)) != QNN_SUCCESS) { e="full update W2 graphAddNode="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  started = Clock::now();
  st = i->api.graphFinalize(f.graph, nullptr, nullptr);
  apiTrace_.fullStepGraphFinalizeCalled = true;
  apiTrace_.fullStepGraphFinalizeResult = QNN_GET_ERROR_CODE(st);
  metrics_.graphFinalizeUs += elapsedUs(started); ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) { e="full step graphFinalize="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
#ifndef NDEBUG
  if (options_.failGraphFinalize) {
    apiTrace_.failureInjectionEnabled = true;
    apiTrace_.failureInjectionPoint = "graphFinalize";
    apiTrace_.failureInjectionCall = 0;
    apiTrace_.lastQnnResult = QNN_GET_ERROR_CODE(st);
    apiTrace_.effectiveResult = -9002;
    e = "injected graphFinalize failure after QNN success";
    return false;
  }
#endif
  diagnostics_ += "full_step_graph_create=success\nfull_step_graph_finalize=success\n"
                  "full_step_graph_name=phonelm_mlp_training_step\n"
                  "full_step_nodes=16\nfull_step_execute_count_per_step=1\n"
                  "full_step_loss_backend=HTP\nfull_step_dp_backend=HTP\n"
                  "full_step_optimizer_backend=HTP\nfull_step_weight_handoff=CPU_PING_PONG_BINDING\n"
                  "full_step_in_place_alias=false\nfull_step_static_grad_scale=true\n"
                  "full_step_dynamic_learning_rate=true\nfull_step_reduction_axes=0,1\n"
                  "full_step_reduce_keep_dims=true\nfull_step_loss_shape=1x1\n";
  return true;
}

bool Runtime::executeMlpFullStep(const std::vector<float>& input,
                                 const std::vector<float>& target,
                                 const std::vector<float>& w1Current,
                                 const std::vector<float>& w2Current,
                                 float learningRate, MlpFullStepOutputs& out,
                                 std::string& e) {
  auto* i = impl_;
  if (!i || !i->mlpFullStep.graph) { e="full step graph not prepared"; return false; }
  auto& f = i->mlpFullStep;
  const size_t xn=size_t(f.xDims[0])*f.xDims[1], yn=size_t(f.yDims[0])*f.yDims[1];
  const size_t w1n=size_t(f.w1Dims[0])*f.w1Dims[1], w2n=size_t(f.w2Dims[0])*f.w2Dims[1];
  const size_t hn=size_t(f.hiddenDims[0])*f.hiddenDims[1];
  if (input.size()!=xn || target.size()!=yn || w1Current.size()!=w1n || w2Current.size()!=w2n) { e="full step buffer size mismatch"; return false; }
  auto bindStarted=Clock::now();
  f.inputs[0].v1.clientBuf={const_cast<float*>(input.data()),uint32_t(xn*sizeof(float))};
  f.inputs[1].v1.clientBuf={const_cast<float*>(target.data()),uint32_t(yn*sizeof(float))};
  f.inputs[2].v1.clientBuf={const_cast<float*>(w1Current.data()),uint32_t(w1n*sizeof(float))};
  f.inputs[3].v1.clientBuf={const_cast<float*>(w2Current.data()),uint32_t(w2n*sizeof(float))};
  f.inputs[4].v1.clientBuf={&learningRate,uint32_t(sizeof(float))};
  metrics_.inputBindUs.push_back(elapsedUs(bindStarted));
  out.w1Next.resize(w1n); out.w2Next.resize(w2n);
  Qnn_Tensor_t outputs[12]={f.loss,f.w1Next,f.w2Next,f.prediction,f.error,f.dPrediction,
                            f.dW2,f.dHidden,f.mask,f.dZ1,f.dW1,f.hidden};
  bindStarted=Clock::now();
  outputs[0].v1.clientBuf={&out.loss,uint32_t(sizeof(float))};
  outputs[1].v1.clientBuf={out.w1Next.data(),uint32_t(w1n*sizeof(float))};
  outputs[2].v1.clientBuf={out.w2Next.data(),uint32_t(w2n*sizeof(float))};
  uint32_t outputCount=3;
  if (f.diagnosticOutputs) {
    out.prediction.resize(yn); out.hidden.resize(hn); out.error.resize(yn); out.dPrediction.resize(yn);
    out.dW2.resize(w2n); out.dHidden.resize(hn); out.mask.resize(hn);
    out.dZ1.resize(hn); out.dW1.resize(w1n);
    outputs[3].v1.clientBuf={out.prediction.data(),uint32_t(yn*sizeof(float))};
    outputs[4].v1.clientBuf={out.error.data(),uint32_t(yn*sizeof(float))};
    outputs[5].v1.clientBuf={out.dPrediction.data(),uint32_t(yn*sizeof(float))};
    outputs[6].v1.clientBuf={out.dW2.data(),uint32_t(w2n*sizeof(float))};
    outputs[7].v1.clientBuf={out.dHidden.data(),uint32_t(hn*sizeof(float))};
    outputs[8].v1.clientBuf={out.mask.data(),uint32_t(hn)};
    outputs[9].v1.clientBuf={out.dZ1.data(),uint32_t(hn*sizeof(float))};
    outputs[10].v1.clientBuf={out.dW1.data(),uint32_t(w1n*sizeof(float))};
    outputs[11].v1.clientBuf={out.hidden.data(),uint32_t(hn*sizeof(float))};
    outputCount=12;
  } else {
    out.prediction.clear(); out.hidden.clear(); out.error.clear(); out.dPrediction.clear(); out.dW2.clear();
    out.dHidden.clear(); out.mask.clear(); out.dZ1.clear(); out.dW1.clear();
  }
  metrics_.outputBindUs.push_back(elapsedUs(bindStarted));
  const auto started=Clock::now();
  const auto st=i->api.graphExecute(f.graph,f.inputs,5,outputs,outputCount,nullptr,nullptr);
  metrics_.executeUs.push_back(elapsedUs(started)); ++metrics_.graphExecuteCount;
  const auto callIndex = apiTrace_.graphExecuteAttemptCount;
#ifndef NDEBUG
  const bool injected = st == QNN_SUCCESS && options_.failGraphExecuteAt >= 0 &&
                        callIndex == static_cast<std::uint64_t>(options_.failGraphExecuteAt);
#else
  const bool injected = false;
#endif
  const int effectiveResult = injected ? -9001 : QNN_GET_ERROR_CODE(st);
  if (injected) {
    apiTrace_.failureInjectionEnabled = true;
    apiTrace_.failureInjectionPoint = "graphExecute";
    apiTrace_.failureInjectionCall = static_cast<std::int64_t>(callIndex);
  }
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), effectiveResult,
                           st == QNN_SUCCESS && !injected);
  if (injected) { e="injected graphExecute failure after QNN success"; return false; }
  if (st!=QNN_SUCCESS) { e="full step graphExecute="+std::to_string(QNN_GET_ERROR_CODE(st)); return false; }
  return true;
}