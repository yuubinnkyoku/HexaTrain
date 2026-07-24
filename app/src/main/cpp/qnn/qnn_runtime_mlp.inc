template <class I, class S>
static bool prepareSlot(I *i, RuntimeMetrics &metrics, S &s,
                        const char *graphName, uint32_t a0, uint32_t a1,
                        uint32_t b0, uint32_t b1, uint32_t o0, uint32_t o1,
                        bool trans0, bool trans1, std::string &error) {
  s.d0[0] = a0;
  s.d0[1] = a1;
  s.d1[0] = b0;
  s.d1[1] = b1;
  s.doo[0] = o0;
  s.doo[1] = o1;
  s.n0 = a0 * a1;
  s.n1 = b0 * b1;
  s.no = o0 * o1;
  tensor(s.in[0], "a", QNN_TENSOR_TYPE_APP_WRITE, s.d0);
  tensor(s.in[1], "b", QNN_TENSOR_TYPE_APP_WRITE, s.d1);
  tensor(s.out, "out", QNN_TENSOR_TYPE_APP_READ, s.doo);
  auto st = i->api.graphCreate(i->context, graphName, nullptr, &s.graph);
  ++metrics.graphCreateCount;
  if (st != QNN_SUCCESS) {
    error = std::string(graphName) +
            " graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  for (auto &t : s.in) {
    st = i->api.tensorCreateGraphTensor(s.graph, &t);
    if (st != QNN_SUCCESS) {
      error = std::string(graphName) +
              " input tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  st = i->api.tensorCreateGraphTensor(s.graph, &s.out);
  if (st != QNN_SUCCESS) {
    error = std::string(graphName) +
            " output tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Param_t ps[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  Qnn_Scalar_t sc[2] = {QNN_SCALAR_INIT, QNN_SCALAR_INIT};
  int pc = 0;
  if (trans0) {
    sc[pc].dataType = QNN_DATATYPE_BOOL_8;
    sc[pc].bool8Value = true;
    ps[pc].paramType = QNN_PARAMTYPE_SCALAR;
    ps[pc].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    ps[pc].scalarParam = sc[pc];
    ++pc;
  }
  if (trans1) {
    sc[pc].dataType = QNN_DATATYPE_BOOL_8;
    sc[pc].bool8Value = true;
    ps[pc].paramType = QNN_PARAMTYPE_SCALAR;
    ps[pc].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    ps[pc].scalarParam = sc[pc];
    ++pc;
  }
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "matmul";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_MAT_MUL;
  op.v1.numOfParams = pc;
  op.v1.params = pc ? ps : nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = s.in;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &s.out;
  st = i->api.graphAddNode(s.graph, op);
  if (st != QNN_SUCCESS) {
    error = std::string(graphName) +
            " graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  st = i->api.graphFinalize(s.graph, nullptr, nullptr);
  ++metrics.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    error = std::string(graphName) +
            " graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}
template <class I, class S>
static bool executeSlot(I *i, RuntimeMetrics &metrics, Runtime &runtime, S &s,
                        const std::vector<float> &a,
                        const std::vector<float> &b, std::vector<float> &o,
                        std::string &error) {
  if (!s.graph || a.size() != s.n0 || b.size() != s.n1) {
    error = "slot buffer size mismatch";
    return false;
  }
  o.resize(s.no);
  s.in[0].v1.clientBuf = {const_cast<float *>(a.data()),
                          uint32_t(a.size() * sizeof(float))};
  s.in[1].v1.clientBuf = {const_cast<float *>(b.data()),
                          uint32_t(b.size() * sizeof(float))};
  s.out.v1.clientBuf = {o.data(), uint32_t(o.size() * sizeof(float))};
  auto st = i->api.graphExecute(s.graph, s.in, 2, &s.out, 1, nullptr, nullptr);
  ++metrics.graphExecuteCount;
  runtime.recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st), st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    error = "graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}

bool Runtime::prepareInputGradientMatMul(uint32_t b, uint32_t in, uint32_t out,
                                         std::string &e) {
  bool ok = prepareSlot(impl_, metrics_, impl_->dx, "phonelm_dx", b, out, in,
                        out, b, in, false, true, e);
  if (ok)
    diagnostics_ += "dx_graph_create=success\ndx_graph_finalize=success\ndx_"
                    "transpose_implementation=QNN_MATMUL_TRANSPOSE_IN1\n";
  return ok;
}
bool Runtime::executeInputGradient(const std::vector<float> &dp,
                                   const std::vector<float> &w,
                                   std::vector<float> &dx, std::string &e) {
  return executeSlot(impl_, metrics_, *this, impl_->dx, dp, w, dx, e);
}

bool Runtime::prepareMlp(uint32_t b, uint32_t in, uint32_t h, uint32_t out,
                         std::string &e, bool prepareSplitBackward) {
  if (!impl_ || !impl_->context) {
    e = "runtime not initialized";
    return false;
  }
  auto *i = impl_;
  i->batch = b;
  i->inDim = in;
  i->hidDim = h;
  i->outDim = out;
  i->xDims[0] = b;
  i->xDims[1] = in;
  i->w1Dims[0] = in;
  i->w1Dims[1] = h;
  i->w2Dims[0] = h;
  i->w2Dims[1] = out;
  i->zDims[0] = b;
  i->zDims[1] = h;
  i->pDims[0] = b;
  i->pDims[1] = out;
  tensor(i->mlpInputs[0], "mlp_x", QNN_TENSOR_TYPE_APP_WRITE, i->xDims);
  tensor(i->mlpInputs[1], "mlp_w1", QNN_TENSOR_TYPE_APP_WRITE, i->w1Dims);
  tensor(i->mlpInputs[2], "mlp_w2", QNN_TENSOR_TYPE_APP_WRITE, i->w2Dims);
  tensor(i->mlpZ, "mlp_z1", QNN_TENSOR_TYPE_NATIVE, i->zDims);
  tensor(i->mlpH, "mlp_h", QNN_TENSOR_TYPE_APP_READ, i->zDims);
  tensor(i->mlpP, "mlp_p", QNN_TENSOR_TYPE_APP_READ, i->pDims);
  auto st = i->api.graphCreate(i->context, "phonelm_mlp_forward", nullptr,
                               &i->mlpForward);
  ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) {
    e = "mlp forward graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  for (auto &t : i->mlpInputs) {
    st = i->api.tensorCreateGraphTensor(i->mlpForward, &t);
    if (st != QNN_SUCCESS) {
      e = "mlp input tensorCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  for (auto *t : {&i->mlpZ, &i->mlpH, &i->mlpP}) {
    st = i->api.tensorCreateGraphTensor(i->mlpForward, t);
    if (st != QNN_SUCCESS) {
      e = "mlp output tensorCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  auto add = [&](const char *n, const char *type, Qnn_Tensor_t *ins,
                 uint32_t ni, Qnn_Tensor_t *outs, uint32_t no) {
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = n;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = type;
    op.v1.numOfInputs = ni;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = no;
    op.v1.outputTensors = outs;
    return i->api.graphAddNode(i->mlpForward, op);
  };
  Qnn_Tensor_t mm1in[2] = {i->mlpInputs[0], i->mlpInputs[1]};
  if ((st = add("mlp_mm1", QNN_OP_MAT_MUL, mm1in, 2, &i->mlpZ, 1)) !=
      QNN_SUCCESS) {
    e = "mlp MatMul1=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  if ((st = add("mlp_relu", QNN_OP_RELU, &i->mlpZ, 1, &i->mlpH, 1)) !=
      QNN_SUCCESS) {
    e = "mlp Relu=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Tensor_t mm2in[2] = {i->mlpH, i->mlpInputs[2]};
  if ((st = add("mlp_mm2", QNN_OP_MAT_MUL, mm2in, 2, &i->mlpP, 1)) !=
      QNN_SUCCESS) {
    e = "mlp MatMul2=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  st = i->api.graphFinalize(i->mlpForward, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    e = "mlp forward graphFinalize=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  if (prepareSplitBackward) {
    if (!prepareSlot(i, metrics_, i->dw2, "phonelm_mlp_dw2", b, h, b, out, h,
                     out, true, false, e))
      return false;
    if (!prepareSlot(i, metrics_, i->dh, "phonelm_mlp_dh", b, out, h, out, b,
                     h, false, true, e))
      return false;
    if (!prepareSlot(i, metrics_, i->dw1, "phonelm_mlp_dw1", b, in, b, h, in,
                     h, true, false, e))
      return false;
  }
  diagnostics_ += "mlp_forward_graph_create=success\nmlp_forward_graph_"
                  "finalize=success\nmlp_relu_backend=HTP\nmlp_backward_graphs=";
  diagnostics_ += prepareSplitBackward ? "dw2,dh,dw1\n" : "none\n";
  diagnostics_ += "backend_device_context_shared=true\n";
  return true;
}
bool Runtime::setMlpWeights(const std::vector<float> &w1,
                            const std::vector<float> &w2, std::string &e) {
  auto *i = impl_;
  if (!i || !i->mlpForward || w1.size() != i->inDim * i->hidDim ||
      w2.size() != i->hidDim * i->outDim) {
    e = "MLP weight size mismatch";
    return false;
  }
  i->mlpW1 = w1;
  i->mlpW2 = w2;
  ++metrics_.runtimeWeightUpdateCount;
  return true;
}
bool Runtime::executeMlpForward(const std::vector<float> &x,
                                std::vector<float> &h, std::vector<float> &p,
                                std::string &e) {
  auto *i = impl_;
  if (!i || !i->mlpForward || x.size() != i->batch * i->inDim) {
    e = "MLP forward input mismatch";
    return false;
  }
  h.resize(i->batch * i->hidDim);
  p.resize(i->batch * i->outDim);
  i->mlpInputs[0].v1.clientBuf = {const_cast<float *>(x.data()),
                                  uint32_t(x.size() * 4)};
  i->mlpInputs[1].v1.clientBuf = {i->mlpW1.data(),
                                  uint32_t(i->mlpW1.size() * 4)};
  i->mlpInputs[2].v1.clientBuf = {i->mlpW2.data(),
                                  uint32_t(i->mlpW2.size() * 4)};
  Qnn_Tensor_t outs[2] = {i->mlpH, i->mlpP};
  outs[0].v1.clientBuf = {h.data(), uint32_t(h.size() * 4)};
  outs[1].v1.clientBuf = {p.data(), uint32_t(p.size() * 4)};
  auto st = i->api.graphExecute(i->mlpForward, i->mlpInputs, 3, outs, 2,
                                nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st), st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    e = "mlp forward graphExecute=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}
bool Runtime::executeMlpSecondBackward(const std::vector<float> &h,
                                       const std::vector<float> &dp,
                                       std::vector<float> &dw2,
                                       std::vector<float> &dh, std::string &e) {
  return executeSlot(impl_, metrics_, *this, impl_->dw2, h, dp, dw2, e) &&
         executeSlot(impl_, metrics_, *this, impl_->dh, dp, impl_->mlpW2, dh, e);
}
bool Runtime::executeMlpFirstBackward(const std::vector<float> &x,
                                      const std::vector<float> &dz,
                                      std::vector<float> &dw1, std::string &e) {
  return executeSlot(impl_, metrics_, *this, impl_->dw1, x, dz, dw1, e);
}
bool Runtime::prepareReluBackward(uint32_t b, uint32_t h, std::string &e) {
  auto *i = impl_;
  if (!i || !i->context) {
    e = "runtime not initialized";
    return false;
  }
  auto &r = i->reluBackward;
  r.dims[0] = b;
  r.dims[1] = h;
  r.elements = b * h;
  r.zero.assign(r.elements, 0.0f);
  tensor(r.inputs[0], "relu_bw_activation", QNN_TENSOR_TYPE_APP_WRITE, r.dims);
  tensor(r.inputs[1], "relu_bw_dh", QNN_TENSOR_TYPE_APP_WRITE, r.dims);
  tensor(r.inputs[2], "relu_bw_zero", QNN_TENSOR_TYPE_APP_WRITE, r.dims);
  tensor(r.mask, "relu_bw_mask", QNN_TENSOR_TYPE_APP_READ, r.dims,
         QNN_DATATYPE_BOOL_8);
  tensor(r.dz, "relu_bw_dz", QNN_TENSOR_TYPE_APP_READ, r.dims);
  auto st = i->api.graphCreate(i->context, "phonelm_relu_backward", nullptr,
                               &r.graph);
  ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) {
    e = "relu backward graphCreate=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  for (auto &t : r.inputs) {
    if ((st = i->api.tensorCreateGraphTensor(r.graph, &t)) != QNN_SUCCESS) {
      e = "relu backward input tensor=" +
          std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  for (auto *t : {&r.mask, &r.dz}) {
    if ((st = i->api.tensorCreateGraphTensor(r.graph, t)) != QNN_SUCCESS) {
      e = "relu backward output tensor=" +
          std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  Qnn_Tensor_t greaterInputs[2] = {r.inputs[0], r.inputs[2]};
  Qnn_OpConfig_t greater = QNN_OPCONFIG_INIT;
  greater.v1.name = "relu_bw_greater";
  greater.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  greater.v1.typeName = QNN_OP_ELEMENT_WISE_GREATER;
  greater.v1.numOfInputs = 2;
  greater.v1.inputTensors = greaterInputs;
  greater.v1.numOfOutputs = 1;
  greater.v1.outputTensors = &r.mask;
  if ((st = i->api.graphAddNode(r.graph, greater)) != QNN_SUCCESS) {
    e = "relu backward Greater graphAddNode=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Tensor_t selectInputs[3] = {r.mask, r.inputs[1], r.inputs[2]};
  Qnn_OpConfig_t select = QNN_OPCONFIG_INIT;
  select.v1.name = "relu_bw_select";
  select.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  select.v1.typeName = QNN_OP_ELEMENT_WISE_SELECT;
  select.v1.numOfInputs = 3;
  select.v1.inputTensors = selectInputs;
  select.v1.numOfOutputs = 1;
  select.v1.outputTensors = &r.dz;
  if ((st = i->api.graphAddNode(r.graph, select)) != QNN_SUCCESS) {
    e = "relu backward Select graphAddNode=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  st = i->api.graphFinalize(r.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    e = "relu backward graphFinalize=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  diagnostics_ +=
      "relu_backward_graph_create=success\nrelu_backward_graph_finalize=success\n"
      "relu_backward_op_method=ElementWiseGreater+ElementWiseSelect\n"
      "relu_backward_mask_datatype=BOOL_8\nrelu_backward_zero_rule=GT_ZERO\n";
  return true;
}

bool Runtime::executeReluBackward(const std::vector<float> &activation,
                                  const std::vector<float> &dh,
                                  std::vector<std::uint8_t> &mask,
                                  std::vector<float> &dz, std::string &e) {
  auto *i = impl_;
  if (!i || !i->reluBackward.graph ||
      activation.size() != i->reluBackward.elements ||
      dh.size() != i->reluBackward.elements) {
    e = "relu backward buffer size mismatch";
    return false;
  }
  auto &r = i->reluBackward;
  mask.resize(r.elements);
  dz.resize(r.elements);
  r.inputs[0].v1.clientBuf = {const_cast<float *>(activation.data()),
                              uint32_t(activation.size() * sizeof(float))};
  r.inputs[1].v1.clientBuf = {const_cast<float *>(dh.data()),
                              uint32_t(dh.size() * sizeof(float))};
  r.inputs[2].v1.clientBuf = {r.zero.data(),
                              uint32_t(r.zero.size() * sizeof(float))};
  Qnn_Tensor_t outputs[2] = {r.mask, r.dz};
  outputs[0].v1.clientBuf = {mask.data(), uint32_t(mask.size())};
  outputs[1].v1.clientBuf = {dz.data(),
                             uint32_t(dz.size() * sizeof(float))};
  const auto st =
      i->api.graphExecute(r.graph, r.inputs, 3, outputs, 2, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st), st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    e = "relu backward graphExecute=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}

bool Runtime::prepareMlpFusedBackward(bool diagnosticOutputs, std::string &e) {
  auto *i = impl_;
  if (!i || !i->mlpForward) {
    e = "MLP forward must be prepared before fused backward";
    return false;
  }
  auto &f = i->fusedBackward;
  f.diagnosticOutputs = diagnosticOutputs;
  f.dpDims[0] = i->batch;
  f.dpDims[1] = i->outDim;
  f.dw2Dims[0] = i->hidDim;
  f.dw2Dims[1] = i->outDim;
  f.dw1Dims[0] = i->inDim;
  f.dw1Dims[1] = i->hidDim;
  f.zero.assign(i->batch * i->hidDim, 0.0f);
  tensor(f.inputs[0], "fused_x", QNN_TENSOR_TYPE_APP_WRITE, i->xDims);
  tensor(f.inputs[1], "fused_h", QNN_TENSOR_TYPE_APP_WRITE, i->zDims);
  tensor(f.inputs[2], "fused_dp", QNN_TENSOR_TYPE_APP_WRITE, f.dpDims);
  tensor(f.inputs[3], "fused_w2", QNN_TENSOR_TYPE_APP_WRITE, i->w2Dims);
  tensor(f.inputs[4], "fused_zero", QNN_TENSOR_TYPE_APP_WRITE, i->zDims);
  tensor(f.dw2, "fused_dw2", QNN_TENSOR_TYPE_APP_READ, f.dw2Dims);
  const auto intermediateType = diagnosticOutputs ? QNN_TENSOR_TYPE_APP_READ
                                                   : QNN_TENSOR_TYPE_NATIVE;
  tensor(f.dh, "fused_dh", intermediateType, i->zDims);
  tensor(f.mask, "fused_mask", intermediateType, i->zDims,
         QNN_DATATYPE_BOOL_8);
  tensor(f.dz, "fused_dz", intermediateType, i->zDims);
  tensor(f.dw1, "fused_dw1", QNN_TENSOR_TYPE_APP_READ, f.dw1Dims);
  auto st = i->api.graphCreate(i->context, "phonelm_mlp_fused_backward", nullptr,
                               &f.graph);
  ++metrics_.graphCreateCount;
  if (st != QNN_SUCCESS) {
    e = "fused backward graphCreate=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  for (auto &t : f.inputs) {
    if ((st = i->api.tensorCreateGraphTensor(f.graph, &t)) != QNN_SUCCESS) {
      e = "fused backward input tensor=" +
          std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  for (auto *t : {&f.dw2, &f.dh, &f.mask, &f.dz, &f.dw1}) {
    if ((st = i->api.tensorCreateGraphTensor(f.graph, t)) != QNN_SUCCESS) {
      e = "fused backward tensor=" + std::to_string(QNN_GET_ERROR_CODE(st));
      return false;
    }
  }
  auto addMatMul = [&](const char *name, Qnn_Tensor_t *inputs,
                       Qnn_Tensor_t *output, const char *transposeName) {
    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType = QNN_DATATYPE_BOOL_8;
    scalar.bool8Value = true;
    Qnn_Param_t param = QNN_PARAM_INIT;
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = transposeName;
    param.scalarParam = scalar;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = name;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 1;
    op.v1.params = &param;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = output;
    return i->api.graphAddNode(f.graph, op);
  };
  Qnn_Tensor_t dw2Inputs[2] = {f.inputs[1], f.inputs[2]};
  if ((st = addMatMul("fused_dw2_mm", dw2Inputs, &f.dw2,
                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0)) != QNN_SUCCESS) {
    e = "fused dW2 graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Tensor_t dhInputs[2] = {f.inputs[2], f.inputs[3]};
  if ((st = addMatMul("fused_dh_mm", dhInputs, &f.dh,
                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1)) != QNN_SUCCESS) {
    e = "fused dH graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  auto addSimple = [&](const char *name, const char *type, Qnn_Tensor_t *inputs,
                       uint32_t inputCount, Qnn_Tensor_t *output) {
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = name;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = type;
    op.v1.numOfInputs = inputCount;
    op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = output;
    return i->api.graphAddNode(f.graph, op);
  };
  Qnn_Tensor_t greaterInputs[2] = {f.inputs[1], f.inputs[4]};
  if ((st = addSimple("fused_relu_greater", QNN_OP_ELEMENT_WISE_GREATER,
                      greaterInputs, 2, &f.mask)) != QNN_SUCCESS) {
    e = "fused Greater graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Tensor_t selectInputs[3] = {f.mask, f.dh, f.inputs[4]};
  if ((st = addSimple("fused_relu_select", QNN_OP_ELEMENT_WISE_SELECT,
                      selectInputs, 3, &f.dz)) != QNN_SUCCESS) {
    e = "fused Select graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  Qnn_Tensor_t dw1Inputs[2] = {f.inputs[0], f.dz};
  if ((st = addMatMul("fused_dw1_mm", dw1Inputs, &f.dw1,
                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0)) != QNN_SUCCESS) {
    e = "fused dW1 graphAddNode=" + std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  st = i->api.graphFinalize(f.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (st != QNN_SUCCESS) {
    e = "fused backward graphFinalize=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  diagnostics_ +=
      "fused_backward_graph_create=success\nfused_backward_graph_finalize=success\n"
      "fused_backward_nodes=dW2,dH,Greater,Select,dW1\n"
      "fused_backward_execute_count_per_step=1\nhtp_relu_backward_used=true\n"
      "htp_fused_backward_used=true\n";
  return true;
}

bool Runtime::executeMlpFusedBackward(
    const std::vector<float> &x, const std::vector<float> &h,
    const std::vector<float> &dp, std::vector<float> &dw2,
    std::vector<float> &dh, std::vector<std::uint8_t> &mask,
    std::vector<float> &dz, std::vector<float> &dw1, std::string &e) {
  auto *i = impl_;
  if (!i || !i->fusedBackward.graph || x.size() != i->batch * i->inDim ||
      h.size() != i->batch * i->hidDim ||
      dp.size() != i->batch * i->outDim) {
    e = "fused backward buffer size mismatch";
    return false;
  }
  auto &f = i->fusedBackward;
  dw2.resize(i->hidDim * i->outDim);
  dw1.resize(i->inDim * i->hidDim);
  f.inputs[0].v1.clientBuf = {const_cast<float *>(x.data()),
                              uint32_t(x.size() * sizeof(float))};
  f.inputs[1].v1.clientBuf = {const_cast<float *>(h.data()),
                              uint32_t(h.size() * sizeof(float))};
  f.inputs[2].v1.clientBuf = {const_cast<float *>(dp.data()),
                              uint32_t(dp.size() * sizeof(float))};
  f.inputs[3].v1.clientBuf = {i->mlpW2.data(),
                              uint32_t(i->mlpW2.size() * sizeof(float))};
  f.inputs[4].v1.clientBuf = {f.zero.data(),
                              uint32_t(f.zero.size() * sizeof(float))};
  Qnn_Tensor_t outputs[5] = {f.dw2, f.dh, f.mask, f.dz, f.dw1};
  outputs[0].v1.clientBuf = {dw2.data(),
                             uint32_t(dw2.size() * sizeof(float))};
  outputs[4].v1.clientBuf = {dw1.data(),
                             uint32_t(dw1.size() * sizeof(float))};
  uint32_t outputCount = 2;
  Qnn_Tensor_t benchmarkOutputs[2] = {outputs[0], outputs[4]};
  Qnn_Tensor_t *executeOutputs = benchmarkOutputs;
  if (f.diagnosticOutputs) {
    dh.resize(i->batch * i->hidDim);
    mask.resize(i->batch * i->hidDim);
    dz.resize(i->batch * i->hidDim);
    outputs[1].v1.clientBuf = {dh.data(),
                               uint32_t(dh.size() * sizeof(float))};
    outputs[2].v1.clientBuf = {mask.data(), uint32_t(mask.size())};
    outputs[3].v1.clientBuf = {dz.data(),
                               uint32_t(dz.size() * sizeof(float))};
    outputCount = 5;
    executeOutputs = outputs;
  } else {
    dh.clear();
    mask.clear();
    dz.clear();
  }
  const auto st = i->api.graphExecute(f.graph, f.inputs, 5, executeOutputs,
                                      outputCount, nullptr, nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(st), QNN_GET_ERROR_CODE(st), st == QNN_SUCCESS);
  if (st != QNN_SUCCESS) {
    e = "fused backward graphExecute=" +
        std::to_string(QNN_GET_ERROR_CODE(st));
    return false;
  }
  return true;
}