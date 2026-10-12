// Diagnostic graph: use the production shared forward/backward builder, with
// application-visible intermediates. No optimizer, dataset, or CPU fallback.
bool Runtime::prepareLayerNormStability(uint32_t rows, uint32_t dimension,
                                      float epsilon, bool adaptive, std::string& error, bool observe) {
  if (!impl_ || !impl_->context || !rows || rows > 64 || !dimension ||
      dimension > 256 || !std::isfinite(epsilon) || epsilon <= 0 ||
      impl_->layerNormStability.graph) { error = "LN_STABILITY_INVALID_CONFIGURATION"; return false; }
  auto& g = impl_->layerNormStability;
  g.rows = rows; g.dimension = dimension; g.epsilon = epsilon;
  g.halfEpsilon = epsilon * .25f; g.d = float(dimension); g.invD = 1.f / dimension;
  if (!(g.halfEpsilon > 0)) { error = "LN_STABILITY_EPSILON_UNREPRESENTABLE"; return false; }
  const auto configs = buildHtpGraphConfigs(options_);
  if (configs.failClosedInvalid) { error = "LN_STABILITY_PRECISION_CONFIGURATION"; return false; }
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_ln_stability", configs.array, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) { error = "LN_STABILITY_CREATE=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  g.tensors.reserve(64); g.names.reserve(64); g.dims.reserve(64);
  auto add = [&](const char* name, Qnn_TensorType_t type, std::vector<uint32_t> dims,
                 Qnn_DataType_t dtype = QNN_DATATYPE_FLOAT_32, void* data = nullptr, uint32_t bytes = 0) {
    const int id = int(g.tensors.size());
    g.names.emplace_back(name); g.dims.push_back(std::move(dims));
    g.tensors.push_back(QNN_TENSOR_INIT);
    tensor(g.tensors.back(), g.names.back().c_str(), type, g.dims.back().data(), dtype, uint32_t(g.dims.back().size()));
    if (data) g.tensors.back().v1.clientBuf = {data, bytes};
    if (type == QNN_TENSOR_TYPE_APP_READ) g.outputs.push_back(uint32_t(id));
    return id;
  };
  const std::vector<uint32_t> full{rows, dimension}, row{rows, 1}, scalar{1, 1}, param{dimension};
  const auto W = QNN_TENSOR_TYPE_APP_WRITE, R = QNN_TENSOR_TYPE_APP_READ, N = QNN_TENSOR_TYPE_NATIVE, S = QNN_TENSOR_TYPE_STATIC;
  const auto I = observe ? R : N;
  add("x", W, full); add("dy", W, full); add("gamma", W, param); add("beta", W, param);
  // Order is part of the diagnostic API; intermediate taps follow final outputs.
  const int y = add("y", R, full), xhat = add("xhat", R, full), dx = add("dx", R, full);
  const int dg = add("dg", R, param), db = add("db", R, param);
  const int mean = add("mean", I, row), c = add("centered", I, full), cs = add("centered_s", I, full);
  const int square = add("square", I, full), var = add("var_s", I, row), ve = add("var_eps_s", I, row);
  const int is = add("inverse_s", I, row), inv = add("inverse", I, row), xg = add("xhat_gamma", N, full);
  const int axis = add("last_axis", S, {1}, QNN_DATATYPE_UINT_32, &g.lastAxis, 4);
  const int ra = add("row_axis", S, {1}, QNN_DATATYPE_UINT_32, &g.rowAxis, 4);
  auto stat = [&](const char* n, float& value) { return add(n, S, scalar, QNN_DATATYPE_FLOAT_32, &value, 4); };
  const int sc = stat("scale", g.scale), ep = stat("epsilon", g.epsilon), d = stat("dimension", g.d), id = stat("inv_d", g.invD);
  const int th = stat("threshold", g.threshold), half = stat("half", g.half), he = stat("half_epsilon", g.halfEpsilon);
  std::array<int, 5> aids{{-1,-1,-1,-1,-1}};
  if (adaptive) {
    aids = {{add("abs_c", N, full), add("max_c", I, row),
             add("condition", N, row, QNN_DATATYPE_BOOL_8),
             add("selected_scale", I, row), add("selected_epsilon", I, row)}};
  }
  const int dh = add("dxhat", I, full), sum = add("sum_dxhat", I, row), prod = add("dxhat_xhat", I, full);
  const int dyx = add("dy_xhat", I, full), sump = add("sum_prod", I, row), ddh = add("d_dxhat", I, full);
  const int first = add("first", I, full), xs = add("xhat_sum", I, full), bracket = add("bracket", I, full);
  const int ivd = add("inv_over_d", I, row);
  for (auto& t : g.tensors) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, &t);
    if (status != QNN_SUCCESS) { error = "LN_STABILITY_TENSOR=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  }
  std::unordered_set<std::string> names;
  TransformerTrainingLayerNormBuilder b{impl_->api, g.graph, g.tensors.data(), axis, ra, sc, ep, d, id, error, g.nodes, &names};
  b.adaptiveIds = aids; b.adaptiveThreshold = th; b.adaptiveHalf = half; b.adaptiveHalfEpsilon = he;
  if (!b.forward("ln", 0, 2, 3, mean, c, cs, square, var, ve, is, inv, xhat, xg, y) ||
      !b.backward("lnb", 1, 2, xhat, inv, dh, sum, prod, dyx, sump, ddh, first, xs, bracket, ivd, dx, dg, db)) return false;
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) { error = "LN_STABILITY_FINALIZE=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  diagnostics_ += "ln_stability_nodes=" + std::to_string(g.nodes) + "\nln_stability_tensors=" + std::to_string(g.tensors.size()) + "\n";
  return true;
}

bool Runtime::executeLayerNormStability(const std::vector<float>& x, const std::vector<float>& dy,
    const std::vector<float>& gamma, const std::vector<float>& beta,
    LayerNormStabilityOutputs& out, std::string& error) {
  auto& g = impl_->layerNormStability;
  if (!g.graph || x.size() != size_t(g.rows)*g.dimension || dy.size() != x.size() ||
      gamma.size() != g.dimension || beta.size() != g.dimension) { error = "LN_STABILITY_SHAPE"; return false; }
  const std::vector<float>* data[4]{&x,&dy,&gamma,&beta};
  Qnn_Tensor_t inputs[4];
  for (int i=0;i<4;++i) { inputs[i]=g.tensors[i]; inputs[i].v1.clientBuf={const_cast<float*>(data[i]->data()),uint32_t(data[i]->size()*4)}; }
  out.taps.resize(g.outputs.size()-5);
  std::vector<std::vector<float>*> buffers{&out.values.output,&out.values.normalized,&out.values.dInput,&out.values.dGamma,&out.values.dBeta};
  for (auto& tap:out.taps) buffers.push_back(&tap);
  std::vector<Qnn_Tensor_t> outputs;
  for (size_t i=0;i<g.outputs.size();++i) {
    auto t=g.tensors[g.outputs[i]]; size_t count=1;
    for (uint32_t j=0;j<t.v1.rank;++j) count*=t.v1.dimensions[j];
    buffers[i]->assign(count, std::numeric_limits<float>::quiet_NaN());
    t.v1.clientBuf={buffers[i]->data(),uint32_t(count*4)}; outputs.push_back(t);
  }
  const auto status=impl_->api.graphExecute(g.graph,inputs,4,outputs.data(),uint32_t(outputs.size()),nullptr,nullptr);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),QNN_GET_ERROR_CODE(status),status==QNN_SUCCESS);
  if(status!=QNN_SUCCESS){error="LN_STABILITY_EXECUTE="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}
  return true;
}
