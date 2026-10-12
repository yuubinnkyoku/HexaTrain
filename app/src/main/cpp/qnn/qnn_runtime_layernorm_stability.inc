// Diagnostic graph: use the production shared forward/backward builder, with
// application-visible intermediates. No optimizer, dataset, or CPU fallback.
bool Runtime::prepareLayerNormStability(uint32_t rows, uint32_t dimension,
                                      float epsilon, bool adaptive, std::string& error, bool observe, uint32_t variant) {
  if (!impl_ || !impl_->context || !rows || rows > 64 || !dimension ||
      dimension > 256 || !std::isfinite(epsilon) || epsilon <= 0 ||
      impl_->layerNormStability.graph || variant > 3) { error = "LN_STABILITY_INVALID_CONFIGURATION"; return false; }
  auto& g = impl_->layerNormStability;
  g.rows = rows; g.dimension = dimension; g.epsilon = epsilon;
  g.halfEpsilon = epsilon * .25f; g.d = float(dimension); g.invD = 1.f / dimension;
  // Diagnostic candidates only. The production training builder is unchanged.
  if (variant == 3) { g.scale = 8.f; g.epsilon = epsilon * 64.f; g.halfEpsilon = epsilon * .25f; }
  if (!(g.halfEpsilon > 0)) { error = "LN_STABILITY_EPSILON_UNREPRESENTABLE"; return false; }
  const auto configs = buildHtpGraphConfigs(options_);
  if (configs.failClosedInvalid) { error = "LN_STABILITY_PRECISION_CONFIGURATION"; return false; }
  if (!configs.description.empty()) diagnostics_ += configs.description + "\n";
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_ln_stability", configs.array, &g.graph);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) { error = "LN_STABILITY_CREATE=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  if(configs.array && impl_->api.graphSetConfig){status=impl_->api.graphSetConfig(g.graph,configs.array);
    if(status!=QNN_SUCCESS){error="LN_STABILITY_SET_CONFIG="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}
    diagnostics_+="ln_precision_config_set=SUCCESS\n";}
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
  const int dyx = add("dy_xhat", I, full), sump = add("sum_prod", I, row), ddh = add("d_dxhat", variant==2?N:I, full);
  const int first = add("first", I, full), xs = add("xhat_sum", I, full), bracket = add("bracket", I, full);
  const int ivd = add("inv_over_d", variant==2?N:I, row);
  int target=-1, denominator=-1, scaleSquare=-1;
  if (variant == 1) {
    target=stat("range_target",g.rangeTarget);
    denominator=add("range_denominator",N,row);
    scaleSquare=add("scale_square",N,row);
  }
  for (auto& t : g.tensors) {
    status = impl_->api.tensorCreateGraphTensor(g.graph, &t);
    if (status != QNN_SUCCESS) { error = "LN_STABILITY_TENSOR=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  }
  std::unordered_set<std::string> names;
  TransformerTrainingLayerNormBuilder b{impl_->api, g.graph, g.tensors.data(), axis, ra, sc, ep, d, id, error, g.nodes, &names};
  b.adaptiveIds = aids; b.adaptiveThreshold = th; b.adaptiveHalf = half; b.adaptiveHalfEpsilon = he;
  if (variant != 1) {
    if (!b.forward("ln", 0, 2, 3, mean, c, cs, square, var, ve, is, inv, xhat, xg, y)) return false;
  } else {
    // s=128/max(128,max(abs(c))); epsilon_s=epsilon*s*s.
    // Square is downstream of scaling, including unselected/constant rows.
    if (!adaptive || !b.reduce("ln_mean",QNN_OP_REDUCE_MEAN,QNN_OP_REDUCE_MEAN_PARAM_AXES,QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,0,axis,true,mean,"mean") ||
        !b.binary("ln_center",QNN_OP_ELEMENT_WISE_SUBTRACT,0,mean,c,"center") ||
        !b.unary("ln_abs",QNN_OP_ELEMENT_WISE_ABS,c,aids[0],"abs") ||
        !b.reduce("ln_max",QNN_OP_REDUCE_MAX,QNN_OP_REDUCE_MAX_PARAM_AXES,QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS,aids[0],axis,true,aids[1],"max") ||
        !b.binary("ln_den",QNN_OP_ELEMENT_WISE_MAXIMUM,aids[1],target,denominator,"denominator") ||
        !b.binary("ln_scale",QNN_OP_ELEMENT_WISE_DIVIDE,target,denominator,aids[3],"scale") ||
        !b.binary("ln_s2",QNN_OP_ELEMENT_WISE_MULTIPLY,aids[3],aids[3],scaleSquare,"scale square") ||
        !b.binary("ln_epsilon",QNN_OP_ELEMENT_WISE_MULTIPLY,ep,scaleSquare,aids[4],"epsilon scale") ||
        !b.binary("ln_cs",QNN_OP_ELEMENT_WISE_MULTIPLY,c,aids[3],cs,"center scale") ||
        !b.binary("ln_square",QNN_OP_ELEMENT_WISE_MULTIPLY,cs,cs,square,"square") ||
        !b.reduce("ln_var",QNN_OP_REDUCE_MEAN,QNN_OP_REDUCE_MEAN_PARAM_AXES,QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,square,axis,true,var,"variance") ||
        !b.binary("ln_ve",QNN_OP_ELEMENT_WISE_ADD,var,aids[4],ve,"epsilon") ||
        !b.unary("ln_is",QNN_OP_ELEMENT_WISE_RSQRT,ve,is,"rsqrt") ||
        !b.binary("ln_inv",QNN_OP_ELEMENT_WISE_MULTIPLY,is,aids[3],inv,"inverse") ||
        !b.binary("ln_xhat",QNN_OP_ELEMENT_WISE_MULTIPLY,c,inv,xhat,"xhat") ||
        !b.binary("ln_gamma",QNN_OP_ELEMENT_WISE_MULTIPLY,xhat,2,xg,"gamma") ||
        !b.binary("ln_beta",QNN_OP_ELEMENT_WISE_ADD,xg,3,y,"beta")) return false;
  }
  if (variant != 2) {
    if (!b.backward("lnb", 1, 2, xhat, inv, dh, sum, prod, dyx, sump, ddh, first, xs, bracket, ivd, dx, dg, db)) return false;
  } else {
    // Mean form avoids multiplying the bracket by D before subtraction.
    if (!b.binary("b_dh",QNN_OP_ELEMENT_WISE_MULTIPLY,1,2,dh,"dxhat") ||
        !b.reduce("b_mean",QNN_OP_REDUCE_MEAN,QNN_OP_REDUCE_MEAN_PARAM_AXES,QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,dh,axis,true,sum,"mean") ||
        !b.binary("b_prod",QNN_OP_ELEMENT_WISE_MULTIPLY,dh,xhat,prod,"product") ||
        !b.reduce("b_meanprod",QNN_OP_REDUCE_MEAN,QNN_OP_REDUCE_MEAN_PARAM_AXES,QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,prod,axis,true,sump,"mean product") ||
        !b.binary("b_first",QNN_OP_ELEMENT_WISE_SUBTRACT,dh,sum,first,"first") ||
        !b.binary("b_projection",QNN_OP_ELEMENT_WISE_MULTIPLY,xhat,sump,xs,"projection") ||
        !b.binary("b_bracket",QNN_OP_ELEMENT_WISE_SUBTRACT,first,xs,bracket,"bracket") ||
        !b.binary("b_dx",QNN_OP_ELEMENT_WISE_MULTIPLY,bracket,inv,dx,"dx") ||
        !b.binary("b_dyx",QNN_OP_ELEMENT_WISE_MULTIPLY,1,xhat,dyx,"dyx") ||
        !b.reduce("b_dg",QNN_OP_REDUCE_SUM,QNN_OP_REDUCE_SUM_PARAM_AXES,QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,dyx,ra,false,dg,"dg") ||
        !b.reduce("b_db",QNN_OP_REDUCE_SUM,QNN_OP_REDUCE_SUM_PARAM_AXES,QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,1,ra,false,db,"db")) return false;
    // Unused diagnostic taps must be NATIVE (no producer).
  }
  status = impl_->api.graphFinalize(g.graph, nullptr, nullptr);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) { error = "LN_STABILITY_FINALIZE=" + std::to_string(QNN_GET_ERROR_CODE(status)); return false; }
  diagnostics_ += "ln_stability_nodes=" + std::to_string(g.nodes) + "\nln_stability_tensors=" + std::to_string(g.tensors.size()) + "\n";
  return true;
}

// Independent primitive probe; no LayerNorm composition or CPU execution path.
bool Runtime::prepareLayerNormArithmetic(uint32_t rows,uint32_t dimension,std::string& error) {
  if(!impl_||!impl_->context||!rows||rows>64||!dimension||dimension>256||impl_->layerNormStability.graph){error="LN_ARITH_CONFIG";return false;}
  auto& g=impl_->layerNormStability;g.rows=rows;g.dimension=dimension;
  auto configs=buildHtpGraphConfigs(options_);if(configs.failClosedInvalid){error="LN_ARITH_PRECISION";return false;}
  if(!configs.description.empty())diagnostics_+=configs.description+"\n";
  auto status=impl_->api.graphCreate(impl_->context,"ln_arithmetic",configs.array,&g.graph);++metrics_.graphCreateCount;
  if(status!=QNN_SUCCESS){error="LN_ARITH_CREATE="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}
  if(configs.array && impl_->api.graphSetConfig){status=impl_->api.graphSetConfig(g.graph,configs.array);
    if(status!=QNN_SUCCESS){error="LN_ARITH_SET_CONFIG="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}
    diagnostics_+="ln_precision_config_set=SUCCESS\n";}
  g.tensors.reserve(32);g.names.reserve(32);g.dims.reserve(32);
  auto add=[&](const char* n,Qnn_TensorType_t type,std::vector<uint32_t> dims,Qnn_DataType_t dtype=QNN_DATATYPE_FLOAT_32,void* data=nullptr){
    int i=int(g.tensors.size());g.names.emplace_back(n);g.dims.push_back(std::move(dims));g.tensors.push_back(QNN_TENSOR_INIT);
    tensor(g.tensors.back(),g.names.back().c_str(),type,g.dims.back().data(),dtype,uint32_t(g.dims.back().size()));
    if(data)g.tensors.back().v1.clientBuf={data,4};if(type==QNN_TENSOR_TYPE_APP_READ)g.outputs.push_back(i);return i;};
  auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ;
  add("a",W,{rows,dimension});add("b",W,{rows,dimension});
  for(auto n:{"add","subtract","multiply","divide","sqrt","rsqrt","reciprocal"})add(n,R,{rows,dimension});
  add("mean",R,{rows,1});add("sum",R,{rows,1});
  int axis=add("axis",QNN_TENSOR_TYPE_STATIC,{1},QNN_DATATYPE_UINT_32,&g.lastAxis);
  int one=add("one",QNN_TENSOR_TYPE_STATIC,{1,1},QNN_DATATYPE_FLOAT_32,&g.scale);
  for(auto& t:g.tensors){status=impl_->api.tensorCreateGraphTensor(g.graph,&t);if(status!=QNN_SUCCESS){error="LN_ARITH_TENSOR="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}}
  std::unordered_set<std::string> names;
  TransformerTrainingLayerNormBuilder builder{impl_->api,g.graph,g.tensors.data(),axis,axis,axis,axis,axis,axis,error,g.nodes,&names};
  if(!builder.binary("add",QNN_OP_ELEMENT_WISE_ADD,0,1,2,"add")||!builder.binary("sub",QNN_OP_ELEMENT_WISE_SUBTRACT,0,1,3,"subtract")||
     !builder.binary("mul",QNN_OP_ELEMENT_WISE_MULTIPLY,0,1,4,"multiply")||!builder.binary("div",QNN_OP_ELEMENT_WISE_DIVIDE,0,1,5,"divide")||
     !builder.unary("sqrt",QNN_OP_ELEMENT_WISE_SQUARE_ROOT,0,6,"sqrt")||!builder.unary("rsqrt",QNN_OP_ELEMENT_WISE_RSQRT,0,7,"rsqrt")||
     !builder.binary("reciprocal",QNN_OP_ELEMENT_WISE_DIVIDE,one,0,8,"reciprocal via divide")||
     !builder.reduce("mean",QNN_OP_REDUCE_MEAN,QNN_OP_REDUCE_MEAN_PARAM_AXES,QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,0,axis,true,9,"mean")||
     !builder.reduce("sum",QNN_OP_REDUCE_SUM,QNN_OP_REDUCE_SUM_PARAM_AXES,QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,0,axis,true,10,"sum"))return false;
  status=impl_->api.graphFinalize(g.graph,nullptr,nullptr);++metrics_.graphFinalizeCount;
  if(status!=QNN_SUCCESS){error="LN_ARITH_FINALIZE="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}return true;
}
bool Runtime::executeLayerNormArithmetic(const std::vector<float>& a,const std::vector<float>& b,std::vector<std::vector<float>>& out,std::string& error){
  auto& g=impl_->layerNormStability;if(!g.graph||g.outputs.size()!=9||a.size()!=size_t(g.rows)*g.dimension||b.size()!=a.size()){error="LN_ARITH_SHAPE";return false;}
  Qnn_Tensor_t inputs[2]{g.tensors[0],g.tensors[1]};inputs[0].v1.clientBuf={const_cast<float*>(a.data()),uint32_t(a.size()*4)};inputs[1].v1.clientBuf={const_cast<float*>(b.data()),uint32_t(b.size()*4)};
  out.resize(9);std::vector<Qnn_Tensor_t> outputs;
  for(size_t i=0;i<9;++i){auto t=g.tensors[g.outputs[i]];size_t n=1;for(uint32_t j=0;j<t.v1.rank;++j)n*=t.v1.dimensions[j];out[i].assign(n,std::numeric_limits<float>::quiet_NaN());t.v1.clientBuf={out[i].data(),uint32_t(n*4)};outputs.push_back(t);}
  auto status=impl_->api.graphExecute(g.graph,inputs,2,outputs.data(),9,nullptr,nullptr);++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),QNN_GET_ERROR_CODE(status),status==QNN_SUCCESS);
  if(status!=QNN_SUCCESS){error="LN_ARITH_EXECUTE="+std::to_string(QNN_GET_ERROR_CODE(status));return false;}return true;
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
