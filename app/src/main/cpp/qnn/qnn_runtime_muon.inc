namespace {

Qnn_ErrorHandle_t addMuonBinary(
    QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
    const char* name, const char* operation, const Qnn_Tensor_t& first,
    const Qnn_Tensor_t& second, Qnn_Tensor_t* output) {
  Qnn_Tensor_t inputs[2]{first, second};
  return addSimpleNode(api, graph, name, operation, inputs, 2, output);
}

Qnn_ErrorHandle_t addMuonUnary(
    QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
    const char* name, const char* operation, const Qnn_Tensor_t& input,
    Qnn_Tensor_t* output) {
  Qnn_Tensor_t copy = input;
  return addSimpleNode(api, graph, name, operation, &copy, 1, output);
}

Qnn_ErrorHandle_t addMuonMatMul(
    QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
    const char* name, const Qnn_Tensor_t& first, const Qnn_Tensor_t& second,
    bool transposeSecond, Qnn_Tensor_t* output) {
  Qnn_Tensor_t inputs[2]{first, second};
  Qnn_Scalar_t transpose = QNN_SCALAR_INIT;
  transpose.dataType = QNN_DATATYPE_BOOL_8;
  transpose.bool8Value = transposeSecond;
  Qnn_Param_t parameter = QNN_PARAM_INIT;
  parameter.paramType = QNN_PARAMTYPE_SCALAR;
  parameter.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
  parameter.scalarParam = transpose;
  Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
  operation.v1.name = name;
  operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  operation.v1.typeName = QNN_OP_MAT_MUL;
  operation.v1.numOfParams = transposeSecond ? 1 : 0;
  operation.v1.params = transposeSecond ? &parameter : nullptr;
  operation.v1.numOfInputs = 2;
  operation.v1.inputTensors = inputs;
  operation.v1.numOfOutputs = 1;
  operation.v1.outputTensors = output;
  return api.graphAddNode(graph, operation);
}

Qnn_ErrorHandle_t addMuonReduce(
    QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
    const char* name, const Qnn_Tensor_t& input, const Qnn_Tensor_t& axes,
    Qnn_Tensor_t* output) {
  Qnn_Param_t parameters[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  parameters[0].paramType = QNN_PARAMTYPE_TENSOR;
  parameters[0].name = QNN_OP_REDUCE_SUM_PARAM_AXES;
  parameters[0].tensorParam = axes;
  parameters[1].paramType = QNN_PARAMTYPE_SCALAR;
  parameters[1].name = QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS;
  parameters[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  parameters[1].scalarParam.bool8Value = true;
  Qnn_Tensor_t inputCopy = input;
  Qnn_OpConfig_t operation = QNN_OPCONFIG_INIT;
  operation.v1.name = name;
  operation.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  operation.v1.typeName = QNN_OP_REDUCE_SUM;
  operation.v1.numOfParams = 2;
  operation.v1.params = parameters;
  operation.v1.numOfInputs = 1;
  operation.v1.inputTensors = &inputCopy;
  operation.v1.numOfOutputs = 1;
  operation.v1.outputTensors = output;
  return api.graphAddNode(graph, operation);
}

bool muonFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

bool Runtime::prepareMuonOptimizer(uint32_t squareBatch,
                                   uint32_t rectangularBatch,
                                   uint32_t rows,
                                   uint32_t rectangularColumns,
                                   uint32_t nsSteps,
                                   bool diagnosticOutputs,
                                   std::string& error) {
  using MuonGraph = Impl::MuonOptimizerGraph;
  if (!impl_ || !impl_->context || squareBatch == 0 ||
      rectangularBatch == 0 || rows == 0 || rectangularColumns < rows ||
      nsSteps != 5) {
    error = "HTP_MUON_PREPARE_INVALID_RUNTIME_OR_SHAPE";
    return false;
  }
  auto& graph = impl_->muonOptimizer;
  graph = {};
  graph.nsSteps = nsSteps;
  graph.diagnosticOutputs = diagnosticOutputs;
  const auto configure = [&](MuonGraph::Branch* branch, uint32_t batch,
                             uint32_t columns) {
    branch->batch = batch;
    branch->rows = rows;
    branch->columns = columns;
    branch->elements = batch * rows * columns;
    branch->matrixDims[0] = batch;
    branch->matrixDims[1] = rows;
    branch->matrixDims[2] = columns;
    branch->gramDims[0] = batch;
    branch->gramDims[1] = rows;
    branch->gramDims[2] = rows;
    branch->normDims[0] = batch;
    branch->normDims[1] = 1;
    branch->normDims[2] = 1;
  };
  configure(&graph.square, squareBatch, rows);
  configure(&graph.rectangular, rectangularBatch, rectangularColumns);
  const HtpGraphConfigs htpConfigs = buildHtpGraphConfigs(options_);
  if (htpConfigs.failClosedInvalid) {
    error = "HTP_MUON_GRAPH_CREATE_INVALID_PRECISION_OPTION";
    return false;
  }
  auto started = Clock::now();
  auto status = impl_->api.graphCreate(impl_->context, "phonelm_htp_muon_v1",
                                       htpConfigs.array, &graph.graph);
  metrics_.graphCreateUs += elapsedUs(started);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_GRAPH_CREATE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if (!htpConfigs.description.empty())
    diagnostics_ += htpConfigs.description + "\n";
  const auto make = [&](Qnn_Tensor_t* value, const std::string& name,
                        Qnn_TensorType_t type, uint32_t* dims,
                        uint32_t rank = 3,
                        Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32) {
    graph.names.push_back(name);
    tensor(*value, graph.names.back().c_str(), type, dims, dataType, rank);
    ++graph.tensorCount;
  };
  make(&graph.learningRate, "muon_learning_rate", QNN_TENSOR_TYPE_APP_WRITE,
       graph.scalarDims);
  const auto makeStatic = [&](Qnn_Tensor_t* value, const char* name,
                              float* data) {
    make(value, name, QNN_TENSOR_TYPE_STATIC, graph.scalarDims);
    value->v1.clientBuf = {data, sizeof(float)};
  };
  makeStatic(&graph.momentum, "muon_momentum_coefficient",
             &graph.momentumData);
  makeStatic(&graph.gradientCoefficient, "muon_gradient_coefficient",
             &graph.gradientCoefficientData);
  makeStatic(&graph.nsA, "muon_ns_a", &graph.nsAData);
  makeStatic(&graph.nsB, "muon_ns_b", &graph.nsBData);
  makeStatic(&graph.nsC, "muon_ns_c", &graph.nsCData);
  makeStatic(&graph.normScale, "muon_normalization_scale",
             &graph.normScaleData);
  makeStatic(&graph.epsilon, "muon_ns_epsilon", &graph.epsilonData);
  const auto makeBranch = [&](MuonGraph::Branch* branch,
                              const std::string& prefix) {
    for (std::size_t index = 0; index < MuonGraph::TENSOR_COUNT; ++index) {
      Qnn_TensorType_t type = QNN_TENSOR_TYPE_NATIVE;
      if (index <= MuonGraph::UPDATE_SCALE) type = QNN_TENSOR_TYPE_APP_WRITE;
      if (index == MuonGraph::NEXT_MOMENTUM ||
          index == MuonGraph::NEXT_WEIGHT)
        type = QNN_TENSOR_TYPE_APP_READ;
      const std::size_t firstIterationOutput =
          MuonGraph::ITERATION_BEGIN + MuonGraph::ITERATION_STRIDE - 1;
      const std::size_t finalIterationOutput =
          MuonGraph::SCALED_ORTHOGONAL - 1;
      if (diagnosticOutputs &&
          (index == finalIterationOutput ||
           (branch == &graph.square &&
            (index == MuonGraph::NORMALIZED ||
             index == firstIterationOutput))))
        type = QNN_TENSOR_TYPE_APP_READ;
      uint32_t* dims = branch->matrixDims;
      if (index == MuonGraph::NORM_SQUARED || index == MuonGraph::NORM ||
          index == MuonGraph::DENOMINATOR ||
          index == MuonGraph::UPDATE_SCALE)
        dims = branch->normDims;
      if (index >= MuonGraph::ITERATION_BEGIN &&
          index < MuonGraph::SCALED_ORTHOGONAL) {
        const std::size_t within =
            (index - MuonGraph::ITERATION_BEGIN) % MuonGraph::ITERATION_STRIDE;
        if (within <= 4) dims = branch->gramDims;
      }
      make(&branch->tensors[index], prefix + "_tensor_" +
               std::to_string(index), type, dims);
    }
    make(&branch->axes, prefix + "_norm_axes", QNN_TENSOR_TYPE_STATIC,
         branch->axesDims, 1, QNN_DATATYPE_UINT_32);
    branch->axes.v1.clientBuf = {branch->axesData,
                                 uint32_t(sizeof(branch->axesData))};
  };
  makeBranch(&graph.square, "muon_square");
  makeBranch(&graph.rectangular, "muon_rectangular");
  for (Qnn_Tensor_t* constant :
       {&graph.learningRate, &graph.momentum, &graph.gradientCoefficient,
        &graph.nsA, &graph.nsB, &graph.nsC, &graph.normScale,
        &graph.epsilon}) {
    status = impl_->api.tensorCreateGraphTensor(graph.graph, constant);
    if (status != QNN_SUCCESS) {
      error = "HTP_MUON_TENSOR_CREATE=" +
              std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  for (MuonGraph::Branch* branch : {&graph.square, &graph.rectangular}) {
    for (auto& value : branch->tensors) {
      status = impl_->api.tensorCreateGraphTensor(graph.graph, &value);
      if (status != QNN_SUCCESS) {
        error = "HTP_MUON_TENSOR_CREATE=" +
                std::to_string(QNN_GET_ERROR_CODE(status));
        return false;
      }
    }
    status = impl_->api.tensorCreateGraphTensor(graph.graph, &branch->axes);
    if (status != QNN_SUCCESS) {
      error = "HTP_MUON_AXES_TENSOR_CREATE=" +
              std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  const auto addBranch = [&](MuonGraph::Branch* branch,
                             const std::string& prefix) {
    auto& t = branch->tensors;
    const auto binary = [&](const std::string& name, const char* operation,
                            const Qnn_Tensor_t& first,
                            const Qnn_Tensor_t& second, Qnn_Tensor_t* output) {
      const auto result = addMuonBinary(impl_->api, graph.graph, name.c_str(),
                                        operation, first, second, output);
      if (result == QNN_SUCCESS) ++graph.nodeCount;
      return result;
    };
    if ((status = binary(prefix + "_momentum_scaled",
                         QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::PRIOR_MOMENTUM], graph.momentum,
                         &t[MuonGraph::MOMENTUM_SCALED])) != QNN_SUCCESS ||
        (status = binary(prefix + "_gradient_scaled",
                         QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::GRADIENT], graph.gradientCoefficient,
                         &t[MuonGraph::GRADIENT_SCALED])) != QNN_SUCCESS ||
        (status = binary(prefix + "_momentum_next", QNN_OP_ELEMENT_WISE_ADD,
                         t[MuonGraph::MOMENTUM_SCALED],
                         t[MuonGraph::GRADIENT_SCALED],
                         &t[MuonGraph::NEXT_MOMENTUM])) != QNN_SUCCESS ||
        (status = binary(prefix + "_nesterov_momentum_scaled",
                         QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::NEXT_MOMENTUM], graph.momentum,
                         &t[MuonGraph::NESTEROV_MOMENTUM_SCALED])) !=
            QNN_SUCCESS ||
        (status = binary(prefix + "_nesterov", QNN_OP_ELEMENT_WISE_ADD,
                         t[MuonGraph::GRADIENT_SCALED],
                         t[MuonGraph::NESTEROV_MOMENTUM_SCALED],
                         &t[MuonGraph::NESTEROV])) != QNN_SUCCESS ||
        (status = binary(prefix + "_normalization_scale",
                         QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::NESTEROV], graph.normScale,
                         &t[MuonGraph::NORM_INPUT])) != QNN_SUCCESS ||
        (status = binary(prefix + "_square", QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::NORM_INPUT], t[MuonGraph::NORM_INPUT],
                         &t[MuonGraph::SQUARED])) != QNN_SUCCESS)
      return false;
    status = addMuonReduce(impl_->api, graph.graph,
                           (prefix + "_reduce_axes_1_2").c_str(),
                           t[MuonGraph::SQUARED], branch->axes,
                           &t[MuonGraph::NORM_SQUARED]);
    if (status != QNN_SUCCESS) return false;
    ++graph.nodeCount;
    status = addMuonUnary(impl_->api, graph.graph,
                          (prefix + "_sqrt").c_str(),
                          QNN_OP_ELEMENT_WISE_SQUARE_ROOT,
                          t[MuonGraph::NORM_SQUARED], &t[MuonGraph::NORM]);
    if (status != QNN_SUCCESS) return false;
    ++graph.nodeCount;
    if ((status = binary(prefix + "_epsilon", QNN_OP_ELEMENT_WISE_ADD,
                         t[MuonGraph::NORM], graph.epsilon,
                         &t[MuonGraph::DENOMINATOR])) != QNN_SUCCESS ||
        (status = binary(prefix + "_normalize", QNN_OP_ELEMENT_WISE_DIVIDE,
                         t[MuonGraph::NORM_INPUT], t[MuonGraph::DENOMINATOR],
                         &t[MuonGraph::NORMALIZED])) != QNN_SUCCESS)
      return false;
    Qnn_Tensor_t* x = &t[MuonGraph::NORMALIZED];
    for (uint32_t iteration = 0; iteration < graph.nsSteps; ++iteration) {
      const std::size_t base = MuonGraph::ITERATION_BEGIN +
                               iteration * MuonGraph::ITERATION_STRIDE;
      const std::string iter = prefix + "_ns_" + std::to_string(iteration + 1);
      status = addMuonMatMul(impl_->api, graph.graph,
                             (iter + "_xx_t").c_str(), *x, *x, true,
                             &t[base]);
      if (status != QNN_SUCCESS) return false;
      ++graph.nodeCount;
      status = addMuonMatMul(impl_->api, graph.graph,
                             (iter + "_a2").c_str(), t[base], t[base], false,
                             &t[base + 1]);
      if (status != QNN_SUCCESS) return false;
      ++graph.nodeCount;
      if ((status = binary(iter + "_b_a", QNN_OP_ELEMENT_WISE_MULTIPLY,
                           t[base], graph.nsB, &t[base + 2])) != QNN_SUCCESS ||
          (status = binary(iter + "_c_a2", QNN_OP_ELEMENT_WISE_MULTIPLY,
                           t[base + 1], graph.nsC,
                           &t[base + 3])) != QNN_SUCCESS ||
          (status = binary(iter + "_bmat", QNN_OP_ELEMENT_WISE_ADD,
                           t[base + 2], t[base + 3],
                           &t[base + 4])) != QNN_SUCCESS)
        return false;
      status = addMuonMatMul(impl_->api, graph.graph,
                             (iter + "_bx").c_str(), t[base + 4], *x, false,
                             &t[base + 5]);
      if (status != QNN_SUCCESS) return false;
      ++graph.nodeCount;
      if ((status = binary(iter + "_a_x", QNN_OP_ELEMENT_WISE_MULTIPLY,
                           *x, graph.nsA, &t[base + 6])) != QNN_SUCCESS ||
          (status = binary(iter + "_x_next", QNN_OP_ELEMENT_WISE_ADD,
                           t[base + 6], t[base + 5],
                           &t[base + 7])) != QNN_SUCCESS)
        return false;
      x = &t[base + 7];
    }
    if ((status = binary(prefix + "_aspect_scale",
                         QNN_OP_ELEMENT_WISE_MULTIPLY, *x,
                         t[MuonGraph::UPDATE_SCALE],
                         &t[MuonGraph::SCALED_ORTHOGONAL])) != QNN_SUCCESS ||
        (status = binary(prefix + "_learning_rate_scale",
                         QNN_OP_ELEMENT_WISE_MULTIPLY,
                         t[MuonGraph::SCALED_ORTHOGONAL], graph.learningRate,
                         &t[MuonGraph::SCALED_UPDATE])) != QNN_SUCCESS ||
        (status = binary(prefix + "_weight_next",
                         QNN_OP_ELEMENT_WISE_SUBTRACT,
                         t[MuonGraph::CURRENT], t[MuonGraph::SCALED_UPDATE],
                         &t[MuonGraph::NEXT_WEIGHT])) != QNN_SUCCESS)
      return false;
    return true;
  };
  if (!addBranch(&graph.square, "muon_square") ||
      !addBranch(&graph.rectangular, "muon_rectangular")) {
    error = "HTP_MUON_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  started = Clock::now();
  status = impl_->api.graphFinalize(graph.graph, nullptr, nullptr);
  metrics_.graphFinalizeUs += elapsedUs(started);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_GRAPH_FINALIZE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ +=
      "muon_optimizer_backend=HTP\n"
      "muon_state_residency=HOST_PING_PONG\n"
      "muon_parameter_handoff=CPU_PACKED_PING_PONG\n"
      "aux_adam_backend=CPU\n"
      "muon_optimizer_graph_count=1\n"
      "muon_optimizer_execute_count_per_update=1\n"
      "muon_square_batch=" + std::to_string(squareBatch) +
      "\nmuon_rectangular_batch=" + std::to_string(rectangularBatch) +
      "\nmuon_reduce_axes=1,2\nmuon_reduce_keep_dims=true\n"
      "muon_normalization_prescale=1024\n"
      "muon_app_tensor_datatype=FLOAT32\nmuon_native_tensor_datatype=FLOAT32\n"
      "muon_graph_node_count=" + std::to_string(graph.nodeCount) +
      "\nmuon_graph_tensor_count=" + std::to_string(graph.tensorCount) + "\n";
  return true;
}

bool Runtime::executeMuonOptimizer(
    const std::vector<float>& currentSquare,
    const std::vector<float>& gradientSquare,
    const std::vector<float>& momentumSquare,
    const std::vector<float>& scaleSquare,
    const std::vector<float>& currentRectangular,
    const std::vector<float>& gradientRectangular,
    const std::vector<float>& momentumRectangular,
    const std::vector<float>& scaleRectangular, float learningRate,
    MuonOptimizerOutputs& outputs, std::string& error) {
  using MuonGraph = Impl::MuonOptimizerGraph;
  auto& graph = impl_->muonOptimizer;
  const auto validBranch = [](const MuonGraph::Branch& branch,
                              const std::vector<float>& current,
                              const std::vector<float>& gradient,
                              const std::vector<float>& momentum,
                              const std::vector<float>& scale) {
    return current.size() == branch.elements &&
           gradient.size() == branch.elements &&
           momentum.size() == branch.elements && scale.size() == branch.batch &&
           muonFinite(current) && muonFinite(gradient) &&
           muonFinite(momentum) && muonFinite(scale);
  };
  if (!graph.graph || !std::isfinite(learningRate) || learningRate <= 0.0f ||
      !validBranch(graph.square, currentSquare, gradientSquare, momentumSquare,
                   scaleSquare) ||
      !validBranch(graph.rectangular, currentRectangular,
                   gradientRectangular, momentumRectangular,
                   scaleRectangular)) {
    error = "HTP_MUON_EXECUTE_INPUT_INVALID";
    return false;
  }
  Qnn_Tensor_t inputs[9] = {
      graph.square.tensors[MuonGraph::CURRENT],
      graph.square.tensors[MuonGraph::GRADIENT],
      graph.square.tensors[MuonGraph::PRIOR_MOMENTUM],
      graph.square.tensors[MuonGraph::UPDATE_SCALE],
      graph.rectangular.tensors[MuonGraph::CURRENT],
      graph.rectangular.tensors[MuonGraph::GRADIENT],
      graph.rectangular.tensors[MuonGraph::PRIOR_MOMENTUM],
      graph.rectangular.tensors[MuonGraph::UPDATE_SCALE], graph.learningRate};
  const std::vector<float>* data[8] = {
      &currentSquare, &gradientSquare, &momentumSquare, &scaleSquare,
      &currentRectangular, &gradientRectangular, &momentumRectangular,
      &scaleRectangular};
  auto started = Clock::now();
  for (std::size_t index = 0; index < 8; ++index)
    inputs[index].v1.clientBuf = {
        const_cast<float*>(data[index]->data()),
        uint32_t(data[index]->size() * sizeof(float))};
  inputs[8].v1.clientBuf = {&learningRate, sizeof(float)};
  constexpr uint32_t poisonBits = 0x7fc0a55au;
  float poison = 0.0f;
  std::memcpy(&poison, &poisonBits, sizeof(poison));
  outputs.nextMomentumSquare.assign(graph.square.elements, poison);
  outputs.nextWeightsSquare.assign(graph.square.elements, poison);
  outputs.nextMomentumRectangular.assign(graph.rectangular.elements, poison);
  outputs.nextWeightsRectangular.assign(graph.rectangular.elements, poison);
  std::vector<Qnn_Tensor_t> outputTensors = {
      graph.square.tensors[MuonGraph::NEXT_MOMENTUM],
      graph.square.tensors[MuonGraph::NEXT_WEIGHT],
      graph.rectangular.tensors[MuonGraph::NEXT_MOMENTUM],
      graph.rectangular.tensors[MuonGraph::NEXT_WEIGHT]};
  std::vector<std::vector<float>*> outputData = {
      &outputs.nextMomentumSquare, &outputs.nextWeightsSquare,
      &outputs.nextMomentumRectangular, &outputs.nextWeightsRectangular};
  if (graph.diagnosticOutputs) {
    outputs.normalizedSquare.assign(graph.square.elements, poison);
    outputs.nsIteration1Square.assign(graph.square.elements, poison);
    outputs.orthogonalSquare.assign(graph.square.elements, poison);
    outputs.orthogonalRectangular.assign(graph.rectangular.elements, poison);
    outputTensors.push_back(graph.square.tensors[MuonGraph::NORMALIZED]);
    outputTensors.push_back(graph.square.tensors[
        MuonGraph::ITERATION_BEGIN + MuonGraph::ITERATION_STRIDE - 1]);
    outputTensors.push_back(graph.square.tensors[
        MuonGraph::SCALED_ORTHOGONAL - 1]);
    outputTensors.push_back(graph.rectangular.tensors[
        MuonGraph::SCALED_ORTHOGONAL - 1]);
    outputData.push_back(&outputs.normalizedSquare);
    outputData.push_back(&outputs.nsIteration1Square);
    outputData.push_back(&outputs.orthogonalSquare);
    outputData.push_back(&outputs.orthogonalRectangular);
  }
  for (std::size_t index = 0; index < outputData.size(); ++index)
    outputTensors[index].v1.clientBuf = {
        outputData[index]->data(),
        uint32_t(outputData[index]->size() * sizeof(float))};
  outputs.bindUs = elapsedUs(started);
  started = Clock::now();
  const auto status = impl_->api.graphExecute(
      graph.graph, inputs, 9, outputTensors.data(),
      uint32_t(outputTensors.size()), nullptr, nullptr);
  outputs.executeUs = elapsedUs(started);
  metrics_.executeUs.push_back(outputs.executeUs);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_GRAPH_EXECUTE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  std::uint64_t poisonRemaining = 0;
  for (const auto* values : outputData)
    for (const float value : *values) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      poisonRemaining += bits == poisonBits;
    }
  if (poisonRemaining != 0) {
    error = "HTP_MUON_OUTPUT_POISON_REMAINING=" +
            std::to_string(poisonRemaining);
    return false;
  }
  for (const auto* values : outputData)
    if (!muonFinite(*values)) {
      error = "HTP_MUON_OUTPUT_NONFINITE";
      return false;
    }
  return true;
}
