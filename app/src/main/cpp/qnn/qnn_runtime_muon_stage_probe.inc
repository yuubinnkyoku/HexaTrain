// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Diagnostic-only single-matrix Newton-Schulz stage probe graph.
//
// This graph is intentionally separate from the production Muon optimizer
// graph: it binds the frozen 64x64 normalized input directly and exposes
// the exact iteration-1 operation order as APP_READ tensors:
//
//   A  = X0 @ X0^T            (MatMul, transposeIn1=true)
//   A2 = A @ A                (MatMul, no transpose)
//   B  = b*A + c*A2           (ElementWiseMultiply x2, ElementWiseAdd)
//   BX = B @ X0               (MatMul, no transpose)
//   X1 = a*X0 + BX            (ElementWiseMultiply, ElementWiseAdd)
//
// A standalone A=XX^T MatMul node with the identical configuration is added
// as an independent verification of the first MatMul backend path.
//
// None of the production Muon graph, optimizer state, packing, thresholds,
// NS coefficients, or NS step counts is changed by this diagnostic graph.

#include <algorithm>
#include <array>
#include <cstring>
#include <list>

namespace {

Qnn_ErrorHandle_t addStageMatMul(
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

Qnn_ErrorHandle_t addStageBinary(
    QNN_INTERFACE_VER_TYPE& api, Qnn_GraphHandle_t graph,
    const char* name, const char* operation, const Qnn_Tensor_t& first,
    const Qnn_Tensor_t& second, Qnn_Tensor_t* output) {
  Qnn_Tensor_t inputs[2]{first, second};
  return addSimpleNode(api, graph, name, operation, inputs, 2, output);
}

bool stageFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

bool Runtime::prepareMuonNsStageProbe(std::string& error) {
  if (!impl_ || !impl_->context) {
    error = "HTP_MUON_STAGE_PROBE_INVALID_RUNTIME";
    return false;
  }
  auto& probe = impl_->muonNsStageProbe;
  probe = {};
  constexpr std::uint32_t kRows = MuonNsStageOutputs::kRows;
  constexpr std::uint32_t kColumns = MuonNsStageOutputs::kColumns;
  probe.matrixDims[0] = 1;
  probe.matrixDims[1] = kRows;
  probe.matrixDims[2] = kColumns;
  probe.gramDims[0] = 1;
  probe.gramDims[1] = kRows;
  probe.gramDims[2] = kRows;
  probe.elements = kRows * kColumns;
  probe.gramElements = kRows * kRows;
  const HtpGraphConfigs htpConfigs = buildHtpGraphConfigs(options_);
  if (htpConfigs.failClosedInvalid) {
    error = "HTP_MUON_STAGE_PROBE_INVALID_PRECISION_OPTION";
    return false;
  }
  auto started = Clock::now();
  auto status = impl_->api.graphCreate(impl_->context,
                                       "phonelm_htp_muon_ns_stage_probe",
                                       htpConfigs.array, &probe.graph);
  metrics_.graphCreateUs += elapsedUs(started);
  ++metrics_.graphCreateCount;
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_CREATE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  if (!htpConfigs.description.empty())
    diagnostics_ += htpConfigs.description + "\n";
  const auto make = [&](Qnn_Tensor_t* value, const std::string& name,
                        Qnn_TensorType_t type, uint32_t* dims) {
    probe.names.push_back(name);
    tensor(*value, probe.names.back().c_str(), type, dims,
           QNN_DATATYPE_FLOAT_32, 3);
    ++probe.tensorCount;
  };
  make(&probe.input, "muon_stage_probe_x0", QNN_TENSOR_TYPE_APP_WRITE,
       probe.matrixDims);
  make(&probe.a, "muon_stage_probe_a", QNN_TENSOR_TYPE_APP_READ,
       probe.gramDims);
  make(&probe.a2, "muon_stage_probe_a2", QNN_TENSOR_TYPE_APP_READ,
       probe.gramDims);
  make(&probe.scaledA, "muon_stage_probe_scaled_a", QNN_TENSOR_TYPE_NATIVE,
       probe.gramDims);
  make(&probe.scaledA2, "muon_stage_probe_scaled_a2", QNN_TENSOR_TYPE_NATIVE,
       probe.gramDims);
  make(&probe.b, "muon_stage_probe_b", QNN_TENSOR_TYPE_APP_READ,
       probe.gramDims);
  make(&probe.bx, "muon_stage_probe_bx", QNN_TENSOR_TYPE_APP_READ,
       probe.matrixDims);
  make(&probe.scaledX, "muon_stage_probe_scaled_x", QNN_TENSOR_TYPE_NATIVE,
       probe.matrixDims);
  make(&probe.x1, "muon_stage_probe_x1", QNN_TENSOR_TYPE_APP_READ,
       probe.matrixDims);
  make(&probe.standaloneA, "muon_stage_probe_standalone_a",
       QNN_TENSOR_TYPE_APP_READ, probe.gramDims);
  const auto makeStatic = [&](Qnn_Tensor_t* value, const char* name,
                              float* data) {
    make(value, name, QNN_TENSOR_TYPE_STATIC, probe.scalarDims);
    value->v1.clientBuf = {data, sizeof(float)};
  };
  makeStatic(&probe.nsA, "muon_stage_probe_ns_a", &probe.nsAData);
  makeStatic(&probe.nsB, "muon_stage_probe_ns_b", &probe.nsBData);
  makeStatic(&probe.nsC, "muon_stage_probe_ns_c", &probe.nsCData);
  for (Qnn_Tensor_t* constant : {&probe.input, &probe.nsA, &probe.nsB,
                                 &probe.nsC}) {
    status = impl_->api.tensorCreateGraphTensor(probe.graph, constant);
    if (status != QNN_SUCCESS) {
      error = "HTP_MUON_STAGE_PROBE_TENSOR_CREATE=" +
              std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  for (Qnn_Tensor_t* value :
       {&probe.a, &probe.a2, &probe.scaledA, &probe.scaledA2, &probe.b,
        &probe.bx, &probe.scaledX, &probe.x1, &probe.standaloneA}) {
    status = impl_->api.tensorCreateGraphTensor(probe.graph, value);
    if (status != QNN_SUCCESS) {
      error = "HTP_MUON_STAGE_PROBE_TENSOR_CREATE=" +
              std::to_string(QNN_GET_ERROR_CODE(status));
      return false;
    }
  }
  status = addStageMatMul(impl_->api, probe.graph, "muon_stage_probe_xx_t",
                          probe.input, probe.input, true, &probe.a);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  ++probe.nodeCount;
  status = addStageMatMul(impl_->api, probe.graph, "muon_stage_probe_a2",
                          probe.a, probe.a, false, &probe.a2);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  ++probe.nodeCount;
  if ((status = addStageBinary(impl_->api, probe.graph,
                               "muon_stage_probe_b_a",
                               QNN_OP_ELEMENT_WISE_MULTIPLY, probe.a, probe.nsB,
                               &probe.scaledA)) != QNN_SUCCESS ||
      (status = addStageBinary(impl_->api, probe.graph,
                               "muon_stage_probe_c_a2",
                               QNN_OP_ELEMENT_WISE_MULTIPLY, probe.a2,
                               probe.nsC, &probe.scaledA2)) != QNN_SUCCESS ||
      (status = addStageBinary(impl_->api, probe.graph, "muon_stage_probe_b",
                               QNN_OP_ELEMENT_WISE_ADD, probe.scaledA,
                               probe.scaledA2, &probe.b)) != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  status = addStageMatMul(impl_->api, probe.graph, "muon_stage_probe_bx",
                          probe.b, probe.input, false, &probe.bx);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  ++probe.nodeCount;
  if ((status = addStageBinary(impl_->api, probe.graph,
                               "muon_stage_probe_a_x",
                               QNN_OP_ELEMENT_WISE_MULTIPLY, probe.input,
                               probe.nsA, &probe.scaledX)) != QNN_SUCCESS ||
      (status = addStageBinary(impl_->api, probe.graph,
                               "muon_stage_probe_x_next",
                               QNN_OP_ELEMENT_WISE_ADD, probe.scaledX,
                               probe.bx, &probe.x1)) != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  // Standalone A verification: identical MatMul configuration on the same
  // input, independent of the chained iteration-1 path.
  status = addStageMatMul(impl_->api, probe.graph,
                          "muon_stage_probe_standalone_xx_t", probe.input,
                          probe.input, true, &probe.standaloneA);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_ADD_NODE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  ++probe.nodeCount;
  started = Clock::now();
  status = impl_->api.graphFinalize(probe.graph, nullptr, nullptr);
  metrics_.graphFinalizeUs += elapsedUs(started);
  ++metrics_.graphFinalizeCount;
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_FINALIZE=" +
            std::to_string(QNN_GET_ERROR_CODE(status));
    return false;
  }
  diagnostics_ +=
      "muon_ns_stage_probe_backend=HTP\n"
      "muon_ns_stage_probe_graph_count=1\n"
      "muon_ns_stage_probe_batch=1\n"
      "muon_ns_stage_probe_rows=64\n"
      "muon_ns_stage_probe_columns=64\n"
      "muon_ns_stage_probe_app_tensor_datatype=FLOAT32\n"
      "muon_ns_stage_probe_native_tensor_datatype=FLOAT32\n"
      "muon_ns_stage_probe_graph_node_count=" +
      std::to_string(probe.nodeCount) +
      "\nmuon_ns_stage_probe_graph_tensor_count=" +
      std::to_string(probe.tensorCount) + "\n";
  return true;
}

bool Runtime::executeMuonNsStageProbe(const std::vector<float>& normalizedInput,
                                      MuonNsStageOutputs& outputs,
                                      std::string& error) {
  auto& probe = impl_->muonNsStageProbe;
  constexpr std::uint32_t kRows = MuonNsStageOutputs::kRows;
  constexpr std::uint32_t kColumns = MuonNsStageOutputs::kColumns;
  if (!probe.graph ||
      normalizedInput.size() != std::size_t(kRows) * kColumns ||
      !stageFinite(normalizedInput)) {
    error = "HTP_MUON_STAGE_PROBE_EXECUTE_INPUT_INVALID";
    return false;
  }
  Qnn_Tensor_t input = probe.input;
  auto started = Clock::now();
  input.v1.clientBuf = {const_cast<float*>(normalizedInput.data()),
                        uint32_t(normalizedInput.size() * sizeof(float))};
  constexpr uint32_t poisonBits = 0x7fc0a55au;
  float poison = 0.0f;
  std::memcpy(&poison, &poisonBits, sizeof(poison));
  outputs.x0 = normalizedInput;
  outputs.a.assign(probe.gramElements, poison);
  outputs.a2.assign(probe.gramElements, poison);
  outputs.b.assign(probe.gramElements, poison);
  outputs.bx.assign(probe.elements, poison);
  outputs.x1.assign(probe.elements, poison);
  outputs.standaloneA.assign(probe.gramElements, poison);
  std::vector<Qnn_Tensor_t> outputTensors = {
      probe.a, probe.a2, probe.b, probe.bx, probe.x1, probe.standaloneA};
  std::vector<std::vector<float>*> outputData = {
      &outputs.a, &outputs.a2, &outputs.b,
      &outputs.bx, &outputs.x1, &outputs.standaloneA};
  for (std::size_t index = 0; index < outputData.size(); ++index)
    outputTensors[index].v1.clientBuf = {
        outputData[index]->data(),
        uint32_t(outputData[index]->size() * sizeof(float))};
  outputs.bindUs = elapsedUs(started);
  started = Clock::now();
  const auto status =
      impl_->api.graphExecute(probe.graph, &input, 1, outputTensors.data(),
                              uint32_t(outputTensors.size()), nullptr, nullptr);
  outputs.executeUs = elapsedUs(started);
  metrics_.executeUs.push_back(outputs.executeUs);
  ++metrics_.graphExecuteCount;
  recordGraphExecuteResult(QNN_GET_ERROR_CODE(status),
                           QNN_GET_ERROR_CODE(status), status == QNN_SUCCESS);
  if (status != QNN_SUCCESS) {
    error = "HTP_MUON_STAGE_PROBE_GRAPH_EXECUTE=" +
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
    error = "HTP_MUON_STAGE_PROBE_OUTPUT_POISON_REMAINING=" +
            std::to_string(poisonRemaining);
    return false;
  }
  for (const auto* values : outputData)
    if (!stageFinite(*values)) {
      error = "HTP_MUON_STAGE_PROBE_OUTPUT_NONFINITE";
      return false;
    }
  return true;
}
