// Host-only construction audit. Never linked into a DSP package or device probe.
#define main inherited_saved_p_graph_main
#include "one_layer_graph_test.cpp"
#undef main
struct Edge {std::string type;std::vector<unsigned> in,out;bool transpose0=false;};
std::vector<Edge> edges;
Qnn_ErrorHandle_t savedNode(Qnn_GraphHandle_t g,Qnn_OpConfig_t cfg){auto& c=cfg.v1;Edge e;e.type=c.typeName;for(unsigned j=0;j<c.numOfInputs;++j)e.in.push_back(c.inputTensors[j].v1.id);for(unsigned j=0;j<c.numOfOutputs;++j)e.out.push_back(c.outputTensors[j].v1.id);for(unsigned j=0;j<c.numOfParams;++j)if(!std::strcmp(c.params[j].name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))e.transpose0=c.params[j].scalarParam.bool8Value;edges.push_back(e);
 if(e.type=="ForwardSoftmaxNoReceipt"){if(c.numOfInputs!=2||c.numOfOutputs!=2)return 1;for(unsigned j=0;j<2;++j){auto& t=c.outputTensors[j].v1;if(t.type!=QNN_TENSOR_TYPE_NATIVE||t.rank!=4||t.dimensions[2]!=32||t.dimensions[3]!=32)return 1;}return QNN_SUCCESS;}return node(g,cfg);}
int main(){api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=savedNode;api.graphFinalize=finish;DiagnosticGraph g("savedPNativeOwnershipHostOnly");g.receiptEnabled=false;unsigned masked[2],p[2],dp[2],dc[2],ds[2],dv[2];
 for(unsigned h=0;h<2;++h){auto mat=[&](std::string name,Qnn_TensorType_t type){return g.tensor((name+std::to_string(h)).c_str(),type,{1,1,32,32});};masked[h]=mat("masked",QNN_TENSOR_TYPE_APP_WRITE);p[h]=mat("savedP",QNN_TENSOR_TYPE_NATIVE);dp[h]=mat("dp",QNN_TENSOR_TYPE_APP_WRITE);dc[h]=mat("dc",QNN_TENSOR_TYPE_APP_WRITE);ds[h]=mat("ds",QNN_TENSOR_TYPE_APP_READ);dv[h]=mat("dv",QNN_TENSOR_TYPE_APP_READ);}
 g.node("p_producer","ForwardSoftmax","HexaFp32",{masked[0],masked[1]},{p[0],p[1]});g.node("softmax_backward","SoftmaxBackward","HexaFp32",{dp[0],dp[1],p[0],p[1]},{ds[0],ds[1]});for(unsigned h=0;h<2;++h)op::mm(g,"dvalue"+std::to_string(h),p[h],dc[h],dv[h],true,false);g.finalize();
 if(edges.size()!=4||edges[1].type!="SoftmaxBackwardNoReceipt")return 1;
 for(unsigned h=0;h<2;++h){unsigned id=g.tensors[p[h]].v1.id;if(edges[0].out[h]!=id||edges[1].in[2+h]!=id||edges[2+h].in[0]!=id||!edges[2+h].transpose0)return 2;if(std::find(g.inputs.begin(),g.inputs.end(),p[h])!=g.inputs.end()||std::find(g.outputs.begin(),g.outputs.end(),p[h])!=g.outputs.end())return 3;}
 for(auto& t:g.tensors)if(t.v1.dataType==QNN_DATATYPE_INT_32)return 4;
 std::puts("saved_p_ownership_host=PASS same_native_P_IDs=1 softmax_backward=1 dvalue_transpose0=1 host_P_reinjection=0 receipt_tensors=0 DSP_BACKWARD_NOT_EXECUTED=1 device_calls=0");return 0;}
