// Construction proof only. Captures the QNN IDs actually passed to graphAddNode.
#define main saved_p_inherited_graph_main
#include "one_layer_graph_test.cpp"
#undef main
#include "saved_p_backward_graph.h"
struct SavedEdge {std::string name,type,package;std::vector<unsigned> in,out;bool t0=false,t1=false;};
std::vector<SavedEdge> savedEdges;
Qnn_ErrorHandle_t savedNode(Qnn_GraphHandle_t graph,Qnn_OpConfig_t config) {
  auto& c=config.v1;SavedEdge e;e.name=c.name;e.type=c.typeName;e.package=c.packageName;
  for(unsigned i=0;i<c.numOfInputs;++i)e.in.push_back(c.inputTensors[i].v1.id);
  for(unsigned i=0;i<c.numOfOutputs;++i)e.out.push_back(c.outputTensors[i].v1.id);
  for(unsigned i=0;i<c.numOfParams;++i) {
    auto& p=c.params[i];
    if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))e.t0=p.scalarParam.bool8Value;
    if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))e.t1=p.scalarParam.bool8Value;
    if(!std::strcmp(p.name,QNN_OP_REDUCE_SUM_PARAM_AXES)) {
      auto& t=p.tensorParam.v1;
      if(t.dataType!=QNN_DATATYPE_UINT_32||t.rank!=1||t.dimensions[0]!=1||
         t.type!=QNN_TENSOR_TYPE_STATIC||*static_cast<const unsigned*>(t.clientBuf.data)!=3)return 1;
    }
    if(!std::strcmp(p.name,QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS)&&!p.scalarParam.bool8Value)return 1;
  }
  savedEdges.push_back(e);
  if(e.package=="qti.aisw")return node(graph,config); // Real shape validator.
  return QNN_SUCCESS; // Custom ABI correctness belongs to the protected host suites.
}
int main() {
  api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;
  api.graphAddNode=savedNode;api.graphFinalize=finish;
  unsigned graphs=0;
  for(auto mode:{savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue,savedp::Mode::Forward,
      savedp::Mode::NativeSoftmax,savedp::Mode::NativeDValue,savedp::Mode::Dual})
    for(bool obs:{true,false})for(bool tap:{true,false}) {
      savedEdges.clear();savedp::Graph z(mode,obs,tap);++graphs;
      auto find=[&](const std::string& n)->const SavedEdge& {
        auto it=std::find_if(savedEdges.begin(),savedEdges.end(),[&](auto& e){return e.name==n;});
        if(it==savedEdges.end())throw std::runtime_error("missing graph node");return *it;
      };
      if(z.p[0]==z.p[1])return 2;
      for(unsigned h=0;h<2;++h) {
        auto id=z.g.tensors[z.p[h]].v1.id;auto n="h"+std::to_string(h)+"_";
        if(z.native) {
          auto& producer=find("softmax");if(producer.out.at(h)!=id)return 3;
          unsigned producers=0;for(auto& e:savedEdges)producers+=std::count(e.out.begin(),e.out.end(),id);
          if(producers!=1||std::find(z.g.inputs.begin(),z.g.inputs.end(),z.p[h])!=z.g.inputs.end())return 4;
          if(!tap&&mode!=savedp::Mode::Forward&&z.g.tensors[z.p[h]].v1.type!=QNN_TENSOR_TYPE_NATIVE)return 5;
        }
        if(z.softmax) {
          auto& a=find(n+"softmax_product");auto& b=find(n+"softmax_dot");
          auto& c=find(n+"softmax_centered");auto& d=find(n+"dscores");
          if(a.in.at(1)!=id||d.in.at(0)!=id||b.in.at(0)!=a.out.at(0)||
             c.in.at(1)!=b.out.at(0)||d.in.at(1)!=c.out.at(0)||a.package!="qti.aisw"||
             b.type!=QNN_OP_REDUCE_SUM||c.type!=QNN_OP_ELEMENT_WISE_SUBTRACT||d.type!=QNN_OP_ELEMENT_WISE_MULTIPLY)return 6;
        }
        if(z.dvalue){auto& v=find(n+"dvalue");if(v.in.at(0)!=id||!v.t0||v.t1||v.type!=QNN_OP_MAT_MUL)return 7;}
      }
      for(auto& tensor:z.g.tensors) {
        if(!obs&&tensor.v1.dataType==QNN_DATATYPE_INT_32)return 8;
        if(tensor.v1.dataType!=QNN_DATATYPE_INT_32&&tensor.v1.dataType!=QNN_DATATYPE_UINT_32&&
           tensor.v1.dataType!=QNN_DATATYPE_FLOAT_32)return 9;
      }
      for(auto& e:savedEdges)if(e.type.find("SoftmaxBackward")!=std::string::npos)return 10;
    }
  std::printf("saved_p_backward_graph_host=PASS graphs=%u same_P_ID=1 producer_count=1 native_no_reinjection=1 production_primitives=1 reduce_axis=3 keep_dims=1 transpose0=1 transpose1=0 shape_validator=PASS device_calls=0\n",graphs);
  return 0;
}
