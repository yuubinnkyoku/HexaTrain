#define main inherited_graph_main
#include "one_layer_graph_test.cpp"
#undef main
#include "ordered_qkv_graph.h"
#include "qk_scale_graph.h"
unsigned newNodes=0;
Qnn_ErrorHandle_t frontierNode(Qnn_GraphHandle_t g,Qnn_OpConfig_t config){auto& c=config.v1;std::string t=c.typeName;bool q=t.rfind("ForwardQKRaw",0)==0,s=t.rfind("BackwardScoreScale",0)==0;
 if(q||s){bool obs=t.find("NoReceipt")==std::string::npos;unsigned nx=q?4:5,ny=q?2:4;if(c.numOfInputs!=nx||c.numOfOutputs!=ny+obs)return 1;for(unsigned j=0;j<nx+ny;++j){auto& v=(j<nx?c.inputTensors[j]:c.outputTensors[j-nx]).v1;if(v.rank!=4||v.dimensions[0]!=1||v.dimensions[1]!=1||v.dimensions[2]!=(s&&j==4?1u:32u)||v.dimensions[3]!=(s&&j==4?1u:32u)||v.dataType!=QNN_DATATYPE_FLOAT_32)return 1;}if(obs&&(c.outputTensors[ny].v1.dataType!=QNN_DATATYPE_INT_32||c.outputTensors[ny].v1.dimensions[3]!=(q?6152u:8201u)))return 1;++newNodes;return QNN_SUCCESS;}return node(g,config);
}
int main(){api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=frontierNode;api.graphFinalize=finish;unsigned count=0;
 for(bool obs:{false,true})for(bool tap:{false,true}){for(unsigned next=0;next<3;++next){qkscalestudy::Forward g("forwardShape"+std::to_string(count++),obs,tap,next);if(!tap)for(auto id:g.heads)if(g.g.tensors[id].v1.type!=QNN_TENSOR_TYPE_NATIVE)return 2;unsigned nr=0;for(auto t:g.g.tensors)nr+=t.v1.dataType==QNN_DATATYPE_INT_32;if(!obs&&nr)return 3;}
  for(unsigned domain:{1u,2u})for(unsigned next:{3u,4u}){OneGraph g("backShape"+std::to_string(count++),tap,false,false,false,true,true,true,obs,true,true,true,domain,true,0,next);if(!tap)for(auto key:{"h0_ds","h0_dqraw","h0_dkraw","h1_dqraw","h1_dkraw"})if(g.g.tensors[g.out.at(key)].v1.type!=QNN_TENSOR_TYPE_NATIVE)return 4;unsigned nr=0;for(auto t:g.g.tensors)nr+=t.v1.dataType==QNN_DATATYPE_INT_32;if(!obs&&nr)return 5;}
 }if(count!=28||newNodes!=24)return 6;std::printf("qk_scale_graph_host=PASS graphs=%u new_nodes=%u native_ownership=1 no_receipt_no_int=1\n",count,newNodes);
}
