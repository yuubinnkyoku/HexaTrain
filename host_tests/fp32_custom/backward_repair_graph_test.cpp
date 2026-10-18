#define main unusedOneGraphMain
#include "one_layer_graph_test.cpp"
#undef main
#include "backward_repair_graph.h"
struct SavedEdge {std::string name,type,package;std::vector<unsigned> in,out;};
std::vector<SavedEdge> savedEdges;
Qnn_ErrorHandle_t savedNode(Qnn_GraphHandle_t graph,Qnn_OpConfig_t config){auto& c=config.v1;SavedEdge e{c.name,c.typeName,c.packageName,{},{}};
 for(unsigned i=0;i<c.numOfInputs;++i)e.in.push_back(c.inputTensors[i].v1.id);
 for(unsigned i=0;i<c.numOfOutputs;++i)e.out.push_back(c.outputTensors[i].v1.id);
 savedEdges.push_back(e);return e.package=="qti.aisw"?node(graph,config):QNN_SUCCESS;
}
std::vector<std::string> topology(DiagnosticGraph& g,const std::vector<SavedEdge>& edges,unsigned count){
 auto name=[&](unsigned id){for(unsigned i=0;i<g.tensors.size();++i)if(g.tensors[i].v1.id==id)return g.names[i];throw std::runtime_error("unknown QNN id");};
 std::vector<std::string> out;for(unsigned i=0;i<count;++i){auto& e=edges.at(i);std::string s=e.name+":"+e.package+":"+e.type;for(auto id:e.in)s+="|in="+name(id);for(auto id:e.out)s+="|out="+name(id);out.push_back(s);}return out;
}
int main(){try{api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=savedNode;api.graphFinalize=finish;unsigned graphs=0;
 for(bool o:{true,false})for(bool t:{true,false}){
  savedEdges.clear();savedp::Graph old(savedp::Mode::Forward,o,t);auto upstream=topology(old.g,savedEdges,8);
  for(auto mode:{savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue,savedp::Mode::Forward,savedp::Mode::NativeSoftmax,savedp::Mode::NativeDValue,savedp::Mode::Dual}){
   savedEdges.clear();repairprobe::Graph z(mode,o,t);++graphs;
   if(z.native&&topology(z.g,savedEdges,8)!=upstream)return 1;
   for(unsigned h=0;h<2;++h){auto pid=z.g.tensors[z.p[h]].v1.id;unsigned producers=0,consumers=0;for(auto& e:savedEdges){producers+=std::count(e.out.begin(),e.out.end(),pid);if(e.package=="HexaBackwardRepair"){if(e.in.size()!=2||e.in[0]==e.in[1])return 2;if(e.in[0]==pid)++consumers;}}
    if(z.native&&(producers!=1||std::find(z.g.inputs.begin(),z.g.inputs.end(),z.p[h])!=z.g.inputs.end()))return 3;
    if(consumers!=unsigned(z.softmax)+unsigned(z.dvalue))return 4;
    if(z.native&&!t&&mode!=savedp::Mode::Forward&&z.g.tensors[z.p[h]].v1.type!=QNN_TENSOR_TYPE_NATIVE)return 5;
    if(z.p[h]==z.p[1-h])return 6;
    if(z.softmax&&o&&z.g.size(z.sr[h])!=5184)return 7;if(z.dvalue&&o&&z.g.size(z.vr[h])!=3104)return 8;
   }
   for(auto& x:z.g.tensors)if(!o&&x.v1.dataType==QNN_DATATYPE_INT_32)return 9;
   if(z.g.nodes!=(z.native?8u:0u)+2u*(unsigned(z.softmax)+unsigned(z.dvalue)))return 10;
  }
 }
 std::printf("backward_repair_graph=PASS graphs=%u upstream_topology_identical=1 same_P_ID=1 one_producer=1 heads_separate=1 no_native_reinjection=1 no_receipt=1 HOST_CONSTRUCTION_ONLY=1\n",graphs);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
