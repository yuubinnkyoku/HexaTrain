#define main inherited_graph_main
#include "one_layer_graph_test.cpp"
#undef main
#include "ordered_hvx_graph.h"
unsigned hvxNodes=0;
Qnn_ErrorHandle_t orderedNode(Qnn_GraphHandle_t graph,Qnn_OpConfig_t config){
    auto& c=config.v1;std::string type=c.typeName;
    if(type=="FFN2Hvx"||type=="FFN2HvxNoReceipt"){
        bool obs=type=="FFN2Hvx";if(c.numOfInputs!=2||c.numOfOutputs!=(obs?2u:1u))return 1;
        for(unsigned j=0;j<3;++j){auto& t=(j<2?c.inputTensors[j]:c.outputTensors[0]).v1;
            if(t.rank!=4||t.dimensions[0]!=1||t.dimensions[1]!=1||t.dimensions[2]!=(j==1?128u:32u)||t.dimensions[3]!=(j==0?128u:64u)||t.dataType!=QNN_DATATYPE_FLOAT_32)return 1;}
        if(obs&&(c.outputTensors[1].v1.dataType!=QNN_DATATYPE_INT_32||c.outputTensors[1].v1.dimensions[3]!=14352))return 1;
        ++hvxNodes;
    }
    return node(graph,config);
}
int main(){
    api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=orderedNode;api.graphFinalize=finish;
    unsigned count=0;
    for(unsigned stage=0;stage<3;++stage)for(bool hvx:{false,true})for(bool obs:{false,true})for(bool tap:{false,true}){
        if(!stage&&tap)continue;
        orderedhvxstudy::Graph g("orderedShape"+std::to_string(count),stage,hvx,obs,tap);
        if(g.g.nodes!=(stage==0?1u:stage==1?2u:4u))return 1;
        if(g.g.inputs.size()!=(stage==0?2u:stage==1?3u:6u))return 2;
        unsigned receipts=0;for(auto& t:g.g.tensors)receipts+=t.v1.dataType==QNN_DATATYPE_INT_32;
        if(receipts!=(obs?(stage==0?1u:stage==1?2u:4u):0u))return 3;
        if(stage&&g.g.tensors[g.mid].v1.type!=(tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE))return 4;
        if(stage==2)for(auto id:{g.x,g.hat,g.iv})if(g.g.tensors[id].v1.type!=(tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE))return 5;
        if(g.g.tensors[g.a].v1.type!=QNN_TENSOR_TYPE_APP_WRITE||g.g.tensors[g.b].v1.type!=QNN_TENSOR_TYPE_APP_WRITE)return 6;
        ++count;
    }
    if(count!=20||hvxNodes!=10)return 7;
    std::printf("ordered_hvx_graph_host=PASS graphs=%u hvx_nodes=%u native_ownership=1 no_receipt_no_int=1\n",count,hvxNodes);
}
