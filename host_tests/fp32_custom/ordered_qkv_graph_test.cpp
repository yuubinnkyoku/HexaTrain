#define main inherited_graph_main
#include "one_layer_graph_test.cpp"
#undef main
#include "ordered_qkv_graph.h"
unsigned qkvNodes=0;
Qnn_ErrorHandle_t qkvNode(Qnn_GraphHandle_t graph,Qnn_OpConfig_t config){
    auto& c=config.v1;std::string type=c.typeName;
    if(type.rfind("ProjectionQKV",0)==0){bool obs=type.find("NoReceipt")==std::string::npos,hvx=type.find("Hvx")!=std::string::npos;
        if(c.numOfInputs!=4||c.numOfOutputs!=(obs?4u:3u))return 1;
        for(unsigned i=0;i<7;++i){auto& t=(i<4?c.inputTensors[i]:c.outputTensors[i-4]).v1;if(t.rank!=4||t.dimensions[0]!=1||t.dimensions[1]!=1||t.dimensions[2]!=(i>0&&i<4?64u:32u)||t.dimensions[3]!=64||t.dataType!=QNN_DATATYPE_FLOAT_32)return 1;}
        if(obs&&(c.outputTensors[3].v1.dataType!=QNN_DATATYPE_INT_32||c.outputTensors[3].v1.dimensions[3]!=(hvx?20504u:20488u)))return 1;++qkvNodes;
    }return node(graph,config);
}
int main(){api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=qkvNode;api.graphFinalize=finish;unsigned count=0;
    for(unsigned stage=0;stage<3;++stage)for(bool hvx:{false,true})for(bool obs:{false,true})for(bool tap:{false,true}){
        orderedqkvstudy::Graph g("qkvShape"+std::to_string(count),stage,hvx,obs,tap);
        if(g.g.nodes!=(stage==0?1u:stage==1?3u:4u)||g.g.inputs.size()!=(stage?7u:4u))return 2;
        unsigned nr=0;for(auto& t:g.g.tensors)nr+=t.v1.dataType==QNN_DATATYPE_INT_32;if(nr!=(obs?(stage==0?1u:stage==1?3u:4u):0u))return 3;
        if(stage)for(auto id:{g.y,g.hat,g.iv})if(g.g.tensors[id].v1.type!=(tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE))return 4;
        if(stage==2)for(auto id:g.q)if(g.g.tensors[id].v1.type!=(tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE))return 5;
        if(g.g.tensors[g.x].v1.type!=QNN_TENSOR_TYPE_APP_WRITE)return 6;for(auto id:g.w)if(g.g.tensors[id].v1.type!=QNN_TENSOR_TYPE_APP_WRITE)return 7;++count;
    }if(count!=24||qkvNodes!=24)return 8;std::printf("ordered_qkv_graph_host=PASS graphs=%u qkv_nodes=%u native_ownership=1 no_receipt_no_int=1\n",count,qkvNodes);
}
