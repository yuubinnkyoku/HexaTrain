#define ONE_GRAPH_HOST_AUDIT
#include "one_layer_probe.cpp"
#include "upstream_reference.h"
#include "ln1_frontier_probe.h"
#include "attention_ln1_probe.h"
#include "../../app/src/main/cpp/qnn/qnn_graph_shape_validator.h"
#include <iostream>
namespace shape=phonelm::qnn::shape;
unsigned checked=0;
Qnn_ErrorHandle_t create(Qnn_ContextHandle_t,const char*,const QnnGraph_Config_t**,Qnn_GraphHandle_t* handle){*handle=reinterpret_cast<void*>(1);return QNN_SUCCESS;}
Qnn_ErrorHandle_t tensorCreate(Qnn_GraphHandle_t,Qnn_Tensor_t* t){t->v1.id=++checked;return QNN_SUCCESS;}
Qnn_ErrorHandle_t finish(Qnn_GraphHandle_t,Qnn_ProfileHandle_t,Qnn_SignalHandle_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t node(Qnn_GraphHandle_t,Qnn_OpConfig_t config){auto& c=config.v1;if(std::strcmp(c.packageName,"HexaFp32")==0){
 if(!std::strcmp(c.typeName,"BroadcastMultiply")){if(c.numOfInputs!=4||c.numOfOutputs!=3)return 1;for(unsigned i=0;i<7;++i){auto& t=i<4?c.inputTensors[i]:c.outputTensors[i-4];unsigned rows=(i==6?1:32),cols=(i==6?4168:i==2||i==3?1:32);if(t.v1.rank!=4||t.v1.dimensions[0]!=1||t.v1.dimensions[1]!=1||t.v1.dimensions[2]!=rows||t.v1.dimensions[3]!=cols||t.v1.dataType!=(i==6?QNN_DATATYPE_INT_32:QNN_DATATYPE_FLOAT_32))return 1;}++checked;}
 if(std::string(c.typeName).rfind("MatMulDProbabilities",0)==0){bool free=std::string(c.typeName).find("NoReceipt")!=std::string::npos;if(c.numOfInputs!=4||c.numOfOutputs!=(free?2u:3u))return 1;for(unsigned i=0;i<6;++i){auto& t=i<4?c.inputTensors[i]:c.outputTensors[i-4];if(t.v1.rank!=4||t.v1.dimensions[0]!=1||t.v1.dimensions[1]!=1||t.v1.dimensions[2]!=32||t.v1.dimensions[3]!=32||t.v1.dataType!=QNN_DATATYPE_FLOAT_32)return 1;}if(!free&&c.outputTensors[2].v1.dimensions[3]!=6152)return 1;++checked;}
 return QNN_SUCCESS;}
 shape::Node n;n.name=c.name;std::string type=c.typeName;
 if(type==QNN_OP_MAT_MUL)n.op=shape::Op::MatMul;else if(type==QNN_OP_REDUCE_SUM||type==QNN_OP_REDUCE_MEAN){n.op=type==QNN_OP_REDUCE_SUM?shape::Op::ReduceSum:shape::Op::ReduceMean;n.axes={3};n.keepDims=true;}else if(type==QNN_OP_SOFTMAX){n.op=shape::Op::Softmax;n.softmaxAxis=3;}else if(type==QNN_OP_SIGMOID||type==QNN_OP_ELEMENT_WISE_RSQRT)n.op=shape::Op::ElementWiseUnary;else if(type==QNN_OP_ELEMENT_WISE_GREATER)n.op=shape::Op::ElementWiseComparison;else if(type==QNN_OP_ELEMENT_WISE_SELECT)n.op=shape::Op::ElementWiseSelect;else n.op=shape::Op::ElementWiseBinary;
 auto wrap=[](const Qnn_Tensor_t& t){shape::Tensor x;x.name=t.v1.name;x.shape={t.v1.dimensions,t.v1.dimensions+t.v1.rank};x.dtype=t.v1.dataType==QNN_DATATYPE_BOOL_8?"BOOL_8":"FLOAT_32";return x;};
 for(unsigned i=0;i<c.numOfInputs;++i)n.inputs.push_back(wrap(c.inputTensors[i]));for(unsigned i=0;i<c.numOfOutputs;++i)n.outputs.push_back(wrap(c.outputTensors[i]));
 for(unsigned i=0;i<c.numOfParams;++i){auto& p=c.params[i];if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))n.transposeInput0=p.scalarParam.bool8Value;if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))n.transposeInput1=p.scalarParam.bool8Value;}
 auto result=shape::validate(n);if(!result.ok){std::cerr<<n.name<<":"<<result.error;return 1;}++checked;return QNN_SUCCESS;
}
int main(int argc,char** argv){api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=node;api.graphFinalize=finish;
 for(auto mode:{"standalone","integratedNative","integratedTap","integratedFused","producedTap","producedSegmentTap"}){std::string m=mode;OneGraph g(m,m.find("Tap")!=std::string::npos||m=="integratedFused",m=="integratedFused",m!="standalone"&&m!="producedSegmentTap",m=="producedTap"||m=="producedSegmentTap");}
 if(argc==2){auto fs=one::read(argv[1]);std::vector<one::FV> weights;for(auto& z:fs)weights.push_back(one::f(one::reference(z,true).at("dw")));accumulation(fs,weights);}
 for(bool tap:{false,true}){OneGraph g(tap?"integratedSplitTap":"integratedSplitNative",tap,false,false,false);}
 for(bool next:{false,true})for(bool tap:{false,true}){OneGraph g("broadcastGraphAudit",tap,false,false,false,true,next);}
 for(bool tap:{false,true}){OneGraph g(tap?"upstreamLn1Tap":"upstreamLn1Native",tap,false,false,true);}
 for(bool receipt:{false,true})for(bool tap:{false,true})for(bool consumer:{false,true}){OneGraph g("dpGraphAudit",tap,false,false,false,true,true,true,receipt,consumer);}
 for(unsigned v=0;v<3;++v)for(bool receipt:{false,true})for(bool tap:{false,true}){LnStudy ln(v,tap,receipt,false,0);}
 for(unsigned k:{1024u,128u}){LnStudy ln(2,false,true,true,k);}
 std::cout<<"one_layer_graph_host_audit=PASS checked="<<checked<<" generalized_shape_validator=PASS\n";
}
