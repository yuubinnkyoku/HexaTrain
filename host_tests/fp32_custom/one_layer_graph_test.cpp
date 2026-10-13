#define ONE_GRAPH_HOST_AUDIT
#include "one_layer_probe.cpp"
#include "../../app/src/main/cpp/qnn/qnn_graph_shape_validator.h"
#include <iostream>
namespace shape=phonelm::qnn::shape;
unsigned checked=0;
Qnn_ErrorHandle_t create(Qnn_ContextHandle_t,const char*,const QnnGraph_Config_t**,Qnn_GraphHandle_t* handle){*handle=reinterpret_cast<void*>(1);return QNN_SUCCESS;}
Qnn_ErrorHandle_t tensorCreate(Qnn_GraphHandle_t,Qnn_Tensor_t* t){t->v1.id=++checked;return QNN_SUCCESS;}
Qnn_ErrorHandle_t finish(Qnn_GraphHandle_t,Qnn_ProfileHandle_t,Qnn_SignalHandle_t){return QNN_SUCCESS;}
Qnn_ErrorHandle_t node(Qnn_GraphHandle_t,Qnn_OpConfig_t config){auto& c=config.v1;if(std::strcmp(c.packageName,"HexaFp32")==0)return QNN_SUCCESS;
 shape::Node n;n.name=c.name;std::string type=c.typeName;
 if(type==QNN_OP_MAT_MUL)n.op=shape::Op::MatMul;else if(type==QNN_OP_REDUCE_SUM){n.op=shape::Op::ReduceSum;n.axes={3};n.keepDims=true;}else if(type==QNN_OP_SOFTMAX){n.op=shape::Op::Softmax;n.softmaxAxis=3;}else if(type==QNN_OP_SIGMOID)n.op=shape::Op::ElementWiseUnary;else if(type==QNN_OP_ELEMENT_WISE_GREATER)n.op=shape::Op::ElementWiseComparison;else if(type==QNN_OP_ELEMENT_WISE_SELECT)n.op=shape::Op::ElementWiseSelect;else n.op=shape::Op::ElementWiseBinary;
 auto wrap=[](const Qnn_Tensor_t& t){shape::Tensor x;x.name=t.v1.name;x.shape={t.v1.dimensions,t.v1.dimensions+t.v1.rank};x.dtype=t.v1.dataType==QNN_DATATYPE_BOOL_8?"BOOL_8":"FLOAT_32";return x;};
 for(unsigned i=0;i<c.numOfInputs;++i)n.inputs.push_back(wrap(c.inputTensors[i]));for(unsigned i=0;i<c.numOfOutputs;++i)n.outputs.push_back(wrap(c.outputTensors[i]));
 for(unsigned i=0;i<c.numOfParams;++i){auto& p=c.params[i];if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0))n.transposeInput0=p.scalarParam.bool8Value;if(!std::strcmp(p.name,QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1))n.transposeInput1=p.scalarParam.bool8Value;}
 auto result=shape::validate(n);if(!result.ok){std::cerr<<n.name<<":"<<result.error;return 1;}++checked;return QNN_SUCCESS;
}
int main(int argc,char** argv){api.graphCreate=create;api.tensorCreateGraphTensor=tensorCreate;api.graphAddNode=node;api.graphFinalize=finish;
 for(auto mode:{"standalone","integratedNative","integratedTap","integratedFused","producedTap","producedSegmentTap"}){std::string m=mode;OneGraph g(m,m.find("Tap")!=std::string::npos||m=="integratedFused",m=="integratedFused",m!="standalone"&&m!="producedSegmentTap",m=="producedTap"||m=="producedSegmentTap");}
 if(argc==2){auto fs=one::read(argv[1]);std::vector<one::FV> weights;for(auto& z:fs)weights.push_back(one::f(one::reference(z,true).at("dw")));accumulation(fs,weights);}
 for(bool tap:{false,true}){OneGraph g(tap?"integratedSplitTap":"integratedSplitNative",tap,false,false,false);}
 std::cout<<"one_layer_graph_host_audit=PASS checked="<<checked<<" generalized_shape_validator=PASS\n";
}
