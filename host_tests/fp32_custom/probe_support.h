#pragma once
#include <QnnInterface.h>
#include <QnnOpPackage.h>
#include <QnnOpDef.h>
#ifndef ONE_GRAPH_HOST_AUDIT
#include <dlfcn.h>
#endif
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#ifndef ONE_GRAPH_HOST_AUDIT
#include <sys/resource.h>
#endif
#include <map>
#include "../../app/src/main/cpp/qnn/layernorm_redesign_reference.h"
#include "../hvx_rpc/fp32_boundary_fixture.h"
using API=QNN_INTERFACE_VER_TYPE;
inline API api{};inline Qnn_BackendHandle_t backend{};inline Qnn_DeviceHandle_t device{};inline Qnn_ContextHandle_t context{};
inline void checkImpl(const char* stage,Qnn_ErrorHandle_t rc){std::printf("api,%s,%llu\n",stage,(unsigned long long)QNN_GET_ERROR_CODE(rc));std::fflush(stdout);if(rc!=QNN_SUCCESS)std::exit(2);}
#define check(stage, expression) do {auto apiStarted=std::chrono::steady_clock::now();auto apiRc=(expression);double apiUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-apiStarted).count();std::printf("api_us,%s,%.3f\n",stage,apiUs);checkImpl(stage,apiRc);}while(0)
inline void log(const char* f,QnnLog_Level_t,uint64_t,va_list a){std::vfprintf(stderr,f,a);std::fputc('\n',stderr);}
inline uint32_t word(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
inline float from(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
inline uint64_t hashWords(const std::vector<float>& values,uint64_t hash=14695981039346656037ull){for(float f:values){uint32_t u=word(f);for(unsigned j=0;j<4;++j){hash^=(u>>(8*j))&255;hash*=1099511628211ull;}}return hash;}
#ifndef ONE_GRAPH_HOST_AUDIT
inline void initializeProbe(){
 void* lib=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_GLOBAL);if(!lib){std::puts(dlerror());std::exit(2);}
 const QnnInterface_t** ps=nullptr;uint32_t count=0;
 auto providers=(Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(lib,"QnnInterface_getProviders");if(!providers)std::exit(2);
 check("getProviders",providers(&ps,&count));for(unsigned j=0;j<count;++j)if(ps[j]->apiVersion.coreApiVersion.major==2&&ps[j]->apiVersion.coreApiVersion.minor==37)api=ps[j]->QNN_INTERFACE_VER_NAME;
 if(!api.backendCreate)std::exit(2);Qnn_LogHandle_t logger{};check("logCreate",api.logCreate(log,QNN_LOG_LEVEL_INFO,&logger));
 check("backendCreate",api.backendCreate(logger,nullptr,&backend));
 check("registerCPU",api.backendRegisterOpPackage(backend,"libHexaFp32Host.so","HexaFp32InterfaceProvider","CPU"));
 check("registerHTP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","HTP"));
 check("deviceCreate",api.deviceCreate(logger,nullptr,&device));check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
}
inline void closeProbe(){check("contextFree",api.contextFree(context,nullptr));check("deviceFree",api.deviceFree(device));check("backendFree",api.backendFree(backend));}
#endif
// Shared QNN helper for variable shapes and multiple saved/output tensors.
struct DiagnosticGraph {
 Qnn_GraphHandle_t graph{};std::vector<Qnn_Tensor_t> tensors;std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
 std::vector<unsigned> inputs,outputs;unsigned nodes=0;double executeUs=0;bool receiptEnabled=true;
 std::vector<std::vector<float>> constants;
 DiagnosticGraph(const std::string& name){tensors.reserve(512);names.reserve(512);shapes.reserve(512);constants.reserve(512);check("graphCreate",api.graphCreate(context,name.c_str(),nullptr,&graph));}
 unsigned tensor(const char* name,Qnn_TensorType_t type,std::vector<uint32_t> shape,Qnn_DataType_t dtype=QNN_DATATYPE_FLOAT_32){
  if(!receiptEnabled&&dtype==QNN_DATATYPE_INT_32)return ~0u;
  if(names.size()>=512||std::find(names.begin(),names.end(),name)!=names.end()){std::puts("DIAGNOSTIC_NAME_OR_CAPACITY_FAIL");std::exit(3);}
  unsigned id=tensors.size();names.emplace_back(name);shapes.push_back(shape);Qnn_Tensor_t t=QNN_TENSOR_INIT;
  t.v1.name=names.back().c_str();t.v1.type=type;t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;t.v1.dataType=dtype;t.v1.rank=shape.size();t.v1.dimensions=shapes.back().data();t.v1.memType=QNN_TENSORMEMTYPE_RAW;
  tensors.push_back(t);check("tensorCreate",api.tensorCreateGraphTensor(graph,&tensors.back()));
  if(type==QNN_TENSOR_TYPE_APP_WRITE)inputs.push_back(id);if(type==QNN_TENSOR_TYPE_APP_READ)outputs.push_back(id);return id;
 }
 unsigned constant(const char* name,std::vector<uint32_t> shape,std::vector<float> data,Qnn_DataType_t dtype=QNN_DATATYPE_FLOAT_32){
  if(names.size()>=512||std::find(names.begin(),names.end(),name)!=names.end()){std::puts("DIAGNOSTIC_NAME_OR_CAPACITY_FAIL");std::exit(3);}
  unsigned id=tensors.size();names.emplace_back(name);shapes.push_back(shape);constants.push_back(std::move(data));Qnn_Tensor_t t=QNN_TENSOR_INIT;
  t.v1.name=names.back().c_str();t.v1.type=QNN_TENSOR_TYPE_STATIC;t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;t.v1.dataType=dtype;t.v1.rank=shape.size();t.v1.dimensions=shapes.back().data();t.v1.memType=QNN_TENSORMEMTYPE_RAW;t.v1.clientBuf={constants.back().data(),uint32_t(constants.back().size()*4)};
  tensors.push_back(t);check("tensorCreate",api.tensorCreateGraphTensor(graph,&tensors.back()));return id;
 }
 unsigned size(unsigned id)const{unsigned n=1;for(auto d:shapes[id])n*=d;return n;}
 void node(const char* name,const char* type,const char* package,std::vector<unsigned> ins,std::vector<unsigned> outs,std::vector<Qnn_Param_t> params={}){
  std::string freeType;if(!receiptEnabled&&!std::strcmp(package,"HexaFp32")){freeType=std::string(type)+"NoReceipt";type=freeType.c_str();outs.erase(std::remove(outs.begin(),outs.end(),~0u),outs.end());}
  std::vector<Qnn_Tensor_t> i,o;for(auto t:ins)i.push_back(tensors[t]);for(auto t:outs)o.push_back(tensors[t]);Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
  op.v1.name=name;op.v1.typeName=type;op.v1.packageName=package;op.v1.numOfInputs=i.size();op.v1.inputTensors=i.data();op.v1.numOfOutputs=o.size();op.v1.outputTensors=o.data();op.v1.numOfParams=params.size();op.v1.params=params.data();
  check("graphAddNode",api.graphAddNode(graph,op));++nodes;
 }
 void finalize(){check("graphFinalize",api.graphFinalize(graph,nullptr,nullptr));std::printf("graph,tensors=%zu,nodes=%u\n",tensors.size(),nodes);}
 std::map<unsigned,std::vector<uint32_t>> run(const std::map<unsigned,std::vector<float>>& values){
  auto allocationStart=std::chrono::steady_clock::now();auto before=values;std::vector<Qnn_Tensor_t> i,o;std::map<unsigned,std::vector<uint32_t>> buffers;
  for(auto id:inputs){if(values.at(id).size()!=size(id))std::exit(3);auto t=tensors[id];t.v1.clientBuf={(void*)values.at(id).data(),uint32_t(size(id)*4)};i.push_back(t);}
  for(auto id:outputs){auto& b=buffers[id];b.assign(size(id)+8,0xa55aa55a);for(unsigned j=4;j<size(id)+4;++j)b[j]=0x7fc0dead;auto t=tensors[id];t.v1.clientBuf={b.data()+4,uint32_t(size(id)*4)};o.push_back(t);}
  auto started=std::chrono::steady_clock::now();std::printf("allocation_copy_us,%.3f\n",std::chrono::duration<double,std::micro>(started-allocationStart).count());
  check("graphExecute",api.graphExecute(graph,i.data(),i.size(),o.data(),o.size(),nullptr,nullptr));executeUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
  for(auto id:inputs)if(std::memcmp(values.at(id).data(),before.at(id).data(),size(id)*4)){std::puts("INPUT_MUTATION");std::exit(3);}
  for(auto id:outputs){auto& b=buffers[id];for(unsigned j=0;j<4;++j)if(b[j]!=0xa55aa55a||b[size(id)+4+j]!=0xa55aa55a){std::puts("CANARY_FAIL");std::exit(3);}
   std::vector<uint32_t> actual(b.begin()+4,b.end()-4);b=actual;
   for(auto w:b){if(w==0x7fc0dead){std::puts("POISON_FAIL");std::exit(3);}if(tensors[id].v1.dataType==QNN_DATATYPE_FLOAT_32&&!std::isfinite(from(w))){std::puts("NONFINITE");std::exit(3);}}
  }return buffers;
 }
};
inline std::vector<float> decode(const std::vector<uint32_t>& bits){std::vector<float> v;for(auto b:bits)v.push_back(from(b));return v;}
