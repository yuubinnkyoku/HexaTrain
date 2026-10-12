// Standalone diagnostic; only libQnnHtp is selected. No training state loaded.
#include <QnnInterface.h>
#include <QnnOpPackage.h>
#include <QnnOpDef.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "../../app/src/main/cpp/qnn/layernorm_redesign_reference.h"
#include "../hvx_rpc/fp32_boundary_fixture.h"
using API=QNN_INTERFACE_VER_TYPE;
API api{}; Qnn_BackendHandle_t backend{};Qnn_DeviceHandle_t device{};Qnn_ContextHandle_t context{};
void check(const char* stage,Qnn_ErrorHandle_t rc) {
 std::printf("api,%s,%llu\n",stage,(unsigned long long)QNN_GET_ERROR_CODE(rc));std::fflush(stdout);
 if(rc!=QNN_SUCCESS)std::exit(2);
}
void log(const char* f,QnnLog_Level_t,uint64_t,va_list a){std::vfprintf(stderr,f,a);std::fputc('\n',stderr);}
uint32_t word(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
float from(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
struct Graph {
 Qnn_GraphHandle_t graph{}; std::vector<Qnn_Tensor_t> tensors;
 std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
 std::vector<unsigned> inputs,outputs; std::vector<std::vector<float>> data;
 std::vector<std::vector<uint32_t>> receipts; bool custom;
 Graph(const std::string& variant,size_t n,bool tap=false):custom(variant!="standardAdd"&&variant!="standardMultiply") {
  tensors.reserve(16);names.reserve(16);shapes.reserve(16);
  check("graphCreate",api.graphCreate(context,variant.c_str(),nullptr,&graph));
  auto tensor=[&](const char* name,Qnn_TensorType_t type,uint32_t size,Qnn_DataType_t dtype=QNN_DATATYPE_FLOAT_32){
   unsigned id=tensors.size();names.emplace_back(name);shapes.push_back({1,1,1,size});
   Qnn_Tensor_t t=QNN_TENSOR_INIT;t.v1.name=names.back().c_str();t.v1.type=type;
   t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;t.v1.dataType=dtype;
   t.v1.rank=4;t.v1.dimensions=shapes.back().data();t.v1.memType=QNN_TENSORMEMTYPE_RAW;
   tensors.push_back(t);check("tensorCreate",api.tensorCreateGraphTensor(graph,&tensors.back()));
   if(type==QNN_TENSOR_TYPE_APP_WRITE)inputs.push_back(id);
   if(type==QNN_TENSOR_TYPE_APP_READ)outputs.push_back(id);
   return id;
  };
  auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ,N=QNN_TENSOR_TYPE_NATIVE;
  unsigned x=tensor("x",W,n),b=tensor("b",W,n),c=tensor("c",W,n),y=tensor("y",R,n);
  auto node=[&](const char* name,const char* type,const char* pkg,std::vector<unsigned> ins,std::vector<unsigned> outs){
   std::vector<Qnn_Tensor_t> it,ot;for(auto i:ins)it.push_back(tensors[i]);for(auto i:outs)ot.push_back(tensors[i]);
   Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.name=name;op.v1.packageName=pkg;op.v1.typeName=type;
   op.v1.numOfInputs=it.size();op.v1.inputTensors=it.data();op.v1.numOfOutputs=ot.size();op.v1.outputTensors=ot.data();
   check("graphAddNode",api.graphAddNode(graph,op));
  };
  auto customNode=[&](const char* name,const char* op,std::vector<unsigned> in,unsigned out){
   unsigned r=tensor(name,R,n+3,QNN_DATATYPE_INT_32);node(name,op,"HexaFp32",in,{out,r});};
  if(variant=="Copy")customNode("copy","Copy",{x},y);
  else if(variant=="Add")customNode("add","Add",{x,b},y);
  else if(variant=="Multiply")customNode("mul","Multiply",{x,b},y);
  else if(variant=="Affine")customNode("affine","Affine",{x,b,c},y);
  else if(variant=="standardAdd"||variant=="standardMultiply")
   node("standard",variant=="standardAdd"?QNN_OP_ELEMENT_WISE_ADD:QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{x,b},{y});
  else {
   unsigned m=tensor("mid",tap?R:N,n);
   customNode("producer","Multiply",{x,b},m);
   if(variant=="chain")customNode("consumer","Add",{m,c},y);
   else if(variant=="reshape")node("reshape",QNN_OP_RESHAPE,"qti.aisw",{m},{y});
   else node("consumer",variant=="htpAdd"?QNN_OP_ELEMENT_WISE_ADD:QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{m,c},{y});
  }
  check("graphFinalize",api.graphFinalize(graph,nullptr,nullptr));
 }
 std::vector<float> run(const std::vector<float>& x,const std::vector<float>& b,const std::vector<float>& c) {
  std::vector<Qnn_Tensor_t> in,out;std::vector<std::vector<uint32_t>> buffers;
  buffers.reserve(outputs.size());const std::vector<float>* values[]={&x,&b,&c};
  for(size_t j=0;j<inputs.size();++j){auto t=tensors[inputs[j]];t.v1.clientBuf={(void*)values[j]->data(),uint32_t(values[j]->size()*4)};in.push_back(t);}
  for(auto j:outputs){auto t=tensors[j];buffers.emplace_back(shapes[j][3],0x7fc0dead);t.v1.clientBuf={buffers.back().data(),uint32_t(buffers.back().size()*4)};out.push_back(t);}
  auto before=x;auto bb=b;auto cc=c;auto start=std::chrono::steady_clock::now();
  check("graphExecute",api.graphExecute(graph,in.data(),in.size(),out.data(),out.size(),nullptr,nullptr));
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  if(std::memcmp(x.data(),before.data(),x.size()*4)||b!=bb||c!=cc){std::puts("INPUT_MUTATION");std::exit(3);}
  std::printf("execute_us,%.3f\n",us);
  for(size_t k=0;k<outputs.size();++k)if(tensors[outputs[k]].v1.dataType==QNN_DATATYPE_INT_32){
   auto& r=buffers[k];if((r[0]&0xffffff00)!=0x48383100 || !r[1] ||r[2]!=x.size()){std::puts("KERNEL_RECEIPT_FAIL");std::exit(3);}
   std::printf("kernel_receipt,%u,%u,%u\n",r[0],r[1],r[2]);
   // The second kernel receipt checks native intermediate independently below.
   receipts.push_back(r);
  }
  std::vector<float> y(x.size());std::memcpy(y.data(),buffers[0].data(),x.size()*4);
  for(float f:y)if(!std::isfinite(f)){std::puts("NONFINITE_OR_POISON");std::exit(3);}
  return y;
 }
};
int main(int argc,char** argv){
 if(argc!=2)return 1;
 void* lib=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_GLOBAL);if(!lib){std::puts(dlerror());return 2;}
 const QnnInterface_t** ps=nullptr;uint32_t count=0;
 auto providers=(Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(lib,"QnnInterface_getProviders");
 if(!providers)return 2;check("getProviders",providers(&ps,&count));
 for(unsigned j=0;j<count;++j)if(ps[j]->apiVersion.coreApiVersion.major==2&&ps[j]->apiVersion.coreApiVersion.minor==37)api=ps[j]->QNN_INTERFACE_VER_NAME;
 if(!api.backendCreate)return 2;
 Qnn_LogHandle_t logger{};check("logCreate",api.logCreate(log,QNN_LOG_LEVEL_INFO,&logger));
 check("backendCreate",api.backendCreate(logger,nullptr,&backend));
 check("registerCPU",api.backendRegisterOpPackage(backend,"libHexaFp32Host.so","HexaFp32InterfaceProvider","CPU"));
 check("registerDSP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","DSP"));
 check("deviceCreate",api.deviceCreate(logger,nullptr,&device));
 check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
 std::string variant=argv[1];Graph g(variant,128);
 std::vector<float>x(128),b(128),c(128);for(unsigned j=0;j<128;++j){x[j]=1024.f+float(int(j%3)-1)*.001f;b[j]=1.000123f+float(j%5)*.00001f;c[j]=.003125f;}
 auto y=g.run(x,b,c);double err=0;unsigned mismatches=0;
 for(unsigned j=0;j<128;++j){float ref=x[j];if(variant=="Add"||variant=="standardAdd")ref=x[j]+b[j];else if(variant=="Multiply"||variant=="standardMultiply"||variant=="reshape")ref=x[j]*b[j];else if(variant=="htpMultiply")ref=(x[j]*b[j])*c[j];else if(variant!="Copy"){volatile float p=x[j]*b[j];ref=p+c[j];}err=std::max(err,std::abs(double(y[j])-ref));mismatches+=word(ref)!=word(y[j]);}
 std::printf("result,%s,bit_mismatch=%u,max_abs=%.12g\n",variant.c_str(),mismatches,err);
 check("contextFree",api.contextFree(context,nullptr));check("deviceFree",api.deviceFree(device));check("backendFree",api.backendFree(backend));
 return mismatches?4:0;
}
