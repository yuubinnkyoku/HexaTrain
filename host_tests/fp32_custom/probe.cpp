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
#include <sys/resource.h>
#include "../../app/src/main/cpp/qnn/layernorm_redesign_reference.h"
#include "../hvx_rpc/fp32_boundary_fixture.h"
using API=QNN_INTERFACE_VER_TYPE;
API api{}; Qnn_BackendHandle_t backend{};Qnn_DeviceHandle_t device{};Qnn_ContextHandle_t context{};
void checkImpl(const char* stage,Qnn_ErrorHandle_t rc) {
 std::printf("api,%s,%llu\n",stage,(unsigned long long)QNN_GET_ERROR_CODE(rc));std::fflush(stdout);
 if(rc!=QNN_SUCCESS)std::exit(2);
}
#define check(stage, expression) do {auto apiStarted=std::chrono::steady_clock::now();auto apiRc=(expression);double apiUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-apiStarted).count();std::printf("api_us,%s,%.3f\n",stage,apiUs);checkImpl(stage,apiRc);}while(0)
void log(const char* f,QnnLog_Level_t,uint64_t,va_list a){std::vfprintf(stderr,f,a);std::fputc('\n',stderr);}
uint32_t word(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
float from(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
uint64_t hashWords(const std::vector<float>& values,uint64_t hash=14695981039346656037ull){
 for(float f:values){uint32_t u=word(f);for(unsigned j=0;j<4;++j){hash^=(u>>(8*j))&255;hash*=1099511628211ull;}}return hash;
}
struct Graph {
 Qnn_GraphHandle_t graph{}; std::vector<Qnn_Tensor_t> tensors;
 std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
 std::vector<unsigned> inputs,outputs; std::vector<std::vector<float>> data;
 std::vector<std::vector<uint32_t>> receipts; bool custom;std::string variant;double executeUs=0;unsigned lastCounter=0;
 Graph(const std::string& variant,size_t n,bool tap=false):custom(variant!="standardAdd"&&variant!="standardMultiply"),variant(variant) {
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
   unsigned r=tensor(name,R,3*n+8,QNN_DATATYPE_INT_32);node(name,op,"HexaFp32",in,{out,r});};
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
  std::printf("graph,%s,tensors=%zu,nodes=%d,custom_scratch=0,custom_vtcm=0\n",variant.c_str(),tensors.size(),variant=="Copy"||variant=="Add"||variant=="Multiply"||variant=="Affine"||!custom?1:2);
 }
 std::vector<float> run(const std::vector<float>& x,const std::vector<float>& b,const std::vector<float>& c) {
  auto allocationStarted=std::chrono::steady_clock::now();receipts.clear();std::vector<Qnn_Tensor_t> in,out;std::vector<std::vector<uint32_t>> buffers;
  buffers.reserve(outputs.size());const std::vector<float>* values[]={&x,&b,&c};
  for(size_t j=0;j<inputs.size();++j){auto t=tensors[inputs[j]];t.v1.clientBuf={(void*)values[j]->data(),uint32_t(values[j]->size()*4)};in.push_back(t);}
  for(auto j:outputs){auto t=tensors[j];buffers.emplace_back(shapes[j][3],0x7fc0dead);t.v1.clientBuf={buffers.back().data(),uint32_t(buffers.back().size()*4)};out.push_back(t);}
  auto before=x;auto bb=b;auto cc=c;auto start=std::chrono::steady_clock::now();
  std::printf("host_allocate_poison_copy_us,%.3f\n",std::chrono::duration<double,std::micro>(start-allocationStarted).count());
  check("graphExecute",api.graphExecute(graph,in.data(),in.size(),out.data(),out.size(),nullptr,nullptr));
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();executeUs=us;
  if(std::memcmp(x.data(),before.data(),x.size()*4)||std::memcmp(b.data(),bb.data(),b.size()*4)||std::memcmp(c.data(),cc.data(),c.size()*4)){std::puts("INPUT_MUTATION");std::exit(3);}
  std::printf("execute_us,%.3f\n",us);
  for(size_t k=0;k<outputs.size();++k)if(tensors[outputs[k]].v1.dataType==QNN_DATATYPE_INT_32){
   auto& r=buffers[k];if((r[0]&0xffffff00)!=0x48383100 || !r[1] ||r[2]!=x.size()){std::puts("KERNEL_RECEIPT_FAIL");std::exit(3);}
   if(r[1]<=lastCounter){std::puts("KERNEL_COUNTER_FAIL");std::exit(3);}lastCounter=r[1];
   std::printf("kernel_receipt,%u,%u,%u,kernel_us=%u\n",r[0],r[1],r[2],r[4]);
   if(r[5]!=81||r[6]!=32||r[7]!=0){std::puts("KERNEL_IDENTITY_FAIL");std::exit(3);}
   for(unsigned a=0;a<3;++a)for(size_t j=0;j<x.size();++j){
    uint32_t expected=0;
    if(a<r[3]){
     float f=(*values[a])[j];
     if(!receipts.empty() && variant=="chain"){if(a==0)f=x[j]*b[j];else f=c[j];}
     expected=word(f);
    }
    if(r[8+a*x.size()+j]!=expected){std::printf("OPERAND_RECEIPT_FAIL,%u,%zu\n",a,j);std::exit(3);}
   }
   receipts.push_back(r);
  }
  std::vector<float> y(x.size());std::memcpy(y.data(),buffers[0].data(),x.size()*4);
  for(size_t k=0;k<outputs.size();++k)if(tensors[outputs[k]].v1.dataType==QNN_DATATYPE_FLOAT_32){
   for(size_t j=0;j<buffers[k].size();++j)if(!std::isfinite(from(buffers[k][j]))){std::puts("TAP_NONFINITE_OR_POISON");std::exit(3);}
  }
  for(float f:y)if(!std::isfinite(f)){std::puts("NONFINITE_OR_POISON");std::exit(3);}
  std::printf("io_hash,%016llx,%016llx\n",(unsigned long long)hashWords(c,hashWords(b,hashWords(x))),(unsigned long long)hashWords(y));
  std::printf("host_receipt_finite_validation_us,%.3f\n",std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()-us);
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
 check("registerHTP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","HTP"));
 check("deviceCreate",api.deviceCreate(logger,nullptr,&device));
 check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
 std::string variant=argv[1];bool tap=false;if(variant=="chainTap"){variant="chain";tap=true;}Graph g(variant,128,tap);
 std::vector<float>x(128),b(128),c(128);for(unsigned j=0;j<128;++j){x[j]=1024.f+float(int(j%3)-1)*.001f;b[j]=1.000123f+float(j%5)*.00001f;c[j]=.003125f;}
 auto y=g.run(x,b,c);double err=0;unsigned mismatches=0;
 for(unsigned j=0;j<128;++j){float ref=x[j];if(variant=="Add"||variant=="standardAdd")ref=x[j]+b[j];else if(variant=="Multiply"||variant=="standardMultiply"||variant=="reshape")ref=x[j]*b[j];else if(variant=="htpMultiply")ref=(x[j]*b[j])*c[j];else if(variant!="Copy"){volatile float p=x[j]*b[j];ref=p+c[j];}err=std::max(err,std::abs(double(y[j])-ref));mismatches+=word(ref)!=word(y[j]);}
 std::printf("result,%s,bit_mismatch=%u,max_abs=%.12g\n",variant.c_str(),mismatches,err);
 unsigned failures=0;
 auto evaluate=[&](const char* recipe,const char* field,const std::vector<float>& input,const std::vector<double>& ideal,
                   const std::vector<float>& coeff,const std::vector<float>& addend,const std::vector<double>* exactOutput=nullptr){
  auto out=g.run(input,coeff,addend);double fpError=0,doubleError=0;unsigned bitError=0;
  for(size_t j=0;j<input.size();++j){
   float fp=input[j];double d=ideal[j];
   if(variant=="Add"||variant=="standardAdd"){fp=input[j]+coeff[j];d+=coeff[j];}
   else if(variant=="Multiply"||variant=="standardMultiply"||variant=="reshape"){fp=input[j]*coeff[j];d*=coeff[j];}
   else if(variant=="htpMultiply"){volatile float p=input[j]*coeff[j];fp=p*addend[j];d=d*coeff[j]*addend[j];}
   else if(variant!="Copy"){volatile float p=input[j]*coeff[j];fp=p+addend[j];d=d*coeff[j]+addend[j];}
   if(exactOutput)d=(*exactOutput)[j];
   fpError=std::max(fpError,std::abs(double(out[j])-fp));doubleError=std::max(doubleError,std::abs(double(out[j])-d));bitError+=word(out[j])!=word(fp);
  }
  bool pass=doubleError<=.02 && (!g.custom||bitError==0);
  failures+=!pass;std::printf("metric,%s,%s,%s,fp_abs=%.12g,double_abs=%.12g,bits=%u,gate=%d\n",variant.c_str(),recipe,field,fpError,doubleError,bitError,pass);
 };
 // Copy's integer fixture includes signed zero, off-FP16 values, subnormals,
 // full finite range and seed97 raw words. It proves storage, not arithmetic.
 if(variant=="Copy"){
  for(size_t j=0;j<128;++j)x[j]=from(hexatrain_boundary_bits(j));
  auto raw=g.run(x,b,c);unsigned bad=0;for(size_t j=0;j<128;++j)bad+=word(x[j])!=word(raw[j]);
  std::printf("raw_copy,bit_mismatch=%u\n",bad);failures+=bad!=0;
 }else if(variant=="Add"||variant=="Multiply"){
  for(size_t j=0;j<128;++j){x[j]=from(hexatrain_boundary_bits(j));b[j]=variant=="Add"?from(1):.5f;}
  auto raw=g.run(x,b,c);unsigned bad=0;
  for(size_t j=0;j<128;++j){float f=variant=="Add"?x[j]+b[j]:x[j]*b[j];bad+=word(f)!=word(raw[j]);}
  std::printf("raw_arithmetic,%s,bit_mismatch=%u\n",variant.c_str(),bad);failures+=bad!=0;
 }
 namespace ln=phonelm::qnn::lnvalidation;
 for(const auto& fixture:ln::redesignCases(64)) {
  auto host=ln::anchoredFp32Oracle(fixture,64);auto ref=ln::doubleReference(fixture,64);
  const std::vector<float>* hf[]={&host.dx,&host.dg,&host.db};
  const std::vector<double>* df[]={&ref.dx,&ref.dg,&ref.db};const char* fields[]={"dx","dg","db"};
  for(unsigned k=0;k<3;++k){std::vector<float> xx(128);std::vector<double> ideal(128);
   for(unsigned j=0;j<128;++j){xx[j]=(*hf[k])[j%hf[k]->size()];ideal[j]=(*df[k])[j%df[k]->size()];b[j]=1.000123f+float(j%5)*.00001f;c[j]=fixture.dy[j];}
   evaluate(fixture.name.c_str(),fields[k],xx,ideal,b,c);
  }
  // Real LN backward tail: bracket * (inverse / D), then residual Add.
  // Inputs are independently derived double stages rounded to FP32. No LN
  // kernel is claimed here. Consumer double oracle is ideal dX + residual.
  if(variant=="chain"||variant=="Affine"){
   auto stage=ln::stages(fixture,64);std::vector<float> bracket(128);std::vector<double> ideal(128),exact(128);
   for(unsigned j=0;j<128;++j){bracket[j]=float(stage[19][j]);ideal[j]=stage[19][j];b[j]=float(stage[20][j/64]);c[j]=fixture.dy[j];exact[j]=ref.dx[j]+double(c[j]);}
   evaluate(fixture.name.c_str(),"ln_tail_residual",bracket,ideal,b,c,&exact);
  }
 }
 std::printf("summary,%s,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0\n",variant.c_str(),failures);
 if(!failures && g.custom){
  auto fixture=ln::redesignCases(64)[1];auto host=ln::anchoredFp32Oracle(fixture,64);
  x=host.dx;for(unsigned j=0;j<128;++j){b[j]=1.000123f;c[j]=fixture.dy[j];}
  for(unsigned j=0;j<33;++j){auto output=g.run(x,b,c);std::printf("perf,%s,%u,execute_us=%.3f\n",variant.c_str(),j,g.executeUs);}
 }
 struct rusage usage{};getrusage(RUSAGE_SELF,&usage);std::printf("peak_host_rss_kib,%ld\n",usage.ru_maxrss);
 check("contextFree",api.contextFree(context,nullptr));check("deviceFree",api.deviceFree(device));check("backendFree",api.backendFree(backend));
 return failures || (g.custom&&mismatches)?4:0;
}
