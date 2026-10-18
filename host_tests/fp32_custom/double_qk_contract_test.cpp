#include <QnnOpPackage.h>
#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "one_layer_reference.h"
#include "double_qk_fixture.h"
struct QHPI_Tensor{QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
unsigned registered=0;QHPI_OpInfo_v1* registry=nullptr;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1* p,const char* name){if(std::strcmp(name,"HexaDoubleQk"))return QHPI_ERROR_FATAL;registered=n;registry=p;return QHPI_SUCCESS;}
#include "double_qk_package.cpp"
void demand(bool ok){if(!ok)throw std::runtime_error("double QK contract");}
QHPI_Tensor tensor(unsigned r,unsigned c,void* p,unsigned dt=QHPI_FLOAT32){return {{4,{1,1,r,c}},dt,p};}
int main(int argc,char** argv){try{QnnOpPackage_Interface_t api{};demand(HexaDoubleQkInterfaceProvider(&api)==QNN_SUCCESS);demand(api.v1_4.init(nullptr)==QNN_SUCCESS);const QnnOpPackage_Info_t* info=nullptr;demand(api.v1_4.getInfo(&info)==QNN_SUCCESS&&info->numOperations==2&&qhpi_init()&&registered==2);for(unsigned i=0;i<2;++i){Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.packageName="HexaDoubleQk";op.v1.typeName=doubleqkpackage::names[i];op.v1.numOfInputs=4;op.v1.numOfOutputs=3-i;demand(api.v1_4.validateOpConfig(op)==QNN_SUCCESS);++op.v1.numOfOutputs;demand(api.v1_4.validateOpConfig(op)!=QNN_SUCCESS);}
#ifndef DOUBLE_QK_HOST_CONTRACT_TEST
 for(unsigned i=0;i<2;++i)demand(registry[i].kernels[0].function(nullptr,0,nullptr,0,nullptr)==QHPI_ERROR_FATAL);
 std::puts("double_qk_prepare=PASS cpu_arithmetic_rejected=2 invalid_signature_rejected=2");
#else
 demand(argc==2);auto cs=doubleqkfixture::load(argv[1]);unsigned checks=0;
 for(auto& c:cs)for(bool obs:{true,false}){std::array<std::vector<float>,4> before={c.heads[0],c.heads[1],c.heads[2],c.heads[3]};std::vector<uint32_t> buffers[3];QHPI_Tensor it[4],ot[3];const QHPI_Tensor* in[4];QHPI_Tensor* out[3];for(unsigned j=0;j<4;++j){it[j]=tensor(32,32,c.heads[j].data());in[j]=it+j;}for(unsigned j=0;j<3;++j){unsigned n=j==2?6176:1024;buffers[j].assign(n+8,0xa55aa55a);std::fill(buffers[j].begin()+4,buffers[j].end()-4,0x7fc0dead);ot[j]=tensor(j==2?1:32,j==2?6176:32,buffers[j].data()+4,j==2?QHPI_INT32:QHPI_FLOAT32);out[j]=ot+j;}auto fn=obs?doubleqkpackage::kernel<true>:doubleqkpackage::kernel<false>;demand(fn(nullptr,2+obs,out,4,in)==QHPI_SUCCESS);for(unsigned j=0;j<2;++j)demand(!std::memcmp(c.expected[1][j].data(),buffers[j].data()+4,4096));for(unsigned j=0;j<2+obs;++j){demand(std::find(buffers[j].begin()+4,buffers[j].end()-4,0x7fc0dead)==buffers[j].end()-4);for(unsigned k=0;k<4;++k)demand(buffers[j][k]==0xa55aa55a&&buffers[j][buffers[j].size()-1-k]==0xa55aa55a);}for(unsigned j=0;j<4;++j)demand(!std::memcmp(c.heads[j].data(),before[j].data(),4096));
  if(obs){auto r=buffers[2].data()+4;demand(r[0]==0x43503401u&&r[2]==4096&&r[3]==2048&&r[24]==1);for(unsigned j=0;j<4;++j)demand(!std::memcmp(r+32+j*1024,c.heads[j].data(),4096));for(unsigned j=0;j<2;++j)demand(!std::memcmp(r+4128+j*1024,buffers[j].data()+4,4096));}else demand(std::all_of(buffers[2].begin()+4,buffers[2].end()-4,[](unsigned x){return x==0x7fc0dead;}));
  it[0].shape.dims[3]=31;demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);it[0].shape.dims[3]=32;void* old=ot[0].data;ot[0].data=c.heads[1].data();demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);ot[0].data=ot[1].data;demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);ot[0].data=static_cast<char*>(old)+1;demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);ot[0].data=old;if(obs){ot[2].data=old;demand(fn(nullptr,3,out,4,in)==QHPI_ERROR_FATAL);ot[2].data=buffers[2].data()+4;}
  float x=c.heads[0][0];c.heads[0][0]=NAN;demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);c.heads[0][0]=INFINITY;demand(fn(nullptr,2+obs,out,4,in)==QHPI_ERROR_FATAL);c.heads[0][0]=x;++checks;
 }std::printf("double_qk_kernel_host=PASS calls=%u words=1 guards=1 complete_write=1 immutable=1 invalid_shape=1 invalid_overlap=1 alignment=1 nonfinite_rejection=1 HOST_CONTRACT_ONLY=1\n",checks);
#endif
 demand(api.v1_4.terminate()==QNN_SUCCESS);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
