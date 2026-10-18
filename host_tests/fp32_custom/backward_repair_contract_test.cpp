#include <QnnOpPackage.h>
#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "double_qk_fixture.h"
#include "saved_p_backward_reference_lib.h"
struct QHPI_Tensor{QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
unsigned registered=0;QHPI_OpInfo_v1* registry=nullptr;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1* p,const char* name){if(std::strcmp(name,"HexaBackwardRepair"))return QHPI_ERROR_FATAL;registered=n;registry=p;return QHPI_SUCCESS;}
#include "backward_repair_package.cpp"
void demand(bool v){if(!v)throw std::runtime_error("backward repair Host contract");}
QHPI_Tensor tensor(unsigned rank,unsigned r,unsigned c,void* p,unsigned dt=QHPI_FLOAT32){QHPI_Shape s{};s.rank=rank;if(rank==4){s.dims[0]=s.dims[1]=1;s.dims[2]=r;s.dims[3]=c;}else{s.dims[0]=r;s.dims[1]=c;}return {s,dt,p};}
int main(int argc,char** argv){try{
 QnnOpPackage_Interface_t api{};demand(HexaBackwardRepairInterfaceProvider(&api)==QNN_SUCCESS);demand(api.v1_4.init(nullptr)==QNN_SUCCESS);const QnnOpPackage_Info_t* info=nullptr;demand(api.v1_4.getInfo(&info)==QNN_SUCCESS&&info->numOperations==6&&qhpi_init()&&registered==6);
 for(unsigned i=0;i<6;++i){Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.packageName="HexaBackwardRepair";op.v1.typeName=repairpackage::names[i];op.v1.numOfInputs=2;op.v1.numOfOutputs=repairpackage::outputCount[i];demand(api.v1_4.validateOpConfig(op)==QNN_SUCCESS);++op.v1.numOfOutputs;demand(api.v1_4.validateOpConfig(op)!=QNN_SUCCESS);}
#ifndef BACKWARD_REPAIR_HOST_CONTRACT
 for(unsigned i=0;i<6;++i)demand(registry[i].kernels[0].function(nullptr,0,nullptr,0,nullptr)==QHPI_ERROR_FATAL);
 std::puts("backward_repair_prepare=PASS arithmetic_rejected=6");
#else
 demand(argc==3);auto cs=doubleqkfixture::load(argv[1]);FILE* f=std::fopen(argv[2],"rb");demand(f);unsigned hdr[4];s2fixture::read(f,hdr,4);demand(hdr[0]==0x53425001&&hdr[1]==1&&hdr[2]==25&&hdr[3]==8);
 using Grad=std::array<std::array<std::vector<float>,2>,2>;std::vector<Grad> gradients(33);for(auto& g:gradients)for(auto& h:g)for(auto& v:h){v.resize(1024);s2fixture::read(f,v.data(),1024);}demand(std::fgetc(f)==EOF);std::fclose(f);
 unsigned calls=0;double maxima[2]={0,0};
 for(unsigned fi=0;fi<60;++fi)for(unsigned gi=0;gi<(fi<44?25u:8u);++gi)for(unsigned h=0;h<2;++h)for(unsigned rank:{2u,4u})for(unsigned op=0;op<6;++op){
  bool value=op>=4,tap=op==2||op==3,observe=!(op&1);unsigned mathOut=tap?4:1,rcount=value?3104:5184;
  auto p=cs[fi].hostP[h],g=gradients[fi<44?gi:25+gi][h][value?1:0];auto bp=p,bg=g;
  std::vector<unsigned> buffers[5];QHPI_Tensor it[2]={tensor(rank,32,32,p.data()),tensor(rank,32,32,g.data())},ot[5];const QHPI_Tensor* in[]={it,it+1};QHPI_Tensor* out[5];
  for(unsigned j=0;j<mathOut+observe;++j){unsigned n=j==mathOut?rcount:j==2?32:1024;buffers[j].assign(n+8,0xa55aa55a);std::fill(buffers[j].begin()+4,buffers[j].end()-4,0x7fc0dead);ot[j]=tensor(rank,j==mathOut?1:32,j==mathOut?n:j==2?1:32,buffers[j].data()+4,j==mathOut?QHPI_INT32:QHPI_FLOAT32);out[j]=ot+j;}
  auto fn=registry[op].kernels[0].function;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_SUCCESS);
  auto ref=value?phonelm::saved_p_backward::DValueOrderedFp32(p,g):phonelm::saved_p_backward::SoftmaxBackwardOrderedFp32(p,g);
  auto exact=value?phonelm::saved_p_backward::DValueDouble(p,g):phonelm::saved_p_backward::SoftmaxBackwardDouble(p,g);
  demand(!std::memcmp(ref.data(),buffers[0].data()+4,4096));for(unsigned i=0;i<1024;++i)maxima[value]=std::max(maxima[value],std::fabs(double(ref[i])-exact[i]));
  for(unsigned j=0;j<mathOut+observe;++j){demand(std::find(buffers[j].begin()+4,buffers[j].end()-4,0x7fc0dead)==buffers[j].end()-4);for(unsigned k=0;k<4;++k)demand(buffers[j][k]==0xa55aa55a&&buffers[j][buffers[j].size()-1-k]==0xa55aa55a);}
  demand(!std::memcmp(p.data(),bp.data(),4096)&&!std::memcmp(g.data(),bg.data(),4096));
  if(observe){auto rec=buffers[mathOut].data()+4;demand(rec[0]==0x43505000u+value);demand(!std::memcmp(rec+32,p.data(),4096)&&!std::memcmp(rec+1056,g.data(),4096)&&!std::memcmp(rec+2080,ref.data(),4096));if(tap)demand(!std::memcmp(rec+3104,buffers[1].data()+4,4096)&&!std::memcmp(rec+4128,buffers[2].data()+4,128)&&!std::memcmp(rec+4160,buffers[3].data()+4,4096));}
  // Bounded negative ABI cases on every specialization/rank (first fixture).
  if(fi==0&&gi==0&&h==0){void* old=ot[0].data;ot[0].data=p.data();demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);ot[0].data=static_cast<char*>(old)+1;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);ot[0].data=old;it[0].shape.dims[rank-1]=31;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);it[0].shape.dims[rank-1]=32;it[0].dtype=QHPI_INT32;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);it[0].dtype=QHPI_FLOAT32;float save=p[0];for(float bad:{NAN,INFINITY}){p[0]=bad;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);}p[0]=save;if(observe){ot[mathOut].data=old;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);}if(tap){ot[1].data=old;demand(fn(nullptr,mathOut+observe,out,2,in)==QHPI_ERROR_FATAL);}}
  ++calls;
 }
 demand(maxima[0]<=.002&&maxima[1]<=.01);std::printf("backward_repair_host=PASS calls=%u softmax_maxAbs=%.17g dvalue_maxAbs=%.17g word_identity=1 rank2_rank4=1 guards=1 immutable=1 complete_writes=1 invalid_abi_rejection=1 HOST_ONLY=1\n",calls,maxima[0],maxima[1]);
#endif
 demand(api.v1_4.terminate()==QNN_SUCCESS);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 4;}}
