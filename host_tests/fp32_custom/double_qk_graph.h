#pragma once
#include "double_qk_initialize.h"
#include "double_qk_fixture.h"
namespace doubleqkprobe {
// Construction only: existing arithmetic is called through its immutable ABI.
struct Graph {
 DiagnosticGraph g;unsigned stage;bool high,observed,tap;unsigned x=~0u,ga,be,dy,y,h,iv,dx,dg,db,lr=~0u,br=~0u,qr=~0u,hr=~0u,rr=~0u,sr=~0u,mr=~0u,pr=~0u;std::array<unsigned,3>w{},q{};std::array<unsigned,6>heads{};std::array<unsigned,4>input{};std::array<unsigned,2>raw{},scaled{},masked{},p{};double finalizeUs=0,firstUs=0;unsigned lastSoftmax=0;
 Graph(unsigned s,bool d,bool o,bool t):g("doubleQk"+std::to_string(s)+std::to_string(d)+std::to_string(o)+std::to_string(t)),stage(s),high(d),observed(o),tap(t){
  g.receiptEnabled=o;auto mid=t?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
  auto mat=[&](std::string n,Qnn_TensorType_t type,unsigned r=32,unsigned c=32){return g.tensor(n.c_str(),type,{1,1,r,c});};
  auto rec=[&](const char* n,unsigned count){return g.tensor(n,QNN_TENSOR_TYPE_APP_READ,{1,1,1,count},QNN_DATATYPE_INT_32);};
  if(!s){for(unsigned j=0;j<4;++j)input[j]=mat("input"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE);}
  else {
   x=mat("x",QNN_TENSOR_TYPE_APP_WRITE,32,64);ga=mat("gamma",QNN_TENSOR_TYPE_APP_WRITE,1,64);be=mat("beta",QNN_TENSOR_TYPE_APP_WRITE,1,64);dy=mat("dy",QNN_TENSOR_TYPE_APP_WRITE,32,64);y=mat("y",mid,32,64);h=mat("h",mid,32,64);iv=mat("iv",mid,32,1);auto eps=g.constant("eps",{1,1,1,1},{1e-5f});lr=rec("ln_receipt",6313);std::vector<unsigned> lo={y,h,iv};if(o)lo.push_back(lr);g.node("ln1",o?"S2":"S2NoReceipt","HexaSelectiveLn1",{x,ga,be,eps},lo);
   dx=mat("dx",QNN_TENSOR_TYPE_APP_READ,32,64);dg=mat("dg",QNN_TENSOR_TYPE_APP_READ,1,64);db=mat("db",QNN_TENSOR_TYPE_APP_READ,1,64);br=rec("back_receipt",6248);g.node("local_backward","LnBackward","HexaFp32",{dy,ga,h,iv},{dx,dg,db,br});
   for(unsigned j=0;j<3;++j){w[j]=mat("w"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE,64,64);q[j]=mat("qkv"+std::to_string(j),mid,32,64);}qr=rec("mixed_receipt",20512);std::vector<unsigned> qo={q[0],q[1],q[2]};if(o)qo.push_back(qr);g.node("mixed",o?"MixedPrecisionQKV":"MixedPrecisionQKVNoReceipt","HexaMixedQkv",{y,w[0],w[1],w[2]},qo);
   for(unsigned j=0;j<6;++j)heads[j]=mat("head"+std::to_string(j),j>=4?QNN_TENSOR_TYPE_APP_READ:mid);hr=rec("head_receipt",12296);g.node("heads","QKVHeads","HexaFp32",{q[0],q[1],q[2]},{heads[0],heads[1],heads[2],heads[3],heads[4],heads[5],hr});for(unsigned j=0;j<4;++j)input[j]=heads[j];
  }
  for(unsigned j=0;j<2;++j)raw[j]=mat("raw"+std::to_string(j),s==2?mid:QNN_TENSOR_TYPE_APP_READ);rr=rec("qk_receipt",d?6176:6152);
  if(d){std::vector<unsigned> ro={raw[0],raw[1]};if(o)ro.push_back(rr);g.node("double_qk",o?"ForwardRawQkDouble":"ForwardRawQkDoubleNoReceipt","HexaDoubleQk",{input[0],input[1],input[2],input[3]},ro);}else g.node("fp32_qk","ForwardQKRaw","HexaFp32",{input[0],input[1],input[2],input[3]},{raw[0],raw[1],rr});
  if(s==2){for(unsigned j=0;j<2;++j){scaled[j]=mat("scaled"+std::to_string(j),mid);masked[j]=mat("masked"+std::to_string(j),mid);p[j]=mat("p"+std::to_string(j),QNN_TENSOR_TYPE_APP_READ);}auto sc=g.constant("scale",{1,1,1,1},{1.f/std::sqrt(32.f)});std::vector<float> mask(1024,0.f);for(unsigned r=0;r<32;++r)for(unsigned c=r+1;c<32;++c)mask[r*32+c]=-10000.f;auto mc=g.constant("mask",{1,1,32,32},mask);sr=rec("scale_receipt",4105);mr=rec("mask_receipt",5128);pr=rec("softmax_receipt",8328);g.node("scale","ForwardScoreScale","HexaFp32",{raw[0],raw[1],sc},{scaled[0],scaled[1],sr});g.node("mask","ForwardMaskAdd","HexaFp32",{scaled[0],scaled[1],mc},{masked[0],masked[1],mr});g.node("softmax","ForwardSoftmax","HexaFp32",{masked[0],masked[1]},{p[0],p[1],pr});}
  auto start=std::chrono::steady_clock::now();g.finalize();finalizeUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
 }
 auto run(const doubleqkfixture::Case& c,const mixedfixture::Case& m,const s2fixture::Case& ln){std::map<unsigned,std::vector<float>> in;if(stage){in[x]=ln.x;in[ga]=ln.ga;in[be]=ln.be;in[dy]=ln.dy;for(unsigned j=0;j<3;++j)in[w[j]]=m.w[j];}else for(unsigned j=0;j<4;++j)in[input[j]]=c.heads[j];auto out=g.run(in);if(!firstUs)firstUs=g.executeUs;return out;}
};
}
