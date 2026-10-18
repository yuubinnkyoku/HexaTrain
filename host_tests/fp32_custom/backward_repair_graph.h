#pragma once
#include "saved_p_backward_graph.h"
namespace repairprobe {
struct Graph {
 DiagnosticGraph g;
 savedp::Mode mode;bool observed,tap,native,softmax,dvalue;
 std::array<unsigned,2> p{},dp{},dc{},product{},dot{},centered{},ds{},dv{},sr{~0u,~0u},vr{~0u,~0u};
 unsigned x=~0u,ga=~0u,be=~0u,dy=~0u;std::array<unsigned,3> w{};double finalizeUs=0;
 Graph(savedp::Mode m,bool o,bool t):g(std::string("repair_")+savedp::label(m)+std::to_string(o)+std::to_string(t)),
 mode(m),observed(o),tap(t),native(m!=savedp::Mode::StandaloneSoftmax&&m!=savedp::Mode::StandaloneDValue),
 softmax(m==savedp::Mode::StandaloneSoftmax||m==savedp::Mode::NativeSoftmax||m==savedp::Mode::Dual),
 dvalue(m==savedp::Mode::StandaloneDValue||m==savedp::Mode::NativeDValue||m==savedp::Mode::Dual){
  g.receiptEnabled=o;auto mid=t?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
  auto mat=[&](const std::string& n,Qnn_TensorType_t type,unsigned r=32,unsigned c=32){return g.tensor(n.c_str(),type,{1,1,r,c});};
  auto rec=[&](const char* n,unsigned count){return g.tensor(n,QNN_TENSOR_TYPE_APP_READ,{1,1,1,count},QNN_DATATYPE_INT_32);};
  if(native){
   // Verbatim topology/math of saved_p_backward_graph.h upstream. The existing
   // packages, local LN backward control, tensor directions and scale stay fixed.
   x=mat("x",QNN_TENSOR_TYPE_APP_WRITE,32,64);ga=mat("gamma",QNN_TENSOR_TYPE_APP_WRITE,1,64);be=mat("beta",QNN_TENSOR_TYPE_APP_WRITE,1,64);dy=mat("dy",QNN_TENSOR_TYPE_APP_WRITE,32,64);
   auto y=mat("y",mid,32,64),h=mat("h",mid,32,64),iv=mat("iv",mid,32,1);auto eps=g.constant("eps",{1,1,1,1},{1e-5f});
   std::vector<unsigned> lo={y,h,iv};if(o)lo.push_back(rec("ln_receipt",6313));g.node("ln1",o?"S2":"S2NoReceipt","HexaSelectiveLn1",{x,ga,be,eps},lo);
   auto dx=mat("dx",QNN_TENSOR_TYPE_APP_READ,32,64),dg=mat("dg",QNN_TENSOR_TYPE_APP_READ,1,64),db=mat("db",QNN_TENSOR_TYPE_APP_READ,1,64);
   g.node("local_backward","LnBackward","HexaFp32",{dy,ga,h,iv},{dx,dg,db,rec("ln_back_receipt",6248)});
   std::array<unsigned,3> q{};for(unsigned j=0;j<3;++j){w[j]=mat("w"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE,64,64);q[j]=mat("qkv"+std::to_string(j),mid,32,64);}
   std::vector<unsigned> qo={q[0],q[1],q[2]};if(o)qo.push_back(rec("mixed_receipt",20512));g.node("mixed",o?"MixedPrecisionQKV":"MixedPrecisionQKVNoReceipt","HexaMixedQkv",{y,w[0],w[1],w[2]},qo);
   std::array<unsigned,6> heads{};for(unsigned j=0;j<6;++j)heads[j]=mat("head"+std::to_string(j),j>=4?QNN_TENSOR_TYPE_APP_READ:mid);
   g.node("heads","QKVHeads","HexaFp32",{q[0],q[1],q[2]},{heads[0],heads[1],heads[2],heads[3],heads[4],heads[5],rec("head_receipt",12296)});
   std::array<unsigned,2> raw{},scaled{},masked{};for(unsigned j=0;j<2;++j){raw[j]=mat("raw"+std::to_string(j),mid);scaled[j]=mat("scaled"+std::to_string(j),mid);masked[j]=mat("masked"+std::to_string(j),mid);p[j]=mat("p"+std::to_string(j),m==savedp::Mode::Forward?QNN_TENSOR_TYPE_APP_READ:mid);}
   std::vector<unsigned> ro={raw[0],raw[1]};if(o)ro.push_back(rec("qk_receipt",6176));g.node("double_qk",o?"ForwardRawQkDouble":"ForwardRawQkDoubleNoReceipt","HexaDoubleQk",{heads[0],heads[1],heads[2],heads[3]},ro);
   auto sc=g.constant("scale",{1,1,1,1},{1.f/std::sqrt(32.f)});std::vector<float> mask(1024,0.f);for(unsigned r=0;r<32;++r)for(unsigned c=r+1;c<32;++c)mask[r*32+c]=-10000.f;auto mc=g.constant("mask",{1,1,32,32},mask);
   g.node("scale","ForwardScoreScale","HexaFp32",{raw[0],raw[1],sc},{scaled[0],scaled[1],rec("scale_receipt",4105)});
   g.node("mask","ForwardMaskAdd","HexaFp32",{scaled[0],scaled[1],mc},{masked[0],masked[1],rec("mask_receipt",5128)});
   g.node("softmax","ForwardSoftmax","HexaFp32",{masked[0],masked[1]},{p[0],p[1],rec("p_receipt",8328)});
  }else for(unsigned j=0;j<2;++j)p[j]=mat("p"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE);
  for(unsigned h=0;h<2;++h){auto n="h"+std::to_string(h)+"_";
   if(softmax){dp[h]=mat(n+"dp",QNN_TENSOR_TYPE_APP_WRITE);ds[h]=mat(n+"ds",QNN_TENSOR_TYPE_APP_READ);std::vector<unsigned> out={ds[h]};
    if(t){product[h]=mat(n+"product",QNN_TENSOR_TYPE_APP_READ);dot[h]=mat(n+"dot",QNN_TENSOR_TYPE_APP_READ,32,1);centered[h]=mat(n+"centered",QNN_TENSOR_TYPE_APP_READ);out.insert(out.end(),{product[h],dot[h],centered[h]});}
    if(o){sr[h]=rec((n+"soft_receipt").c_str(),5184);out.push_back(sr[h]);}
    g.node((n+"backward").c_str(),t?(o?"SoftmaxTap":"SoftmaxTapNoReceipt"):(o?"Softmax":"SoftmaxNoReceipt"),"HexaBackwardRepair",{p[h],dp[h]},out);
   }
   if(dvalue){dc[h]=mat(n+"dc",QNN_TENSOR_TYPE_APP_WRITE);dv[h]=mat(n+"dv",QNN_TENSOR_TYPE_APP_READ);std::vector<unsigned> out={dv[h]};if(o){vr[h]=rec((n+"value_receipt").c_str(),3104);out.push_back(vr[h]);}
    g.node((n+"dvalue").c_str(),o?"DValueTN32":"DValueTN32NoReceipt","HexaBackwardRepair",{p[h],dc[h]},out);
   }
  }
  auto begin=std::chrono::steady_clock::now();g.finalize();finalizeUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
 }
};
}
