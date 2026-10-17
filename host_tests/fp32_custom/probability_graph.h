#pragma once
#include "probability_contract.h"
namespace probabilitystudy {
struct Tail {
 std::array<unsigned,2> raw{},scaled{},masked{},p{},final{},ar{};unsigned sr=~0u,mr=~0u,pr=~0u,br=~0u;std::array<unsigned,2> gradient{};unsigned stop=0,start=0;bool customMask=false;
 void build(DiagnosticGraph& g,std::array<unsigned,2> input,unsigned first,unsigned last,bool tap,bool cm=false,bool cs=false){start=first;stop=last;customMask=cm;auto mid=tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
  auto tensor=[&](std::string n,bool terminal){return g.tensor(n.c_str(),terminal?QNN_TENSOR_TYPE_APP_READ:mid,{1,1,32,32});};
  auto rec=[&](std::string n,unsigned count){return g.tensor(n.c_str(),QNN_TENSOR_TYPE_APP_READ,{1,1,1,count},QNN_DATATYPE_INT_32);};
  if(first==0){raw=input;for(unsigned h=0;h<2;++h)scaled[h]=tensor("scaled"+std::to_string(h),last==0);sr=rec("scale_receipt",4105);auto s=g.constant("forward_scale",{1,1,1,1},{probabilitycontract::scale()});g.node("forward_scale","ForwardScoreScale","HexaFp32",{raw[0],raw[1],s},{scaled[0],scaled[1],sr});input=scaled;}
  if(first<=1&&last>=1){scaled=input;auto m=g.constant("source_mask",{1,1,32,32},probabilitycontract::mask());for(unsigned h=0;h<2;++h)masked[h]=tensor("masked"+std::to_string(h),last==1);if(cm){mr=rec("mask_receipt",5128);g.node("paired_mask","ForwardMaskAdd","HexaFp32",{input[0],input[1],m},{masked[0],masked[1],mr});}else for(unsigned h=0;h<2;++h)g.node(("mask"+std::to_string(h)).c_str(),QNN_OP_ELEMENT_WISE_ADD,"qti.aisw",{input[h],m},{masked[h]});input=masked;}
  if(first<=2&&last>=2){masked=input;for(unsigned h=0;h<2;++h)p[h]=tensor("p"+std::to_string(h),last==2);if(cs){pr=rec("forward_softmax_receipt",8328);g.node("forward_softmax_pair","ForwardSoftmax","HexaFp32",{input[0],input[1]},{p[0],p[1],pr});}else for(unsigned h=0;h<2;++h)g.node(("forward_softmax"+std::to_string(h)).c_str(),QNN_OP_SOFTMAX,"qti.aisw",{input[h]},{p[h]},{op::integer(QNN_OP_SOFTMAX_PARAM_AXIS,3),op::real(QNN_OP_SOFTMAX_PARAM_BETA,1.f)});input=p;}
  if(last>=3){p=input;for(unsigned h=0;h<2;++h){gradient[h]=g.tensor(("gradient"+std::to_string(h)).c_str(),QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,32});final[h]=tensor("backward_final"+std::to_string(h),true);}if(last==3){br=rec("backward_receipt",10312);g.node("saved_p_backward","SoftmaxBackward","HexaFp32",{gradient[0],gradient[1],p[0],p[1]},{final[0],final[1],br});}else for(unsigned h=0;h<2;++h)op::mm(g,"saved_p_dvalue"+std::to_string(h),p[h],gradient[h],final[h],true,false);}
  else final=input;
 }
};
inline void finish(DiagnosticGraph& g,const std::string& name){auto start=std::chrono::steady_clock::now();g.finalize();std::printf("probfinalize,%s,us=%.3f\n",name.c_str(),std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
#ifndef ONE_GRAPH_HOST_AUDIT
 closurestudy::describe(g,name);
#endif
}
struct Standalone {DiagnosticGraph g;Tail t;std::array<unsigned,2>x;std::string name;bool observed;
 Standalone(std::string n,unsigned stage,bool obs,bool cm=false,bool cs=false):g(n),name(n),observed(obs){g.receiptEnabled=obs;for(unsigned h=0;h<2;++h)x[h]=g.tensor(("input"+std::to_string(h)).c_str(),QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,32});t.build(g,x,stage,stage,false,cm,cs);finish(g,n);}
};
struct Native:orderedqkvstudy::Graph {Tail t;std::array<unsigned,2>raw;unsigned rr=~0u,lastRaw=0,lastScale=0,lastAdd=0,lastBack=0;
 Native(std::string n,unsigned stage,bool obs,bool tap,bool cm=false,bool cs=false):Graph(n,2,true,obs,tap,true){for(unsigned h=0;h<2;++h)raw[h]=g.tensor(("raw"+std::to_string(h)).c_str(),tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE,{1,1,32,32});rr=g.tensor("raw_receipt",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6152},QNN_DATATYPE_INT_32);g.node("forward_qk","ForwardQKRaw","HexaFp32",{heads[0],heads[1],heads[2],heads[3]},{raw[0],raw[1],rr});t.build(g,raw,0,stage,tap,cm,cs);finish(g,n);}
};
}
