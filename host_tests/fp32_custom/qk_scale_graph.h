#pragma once
namespace qkscalestudy {
// 0 custom raw, 1 standard raw negative, 2 custom raw -> standard scale.
struct Forward : orderedqkvstudy::Graph {
 std::array<unsigned,2> scores,scaled;unsigned rr;unsigned next;
 Forward(std::string n,bool obs,bool tap,unsigned mode):Graph(n,2,true,obs,tap,true),next(mode){
  rr=mode==1?~0u:g.tensor("qk_receipt",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6152},QNN_DATATYPE_INT_32);
  for(unsigned h=0;h<2;++h)scores[h]=g.tensor(("scores"+std::to_string(h)).c_str(),mode==2&&!tap?QNN_TENSOR_TYPE_NATIVE:QNN_TENSOR_TYPE_APP_READ,{1,1,32,32});
  if(mode==1)for(unsigned h=0;h<2;++h)op::mm(g,"standard_qk"+std::to_string(h),heads[h],heads[2+h],scores[h],false,true);
  else g.node("raw_qk","ForwardQKRaw","HexaFp32",{heads[0],heads[1],heads[2],heads[3]},{scores[0],scores[1],rr});
  if(mode==2){auto scale=g.constant("scale",{1,1,1,1},{1.f/std::sqrt(float(64/2))});for(unsigned h=0;h<2;++h){scaled[h]=g.tensor(("scaled"+std::to_string(h)).c_str(),QNN_TENSOR_TYPE_APP_READ,{1,1,32,32});g.node(("standard_scale"+std::to_string(h)).c_str(),QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{scores[h],scale},{scaled[h]});}}
  auto start=std::chrono::steady_clock::now();g.finalize();std::printf("frontierfinalize,%s,us=%.3f\n",n.c_str(),std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
 }
};
}
