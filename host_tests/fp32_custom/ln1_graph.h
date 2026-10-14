#pragma once
struct Ln1Graph {
 std::vector<unsigned> ids;
 inline static const char* names[]={"mean","center","center_scale","square","variance","variance_eps","inverse_scaled","inverse","xhat","gamma","beta"};
 Ln1Graph(DiagnosticGraph& g,unsigned x,unsigned gamma,unsigned beta,bool taps,unsigned end=10,bool terminal=true){
  auto axis=g.constant("ln1_axis",{1},{from(3)},QNN_DATATYPE_UINT_32),one=g.constant("ln1_scale",{1,1,1,1},{1.f}),eps=g.constant("ln1_epsilon",{1,1,1,1},{1e-5f});
  for(unsigned stage=0;stage<=end;++stage){bool row=stage==0||(stage>=4&&stage<=7);auto name="ln1_"+std::string(names[stage]);auto y=g.tensor(name.c_str(),taps||(terminal&&stage==end)?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE,{1,1,32,row?1u:64u});ids.push_back(y);
   if(stage==0||stage==4){Qnn_Param_t a=QNN_PARAM_INIT;a.name=QNN_OP_REDUCE_MEAN_PARAM_AXES;a.paramType=QNN_PARAMTYPE_TENSOR;a.tensorParam=g.tensors[axis];g.node(name.c_str(),QNN_OP_REDUCE_MEAN,"qti.aisw",{stage==0?x:ids[3]},{y},{a,op::boolean(QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS,true)});}
   else if(stage==6)g.node(name.c_str(),QNN_OP_ELEMENT_WISE_RSQRT,"qti.aisw",{ids[5]},{y});
   else{unsigned a=0,b=0;const char* type=QNN_OP_ELEMENT_WISE_MULTIPLY;switch(stage){case 1:a=x;b=ids[0];type=QNN_OP_ELEMENT_WISE_SUBTRACT;break;case 2:a=ids[1];b=one;break;case 3:a=b=ids[2];break;case 5:a=ids[4];b=eps;type=QNN_OP_ELEMENT_WISE_ADD;break;case 7:a=ids[6];b=one;break;case 8:a=ids[1];b=ids[7];break;case 9:a=ids[8];b=gamma;break;case 10:a=ids[9];b=beta;type=QNN_OP_ELEMENT_WISE_ADD;break;}g.node(name.c_str(),type,"qti.aisw",{a,b},{y});}
  }
 }
};
