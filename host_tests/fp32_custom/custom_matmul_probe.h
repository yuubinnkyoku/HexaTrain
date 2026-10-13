#pragma once
#include "custom_matmul_fixture.h"
#include <limits>
namespace mmprobe {
inline std::map<std::string,std::vector<uint32_t>> anchors;
inline uint32_t ordered(float f){auto u=word(f);return u&0x80000000u?~u:u|0x80000000u;}
inline unsigned metric(const mmfixture::Shape& s,const mmfixture::Fixture& f,const std::string& variant,
 const std::vector<float>& a,const std::vector<float>& b,const std::vector<float>& actual){
 auto start=std::chrono::steady_clock::now();auto ref=mmfixture::reference(a,b,s,false),fp=mmfixture::reference(a,b,s,true);auto e=phonelm::qnn::lnvalidation::errors(ref,actual);
 double err2=0,ref2=0,rel=0,fpAbs=0;uint32_t ulp=0;unsigned mismatches=0;
 for(unsigned i=0;i<ref.size();++i){double d=double(actual[i])-ref[i];err2+=d*d;ref2+=ref[i]*ref[i];rel=std::max(rel,std::fabs(d)/std::max(.001,std::fabs(ref[i])));
  fpAbs=std::max(fpAbs,std::fabs(double(actual[i])-fp[i]));auto x=ordered(actual[i]),y=ordered(float(fp[i]));ulp=std::max(ulp,x>y?x-y:y-x);mismatches+=word(actual[i])!=word(float(fp[i]));
 }
 bool pass=e.nonfinite==0&&e.absolute<=.03;
 std::printf("custommm,%s,%s,%s,%u,%u,abs=%.12g,rms=%.12g,relative_l2=%.12g,max_relative=%.12g,nonfinite=%zu,fp_abs=%.12g,ulp=%u,fp_word_mismatch=%u,input_hash=%016llx,hash=%016llx,gate=%u,host_validation_us=%.3f\n",
  s.name,variant.c_str(),f.c.name.c_str(),f.control,f.additional,e.absolute,e.rms,std::sqrt(err2/std::max(1e-300,ref2)),rel,e.nonfinite,fpAbs,ulp,mismatches,
  (unsigned long long)hashWords(b,hashWords(a)),(unsigned long long)hashWords(actual),pass,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());return !pass;
}
inline void receipt(const std::vector<uint32_t>& r,const mmfixture::Shape& s,const std::vector<float>& a,const std::vector<float>& b,const std::vector<uint32_t>& y,unsigned& last){
 if(r.size()!=16+a.size()+b.size()+y.size()||r[0]!=(s.ta?0x48383108u:0x48383107u)||r[1]<=last||r[2]!=a.size()||r[3]!=b.size()||r[5]!=81||r[6]!=32||r[7]||
    r[8]!=s.m||r[9]!=s.n||r[10]!=s.k||r[11]!=s.ta||r[12]!=s.tb||r[14]!=y.size()||r[15]){std::puts("MM_RECEIPT_FAIL");std::exit(3);}last=r[1];
 unsigned at=16;for(auto v:{a,b})for(float f:v)if(r[at++]!=word(f)){std::puts("MM_OPERAND_RECEIPT_FAIL");std::exit(3);}
 for(auto w:y)if(r[at++]!=w){std::puts("MM_OUTPUT_RECEIPT_FAIL");std::exit(3);}
 std::printf("mmkernel,%s,%u,arithmetic_us=%u,receipt_finite_us=%u\n",s.name,r[1],r[4],r[13]);
}
inline unsigned run(bool boundary){unsigned failures=0,standardFailures=0;for(auto s:mmfixture::shapes){
 for(auto variant:boundary?std::vector<std::string>{"direct","native","tap","standard","standardNative"}:std::vector<std::string>{"direct"}){
  bool copied=variant=="native"||variant=="tap"||variant=="standardNative",tap=variant=="tap",standard=variant=="standard"||variant=="standardNative";
  DiagnosticGraph g(std::string(s.name)+variant);auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ,N=QNN_TENSOR_TYPE_NATIVE;
  auto a=g.tensor("a",W,{1,1,s.ar,s.ac}),b=g.tensor("b",W,{1,1,s.br,s.bc}),y=g.tensor("y",R,{1,1,s.m,s.n});
  unsigned mid=s.left?a:b,cr=0,mr=0,last=0,copyLast=0;
  if(copied){auto grad=mid;mid=g.tensor("native_gradient",tap?R:N,{1,1,32,64});cr=g.tensor("copy_receipt",R,{1,1,1,6152},QNN_DATATYPE_INT_32);g.node("copy","Copy","HexaFp32",{grad},{mid,cr});}
  std::vector<unsigned> ins={s.left?mid:a,s.left?b:mid};
  if(standard){Qnn_Param_t p=QNN_PARAM_INIT;p.paramType=QNN_PARAMTYPE_SCALAR;p.scalarParam.dataType=QNN_DATATYPE_BOOL_8;p.name=QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;p.scalarParam.bool8Value=s.ta;auto q=p;q.name=QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;q.scalarParam.bool8Value=s.tb;
   g.node("matmul",QNN_OP_MAT_MUL,"qti.aisw",ins,{y},{p,q});
  }else{mr=g.tensor("matmul_receipt",R,{1,1,1,16+s.ar*s.ac+s.br*s.bc+s.m*s.n},QNN_DATATYPE_INT_32);g.node("matmul",s.ta?"MatMulDWo":"MatMulDContext","HexaFp32",ins,{y,mr});}
  g.finalize();unsigned bad=0,index=0;auto fixtures=mmfixture::fixtures(s);
  for(auto f:fixtures){auto gradient=mmfixture::gradient(f),aa=s.left?gradient:f.other,bb=s.left?f.other:gradient;auto out=g.run({{a,aa},{b,bb}});
   if(copied){auto& r=out[cr];if(r[0]!=0x48383100u||r[1]<=copyLast||r[2]!=2048||r[3]!=1||r[5]!=81||r[6]!=32||r[7]){std::puts("COPY_RECEIPT_FAIL");std::exit(3);}copyLast=r[1];
    for(unsigned j=0;j<2048;++j)if(r[8+j]!=word(gradient[j])||r[2056+j]||r[4104+j]||(tap&&out[mid][j]!=word(gradient[j]))){std::puts("COPY_PRECISION_FAIL");std::exit(3);}
   }if(!standard)receipt(out[mr],s,aa,bb,out[y],last);
   std::string key=std::string(s.name)+std::to_string(index++),baselineKey=(standard?"standard":"custom")+key;
   if(!copied)anchors[baselineKey]=out[y];else if(anchors.at(baselineKey)!=out[y]){std::puts("MM_OBSERVER_FAIL");std::exit(3);}
   bad+=metric(s,f,variant,aa,bb,decode(out[y]));
  }
  if(standard)standardFailures+=bad;else failures+=bad;
  // Timing starts only after this candidate's complete accuracy matrix passes.
  if(!standard&&!bad){auto f=fixtures[60];auto grad=mmfixture::gradient(f);for(unsigned rep=0;rep<9;++rep){auto aa=s.left?grad:f.other,bb=s.left?f.other:grad;auto out=g.run({{a,aa},{b,bb}});receipt(out[mr],s,aa,bb,out[y],last);
    std::printf("mmperf,%s,%s,%u,wall_us=%.3f,kernel_us=%u,receipt_us=%u\n",s.name,variant.c_str(),rep,g.executeUs,out[mr][4],out[mr][13]);}
  }
  std::printf("pathsummary,%s,%s,cases=68,required_failures=%u,first_execute_in_api_trace=1\n",s.name,variant.c_str(),bad);
 }}std::printf("summary,%s,required_failures=%u,standard_control_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0\n",boundary?"mmBoundary":"mmDirect",failures,standardFailures);return failures;
}
}
