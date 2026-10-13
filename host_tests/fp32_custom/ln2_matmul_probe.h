#pragma once
// Built only after frozen-input direct/native Custom MatMul Device success.
// External inputs are X/gamma/beta/dY/residual/Wo/context. No injected dX/dresidual.
namespace ln2probe {
namespace ln=phonelm::qnn::lnvalidation;
inline unsigned lnMetric(const std::string& variant,const mmfixture::Fixture& f,const char* field,
 const std::vector<double>& ref,const std::vector<float>& host,const std::vector<float>& actual,double limit){
 auto e=ln::errors(ref,actual),h=ln::errors(host,actual);bool pass=e.nonfinite==0&&e.absolute<=limit;
 std::printf("chainln,ln,%s,%s,%u,%u,%s,abs=%.12g,rms=%.12g,relative=%.12g,host_abs=%.12g,hash=%016llx,gate=%u\n",variant.c_str(),f.c.name.c_str(),f.control,f.additional,field,e.absolute,e.rms,e.relative,h.absolute,(unsigned long long)hashWords(actual),pass);return !pass;
}
inline std::vector<uint32_t> words(const std::vector<uint32_t>& r,unsigned at,unsigned n){if(at+n>r.size()){std::puts("CHAIN_RECEIPT_SIZE_FAIL");std::exit(3);}return {r.begin()+at,r.begin()+at+n};}
inline void equality(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,const char* error){if(a!=b){std::puts(error);std::exit(3);}}
inline unsigned run(){unsigned failures=0;auto contextFixtures=mmprobe::frozenFixtures(mmfixture::shapes[0]),weightFixtures=mmprobe::frozenFixtures(mmfixture::shapes[1]);
 std::map<unsigned,std::vector<uint32_t>> contextAnchor,weightAnchor;
 for(auto variant:std::vector<std::string>{"native","tap","fused"}){bool tap=variant=="tap",fused=variant=="fused";
  DiagnosticGraph g("real_dsp_ln2_"+variant);auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ,N=QNN_TENSOR_TYPE_NATIVE;
  auto x=g.tensor("x",W,{1,1,32,64}),gamma=g.tensor("gamma",W,{1,1,1,64}),beta=g.tensor("beta",W,{1,1,1,64}),eps=g.tensor("eps",W,{1,1,1,1});
  auto dy=g.tensor("dy",W,{1,1,32,64}),residual=g.tensor("residual",W,{1,1,32,64}),wo=g.tensor("wo",W,{1,1,64,64}),ctx=g.tensor("context",W,{1,1,32,64});
  auto y=g.tensor("y",R,{1,1,32,64}),h=g.tensor("xhat",tap?R:N,{1,1,32,64}),inv=g.tensor("inverse",tap?R:N,{1,1,32,1});
  auto fr=g.tensor("forward_receipt",R,{1,1,1,4265},QNN_DATATYPE_INT_32);g.node("forward","LnForward","HexaFp32",{x,gamma,beta,eps},{y,h,inv,fr});
  auto dx=g.tensor(fused?"fused_residual":"dx",tap?R:N,{1,1,32,64}),dg=g.tensor("dg",R,{1,1,1,64}),db=g.tensor("db",R,{1,1,1,64});
  auto br=g.tensor("backward_receipt",R,{1,1,1,fused?10344u:6248u},QNN_DATATYPE_INT_32);
  if(fused)g.node("backward","LnResidual","HexaFp32",{dy,gamma,h,inv,residual},{dx,dg,db,br});
  else g.node("backward","LnBackward","HexaFp32",{dy,gamma,h,inv},{dx,dg,db,br});
  unsigned result=dx,ar=0;if(!fused){result=g.tensor("dresidual1",tap?R:N,{1,1,32,64});ar=g.tensor("add_receipt",R,{1,1,1,8200},QNN_DATATYPE_INT_32);g.node("residual","Add","HexaFp32",{dx,residual},{result,ar});}
  auto dc=g.tensor("dcontext",R,{1,1,32,64}),dw=g.tensor("dwo",R,{1,1,64,64});
  auto cr=g.tensor("dcontext_receipt",R,{1,1,1,8208},QNN_DATATYPE_INT_32),wr=g.tensor("dwo_receipt",R,{1,1,1,8208},QNN_DATATYPE_INT_32);
  g.node("dcontext","MatMulDContext","HexaFp32",{result,wo},{dc,cr});g.node("dwo","MatMulDWo","HexaFp32",{ctx,result},{dw,wr});g.finalize();
  unsigned lastC=0,lastW=0,lastF=0,lastB=0,lastA=0,bad=0;
  auto inputs=[&](unsigned index){auto& a=contextFixtures[index];auto& b=weightFixtures[index];
   if(a.c.x!=b.c.x||a.c.gamma!=b.c.gamma||a.c.beta!=b.c.beta||a.c.dy!=b.c.dy||a.residual!=b.residual){std::puts("CHAIN_EXTERNAL_FIXTURE_FAIL");std::exit(3);}
   return std::map<unsigned,std::vector<float>>{{x,a.c.x},{gamma,a.c.gamma},{beta,a.c.beta},{eps,{1e-5f}},{dy,a.c.dy},{residual,a.residual},{wo,a.other},{ctx,b.other}};
  };
  auto verify=[&](unsigned index,const std::map<unsigned,std::vector<uint32_t>>& out,bool numeric){
   auto& f=contextFixtures[index];auto& wf=weightFixtures[index];auto& fReceipt=out.at(fr);auto& bReceipt=out.at(br);
   if(fReceipt[1]<=lastF||bReceipt[1]<=lastB){std::puts("CHAIN_STALE_RECEIPT");std::exit(3);}lastF=fReceipt[1];lastB=bReceipt[1];
   checkReceipt(fReceipt,4,{f.c.x,f.c.gamma,f.c.beta,{1e-5f}},2080);
   auto hb=words(fReceipt,2185,2048),ib=words(fReceipt,4233,32);auto ah=decode(hb),ai=decode(ib);
   std::vector<std::vector<float>> received={f.c.dy,f.c.gamma,ah,ai};if(fused)received.push_back(f.residual);
   checkReceipt(bReceipt,fused?6:5,received,fused?4096:2048);
   auto dxb=words(bReceipt,fused?6248:4200,2048);auto adx=decode(dxb);std::vector<uint32_t> rb;
   if(fused)rb=words(bReceipt,8296,2048);
   else{auto& a=out.at(ar);if(a[0]!=0x48383101u||a[1]<=lastA||a[2]!=2048||a[3]!=2||a[5]!=81||a[6]!=32||a[7]){std::puts("CHAIN_ADD_RECEIPT_FAIL");std::exit(3);}lastA=a[1];
    equality(words(a,8,2048),dxb,"CHAIN_DX_NATIVE_FAIL");for(unsigned j=0;j<2048;++j)if(a[2056+j]!=word(f.residual[j])||a[4104+j]){std::puts("CHAIN_RESIDUAL_RECEIPT_FAIL");std::exit(3);}rb=words(a,6152,2048);
   }
   auto res=decode(rb);for(unsigned j=0;j<2048;++j)if(word(adx[j]+f.residual[j])!=rb[j]){std::puts("CHAIN_ADD_OUTPUT_FAIL");std::exit(3);}
   // Both consumers must receive the producer's complete output word vector.
   equality(words(out.at(cr),16,2048),rb,"CHAIN_DCONTEXT_NATIVE_FAIL");equality(words(out.at(wr),2064,2048),rb,"CHAIN_DWO_NATIVE_FAIL");
   mmprobe::receipt(out.at(cr),mmfixture::shapes[0],res,f.other,out.at(dc),lastC);mmprobe::receipt(out.at(wr),mmfixture::shapes[1],wf.other,res,out.at(dw),lastW);
   if(tap){equality(out.at(h),hb,"CHAIN_XHAT_TAP_FAIL");equality(out.at(inv),ib,"CHAIN_INVERSE_TAP_FAIL");equality(out.at(dx),dxb,"CHAIN_DX_TAP_FAIL");equality(out.at(result),rb,"CHAIN_RESIDUAL_TAP_FAIL");}
   if(numeric){auto ref=ln::doubleReference(f.c,64);auto host=ln::anchoredFp32Oracle(f.c,64);bool low=f.c.name=="low_variance"||f.c.name=="epsilon_dominated"||f.c.name=="constant"||f.c.name=="large_mean_low_variance";double limit=low?.06:.002;
    bad+=lnMetric(variant,f,"y",ref.y,host.y,decode(out.at(y)),limit);bad+=lnMetric(variant,f,"xhat",ref.xhat,host.xhat,ah,limit);bad+=lnMetric(variant,f,"inverse",ref.inverse,host.inverse,ai,.002);
    bad+=lnMetric(variant,f,"dx",ref.dx,host.dx,adx,.02);bad+=lnMetric(variant,f,"dg",ref.dg,host.dg,decode(out.at(dg)),.02);bad+=lnMetric(variant,f,"db",ref.db,host.db,decode(out.at(db)),.02);
    auto doubleRes=ref.dx;auto hostRes=host.dx;for(unsigned j=0;j<2048;++j){doubleRes[j]+=double(f.residual[j]);hostRes[j]+=f.residual[j];}bad+=lnMetric(variant,f,"residual",doubleRes,hostRes,res,.02);
    auto rc=mmfixture::chainReference(f,mmfixture::shapes[0]),rw=mmfixture::chainReference(wf,mmfixture::shapes[1]);
    bad+=mmprobe::metric(mmfixture::shapes[0],f,variant,res,f.other,decode(out.at(dc)),&rc);bad+=mmprobe::metric(mmfixture::shapes[1],wf,variant,wf.other,res,decode(out.at(dw)),&rw);
    if(variant=="native"){contextAnchor[index]=out.at(dc);weightAnchor[index]=out.at(dw);}else{equality(contextAnchor.at(index),out.at(dc),"CHAIN_CONTEXT_OBSERVER_FAIL");equality(weightAnchor.at(index),out.at(dw),"CHAIN_WO_OBSERVER_FAIL");}
    for(auto s:mmfixture::shapes){auto& fixture=s.left?f:wf;std::printf("chainfixture,%s,%s,%s,%u,%u,ln_hash=%016llx,other_hash=%016llx\n",s.name,variant.c_str(),fixture.c.name.c_str(),fixture.control,fixture.additional,
     (unsigned long long)hashWords(fixture.residual,hashWords(fixture.c.dy,hashWords(fixture.c.beta,hashWords(fixture.c.gamma,hashWords(fixture.c.x))))),(unsigned long long)hashWords(fixture.other));}
   }
  };
  for(unsigned i=0;i<contextFixtures.size();++i){auto out=g.run(inputs(i));verify(i,out,true);}
  failures+=bad;if(!bad)for(unsigned rep=0;rep<9;++rep){auto out=g.run(inputs(60));verify(60,out,false);std::printf("chainperf,segment,%s,%u,wall_us=%.3f,dcontext_us=%u,dwo_us=%u,ln_us=%u,add_us=%u\n",variant.c_str(),rep,g.executeUs,out[cr][4],out[wr][4],out[fr][4]+out[br][4],fused?0:out[ar][4]);}
  std::printf("chainsummary,%s,cases=68,required_failures=%u,tap_fused_final_words_equal=1\n",variant.c_str(),bad);
 }
 std::printf("summary,mmLn2,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0\n",failures);return failures;
}
}
