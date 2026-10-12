// Acceptance is fixed in docs/htp-layernorm-numerical-redesign.md before device runs.
// No production numerical path is changed by this diagnostic suite.
std::string layerNormRedesignCheck(){
  using namespace lnvalidation;std::ostringstream r;r<<std::setprecision(12)<<"LAYERNORM_REDESIGN\nexecution_mode=QNN_HTP_LAYER_NORM_REDESIGN\n";
  uint64_t calls=0;bool accepted=false;
  std::map<std::string,LayerNormBackwardOutputs> anchors;
  const char* labels[]{"PREVIOUS","DYNAMIC_RANGE","MEAN_BACKWARD","SCALE8","FLOAT32_CONFIG","COMPENSATION_CONFIG"};
  auto bits=[](const std::vector<float>& a,const std::vector<float>& b){size_t count=0;if(a.size()!=b.size())return size_t(-1);for(size_t i=0;i<a.size();++i){uint32_t aa,bb;std::memcpy(&aa,&a[i],4);std::memcpy(&bb,&b[i],4);count+=aa!=bb;}return count;};
  auto metric=[&](const char* candidate,uint32_t d,const std::string& name,const std::string& stage,const auto& ref,const auto& out){auto e=errors(ref,out);
    r<<"metric candidate="<<candidate<<" d="<<d<<" name="<<name<<" stage="<<stage<<" abs="<<e.absolute<<" rms="<<e.rms<<" relative="<<e.relative<<" min="<<e.minimum<<" max="<<e.maximum<<" nonfinite="<<e.nonfinite<<'\n';};
  for(uint32_t candidate=0;candidate<6;++candidate){bool all=true;size_t passCount=0,total=0;
    for(uint32_t d:{8u,16u,32u,64u,128u}){
      Runtime tap,native;RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;
      if(candidate==4)o.htpGraphPrecisionMode=2;if(candidate==5)o.htpGraphPrecisionCompensation=2;
      tap.setOptions(o);native.setOptions(o);std::string e;uint32_t v=candidate<4?candidate:0;
      if(!tap.initialize(QnnBackendKind::HTP,e)||!tap.prepareLayerNormStability(2,d,1e-5f,true,e,true,v)||
         !native.initialize(QnnBackendKind::HTP,e)||!native.prepareLayerNormStability(2,d,1e-5f,true,e,false,v))return failure("ln_redesign",e,tap);
      for(const auto& c:redesignCases(d)){
        auto ref=doubleReference(c,d);auto fp=reference(c.x,c.dy,c.gamma,c.beta,d,1e-5f);LayerNormStabilityOutputs h,n;
        if(!tap.executeLayerNormStability(c.x,c.dy,c.gamma,c.beta,h,e)||!native.executeLayerNormStability(c.x,c.dy,c.gamma,c.beta,n,e))return failure("ln_redesign_execute",e,tap);calls+=2;
        auto& a=h.values;size_t mismatch=bits(a.output,n.values.output)+bits(a.normalized,n.values.normalized)+bits(a.dInput,n.values.dInput)+bits(a.dGamma,n.values.dGamma)+bits(a.dBeta,n.values.dBeta);
        const auto key=std::to_string(d)+":"+c.name;size_t priorMismatch=0;
        if(candidate==0)anchors[key]=a;else{const auto& old=anchors.at(key);priorMismatch=bits(old.output,a.output)+bits(old.normalized,a.normalized)+bits(old.dInput,a.dInput)+bits(old.dGamma,a.dGamma)+bits(old.dBeta,a.dBeta);}
        bool low=c.name=="low_variance"||c.name=="epsilon_dominated"||c.name=="constant"||c.name=="large_mean_low_variance"||c.name=="random_seed7"||c.name=="random_seed97";
        double lim=low?.06:.002;bool pass=error(ref.y,a.output)<=lim&&error(ref.xhat,a.normalized)<=lim&&error(ref.dx,a.dInput)<=.02&&error(ref.dg,a.dGamma)<=.02&&error(ref.db,a.dBeta)<=.02&&mismatch==0;
        for(auto& t:h.taps)pass&=finite(t);all&=pass;++total;passCount+=pass;
        r<<"case candidate="<<labels[candidate]<<" d="<<d<<" name="<<c.name<<" y_abs="<<error(ref.y,a.output)<<" xhat_abs="<<error(ref.xhat,a.normalized)<<" dx_abs="<<error(ref.dx,a.dInput)<<" dg_abs="<<error(ref.dg,a.dGamma)<<" db_abs="<<error(ref.db,a.dBeta)<<" fp32_dx_abs="<<error(fp.dx,a.dInput)<<" observer_bit_mismatch="<<mismatch<<" previous_bit_mismatch="<<priorMismatch<<" pass="<<pass<<'\n';
        if(candidate==0){
          auto t=stages(c,d);for(size_t k=0;k<t.size();++k)metric(labels[candidate],d,c.name,"tap"+std::to_string(k),t[k],h.taps[k]);
          metric(labels[candidate],d,c.name,"xhat",ref.xhat,a.normalized);metric(labels[candidate],d,c.name,"dx",ref.dx,a.dInput);
          auto hh=cast<double>(a.normalized),ii=cast<double>(h.taps[7]);
          auto both=replayDx(c,d,hh,ii),invOnly=replayDx(c,d,ref.xhat,ii),xhatOnly=replayDx(c,d,hh,ref.inverse);
          std::vector<double> finalProduct(c.x.size());for(size_t j=0;j<finalProduct.size();++j)finalProduct[j]=double(h.taps[19][j])*h.taps[20][j/d];
          double lattice=0;for(float f:a.dInput)if(std::isfinite(f))lattice=std::max(lattice,std::abs(double(f)-halfLattice(f)));
          r<<"causal d="<<d<<" name="<<c.name<<" inverse_only_dx_abs="<<error(ref.dx,invOnly)<<" xhat_only_dx_abs="<<error(ref.dx,xhatOnly)<<" forward_only_dx_abs="<<error(ref.dx,both)<<" backward_residual_abs="<<error(both,a.dInput)<<" final_multiply_local_abs="<<error(finalProduct,a.dInput)<<" half_lattice_lower_bound="<<latticeLowerBound(ref.dx)<<" observed_lattice_distance="<<lattice<<'\n';
        }
      }r<<tap.apiTraceSummary()<<tap.diagnostics()<<native.apiTraceSummary()<<native.diagnostics();
    }accepted|=all;r<<"candidate label="<<labels[candidate]<<" pass_count="<<passCount<<" total="<<total<<" gate="<<(all?"PASS":"FAIL")<<'\n';
  }
  // Direct LayerNorm is a forward-only isolation, not a backward candidate.
  for(uint32_t d:{8u,16u,32u,64u,128u}){Runtime p;RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;p.setOptions(o);std::string e;
    if(!p.initialize(QnnBackendKind::HTP,e)||!p.prepareLayerNorm(1,2,d,1e-5f,e))return failure("ln_direct",e,p);
    for(const auto& c:redesignCases(d)){std::vector<float> out(c.x.size(),std::numeric_limits<float>::quiet_NaN());if(!p.executeLayerNorm(c.x,out,e))return failure("ln_direct_execute",e,p);++calls;
      metric("DIRECT_FORWARD_ONLY",d,c.name,"xhat",doubleReference(c,d).xhat,out);}
    r<<p.apiTraceSummary()<<p.diagnostics();
  }
  // Fixed positive primitive inputs: zero-denominator/negative sqrt are excluded.
  for(uint32_t config:{0u,2u}){Runtime p;RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;o.htpGraphPrecisionMode=config;p.setOptions(o);std::string e;
    if(!p.initialize(QnnBackendKind::HTP,e)||!p.prepareLayerNormArithmetic(2,64,e))return failure("ln_primitive",e,p);
    const char* op[]{"add","subtract","multiply","divide","sqrt","rsqrt","reciprocal_via_divide","mean","sum"};
    for(float value:{1e-8f,5.96046448e-8f,1e-7f,1e-5f,.001f,1.f,255.75f,255.875f,256.f,269.75f,512.f,1024.f,4096.f}){
      std::ostringstream inputName;inputName<<std::setprecision(12)<<value;
      std::vector<float>a(128,value),b(128,value);if(value==1024.f){for(size_t i=0;i<64;++i)a[i]+=i%2?.001f:-.001f;}
      std::vector<std::vector<float>> out;if(!p.executeLayerNormArithmetic(a,b,out,e))return failure("ln_primitive_execute",e,p);++calls;
      for(size_t k=0;k<9;++k){std::vector<double> ref(k<7?128:2);
        for(size_t i=0;i<ref.size();++i){double aa=a[i],bb=b[i];if(k==0)ref[i]=aa+bb;if(k==1)ref[i]=aa-bb;if(k==2)ref[i]=aa*bb;if(k==3)ref[i]=aa/bb;if(k==4)ref[i]=std::sqrt(aa);if(k==5)ref[i]=1/std::sqrt(aa);if(k==6)ref[i]=1/aa;if(k>=7){double sum=0;for(size_t j=0;j<64;++j)sum+=a[i*64+j];ref[i]=k==7?sum/64:sum;}}
        metric(config?"PRIMITIVE_FLOAT32":"PRIMITIVE_DEFAULT",64,inputName.str(),op[k],ref,out[k]);
        r<<"primitive config="<<config<<" input="<<value<<" op="<<op[k]<<" first="<<out[k][0]<<" last="<<out[k].back()<<'\n';
      }
    }r<<p.apiTraceSummary()<<p.diagnostics();
  }
  r<<"htp_graph_execute_count="<<calls<<"\ncpu_fallback=false\naccepted_leaf_candidate="<<accepted<<"\nstress_60k=BLOCKED_PENDING_FULL_MODEL_GATES\nstatus="<<(accepted?"SUCCESS":"FAILED")<<'\n';return r.str();
}
