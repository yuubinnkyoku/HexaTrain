// Fixed criteria, chosen before device execution from the existing LN tests:
// analytic/numeric 5e-4; all gradients 2e-2; ordinary normalized forward 2e-3.
// Low variance forward retains the previous primitive-composition 6e-2 bound;
// upstream is deliberately NOT reduced (previous test multiplied it by .005).
std::string layerNormStabilityCheck() {
  using namespace lnvalidation;
  const auto legacy=layerNormBackwardCheck();
  if(legacy.find("status=SUCCESS\n")==std::string::npos) return legacy;
  std::ostringstream r;r<<std::setprecision(12)<<"LAYERNORM_STABILITY_VALIDATION\n"
    "execution_mode=QNN_HTP_LAYER_NORM_BACKWARD_CHECK\nmethod=SHARED_TRAINING_BUILDER\n"
    "legacy_backward_gate=PASS\nforward_abs_limit=0.002\nlow_variance_forward_abs_limit=0.06\n"
    "gradient_abs_limit=0.02\nanalytic_numeric_abs_limit=0.0005\nupstream_scale=1\n";
  bool requiredPass=true;uint64_t calls=0;size_t counterexamples=0;bool anyNonfinite=false;
  for(uint32_t d:{8u,64u,128u}) {
    Runtime runtime;RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;runtime.setOptions(o);std::string e;
    if(!runtime.initialize(QnnBackendKind::HTP,e)||!runtime.prepareLayerNormStability(2,d,1e-5f,true,e))return failure("ln_stability",e,runtime);
    for(const auto& c:cases(d)) for(int repeat=0;repeat<3;++repeat){
      LayerNormStabilityOutputs h;
      if(!runtime.executeLayerNormStability(c.x,c.dy,c.gamma,c.beta,h,e))return failure("ln_stability",e,runtime);
      ++calls;auto ref=doubleReference(c,d);auto fp=reference(c.x,c.dy,c.gamma,c.beta,d,1e-5f);
      double y=error(ref.y,h.values.output),xhat=error(ref.xhat,h.values.normalized),dx=error(ref.dx,h.values.dInput),dg=error(ref.dg,h.values.dGamma),db=error(ref.db,h.values.dBeta);
      bool allFinite=true;for(const auto& tap:h.taps)allFinite&=finite(tap);
      allFinite&=finite(h.values.output)&&finite(h.values.normalized)&&finite(h.values.dInput)&&finite(h.values.dGamma)&&finite(h.values.dBeta);
      const bool low=c.name=="low_variance"||c.name=="epsilon_dominated"||c.name=="constant"||c.name=="large_mean_low_variance";
      const double forwardLimit=low?.06:.002;
      bool pass=allFinite&&y<=forwardLimit&&xhat<=forwardLimit&&dx<=.02&&dg<=.02&&db<=.02;
      if(c.required)requiredPass&=pass;else if(!pass)++counterexamples;
      anyNonfinite|=!allFinite;
      r<<"case d="<<d<<" name="<<c.name<<" repeat="<<repeat<<" required="<<c.required<<" finite="<<allFinite
       <<" y_abs="<<y<<" xhat_abs="<<xhat<<" dx_abs="<<dx<<" dg_abs="<<dg<<" db_abs="<<db
       <<" fp32_dx_abs="<<error(fp.dx,h.values.dInput)<<" pass="<<pass<<'\n';
      // Diagnostic order: mean,c,cs,square,var_s,var_eps_s,inv_s,inv,max_c,scale,eps,...
      for(size_t t=0;t<h.taps.size();++t){double max=0;size_t bad=0;for(float v:h.taps[t]){if(!std::isfinite(v))++bad;else max=std::max(max,std::abs(double(v)));}
        r<<"tap d="<<d<<" name="<<c.name<<" repeat="<<repeat<<" index="<<t<<" nonfinite="<<bad<<" max_abs="<<max<<'\n';}
    }
    r<<runtime.apiTraceSummary()<<runtime.diagnostics();
  }
  r<<"htp_graph_execute_count="<<calls<<"\nrequired_correctness_gate="<<(requiredPass?"PASS":"FAIL")
   <<"\nexploratory_counterexamples="<<counterexamples<<"\nobserved_nonfinite="<<anyNonfinite
   <<"\ncpu_fallback=false\nstatus="<<(requiredPass?"SUCCESS":"FAILED")<<'\n';
  return r.str();
}
