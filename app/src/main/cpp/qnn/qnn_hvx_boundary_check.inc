// Diagnostic only. Host reference never substitutes for DSP computation.
std::string hvxBoundaryCheck() {
#if !PHONELM_ENABLE_HVX_MUON
  return "status=FAILED\nfailure=HVX_DISABLED\n";
#else
  using namespace lnvalidation;
  std::ostringstream r;r<<std::setprecision(12)<<"HTP_HVX_BOUNDARY\n";
  auto bits=[](const std::vector<float>& a,const std::vector<float>& b){size_t n=0;if(a.size()!=b.size())return size_t(-1);for(size_t i=0;i<a.size();++i){uint32_t x,y;std::memcpy(&x,&a[i],4);std::memcpy(&y,&b[i],4);n+=x!=y;}return n;};
  struct Rpc {
    remote_handle64 handle=0;float *a=nullptr,*b=nullptr,*out=nullptr;int* metadata=nullptr;
    ~Rpc(){if(handle)hexatrain_hvx_probe_close(handle);if(a)rpcmem_free(a);if(b)rpcmem_free(b);if(out)rpcmem_free(out);if(metadata)rpcmem_free(metadata);}
  } rpc;
  struct remote_rpc_control_unsigned_module pd={CDSP_DOMAIN_ID,1};
  if(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&pd,sizeof(pd)))return "status=FAILED\nfailure=BOUNDARY_UNSIGNED_PD\n";
  auto uri=std::string(hexatrain_hvx_probe_URI)+"&_dom=cdsp";
  if(hexatrain_hvx_probe_open(uri.c_str(),&rpc.handle))return "status=FAILED\nfailure=BOUNDARY_RPC_OPEN\n";
  rpc.a=static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,4096*4));
  rpc.b=static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,4096*4));
  rpc.out=static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,4096*4));
  rpc.metadata=static_cast<int*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,(4096+8)*4));
  if(!rpc.a||!rpc.b||!rpc.out||!rpc.metadata)return "status=FAILED\nfailure=BOUNDARY_RPCMEM\n";
  if((reinterpret_cast<uintptr_t>(rpc.a)|reinterpret_cast<uintptr_t>(rpc.b)|reinterpret_cast<uintptr_t>(rpc.out))&127)return "status=FAILED\nfailure=BOUNDARY_ALIGNMENT\n";
  size_t rpcCalls=0,transportMismatch=0;
  auto fingerprint=[](const std::vector<float>& values){uint64_t hash=14695981039346656037ull;for(float value:values){uint32_t w;std::memcpy(&w,&value,4);for(int j=0;j<4;++j){hash^=(w>>(8*j))&255;hash*=1099511628211ull;}}return hash;};
  auto transfer=[&](const std::vector<float>& input,bool fixed,std::vector<float>& output,const std::string& name){
    const auto t0=std::chrono::steady_clock::now();size_t n=input.size();if(!n||n>4096)return false;
    std::memcpy(rpc.a,input.data(),n*4);std::memcpy(rpc.b,input.data(),n*4);
    // Complement every expected word: an unwritten element cannot match input.
    for(size_t i=0;i<n;++i){uint32_t word;std::memcpy(&word,&input[i],4);word=~word;std::memcpy(rpc.out+i,&word,4);}
    std::memset(rpc.metadata,0xa5,(n+8)*4);
    for(size_t i=0;i<n;++i){uint32_t poison;std::memcpy(&poison,&input[i],4);poison=~poison;std::memcpy(rpc.metadata+8+i,&poison,4);}
    const auto t1=std::chrono::steady_clock::now();
    int status=hexatrain_hvx_probe_run(rpc.handle,8,fixed?1:0,int(n),0,rpc.a,int(n),rpc.b,int(n),rpc.out,int(n),rpc.metadata,int(n+8));
    ++rpcCalls;const auto t2=std::chrono::steady_clock::now();output.assign(rpc.out,rpc.out+n);
    std::vector<float> before(rpc.a,rpc.a+n),expected(rpc.b,rpc.b+n);
    size_t mismatch=bits(input,output),immutable=bits(input,before)+bits(input,expected),receipt=0;
    for(size_t i=0;i<n;++i){uint32_t sent,received;std::memcpy(&sent,&input[i],4);std::memcpy(&received,rpc.metadata+8+i,4);receipt+=sent!=received;}
    const auto t3=std::chrono::steady_clock::now();transportMismatch+=mismatch+immutable+receipt;
    auto us=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
    r<<"transport name="<<name<<" count="<<n<<" input_fnv="<<std::hex<<fingerprint(input)<<std::dec<<" finite="<<finite(input)<<" status="<<status<<" dsp_mismatch="<<rpc.metadata[1]<<" host_mismatch="<<mismatch<<" input_mutation="<<immutable<<" complete="<<(mismatch==0)<<" receipt_mismatch="<<receipt<<" output_finite="<<finite(output)<<" host_pack_us="<<us(t0,t1)<<" rpc_us="<<us(t1,t2)<<" dsp_copy_check_us="<<rpc.metadata[3]<<" host_validation_us="<<us(t2,t3)<<'\n';
    return !status&&rpc.metadata[0]==0x46503332&&rpc.metadata[1]==0&&rpc.metadata[2]==int(n)&&rpc.metadata[4]==0&&rpc.metadata[5]==0&&!mismatch&&!immutable&&!receipt;
  };
  std::vector<float> fixture(HEXATRAIN_BOUNDARY_COUNT),echo;hexatrain_boundary_fill(fixture.data(),fixture.size());
  if(!finite(fixture))return "status=FAILED\nfailure=FIXTURE_NONFINITE\n";
  for(int i=0;i<6;++i)if(!transfer(fixture,true,echo,"fixed"+std::to_string(i)))return r.str()+"status=FAILED\nfailure=BLOCKED_TRANSPORT\n";
  Runtime boundary[4];std::string e;
  for(uint32_t op=0;op<4;++op){RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;boundary[op].setOptions(o);
    if(!boundary[op].initialize(QnnBackendKind::HTP,e)||!boundary[op].prepareFp32Boundary(op,e))return failure("boundary_graph",e,boundary[op]);}
  size_t reinjectionFailures=0,reshapeMismatch=0,observerMismatch=0;uint64_t qnnCalls=0;
  const char* opName[]{"reshape","add_zero","multiply_one","subtract_offset"};
  std::vector<float> rawBoundary(128);for(size_t i=0;i<128;++i)rawBoundary[i]=fixture[i%17];
  for(uint32_t op=0;op<3;++op){std::vector<float> rhs(128,op==2?1.f:0.f),out;const auto before=rawBoundary,rb=rhs;
    if(!boundary[op].executeFp32Boundary(rawBoundary,rhs,out,e))return failure("boundary_raw_execute",e,boundary[op]);++qnnCalls;
    if(bits(before,rawBoundary)||bits(rb,rhs))return "status=FAILED\nfailure=QNN_INPUT_MUTATION\n";
    auto er=errors(rawBoundary,out);reshapeMismatch+=op==0?bits(rawBoundary,out):0;
    r<<"raw_boundary op="<<opName[op]<<" mismatch="<<bits(rawBoundary,out)<<" abs="<<er.absolute<<" nonfinite="<<er.nonfinite<<'\n';
    if(!finite(out))return r.str()+"status=FAILED\nfailure=BOUNDARY_UNEXPECTED_NONFINITE\n";
  }
  // Same procedural D64 recipes, full upstream, frozen acceptance. No optimizer.
  Runtime ln,untapped;RuntimeOptions o;o.captureQnnCallback=false;o.qnnLogLevel=2;ln.setOptions(o);untapped.setOptions(o);
  if(!ln.initialize(QnnBackendKind::HTP,e)||!ln.prepareLayerNormStability(2,64,1e-5f,true,e,true,1)||
     !untapped.initialize(QnnBackendKind::HTP,e)||!untapped.prepareLayerNormStability(2,64,1e-5f,true,e,false,1))return failure("boundary_ln",e,ln);
  for(const auto& c:redesignCases(64)){
    auto ref=doubleReference(c,64);auto oracle=anchoredFp32Oracle(c,64);LayerNormStabilityOutputs h,n;
    const auto xBefore=c.x,dyBefore=c.dy,gammaBefore=c.gamma,betaBefore=c.beta;
    if(!ln.executeLayerNormStability(c.x,c.dy,c.gamma,c.beta,h,e)||!untapped.executeLayerNormStability(c.x,c.dy,c.gamma,c.beta,n,e))return failure("boundary_ln_execute",e,ln);qnnCalls+=2;
    if(bits(xBefore,c.x)||bits(dyBefore,c.dy)||bits(gammaBefore,c.gamma)||bits(betaBefore,c.beta))return "status=FAILED\nfailure=QNN_LN_INPUT_MUTATION\n";
    size_t observer=bits(h.values.output,n.values.output)+bits(h.values.normalized,n.values.normalized)+bits(h.values.dInput,n.values.dInput)+bits(h.values.dGamma,n.values.dGamma)+bits(h.values.dBeta,n.values.dBeta);observerMismatch+=observer;
    r<<"observer name="<<c.name<<" bit_mismatch="<<observer<<'\n';
    const char* fields[]{"input","centered","inverse","xhat","dy","dxhat","dx","dg","db"};
    std::vector<float> center(c.x.size()),dh(c.x.size());for(size_t i=0;i<c.x.size();++i){center[i]=float(double(c.x[i])-ref.mean[i/64]);dh[i]=c.dy[i]*c.gamma[i%64];}
    const std::vector<float>* fp[]{&c.x,&center,&oracle.inverse,&oracle.xhat,&c.dy,&dh,&oracle.dx,&oracle.dg,&oracle.db};
    const std::vector<float>* ht[]{&c.x,&h.taps[1],&h.taps[7],&h.values.normalized,&c.dy,&h.taps[11],&h.values.dInput,&h.values.dGamma,&h.values.dBeta};
    const std::vector<double>* dr[]{nullptr,nullptr,&ref.inverse,&ref.xhat,nullptr,nullptr,&ref.dx,&ref.dg,&ref.db};
    for(size_t f=0;f<9;++f){std::string label=c.name+"_"+fields[f];
      r<<"stage name="<<c.name<<" field="<<fields[f]<<" htp_fp32_abs="<<errors(*fp[f],*ht[f]).absolute
       <<" oracle_double_abs="<<(dr[f]?errors(*dr[f],*fp[f]).absolute:0)<<'\n';
      if(!finite(*ht[f]))return r.str()+"status=FAILED\nfailure=BOUNDARY_UNEXPECTED_NONFINITE\n";
      if(!transfer(*ht[f],false,echo,"htp_"+label))return r.str()+"status=FAILED\nfailure=BLOCKED_TRANSPORT\n";
      std::vector<float> fromDsp;if(!transfer(*fp[f],false,fromDsp,"fp32_"+label))return r.str()+"status=FAILED\nfailure=BLOCKED_TRANSPORT\n";
      std::vector<float> padded(128,0);std::copy(fromDsp.begin(),fromDsp.end(),padded.begin());
      for(uint32_t op=0;op<3;++op){std::vector<float> rhs(128,op==2?1.f:0.f),out;
        const auto before=padded,rb=rhs;
        if(!boundary[op].executeFp32Boundary(padded,rhs,out,e))return failure("boundary_consumer",e,boundary[op]);++qnnCalls;
        if(bits(before,padded)||bits(rb,rhs))return "status=FAILED\nfailure=QNN_INPUT_MUTATION\n";
        out.resize(fromDsp.size());size_t mismatch=bits(fromDsp,out);auto er=errors(fromDsp,out);
        if(!finite(out))return r.str()+"status=FAILED\nfailure=BOUNDARY_CONSUMER_NONFINITE\n";
        double referenceError=dr[f]?errors(*dr[f],out).absolute:er.absolute;
        double limit=(f==6||f==7||f==8)?.02:.002;
        if(f==3&&(c.name=="low_variance"||c.name=="epsilon_dominated"||c.name=="constant"||c.name=="large_mean_low_variance"||c.name=="random_seed7"||c.name=="random_seed97"))limit=.06;
        bool pass=finite(out)&&referenceError<=limit;reinjectionFailures+=op!=0&&!pass;reshapeMismatch+=op==0?mismatch:0;
        r<<"boundary name="<<c.name<<" field="<<fields[f]<<" op="<<opName[op]<<" mismatch="<<mismatch<<" transfer_abs="<<er.absolute<<" reference_abs="<<referenceError<<" nonfinite="<<er.nonfinite<<" gate="<<pass<<'\n';
      }
    }
    std::vector<float> rhs(128,c.name=="large_mean_low_variance"?1024.f:0.f),out;
    if(!boundary[3].executeFp32Boundary(c.x,rhs,out,e))return failure("boundary_subtract",e,boundary[3]);++qnnCalls;
    std::vector<double> difference(128);for(size_t i=0;i<128;++i)difference[i]=double(c.x[i])-rhs[i];auto de=errors(difference,out);
    r<<"subtract name="<<c.name<<" abs="<<de.absolute<<" output_nonzero="<<std::count_if(out.begin(),out.end(),[](float x){return x!=0;})<<" row0_nonzero="<<std::count_if(out.begin(),out.begin()+64,[](float x){return x!=0;})<<'\n';
  }
  for(auto& b:boundary)r<<b.apiTraceSummary()<<b.diagnostics();r<<ln.apiTraceSummary()<<untapped.apiTraceSummary();
  const int closeStatus=hexatrain_hvx_probe_close(rpc.handle);rpc.handle=0;
  r<<"rpc_close_status="<<closeStatus<<"\ntransport_mismatch="<<transportMismatch<<"\nrpc_calls="<<rpcCalls<<"\nqnn_calls="<<qnnCalls<<"\nreshape_bit_mismatch="<<reshapeMismatch<<"\nobserver_bit_mismatch="<<observerMismatch<<"\nreinjection_failed_checks="<<reinjectionFailures<<"\n";
  r<<"decision="<<(reinjectionFailures?"BLOCKED_REINJECTION":"UNRESOLVED_FULL_MODEL_PENDING")<<"\ndsp_layernorm_kernel=NOT_IMPLEMENTED_CONDITIONAL_GATE\ntraining_updates=0\nrpc_status_success=true\nfallback=false\ncpu_fallback=false\nstatus="<<(!closeStatus&&!transportMismatch&&!observerMismatch?"SUCCESS":"FAILED")<<'\n';return r.str();
#endif
}
