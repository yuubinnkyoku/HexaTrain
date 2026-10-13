#include "probe_support.h"
namespace ln=phonelm::qnn::lnvalidation;
ln::Case expand(ln::Case c,unsigned d,unsigned rows){auto x=c.x,dy=c.dy;c.x.resize(rows*d);c.dy.resize(rows*d);for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<d;++j){c.x[r*d+j]=x[(r%2)*d+j];c.dy[r*d+j]=dy[(r%2)*d+j]*(r%3==2?-1.f:1.f);}return c;}
#include "matmul_probe.h"
#include "custom_matmul_probe.h"
unsigned failures=0;
void metric(const char* lane,const std::string& recipe,unsigned d,unsigned rows,const char* field,
 const std::vector<double>& ref,const std::vector<float>& host,const std::vector<float>& actual,double limit){
 auto e=ln::errors(ref,actual),cpu=ln::errors(host,actual);bool pass=e.nonfinite==0&&e.absolute<=limit;
 failures+=!pass;std::printf("lnmetric,%s,%u,%u,%s,%s,abs=%.12g,rms=%.12g,relative=%.12g,host_abs=%.12g,finite=%zu,hash=%016llx,gate=%d\n",lane,d,rows,recipe.c_str(),field,e.absolute,e.rms,e.relative,cpu.absolute,actual.size()-e.nonfinite,(unsigned long long)hashWords(actual),pass);
}
void checkReceipt(const std::vector<uint32_t>& r,unsigned kind,const std::vector<std::vector<float>>& inputs,unsigned extra=0){
 if(r[0]!=0x48383100u+kind||!r[1]||r[2]!=inputs[0].size()||r[3]!=inputs.size()||r[5]!=81||r[6]!=32||r[7]){std::puts("RECEIPT_FAIL");std::exit(3);}
 unsigned at=8;for(auto& in:inputs)for(float f:in)if(r.at(at++)!=word(f)){std::puts("OPERAND_RECEIPT_FAIL");std::exit(3);}
 if(at+extra!=r.size()){std::puts("RECEIPT_SIZE_FAIL");std::exit(3);}std::printf("kernel_receipt,%u,%u,kernel_us=%u\n",kind,r[1],r[4]);
}
#include "ln2_matmul_probe.h"
int main(int argc,char** argv){
 if(argc!=2)return 1;std::string mode=argv[1];initializeProbe();
 if(mode=="mmDirect"||mode=="mmBoundary"){auto failed=mmprobe::run(mode=="mmBoundary");closeProbe();return failed?4:0;}
 if(mode=="mmLn2"){auto failed=ln2probe::run();closeProbe();return failed?4:0;}
 if(mode=="copyNative"){
  DiagnosticGraph g("gradient_native_copy_control");auto x=g.tensor("gradient",QNN_TENSOR_TYPE_APP_WRITE,{32,64});
  auto n=g.tensor("native",QNN_TENSOR_TYPE_NATIVE,{32,64}),y=g.tensor("readback",QNN_TENSOR_TYPE_APP_READ,{32,64});
  auto a=g.tensor("a_receipt",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6152},QNN_DATATYPE_INT_32),b=g.tensor("b_receipt",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6152},QNN_DATATYPE_INT_32);
  g.node("producer","Copy","HexaFp32",{x},{n,a});g.node("consumer","Copy","HexaFp32",{n},{y,b});g.finalize();
  for(auto c:ln::redesignCases(64)){
   c=expand(c,64,32);auto host=ln::anchoredFp32Oracle(c,64);auto gradient=host.dx;for(unsigned j=0;j<2048;++j)gradient[j]+=c.dy[j];
   auto out=g.run({{x,gradient}});unsigned bad=0;for(unsigned id:{a,b}){auto& q=out[id];if(q[0]!=0x48383100||!q[1]||q[2]!=2048||q[3]!=1){std::puts("RECEIPT_FAIL");return 3;}for(unsigned j=0;j<2048;++j)bad+=(q[8+j]!=word(gradient[j])||q[8+2048+j]||q[8+4096+j]);}
   for(unsigned j=0;j<2048;++j)bad+=out[y][j]!=word(gradient[j]);failures+=bad!=0;
   std::printf("copycontrol,%s,words=2048,bit_mismatch=%u,hash=%016llx\n",c.name.c_str(),bad,(unsigned long long)hashWords(decode(out[y])));
  }closeProbe();std::printf("summary,copyNative,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0\n",failures);return failures?4:0;
 }
 if(mode=="matmul"){unsigned failed=laneB();closeProbe();return failed?4:0;}
 bool backward=mode!="lnForward",residual=mode.find("Residual")!=std::string::npos,fused=mode=="lnResidualFused",tap=mode=="lnBackward"||mode=="lnResidualTap";
 std::vector<unsigned> dimensions=mode=="lnMatrix"?std::vector<unsigned>{8,16,32,64,128}:std::vector<unsigned>{64};
 for(unsigned rows:{2u,32u})for(unsigned d:dimensions){
  DiagnosticGraph g(mode+"d"+std::to_string(d)+"r"+std::to_string(rows));auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ,N=QNN_TENSOR_TYPE_NATIVE;
  auto x=g.tensor("x",W,{1,1,rows,d}),gamma=g.tensor("gamma",W,{1,1,1,d}),beta=g.tensor("beta",W,{1,1,1,d}),eps=g.tensor("eps",W,{1,1,1,1});
  unsigned n=rows*d;auto y=g.tensor("y",R,{1,1,rows,d}),h=g.tensor("xhat",!backward||tap?R:N,{1,1,rows,d}),inv=g.tensor("inverse",!backward||tap?R:N,{1,1,rows,1});
  auto receipt=g.tensor("receipt",R,{1,1,1,8+n+2*d+1+n+rows},QNN_DATATYPE_INT_32);
  g.node("forward","LnForward","HexaFp32",{x,gamma,beta,eps},{y,h,inv,receipt});
  unsigned dy=0,dx=0,dg=0,db=0,brec=0,add=0,res=0,arec=0;
  if(backward){dy=g.tensor("dy",W,{1,1,rows,d});dx=g.tensor("dx",residual&&!tap&&!fused?N:R,{1,1,rows,d});dg=g.tensor("dg",R,{1,1,1,d});db=g.tensor("db",R,{1,1,1,d});
   if(residual)add=g.tensor("residual",W,{1,1,rows,d});
   brec=g.tensor("brec",R,{1,1,1,8+2*n+d+rows+n+(fused?n:0)},QNN_DATATYPE_INT_32);
   if(fused)g.node("backward","LnResidual","HexaFp32",{dy,gamma,h,inv,add},{dx,dg,db,brec});
   else g.node("backward","LnBackward","HexaFp32",{dy,gamma,h,inv},{dx,dg,db,brec});
   if(residual&&!fused){res=g.tensor("result",R,{1,1,rows,d});arec=g.tensor("arec",R,{1,1,1,3*n+8},QNN_DATATYPE_INT_32);g.node("residual","Add","HexaFp32",{dx,add},{res,arec});}
  }g.finalize();
  unsigned before=failures;
  for(auto c:ln::redesignCases(d)){
   c=expand(c,d,rows);auto ref=ln::doubleReference(c,d),unused=ref;auto host=ln::anchoredFp32Oracle(c,d);
   std::map<unsigned,std::vector<float>> inputs={{x,c.x},{gamma,c.gamma},{beta,c.beta},{eps,{1e-5f}}};if(backward)inputs[dy]=c.dy;if(residual)inputs[add]=c.dy;
   auto out=g.run(inputs);checkReceipt(out[receipt],4,{c.x,c.gamma,c.beta,{1e-5f}},n+rows);
   unsigned pos=8+n+2*d+1;std::vector<uint32_t> hb(out[receipt].begin()+pos,out[receipt].begin()+pos+n),ib(out[receipt].begin()+pos+n,out[receipt].end());
   auto actualH=decode(hb),actualI=decode(ib);
   bool low=c.name=="low_variance"||c.name=="epsilon_dominated"||c.name=="constant"||c.name=="large_mean_low_variance";double limit=low?.06:.002;
   metric("forward",c.name,d,rows,"y",ref.y,host.y,decode(out[y]),limit);
   metric("forward",c.name,d,rows,"xhat",ref.xhat,host.xhat,actualH,limit);
   metric("forward",c.name,d,rows,"inverse",ref.inverse,host.inverse,actualI,.002);
   if(!backward||tap)if(out[h]!=hb||out[inv]!=ib){std::puts("FORWARD_STORE_FAIL");return 3;}
   if(backward){auto recv=std::vector<std::vector<float>>{c.dy,c.gamma,actualH,actualI};if(fused)recv.push_back(c.dy);checkReceipt(out[brec],fused?6:5,recv,n);
    unsigned at=8+2*n+d+rows+(fused?n:0);std::vector<uint32_t> dxBits(out[brec].begin()+at,out[brec].end());auto actualDx=decode(dxBits);
    metric("backward",c.name,d,rows,"dx",ref.dx,host.dx,actualDx,.02);
    metric("backward",c.name,d,rows,"dg",ref.dg,host.dg,decode(out[dg]),.02);metric("backward",c.name,d,rows,"db",ref.db,host.db,decode(out[db]),.02);
    if(!residual||tap)if(out[dx]!=dxBits){std::puts("BACKWARD_STORE_FAIL");return 3;}
    if(residual){auto expected=ref.dx;auto hostRes=host.dx;for(unsigned j=0;j<n;++j){expected[j]+=c.dy[j];hostRes[j]+=c.dy[j];}
     if(!fused){auto& a=out[arec];if(a[0]!=0x48383101||a[2]!=n||a[3]!=2){std::puts("RESIDUAL_RECEIPT_FAIL");return 3;}for(unsigned j=0;j<n;++j)if(a[8+j]!=dxBits[j]||a[8+n+j]!=word(c.dy[j])||a[8+2*n+j]){std::puts("NATIVE_PRECISION_FAIL");return 3;}std::printf("kernel_receipt,1,%u,kernel_us=%u\n",a[1],a[4]);}
     metric("residual",c.name,d,rows,"result",expected,hostRes,decode(out[fused?dx:res]),.02);
    }
   }
   std::printf("fixture,%u,%u,%s,%016llx\n",d,rows,c.name.c_str(),(unsigned long long)hashWords(c.dy,hashWords(c.beta,hashWords(c.gamma,hashWords(c.x)))));
  }
  if(failures==before){auto c=expand(ln::redesignCases(d)[2],d,rows);std::map<unsigned,std::vector<float>> inputs={{x,c.x},{gamma,c.gamma},{beta,c.beta},{eps,{1e-5f}}};if(backward)inputs[dy]=c.dy;if(residual)inputs[add]=c.dy;for(unsigned j=0;j<9;++j){auto out=g.run(inputs);std::printf("perf,%s,%u,%u,%u,%.3f\n",mode.c_str(),d,rows,j,g.executeUs);}}
 }
 struct rusage usage{};getrusage(RUSAGE_SELF,&usage);std::printf("peak_rss_kib,%ld\n",usage.ru_maxrss);closeProbe();std::printf("summary,%s,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0\n",mode.c_str(),failures);return failures?4:0;
}
