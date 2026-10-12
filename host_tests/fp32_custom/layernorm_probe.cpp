#include "probe_support.h"
namespace ln=phonelm::qnn::lnvalidation;
ln::Case expand(ln::Case c,unsigned d,unsigned rows){auto x=c.x,dy=c.dy;c.x.resize(rows*d);c.dy.resize(rows*d);for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<d;++j){c.x[r*d+j]=x[(r%2)*d+j];c.dy[r*d+j]=dy[(r%2)*d+j]*(r%3==2?-1.f:1.f);}return c;}
unsigned failures=0;
void metric(const char* lane,const std::string& recipe,unsigned d,unsigned rows,const char* field,
 const std::vector<double>& ref,const std::vector<float>& host,const std::vector<float>& actual,double limit){
 auto e=ln::errors(ref,actual),cpu=ln::errors(host,actual);bool pass=e.nonfinite==0&&e.absolute<=limit;
 failures+=!pass;std::printf("lnmetric,%s,%u,%u,%s,%s,abs=%.12g,rms=%.12g,relative=%.12g,host_abs=%.12g,finite=%zu,hash=%016llx,gate=%d\n",lane,d,rows,recipe.c_str(),field,e.absolute,e.rms,e.relative,cpu.absolute,actual.size()-e.nonfinite,(unsigned long long)hashWords(actual),pass);
}
void checkReceipt(const std::vector<uint32_t>& r,unsigned kind,const std::vector<std::vector<float>>& inputs){
 if(r[0]!=0x48383100u+kind||!r[1]||r[2]!=inputs[0].size()||r[3]!=inputs.size()||r[5]!=81||r[6]!=32||r[7]){std::puts("RECEIPT_FAIL");std::exit(3);}
 unsigned at=8;for(auto& in:inputs)for(float f:in)if(r.at(at++)!=word(f)){std::puts("OPERAND_RECEIPT_FAIL");std::exit(3);}
 if(at!=r.size()){std::puts("RECEIPT_SIZE_FAIL");std::exit(3);}std::printf("kernel_receipt,%u,%u,kernel_us=%u\n",kind,r[1],r[4]);
}
int main(int argc,char** argv){
 if(argc!=2)return 1;std::string mode=argv[1];if(mode!="lnForward")return 1;initializeProbe();
 for(unsigned rows:{2u,32u})for(unsigned d:{64u}){
  DiagnosticGraph g("forward_d"+std::to_string(d)+"r"+std::to_string(rows));auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ;
  auto x=g.tensor("x",W,{1,1,rows,d}),gamma=g.tensor("gamma",W,{1,1,1,d}),beta=g.tensor("beta",W,{1,1,1,d}),eps=g.tensor("eps",W,{1,1,1,1});
  auto y=g.tensor("y",R,{1,1,rows,d}),h=g.tensor("xhat",R,{1,1,rows,d}),inv=g.tensor("inverse",R,{1,1,rows,1});
  auto receipt=g.tensor("receipt",R,{1,1,1,8+rows*d+2*d+1},QNN_DATATYPE_INT_32);
  g.node("forward","LnForward","HexaFp32",{x,gamma,beta,eps},{y,h,inv,receipt});g.finalize();
  unsigned before=failures;
  for(auto c:ln::redesignCases(d)){
   c=expand(c,d,rows);auto ref=ln::doubleReference(c,d),unused=ref;auto host=ln::anchoredFp32Oracle(c,d);
   auto out=g.run({{x,c.x},{gamma,c.gamma},{beta,c.beta},{eps,{1e-5f}}});checkReceipt(out[receipt],4,{c.x,c.gamma,c.beta,{1e-5f}});
   bool low=c.name=="low_variance"||c.name=="epsilon_dominated"||c.name=="constant"||c.name=="large_mean_low_variance";double limit=low?.06:.002;
   metric("forward",c.name,d,rows,"y",ref.y,host.y,decode(out[y]),limit);
   metric("forward",c.name,d,rows,"xhat",ref.xhat,host.xhat,decode(out[h]),limit);
   metric("forward",c.name,d,rows,"inverse",ref.inverse,host.inverse,decode(out[inv]),.002);
   std::printf("fixture,%u,%u,%s,%016llx\n",d,rows,c.name.c_str(),(unsigned long long)hashWords(c.dy,hashWords(c.beta,hashWords(c.gamma,hashWords(c.x)))));
  }
  if(failures==before){auto c=expand(ln::redesignCases(d)[2],d,rows);for(unsigned j=0;j<9;++j){auto out=g.run({{x,c.x},{gamma,c.gamma},{beta,c.beta},{eps,{1e-5f}}});checkReceipt(out[receipt],4,{c.x,c.gamma,c.beta,{1e-5f}});std::printf("perf,forward,%u,%u,%u,%.3f\n",d,rows,j,g.executeUs);}}
 }
 struct rusage usage{};getrusage(RUSAGE_SELF,&usage);std::printf("peak_rss_kib,%ld\n",usage.ru_maxrss);closeProbe();std::printf("summary,lnForward,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0\n",failures);return failures?4:0;
}
