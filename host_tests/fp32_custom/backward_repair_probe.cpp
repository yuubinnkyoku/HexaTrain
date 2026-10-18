#include "backward_repair_graph.h"
#define main unusedSavedPMain
#include "saved_p_backward_probe.cpp"
#undef main
namespace repairprobe {
using Words=savedp::Words;
void demand(bool v){if(!v)throw std::runtime_error("repair runtime contract");}
Words asWords(const std::vector<float>& v){Words w;for(float x:v)w.push_back(word(x));return w;}
unsigned differences(const Words& a,const Words& b){demand(a.size()==b.size());unsigned n=0;for(unsigned i=0;i<a.size();++i)n+=a[i]!=b[i];return n;}
std::vector<double> oracle(const std::vector<double>& p,const std::vector<float>& g,bool value){
 std::vector<double> y(1024);if(value){for(unsigned i=0;i<32;++i)for(unsigned j=0;j<32;++j){double s=0;for(unsigned t=0;t<32;++t)s=s+p[t*32+i]*double(g[t*32+j]);y[i*32+j]=s;}}
 else for(unsigned r=0;r<32;++r){double s=0;for(unsigned c=0;c<32;++c)s=s+p[r*32+c]*double(g[r*32+c]);for(unsigned c=0;c<32;++c)y[r*32+c]=p[r*32+c]*(double(g[r*32+c])-s);}return y;
}
std::map<unsigned,std::vector<float>> inputs(Graph& z,unsigned fi,unsigned gi,const std::vector<doubleqkfixture::Case>& cs,const std::vector<mixedfixture::Case>& ms,const std::vector<s2fixture::Case>& ls,const std::vector<savedp::Pair>& gs){
 std::map<unsigned,std::vector<float>> in;
 if(z.native){in[z.x]=ls[fi].x;in[z.ga]=ls[fi].ga;in[z.be]=ls[fi].be;in[z.dy]=ls[fi].dy;for(unsigned j=0;j<3;++j)in[z.w[j]]=ms[fi].w[j];}
 auto& grad=gs[fi<44?gi:25+gi];for(unsigned h=0;h<2;++h){if(!z.native)in[z.p[h]]=cs[fi].hostP[h];if(z.softmax)in[z.dp[h]]=grad[h][0];if(z.dvalue)in[z.dc[h]]=grad[h][1];}return in;
}
void record(unsigned code,unsigned h,const Words& words){unsigned hdr[]={code,h,unsigned(words.size())};savedp::writeRaw(hdr,3);savedp::writeRaw(words.data(),words.size());}
double pct(std::vector<double> x,double q){std::sort(x.begin(),x.end());return x[unsigned(q*(x.size()-1))];}
void initialize(){initializeDoubleQk();check("registerRepairHost",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairHost.so","HexaBackwardRepairInterfaceProvider","CPU"));check("registerRepairDSP",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairDsp.so","HexaBackwardRepairInterfaceProvider","HTP"));}
void performance(const std::vector<doubleqkfixture::Case>& cs,const std::vector<mixedfixture::Case>& ms,const std::vector<s2fixture::Case>& ls,const std::vector<savedp::Pair>& gs){
 for(bool obs:{true,false})for(bool tap:{true,false}){
  for(auto mode:{savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue,savedp::Mode::Forward,savedp::Mode::NativeSoftmax,savedp::Mode::NativeDValue,savedp::Mode::Dual}){
   Graph z(mode,obs,tap);auto in=inputs(z,19,9,cs,ms,ls,gs);std::vector<double> wall,math,receipt;double first=0;
   for(unsigned i=0;i<32;++i){auto out=z.g.run(in);double m=0,r=0;if(obs)for(unsigned h=0;h<2;++h){for(auto id:{z.sr[h],z.vr[h]})if(id!=~0u){auto& rec=out.at(id);m+=rec[4];r+=rec[7];}}
    if(!i)first=z.g.executeUs;else{wall.push_back(z.g.executeUs);if(obs){math.push_back(m);receipt.push_back(r);}}}
   unsigned native=0,host=0;for(unsigned id=0;id<z.g.tensors.size();++id){auto type=z.g.tensors[id].v1.type;if(type==QNN_TENSOR_TYPE_NATIVE)native+=z.g.size(id)*4;if(type==QNN_TENSOR_TYPE_APP_READ||type==QNN_TENSOR_TYPE_APP_WRITE)host+=z.g.size(id)*4;}
   std::printf("{\"kind\":\"repair_performance\",\"mode\":\"%s\",\"observed\":%d,\"tap\":%d,\"samples\":31,\"first_us\":%.3f,\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,\"finalize_us\":%.3f,\"nodes\":%u,\"tensors\":%zu,\"native_bytes\":%u,\"host_bytes\":%u,\"arithmetic_us\":",savedp::label(mode),obs,tap,first,pct(wall,.5),pct(wall,.1),pct(wall,.9),z.finalizeUs,z.g.nodes,z.g.tensors.size(),native,host);
   if(obs)std::printf("%.3f,\"receipt_inclusive_us\":%.3f}",pct(math,.5),pct(receipt,.5));else std::printf("null,\"receipt_inclusive_us\":null}");std::puts("");
  }
  for(auto mode:{savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue}){savedp::Graph z(mode,obs,tap);std::map<unsigned,std::vector<float>> in;for(unsigned h=0;h<2;++h){in[z.p[h]]=cs[19].hostP[h];if(z.softmax)in[z.dp[h]]=gs[9][h][0];if(z.dvalue)in[z.dc[h]]=gs[9][h][1];}std::vector<double> wall;double first=0;for(unsigned i=0;i<32;++i){z.g.run(in);if(!i)first=z.g.executeUs;else wall.push_back(z.g.executeUs);}std::printf("{\"kind\":\"standard_performance\",\"mode\":\"%s\",\"observed\":%d,\"tap\":%d,\"samples\":31,\"first_us\":%.3f,\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,\"finalize_us\":%.3f,\"nodes\":%u,\"tensors\":%zu,\"arithmetic_us\":null,\"receipt_inclusive_us\":null,\"baseline_accuracy\":\"FAIL_PRESERVED\"}\n",savedp::label(mode),obs,tap,first,pct(wall,.5),pct(wall,.1),pct(wall,.9),z.finalizeUs,z.g.nodes,z.g.tensors.size());}
 }
}
}
int main(int argc,char** argv){if(argc!=7)return 1;try{
 using namespace repairprobe;auto cs=doubleqkfixture::load(argv[1]);auto ms=mixedfixture::load(argv[2]);auto ls=s2fixture::load(argv[3]);auto gs=savedp::loadGradients(argv[4]);std::string requested=argv[5];unsigned limit=std::stoul(argv[6]);
 std::vector<savedp::Mode> modes={savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue,savedp::Mode::Forward,savedp::Mode::NativeSoftmax,savedp::Mode::NativeDValue,savedp::Mode::Dual};unsigned modeCode=0;for(;modeCode<modes.size()&&requested!=savedp::label(modes[modeCode]);++modeCode){}
 demand(modeCode<6||requested=="performance");if(modeCode>=3){FILE* permit=std::fopen("native-permit-private.json","rb");demand(permit);std::fclose(permit);}
 savedp::rawEvidence=std::fopen("saved-p-words.bin","wbx");demand(savedp::rawEvidence);unsigned hdr[]={0x42525001,1,modeCode,limit};savedp::writeRaw(hdr,4);initialize();
 if(requested=="performance"){performance(cs,ms,ls,gs);std::fclose(savedp::rawEvidence);closeProbe();std::puts("summary,repair_performance,records=32,samples=31,teardown=1");return 0;}
 auto mode=modes[modeCode];unsigned pairs=0,failures=0;std::map<std::string,Words> anchors,pAnchors;
 for(bool obs:{true,false})for(bool tap:{true,false}){Graph z(mode,obs,tap);
  for(unsigned fi=0;fi<std::min(limit,60u);++fi)for(unsigned gi=0;gi<(mode==savedp::Mode::Forward?1u:fi<44?25u:8u);++gi){auto in=inputs(z,fi,gi,cs,ms,ls,gs);auto out=z.g.run(in);
   for(unsigned h=0;h<2;++h){std::string pk=std::to_string(fi)+":"+std::to_string(h);Words pw;
    if(!z.native)pw=asWords(cs[fi].hostP[h]);else if(tap||mode==savedp::Mode::Forward)pw=out.at(z.p[h]);else if(obs){auto rid=z.softmax?z.sr[h]:z.vr[h];auto& r=out.at(rid);pw=Words(r.begin()+32,r.begin()+1056);}else pw=pAnchors.at(pk);
    if(obs&&tap){if(!gi)pAnchors[pk]=pw;else demand(pAnchors.at(pk)==pw);}else if(z.native)demand(pAnchors.at(pk)==pw);
    if(mode==savedp::Mode::Forward){record(1,h,pw);std::printf("{\"kind\":\"repair_forward\",\"observed\":%d,\"tap\":%d,\"forward\":%u,\"head\":%u,\"hash\":\"%016llx\"}\n",obs,tap,fi,h,(unsigned long long)hashWords(decode(pw)));continue;}
    for(unsigned value=0;value<2;++value){if(value?!z.dvalue:!z.softmax)continue;auto id=value?z.dv[h]:z.ds[h],rid=value?z.vr[h]:z.sr[h];auto& ow=out.at(id);auto output=decode(ow),p=decode(pw);auto& grad=gs[fi<44?gi:25+gi][h][value];auto gw=asWords(grad);
     Words product,dot,center;
     if(obs){auto& r=out.at(rid);demand(r.size()==(value?3104u:5184u)&&r[0]==0x48385000u+value&&r[1]==2048&&r[2]==1024&&r[3]==32&&r[5]==81&&r[6]==value);
      demand(Words(r.begin()+32,r.begin()+1056)==pw&&Words(r.begin()+1056,r.begin()+2080)==gw&&Words(r.begin()+2080,r.begin()+3104)==ow);
      if(!value){product=Words(r.begin()+3104,r.begin()+4128);dot=Words(r.begin()+4128,r.begin()+4160);center=Words(r.begin()+4160,r.end());if(tap)demand(product==out.at(z.product[h])&&dot==out.at(z.dot[h])&&center==out.at(z.centered[h]));}
     }else if(tap&&!value){product=out.at(z.product[h]);dot=out.at(z.dot[h]);center=out.at(z.centered[h]);}
     auto ref=value?phonelm::saved_p_backward::DValueOrderedFp32(p,grad):phonelm::saved_p_backward::SoftmaxBackwardOrderedFp32(p,grad);
     auto local=oracle(std::vector<double>(p.begin(),p.end()),grad,value);auto le=savedp::error(output,local);double total=z.native?0:le.max,pinduced=0;
     if(z.native)for(unsigned ri=0;ri<2;++ri){auto ideal=oracle(cs[fi].idealP[ri][h],grad,value);total=std::max(total,savedp::error(output,ideal).max);std::vector<float> unused;for(unsigned k=0;k<1024;++k)pinduced=std::max(pinduced,std::fabs(local[k]-ideal[k]));}
     auto key=std::to_string(fi)+":"+std::to_string(gi)+":"+std::to_string(h)+":"+std::to_string(value);unsigned mismatch=differences(ow,asWords(ref)),variant=0;if(obs&&tap)anchors[key]=ow;else variant=differences(ow,anchors.at(key));
     bool pass=total<=(value?.01:.002)&&!mismatch&&!variant;failures+=!pass;++pairs;
     std::printf("{\"kind\":\"repair_pair\",\"mode\":\"%s\",\"observed\":%d,\"tap\":%d,\"forward\":%u,\"gradient\":%u,\"head\":%u,\"consumer\":%u,\"local_maxAbs\":%.17g,\"total_maxAbs\":%.17g,\"p_induced_actual_maxAbs\":%.17g,\"word_mismatches\":%u,\"observer_mismatches\":%u,\"output_hash\":\"%016llx\",\"p_visible\":%s,\"pass\":%s}\n",requested.c_str(),obs,tap,fi,gi,h,value,le.max,total,pinduced,mismatch,variant,(unsigned long long)hashWords(output),(!z.native||obs||tap)?"true":"false",pass?"true":"false");
     unsigned rawKey[]={unsigned(obs),unsigned(tap),fi,gi,h,value,3u+unsigned(!product.empty())*3u+unsigned(obs)};savedp::writeRaw(rawKey,7);record(1,h,pw);record(2,h,gw);record(3,h,ow);if(!product.empty()){record(4,h,product);record(5,h,dot);record(6,h,center);}if(obs)record(7,h,out.at(rid));
    }
   }
  }
 }
 std::fclose(savedp::rawEvidence);closeProbe();std::printf("summary,backward_repair,mode=%s,pairs=%u,required_failures=%u,input_mutation=0,finite=1,complete_writes=1,poison_remaining=0,canary=1,teardown=1,cpu_arithmetic_fallback=0\n",requested.c_str(),pairs,failures);return failures?4:0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 5;}}
