// Host counterfactual laboratory. No runtime graph or Custom kernel is changed.
#define PHONELM_ATTRIBUTION_LIBRARY
#include "probability_attribution_test.cpp"
#include "forward_softmax_reference.h"
#include <set>
namespace precision {
using one::Vec;
bool fp32Shift=false; // Separate Host model; never a DSP exp word oracle.
using Case=mmfixture::ln::Case;
double rounded(double x){return double(float(x));}
Vec stored(Vec a){for(auto& v:a)v=rounded(v);return a;}
struct Kahan {float s=0,c=0;void add(float p){float y=p-c,t=s+y;c=(t-s)-y;s=t;}};
// Different arithmetic contracts: 0 ordered FP32; 1 double sum of FP32
// products; 2 double products/sum; 3 Kahan; 4 Neumaier, FP32 products.
Vec matmul(const Vec& a,const Vec& b,unsigned m,unsigned n,unsigned k,bool nt,int mode,bool keep=false){
 Vec out(m*n);for(unsigned r=0;r<m;++r)for(unsigned j=0;j<n;++j){double d=0;float s=0,c=0;Kahan kh;
  for(unsigned q=0;q<k;++q){double x=a[r*k+q],y=b[nt?j*k+q:q*n+j];float p=float(x)*float(y);
   if(mode==2)d+=x*y;else if(mode==1)d+=double(p);else if(mode==3)kh.add(p);
   else if(mode==4){float t=s+p;float z=std::fabs(s)>=std::fabs(p)?(s-t)+p:(p-t)+s;c=c+z;s=t;}
   else s=s+p;
  }double v=mode==2||mode==1?d:mode==3?double(kh.s):mode==4?double(s+c):double(s);
  out[r*n+j]=keep?v:rounded(v);
 }return out;
}
struct Ln {Vec y,h,inv;};
// Full double reference is independent two-pass mean/variance. Source epsilon
// is promoted from its actual float word. Legacy reference is also retained.
Ln idealLn(const Case& c,bool legacy=false){auto x=mmfixture::ln::reference(one::d(c.x),one::d(c.dy),one::d(c.gamma),one::d(c.beta),64,legacy?1e-5:double(1e-5f));return {x.y,x.xhat,x.inverse};}
// mode 1 full double, 2 reduction only, 3 normalization only, 4 affine only.
// Partial paths retain anchor-relative centering and Kahan reduction topology.
Ln layernorm(const Case& c,int mode,bool keep=false){
 auto f=mmfixture::ln::anchoredFp32Oracle(c,64);if(!mode)return {one::d(f.y),one::d(f.xhat),one::d(f.inverse)};
 if(mode==1){auto o=idealLn(c);if(!keep){o.y=stored(o.y);o.h=stored(o.h);o.inv=stored(o.inv);}return o;}
 Ln out;out.y.resize(2048);out.h.resize(2048);out.inv.resize(32);
 for(unsigned r=0;r<32;++r){Vec center(64);double inv=0;
  if(mode==2){double anchor=c.x[r*64],delta=0;for(unsigned j=0;j<64;++j)delta+=double(c.x[r*64+j])-anchor;delta/=64;double vmax=0;for(unsigned j=0;j<64;++j){center[j]=(double(c.x[r*64+j])-anchor)-delta;vmax=std::max(vmax,std::fabs(center[j]));}double sum=0;if(vmax>0)for(double v:center){double z=v/vmax;sum+=z*z;}float rms=float(vmax*std::sqrt(sum/64));inv=double(1.f/std::hypot(rms,std::sqrt(1e-5f)));center=stored(center);}
  else {float anchor=c.x[r*64];Kahan mean;for(unsigned j=0;j<64;++j)mean.add(c.x[r*64+j]-anchor);float delta=mean.s/64.f,amax=0;for(unsigned j=0;j<64;++j){center[j]=double((c.x[r*64+j]-anchor)-delta);amax=std::max(amax,std::fabs(float(center[j])));}Kahan sq;if(amax>0)for(double v:center){float q=float(v)/amax;sq.add(q*q);}float rms=amax*std::sqrt(sq.s/64.f);
   inv=mode==3?1/std::hypot(double(rms),std::sqrt(double(1e-5f))):double(f.inverse[r]);
  }
  out.inv[r]=rounded(inv);for(unsigned j=0;j<64;++j){unsigned i=r*64+j;out.h[i]=rounded(mode==3?center[j]*inv:double(float(center[j])*float(inv)));out.y[i]=mode==4?rounded(double(c.gamma[j])*out.h[i]+c.beta[j]):double(float(c.gamma[j])*float(out.h[i])+c.beta[j]);}
 }return out;
}
struct Candidate {std::string id;int ln=0,proj=0,qk=0;bool keepY=false,keepQ=false,keepRaw=false;};
std::vector<Candidate> candidates(){std::vector<Candidate> v;unsigned bits[]={0,1,2,4,3,5,6,7};for(unsigned i=0;i<8;++i){unsigned b=bits[i];v.push_back({"C"+std::to_string(i),b&1?1:0,b&2?2:0,b&4?2:0});}
 v.push_back({"LN_reduce",2});v.push_back({"LN_norm",3});v.push_back({"LN_affine",4});
 v.push_back({"QK_sum64",0,0,1});v.push_back({"QK_Kahan",0,0,3});v.push_back({"QK_Neumaier",0,0,4});v.push_back({"LN_QK_Kahan",1,0,3});
 v.push_back({"Q_sum64",0,1});v.push_back({"Q_Kahan",0,3});v.push_back({"Q_Neumaier",0,4});v.push_back({"Q_QK_Kahan",0,3,3});
 v.push_back({"B1",1,2,2,true,false,false});v.push_back({"B2",1,2,2,false,true,false});v.push_back({"B3",1,2,2,true,true,false});v.push_back({"B4",1,2,2,true,true,true});return v;
}
struct Forward {Ln ln;Vec q,k;std::array<Vec,2> raw,p;};
Forward forward(const Case& c,const Vec& wq,const Vec& wk,const Candidate& a){Forward f;f.ln=layernorm(c,a.ln,a.keepY);f.q=matmul(f.ln.y,wq,32,64,64,false,a.proj,a.keepQ);f.k=matmul(f.ln.y,wk,32,64,64,false,a.proj,a.keepQ);
 for(unsigned h=0;h<2;++h){f.raw[h]=matmul(one::split(f.q,h),one::split(f.k,h),32,32,32,true,a.qk,a.keepRaw);Vec s=f.raw[h];for(auto& v:s)v=rounded(v*double(probabilitycontract::scale()));s=probabilitycontract::masked(s,true);f.p[h]=fp32Shift?one::d(forwardsoftmaxreference::evaluate(one::f(s)).p):one::softmax(s,true);}return f;
}
std::array<Vec,2> ideal(const Case& c,const Vec& wq,const Vec& wk,bool legacy=false){auto l=idealLn(c,legacy);auto q=one::mm(l.y,wq,32,64,64),k=one::mm(l.y,wk,32,64,64);std::array<Vec,2> out;for(unsigned h=0;h<2;++h)out[h]=one::softmax(probabilitycontract::masked(one::mul(one::mm(one::split(q,h),one::split(k,h),32,32,32,false,true),{double(probabilitycontract::scale())})));return out;}
struct Gradient {Vec dp,dc;};
struct Totals {double p=0,row=0,ds=0,dv=0,dsSigned=0,dvSigned=0,dsActual=0,dvActual=0;unsigned pFail=0,rowFail=0,dsFail=0,dvFail=0,actualFail=0,pairs=0;std::set<std::pair<unsigned,unsigned>> failed;
 unsigned failures()const{return pFail+rowFail+dsFail+dvFail+actualFail;}
 void probability(unsigned fi,unsigned h,const attribution::Stats& s){p=std::max(p,s.abs);row=std::max(row,s.rows);pFail+=s.abs>1e-6;rowFail+=s.rows>2e-6;if(s.abs>1e-6||s.rows>2e-6)failed.insert({fi,h});}
 void add(unsigned fi,unsigned h,const attribution::Impact& b){++pairs;ds=std::max(ds,b.dsBound);dv=std::max(dv,b.dvBound);dsSigned=std::max(dsSigned,b.dsSigned);dvSigned=std::max(dvSigned,b.dvSigned);dsActual=std::max(dsActual,b.dsFp);dvActual=std::max(dvActual,b.dvFp);dsFail+=b.dsBound>.00025;dvFail+=b.dvBound>.001;actualFail+=b.dsFp>.002;actualFail+=b.dvFp>.01;if(b.dsBound>.00025||b.dvBound>.001||b.dsFp>.002||b.dvFp>.01)failed.insert({fi,h});}
 void print(const std::string& id,const char* set,const char* ref)const {std::cout<<"{\"kind\":\"summary\",\"candidate\":\""<<id<<"\",\"set\":\""<<set<<"\",\"reference\":\""<<ref<<"\",\"pairs\":"<<pairs<<",\"p_maxAbs\":"<<p<<",\"row_error\":"<<row<<",\"ds_bound\":"<<ds<<",\"dv_bound\":"<<dv<<",\"ds_signed\":"<<dsSigned<<",\"dv_signed\":"<<dvSigned<<",\"ds_actual\":"<<dsActual<<",\"dv_actual\":"<<dvActual<<",\"p_fail\":"<<pFail<<",\"row_fail\":"<<rowFail<<",\"ds_fail\":"<<dsFail<<",\"dv_fail\":"<<dvFail<<",\"actual_fail\":"<<actualFail<<",\"failures\":"<<failures()<<",\"unique_failed_pairs\":"<<failed.size()<<",\"joint_pass\":"<<(failures()?"false":"true")<<"}\n";}
};
float next(uint32_t& s){s=1664525u*s+1013904223u;return float(int((s>>8)%20001)-10000)/10000.f;}
std::vector<one::Fixture> synthetic(){std::vector<one::Fixture> out;uint32_t seeds[]={20903,49937,81229,104729,130363,155921,181081,205019,230003,255043,280001,305093,330017,355009,380041,405023};float scales[]={.1f,.1f,.1f,.1f,.0001f,.2f,.05f,.1f};for(unsigned ci=0;ci<16;++ci){one::Fixture z;z.index=ci;unsigned p=ci%8;uint32_t s=seeds[ci];for(auto field:{"layerinput","dy","gamma1","beta1","wq","wk"}){unsigned count=std::string(field)=="gamma1"||std::string(field)=="beta1"?64:std::string(field)=="wq"||std::string(field)=="wk"?4096:2048;auto& v=z.v[field];for(unsigned i=0;i<count;++i){float x=next(s);if(std::string(field)=="layerinput"){if(p==1)x=.25f+1e-4f*x;if(p==2)x=1024.f+.01f*x;if(p==3)x=(i%2?1.f:-1.f)+.01f*x;if(p==6)x*=.001f;if(p==7)x*=32.f;}else if(std::string(field)=="gamma1")x=1.f+.25f*x;else if(std::string(field)=="beta1")x*=.03125f;else if(std::string(field)=="wq"||std::string(field)=="wk")x*=scales[p];v.push_back(x);}}z.v["x"]=z.v["layerinput"];z.v["gamma"]=z.v["gamma1"];z.v["beta"]=z.v["beta1"];out.push_back(z);}return out;}
std::vector<std::array<Gradient,2>> syntheticGradients(){std::vector<std::array<Gradient,2>> v;uint32_t seeds[]={7717,12919,17891,22993,27997,33013,37997,43003};float dpScale[]={.25f,2.f,16.f,64.f};for(unsigned i=0;i<8;++i){uint32_t s=seeds[i];std::array<Gradient,2> z;for(auto& h:z){for(unsigned k=0;k<1024;++k)h.dp.push_back(next(s)*dpScale[i%4]);for(unsigned k=0;k<1024;++k)h.dc.push_back(next(s)*(i%2?4.f:.25f));}v.push_back(z);}return v;}
// Float-saved-state replay is independent of the Forward LN implementation.
void savedAudit(const Case& c,const Ln& f,const Ln& ref,unsigned fi,const std::string& id,const char* set){auto h=stored(f.h),inv=stored(f.inv);auto dx=mmfixture::ln::replayDx(c,64,h,inv),dr=mmfixture::ln::replayDx(c,64,ref.h,ref.inv);Vec dg(64),dgRef(64),db(64);double affine=0;for(unsigned i=0;i<2048;++i){unsigned j=i%64;dg[j]+=double(c.dy[i])*h[i];dgRef[j]+=double(c.dy[i])*ref.h[i];db[j]+=c.dy[i];affine=std::max(affine,std::fabs(f.y[i]-double(float(c.gamma[j])*float(h[i])+c.beta[j])));}double ex=attribution::distance(dx,dr),eg=attribution::distance(dg,dgRef);std::cout<<"{\"kind\":\"saved_ln\",\"candidate\":\""<<id<<"\",\"set\":\""<<set<<"\",\"forward\":"<<fi<<",\"y\":"<<attribution::distance(f.y,ref.y)<<",\"xhat\":"<<attribution::distance(h,ref.h)<<",\"inverse\":"<<attribution::distance(inv,ref.inv)<<",\"dx\":"<<ex<<",\"dg\":"<<eg<<",\"db\":0,\"saved_affine_replay_delta\":"<<affine<<",\"representation_backward_gate\":"<<(ex<=.02&&eg<=.02?"true":"false")<<"}\n";}
void evaluate(const std::vector<one::Fixture>& fs,const std::vector<std::array<Gradient,2>>& gs,const char* set){for(auto& a:candidates()){Totals total[2],matched;double referenceDelta=0,savedQ=0,rawLocal=0;
 for(auto& z:fs){auto c=one::lncase(z);c.x=z.v.at("layerinput");c.gamma=z.v.at("gamma1");c.beta=z.v.at("beta1");auto wq=one::d(z.v.at("wq")),wk=one::d(z.v.at("wk"));auto refs=ideal(c,wq,wk),legacy=ideal(c,wq,wk,true);auto f=forward(c,wq,wk,a);savedAudit(c,f.ln,idealLn(c),z.index,a.id,set);savedQ=std::max(savedQ,std::max(attribution::distance(f.q,stored(f.q)),attribution::distance(f.k,stored(f.k))));
 for(unsigned h=0;h<2;++h){auto local=one::mm(one::split(f.q,h),one::split(f.k,h),32,32,32,false,true);rawLocal=std::max(rawLocal,attribution::distance(local,f.raw[h]));referenceDelta=std::max(referenceDelta,attribution::distance(refs[h],legacy[h]));
 for(unsigned ri=0;ri<2;++ri){auto& p=ri?legacy[h]:refs[h];auto st=attribution::stats(p,f.p[h]);total[ri].probability(z.index,h,st);bool sameSet=std::string(set)=="frozen"&&z.index<24;
 std::cout<<"{\"kind\":\"case\",\"candidate\":\""<<a.id<<"\",\"set\":\""<<set<<"\",\"reference\":\""<<(ri?"legacy":"source")<<"\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"p_maxAbs\":"<<st.abs<<",\"row_error\":"<<st.rows<<",\"coordinate\":"<<st.coordinate<<",\"rms\":"<<st.rms<<"}\n";
 for(unsigned gi=0;gi<gs.size();++gi){auto b=attribution::impact(gs[gi][h].dp,gs[gi][h].dc,p,f.p[h]);total[ri].add(z.index,h,b);if(!ri&&sameSet&&gi==z.index){matched.probability(z.index,h,st);matched.add(z.index,h,b);}std::cout<<"{\"kind\":\"pair\",\"candidate\":\""<<a.id<<"\",\"set\":\""<<set<<"\",\"reference\":\""<<(ri?"legacy":"source")<<"\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"gradient\":"<<gi;attribution::impactFields(b);std::cout<<"}\n";}
 }
 }
 }total[0].print(a.id,set,"source");total[1].print(a.id,set,"legacy");if(std::string(set)=="frozen")matched.print(a.id,"same_frozen","source");std::cout<<"{\"kind\":\"boundary\",\"candidate\":\""<<a.id<<"\",\"set\":\""<<set<<"\",\"reference_delta_p\":"<<referenceDelta<<",\"saved_q_rounding\":"<<savedQ<<",\"qk_local_error\":"<<rawLocal<<",\"keep_y\":"<<a.keepY<<",\"keep_q\":"<<a.keepQ<<",\"keep_raw\":"<<a.keepRaw<<"}\n";}
}
void checks(){Vec a={1e8,1,-1e8,3},b(4,1);auto f=matmul(a,b,1,1,4,false,0),d=matmul(a,b,1,1,4,false,2),n=matmul(a,b,1,1,4,false,4);if(f[0]!=3||d[0]!=4||n[0]!=4)throw std::runtime_error("independent cancellation control");for(int mode=0;mode<=4;++mode){Vec x={1,2,3,4,5,6},y={7,8,9,10,11,12};if(matmul(x,y,2,2,3,true,mode)!=Vec({50,68,122,167}))throw std::runtime_error("NT indexing");}uint32_t sb;float s=probabilitycontract::scale();std::memcpy(&sb,&s,4);if(sb!=0x3e3504f3)throw std::runtime_error("scale");}
int run(const char* path){checks();auto original=one::read(path),fs=segmentstudy::lnCases(original),gr=upstreamreference::fullUpstream(original);std::vector<std::array<Gradient,2>> gs;for(auto& z:gr){auto g=one::reference(z,true);std::array<Gradient,2> v;for(unsigned h=0;h<2;++h){auto k="h"+std::to_string(h)+"_";v[h]={g.at(k+"dp"),g.at(k+"dc")};}gs.push_back(v);}if(fs.size()!=44||gs.size()!=25)throw std::runtime_error("coverage");
 evaluate(fs,gs,"frozen");evaluate(synthetic(),syntheticGradients(),"holdout");std::cout<<"{\"kind\":\"completed\",\"candidates\":"<<candidates().size()<<",\"frozen_pairs_per_candidate_per_reference\":2200,\"holdout_pairs_per_candidate_per_reference\":256,\"same_frozen_pairs\":48,\"actual_training_matched\":false,\"dsp_execution\":false}\n";return 0;}
}
#ifndef PHONELM_PRECISION_LIBRARY
int main(int argc,char** argv){std::cout<<std::setprecision(17);try{if(argc!=2&&argc!=3)return 1;if(argc==3){if(std::string(argv[2])!="--fp32-shift")return 1;precision::fp32Shift=true;}return precision::run(argv[1]);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}}
#endif
