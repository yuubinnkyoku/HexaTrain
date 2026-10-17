#define main inherited_graph_main
#include "one_layer_graph_test.cpp"
#undef main
#include "softmax_projection_probe.h"
#include "probability_contract.h"
#include <iomanip>
#include <iostream>
// Host-only counterfactuals. All references use one:: analytic equations and
// LayerNorm references; no Custom arithmetic body is invoked.
namespace attribution {
using one::Vec;
double distance(const Vec& a,const Vec& b){double v=0;for(unsigned i=0;i<a.size();++i)v=std::max(v,std::fabs(a[i]-b[i]));return v;}
double magnitude(const Vec& a){double v=0;for(double x:a)v=std::max(v,std::fabs(x));return v;}
void array(const Vec& a){std::cout<<'[';for(unsigned i=0;i<a.size();++i){if(i)std::cout<<',';std::cout<<a[i];}std::cout<<']';}
struct Stats {double abs=0,rms=0,relative=0,rows=0;unsigned coordinate=0,nearZero=0,nearOne=0,zero=0;};
Stats stats(const Vec& p,const Vec& x){Stats s;double sq=0;for(unsigned r=0;r<32;++r){double sum=0;for(unsigned c=0;c<32;++c){unsigned i=r*32+c;double e=std::fabs(x[i]-p[i]);if(e>s.abs){s.abs=e;s.coordinate=i;}sq+=e*e;s.relative=std::max(s.relative,e/std::max(std::fabs(p[i]),1e-12));sum+=x[i];s.nearZero+=x[i]<1e-8;s.nearOne+=x[i]>1-1e-6;s.zero+=x[i]==0;if(!std::isfinite(x[i])||x[i]<0||(c>r&&x[i]!=0))throw std::runtime_error("causal/finite invariant");}s.rows=std::max(s.rows,std::fabs(sum-1));}s.rms=std::sqrt(sq/1024);return s;}
void softMath(unsigned fi,unsigned h,const Vec& scores){
 Vec expOnly(1024),sumRounded(1024),outputRounded(1024);auto ref=one::softmax(scores);
 for(unsigned r=0;r<32;++r){double mx=-1e300;for(unsigned c=0;c<32;++c)mx=std::max(mx,scores[r*32+c]);double sum=0;float sumf=0;Vec e(32);for(unsigned c=0;c<32;++c){e[c]=float(std::exp(scores[r*32+c]-mx));sum+=e[c];sumf=float(sumf+float(e[c]));}for(unsigned c=0;c<32;++c){expOnly[r*32+c]=e[c]/sum;sumRounded[r*32+c]=e[c]/double(sumf);outputRounded[r*32+c]=float(float(e[c])/sumf);}}
 if(distance(outputRounded,one::softmax(scores,true))!=0)throw std::runtime_error("Host Softmax decomposition");
 std::cout<<"{\"kind\":\"softmath\",\"forward\":"<<fi<<",\"head\":"<<h<<",\"exp_rounding_normalized_error\":"<<distance(ref,expOnly)<<",\"sum_rounding_increment\":"<<distance(expOnly,sumRounded)<<",\"output_rounding_increment\":"<<distance(sumRounded,outputRounded)<<",\"total_local_error\":"<<distance(ref,outputRounded)<<"}\n";
}
struct Impact {double dsSigned=0,dvSigned=0,dsFp=0,dvFp=0,dsBound=0,dvBound=0,t=0,absoluteT=0,firstTerm=0,secondTerm=0,eAt=0,pAt=0,dpAt=0,signedAt=0,cosine=0;unsigned coordinate=0;};
Impact impact(const Vec& dp,const Vec& dc,const Vec& p,const Vec& x){
 Impact b;auto old=probabilitycontract::impact(dp,dc,p,x);b.dsBound=old.first;b.dvBound=old.second;
 auto ds=probabilitycontract::backward(dp,p),signedDs=probabilitycontract::backward(dp,x),fpDs=probabilitycontract::backward(dp,x,true);
 auto dv=one::mm(p,dc,32,32,32,true),signedDv=one::mm(x,dc,32,32,32,true),fpDv=one::mm(x,dc,32,32,32,true,false,true);
 b.dsSigned=distance(ds,signedDs);b.dvSigned=distance(dv,signedDv);b.dsFp=distance(ds,fpDs);b.dvFp=distance(dv,fpDv);
 double independentDsBound=0,independentDvBound=0;
 for(unsigned r=0;r<32;++r){double a=0,t=0,u=0,e2=0,d2=0;for(unsigned c=0;c<32;++c){unsigned i=r*32+c;a+=dp[i]*p[i];t+=dp[i]*(x[i]-p[i]);u+=std::fabs(dp[i]*(x[i]-p[i]));e2+=(x[i]-p[i])*(x[i]-p[i]);d2+=dp[i]*dp[i];}
  for(unsigned c=0;c<32;++c){unsigned i=r*32+c;double e=x[i]-p[i],signedDelta=e*(dp[i]-a)-x[i]*t,term1=std::fabs(e)*std::fabs(dp[i]-a),term2=std::fabs(x[i])*u,bound=term1+term2;
   if(std::fabs(signedDelta-(signedDs[i]-ds[i]))>1e-10||std::fabs(signedDelta)>bound+1e-10)throw std::runtime_error("dScores bound identity");
   if(bound>independentDsBound){independentDsBound=bound;b.coordinate=i;b.t=t;b.absoluteT=u;b.firstTerm=term1;b.secondTerm=term2;b.eAt=e;b.pAt=p[i];b.dpAt=dp[i];b.signedAt=signedDelta;b.cosine=e2*d2>0?t/std::sqrt(e2*d2):0;}}
 }
 for(unsigned c=0;c<32;++c)for(unsigned k=0;k<32;++k){double signedDelta=0,bound=0;for(unsigned r=0;r<32;++r){double term=(x[r*32+c]-p[r*32+c])*dc[r*32+k];signedDelta+=term;bound+=std::fabs(term);}if(std::fabs(signedDelta-(signedDv[c*32+k]-dv[c*32+k]))>1e-10||std::fabs(signedDelta)>bound+1e-10)throw std::runtime_error("dValue bound identity");independentDvBound=std::max(independentDvBound,bound);}
 if(b.dsBound!=independentDsBound||b.dvBound!=independentDvBound)throw std::runtime_error("independent bound mismatch");return b;
}
void impactFields(const Impact& b){std::cout<<",\"ds_signed\":"<<b.dsSigned<<",\"dv_signed\":"<<b.dvSigned<<",\"ds_fp_actual\":"<<b.dsFp<<",\"dv_fp_actual\":"<<b.dvFp<<",\"ds_bound\":"<<b.dsBound<<",\"dv_bound\":"<<b.dvBound<<",\"bound_coordinate\":"<<b.coordinate<<",\"signed_t\":"<<b.t<<",\"absolute_t\":"<<b.absoluteT<<",\"bound_term1\":"<<b.firstTerm<<",\"bound_term2\":"<<b.secondTerm<<",\"e_at_bound\":"<<b.eAt<<",\"p_at_bound\":"<<b.pAt<<",\"dp_at_bound\":"<<b.dpAt<<",\"signed_at_bound\":"<<b.signedAt<<",\"dp_error_cosine\":"<<b.cosine;}
void details(unsigned fi,unsigned h,int gi,const char* predicate,double value,double gate,const Stats& s,const Vec& p,const Vec& x,const Vec& scores,const Vec& fpScores,const Vec& dp,const Vec& dc,const Impact& b){
 unsigned row=s.coordinate/32,col=s.coordinate%32;double top=-1e300,second=top,min=1e300;for(unsigned c=0;c<32;++c){double v=scores[row*32+c];min=std::min(min,v);if(v>top){second=top;top=v;}else second=std::max(second,v);}
 std::cout<<"{\"kind\":\"failure\",\"forward\":"<<fi<<",\"head\":"<<h<<",\"gradient\":"<<gi<<",\"predicate\":\""<<predicate<<"\",\"value\":"<<value<<",\"threshold\":"<<gate<<",\"excess\":"<<value-gate<<",\"p_coordinate\":"<<s.coordinate<<",\"row\":"<<row<<",\"column\":"<<col<<",\"mask_changes_critical\":"<<(col>row?"true":"false")<<",\"p_reference\":"<<p[s.coordinate]<<",\"p_actual\":"<<x[s.coordinate]<<",\"logit_min\":"<<min<<",\"logit_max\":"<<top<<",\"top2_gap\":"<<top-second<<",\"dp_max\":"<<magnitude(dp)<<",\"dc_max\":"<<magnitude(dc);impactFields(b);
 std::cout<<",\"row_p_reference\":";array(Vec(p.begin()+row*32,p.begin()+(row+1)*32));std::cout<<",\"row_p_actual\":";array(Vec(x.begin()+row*32,x.begin()+(row+1)*32));std::cout<<",\"row_logits_reference\":";array(Vec(scores.begin()+row*32,scores.begin()+(row+1)*32));std::cout<<",\"row_logits_fp\":";array(Vec(fpScores.begin()+row*32,fpScores.begin()+(row+1)*32));std::cout<<"}\n";
}
int run(const char* fixture){auto original=one::read(fixture),cs=segmentstudy::lnCases(original),gs=upstreamreference::fullUpstream(original);std::vector<std::map<std::string,Vec>> gr;for(auto& z:gs)gr.push_back(one::reference(z,true));unsigned failures=0,comparisons=0,matched=0;double pMax=0,rowsMax=0,dsMax=0,dvMax=0,dsActual=0,dvActual=0;
 for(auto& z:cs){auto c=one::lncase(z);c.x=z.v.at("layerinput");c.gamma=z.v.at("gamma1");c.beta=z.v.at("beta1");auto ln=mmfixture::ln::doubleReference(c,64);auto lf=mmfixture::ln::anchoredFp32Oracle(c,64);Vec q[2],f[2],lnOnly[2],qOnly[2];
  for(unsigned j=0;j<2;++j){auto w=one::d(z.v.at(j?"wk":"wq"));q[j]=one::mm(ln.y,w,32,64,64);f[j]=one::mm(one::d(lf.y),w,32,64,64,false,false,true);lnOnly[j]=one::mm(one::d(lf.y),w,32,64,64);qOnly[j]=one::mm(one::d(one::f(ln.y)),w,32,64,64,false,false,true);}
  for(unsigned h=0;h<2;++h){auto raw=one::mm(one::split(q[0],h),one::split(q[1],h),32,32,32,false,true),rf=one::mm(one::split(f[0],h),one::split(f[1],h),32,32,32,false,true,true);auto rd=one::mm(one::split(f[0],h),one::split(f[1],h),32,32,32,false,true);
   auto scale=[&](const Vec& a,bool fp=false){return one::mul(a,{double(probabilitycontract::scale())},fp);};auto end=[&](const Vec& a){return one::softmax(probabilitycontract::masked(scale(a)));};
   auto scaled=scale(raw),sf=scale(rf,true),masked=probabilitycontract::masked(scaled),mf=probabilitycontract::masked(sf,true);auto p=one::softmax(masked);auto rounded=one::d(one::f(masked));
   std::vector<Vec> logits={masked,rounded,probabilitycontract::masked(scale(rf)),probabilitycontract::masked(sf),mf,mf};
   std::vector<Vec> paths={p,one::softmax(rounded,true),end(rf),one::softmax(probabilitycontract::masked(sf)),one::softmax(mf),one::softmax(mf,true)};
   // Additional counterfactuals, distinct from the required R0..R5 definitions.
   auto lnP=end(one::mm(one::split(lnOnly[0],h),one::split(lnOnly[1],h),32,32,32,false,true));auto qP=end(one::mm(one::split(qOnly[0],h),one::split(qOnly[1],h),32,32,32,false,true));
   auto local=one::softmax(mf,true),localRef=one::softmax(mf);std::cout<<"{\"kind\":\"stage\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"checkpoint_parameters\":"<<z.checkpoint<<",\"ln_y_error\":"<<distance(ln.y,one::d(lf.y))<<",\"q_error\":"<<distance(q[0],f[0])<<",\"k_error\":"<<distance(q[1],f[1])<<",\"qk_cumulative_error\":"<<distance(raw,rf)<<",\"qk_local_error\":"<<distance(rd,rf)<<",\"scale_cumulative_error\":"<<distance(scaled,sf)<<",\"scale_local_error\":"<<distance(scale(rf),sf)<<",\"mask_cumulative_error\":"<<distance(masked,mf)<<",\"mask_local_error\":"<<distance(probabilitycontract::masked(sf),mf)<<",\"ln_only_p_error\":"<<distance(p,lnP)<<",\"qkv_only_p_error\":"<<distance(p,qP)<<",\"pre_qk_rounding_p_error\":"<<distance(p,end(rd))<<",\"input_quantization_p_error\":"<<distance(p,one::softmax(rounded))<<",\"local_softmax_p_error\":"<<distance(localRef,local)<<",\"r4_r5_error\":"<<distance(paths[4],paths[5])<<"}\n";
   softMath(z.index,h,mf);
   Stats full=stats(p,paths[5]);pMax=std::max(pMax,full.abs);rowsMax=std::max(rowsMax,full.rows);
   for(unsigned path=0;path<6;++path){auto s=stats(p,paths[path]);std::cout<<"{\"kind\":\"path\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"path\":"<<path<<",\"p_maxAbs\":"<<s.abs<<",\"p_rms\":"<<s.rms<<",\"p_relative_floor_1e12\":"<<s.relative<<",\"row_error\":"<<s.rows<<",\"coordinate\":"<<s.coordinate<<",\"near_zero\":"<<s.nearZero<<",\"near_one\":"<<s.nearOne<<",\"zero\":"<<s.zero<<"}\n";
    // Each row distribution remains private. It is never selected by the public exporter.
    for(unsigned r=0;r<32;++r){double mx=-1e300,second=mx;unsigned underflow=0;for(unsigned col=0;col<32;++col){double v=logits[path][r*32+col];if(v>mx){second=mx;mx=v;}else second=std::max(second,v);underflow+=col<=r&&paths[path][r*32+col]==0;}std::cout<<"{\"kind\":\"row\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"path\":"<<path<<",\"row\":"<<r<<",\"max_logit\":"<<mx<<",\"top2_gap\":"<<mx-second<<",\"allowed_underflow\":"<<underflow<<",\"probabilities\":";array(Vec(paths[path].begin()+r*32,paths[path].begin()+(r+1)*32));std::cout<<"}\n";}
    for(unsigned gi=0;gi<gr.size();++gi){auto key="h"+std::to_string(h)+"_";auto& dp=gr[gi].at(key+"dp");auto& dc=gr[gi].at(key+"dc");auto b=impact(dp,dc,p,paths[path]);std::cout<<"{\"kind\":\"impact\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"path\":"<<path<<",\"gradient\":"<<gi<<",\"provenance_same_fixture\":"<<(z.index<24&&z.index==gi?"true":"false");impactFields(b);std::cout<<"}\n";
     if(path==5){++comparisons;matched+=z.index<24&&z.index==gi;dsMax=std::max(dsMax,b.dsBound);dvMax=std::max(dvMax,b.dvBound);dsActual=std::max(dsActual,b.dsFp);dvActual=std::max(dvActual,b.dvFp);if(b.dsBound>probabilitycontract::dsGate/8){++failures;details(z.index,h,int(gi),"ds_bound",b.dsBound,probabilitycontract::dsGate/8,full,p,paths[5],masked,mf,dp,dc,b);}if(b.dvBound>probabilitycontract::dvGate/10){++failures;details(z.index,h,int(gi),"dv_bound",b.dvBound,probabilitycontract::dvGate/10,full,p,paths[5],masked,mf,dp,dc,b);}}
    }
   }
   if(full.abs>probabilitycontract::pGate){++failures;details(z.index,h,-1,"p_maxAbs",full.abs,probabilitycontract::pGate,full,p,paths[5],masked,mf,{}, {},{});}if(full.rows>probabilitycontract::rowGate){++failures;details(z.index,h,-1,"row_error",full.rows,probabilitycontract::rowGate,full,p,paths[5],masked,mf,{}, {},{});}
  }
 }
 std::cout<<"{\"kind\":\"summary\",\"cases\":"<<cs.size()<<",\"heads\":88,\"gradient_cases\":"<<gr.size()<<",\"cartesian_pairs\":"<<comparisons<<",\"same_fixture_pairs\":"<<matched<<",\"failures\":"<<failures<<",\"p_maxAbs\":"<<pMax<<",\"row_error\":"<<rowsMax<<",\"ds_bound\":"<<dsMax<<",\"dv_bound\":"<<dvMax<<",\"ds_actual\":"<<dsActual<<",\"dv_actual\":"<<dvActual<<",\"bound_checks\":\"PASS\",\"old_contract\":\"FAIL\",\"dsp_claim\":false}\n";
 return failures==9&&comparisons==2200&&matched==48&&std::fabs(pMax-1.00750624248969e-6)<1e-18&&std::fabs(dsMax-4.64227603470169e-4)<1e-15?0:4;
}
}
#ifndef PHONELM_ATTRIBUTION_LIBRARY
int main(int argc,char** argv){std::cout<<std::setprecision(17);if(argc!=2)return 1;return attribution::run(argv[1]);}
#endif
