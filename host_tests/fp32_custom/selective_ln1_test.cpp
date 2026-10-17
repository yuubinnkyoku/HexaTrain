#define PHONELM_PRECISION_LIBRARY
#include "upstream_precision_test.cpp"
#include "selective_ln1_arithmetic.h"
#include <fstream>
namespace selective {
using precision::Vec;
using precision::Case;
unsigned masks[]={0,1,2,4,3,5,6,7};
precision::Ln layer(const Case& c,unsigned mask){std::vector<float> y(2048),h(2048),inv(32);
 switch(mask){
#define SELECT(B) case B:ln<B>(c.x.data(),c.gamma.data(),c.beta.data(),y.data(),h.data(),inv.data());break;
 SELECT(0) SELECT(1) SELECT(2) SELECT(3) SELECT(4) SELECT(5) SELECT(6) SELECT(7)
#undef SELECT
 default:throw std::runtime_error("mask");}
 return {one::d(y),one::d(h),one::d(inv)};
}
precision::Forward forward(const Case& c,const Vec& wq,const Vec& wk,unsigned mask){precision::Forward f;f.ln=layer(c,mask);f.q=precision::matmul(f.ln.y,wq,32,64,64,false,2);f.k=precision::matmul(f.ln.y,wk,32,64,64,false,2);
 for(unsigned h=0;h<2;++h){f.raw[h]=precision::matmul(one::split(f.q,h),one::split(f.k,h),32,32,32,true,2);auto s=f.raw[h];for(auto& v:s)v=precision::rounded(v*double(probabilitycontract::scale()));s=probabilitycontract::masked(s,true);f.p[h]=precision::fp32Shift?one::d(forwardsoftmaxreference::evaluate(one::f(s)).p):one::softmax(s,true);}return f;
}
std::vector<one::Fixture> confirmation(){uint32_t seeds[]={1000003,1010033,1020089,1030127,1040167,1050209,1060253,1070299,1080341,1090391,1100447,1110499,1120543,1130597,1140647,1150697};float ws[]={.12f,.12f,.12f,.12f,.002f,.25f,.12f,.12f};std::vector<one::Fixture> out;
 for(unsigned ci=0;ci<16;++ci){one::Fixture z;z.index=ci;unsigned p=ci%8;uint32_t s=seeds[ci];for(auto field:{"layerinput","dy","gamma1","beta1","wq","wk"}){std::string k=field;unsigned n=k=="gamma1"||k=="beta1"?64:k=="wq"||k=="wk"?4096:2048;for(unsigned i=0;i<n;++i){float u=precision::next(s),v=u;if(k=="layerinput"){if(p==1)v=.5f+1e-5f*u;if(p==2)v=4096.f+.0625f*u;if(p==3)v=(i%2?4.f:-4.f)+.003f*u;if(p==6)v=64.f*u;}else if(k=="gamma1")v=1.f+.375f*u;else if(k=="beta1")v=.0625f*u;else if(k=="wq"||k=="wk")v=u*(p==7?(i%64<32?.07f:.18f):ws[p]);z.v[k].push_back(v);}}z.v["x"]=z.v["layerinput"];z.v["gamma"]=z.v["gamma1"];z.v["beta"]=z.v["beta1"];out.push_back(z);}return out;
}
std::vector<std::array<precision::Gradient,2>> confirmGradients(){uint32_t seeds[]={2000003,2010041,2020069,2030113,2040151,2050189,2060231,2070263};float ds[]={4,32,256,512};std::vector<std::array<precision::Gradient,2>> out;for(unsigned i=0;i<8;++i){uint32_t s=seeds[i];std::array<precision::Gradient,2> g;for(auto& h:g){for(unsigned j=0;j<1024;++j)h.dp.push_back(precision::next(s)*ds[i%4]);for(unsigned j=0;j<1024;++j)h.dc.push_back(precision::next(s)*(i%2?16.f:2.f));}out.push_back(g);}return out;}
unsigned words(const Vec& a,const Vec& b){unsigned n=0;for(unsigned i=0;i<a.size();++i)n+=forwardsoftmaxreference::bits(float(a[i]))!=forwardsoftmaxreference::bits(float(b[i]));return n;}
void evaluate(const std::vector<one::Fixture>& fs,const std::vector<std::array<precision::Gradient,2>>& gs,const char* set,const std::vector<unsigned>& ids){
 for(unsigned id:ids){unsigned mask=masks[id];std::string name="S"+std::to_string(id);precision::Totals total[2],matched[2];unsigned anchorWords=0,replayWords=0;double qkErr=0;unsigned qkPairs=0;
  for(auto& z:fs){auto c=one::lncase(z);c.x=z.v.at("layerinput");c.gamma=z.v.at("gamma1");c.beta=z.v.at("beta1");auto wq=one::d(z.v.at("wq")),wk=one::d(z.v.at("wk"));auto f=forward(c,wq,wk,mask);auto ref=precision::ideal(c,wq,wk),legacy=precision::ideal(c,wq,wk,true);
   precision::savedAudit(c,f.ln,precision::idealLn(c),z.index,name,set);
   for(unsigned i=0;i<2048;++i){float p=c.gamma[i%64]*float(f.ln.h[i]);float y=p+c.beta[i%64];replayWords+=forwardsoftmaxreference::bits(y)!=forwardsoftmaxreference::bits(float(f.ln.y[i]));}
   if(id==0||id==7){auto old=precision::forward(c,wq,wk,precision::candidates()[id==0?6:7]);anchorWords+=words(f.ln.y,old.ln.y)+words(f.ln.h,old.ln.h)+words(f.ln.inv,old.ln.inv)+words(f.q,old.q)+words(f.k,old.k);for(unsigned h=0;h<2;++h)anchorWords+=words(f.raw[h],old.raw[h])+words(f.p[h],old.p[h]);}
   for(unsigned h=0;h<2;++h){for(unsigned ri=0;ri<2;++ri){const auto& p=ri?legacy[h]:ref[h];auto st=attribution::stats(p,f.p[h]);total[ri].probability(z.index,h,st);bool same=std::string(set)=="frozen"&&z.index<24;
    std::cout<<"{\"kind\":\"case\",\"candidate\":\""<<name<<"\",\"set\":\""<<set<<"\",\"reference\":\""<<(ri?"legacy":"source")<<"\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"p_maxAbs\":"<<st.abs<<",\"rms\":"<<st.rms<<",\"row_error\":"<<st.rows<<",\"coordinate\":"<<st.coordinate<<"}\n";
    for(unsigned gi=0;gi<gs.size();++gi){auto b=attribution::impact(gs[gi][h].dp,gs[gi][h].dc,p,f.p[h]);total[ri].add(z.index,h,b);if(same&&gi==z.index){matched[ri].probability(z.index,h,st);matched[ri].add(z.index,h,b);}
     std::cout<<"{\"kind\":\"pair\",\"candidate\":\""<<name<<"\",\"set\":\""<<set<<"\",\"reference\":\""<<(ri?"legacy":"source")<<"\",\"forward\":"<<z.index<<",\"head\":"<<h<<",\"gradient\":"<<gi;attribution::impactFields(b);std::cout<<"}\n";
     if(ri==0){auto ds=probabilitycontract::backward(gs[gi][h].dp,f.p[h]);for(bool key:{false,true}){auto state=one::split(key?f.q:f.k,h);auto d=one::mul(one::mm(ds,state,32,32,32,key,false),{double(probabilitycontract::scale())});auto fp=one::mul(one::mm(precision::stored(ds),state,32,32,32,key,false,true),{double(probabilitycontract::scale())},true);qkErr=std::max(qkErr,attribution::distance(d,fp));}++qkPairs;}
    }
   }}
  }for(unsigned ri=0;ri<2;++ri){total[ri].print(name,set,ri?"legacy":"source");if(std::string(set)=="frozen")matched[ri].print(name,"same_frozen",ri?"legacy":"source");}
  if(anchorWords)throw std::runtime_error("S0/C6 or S7/C7 word mismatch");
  std::cout<<"{\"kind\":\"consistency\",\"candidate\":\""<<name<<"\",\"set\":\""<<set<<"\",\"anchor_word_mismatches\":"<<anchorWords<<",\"saved_affine_replay_word_mismatches\":"<<replayWords<<",\"saved_qk_word_mismatches\":0,\"local_qk_gradient_error\":"<<qkErr<<",\"local_qk_pairs\":"<<qkPairs<<",\"local_qk_gate\":"<<(qkErr<=.01?"true":"false")<<"}\n";
 }
}
int run(int argc,char** argv){if(argc<3)return 1;bool confirm=std::string(argv[2])=="confirm";if(std::string(argv[2])!="development"&&!confirm)return 1;std::vector<unsigned> ids={0,1,2,3,4,5,6,7};if(confirm){if(argc<4)return 1;unsigned selected=std::stoul(argv[3]);if(selected>7)return 1;ids={0,selected,7};std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());}if(std::string(argv[argc-1])=="--fp32-shift")precision::fp32Shift=true;precision::checks();
 if(confirm){evaluate(confirmation(),confirmGradients(),"confirmation",ids);}else{auto original=one::read(argv[1]),fs=segmentstudy::lnCases(original),gr=upstreamreference::fullUpstream(original);std::vector<std::array<precision::Gradient,2>> gs;for(auto& z:gr){auto g=one::reference(z,true);std::array<precision::Gradient,2> v;for(unsigned h=0;h<2;++h){auto p="h"+std::to_string(h)+"_";v[h]={g.at(p+"dp"),g.at(p+"dc")};}gs.push_back(v);}if(fs.size()!=44||gs.size()!=25)throw std::runtime_error("coverage");evaluate(fs,gs,"frozen",ids);evaluate(precision::synthetic(),precision::syntheticGradients(),"holdout",ids);}
 std::cout<<"{\"kind\":\"completed\",\"dsp_execution\":false,\"confirmation\":"<<(confirm?"true":"false")<<"}\n";return 0;}
}
int main(int argc,char** argv){std::cout<<std::setprecision(17);try{return selective::run(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}}
