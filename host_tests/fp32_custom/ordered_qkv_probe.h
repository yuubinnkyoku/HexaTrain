#pragma once
#include "ordered_qkv_controls.h"
#include "ordered_qkv_graph.h"
namespace orderedqkvstudy {
using closurestudy::require;
using orderedhvxstudy::words;
struct Case {std::string name;one::FV supplied;std::array<one::FV,3>w;mmfixture::ln::Case ln;};
inline std::vector<Case> cases(const std::vector<one::Fixture>& fs){
    std::vector<Case> out;for(auto& z:segmentstudy::lnCases(fs)){Case c;c.name="ln_"+std::to_string(z.index);c.ln=one::lncase(z);c.ln.x=z.v.at("layerinput");c.ln.gamma=z.v.at("gamma1");c.ln.beta=z.v.at("beta1");
        c.supplied=one::f(mmfixture::ln::doubleReference(c.ln,64).y);for(unsigned p=0;p<3;++p)c.w[p]=z.v.at(p==0?"wq":p==1?"wk":"wv");out.push_back(std::move(c));}return out;
}
inline std::map<unsigned,one::FV> inputs(const Graph& g,const Case& c){
    std::map<unsigned,one::FV> in={{g.x,g.stage?c.ln.x:c.supplied}};for(unsigned p=0;p<3;++p)in[g.w[p]]=c.w[p];
    if(g.stage){in[g.ga]=c.ln.gamma;in[g.be]=c.ln.beta;in[g.dy]=c.ln.dy;}return in;
}
inline std::map<std::string,one::FV> finals(const Graph& g,const std::map<unsigned,std::vector<uint32_t>>& o){
    std::map<std::string,one::FV> v;if(g.stage<2)for(unsigned p=0;p<3;++p)v["qkv"+std::to_string(p)]=decode(o.at(g.q[p]));
    else for(unsigned p=0;p<6;++p)v["head"+std::to_string(p)]=decode(o.at(g.heads[p]));
    if(g.stage){v["dx"]=decode(o.at(g.dx));v["dg"]=decode(o.at(g.dg));v["db"]=decode(o.at(g.db));}return v;
}
inline void qreceipt(Graph& g,const Case& c,const std::map<unsigned,std::vector<uint32_t>>& o,const one::FV& received,const std::array<one::FV,3>& actual,unsigned ci){
    const auto& r=o.at(g.qr);unsigned h=g.hvx?24:8;
    require(r.size()==h+20480&&r[0]==(g.hvx?0x48383115u:0x4838310du)&&r[1]>g.lastQ&&r[2]==14336&&r[3]==6144&&r[5]==81&&r[6]==32);g.lastQ=r[1];
    require(sameFpWords(closurestudy::slice(r,h,2048),received));
    for(unsigned p=0;p<3;++p)require(sameFpWords(closurestudy::slice(r,h+2048+p*4096,4096),c.w[p])&&sameFpWords(closurestudy::slice(r,h+14336+p*2048,2048),actual[p]));
    if(g.hvx){require(r[15]==128&&r[16]==0&&r[17]==64&&r[18]==64&&r[19]==3&&r[20]==1&&r[21]==0&&r[22]==0&&r[23]==0);
        std::printf("qkvalignment,%s,%u,y=%u,wq=%u,wk=%u,wv=%u,q=%u,k=%u,v=%u,vector_bytes=128,scratch_bytes=0\n",g.name.c_str(),ci,r[8],r[9],r[10],r[11],r[12],r[13],r[14]);}
    std::printf("qkvreceipt,%s,%u,marker=%08x,all_words=1,math_us=%u,window_us=%u\n",g.name.c_str(),ci,r[0],r[4],r[7]);
}
inline std::map<std::string,one::FV> audit(Graph& g,const Case& c,const std::map<unsigned,std::vector<uint32_t>>& o,unsigned ci,unsigned& accuracy,unsigned& identity){
    auto values=finals(g,o);one::FV received=c.supplied;one::Vec idealY=one::d(c.supplied);
    if(g.stage){auto ref=mmfixture::ln::doubleReference(c.ln,64);auto host=mmfixture::ln::anchoredFp32Oracle(c.ln,64);received=host.y;idealY=ref.y;
        // Hidden NoReceipt operands below are Host oracles, never observations.
        if(g.observed)received=closurestudy::slice(o.at(g.qr),g.hvx?24:8,2048);else if(g.tap)received=decode(o.at(g.y));
        for(auto key:{"dx","dg","db"})accuracy+=metric(g.name,ci,key,std::string(key)=="dx"?ref.dx:std::string(key)=="dg"?ref.dg:ref.db,one::d(std::string(key)=="dx"?host.dx:std::string(key)=="dg"?host.dg:host.db),values.at(key),.02);
        if(g.observed||g.tap){values["y"]=received;auto hat=g.observed?closurestudy::slice(o.at(g.fr),2185,2048):decode(o.at(g.hat));auto iv=g.observed?closurestudy::slice(o.at(g.fr),4233,32):decode(o.at(g.iv));values["hat"]=hat;values["inverse"]=iv;
            accuracy+=metric(g.name,ci,"y",ref.y,one::d(host.y),received,.002);accuracy+=metric(g.name,ci,"hat",ref.xhat,one::d(host.xhat),hat,.002);accuracy+=metric(g.name,ci,"inverse",ref.inverse,one::d(host.inverse),iv,.002);
            if(g.tap)require(sameFpWords(received,decode(o.at(g.y)))&&sameFpWords(hat,decode(o.at(g.hat)))&&sameFpWords(iv,decode(o.at(g.iv))));
            if(g.observed){const auto& f=o.at(g.fr);const auto& b=o.at(g.br);
                require(f.size()==4265&&b.size()==6248&&f[0]==0x48383104u&&b[0]==0x48383105u&&f[1]>g.lastF&&b[1]>g.lastB&&f[2]==2048&&f[3]==4&&b[2]==2048&&b[3]==4&&f[5]==81&&b[5]==81);g.lastF=f[1];g.lastB=b[1];
                require(sameFpWords(closurestudy::slice(f,8,2048),c.ln.x)&&sameFpWords(closurestudy::slice(f,2056,64),c.ln.gamma)&&sameFpWords(closurestudy::slice(f,2120,64),c.ln.beta)&&f[2184]==word(1e-5f));
                require(sameFpWords(closurestudy::slice(b,8,2048),c.ln.dy)&&sameFpWords(closurestudy::slice(b,2056,64),c.ln.gamma)&&sameFpWords(closurestudy::slice(b,2120,2048),hat)&&sameFpWords(closurestudy::slice(b,4168,32),iv)&&sameFpWords(closurestudy::slice(b,4200,2048),values.at("dx")));
                one::FV reconstructed(2048);for(unsigned j=0;j<2048;++j){volatile float product=c.ln.gamma[j%64]*hat[j];reconstructed[j]=product+c.ln.beta[j%64];}
                require(sameFpWords(received,reconstructed));
                std::printf("qkvnative,%s,%u,ln_input_words=1,consumer_y_words=1,producer_y_tap_words=%u,saved_affine_y_words=1,saved_state_words=1,reference_fed_dy=1\n",g.name.c_str(),ci,g.tap);
            }
        }
    }
    std::array<one::FV,3> actual,fp;for(unsigned p=0;p<3;++p){auto ref=one::mm(idealY,one::d(c.w[p]),32,64,64);fp[p]=one::f(one::mm(one::d(received),one::d(c.w[p]),32,64,64,false,false,true));
        actual[p]=o.count(g.q[p])?decode(o.at(g.q[p])):g.observed?closurestudy::slice(o.at(g.qr),(g.hvx?24:8)+14336+p*2048,2048):fp[p];
        if(o.count(g.q[p])||g.observed){auto key="qkv"+std::to_string(p);values[key]=actual[p];accuracy+=metric(g.name,ci,key,ref,one::d(fp[p]),actual[p],.002);identity+=words(actual[p],fp[p],g.name,ci,("host_ordered_"+key).c_str());}
        if(g.stage==2)for(unsigned h=0;h<2;++h){auto key="head"+std::to_string(p*2+h);accuracy+=metric(g.name,ci,key,one::split(ref,h),one::split(one::d(fp[p]),h),values.at(key),.002);
            identity+=words(values.at(key),one::f(one::split(one::d(actual[p]),h)),g.name,ci,(std::string(g.observed||g.tap?"lossless_observed_":"oracle_slice_")+key).c_str());}
    }
    if(g.observed){qreceipt(g,c,o,received,actual,ci);
        if(g.stage==2){auto& r=o.at(g.hr);require(r.size()==12296&&r[0]==0x48383113u&&r[1]>g.lastH&&r[2]==6144&&r[3]==6144&&r[5]==81&&r[6]==32);g.lastH=r[1];
            for(unsigned p=0;p<3;++p)require(sameFpWords(closurestudy::slice(r,8+p*2048,2048),actual[p]));for(unsigned p=0;p<6;++p)require(sameFpWords(closurestudy::slice(r,6152+p*1024,1024),values.at("head"+std::to_string(p))));
            std::printf("qkvheads,%s,%u,producer_consumer_words=1,lossless_words=1\n",g.name.c_str(),ci);}
    }return values;
}
inline void matrix(unsigned stage,const std::vector<Case>& cs,unsigned& accuracy,unsigned& identity){
    std::vector<std::map<std::string,one::FV>> anchors(cs.size());std::array<std::vector<std::map<std::string,one::FV>>,4> scalar;for(auto& v:scalar)v.resize(cs.size());
    for(unsigned mode=0;mode<8;++mode){Graph g("qkvAccuracy"+std::to_string(stage)+std::to_string(mode),stage,mode>=4,mode%4<2,mode%2);closurestudy::describe(g.g,g.name);
        for(unsigned ci=0;ci<cs.size();++ci){auto out=g.g.run(inputs(g,cs[ci]));if(!ci)std::printf("qkvfirst,%s,wall_us=%.3f\n",g.name.c_str(),g.g.executeUs);auto v=audit(g,cs[ci],out,ci,accuracy,identity);
            if(!mode)anchors[ci]=v;else for(auto& e:v)identity+=words(e.second,anchors[ci].at(e.first),g.name,ci,("scalar_anchor_"+e.first).c_str());
            if(mode<4)scalar[mode][ci]=v;else for(auto& e:v)identity+=words(e.second,scalar[mode%4][ci].at(e.first),g.name,ci,("same_variant_scalar_"+e.first).c_str());
        }
    }std::printf("qkvgate,stage%u,accuracy_failures=%u,word_mismatches=%u\n",stage,accuracy,identity);
}
inline void paired(const Case& c,unsigned& accuracy,unsigned& identity){
    for(unsigned stage=0;stage<3;++stage)for(bool obs:{true,false}){
        std::string n="qkvPerf"+std::to_string(stage)+std::to_string(obs);Graph scalar(n+"S",stage,false,obs),hvx(n+"H",stage,true,obs);Graph* gs[]={&scalar,&hvx};std::map<std::string,one::FV> anchor;
        for(unsigned k=0;k<2;++k){auto& g=*gs[k];closurestudy::describe(g.g,g.name);auto out=g.g.run(inputs(g,c));std::printf("qkvfirst,%s,wall_us=%.3f\n",g.name.c_str(),g.g.executeUs);auto v=audit(g,c,out,0,accuracy,identity);if(!k)anchor=finals(g,out);else for(auto& e:finals(g,out))identity+=words(e.second,anchor.at(e.first),g.name,0,("scalar_pair_"+e.first).c_str());}
        if(accuracy||identity)return;
        auto execute=[&](unsigned kind,unsigned cycle,unsigned order,bool steady){auto& g=*gs[kind];auto out=g.g.run(inputs(g,c));for(auto& e:finals(g,out))require(sameFpWords(e.second,anchor.at(e.first)));
            // Receipts are checked in full for warmups and every steady sample.
            unsigned a=0,i=0;if(obs)audit(g,c,out,0,a,i);require(!a&&!i);
            if(steady){std::printf("qkvperf,%s,stage=%u,observed=%u,hvx=%u,cycle=%u,order=%u,wall_us=%.3f",g.name.c_str(),stage,obs,kind,cycle,order,g.g.executeUs);if(obs)std::printf(",math_us=%u,window_us=%u",out.at(g.qr)[4],out.at(g.qr)[7]);std::puts(",final_words=1");}
        };
        for(unsigned warm=0;warm<2;++warm)for(unsigned k=0;k<2;++k)execute(k,0,warm,false);
        for(unsigned cycle=0;cycle<8;++cycle)for(unsigned order=0;order<4;++order){unsigned k=order==1||order==2;if(cycle%2)k=1-k;execute(k,cycle,order,true);}
    }
}
inline int run(const std::vector<one::Fixture>& fs){
    auto native=cases(fs);require(native.size()==44);auto all=native;
    for(auto& control:qkvcontrols::cases()){Case c;c.name=control.name;c.supplied=control.a;c.w=control.w;all.push_back(std::move(c));}
    for(unsigned ci=0;ci<all.size();++ci)std::printf("qkvfixture,%u,name=%s,collection=%s\n",ci,all[ci].name.c_str(),ci<44?"ln_stage_accuracy":"independent_arithmetic_control");
    unsigned accuracy=0,identity=0;matrix(0,all,accuracy,identity);
    if(!accuracy&&!identity)matrix(1,native,accuracy,identity);
    if(!accuracy&&!identity)matrix(2,native,accuracy,identity);
    if(!accuracy&&!identity)paired(native.front(),accuracy,identity);
    struct rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0);std::printf("qkv_peak_rss_kib,%ld\n",usage.ru_maxrss);
    std::printf("summary,oneQkvHvx,accuracy_failures=%u,word_mismatches=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0\n",accuracy,identity);
    return accuracy||identity?4:0;
}
}
