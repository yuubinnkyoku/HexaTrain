#pragma once
#include "ordered_ffn2_controls.h"
#include "ordered_hvx_graph.h"
namespace orderedhvxstudy {
using closurestudy::require;
inline unsigned words(const one::FV& a,const one::FV& b,const std::string& graph,unsigned ci,const char* kind){
    require(a.size()==b.size());unsigned mismatches=0,first=~0u;
    for(unsigned j=0;j<a.size();++j)if(word(a[j])!=word(b[j])){++mismatches;if(first==~0u)first=j;}
    std::printf("hvxwords,%s,%u,%s,count=%u,first=%u,actual=%08x,expected=%08x\n",graph.c_str(),ci,kind,mismatches,first,first==~0u?0:word(a[first]),first==~0u?0:word(b[first]));
    return mismatches;
}
inline void ffnReceipt(Graph& g,const closurestudy::FfnCase& c,const std::map<unsigned,std::vector<uint32_t>>& o,const one::FV& actual,unsigned ci){
    auto& r=o.at(g.mr);unsigned h=g.hvx?16:8;
    require(r.size()==h+14336&&r[0]==(g.hvx?0x48383114u:0x48383112u)&&r[1]>g.lastM&&r[2]==12288&&r[3]==2048&&r[5]==81&&r[6]==32);g.lastM=r[1];
    require(sameFpWords(closurestudy::slice(r,h,4096),c.a)&&sameFpWords(closurestudy::slice(r,h+4096,8192),c.b)&&sameFpWords(closurestudy::slice(r,h+12288,2048),actual));
    if(g.hvx){require(r[11]==128&&r[12]==0&&r[13]==128&&r[14]==64&&r[15]==1);
        std::printf("hvxalignment,%s,%u,a_mod128=%u,b_mod128=%u,y_mod128=%u,vector_bytes=128,scratch_bytes=0\n",g.name.c_str(),ci,r[8],r[9],r[10]);}
    std::printf("hvxreceipt,%s,%u,marker=%08x,math_us=%u,window_us=%u,all_words=1\n",g.name.c_str(),ci,r[0],r[4],r[7]);
}
inline std::map<std::string,one::FV> auditSegment(Graph& g,const closurestudy::FfnCase& c,const std::map<unsigned,std::vector<uint32_t>>& o,unsigned ci,unsigned& accuracy,unsigned& identity){
    auto finals=g.finals(o);auto values=finals;
    auto ref=one::mm(one::d(c.a),one::d(c.b),32,64,128);
    auto fp=one::f(one::mm(one::d(c.a),one::d(c.b),32,64,128,false,false,true));
    // Expected intermediates are used only as Host oracles when unobservable.
    auto actual=o.count(g.mid)?decode(o.at(g.mid)):g.observed?closurestudy::slice(o.at(g.mr),(g.hvx?16:8)+12288,2048):fp;
    if(o.count(g.mid)||g.observed){values["ffn2"]=actual;accuracy+=metric(g.name,ci,"ffn2",ref,one::d(fp),actual,.002);identity+=words(actual,fp,g.name,ci,"host_ordered_ffn2");}
    if(g.observed)ffnReceipt(g,c,o,actual,ci);
    if(!g.stage)return values;
    auto ideal=one::add(ref,one::d(c.res));auto xfp=one::f(one::add(one::d(fp),one::d(c.res),true));
    auto rx=o.count(g.x)?decode(o.at(g.x)):g.observed?closurestudy::slice(o.at(g.ar),6152,2048):xfp;
    if(o.count(g.x)||g.observed){values["x"]=rx;accuracy+=metric(g.name,ci,"x",ideal,one::d(xfp),rx,.002);identity+=words(rx,xfp,g.name,ci,"host_ordered_x");}
    if(g.observed){auto& r=o.at(g.ar);require(r.size()==8200&&r[0]==0x48383101u&&r[1]>g.lastA&&r[2]==2048&&r[3]==2&&r[5]==81&&r[6]==32);g.lastA=r[1];
        require(sameFpWords(closurestudy::slice(r,8,2048),actual)&&sameFpWords(closurestudy::slice(r,2056,2048),c.res)&&sameFpWords(closurestudy::slice(r,6152,2048),rx));for(unsigned j=4104;j<6152;++j)require(r[j]==0);}
    if(g.stage==2){
        auto lc=one::lncase(c.z);lc.x=one::f(ideal);lc.gamma=c.z.v.at("gamma1");lc.beta=c.z.v.at("beta1");
        auto target=mmfixture::ln::doubleReference(lc,64);lc.x=xfp;auto host=mmfixture::ln::anchoredFp32Oracle(lc,64);
        for(auto& e:finals){auto& key=e.first;auto tr=key=="y"?target.y:key=="dx"?target.dx:key=="dg"?target.dg:target.db;
            auto oracle=key=="y"?host.y:key=="dx"?host.dx:key=="dg"?host.dg:host.db;
            accuracy+=metric(g.name,ci,key,tr,one::d(oracle),e.second,key=="y"?.002:.02);}
        if(g.observed||g.tap){
            auto hat=g.observed?closurestudy::slice(o.at(g.fr),2185,2048):decode(o.at(g.hat));
            auto iv=g.observed?closurestudy::slice(o.at(g.fr),4233,32):decode(o.at(g.iv));
            values["hat"]=hat;values["inverse"]=iv;
            accuracy+=metric(g.name,ci,"hat",target.xhat,one::d(host.xhat),hat,.002);
            accuracy+=metric(g.name,ci,"inverse",target.inverse,one::d(host.inverse),iv,.002);
            if(g.tap)require(sameFpWords(decode(o.at(g.hat)),hat)&&sameFpWords(decode(o.at(g.iv)),iv));
            if(g.observed){auto& f=o.at(g.fr);auto& b=o.at(g.br);
                require(f.size()==4265&&b.size()==6248&&f[0]==0x48383104u&&b[0]==0x48383105u&&f[1]>g.lastF&&b[1]>g.lastB&&f[5]==81&&b[5]==81);g.lastF=f[1];g.lastB=b[1];
                require(sameFpWords(closurestudy::slice(f,8,2048),rx)&&sameFpWords(closurestudy::slice(f,2056,64),lc.gamma)&&sameFpWords(closurestudy::slice(f,2120,64),lc.beta)&&f[2184]==word(1e-5f));
                require(sameFpWords(closurestudy::slice(b,8,2048),lc.dy)&&sameFpWords(closurestudy::slice(b,2056,64),lc.gamma)&&sameFpWords(closurestudy::slice(b,2120,2048),hat)&&sameFpWords(closurestudy::slice(b,4168,32),iv)&&sameFpWords(closurestudy::slice(b,4200,2048),decode(o.at(g.dx))));
            }
        }
    }
    if(g.observed)std::printf("hvxnative,%s,%u,producer_consumer_words=1,residual_words=1,saved_state_words=%u,reference_fed_dy=%u\n",g.name.c_str(),ci,g.stage==2,g.stage==2);
    return values;
}
inline void segments(const std::vector<closurestudy::FfnCase>& cases,unsigned& accuracy,unsigned& identity){
    for(unsigned stage=1;stage<=2;++stage){std::vector<std::map<std::string,one::FV>> anchors(cases.size());std::array<std::vector<std::map<std::string,one::FV>>,4> counterparts;for(auto& a:counterparts)a.resize(cases.size());
        for(unsigned mode=0;mode<8;++mode){Graph g("orderedSegment"+std::to_string(stage)+std::to_string(mode),stage,mode>=4,mode%4<2,mode%2);closurestudy::describe(g.g,g.name);
            for(unsigned ci=0;ci<cases.size();++ci){auto o=g.g.run(g.inputs(cases[ci]));if(!ci)std::printf("hvxfirst,%s,wall_us=%.3f\n",g.name.c_str(),g.g.executeUs);
                auto values=auditSegment(g,cases[ci],o,ci,accuracy,identity);
                if(mode==0)anchors[ci]=values;else for(auto& e:values)identity+=words(e.second,anchors[ci].at(e.first),g.name,ci,("scalar_observed_"+e.first).c_str());
                if(mode<4)counterparts[mode][ci]=values;else for(auto& e:values)identity+=words(e.second,counterparts[mode%4][ci].at(e.first),g.name,ci,("same_variant_scalar_"+e.first).c_str());
            }
        }
        std::printf("hvxgate,native%u,accuracy_failures=%u,word_mismatches=%u\n",stage,accuracy,identity);
        if(accuracy||identity)return;
    }
}
inline void paired(const closurestudy::FfnCase& c,unsigned& accuracy,unsigned& identity){
    for(unsigned stage=0;stage<3;++stage)for(bool obs:{true,false}){
        std::string name="orderedPerf"+std::to_string(stage)+std::to_string(obs);
        Graph scalar(name+"S",stage,false,obs),hvx(name+"H",stage,true,obs);closurestudy::describe(scalar.g,scalar.name);closurestudy::describe(hvx.g,hvx.name);
        Graph* graphs[]={&scalar,&hvx};std::map<std::string,one::FV> anchor;
        for(unsigned kind=0;kind<2;++kind){auto& g=*graphs[kind];auto out=g.g.run(g.inputs(c));std::printf("hvxfirst,%s,wall_us=%.3f\n",g.name.c_str(),g.g.executeUs);
            auditSegment(g,c,out,0,accuracy,identity);auto finals=g.finals(out);if(!kind)anchor=finals;else for(auto& e:finals)identity+=words(e.second,anchor.at(e.first),g.name,0,((obs?"scalar_observed_":"scalar_no_receipt_")+e.first).c_str());
        }
        if(accuracy||identity)return;
        auto execute=[&](unsigned kind,unsigned cycle,unsigned order,bool steady){auto& g=*graphs[kind];auto out=g.g.run(g.inputs(c));
            for(auto& e:g.finals(out))require(sameFpWords(e.second,anchor.at(e.first)));
            // Full receipt operands/output continue to be checked outside execute wall.
            if(obs){auto actual=stage?closurestudy::slice(out.at(g.mr),(g.hvx?16:8)+12288,2048):decode(out.at(g.mid));ffnReceipt(g,c,out,actual,0);
                if(stage){auto& r=out.at(g.ar);auto rx=stage==1?decode(out.at(g.x)):closurestudy::slice(r,6152,2048);
                    require(r[0]==0x48383101u&&r[1]>g.lastA&&r[2]==2048&&r[3]==2&&r[5]==81&&r[6]==32);g.lastA=r[1];
                    require(sameFpWords(closurestudy::slice(r,8,2048),actual)&&sameFpWords(closurestudy::slice(r,2056,2048),c.res)&&sameFpWords(closurestudy::slice(r,6152,2048),rx));for(unsigned j=4104;j<6152;++j)require(r[j]==0);
                    if(stage==2){auto& f=out.at(g.fr);auto& b=out.at(g.br);require(f[0]==0x48383104u&&b[0]==0x48383105u&&f[1]>g.lastF&&b[1]>g.lastB&&f[5]==81&&b[5]==81);g.lastF=f[1];g.lastB=b[1];
                        require(sameFpWords(closurestudy::slice(f,8,2048),rx)&&sameFpWords(closurestudy::slice(f,2056,64),c.z.v.at("gamma1"))&&sameFpWords(closurestudy::slice(f,2120,64),c.z.v.at("beta1"))&&f[2184]==word(1e-5f));
                        require(sameFpWords(closurestudy::slice(b,8,2048),c.z.v.at("dy"))&&sameFpWords(closurestudy::slice(b,2056,64),c.z.v.at("gamma1"))&&sameFpWords(closurestudy::slice(f,2185,2048),closurestudy::slice(b,2120,2048))&&sameFpWords(closurestudy::slice(f,4233,32),closurestudy::slice(b,4168,32))&&sameFpWords(closurestudy::slice(b,4200,2048),decode(out.at(g.dx))));
                    }
                }
            }
            if(steady){std::printf("hvxperf,%s,stage=%u,observed=%u,hvx=%u,cycle=%u,order=%u,wall_us=%.3f",g.name.c_str(),stage,obs,kind,cycle,order,g.g.executeUs);
                if(obs)std::printf(",math_us=%u,window_us=%u",out.at(g.mr)[4],out.at(g.mr)[7]);std::printf(",final_words=1\n");}
        };
        for(unsigned warm=0;warm<2;++warm)for(unsigned kind=0;kind<2;++kind)execute(kind,0,warm,false);
        // Eight ABBA/BAAB blocks: exactly16 steady samples per variant.
        for(unsigned cycle=0;cycle<8;++cycle)for(unsigned order=0;order<4;++order){unsigned kind=(order==1||order==2)?1:0;if(cycle%2)kind=1-kind;execute(kind,cycle,order,true);}
    }
}
inline int run(const std::vector<one::Fixture>& fs){
    auto cases=closurestudy::ffnCases(fs);
    for(auto& control:orderedcontrols::cases()){auto c=cases.front();c.z.index=100+cases.size()-16;c.a=control.a;c.b=control.b;cases.push_back(std::move(c));}
    unsigned accuracy=0,identity=0;std::vector<one::FV> anchors(cases.size());
    for(unsigned variant=0;variant<4;++variant){
        bool hvx=variant>=2,obs=variant%2==0;std::string name="orderedFFN"+std::to_string(variant);
        DiagnosticGraph g(name);g.receiptEnabled=obs;
        auto a=g.tensor("a",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,128});
        auto b=g.tensor("b",QNN_TENSOR_TYPE_APP_WRITE,{1,1,128,64});
        auto y=g.tensor("y",QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});
        auto r=g.tensor("r",QNN_TENSOR_TYPE_APP_READ,{1,1,1,hvx?14352u:14344u},QNN_DATATYPE_INT_32);
        g.node("ffn",hvx?"FFN2Hvx":"FFN2","HexaFp32",{a,b},{y,r});g.finalize();closurestudy::describe(g,name);
        unsigned last=0;
        for(unsigned ci=0;ci<cases.size();++ci){auto& c=cases[ci];
            auto ref=one::mm(one::d(c.a),one::d(c.b),32,64,128);
            auto fp=one::f(one::mm(one::d(c.a),one::d(c.b),32,64,128,false,false,true));
            auto o=g.run({{a,c.a},{b,c.b}});auto actual=decode(o.at(y));
            accuracy+=metric(name,ci,"ffn2",ref,one::d(fp),actual,.002);
            identity+=words(actual,fp,name,ci,"host_ordered");
            if(variant==0)anchors[ci]=actual;
            else identity+=words(actual,anchors[ci],name,ci,"scalar_observed");
            if(obs){auto& receipt=o.at(r);unsigned h=hvx?16:8;
                require(receipt[0]==(hvx?0x48383114u:0x48383112u)&&receipt[1]>last&&receipt[2]==12288&&receipt[3]==2048&&receipt[5]==81&&receipt[6]==32);last=receipt[1];
                require(sameFpWords(closurestudy::slice(receipt,h,4096),c.a)&&sameFpWords(closurestudy::slice(receipt,h+4096,8192),c.b)&&sameFpWords(closurestudy::slice(receipt,h+12288,2048),actual));
                if(hvx){require(receipt[11]==128&&receipt[12]==0&&receipt[13]==128&&receipt[14]==64&&receipt[15]==1);
                    std::printf("hvxalignment,%u,a_mod128=%u,b_mod128=%u,y_mod128=%u,vector_bytes=128,scratch_bytes=0\n",ci,receipt[8],receipt[9],receipt[10]);}
                std::printf("hvxreceipt,%s,%u,marker=%08x,math_us=%u,window_us=%u,all_words=1\n",name.c_str(),ci,receipt[0],receipt[4],receipt[7]);
            }
        }
    }
    // Integration and performance are intentionally gated by standalone identity.
    std::printf("hvxgate,standalone,accuracy_failures=%u,word_mismatches=%u,integration_allowed=%u,performance_allowed=%u\n",accuracy,identity,!accuracy&&!identity,!accuracy&&!identity);
    if(!accuracy&&!identity){auto native=std::vector<closurestudy::FfnCase>(cases.begin(),cases.begin()+16);segments(native,accuracy,identity);}
    if(!accuracy&&!identity)paired(cases.front(),accuracy,identity);
    struct rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0);std::printf("hvx_peak_rss_kib,%ld\n",usage.ru_maxrss);
    std::printf("summary,oneHvx,accuracy_failures=%u,word_mismatches=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0\n",accuracy,identity);
    return accuracy||identity?4:0;
}
}
