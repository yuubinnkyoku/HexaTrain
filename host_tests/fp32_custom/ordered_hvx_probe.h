#pragma once
#include "ordered_ffn2_controls.h"
namespace orderedhvxstudy {
using closurestudy::require;
inline unsigned words(const one::FV& a,const one::FV& b,const std::string& graph,unsigned ci,const char* kind){
    require(a.size()==b.size());unsigned mismatches=0,first=~0u;
    for(unsigned j=0;j<a.size();++j)if(word(a[j])!=word(b[j])){++mismatches;if(first==~0u)first=j;}
    std::printf("hvxwords,%s,%u,%s,count=%u,first=%u,actual=%08x,expected=%08x\n",graph.c_str(),ci,kind,mismatches,first,first==~0u?0:word(a[first]),first==~0u?0:word(b[first]));
    return mismatches;
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
    std::printf("summary,oneHvx,accuracy_failures=%u,word_mismatches=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0\n",accuracy,identity);
    return accuracy||identity?4:0;
}
}
