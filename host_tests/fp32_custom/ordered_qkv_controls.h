#pragma once
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
namespace qkvcontrols {
struct Case {std::string name;std::vector<float> a;std::array<std::vector<float>,3> w;};
inline std::vector<Case> cases(){
    const char* names[]={"ordinary","constant","low_variance","large_offset","mixed_signs","cancellation","sparse","fixed_random","asymmetric","signed_zero","subnormal","large_small","transpose_sensitive"};
    unsigned seed=0x8164123u;std::vector<Case> all;
    for(unsigned kind=0;kind<13;++kind){Case c;c.name=names[kind];c.a.resize(2048);for(auto& w:c.w)w.resize(4096);
        for(unsigned m=0;m<32;++m)for(unsigned q=0;q<64;++q){seed=seed*1664525u+1013904223u;float x=float(int(seed>>16)-32768)/65536.f;
            if(kind==1)x=.125f;if(kind==2)x=1.f+float(int(q%3)-1)/4096.f;
            if(kind==3)x=1024.f+float(int((m+q)%7)-3)/16.f;
            if(kind==5)x=1;if(kind==6&&q%13)x=0;if(kind==8||kind==12)x=float(int(m*7+q%11)-100)/256.f;
            if(kind==9)x=q%2?-0.f:0.f;if(kind==10)x=std::ldexp(float(1+(m+q)%3),-129);
            if(kind==11)x=std::ldexp(float(1+m%3),40);
            c.a[m*64+q]=x;}
        for(unsigned p=0;p<3;++p)for(unsigned q=0;q<64;++q)for(unsigned n=0;n<64;++n){seed=seed*1664525u+1013904223u;float x=float(int(seed>>16)-32768)/262144.f;
            if(kind==3)x*=1.f/1024;if(kind==4)x=(n%2?-1.f:1.f)*float(1+(q+p)%7)/64.f;
            if(kind==5)x=(q%2?-1.f:1.f)*float(p+1)/32.f;
            if(kind==8||kind==12)x=float(int(q*3+n*7+p*29)-300)/1024.f;
            if(kind==9)x=(n+p)%2?-1.f:1.f;if(kind==10)x=.25f*float(p+1);
            if(kind==11)x=std::ldexp(float(int((q+n+p)%5)-2),-40);
            c.w[p][q*64+n]=x;}
        all.push_back(std::move(c));
    }return all;
}
}
