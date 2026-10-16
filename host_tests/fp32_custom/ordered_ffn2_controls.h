#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <limits>
namespace orderedcontrols {
struct Case {std::string name;std::vector<float> a,b;};
inline std::vector<Case> cases(){
    std::vector<Case> v;
    const char* names[]={"ordinary","relu","sparse","zero","signed_zero","mixed_weights",
        "large_small","cancellation","near_subnormal","random","asymmetric","finite_extremes"};
    unsigned seed=0x81ff123u;
    for(unsigned kind=0;kind<12;++kind){Case c{names[kind],std::vector<float>(4096),std::vector<float>(8192)};
        for(unsigned m=0;m<32;++m)for(unsigned q=0;q<128;++q){
            seed=1664525u*seed+1013904223u;float x=float(int(seed>>16)-32768)/65536.f;
            if(kind==1)x=std::fabs(x);if(kind==2&&q%17)x=0;
            if(kind==3)x=0;if(kind==4)x=(q%2)?-0.f:0.f;
            if(kind==6)x=std::ldexp(float(1+m%3),40);
            if(kind==7)x=1;
            if(kind==8)x=std::ldexp(float(1+(m+q)%3),-126);
            if(kind==10)x=float(int(m*7+q%9)-100)/128.f;
            if(kind==11)x=(q%2)?std::numeric_limits<float>::min():std::numeric_limits<float>::max();
            c.a[m*128+q]=x;
        }
        for(unsigned q=0;q<128;++q)for(unsigned n=0;n<64;++n){
            seed=1664525u*seed+1013904223u;float x=float(int(seed>>16)-32768)/65536.f;
            if(kind==4)x=n%2?-1.f:1.f;
            if(kind==5)x=(n%2?-1.f:1.f)*float(1+q%7)/16.f;
            if(kind==6)x=std::ldexp(float(int((q+n)%5)-2),-40);
            if(kind==7)x=(q%2?-1.f:1.f)*.125f;
            if(kind==8)x=.25f;
            if(kind==10)x=float(int(q*3+n%11)-180)/256.f;
            if(kind==11)x=std::numeric_limits<float>::min();
            c.b[q*64+n]=x;
        }v.push_back(std::move(c));
    }return v;
}
}
