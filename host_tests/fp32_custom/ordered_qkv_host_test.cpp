// CPU controls validate scalar contracts; HVX on Host must fail explicitly.
#define main inherited_segment_main
#include "segment_kernel_test.cpp"
#undef main
#include "ordered_qkv_controls.h"
#include <stdexcept>
#define need(b) do { if(!(b)) throw std::runtime_error(std::string("ordered QKV host contract line=")+std::to_string(__LINE__)+" expression="+#b); } while(false)
int main(){
    need(qhpi_init()&&registered==51&&info.numOperations==51);
    need(std::string(names[23])=="ProjectionQKV"&&std::string(names[24])=="ProjectionQKVNoReceipt");
    need(std::string(names[39])=="ProjectionQKVHvx"&&std::string(names[40])=="ProjectionQKVHvxNoReceipt");
    unsigned calls=0;double worst=0;
    for(auto c:qkvcontrols::cases())for(bool observed:{false,true}){
        auto before=c;std::array<std::vector<uint32_t>,4> b;for(unsigned p=0;p<4;++p){unsigned n=p==3?hvxqkv::receiptWords:2048;b[p].assign(n+8,0xa55aa55a);std::fill(b[p].begin()+4,b[p].end()-4,0x7fc0dead);}
        QHPI_Tensor inT[]={tensor(32,64,c.a.data()),tensor(64,64,c.w[0].data()),tensor(64,64,c.w[1].data()),tensor(64,64,c.w[2].data())};const QHPI_Tensor* in[]={inT,inT+1,inT+2,inT+3};
        QHPI_Tensor outT[]={tensor(32,64,b[0].data()+4),tensor(32,64,b[1].data()+4),tensor(32,64,b[2].data()+4),tensor(1,20488,b[3].data()+4,QHPI_INT32)};QHPI_Tensor* out[]={outT,outT+1,outT+2,outT+3};
        auto scalar=observed?segmentcustom::kernel<true,true>:segmentcustom::kernel<true,false>;
        need(scalar(nullptr,3+observed,out,4,in)==QHPI_SUCCESS);
        for(unsigned p=0;p<3;++p){auto fp=one::f(one::mm(one::d(c.a),one::d(c.w[p]),32,64,64,false,false,true));auto ref=one::mm(one::d(c.a),one::d(c.w[p]),32,64,64);need(!std::memcmp(fp.data(),b[p].data()+4,8192));
            auto e=mmfixture::ln::errors(ref,fp);need(!e.nonfinite&&e.absolute<=.002);worst=std::max(worst,e.absolute);
            for(unsigned j=0;j<4;++j)need(b[p][j]==0xa55aa55a&&b[p][b[p].size()-1-j]==0xa55aa55a);
            need(std::find(b[p].begin()+4,b[p].end()-4,0x7fc0dead)==b[p].end()-4);}
        if(observed){auto r=b[3].data()+4;need(r[0]==0x4350550du&&r[2]==14336&&r[3]==6144);need(!std::memcmp(r+8,c.a.data(),8192));
            for(unsigned p=0;p<3;++p){need(!std::memcmp(r+2056+p*4096,c.w[p].data(),16384));need(!std::memcmp(r+14344+p*2048,b[p].data()+4,8192));}}
        need(!std::memcmp(c.a.data(),before.a.data(),8192));for(unsigned p=0;p<3;++p)need(!std::memcmp(c.w[p].data(),before.w[p].data(),16384));
        outT[3].shape.dims[3]=hvxqkv::receiptWords;auto snapshot=b;
        need((observed?hvxqkv::kernel<true>:hvxqkv::kernel<false>)(nullptr,3+observed,out,4,in)==QHPI_ERROR_FATAL);need(b==snapshot);
        outT[3].shape.dims[3]=20488;c.a[0]=INFINITY;need(scalar(nullptr,3+observed,out,4,in)==QHPI_ERROR_FATAL);c.a[0]=NAN;need(scalar(nullptr,3+observed,out,4,in)==QHPI_ERROR_FATAL);
        std::fill(c.a.begin(),c.a.end(),std::numeric_limits<float>::max());for(auto& w:c.w)std::fill(w.begin(),w.end(),2.f);need(scalar(nullptr,3+observed,out,4,in)==QHPI_ERROR_FATAL);
        c=before; // Negative inputs never carry into the next finite variant.
        ++calls;
    }
    std::printf("ordered_qkv_host=PASS controls=13 calls=%u maxAbs=%.12g scalar_words=1 guards=1 immutable=1 hvx_host_execution_rejected=1 negative_separate=1\n",calls,worst);
}
