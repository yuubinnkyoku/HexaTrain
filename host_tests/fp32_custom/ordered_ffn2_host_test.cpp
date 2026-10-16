// Host arithmetic/ABI controls. This deliberately does not mock HVX arithmetic.
#define ORDERED_FFN2_HOST_EXTENSION
#include "closure_kernel_test.cpp"
#include "ordered_ffn2_controls.h"
int main(){
    demand(qhpi_init()&&registered==45);
    demand(info.numOperations==45);
    for(unsigned i=0;i<37;++i)demand(std::string(names[i]).find("Hvx")==std::string::npos);
    demand(std::string(names[33])=="FFN2"&&std::string(names[34])=="FFN2NoReceipt");
    demand(std::string(names[37])=="FFN2Hvx"&&std::string(names[38])=="FFN2HvxNoReceipt");
    unsigned calls=0;double worst=0;
    for(auto& c:orderedcontrols::cases()){
        test<closurecontract::FFN2,true>({c.a,c.b},calls,worst);
        test<closurecontract::FFN2,false>({c.a,c.b},calls,worst);
        std::vector<float> y(2048);std::vector<unsigned> receipt(hvxffn2::receiptWords,0x7fc0dead);
        auto a=tensor(32,128,c.a.data()),b=tensor(128,64,c.b.data()),o=tensor(32,64,y.data());
        auto r=tensor(1,receipt.size(),receipt.data(),QHPI_INT32);
        const QHPI_Tensor* in[]={&a,&b};QHPI_Tensor* out[]={&o,&r};
        demand(hvxffn2::kernel<true>(nullptr,2,out,2,in)==QHPI_ERROR_FATAL);
        demand(hvxffn2::kernel<false>(nullptr,1,out,2,in)==QHPI_ERROR_FATAL);
        for(auto u:receipt)demand(u==0x7fc0dead);
        // Negative workload controls are separate from the finite accuracy set.
        r.shape.dims[3]=closurecontract::receiptWords(closurecontract::FFN2);
        auto before=c.a;c.a[0]=INFINITY;
        demand((closurecustom::kernel<closurecontract::FFN2,true>(nullptr,2,out,2,in))==QHPI_ERROR_FATAL);
        c.a[0]=NAN;
        demand((closurecustom::kernel<closurecontract::FFN2,true>(nullptr,2,out,2,in))==QHPI_ERROR_FATAL);
        c.a=before;a.data=c.a.data();
        std::fill(c.a.begin(),c.a.end(),std::numeric_limits<float>::max());
        std::fill(c.b.begin(),c.b.end(),2.f);
        // Use the scalar receipt size; the overflow must be rejected after math.
        demand((closurecustom::kernel<closurecontract::FFN2,true>(nullptr,2,out,2,in))==QHPI_ERROR_FATAL);
    }
    std::printf("ordered_ffn2_host=PASS controls=12 calls=%u maxAbs=%.12g scalar_words=1 hvx_host_execution_rejected=1 negative_separate=1\n",calls,worst);
    return 0;
}
