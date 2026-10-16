#pragma once
// Separate diagnostic ABI; the scalar FFN2 Op and receipt remain immutable.
namespace hvxffn2 {
constexpr unsigned marker=0x48383114u, inputWords=12288, outputWords=2048;
constexpr unsigned headerWords=16, receiptWords=headerWords+inputWords+outputWords;
#ifdef __hexagon__
extern "C" void hexatrain_ordered_hvx_ffn2(const float*,const float*,float*);
#endif
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,
    uint32_t ni,const QHPI_Tensor* const* in) {
    if(ni!=2||no!=(Observe?2u:1u))return QHPI_ERROR_FATAL;
    if(!mmcustom::matrix(in[0],32,128)||!mmcustom::matrix(in[1],128,64)||
       !mmcustom::matrix(out[0],32,64))return QHPI_ERROR_FATAL;
    const float* a=lncustom::data(in[0]);const float* b=lncustom::data(in[1]);
    float* y=lncustom::data(out[0]);if(!a||!b||!y)return QHPI_ERROR_FATAL;
    unsigned* r=nullptr;
    if constexpr(Observe){
        if(qhpi_tensor_type(out[1])!=QHPI_INT32||lncustom::count(out[1])!=receiptWords)return QHPI_ERROR_FATAL;
        r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[1]));if(!r)return QHPI_ERROR_FATAL;
    }
    // Host preparation never emulates successful HVX execution.
#ifndef __hexagon__
    return QHPI_ERROR_FATAL;
#else
    static unsigned calls=0;
    auto start=lncustom::now();
    for(unsigned j=0;j<4096;++j)if(!std::isfinite(a[j]))return QHPI_ERROR_FATAL;
    for(unsigned j=0;j<8192;++j)if(!std::isfinite(b[j]))return QHPI_ERROR_FATAL;
    if constexpr(Observe){std::memcpy(r+headerWords,a,16384);std::memcpy(r+headerWords+4096,b,32768);}
    auto math=lncustom::now();
    hexatrain_ordered_hvx_ffn2(a,b,y);
    auto end=lncustom::now();
    for(unsigned j=0;j<2048;++j)if(!std::isfinite(y[j]))return QHPI_ERROR_FATAL;
    if constexpr(Observe){
        std::memcpy(r+headerWords+inputWords,y,8192);
        r[0]=marker;r[1]=++calls;r[2]=inputWords;r[3]=outputWords;r[4]=end-math;
        r[5]=81;r[6]=32;r[7]=lncustom::now()-start;
        r[8]=reinterpret_cast<uintptr_t>(a)%128;r[9]=reinterpret_cast<uintptr_t>(b)%128;
        r[10]=reinterpret_cast<uintptr_t>(y)%128;r[11]=128;r[12]=0;
        r[13]=128;r[14]=64;r[15]=1;
    }
    return QHPI_SUCCESS;
#endif
}
inline bool registerOps(){
    static QHPI_Tensor_Signature_v1 in[2],out[2];
    for(auto& s:in)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
    out[0]=in[0];out[1]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
    static QHPI_Kernel_v1 kernels[2]={};static QHPI_OpInfo_v1 ops[2]={};
    const char* names[]={"HexaFp32::FFN2Hvx","HexaFp32::FFN2HvxNoReceipt"};
    QHPI_Plugin_Function fn[]={kernel<true>,kernel<false>};
    for(unsigned j=0;j<2;++j){auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];
        k.resources=QHPI_RESOURCE_HVX;k.min_inputs=2;k.input_signature=in;
        k.min_outputs=2-j;k.output_signature=out;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
    return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
