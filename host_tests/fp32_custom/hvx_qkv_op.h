#pragma once
namespace hvxqkv {
constexpr unsigned marker=0x48383115u,inputWords=14336,outputWords=6144;
constexpr unsigned headerWords=24,receiptWords=headerWords+inputWords+outputWords;
#ifdef __hexagon__
extern "C" void hexatrain_ordered_hvx_qkv_nn64(const float*,const float*,float*);
#endif
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,
    uint32_t ni,const QHPI_Tensor* const* in){
    if(ni!=4||no!=(Observe?4u:3u))return QHPI_ERROR_FATAL;
    const float* x[4];float* y[3];
    for(unsigned i=0;i<4;++i){if(!mmcustom::matrix(in[i],i?64:32,64))return QHPI_ERROR_FATAL;x[i]=lncustom::data(in[i]);if(!x[i])return QHPI_ERROR_FATAL;}
    for(unsigned i=0;i<3;++i){if(!mmcustom::matrix(out[i],32,64))return QHPI_ERROR_FATAL;y[i]=lncustom::data(out[i]);if(!y[i])return QHPI_ERROR_FATAL;}
    // Outputs must be disjoint from all input ranges and from each other.
    auto overlaps=[](const void* a,unsigned an,const void* b,unsigned bn){auto p=reinterpret_cast<uintptr_t>(a),q=reinterpret_cast<uintptr_t>(b);return p<=q?q-p<an:p-q<bn;};
    for(unsigned i=0;i<3;++i){for(unsigned j=0;j<4;++j)if(overlaps(y[i],8192,x[j],j?16384:8192))return QHPI_ERROR_FATAL;
        for(unsigned j=0;j<i;++j)if(overlaps(y[i],8192,y[j],8192))return QHPI_ERROR_FATAL;}
    unsigned* r=nullptr;
    if constexpr(Observe){if(qhpi_tensor_type(out[3])!=QHPI_INT32||lncustom::count(out[3])!=receiptWords)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[3]));if(!r)return QHPI_ERROR_FATAL;
        for(unsigned j=0;j<4;++j)if(overlaps(r,receiptWords*4,x[j],j?16384:8192))return QHPI_ERROR_FATAL;
        for(auto p:y)if(overlaps(r,receiptWords*4,p,8192))return QHPI_ERROR_FATAL;}
#ifndef __hexagon__
    return QHPI_ERROR_FATAL;
#else
    static unsigned calls=0;auto start=lncustom::now();unsigned at=headerWords;
    for(unsigned i=0;i<4;++i){unsigned n=i?4096:2048;for(unsigned j=0;j<n;++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;
        if constexpr(Observe)std::memcpy(r+at,x[i],n*4);at+=n;}
    auto math=lncustom::now();
    for(unsigned i=0;i<3;++i)hexatrain_ordered_hvx_qkv_nn64(x[0],x[i+1],y[i]);
    auto end=lncustom::now();
    for(unsigned i=0;i<3;++i){for(unsigned j=0;j<2048;++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;
        if constexpr(Observe)std::memcpy(r+headerWords+inputWords+i*2048,y[i],8192);}
    if constexpr(Observe){
        r[0]=marker;r[1]=++calls;r[2]=inputWords;r[3]=outputWords;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;
        for(unsigned i=0;i<4;++i)r[8+i]=reinterpret_cast<uintptr_t>(x[i])%128;
        for(unsigned i=0;i<3;++i)r[12+i]=reinterpret_cast<uintptr_t>(y[i])%128;
        r[15]=128;r[16]=0;r[17]=64;r[18]=64;r[19]=3;r[20]=1;r[21]=r[22]=r[23]=0;
    }
    return QHPI_SUCCESS;
#endif
}
inline bool registerOps(){
    static QHPI_Tensor_Signature_v1 in[4],out[4];for(auto& s:in)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
    for(unsigned i=0;i<3;++i)out[i]=in[0];out[3]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
    static QHPI_Kernel_v1 kernels[2]={};static QHPI_OpInfo_v1 ops[2]={};
    const char* names[]={"HexaFp32::ProjectionQKVHvx","HexaFp32::ProjectionQKVHvxNoReceipt"};QHPI_Plugin_Function fn[]={kernel<true>,kernel<false>};
    for(unsigned i=0;i<2;++i){auto& k=kernels[i];k.function_name=names[i];k.function=fn[i];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=in;k.min_outputs=4-i;k.output_signature=out;
        ops[i].name=names[i];ops[i].num_kernels=1;ops[i].kernels=&k;}
    return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
