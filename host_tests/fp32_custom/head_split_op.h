#pragma once
// The single additional operation type: lossless T32/D64/H2 selector replacement.
namespace headcustom {
inline void copyHeads(const float* x,float* a,float* b){
 for(unsigned row=0;row<32;++row){std::memcpy(a+row*32,x+row*64,128);std::memcpy(b+row*32,x+row*64+32,128);}
}
template<bool Observe> uint32_t kernelImpl(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** o,uint32_t ni,const QHPI_Tensor* const* in){static unsigned calls=0;
 if(ni!=1||no!=(Observe?3u:2u)||!mmcustom::matrix(in[0],32,64)||!mmcustom::matrix(o[0],32,32)||!mmcustom::matrix(o[1],32,32))return QHPI_ERROR_FATAL;
 if constexpr(Observe)if(qhpi_tensor_type(o[2])!=QHPI_INT32||lncustom::count(o[2])!=4104)return QHPI_ERROR_FATAL;
 auto x=lncustom::data(in[0]),a=lncustom::data(o[0]),b=lncustom::data(o[1]);unsigned* r=nullptr;if constexpr(Observe)r=static_cast<unsigned*>(qhpi_tensor_raw_data(o[2]));if(!x||!a||!b||(Observe&&!r))return QHPI_ERROR_FATAL;auto started=lncustom::now();
 for(unsigned j=0;j<2048;++j)if(!std::isfinite(x[j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+8,x,8192);
 copyHeads(x,a,b);if constexpr(Observe){std::memcpy(r+2056,a,4096);std::memcpy(r+3080,b,4096);
 #ifdef __hexagon__
 r[0]=0x48383109u;
 #else
 r[0]=0x43505509u;
 #endif
 r[1]=++calls;r[2]=2048;r[3]=2;r[4]=lncustom::now()-started;r[5]=81;r[6]=32;r[7]=0;}return QHPI_SUCCESS;
}
uint32_t kernel(QHPI_RuntimeHandle* rt,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){return kernelImpl<true>(rt,no,out,ni,in);}
QHPI_Tensor_Signature_v1 ins[1]={},outs[3]={};QHPI_Kernel_v1 k{};QHPI_OpInfo_v1 op{};
bool registerOps(){ins[0]={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};k.function_name="HexaFp32::HeadSplit";k.function=kernel;k.resources=QHPI_RESOURCE_HVX;k.min_inputs=1;k.input_signature=ins;k.min_outputs=3;k.output_signature=outs;op.name=k.function_name;op.num_kernels=1;op.kernels=&k;return qhpi_register_ops_v1(1,&op,"HexaFp32")==QHPI_SUCCESS;}
}
