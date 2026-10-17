#pragma once
// Diagnostic pairing only: both additions invoke the EXISTING AddNoReceipt
// arithmetic body. No new Add arithmetic, mask semantics or approximation.
namespace forwardmask {
constexpr unsigned receiptWords=5128;
template<bool Observe> uint32_t kernel(QHPI_RuntimeHandle* runtime,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned serial=0;if(ni!=3||no!=2+unsigned(Observe))return QHPI_ERROR_FATAL;
 const float* x[3];float* y[2];for(unsigned j=0;j<3;++j){if(!mmcustom::matrix(in[j],32,32))return QHPI_ERROR_FATAL;x[j]=lncustom::data(in[j]);if(!x[j])return QHPI_ERROR_FATAL;}
 for(unsigned j=0;j<2;++j){if(!mmcustom::matrix(out[j],32,32))return QHPI_ERROR_FATAL;y[j]=lncustom::data(out[j]);if(!y[j])return QHPI_ERROR_FATAL;}
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[2])!=QHPI_INT32||lncustom::count(out[2])!=receiptWords)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!r)return QHPI_ERROR_FATAL;}
 auto overlap=[](const void* a,unsigned na,const void* b,unsigned nb){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<y+nb&&y<x+na;};
 for(unsigned h=0;h<2;++h){for(unsigned j=0;j<3;++j)if(overlap(y[h],4096,x[j],4096))return QHPI_ERROR_FATAL;if(overlap(y[h],4096,y[1-h],4096))return QHPI_ERROR_FATAL;if constexpr(Observe)if(overlap(r,receiptWords*4,y[h],4096))return QHPI_ERROR_FATAL;}
 for(unsigned j=0;j<3;++j){if constexpr(Observe)if(overlap(r,receiptWords*4,x[j],4096))return QHPI_ERROR_FATAL;for(unsigned i=0;i<1024;++i)if(!std::isfinite(x[j][i]))return QHPI_ERROR_FATAL;}
 for(unsigned row=0;row<32;++row)for(unsigned col=0;col<32;++col){uint32_t bits;std::memcpy(&bits,x[2]+row*32+col,4);if(bits!=(col>row?0xc61c4000u:0u))return QHPI_ERROR_FATAL;}
 auto start=lncustom::now();if constexpr(Observe)for(unsigned j=0;j<3;++j)std::memcpy(r+8+j*1024,x[j],4096);
 auto math=lncustom::now();for(unsigned h=0;h<2;++h){const QHPI_Tensor* args[]={in[h],in[2]};QHPI_Tensor* result[]={out[h]};if(::kernel<1,false>(runtime,1,result,2,args)!=QHPI_SUCCESS)return QHPI_ERROR_FATAL;}auto end=lncustom::now();
 for(unsigned h=0;h<2;++h){for(unsigned i=0;i<1024;++i)if(!std::isfinite(y[h][i]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+3080+h*1024,y[h],4096);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383119u;
 #else
 r[0]=0x43505519u;
 #endif
 r[1]=++serial;r[2]=3072;r[3]=2048;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[3]={},outs[3]={};QHPI_Kernel_v1 kernels[2]={};QHPI_OpInfo_v1 ops[2]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};const char* names[]={"HexaFp32::ForwardMaskAdd","HexaFp32::ForwardMaskAddNoReceipt"};for(unsigned j=0;j<2;++j){auto& k=kernels[j];k.function_name=names[j];k.function=j?kernel<false>:kernel<true>;k.resources=QHPI_RESOURCE_HVX;k.min_inputs=3;k.input_signature=ins;k.min_outputs=3-j;k.output_signature=outs;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;}
}
