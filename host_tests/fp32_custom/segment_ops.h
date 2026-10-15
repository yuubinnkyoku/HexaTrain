#pragma once
// Diagnostic paired-head product and shared ordered NN projection geometry.
// Each graph invokes each specialization once, so counters have one writer.
namespace segmentcustom {
template<bool Projection,bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned calls=0;
 constexpr unsigned outputs=Projection?3:2,inputs=4,yn=Projection?2048:1024;
 constexpr unsigned sizes[4]={Projection?2048u:1024u,Projection?4096u:1024u,Projection?4096u:1024u,Projection?4096u:1024u};
 constexpr unsigned inputWords=Projection?14336:4096,total=8+inputWords+outputs*yn;
 if(ni!=inputs||no!=outputs+(Observe?1:0))return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<inputs;++i)if(!mmcustom::matrix(in[i],Projection?(i?64:32):32,Projection?64:32))return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<outputs;++i)if(!mmcustom::matrix(out[i],32,Projection?64:32))return QHPI_ERROR_FATAL;
 unsigned* r=nullptr;if constexpr(Observe){if(qhpi_tensor_type(out[outputs])!=QHPI_INT32||lncustom::count(out[outputs])!=total)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[outputs]));if(!r)return QHPI_ERROR_FATAL;}
 const float* x[4];float* y[outputs];auto start=lncustom::now();unsigned at=8;
 for(unsigned i=0;i<inputs;++i){x[i]=lncustom::data(in[i]);if(!x[i])return QHPI_ERROR_FATAL;for(unsigned j=0;j<sizes[i];++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+at,x[i],sizes[i]*4);at+=sizes[i];}
 for(unsigned i=0;i<outputs;++i){y[i]=lncustom::data(out[i]);if(!y[i])return QHPI_ERROR_FATAL;}
 auto math=lncustom::now();
 if constexpr(Projection){for(unsigned i=0;i<3;++i)mmcustom::ordered<32,64,64,false,false>(x[0],x[i+1],y[i]);}
 else for(unsigned h=0;h<2;++h)for(unsigned j=0;j<1024;++j)y[h][j]=elementProduct(x[h][j],x[h+2][j]);
 auto end=lncustom::now();for(unsigned i=0;i<outputs;++i){for(unsigned j=0;j<yn;++j)if(!std::isfinite(y[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+at+i*yn,y[i],yn*4);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383100u+(Projection?13:12);
 #else
 r[0]=0x43505500u+(Projection?13:12);
 #endif
 r[1]=++calls;r[2]=inputWords;r[3]=outputs*yn;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[4]={},outs[2][4]={};QHPI_Kernel_v1 kernels[4]={};QHPI_OpInfo_v1 ops[4]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};for(auto& a:outs)for(auto& s:a)s=ins[0];outs[0][2]=outs[1][3]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 static QHPI_Tensor_Signature_v1 freeOut[3];for(auto& s:freeOut)s=ins[0];
 const char* names[]={"HexaFp32::SoftmaxProduct","HexaFp32::SoftmaxProductNoReceipt","HexaFp32::ProjectionQKV","HexaFp32::ProjectionQKVNoReceipt"};
 QHPI_Plugin_Function fn[]={kernel<false,true>,kernel<false,false>,kernel<true,true>,kernel<true,false>};
 for(unsigned j=0;j<4;++j){auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=ins;k.min_outputs=(j<2?2:3)+(j%2?0:1);k.output_signature=j%2?freeOut:outs[j/2];ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}
 return qhpi_register_ops_v1(4,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
