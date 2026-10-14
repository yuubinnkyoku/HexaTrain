#pragma once
// Diagnostic-only G1 T32/Dhead32 row broadcast. One launch handles both heads,
// avoiding shared receipt counters between concurrently scheduled head nodes.
namespace broadcastcustom {
constexpr unsigned receiptWords=4168; // header + A0/A1 + G0/G1 + Y0/Y1
uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** o,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned calls=0; // synchronous diagnostic executions only
 if(ni!=4||no!=3)return QHPI_ERROR_FATAL;
 for(unsigned h=0;h<2;++h)if(!mmcustom::matrix(in[h],32,32)||!mmcustom::matrix(in[2+h],32,1)||!mmcustom::matrix(o[h],32,32))return QHPI_ERROR_FATAL;
 auto rs=qhpi_tensor_shape(o[2]);
 if(qhpi_tensor_type(o[2])!=QHPI_INT32||rs.rank!=4||rs.dims[0]!=1||rs.dims[1]!=1||rs.dims[2]!=1||rs.dims[3]!=receiptWords)return QHPI_ERROR_FATAL;
 const float* a[2]={lncustom::data(in[0]),lncustom::data(in[1])};
 const float* g[2]={lncustom::data(in[2]),lncustom::data(in[3])};
 float* y[2]={lncustom::data(o[0]),lncustom::data(o[1])};auto r=static_cast<unsigned*>(qhpi_tensor_raw_data(o[2]));
 if(!a[0]||!a[1]||!g[0]||!g[1]||!y[0]||!y[1]||!r)return QHPI_ERROR_FATAL;
 auto start=lncustom::now();
 for(unsigned h=0;h<2;++h){for(unsigned i=0;i<1024;++i)if(!std::isfinite(a[h][i]))return QHPI_ERROR_FATAL;for(unsigned row=0;row<32;++row)if(!std::isfinite(g[h][row]))return QHPI_ERROR_FATAL;
  std::memcpy(r+8+h*1024,a[h],4096);std::memcpy(r+2056+h*32,g[h],128);}
 auto mathStart=lncustom::now();
 for(unsigned h=0;h<2;++h)for(unsigned row=0;row<32;++row)for(unsigned col=0;col<32;++col){volatile float product=a[h][row*32+col]*g[h][row];y[h][row*32+col]=product;}
 auto mathEnd=lncustom::now();
 for(unsigned h=0;h<2;++h){for(unsigned i=0;i<1024;++i)if(!std::isfinite(y[h][i]))return QHPI_ERROR_FATAL;std::memcpy(r+2120+h*1024,y[h],4096);}
 #ifdef __hexagon__
 r[0]=0x4838310au;
 #else
 r[0]=0x4350550au;
 #endif
 r[1]=++calls;r[2]=2048;r[3]=64;r[4]=mathEnd-mathStart;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[4]={},outs[3]={};QHPI_Kernel_v1 k{};QHPI_OpInfo_v1 op{};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};k.function_name="HexaFp32::BroadcastMultiply";k.function=kernel;k.resources=QHPI_RESOURCE_HVX;k.min_inputs=4;k.input_signature=ins;k.min_outputs=3;k.output_signature=outs;op.name=k.function_name;op.num_kernels=1;op.kernels=&k;return qhpi_register_ops_v1(1,&op,"HexaFp32")==QHPI_SUCCESS;}
}
