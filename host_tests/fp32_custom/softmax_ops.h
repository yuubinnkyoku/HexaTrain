#pragma once
// Added only after independent accurate-input standard stages exceeded gates.
// Increasing-column FP32 sum, no contraction, no extra mask/scale/clipping.
namespace softmaxcustom {
inline unsigned bits(float x){unsigned v;std::memcpy(&v,&x,4);return v;}
template<unsigned Kind,bool Observe> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 static unsigned serial=0;constexpr unsigned inputs=Kind==0?2:4,yn=Kind==0?32:1024,inputWords=Kind==0?2048:Kind==1?2112:4096,total=8+inputWords+(Kind==2?6208:2*yn);
 if(ni!=inputs||no!=(Observe?3u:2u))return QHPI_ERROR_FATAL;const float* x[4]={};float* y[2]={};unsigned* r=nullptr;
 if constexpr(Observe){if(qhpi_tensor_type(out[2])!=QHPI_INT32||lncustom::count(out[2])!=total)return QHPI_ERROR_FATAL;r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[2]));if(!r)return QHPI_ERROR_FATAL;}
 auto start=lncustom::now();unsigned at=8;for(unsigned i=0;i<inputs;++i){unsigned cols=Kind==1&&i>=2?1:32;if(!mmcustom::matrix(in[i],32,cols))return QHPI_ERROR_FATAL;x[i]=lncustom::data(in[i]);if(!x[i])return QHPI_ERROR_FATAL;for(unsigned j=0;j<32*cols;++j)if(!std::isfinite(x[i][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+at,x[i],128*cols);at+=32*cols;}
 for(unsigned h=0;h<2;++h){if(!mmcustom::matrix(out[h],32,Kind==0?1:32))return QHPI_ERROR_FATAL;y[h]=lncustom::data(out[h]);if(!y[h])return QHPI_ERROR_FATAL;}
 auto math=lncustom::now();for(unsigned h=0;h<2;++h)for(unsigned row=0;row<32;++row){
  if constexpr(Kind==0){float sum=0;for(unsigned col=0;col<32;++col)sum=sum+x[h][row*32+col];y[h][row]=sum;}
  if constexpr(Kind==1)for(unsigned col=0;col<32;++col)y[h][row*32+col]=x[h][row*32+col]-x[h+2][row];
  if constexpr(Kind==2){float product[32],sum=0;for(unsigned col=0;col<32;++col){unsigned j=row*32+col;product[col]=elementProduct(x[h][j],x[h+2][j]);sum=sum+product[col];}for(unsigned col=0;col<32;++col){unsigned j=row*32+col;volatile float centered=x[h][j]-sum;y[h][j]=elementProduct(x[h+2][j],centered);if constexpr(Observe){r[4104+h*1024+j]=bits(product[col]);r[6216+h*1024+j]=bits(centered);}}if constexpr(Observe)r[6152+h*32+row]=bits(sum);}
 }
 auto end=lncustom::now();for(unsigned h=0;h<2;++h){for(unsigned j=0;j<yn;++j)if(!std::isfinite(y[h][j]))return QHPI_ERROR_FATAL;if constexpr(Observe)std::memcpy(r+(Kind==2?8264:8+inputWords)+h*yn,y[h],yn*4);}
 if constexpr(Observe){
 #ifdef __hexagon__
 r[0]=0x48383100u+14+Kind;
 #else
 r[0]=0x43505500u+14+Kind;
 #endif
 r[1]=++serial;r[2]=inputWords;r[3]=Kind==2?6208:2*yn;r[4]=end-math;r[5]=81;r[6]=32;r[7]=lncustom::now()-start;}
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[4]={},outs[3]={},freeOut[2]={};QHPI_Kernel_v1 kernels[6]={};QHPI_OpInfo_v1 ops[6]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};outs[0]=outs[1]=freeOut[0]=freeOut[1]=ins[0];outs[2]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};const char* names[]={"HexaFp32::SoftmaxReduce","HexaFp32::SoftmaxReduceNoReceipt","HexaFp32::SoftmaxCenter","HexaFp32::SoftmaxCenterNoReceipt","HexaFp32::SoftmaxBackward","HexaFp32::SoftmaxBackwardNoReceipt"};QHPI_Plugin_Function fn[]={kernel<0,true>,kernel<0,false>,kernel<1,true>,kernel<1,false>,kernel<2,true>,kernel<2,false>};for(unsigned j=0;j<6;++j){auto& k=kernels[j];k.function_name=names[j];k.function=fn[j];k.resources=QHPI_RESOURCE_HVX;k.min_inputs=j<2?2:4;k.input_signature=ins;k.min_outputs=j%2?2:3;k.output_signature=j%2?freeOut:outs;ops[j].name=names[j];ops[j].num_kernels=1;ops[j].kernels=&k;}return qhpi_register_ops_v1(6,ops,"HexaFp32")==QHPI_SUCCESS;}
}
