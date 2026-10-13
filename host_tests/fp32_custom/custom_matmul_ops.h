#pragma once
// Two diagnostic geometries only. Physical row-major inputs are never transposed.
namespace mmcustom {
unsigned serial=0;
bool matrix(const QHPI_Tensor* t,unsigned rows,unsigned cols){auto s=qhpi_tensor_shape(t);
 return qhpi_tensor_type(t)==QHPI_FLOAT32&&s.rank==4&&s.dims[0]==1&&s.dims[1]==1&&s.dims[2]==rows&&s.dims[3]==cols;
}
template<bool Dwo> uint32_t kernel(QHPI_RuntimeHandle*,uint32_t no,QHPI_Tensor** out,uint32_t ni,const QHPI_Tensor* const* in){
 constexpr unsigned M=Dwo?64:32,N=64,K=Dwo?32:64,AN=2048,BN=Dwo?2048:4096,YN=M*N;
 if(ni!=2||no!=2||!matrix(in[0],32,64)||!matrix(in[1],Dwo?32:64,64)||!matrix(out[0],M,N)||
    qhpi_tensor_type(out[1])!=QHPI_INT32||lncustom::count(out[1])!=16+AN+BN+YN)return QHPI_ERROR_FATAL;
 auto a=lncustom::data(in[0]),b=lncustom::data(in[1]),y=lncustom::data(out[0]);auto r=static_cast<unsigned*>(qhpi_tensor_raw_data(out[1]));
 if(!a||!b||!y||!r)return QHPI_ERROR_FATAL;
 auto receiptStarted=lncustom::now();for(unsigned i=0;i<AN;++i)if(!std::isfinite(a[i]))return QHPI_ERROR_FATAL;
 for(unsigned i=0;i<BN;++i)if(!std::isfinite(b[i]))return QHPI_ERROR_FATAL;
 std::memcpy(r+16,a,4*AN);std::memcpy(r+16+AN,b,4*BN);auto arithmeticStarted=lncustom::now();
 for(unsigned i=0;i<M;++i)for(unsigned j=0;j<N;++j){float sum=0;
  for(unsigned q=0;q<K;++q){float av=Dwo?a[q*64+i]:a[i*64+q],bv=Dwo?b[q*64+j]:b[j*64+q];
   volatile float product=av*bv;sum=sum+product;
  }y[i*N+j]=sum;
 }
 auto arithmeticEnded=lncustom::now();for(unsigned i=0;i<YN;++i)if(!std::isfinite(y[i]))return QHPI_ERROR_FATAL;
 std::memcpy(r+16+AN+BN,y,4*YN);
 #ifdef __hexagon__
 r[0]=0x48383100u+(Dwo?8:7);
 #else
 r[0]=0x43505500u+(Dwo?8:7);
 #endif
 r[1]=++serial;r[2]=AN;r[3]=BN;r[4]=unsigned(arithmeticEnded-arithmeticStarted);r[5]=81;r[6]=32;r[7]=0;
 r[8]=M;r[9]=N;r[10]=K;r[11]=Dwo;r[12]=!Dwo;r[13]=unsigned((arithmeticStarted-receiptStarted)+(lncustom::now()-arithmeticEnded));r[14]=YN;r[15]=0;
 return QHPI_SUCCESS;
}
QHPI_Tensor_Signature_v1 ins[2]={},outs[2]={};QHPI_Kernel_v1 kernels[2]={};QHPI_OpInfo_v1 ops[2]={};
bool registerOps(){for(auto& s:ins)s={QHPI_FLOAT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 outs[0]=ins[0];outs[1]={QHPI_INT32,QHPI_LAYOUT_FLAT_4,QHPI_STORAGE_DIRECT,QHPI_MEM_LOC_DDR_ONLY};
 const char* names[]={"HexaFp32::MatMulDContext","HexaFp32::MatMulDWo"};QHPI_Plugin_Function functions[]={kernel<false>,kernel<true>};
 for(unsigned i=0;i<2;++i){auto& k=kernels[i];k.function_name=names[i];k.function=functions[i];k.resources=QHPI_RESOURCE_HVX;
  k.min_inputs=2;k.input_signature=ins;k.min_outputs=2;k.output_signature=outs;ops[i].name=names[i];ops[i].num_kernels=1;ops[i].kernels=&k;
 }return qhpi_register_ops_v1(2,ops,"HexaFp32")==QHPI_SUCCESS;
}
}
