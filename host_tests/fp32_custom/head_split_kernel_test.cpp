#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
#include "one_layer_reference.h"
struct QHPI_Tensor {QHPI_Shape shape;QHPI_Element_Type dtype;void* data;};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor* t){return t->shape;}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor* t){return t->dtype;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t){return t->data;}
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t,QHPI_OpInfo_v1*,const char*){return QHPI_SUCCESS;}
#include "package.cpp"
QHPI_Tensor t(unsigned rows,unsigned cols,void* p,unsigned dtype=QHPI_FLOAT32){return {{4,{1,1,rows,cols}},dtype,p};}
int main(int argc,char** argv){if(argc!=2)return 1;auto fs=one::read(argv[1]);for(auto& z:fs){auto input=one::f(one::reference(z,true).at("dc"));auto saved=input;one::FV a(1024,NAN),b=a;std::vector<unsigned> receipt(4104,0x7fc0dead);auto x=t(32,64,input.data()),at=t(32,32,a.data()),bt=t(32,32,b.data()),r=t(1,4104,receipt.data(),QHPI_INT32);const QHPI_Tensor* ins[]={&x};QHPI_Tensor* outs[]={&at,&bt,&r};
 if(headcustom::kernel(nullptr,3,outs,1,ins)!=QHPI_SUCCESS)return 2;for(unsigned h=0;h<2;++h){auto expected=one::f(one::split(one::d(input),h));if(expected!=(h?b:a)||std::memcmp(receipt.data()+2056+h*1024,expected.data(),4096))return 3;}if(std::memcmp(receipt.data()+8,input.data(),8192)||input!=saved)return 4;
 x.dtype=QHPI_FLOAT16;if(headcustom::kernel(nullptr,3,outs,1,ins)!=QHPI_ERROR_FATAL)return 5;x.dtype=QHPI_FLOAT32;x.shape.dims[3]=32;if(headcustom::kernel(nullptr,3,outs,1,ins)!=QHPI_ERROR_FATAL)return 6;x.shape.dims[3]=64;input[0]=INFINITY;if(headcustom::kernel(nullptr,3,outs,1,ins)!=QHPI_ERROR_FATAL)return 7;
 }std::puts("head_split_host_kernel=PASS cases=24 words=49152 receipts=PASS shape_dtype_nonfinite=PASS");}
