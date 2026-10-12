#pragma once
// Lane B is independent of the DSP LayerNorm kernels.
struct MatShape {const char* name;unsigned aRows,aCols,bRows,bCols;bool ta,tb,gradientLeft;};
inline std::vector<double> matRef(const std::vector<float>& a,const std::vector<float>& b,const MatShape& s,bool fp32){
 unsigned m=s.ta?s.aCols:s.aRows,k=s.ta?s.aRows:s.aCols,n=s.tb?s.bRows:s.bCols;
 std::vector<double> out(m*n);for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j){double sum=0;float fsum=0;
  for(unsigned h=0;h<k;++h){float x=a[s.ta?h*s.aCols+i:i*s.aCols+h],y=b[s.tb?j*s.bCols+h:h*s.bCols+j];
   if(fp32){volatile float product=x*y;fsum=fsum+product;}else sum+=double(x)*double(y);}
  out[i*n+j]=fp32?double(fsum):sum;
 }return out;
}
inline unsigned laneB(){
 unsigned failures=0;const MatShape shapes[]={
  {"dwo",32,64,32,64,true,false,false},{"dcontext",32,64,64,64,false,true,true},
  {"dw2",32,128,32,64,true,false,false},{"drelu",32,64,128,64,false,true,true},
  {"dembedding",32,1024,32,64,true,false,false}};
 for(auto s:shapes)for(bool copied:{false,true}){
  unsigned m=s.ta?s.aCols:s.aRows,k=s.ta?s.aRows:s.aCols,n=s.tb?s.bRows:s.bCols;
  DiagnosticGraph g(std::string(s.name)+(copied?"_native":"_direct"));auto W=QNN_TENSOR_TYPE_APP_WRITE,R=QNN_TENSOR_TYPE_APP_READ,N=QNN_TENSOR_TYPE_NATIVE;
  auto a=g.tensor("a",W,{s.aRows,s.aCols}),b=g.tensor("b",W,{s.bRows,s.bCols}),y=g.tensor("y",R,{m,n});
  auto grad=s.gradientLeft?a:b;unsigned mid=grad,receipt=0;
  if(copied){mid=g.tensor("native_gradient",N,s.gradientLeft?std::vector<uint32_t>{s.aRows,s.aCols}:std::vector<uint32_t>{s.bRows,s.bCols});receipt=g.tensor("receipt",R,{1,1,1,3*2048+8},QNN_DATATYPE_INT_32);g.node("copy","Copy","HexaFp32",{grad},{mid,receipt});}
  Qnn_Param_t p0=QNN_PARAM_INIT,p1=QNN_PARAM_INIT;p0.name=QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;p0.paramType=QNN_PARAMTYPE_SCALAR;p0.scalarParam.dataType=QNN_DATATYPE_BOOL_8;p0.scalarParam.bool8Value=s.ta;
  p1=p0;p1.name=QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;p1.scalarParam.bool8Value=s.tb;
  g.node("matmul",QNN_OP_MAT_MUL,"qti.aisw",{s.gradientLeft?mid:a,s.gradientLeft?b:mid},{y},{p0,p1});g.finalize();
  for(auto c:phonelm::qnn::lnvalidation::redesignCases(64)){
   c=expand(c,64,32);auto oracle=phonelm::qnn::lnvalidation::anchoredFp32Oracle(c,64);auto ideal=phonelm::qnn::lnvalidation::doubleReference(c,64);
   std::vector<float> gradient=oracle.dx;for(unsigned j=0;j<gradient.size();++j)gradient[j]+=c.dy[j];
   for(unsigned control=0;control<3;++control){
    std::vector<float> other(s.gradientLeft?s.bRows*s.bCols:s.aRows*s.aCols);uint32_t seed=97;
    for(unsigned j=0;j<other.size();++j){seed=1664525u*seed+1013904223u;other[j]=float(int(seed>>8)%2001-1000)/1000.f*(s.gradientLeft?.125f:1.f);}
    if(control==1){std::fill(other.begin(),other.end(),0.f);if(s.gradientLeft){for(unsigned j=0;j<s.bRows;++j)other[j*s.bCols+j%s.bCols]=1.f;}else for(unsigned r=0;r<s.aRows;++r)other[r*s.aCols+r%s.aCols]=1.f;}
    if(control==2){for(unsigned j=0;j<other.size();++j)other[j]=(j%2?.125f:-.125f);}
    if(std::string(s.name)=="dembedding"){std::fill(other.begin(),other.end(),0.f);for(unsigned r=0;r<32;++r)other[r*1024+(r%8)]=1.f;}
    auto aa=s.gradientLeft?gradient:other,bb=s.gradientLeft?other:gradient;
    auto ref=matRef(aa,bb,s,false),fp=matRef(aa,bb,s,true);auto out=g.run({{a,aa},{b,bb}});auto actual=decode(out[y]);
    if(copied){auto& r=out[receipt];if(r[0]!=0x48383100||r[2]!=2048||r[3]!=1){std::puts("RECEIPT_FAIL");std::exit(3);}for(unsigned j=0;j<2048;++j)if(r[8+j]!=word(gradient[j])||r[8+2048+j]||r[8+4096+j]){std::puts("COPY_PRECISION_FAIL");std::exit(3);}std::printf("kernel_receipt,0,%u,kernel_us=%u\n",r[1],r[4]);}
    auto e=phonelm::qnn::lnvalidation::errors(ref,actual),f=phonelm::qnn::lnvalidation::errors(fp,actual);bool pass=e.absolute<=.03&&e.nonfinite==0;failures+=!pass;
    std::printf("mmmetric,%s,%d,%s,%u,m=%u,k=%u,n=%u,abs=%.12g,rms=%.12g,relative=%.12g,fp_abs=%.12g,hash=%016llx,input_hash=%016llx,gate=%d,extra002=%d\n",s.name,copied,c.name.c_str(),control,m,k,n,e.absolute,e.rms,e.relative,f.absolute,(unsigned long long)hashWords(actual),(unsigned long long)hashWords(bb,hashWords(aa)),pass,e.absolute<=.02);
   }
  }
 }
 std::printf("summary,matmul,required_failures=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0\n",failures);return failures;
}
