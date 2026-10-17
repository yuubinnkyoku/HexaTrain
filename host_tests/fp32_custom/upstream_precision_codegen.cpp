// Compile/disassembly feasibility only. Not registered, linked or run on DSP.
extern "C" double precision_add(double a,double b){return a+b;}
extern "C" double precision_mul(double a,double b){return a*b;}
extern "C" double precision_div(double a,double b){return a/b;}
extern "C" double sqrt(double);
extern "C" double precision_ln_inverse(double v){return 1/sqrt(v+double(1e-5f));}
extern "C" float precision_dot64(const float* a,const float* b,unsigned k){
 double s=0;for(unsigned q=0;q<k;++q)s+=double(a[q])*double(b[q]);return float(s);
}
extern "C" float precision_rounded_product_dot64(const float* a,const float* b,unsigned k){
 double s=0;for(unsigned q=0;q<k;++q){volatile float p=a[q]*b[q];s+=double(p);}return float(s);
}
extern "C" float precision_kahan(const float* a,const float* b,unsigned k){
 float s=0,c=0;for(unsigned q=0;q<k;++q){volatile float p=a[q]*b[q];float y=p-c,t=s+y;c=(t-s)-y;s=t;}return s;
}
