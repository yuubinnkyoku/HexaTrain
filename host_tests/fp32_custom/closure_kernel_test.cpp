#define main old_segment_main
#include "segment_kernel_test.cpp"
#undef main
#include <stdexcept>
void demand(bool b){if(!b)throw std::runtime_error("closure host contract");}
template<closurecontract::Kind K,bool Obs> void test(const std::vector<one::FV>& x,unsigned& count,double& worst){
 constexpr auto c=closurecontract::contracts[K];constexpr unsigned yn=K==closurecontract::FFN2?2048:1024,nr=closurecontract::receiptWords(K);
 auto before=x;std::vector<QHPI_Tensor> it,ot;std::vector<const QHPI_Tensor*> in;std::vector<QHPI_Tensor*> out;std::vector<std::vector<uint32_t>> buf(c.outputs+1);it.reserve(c.inputs);ot.reserve(c.outputs+1);
 for(unsigned i=0;i<c.inputs;++i)it.push_back(tensor(K==closurecontract::FFN2&&i?128:32,K==closurecontract::FFN2?(i?64:128):K==closurecontract::QKVHeads?64:32,const_cast<float*>(x[i].data())));
 for(auto& t:it)in.push_back(&t);
 for(unsigned i=0;i<c.outputs+1;++i){buf[i].assign((i==c.outputs?nr:yn)+8,0xa55aa55a);std::fill(buf[i].begin()+4,buf[i].end()-4,0x7fc0dead);ot.push_back(tensor(i==c.outputs?1:32,i==c.outputs?nr:K==closurecontract::FFN2?64:32,buf[i].data()+4,i==c.outputs?QHPI_INT32:QHPI_FLOAT32));}
 for(auto& t:ot)out.push_back(&t);
 auto fn=closurecustom::kernel<K,Obs>;demand(fn(nullptr,c.outputs+Obs,out.data(),c.inputs,in.data())==QHPI_SUCCESS);
 std::vector<one::FV> actual;
 for(unsigned j=0;j<c.outputs;++j){one::Vec ref,fp;if constexpr(K==closurecontract::RawDQDK){unsigned h=j/2;ref=one::mm(one::d(x[h]),one::d(x[(j%2?4:2)+h]),32,32,32,j%2);fp=one::mm(one::d(x[h]),one::d(x[(j%2?4:2)+h]),32,32,32,j%2,false,true);}else if constexpr(K==closurecontract::FFN2){ref=one::mm(one::d(x[0]),one::d(x[1]),32,64,128);fp=one::mm(one::d(x[0]),one::d(x[1]),32,64,128,false,false,true);}else{ref=one::split(one::d(x[j/2]),j%2);fp=ref;}
  one::FV a(yn);std::memcpy(a.data(),buf[j].data()+4,yn*4);auto f=one::f(fp);demand(!std::memcmp(a.data(),f.data(),yn*4));auto e=mmfixture::ln::errors(ref,a);demand(!e.nonfinite&&e.absolute<=c.gate);worst=std::max(worst,e.absolute);actual.push_back(a);
 }
 if constexpr(Obs){auto rr=buf[c.outputs].data()+4;demand(rr[0]==0x43505500u+c.marker&&rr[1]>0&&rr[2]==c.inputWords&&rr[3]==c.outputWords&&rr[5]==81&&rr[6]==32);unsigned at=8;for(auto& a:x){demand(!std::memcmp(rr+at,a.data(),a.size()*4));at+=a.size();}for(auto& a:actual){demand(!std::memcmp(rr+at,a.data(),a.size()*4));at+=a.size();}demand(at==nr);}else for(auto p:std::vector<uint32_t>(buf.back().begin()+4,buf.back().end()-4))demand(p==0x7fc0dead);
 for(unsigned i=0;i<c.outputs+Obs;++i){for(unsigned j=0;j<4;++j)demand(buf[i][j]==0xa55aa55a&&buf[i][buf[i].size()-1-j]==0xa55aa55a);demand(std::find(buf[i].begin()+4,buf[i].end()-4,0x7fc0dead)==buf[i].end()-4);}demand(x==before);
 demand(fn(nullptr,c.outputs+Obs,out.data(),c.inputs-1,in.data())==QHPI_ERROR_FATAL);it[0].shape.dims[3]--;demand(fn(nullptr,c.outputs+Obs,out.data(),c.inputs,in.data())==QHPI_ERROR_FATAL);it[0].shape.dims[3]++;auto v=x;v[0][0]=INFINITY;it[0].data=v[0].data();demand(fn(nullptr,c.outputs+Obs,out.data(),c.inputs,in.data())==QHPI_ERROR_FATAL);++count;
}
#ifndef ORDERED_FFN2_HOST_EXTENSION
int main(int argc,char** argv){if(argc!=4)return 1;demand(qhpi_init()&&registered==41);auto fs=one::read(argv[1]);unsigned calls=0;double worst=0;for(auto& z:upstreamreference::fullUpstream(fs)){auto fp=one::reference(z,true);std::vector<one::FV>x(6);for(unsigned h=0;h<2;++h){auto p="h"+std::to_string(h)+"_";x[h]=one::f(fp.at(p+"ds"));x[2+h]=z.v.at(p+"k");x[4+h]=z.v.at(p+"q");}test<closurecontract::RawDQDK,true>(x,calls,worst);test<closurecontract::RawDQDK,false>(x,calls,worst);}
 auto stress=attentionfixture::matrix({});for(unsigned i=0;i<stress.size();++i){std::vector<one::FV>x(6);for(unsigned h=0;h<2;++h){x[h]=stress[i].a[h];x[2+h]=stress[i].b[h];x[4+h]=stress[(i+3)%stress.size()].b[1-h];}test<closurecontract::RawDQDK,true>(x,calls,worst);test<closurecontract::RawDQDK,false>(x,calls,worst);}
 std::printf("closure_host_raw=PASS calls=%u maxAbs=%.12g\n",calls,worst);
 // Frozen producer binary format is read independently of the device reader.
 FILE* f=std::fopen(argv[3],"rb");demand(f);uint32_t magic,n;demand(std::fread(&magic,4,1,f)==1&&std::fread(&n,4,1,f)==1&&magic==0x3152504c&&n==24);unsigned frozen=0;double fw=0;for(unsigned i=0;i<n;++i){uint64_t anchor;uint32_t k;demand(std::fread(&anchor,8,1,f)==1&&std::fread(&k,4,1,f)==1&&k==(i<16?1024u:128u));std::vector<one::FV>x={one::FV(32*k),one::FV(k*64)};one::FV residual(2048);for(auto& a:x)demand(std::fread(a.data(),4,a.size(),f)==a.size());demand(std::fread(residual.data(),4,residual.size(),f)==residual.size());if(k==128){test<closurecontract::FFN2,true>(x,calls,fw);test<closurecontract::FFN2,false>(x,calls,fw);++frozen;}}demand(frozen==8&&std::fgetc(f)==EOF);std::fclose(f);std::printf("closure_host_ffn_frozen=PASS cases=%u maxAbs=%.12g\n",frozen,fw);
 // QKV selection includes non-symmetric signed-zero words and every channel.
 for(unsigned kind=0;kind<12;++kind){std::vector<one::FV>x(3,one::FV(2048));for(unsigned j=0;j<3;++j)for(unsigned i=0;i<2048;++i)x[j][i]=kind==0?(i%2?-0.f:0.f):float(int((i*31+j*17+kind*11)%2001)-1000)*.013f;test<closurecontract::QKVHeads,true>(x,calls,worst);test<closurecontract::QKVHeads,false>(x,calls,worst);}
 // ReLU-like, sparse, mixed weight, cancellation and magnitude stress for K128.
 uint32_t seed=18211;for(unsigned kind=0;kind<16;++kind){std::vector<one::FV>x={one::FV(4096),one::FV(8192)};for(unsigned j=0;j<4096;++j){seed=1664525u*seed+1013904223u;x[0][j]=float(seed>>16)/65536.f;if(kind%8==1&&j%7)x[0][j]=0;if(kind%8==2)x[0][j]=1;if(kind%8==3)x[0][j]*=16;if(kind%8==4)x[0][j]*=1e-8f;if(kind%8==5)x[0][j]=.5f+(j%11)*.0001f;}for(unsigned j=0;j<8192;++j){seed=1664525u*seed+1013904223u;x[1][j]=float(int(seed>>16)-32768)/65536.f;if(kind%8==2)x[1][j]=(j/64%2?-1:1)*.125f;if(kind%8==3)x[1][j]*=.01f;}test<closurecontract::FFN2,true>(x,calls,worst);test<closurecontract::FFN2,false>(x,calls,worst);}
 std::printf("closure_host_kernel=PASS calls=%u maxAbs=%.12g fp_words=1 complete=1 guards=1 immutable=1 negative_contract=1\n",calls,worst);return 0;}
#endif
