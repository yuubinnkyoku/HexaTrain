#include "one_layer_reference.h"
#include "../../app/src/main/cpp/tiny_language_model_cpu.h"
#include "../../app/src/main/cpp/nicopedia_muon_checkpoint.h"
#include <fstream>
#include <iterator>
#include <iostream>
using namespace phonelm;
one::Fixture make(unsigned index,const tiny_lm::Config& config,const qnn::TinyTransformerParameters& model,bool checkpoint){
 auto ids=std::vector<uint32_t>(32);for(unsigned r=0;r<32;++r)ids[r]=(index%8==5?r%4:(r*73+index*31)%1024);
 auto trace=tiny_lm::forwardTraceGeneralized(config,tiny_lm::oneHot(ids,1024),model);unsigned layer=config.numLayers-1;auto& t=trace.layers.at(layer);auto& p=layer?model.layers.at(layer-1):static_cast<const qnn::TinyTransformerLayerParameters&>(model);
 one::Fixture z;z.index=index;z.checkpoint=checkpoint;auto& v=z.v;
 v["x"]=t.residual1;v["gamma"]=p.gamma2;v["beta"]=p.beta2;v["wo"]=p.wo;v["context"]=t.context;v["ln1"]=t.ln1;v["layerinput"]=t.input;
 v["wq"]=p.wq;v["wk"]=p.wk;v["wv"]=p.wv;v["wg"]=p.attentionGateWeight;v["w1"]=p.w1;v["w2"]=p.w2;v["gates"]=t.gates;
 unsigned recipe=index%8;uint32_t seed=97+index*7919;v["residual"].resize(2048);for(unsigned j=0;j<2048;++j){seed=1664525u*seed+1013904223u;float x=float(int(seed>>8)%2001-1000)/1000.f;
  if(recipe==1)x*=8.f;if(recipe==2)x*=.0001f;if(recipe==3)x=(j%2?1.f:-1.f)*(1.f+float(j%11)*.03125f);if(recipe==4)x*=j%3? .001f:8.f;if(recipe==5)x=float(int(j%13)-6)*.125f;if(recipe==7)x=0;v["residual"][j]=x;
 }
 auto drelu=one::mm(one::d(v["residual"]),one::d(p.w2),32,128,64,false,true);for(unsigned j=0;j<4096;++j)if(t.ff1[j]<=0)drelu[j]=0;
 v["dy"]=one::f(one::mm(drelu,one::d(p.w1),32,64,128,false,true));
 for(unsigned h=0;h<2;++h){auto pr="h"+std::to_string(h)+"_";v[pr+"q"]=one::f(one::split(one::d(t.q),h));v[pr+"k"]=one::f(one::split(one::d(t.k),h));v[pr+"v"]=one::f(one::split(one::d(t.v),h));
  auto scores=one::mul(one::mm(one::d(v[pr+"q"]),one::d(v[pr+"k"]),32,32,32,false,true),{double(1.f/std::sqrt(32.f))});for(unsigned r=0;r<32;++r)for(unsigned c=r+1;c<32;++c)scores[r*32+c]-=10000.;v[pr+"p"]=one::f(one::softmax(scores));
  v[pr+"context"]=one::f(one::mm(one::d(v[pr+"p"]),one::d(v[pr+"v"]),32,32,32));auto& gate=v[pr+"gate"];for(unsigned r=0;r<32;++r)gate.push_back(t.gates[r*2+h]);
 }return z;
}
int main(int argc,char** argv){if(argc!=3)return 1;tiny_lm::Config c;c.vocabularySize=1024;c.tokens=32;c.dimension=64;c.feedForwardDimension=128;c.numLayers=19;c.numHeads=2;c.attentionGate=tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
 std::ifstream input(argv[1],std::ios::binary);std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});nicopedia_muon_checkpoint::Checkpoint ck;std::string error;
 if(!nicopedia_muon_checkpoint::decodeCheckpoint(bytes,&ck,&error)||ck.identity.globalStep!=40000){std::cerr<<error;return 2;}qnn::TinyTransformerParameters checkpoint;
 if(!nicopedia_muon_checkpoint::extractParameters(ck,c,1,&checkpoint,&error)){std::cerr<<error;return 3;}
 std::vector<one::Fixture> fs;for(unsigned i=0;i<24;++i){auto model=tiny_lm::initialParameters(c,1+i);if(i<16){c.numLayers=1;model=tiny_lm::initialParameters(c,1+i);if(i>=8){for(auto& x:model.wo)x*=8.f;for(auto& x:model.wv)x*=8.f;}}else c.numLayers=19;
  auto z=make(i,c,i>=16?checkpoint:model,i>=16);auto ref=one::reference(z),fp=one::reference(z,true);double max=0;for(auto& p:ref){auto e=mmfixture::ln::errors(p.second,fp.at(p.first));if(e.nonfinite) return 4;max=std::max(max,e.absolute);}std::printf("host_fixture,%u,checkpoint=%u,fp_double_max=%.12g\n",i,z.checkpoint,max);fs.push_back(z);
 }
 FILE* file=std::fopen(argv[2],"wb");if(!file)return 5;one::write(file,fs);std::fclose(file);auto loaded=one::read(argv[2]);for(unsigned i=0;i<fs.size();++i)if(loaded[i].v!=fs[i].v)return 6;
 std::puts("one_layer_fixture_audit=PASS,count=24,synthetic=16,checkpoint_parameters_layer18=8,training_updates=0");
}
