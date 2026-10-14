// Private bounded producer fragments, read-only checkpoint and fixed CPU trace.
#include "one_layer_reference.h"
#include "../../app/src/main/cpp/tiny_language_model_cpu.h"
#include "../../app/src/main/cpp/nicopedia_muon_checkpoint.h"
#include <fstream>
#include <iterator>
#include <cstring>
using namespace phonelm;
int main(int argc,char** argv){if(argc!=4)return 1;auto fs=one::read(argv[2]);tiny_lm::Config c;c.vocabularySize=1024;c.tokens=32;c.dimension=64;c.feedForwardDimension=128;c.numLayers=19;c.numHeads=2;c.attentionGate=tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
 std::ifstream input(argv[1],std::ios::binary);std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});nicopedia_muon_checkpoint::Checkpoint ck;std::string error;qnn::TinyTransformerParameters checkpoint;if(!nicopedia_muon_checkpoint::decodeCheckpoint(bytes,&ck,&error)||ck.identity.globalStep!=40000||!nicopedia_muon_checkpoint::extractParameters(ck,c,1,&checkpoint,&error))return 2;
 FILE* out=std::fopen(argv[3],"wb");if(!out)return 3;uint32_t header[]={0x3152504c,24};std::fwrite(header,4,2,out);
 for(unsigned i=0;i<24;++i){c.numLayers=i<16?1:19;auto model=i<16?tiny_lm::initialParameters(c,1+i):checkpoint;if(i>=8&&i<16){for(auto& x:model.wo)x*=8.f;for(auto& x:model.wv)x*=8.f;}std::vector<uint32_t> ids(32);for(unsigned r=0;r<32;++r)ids[r]=(i%8==5?r%4:(r*73+i*31)%1024);auto hot=tiny_lm::oneHot(ids,1024);auto trace=tiny_lm::forwardTraceGeneralized(c,hot,model);auto& state=trace.layers.back();if(state.input!=fs[i].v.at("layerinput"))return 4;
  uint64_t hash=14695981039346656037ull;for(float f:state.input){uint32_t u;std::memcpy(&u,&f,4);for(unsigned j=0;j<4;++j){hash^=(u>>(j*8))&255;hash*=1099511628211ull;}}std::fwrite(&hash,8,1,out);uint32_t k=i<16?1024:128;std::fwrite(&k,4,1,out);
  one::FV a,b,res;if(i<16){a=hot;b=model.tokenEmbedding;res.resize(2048);for(unsigned r=0;r<32;++r)for(unsigned j=0;j<64;++j){float f=std::pow(10000.f,-float(j&~1u)/64.f);res[r*64+j]=((j&1u)?std::cos(float(r)*f):std::sin(float(r)*f))*.05f;}}else{a=trace.layers.at(17).relu;b=model.layers.at(16).w2;res=trace.layers.at(17).residual1;}
  // Identity check also catches changes in the production positional definition.
  auto x=one::f(one::mm(one::d(a),one::d(b),32,64,k));for(unsigned j=0;j<2048;++j)x[j]+=res[j];if(x!=state.input)return 5;
  for(auto& v:{a,b,res})std::fwrite(v.data(),4,v.size(),out);
 }std::fclose(out);std::puts("ln1_producer_fixture=PASS cases=24 checkpoint_read_only=1 exact_cpu_trace_identity=1");
}
