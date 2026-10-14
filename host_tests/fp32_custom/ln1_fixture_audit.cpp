// Read-only extension of the frozen one-layer fixture derivation. No data order
// or checkpoint codec changes. Output remains private under build/reports.
#include "one_layer_reference.h"
#include "../../app/src/main/cpp/tiny_language_model_cpu.h"
#include "../../app/src/main/cpp/nicopedia_muon_checkpoint.h"
#include <fstream>
#include <iterator>
#include <cstring>
using namespace phonelm;
uint64_t ln1Anchor(const one::Fixture& z){uint64_t hash=14695981039346656037ull;for(auto key:{"layerinput","ln1"})for(float f:z.v.at(key)){uint32_t u;std::memcpy(&u,&f,4);for(unsigned j=0;j<4;++j){hash^=(u>>(8*j))&255;hash*=1099511628211ull;}}return hash;}
int main(int argc,char** argv){if(argc!=4)return 1;auto fs=one::read(argv[2]);tiny_lm::Config c;c.vocabularySize=1024;c.tokens=32;c.dimension=64;c.feedForwardDimension=128;c.numLayers=19;c.numHeads=2;c.attentionGate=tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
 std::ifstream input(argv[1],std::ios::binary);std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});nicopedia_muon_checkpoint::Checkpoint ck;std::string error;qnn::TinyTransformerParameters checkpoint;if(!nicopedia_muon_checkpoint::decodeCheckpoint(bytes,&ck,&error)||ck.identity.globalStep!=40000||!nicopedia_muon_checkpoint::extractParameters(ck,c,1,&checkpoint,&error))return 2;
 FILE* out=std::fopen(argv[3],"wb");if(!out)return 3;uint32_t header[]={0x314e4c47,24};std::fwrite(header,4,2,out);
 for(unsigned i=0;i<24;++i){c.numLayers=i<16?1:19;auto model=i<16?tiny_lm::initialParameters(c,1+i):checkpoint;if(i>=8&&i<16){for(auto& x:model.wo)x*=8.f;for(auto& x:model.wv)x*=8.f;}std::vector<uint32_t> ids(32);for(unsigned r=0;r<32;++r)ids[r]=(i%8==5?r%4:(r*73+i*31)%1024);auto trace=tiny_lm::forwardTraceGeneralized(c,tiny_lm::oneHot(ids,1024),model);auto& state=trace.layers.back();if(state.input!=fs[i].v.at("layerinput")||state.ln1!=fs[i].v.at("ln1"))return 4;const auto& p=i<16?static_cast<const qnn::TinyTransformerLayerParameters&>(model):model.layers.at(17);uint64_t anchor=ln1Anchor(fs[i]);std::fwrite(&anchor,8,1,out);std::fwrite(p.gamma1.data(),4,64,out);std::fwrite(p.beta1.data(),4,64,out);
 }std::fclose(out);std::puts("ln1_parameters_fixture=PASS cases=24 frozen_state_identity=1 checkpoint_read_only=1");return 0;
}
