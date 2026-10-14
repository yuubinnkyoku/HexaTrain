#define main attentionTestMain
#include "attention_kernel_test.cpp"
#undef main
struct Buffer{std::vector<uint32_t> words;QHPI_Tensor t;Buffer(unsigned r,unsigned c,bool integer=false):words(r*c+8,0xa55aa55a),t(tensor(r,c,words.data()+4,integer?QHPI_INT32:QHPI_FLOAT32)){std::fill(words.begin()+4,words.end()-4,0x7fc0dead);}
 bool guards()const{for(unsigned j=0;j<4;++j)if(words[j]!=0xa55aa55a||words[words.size()-1-j]!=0xa55aa55a)return false;return std::find(words.begin()+4,words.end()-4,0x7fc0dead)==words.end()-4;}};
unsigned pairs=0;
bool pair(QHPI_Plugin_Function observed,QHPI_Plugin_Function free,const std::vector<std::pair<unsigned,unsigned>>& is,const std::vector<one::FV>& inputs,const std::vector<std::pair<unsigned,unsigned>>& os,unsigned receipt){auto saved=inputs;std::vector<QHPI_Tensor> it;std::vector<const QHPI_Tensor*> ins;std::vector<Buffer> a,b;std::vector<QHPI_Tensor*> oa,ob;it.reserve(8);a.reserve(8);b.reserve(8);
 for(unsigned j=0;j<is.size();++j)it.push_back(tensor(is[j].first,is[j].second,const_cast<float*>(inputs[j].data())));for(auto& t:it)ins.push_back(&t);for(auto s:os){a.emplace_back(s.first,s.second);b.emplace_back(s.first,s.second);}a.emplace_back(1,receipt,true);for(auto& v:a)oa.push_back(&v.t);for(auto& v:b)ob.push_back(&v.t);
 if(observed(nullptr,oa.size(),oa.data(),ins.size(),ins.data())||free(nullptr,ob.size(),ob.data(),ins.size(),ins.data()))return false;
 for(unsigned j=0;j<b.size();++j)if(!a[j].guards()||!b[j].guards()||a[j].words!=b[j].words)return false;if(!a.back().guards()||inputs!=saved)return false;++pairs;return true;
}
int main(int argc,char** argv){if(argc!=2)return 1;for(auto& z:one::read(argv[1])){auto ref=one::reference(z,true);auto res=one::f(ref.at("residual")),h=one::f(ref.at("xhat")),inv=one::f(ref.at("inverse"));
 if(!pair(lncustom::forward,lncustom::forwardKernel<false>,{{32,64},{1,64},{1,64},{1,1}},{z.v.at("x"),z.v.at("gamma"),z.v.at("beta"),{1e-5f}},{{32,64},{32,64},{32,1}},4265))return 2;
 if(!pair(lncustom::backward<false>,lncustom::backward<false,false>,{{32,64},{1,64},{32,64},{32,1}},{z.v.at("dy"),z.v.at("gamma"),h,inv},{{32,64},{1,64},{1,64}},6248))return 3;
 if(!pair(kernel<1>,kernel<1,false>,{{32,64},{32,64}},{one::f(ref.at("dx")),z.v.at("residual")},{{32,64}},8200))return 4;
 if(!pair(mmcustom::kernel<false>,mmcustom::kernel<false,false>,{{32,64},{64,64}},{res,z.v.at("wo")},{{32,64}},8208))return 5;
 if(!pair(mmcustom::kernel<true>,mmcustom::kernel<true,false>,{{32,64},{32,64}},{z.v.at("context"),res},{{64,64}},8208))return 6;
 if(!pair(headcustom::kernel,headcustom::kernelImpl<false>,{{32,64}},{one::f(ref.at("dc"))},{{32,32},{32,32}},4104))return 7;
 if(!pair(broadcastcustom::kernel,broadcastcustom::kernelImpl<false>,{{32,32},{32,32},{32,1},{32,1}},{one::f(ref.at("h0_selected")),one::f(ref.at("h1_selected")),z.v.at("h0_gate"),z.v.at("h1_gate")},{{32,32},{32,32}},4168))return 8;
 }std::printf("observer_host_kernel=PASS pairs=%u all_outputs_bitwise=1 poison=1 canary=1 inputs_immutable=1 no_receipt_output=1\n",pairs);
}
