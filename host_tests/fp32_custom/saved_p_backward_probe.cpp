#include "saved_p_backward_graph.h"
#include "saved_p_backward_reference_lib.h"
#include <algorithm>
#include <stdexcept>
namespace savedp {
using Words=std::vector<uint32_t>;
FILE* rawEvidence=nullptr;
void writeRaw(const unsigned* values,unsigned count) {
  if(!rawEvidence||std::fwrite(values,4,count,rawEvidence)!=count)throw std::runtime_error("raw evidence write");
}
using Pair=std::array<std::array<std::vector<float>,2>,2>; // head, dp/dContext
std::vector<Pair> loadGradients(const char* path) {
  FILE* f=std::fopen(path,"rb");if(!f)throw std::runtime_error("gradient file open");
  unsigned h[4];s2fixture::read(f,h,4);
  if(h[0]!=0x53425001||h[1]!=1||h[2]!=25||h[3]!=8)throw std::runtime_error("gradient header");
  std::vector<Pair> out(33);
  for(auto& g:out)for(auto& head:g)for(auto& v:head){v.resize(1024);s2fixture::read(f,v.data(),1024);}
  if(std::fgetc(f)!=EOF)throw std::runtime_error("gradient trailing data");std::fclose(f);return out;
}
struct Error {double max=0,rms=0;unsigned coordinate=0;};
template<class T> Error error(const std::vector<float>& x,const std::vector<T>& y) {
  if(x.size()!=y.size())throw std::runtime_error("reference size");Error e;
  for(unsigned i=0;i<x.size();++i) {
    if(!std::isfinite(x[i])||!std::isfinite(y[i]))throw std::runtime_error("nonfinite reference/output");
    double d=std::fabs(double(x[i])-double(y[i]));if(d>e.max){e.max=d;e.coordinate=i;}e.rms+=d*d;
  }e.rms=std::sqrt(e.rms/x.size());return e;
}
void words(const char* field,unsigned h,const Words& v) {
  unsigned code=!std::strcmp(field,"P")?1:!std::strcmp(field,"gradient")?2:!std::strcmp(field,"output")?3:
    !std::strcmp(field,"product")?4:!std::strcmp(field,"dot")?5:!std::strcmp(field,"centered")?6:0;
  if(!code)throw std::runtime_error("raw field identity");
  unsigned header[]={code,h,unsigned(v.size())};writeRaw(header,3);writeRaw(v.data(),v.size());
}
} // namespace savedp
#ifndef ONE_GRAPH_HOST_AUDIT
int main(int argc,char** argv) {
  if(argc!=7)return 1;
  try {
    auto cs=doubleqkfixture::load(argv[1]);auto ms=mixedfixture::load(argv[2]);auto ls=s2fixture::load(argv[3]);
    auto gs=savedp::loadGradients(argv[4]);std::string requested=argv[5];unsigned limit=std::stoul(argv[6]);
    savedp::Mode mode;
    if(requested=="standalone_softmax")mode=savedp::Mode::StandaloneSoftmax;
    else if(requested=="standalone_dvalue")mode=savedp::Mode::StandaloneDValue;
    else throw std::runtime_error("native execution requires a separately certified standalone permit");
    // Native graphs exist for host construction audit, but cannot bypass the standalone gate.
    savedp::rawEvidence=std::fopen("saved-p-words.bin","wbx");
    if(!savedp::rawEvidence)throw std::runtime_error("exclusive binary evidence creation");
    unsigned header[]={0x53425002,1,mode==savedp::Mode::StandaloneSoftmax?0u:1u,limit};
    savedp::writeRaw(header,4);
    initializeDoubleQk();unsigned failed=0,pairs=0,variantMismatch=0;
    std::map<std::string,savedp::Words> anchors;
    for(bool obs:{true,false})for(bool tap:{true,false}) {
      savedp::Graph z(mode,obs,tap);
      for(unsigned fi=0;fi<std::min(60u,limit);++fi)for(unsigned gi=0;gi<(fi<44?25u:8u);++gi) {
        auto& grad=gs[fi<44?gi:25+gi];
        std::map<unsigned,std::vector<float>> in;
        for(unsigned h=0;h<2;++h){in[z.p[h]]=cs[fi].hostP[h];if(z.softmax)in[z.dp[h]]=grad[h][0];if(z.dvalue)in[z.dc[h]]=grad[h][1];}
        auto out=z.g.run(in);
        for(unsigned h=0;h<2;++h) {
          auto& p=cs[fi].hostP[h];auto& dp=grad[h][0];auto& dc=grad[h][1];
          auto id=z.softmax?z.ds[h]:z.dv[h];auto value=decode(out.at(id));
          auto a=z.softmax?phonelm::saved_p_backward::SoftmaxBackwardDouble(p,dp):phonelm::saved_p_backward::DValueDouble(p,dc);
          auto b=z.softmax?phonelm::saved_p_backward::SoftmaxBackwardOrderedFp32(p,dp):phonelm::saved_p_backward::DValueOrderedFp32(p,dc);
          auto ea=savedp::error(value,a),eb=savedp::error(value,b);double gate=z.softmax?.002:.01;
          std::string key=std::to_string(fi)+":"+std::to_string(gi)+":"+std::to_string(h);
          unsigned mismatches=0;if(obs&&tap)anchors[key]=out.at(id);else {
            auto& anchor=anchors.at(key);for(unsigned k=0;k<anchor.size();++k)mismatches+=anchor[k]!=out.at(id)[k];
          }
          bool pass=ea.max<=gate&&!mismatches;failed+=!pass;variantMismatch+=mismatches;++pairs;
          std::printf("{\"kind\":\"backward_pair\",\"mode\":\"%s\",\"observed\":%d,\"tap\":%d,\"forward\":%u,\"gradient\":%u,\"head\":%u,\"set\":\"%s\",\"maxAbs\":%.17g,\"rms\":%.17g,\"coordinate\":%u,\"ordered_fp32_delta\":%.17g,\"variant_word_mismatches\":%u,\"output_hash\":\"%016llx\",\"accuracy_gate\":%.17g,\"pass\":%s}\n",
            requested.c_str(),obs,tap,fi,gi,h,fi<44?"core":"holdout",ea.max,ea.rms,ea.coordinate,eb.max,mismatches,(unsigned long long)hashWords(value),gate,pass?"true":"false");
          // Private complete operand/result evidence. No receipt claim for standard primitives.
          std::printf("case,%s,%d,%d,%u,%u,%u\n",requested.c_str(),obs,tap,fi,gi,h);
          unsigned rawKey[]={unsigned(obs),unsigned(tap),fi,gi,h,z.softmax&&tap?6u:3u};
          savedp::writeRaw(rawKey,6);
          savedp::Words pw;for(float v:p)pw.push_back(word(v));savedp::words("P",h,pw);
          savedp::Words gw;for(float v:z.softmax?dp:dc)gw.push_back(word(v));savedp::words("gradient",h,gw);
          savedp::words("output",h,out.at(id));
          if(z.softmax&&tap) {
            std::vector<float> product(1024),dot(32),center(1024);
            for(unsigned i=0;i<1024;++i)product[i]=dp[i]*p[i];
            auto ep=savedp::error(decode(out.at(z.product[h])),product);
            for(unsigned r=0;r<32;++r){float s=0;for(unsigned c=0;c<32;++c)s=s+product[r*32+c];dot[r]=s;}
            auto ed=savedp::error(decode(out.at(z.dot[h])),dot);
            for(unsigned i=0;i<1024;++i)center[i]=dp[i]-dot[i/32];
            auto ec=savedp::error(decode(out.at(z.centered[h])),center);
            std::printf("{\"kind\":\"boundary\",\"forward\":%u,\"gradient\":%u,\"head\":%u,\"observed\":%d,\"product_ordered_delta\":%.17g,\"dot_ordered_delta\":%.17g,\"centered_ordered_delta\":%.17g}\n",fi,gi,h,obs,ep.max,ed.max,ec.max);
            savedp::words("product",h,out.at(z.product[h]));savedp::words("dot",h,out.at(z.dot[h]));savedp::words("centered",h,out.at(z.centered[h]));
          }
        }
      }
    }
    if(std::fclose(savedp::rawEvidence))throw std::runtime_error("raw evidence close");
    savedp::rawEvidence=nullptr;
    closeProbe();
    std::printf("summary,saved_p_backward,mode=%s,required_failures=%u,pairs=%u,variant_word_mismatches=%u,cpu_arithmetic_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary=1,teardown=1,native_execution=0\n",requested.c_str(),failed,pairs,variantMismatch);
    return failed?4:0;
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 5;}
}
#endif
