#include "qnn_reproducibility.h"

#include "qnn_runtime.h"
#include "../tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace phonelm::qnn {
namespace {

// Small self-contained SHA-256 implementation.  Keeping it here avoids adding a
// crypto library dependency to the Android native target.
class Sha256 {
 public:
  void update(const std::uint8_t* data, size_t size) {
    bits_ += std::uint64_t(size) * 8;
    while (size) {
      const size_t count = std::min(size, sizeof(block_) - used_);
      std::memcpy(block_.data() + used_, data, count);
      used_ += count; data += count; size -= count;
      if (used_ == block_.size()) { transform(block_.data()); used_ = 0; }
    }
  }
  std::string finish() {
    const std::uint64_t originalBits = bits_;
    const std::uint8_t one = 0x80;
    update(&one, 1);
    const std::uint8_t zero = 0;
    while (used_ != 56) update(&zero, 1);
    std::array<std::uint8_t, 8> length{};
    for (int i = 0; i != 8; ++i) length[7 - i] = std::uint8_t(originalBits >> (i * 8));
    update(length.data(), length.size());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state_)
      for (int i = 3; i >= 0; --i) out << std::setw(2) << ((word >> (i * 8)) & 0xff);
    return out.str();
  }
 private:
  static std::uint32_t ror(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void transform(const std::uint8_t* p) {
    static constexpr std::array<std::uint32_t, 64> k{{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2}};
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) w[i] = (std::uint32_t(p[4*i]) << 24) | (std::uint32_t(p[4*i+1]) << 16) | (std::uint32_t(p[4*i+2]) << 8) | p[4*i+3];
    for (int i = 16; i < 64; ++i) { const auto s0 = ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3); const auto s1 = ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (int i=0;i<64;++i) { const auto s1=ror(e,6)^ror(e,11)^ror(e,25); const auto ch=(e&f)^((~e)&g); const auto t1=h+s1+ch+k[i]+w[i]; const auto s0=ror(a,2)^ror(a,13)^ror(a,22); const auto maj=(a&b)^(a&c)^(b&c); h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+s0+maj; }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
  std::array<std::uint32_t,8> state_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<std::uint8_t,64> block_{}; size_t used_=0; std::uint64_t bits_=0;
};

std::string rawHash(const std::vector<float>& values) { Sha256 h; h.update(reinterpret_cast<const std::uint8_t*>(values.data()), values.size()*sizeof(float)); return h.finish(); }
std::string canonicalHash(const std::vector<float>& values) {
  Sha256 h;
  for (float value : values) {
    std::uint32_t bits;
    if (value == 0.0f) bits = 0;
    else if (std::isnan(value)) bits = 0x7fc00000u;
    else std::memcpy(&bits, &value, sizeof(bits));
    const std::array<std::uint8_t,4> le{{std::uint8_t(bits),std::uint8_t(bits>>8),std::uint8_t(bits>>16),std::uint8_t(bits>>24)}};
    h.update(le.data(), le.size());
  }
  return h.finish();
}

std::vector<float> cpuDembedding(const std::vector<float>& oneHot, const std::vector<float>& dx, uint32_t t, uint32_t v, uint32_t d) {
  std::vector<float> result(size_t(v)*d, 0.0f);
  for (uint32_t token=0;token<v;++token) for(uint32_t col=0;col<d;++col)
    for(uint32_t row=0;row<t;++row) result[size_t(token)*d+col] += oneHot[size_t(row)*v+token]*dx[size_t(row)*d+col];
  return result;
}
std::pair<std::vector<float>,std::vector<float>> batch(const tiny_lm::Config& c, uint32_t index) {
  static const std::array<std::array<uint32_t,4>,4> p{{{{0,1,2,3}},{{4,5,6,7}},{{8,9,8,9}},{{10,11,12,10}}}};
  std::vector<uint32_t> x(c.tokens), y(c.tokens);
  for(uint32_t i=0;i<c.tokens;++i){x[i]=p[index%p.size()][i%4];y[i]=p[index%p.size()][(i+1)%4];}
  return {tiny_lm::oneHot(x,c.vocabularySize),tiny_lm::oneHot(y,c.vocabularySize)};
}
struct Snapshot {
  const char* id;
  int completedStep;
  std::vector<float> oneHot, target, dx, expected;
  TinyTransformerParameters current, firstMoment, secondMoment;
  tiny_lm::StepResult cpu;
};
std::vector<Snapshot> snapshots() {
  tiny_lm::Config c; auto current=tiny_lm::initialParameters(c,1); auto m=current, v=current;
  for (auto member : {&TinyTransformerParameters::gamma1,&TinyTransformerParameters::beta1,&TinyTransformerParameters::wq,&TinyTransformerParameters::wk,&TinyTransformerParameters::wv,&TinyTransformerParameters::wo,&TinyTransformerParameters::gamma2,&TinyTransformerParameters::beta2,&TinyTransformerParameters::w1,&TinyTransformerParameters::w2,&TinyTransformerParameters::tokenEmbedding,&TinyTransformerParameters::outputProjection}) { std::fill((m.*member).begin(),(m.*member).end(),0); std::fill((v.*member).begin(),(v.*member).end(),0); }
  std::vector<Snapshot> result; const std::array<int,3> wanted{{2,10,100}};
  for(int completed=0;completed<=100;++completed) {
    const auto b=batch(c,uint32_t(completed%4));
    const auto r=tiny_lm::forwardBackward(c,b.first,b.second,current,.0003f);
    if(std::find(wanted.begin(),wanted.end(),completed)!=wanted.end()) {
      Snapshot s{completed==2?"E":completed==10?"D":"L",completed,b.first,b.second,
                 r.dEmbeddedInput,{},current,m,v,r};
      s.expected=cpuDembedding(s.oneHot,s.dx,c.tokens,c.vocabularySize,c.dimension);
      result.push_back(std::move(s));
    }
    if(completed==100) break;
    const int step=completed+1;
    const float c1=float(1.0/(1.0-std::pow(.9,double(step))));
    const float c2=float(1.0/(1.0-std::pow(.999,double(step))));
    auto u=tiny_lm::adamUpdate(current,r.gradients,m,v,.0003f,.9f,.999f,1e-8f,c1,c2);
    current=std::move(u.next);m=std::move(u.firstMoment);v=std::move(u.secondMoment);
  }
  return result;
}
std::uint32_t canonicalBits(float value) {
  if (value == 0.0f) return 0;
  if (std::isnan(value)) return 0x7fc00000u;
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
struct Aggregate {
  std::set<std::string> raw, canonical;
  std::vector<float> firstOutput;
  double cpuMaxAbs=0, cpuMeanAbs=0, cpuMaxRel=0;
  double repeatMaxAbs=0, repeatMeanAbs=0;
  size_t nonfinite=0, poison=0, poisonCollisions=0;
  size_t firstRepeatDiff=std::numeric_limits<size_t>::max(), runs=0, attempts=0;
  bool success=true;
};
void observe(Aggregate& a, const std::vector<float>& actual,
             const std::vector<float>& expected, float poison) {
  ++a.runs;
  a.raw.insert(rawHash(actual));
  a.canonical.insert(canonicalHash(actual));
  double cpuSum=0, repeatSum=0;
  for(size_t i=0;i<actual.size();++i) {
    if(expected[i] == poison) ++a.poisonCollisions;
    else if(actual[i] == poison) ++a.poison;
    if(!std::isfinite(actual[i])) ++a.nonfinite;
    const double cpuError=std::abs(double(actual[i])-expected[i]);
    cpuSum+=cpuError;
    a.cpuMaxAbs=std::max(a.cpuMaxAbs,cpuError);
    a.cpuMaxRel=std::max(a.cpuMaxRel,cpuError/std::max(1e-12,std::abs(double(expected[i]))));
    if(!a.firstOutput.empty()) {
      const double repeatError=std::abs(double(actual[i])-a.firstOutput[i]);
      repeatSum+=repeatError;
      a.repeatMaxAbs=std::max(a.repeatMaxAbs,repeatError);
      if(canonicalBits(actual[i]) != canonicalBits(a.firstOutput[i]) &&
         a.firstRepeatDiff == std::numeric_limits<size_t>::max()) {
        a.firstRepeatDiff=i;
      }
    }
  }
  a.cpuMeanAbs += cpuSum/std::max<size_t>(1,actual.size());
  if(!a.firstOutput.empty()) a.repeatMeanAbs += repeatSum/std::max<size_t>(1,actual.size());
  else a.firstOutput=actual;
  if(a.poison != 0 || a.nonfinite != 0) a.success=false;
}
void emit(std::ostringstream& out,const std::string& prefix,const Aggregate& a) {
  out<<prefix<<"_attempts="<<a.attempts<<'\n'
     <<prefix<<"_runs="<<a.runs<<'\n'
     <<prefix<<"_unique_raw_hashes="<<a.raw.size()<<'\n'
     <<prefix<<"_unique_canonical_hashes="<<a.canonical.size()<<'\n'
     <<prefix<<"_representative_raw_hash="<<(a.raw.empty()?"NONE":*a.raw.begin())<<'\n'
     <<prefix<<"_representative_canonical_hash="<<(a.canonical.empty()?"NONE":*a.canonical.begin())<<'\n'
     <<prefix<<"_cpu_max_abs_difference="<<a.cpuMaxAbs<<'\n'
     <<prefix<<"_cpu_mean_abs_difference="<<(a.runs?a.cpuMeanAbs/a.runs:0)<<'\n'
     <<prefix<<"_cpu_max_relative_difference="<<a.cpuMaxRel<<'\n'
     <<prefix<<"_repeat_max_abs_difference="<<a.repeatMaxAbs<<'\n'
     <<prefix<<"_repeat_mean_abs_difference="<<(a.runs>1?a.repeatMeanAbs/(a.runs-1):0)<<'\n'
     <<prefix<<"_nonfinite_count="<<a.nonfinite<<'\n'
     <<prefix<<"_app_read_poison_residual_elements="<<a.poison<<'\n'
     <<prefix<<"_poison_reference_collisions="<<a.poisonCollisions<<'\n'
     <<prefix<<"_physical_guard_integrity=UNSUPPORTED_RUNTIME_VECTOR_API\n"
     <<prefix<<"_qnn_execute_success="<<(a.success&&a.runs==a.attempts?"true":"false")<<'\n'
     <<prefix<<"_first_repeat_different_index="
     <<(a.firstRepeatDiff==std::numeric_limits<size_t>::max()?-1:static_cast<long long>(a.firstRepeatDiff))<<'\n';
}
bool execute(Runtime& rt,const Snapshot& s,float poison,Aggregate& a,std::string& error) {
  ++a.attempts;
  std::vector<float> oneHot(s.oneHot.size(),poison), dx(s.dx.size(),poison), out(s.expected.size(),poison);
  std::copy(s.oneHot.begin(),s.oneHot.end(),oneHot.begin());
  std::copy(s.dx.begin(),s.dx.end(),dx.begin());
  const bool inputClean=
      std::memcmp(oneHot.data(),s.oneHot.data(),oneHot.size()*sizeof(float))==0 &&
      std::memcmp(dx.data(),s.dx.data(),dx.size()*sizeof(float))==0;
  if(!inputClean){error="APP_WRITE poison survived canonical copy";a.success=false;return false;}
  if(!rt.executeMatMul(oneHot,dx,out,error)){a.success=false;return false;} observe(a,out,s.expected,poison); return true;
}
bool runSame(Runtime& rt,const Snapshot& s,int repeats,Aggregate& a,std::string& error) {
  if(!rt.prepareMatMul(32,8,16,true,error)){a.success=false;return false;}
  for(int i=0;i<repeats;++i)
    if(!execute(rt,s,(i&1)?-1.1415926f:1.1415926f,a,error)) return false;
  return true;
}

void forEachParameter(TinyTransformerParameters& p,
                      const std::function<void(std::vector<float>&)>& f) {
  for (auto member : {&TinyTransformerParameters::gamma1,&TinyTransformerParameters::beta1,
       &TinyTransformerParameters::wq,&TinyTransformerParameters::wk,
       &TinyTransformerParameters::wv,&TinyTransformerParameters::wo,
       &TinyTransformerParameters::gamma2,&TinyTransformerParameters::beta2,
       &TinyTransformerParameters::w1,&TinyTransformerParameters::w2,
       &TinyTransformerParameters::tokenEmbedding,
       &TinyTransformerParameters::outputProjection}) f(p.*member);
}
std::vector<float> flattenParameters(const TinyTransformerParameters& p) {
  std::vector<float> flat;
  auto copy=p;
  forEachParameter(copy,[&](std::vector<float>& v){flat.insert(flat.end(),v.begin(),v.end());});
  return flat;
}
void poisonOutputs(TinyTransformerTrainingOutputs& o,
                   const TinyTransformerParameters& shape, float poison) {
  o.output.assign(8*16,poison);
  o.dOutput.assign(8*16,poison);
  o.embeddedInput.assign(8*16,poison);
  o.logits.assign(8*32,poison);
  o.probabilities.assign(8*32,poison);
  o.dLogits.assign(8*32,poison);
  o.dEmbeddedInput.assign(8*16,poison);
  o.gradients=shape;
  o.next=shape;
  forEachParameter(o.gradients,[&](std::vector<float>& v){std::fill(v.begin(),v.end(),poison);});
  forEachParameter(o.next,[&](std::vector<float>& v){std::fill(v.begin(),v.end(),poison);});
}
struct TensorRepeat {
  std::set<std::string> raw, canonical;
  std::map<std::string,size_t> canonicalCounts;
  std::string firstCanonical;
  std::vector<float> first;
  double maxAbs=0;
  size_t firstDiff=std::numeric_limits<size_t>::max(), poisonResidual=0,
         nonfinite=0, runs=0,
         firstDifferentRun=std::numeric_limits<size_t>::max();
};
void observeRepeat(TensorRepeat& a,const std::vector<float>& v,float poison) {
  ++a.runs;
  const auto raw=rawHash(v), canonical=canonicalHash(v);
  a.raw.insert(raw);
  a.canonical.insert(canonical);
  ++a.canonicalCounts[canonical];
  a.poisonResidual+=std::count(v.begin(),v.end(),poison);
  a.nonfinite+=std::count_if(v.begin(),v.end(),
                            [](float value){return !std::isfinite(value);});
  if(a.first.empty()){a.first=v;a.firstCanonical=canonical;return;}
  if(canonical!=a.firstCanonical &&
     a.firstDifferentRun==std::numeric_limits<size_t>::max())
    a.firstDifferentRun=a.runs;
  for(size_t i=0;i<v.size();++i){
    a.maxAbs=std::max(a.maxAbs,std::abs(double(v[i])-a.first[i]));
    if(canonicalBits(v[i])!=canonicalBits(a.first[i]) &&
       a.firstDiff==std::numeric_limits<size_t>::max()) a.firstDiff=i;
  }
}
void emitRepeat(std::ostringstream& out,const std::string& prefix,const TensorRepeat& a) {
  std::ostringstream frequencies;
  bool first=true;
  for(const auto& item:a.canonicalCounts) {
    if(!first) frequencies<<',';
    first=false;
    frequencies<<item.first<<':'<<item.second;
  }
  out<<prefix<<"_unique_raw_hashes="<<a.raw.size()<<'\n'
     <<prefix<<"_unique_canonical_hashes="<<a.canonical.size()<<'\n'
     <<prefix<<"_representative_raw_hash="<<(a.raw.empty()?"NONE":*a.raw.begin())<<'\n'
     <<prefix<<"_representative_canonical_hash="<<(a.canonical.empty()?"NONE":*a.canonical.begin())<<'\n'
     <<prefix<<"_canonical_hash_frequencies="
     <<(frequencies.str().empty()?"NONE":frequencies.str())<<'\n'
     <<prefix<<"_repeat_max_abs_difference="<<a.maxAbs<<'\n'
     <<prefix<<"_first_different_run="
     <<(a.firstDifferentRun==std::numeric_limits<size_t>::max()?-1:
        static_cast<long long>(a.firstDifferentRun))<<'\n'
     <<prefix<<"_first_repeat_different_index="
     <<(a.firstDiff==std::numeric_limits<size_t>::max()?-1:static_cast<long long>(a.firstDiff))<<'\n'
     <<prefix<<"_app_read_poison_residual_elements="<<a.poisonResidual<<'\n'
     <<prefix<<"_nonfinite_elements="<<a.nonfinite<<'\n';
}
double maxAbsDifference(const std::vector<float>& a,const std::vector<float>& b) {
  if(a.size()!=b.size()) return std::numeric_limits<double>::infinity();
  double result=0;
  for(size_t i=0;i<a.size();++i)
    result=std::max(result,std::abs(double(a[i])-b[i]));
  return result;
}
bool runFullGraphFixed(const Snapshot& s,int repeats,std::ostringstream& out,
                       std::string& firstChanging,std::string& error) {
  Runtime rt;
  if(!rt.initialize(QnnBackendKind::HTP,error) ||
     !rt.prepareTinyTransformerTraining(8,16,32,1e-5f,true,error,32)) return false;
  std::map<std::string,TensorRepeat> tensors;
  for(int i=0;i<repeats;++i) {
    const float poison=(i&1)?-1.1415926f:1.1415926f;
    auto input=s.oneHot;
    auto target=s.target;
    auto current=s.current;
    const auto inputBefore=rawHash(input), targetBefore=rawHash(target);
    const auto parameterBefore=rawHash(flattenParameters(current));
    TinyTransformerTrainingOutputs outputs;
    poisonOutputs(outputs,current,poison);
    if(!rt.executeTinyTransformerTraining(input,target,current,0.0003f,outputs,error)) return false;
    if(rawHash(input)!=inputBefore || rawHash(target)!=targetBefore ||
       rawHash(flattenParameters(current))!=parameterBefore) {
      error="APP_WRITE input/state mutated during full graph execute";
      return false;
    }
    observeRepeat(tensors["logits"],outputs.logits,poison);
    observeRepeat(tensors["softmax_probability"],outputs.probabilities,poison);
    observeRepeat(tensors["dlogits"],outputs.dLogits,poison);
    observeRepeat(tensors["output_projection_gradient"],outputs.gradients.outputProjection,poison);
    observeRepeat(tensors["transformer_output"],outputs.output,poison);
    observeRepeat(tensors["embedding_input_gradient"],outputs.dEmbeddedInput,poison);
    observeRepeat(tensors["token_embedding_gradient"],outputs.gradients.tokenEmbedding,poison);
    observeRepeat(tensors["next_token_embedding"],outputs.next.tokenEmbedding,poison);
  }
  static const std::array<const char*,8> order{{"logits","softmax_probability","dlogits",
    "output_projection_gradient","transformer_output","embedding_input_gradient",
    "token_embedding_gradient","next_token_embedding"}};
  bool poisonOk=true;
  for(const char* name:order) {
    emitRepeat(out,std::string("full_graph_")+s.id+"_"+name,tensors[name]);
    if(tensors[name].poisonResidual!=0) poisonOk=false;
    if(firstChanging.empty()&&tensors[name].canonical.size()>1) firstChanging=name;
  }
  out<<"full_graph_"<<s.id<<"_logits_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["logits"].first,s.cpu.logits)<<'\n'
     <<"full_graph_"<<s.id<<"_softmax_probability_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["softmax_probability"].first,s.cpu.probabilities)<<'\n'
     <<"full_graph_"<<s.id<<"_dlogits_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["dlogits"].first,s.cpu.dLogits)<<'\n'
     <<"full_graph_"<<s.id<<"_output_projection_gradient_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["output_projection_gradient"].first,s.cpu.gradients.outputProjection)<<'\n'
     <<"full_graph_"<<s.id<<"_transformer_output_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["transformer_output"].first,s.cpu.transformerOutput)<<'\n'
     <<"full_graph_"<<s.id<<"_embedding_input_gradient_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["embedding_input_gradient"].first,s.cpu.dEmbeddedInput)<<'\n'
     <<"full_graph_"<<s.id<<"_token_embedding_gradient_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["token_embedding_gradient"].first,s.cpu.gradients.tokenEmbedding)<<'\n'
     <<"full_graph_"<<s.id<<"_next_token_embedding_cpu_max_abs_difference="
     <<maxAbsDifference(tensors["next_token_embedding"].first,s.cpu.next.tokenEmbedding)<<'\n';
  if(!poisonOk){error="APP_READ poison remained in full graph output";return false;}
  return true;
}
const char* variantName(TinyTransformerTrainingVariant variant) {
  switch (variant) {
    case TinyTransformerTrainingVariant::FULL: return "full";
    case TinyTransformerTrainingVariant::STOP_AFTER_DINPUT:
      return "stop_after_dinput";
    case TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING:
      return "stop_after_dembedding";
  }
  return "unknown";
}
bool runGraphVariantFixed(const Snapshot& s, TinyTransformerTrainingVariant variant,
                          int repeats, std::ostringstream& out,
                          std::string& error) {
  Runtime rt;
  const std::string prefix=std::string("variant_")+variantName(variant);
  if(!rt.initialize(QnnBackendKind::HTP,error) ||
     !rt.prepareTinyTransformerTraining(8,16,32,1e-5f,true,error,32,variant))
    return false;
  std::map<std::string,TensorRepeat> tensors;
  int attempts=0, successes=0;
  bool appWriteHashesUnchanged=true;
  for(int i=0;i<repeats;++i) {
    ++attempts;
    const float poison=(i&1)?-1.1415926f:1.1415926f;
    auto input=s.oneHot;
    auto target=s.target;
    auto current=s.current;
    const auto inputBefore=rawHash(input), targetBefore=rawHash(target);
    const auto parameterBefore=rawHash(flattenParameters(current));
    TinyTransformerTrainingOutputs outputs;
    poisonOutputs(outputs,current,poison);
    if(!rt.executeTinyTransformerTraining(input,target,current,0.0003f,outputs,error))
      return false;
    ++successes;
    appWriteHashesUnchanged &=
        rawHash(input)==inputBefore && rawHash(target)==targetBefore &&
        rawHash(flattenParameters(current))==parameterBefore;
    if(!appWriteHashesUnchanged) {
      error="APP_WRITE input/state mutated during graph variant execute";
      return false;
    }
    auto audit=[&](const char* name,const std::vector<float>& values) {
      observeRepeat(tensors[name],values,poison);
    };
    audit("logits",outputs.logits);
    audit("transformer_output",outputs.output);
    audit("transformer_output_gradient",outputs.dOutput);
    audit("gradient_gamma1",outputs.gradients.gamma1);
    audit("gradient_beta1",outputs.gradients.beta1);
    audit("gradient_wq",outputs.gradients.wq);
    audit("gradient_wk",outputs.gradients.wk);
    audit("gradient_wv",outputs.gradients.wv);
    audit("gradient_wo",outputs.gradients.wo);
    audit("gradient_gamma2",outputs.gradients.gamma2);
    audit("gradient_beta2",outputs.gradients.beta2);
    audit("gradient_w1",outputs.gradients.w1);
    audit("gradient_w2",outputs.gradients.w2);
    audit("embedded_input",outputs.embeddedInput);
    audit("softmax_probability",outputs.probabilities);
    audit("dlogits",outputs.dLogits);
    audit("embedding_input_gradient",outputs.dEmbeddedInput);
    audit("output_projection_gradient",outputs.gradients.outputProjection);
    if(variant!=TinyTransformerTrainingVariant::STOP_AFTER_DINPUT)
      audit("token_embedding_gradient",outputs.gradients.tokenEmbedding);
    if(variant==TinyTransformerTrainingVariant::FULL) {
      audit("next_gamma1",outputs.next.gamma1);
      audit("next_beta1",outputs.next.beta1);
      audit("next_wq",outputs.next.wq);
      audit("next_wk",outputs.next.wk);
      audit("next_wv",outputs.next.wv);
      audit("next_wo",outputs.next.wo);
      audit("next_gamma2",outputs.next.gamma2);
      audit("next_beta2",outputs.next.beta2);
      audit("next_w1",outputs.next.w1);
      audit("next_w2",outputs.next.w2);
      audit("next_token_embedding",outputs.next.tokenEmbedding);
      audit("next_output_projection",outputs.next.outputProjection);
    }
  }
  emitRepeat(out,prefix+"_transformer_output",tensors["transformer_output"]);
  emitRepeat(out,prefix+"_embedding_input_gradient",
             tensors["embedding_input_gradient"]);
  if(variant!=TinyTransformerTrainingVariant::STOP_AFTER_DINPUT)
    emitRepeat(out,prefix+"_token_embedding_gradient",
               tensors["token_embedding_gradient"]);
  else
    out<<prefix<<"_token_embedding_gradient=NOT_PRESENT_BY_DESIGN\n";
  if(variant==TinyTransformerTrainingVariant::FULL)
    emitRepeat(out,prefix+"_next_token_embedding",
               tensors["next_token_embedding"]);
  else
    out<<prefix<<"_next_token_embedding=NOT_PRESENT_BY_DESIGN\n";
  bool poisonOk=true, finite=true;
  for(const auto& item:tensors) {
    poisonOk&=item.second.poisonResidual==0;
    finite&=item.second.nonfinite==0;
  }
  auto executeUs=rt.metrics().executeUs;
  std::sort(executeUs.begin(),executeUs.end());
  double executeMeanUs=0;
  for(double value:executeUs) executeMeanUs+=value;
  if(!executeUs.empty()) executeMeanUs/=executeUs.size();
  const auto percentile=[&](double q) {
    if(executeUs.empty()) return 0.0;
    return executeUs[std::min(executeUs.size()-1,
                              size_t(q*double(executeUs.size()-1)))];
  };
  out<<prefix<<"_graph_boundary="<<(variant==TinyTransformerTrainingVariant::FULL?
       "lm_output_projection_next":variant==TinyTransformerTrainingVariant::STOP_AFTER_DINPUT?
       "lm_dinput":"lm_dembedding")<<'\n'
     <<prefix<<"_repeats="<<repeats<<'\n'
     <<prefix<<"_qnn_execute_attempts="<<attempts<<'\n'
     <<prefix<<"_qnn_execute_successes="<<successes<<'\n'
     <<prefix<<"_qnn_execute_return_code=0\n"
     <<prefix<<"_execute_us_min="<<percentile(0)<<'\n'
     <<prefix<<"_execute_us_median="<<percentile(.5)<<'\n'
     <<prefix<<"_execute_us_p95="<<percentile(.95)<<'\n'
     <<prefix<<"_execute_us_max="<<percentile(1)<<'\n'
     <<prefix<<"_execute_us_mean="<<executeMeanUs<<'\n'
     <<prefix<<"_app_write_hashes_unchanged="
     <<(appWriteHashesUnchanged?"true":"false")<<'\n'
     <<prefix<<"_app_read_tensors_audited="<<tensors.size()<<'\n'
     <<prefix<<"_app_read_poison_residual_elements="
     <<([&](){size_t n=0;for(const auto& item:tensors)n+=item.second.poisonResidual;
              return n;}())<<'\n'
     <<prefix<<"_nonfinite_elements="
     <<([&](){size_t n=0;for(const auto& item:tensors)n+=item.second.nonfinite;
              return n;}())<<'\n'
     <<prefix<<"_all_outputs_finite="<<(finite?"true":"false")<<'\n'
     <<prefix<<"_numerical_variability_observed="
     <<([&](){for(const auto& item:tensors)
                if(item.second.canonical.size()>1)return true;
              return false;}()?"true":"false")<<'\n';
  if(!poisonOk){error="APP_READ poison remained in graph variant output";return false;}
  if(!finite){error="nonfinite graph variant output";return false;}
  return true;
}
bool runStandalonePrelude(const Snapshot& s,std::ostringstream& out,
                          std::string& error) {
  bool ok=true;
  Aggregate sameGraph;
  {
    Runtime rt;
    if(!rt.initialize(QnnBackendKind::HTP,error) ||
       !runSame(rt,s,100,sameGraph,error)) ok=false;
  }
  emit(out,"prelude_scope_a_same_graph",sameGraph);
  Aggregate recreateGraph;
  {
    Runtime rt;
    if(!rt.initialize(QnnBackendKind::HTP,error)) ok=false;
    else for(int i=0;i<30;++i)
      if(!runSame(rt,s,1,recreateGraph,error)){ok=false;break;}
  }
  emit(out,"prelude_scope_b_recreate_graph_context_reuse",recreateGraph);
  Aggregate recreateContext;
  {
    Runtime rt;
    if(!rt.initialize(QnnBackendKind::HTP,error)) ok=false;
    else for(int i=0;i<20;++i) {
      if(!runSame(rt,s,1,recreateContext,error)){ok=false;break;}
      if(i!=19&&!rt.recreateContext(error)){
        recreateContext.success=false;ok=false;break;
      }
    }
  }
  emit(out,"prelude_scope_c_recreate_context_backend_reuse",recreateContext);
  Aggregate recreateRuntime;
  for(int i=0;i<10;++i) {
    Runtime rt;
    if(!rt.initialize(QnnBackendKind::HTP,error) ||
       !runSame(rt,s,1,recreateRuntime,error)){ok=false;break;}
  }
  emit(out,"prelude_scope_d_backend_runtime_recreate",recreateRuntime);
  for(const Aggregate* aggregate:{&sameGraph,&recreateGraph,&recreateContext,
                                  &recreateRuntime}) {
    ok&=aggregate->success && aggregate->runs==aggregate->attempts &&
        aggregate->canonical.size()==1 && aggregate->poison==0 &&
        aggregate->nonfinite==0;
  }
  out<<"prelude_total_qnn_execute_attempts="
     <<sameGraph.attempts+recreateGraph.attempts+recreateContext.attempts+
       recreateRuntime.attempts<<'\n'
     <<"prelude_qnn_execute_return_code=0\n"
     <<"prelude_all_outputs_deterministic="<<(ok?"true":"false")<<'\n';
  if(!ok&&error.empty()) error="standalone prelude invariant failed";
  return ok;
}
bool runZeroGradientOptimizer(const Snapshot& s,std::ostringstream& out,
                              std::string& error) {
  auto current=flattenParameters(s.current);
  std::vector<float> zero(current.size(),0.0f);
  const auto currentHash=rawHash(current);
  const auto zeroHash=rawHash(zero);
  Runtime rt;
  AdamOptimizerOutputs outputs;
  if(!rt.initialize(QnnBackendKind::HTP,error) ||
     !rt.prepareAdamOptimizer(uint32_t(current.size()),error) ||
     !rt.executeAdamOptimizer(current,zero,zero,zero,.0003f,1.0f,10.0f,
                              1000.0f,outputs,error)) return false;
  const bool inputsUntouched=rawHash(current)==currentHash&&rawHash(zero)==zeroHash;
  const bool firstZero=canonicalHash(outputs.firstMomentNext)==canonicalHash(zero);
  const bool secondZero=canonicalHash(outputs.secondMomentNext)==canonicalHash(zero);
  const bool weightSame=canonicalHash(outputs.weightNext)==canonicalHash(current);
  const double weightMaxAbs=maxAbsDifference(outputs.weightNext,current);
  size_t firstWeightDiff=std::numeric_limits<size_t>::max();
  for(size_t i=0;i<current.size();++i)
    if(canonicalBits(outputs.weightNext[i])!=canonicalBits(current[i])){
      firstWeightDiff=i;break;
    }
  out<<"zero_gradient_optimizer_inputs_untouched="<<(inputsUntouched?"true":"false")<<'\n'
     <<"zero_gradient_optimizer_m_next_zero="<<(firstZero?"true":"false")<<'\n'
     <<"zero_gradient_optimizer_v_next_zero="<<(secondZero?"true":"false")<<'\n'
     <<"zero_gradient_optimizer_weight_next_unchanged="<<(weightSame?"true":"false")<<'\n'
     <<"zero_gradient_optimizer_weight_next_max_abs_difference="<<weightMaxAbs<<'\n'
     <<"zero_gradient_optimizer_weight_next_first_different_index="
     <<(firstWeightDiff==std::numeric_limits<size_t>::max()?-1:
        static_cast<long long>(firstWeightDiff))<<'\n'
     <<"zero_gradient_optimizer_qnn_execute_success=true\n";
  if(!inputsUntouched||!firstZero||!secondZero) {
    error="zero-gradient optimizer state invariant failed";
    return false;
  }
  return true;
}
bool runAdamFixed(const Snapshot& s,int repeats,std::ostringstream& out,
                  std::string& firstChanging,std::string& error) {
  auto current=flattenParameters(s.current);
  auto gradient=flattenParameters(s.cpu.gradients);
  auto first=flattenParameters(s.firstMoment);
  auto second=flattenParameters(s.secondMoment);
  double normSquared=0;
  for(float value:gradient) normSquared+=double(value)*value;
  const float clipScale=std::min(1.0f,float(10.0/(std::sqrt(normSquared)+1e-6)));
  const int step=s.completedStep+1;
  const float firstCorrection=float(1.0/(1.0-std::pow(.9,double(step))));
  const float secondCorrection=float(1.0/(1.0-std::pow(.999,double(step))));
  const auto currentHash=rawHash(current), gradientHash=rawHash(gradient);
  const auto firstHash=rawHash(first), secondHash=rawHash(second);
  Runtime rt;
  if(!rt.initialize(QnnBackendKind::HTP,error) ||
     !rt.prepareAdamOptimizer(uint32_t(current.size()),error)) return false;
  std::map<std::string,TensorRepeat> tensors;
  for(int i=0;i<repeats;++i) {
    const float poison=(i&1)?-1.1415926f:1.1415926f;
    AdamOptimizerOutputs outputs;
    outputs.firstMomentNext.assign(current.size(),poison);
    outputs.secondMomentNext.assign(current.size(),poison);
    outputs.firstMomentHat.assign(current.size(),poison);
    outputs.secondMomentHat.assign(current.size(),poison);
    outputs.secondRoot.assign(current.size(),poison);
    outputs.denominator.assign(current.size(),poison);
    outputs.normalizedUpdate.assign(current.size(),poison);
    outputs.scaledUpdate.assign(current.size(),poison);
    outputs.weightNext.assign(current.size(),poison);
    if(!rt.executeAdamOptimizer(current,gradient,first,second,.0003f,clipScale,
                                firstCorrection,secondCorrection,outputs,error))
      return false;
    if(rawHash(current)!=currentHash||rawHash(gradient)!=gradientHash||
       rawHash(first)!=firstHash||rawHash(second)!=secondHash) {
      error="APP_WRITE optimizer input/state mutated";
      return false;
    }
    observeRepeat(tensors["m_next"],outputs.firstMomentNext,poison);
    observeRepeat(tensors["v_next"],outputs.secondMomentNext,poison);
    observeRepeat(tensors["m_hat"],outputs.firstMomentHat,poison);
    observeRepeat(tensors["v_hat"],outputs.secondMomentHat,poison);
    observeRepeat(tensors["adam_denominator"],outputs.denominator,poison);
    observeRepeat(tensors["normalized_update"],outputs.normalizedUpdate,poison);
    observeRepeat(tensors["update"],outputs.scaledUpdate,poison);
    observeRepeat(tensors["next_weight"],outputs.weightNext,poison);
  }
  static const std::array<const char*,8> order{{"m_next","v_next","m_hat","v_hat",
    "adam_denominator","normalized_update","update","next_weight"}};
  bool poisonOk=true;
  for(const char* name:order) {
    emitRepeat(out,std::string("fixed_adam_")+s.id+"_"+name,tensors[name]);
    if(tensors[name].poisonResidual!=0) poisonOk=false;
    if(firstChanging.empty()&&tensors[name].canonical.size()>1) firstChanging=name;
  }
  out<<"fixed_adam_"<<s.id<<"_step_index="<<step<<'\n'
     <<"fixed_adam_"<<s.id<<"_learning_rate_bits=0x"<<std::hex
     <<canonicalBits(.0003f)<<std::dec<<'\n'
     <<"fixed_adam_"<<s.id<<"_clip_scalar_bits=0x"<<std::hex
     <<canonicalBits(clipScale)<<std::dec<<'\n'
     <<"fixed_adam_"<<s.id<<"_bias_correction_1_bits=0x"<<std::hex
     <<canonicalBits(firstCorrection)<<std::dec<<'\n'
     <<"fixed_adam_"<<s.id<<"_bias_correction_2_bits=0x"<<std::hex
     <<canonicalBits(secondCorrection)<<std::dec<<'\n'
     <<"fixed_adam_"<<s.id<<"_current_parameter_canonical_hash="
     <<canonicalHash(current)<<'\n'
     <<"fixed_adam_"<<s.id<<"_m_canonical_hash="<<canonicalHash(first)<<'\n'
     <<"fixed_adam_"<<s.id<<"_v_canonical_hash="<<canonicalHash(second)<<'\n'
     <<"fixed_adam_"<<s.id<<"_gradient_canonical_hash="<<canonicalHash(gradient)<<'\n';
  if(!poisonOk){error="APP_READ poison remained in Adam output";return false;}
  return true;
}
}

std::string rawFloatSha256(const std::vector<float>& values) {
  return rawHash(values);
}
std::string canonicalFloatSha256(const std::vector<float>& values) {
  return canonicalHash(values);
}

std::string runTinyLmGraphBisection(bool standalonePrelude) {
  std::ostringstream out;
  out<<std::setprecision(9)
     <<"QNN_TINY_LM_GRAPH_BISECTION\n"
     <<"test=fixed_state_graph_prefix_bisection\n"
     <<"shape=B1_T8_V32_D16\n"
     <<"snapshot=E\n"
     <<"snapshot_source=CPU_REFERENCE_ADAM_TRAJECTORY\n"
     <<"manifest_schema_version=2\n"
     <<"manifest_seed=1\n"
     <<"manifest_batch_schedule=pattern_round_robin_4\n"
     <<"repeats_per_variant=100\n"
     <<"fresh_process_control=EXTERNAL_HEADLESS_INSTRUMENTATION\n"
     <<"prelude="<<(standalonePrelude?
       "E_STANDALONE_SCOPE_A100_B30_C20_D10":"NONE_COLD_FULL_FIRST")<<'\n'
     <<"variant_execution_order=full,stop_after_dinput,stop_after_dembedding\n"
     <<"tensor_creation_order=PRESERVED_FULL_ENUM_ORDER\n"
     <<"node_prefix_order=PRESERVED\n"
     <<"app_write_poison_patterns=finite_plus_minus_1.1415926\n"
     <<"app_read_poison_patterns=finite_plus_minus_1.1415926\n"
     <<"physical_guard=UNSUPPORTED_RUNTIME_VECTOR_API\n";
  const auto fixedSnapshots=snapshots();
  if(fixedSnapshots.empty()) return out.str()+"status=FAILED\nerror=no snapshots\n";
  const auto& s=fixedSnapshots.front();
  out<<"snapshot_E_one_hot_raw_hash="<<rawHash(s.oneHot)<<'\n'
     <<"snapshot_E_one_hot_canonical_hash="<<canonicalHash(s.oneHot)<<'\n'
     <<"snapshot_E_target_raw_hash="<<rawHash(s.target)<<'\n'
     <<"snapshot_E_target_canonical_hash="<<canonicalHash(s.target)<<'\n'
     <<"snapshot_E_current_parameter_raw_hash="
     <<rawHash(flattenParameters(s.current))<<'\n'
     <<"snapshot_E_current_parameter_canonical_hash="
     <<canonicalHash(flattenParameters(s.current))<<'\n';
  std::string error;
  bool ok=true;
  if(standalonePrelude&&!runStandalonePrelude(s,out,error)) ok=false;
  for(auto variant:{TinyTransformerTrainingVariant::FULL,
                    TinyTransformerTrainingVariant::STOP_AFTER_DINPUT,
                    TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING}) {
    if(ok&&!runGraphVariantFixed(s,variant,100,out,error)){ok=false;break;}
  }
  out<<"comparison_same_input_state=true\n"
     <<"qnn_execute_all_success="<<(ok?"true":"false")<<'\n'
     <<"status="<<(ok?"SUCCESS":"FAILED")<<'\n'
     <<"error="<<(ok?"none":error)<<'\n';
  return out.str();
}

std::string runTinyLmGraphIsolated(int variantCode) {
  TinyTransformerTrainingVariant variant;
  switch(variantCode) {
    case 0: variant=TinyTransformerTrainingVariant::FULL;break;
    case 1: variant=TinyTransformerTrainingVariant::STOP_AFTER_DINPUT;break;
    case 2: variant=TinyTransformerTrainingVariant::STOP_AFTER_DEMBEDDING;break;
    default:
      return "QNN_TINY_LM_GRAPH_ISOLATED\nstatus=FAILED\n"
             "error=unknown graph variant\n";
  }
  std::ostringstream out;
  out<<std::setprecision(9)
     <<"QNN_TINY_LM_GRAPH_ISOLATED\n"
     <<"test=fixed_state_isolated_graph_variant\n"
     <<"shape=B1_T8_V32_D16\n"
     <<"snapshot=E\n"
     <<"snapshot_source=CPU_REFERENCE_ADAM_TRAJECTORY\n"
     <<"manifest_schema_version=2\n"
     <<"manifest_seed=1\n"
     <<"manifest_batch_schedule=pattern_round_robin_4\n"
     <<"isolated_variant="<<variantName(variant)<<'\n'
     <<"repeats=100\n"
     <<"qnn_graphs_before_variant=0\n"
     <<"fresh_process_control=EXTERNAL_HEADLESS_INSTRUMENTATION\n"
     <<"tensor_creation_order=PRESERVED_FULL_ENUM_ORDER\n"
     <<"node_prefix_order=PRESERVED\n"
     <<"app_write_poison_patterns=finite_plus_minus_1.1415926\n"
     <<"app_read_poison_patterns=finite_plus_minus_1.1415926\n"
     <<"physical_guard=UNSUPPORTED_RUNTIME_VECTOR_API\n";
  const auto fixedSnapshots=snapshots();
  if(fixedSnapshots.empty()) return out.str()+"status=FAILED\nerror=no snapshots\n";
  const auto& s=fixedSnapshots.front();
  out<<"snapshot_E_one_hot_raw_hash="<<rawHash(s.oneHot)<<'\n'
     <<"snapshot_E_one_hot_canonical_hash="<<canonicalHash(s.oneHot)<<'\n'
     <<"snapshot_E_target_raw_hash="<<rawHash(s.target)<<'\n'
     <<"snapshot_E_target_canonical_hash="<<canonicalHash(s.target)<<'\n'
     <<"snapshot_E_current_parameter_raw_hash="
     <<rawHash(flattenParameters(s.current))<<'\n'
     <<"snapshot_E_current_parameter_canonical_hash="
     <<canonicalHash(flattenParameters(s.current))<<'\n';
  std::string error;
  const bool ok=runGraphVariantFixed(s,variant,100,out,error);
  out<<"comparison_same_input_state=true\n"
     <<"qnn_execute_all_success="<<(ok?"true":"false")<<'\n'
     <<"status="<<(ok?"SUCCESS":"FAILED")<<'\n'
     <<"error="<<(ok?"none":error)<<'\n';
  return out.str();
}

std::string runTinyLmDembeddingReproducibility() {
  std::ostringstream out; out<<std::setprecision(9)
    <<"QNN_TINY_LM_REPRODUCIBILITY\n"
    <<"test=lm_dembedding_micrograph\n"
    <<"shape=B1_T8_V32_D16\n"
    <<"operator=OneHotTransposeMatMulDX\n"
    <<"standalone_qnn_node=matmul\n"
    <<"full_graph_candidate_node=lm_dembedding\n"
    <<"first_compared_tensor=lm_dembedding_output\n"
    <<"snapshot_source=CPU_REFERENCE_ADAM_TRAJECTORY\n"
    <<"manifest_schema_version=1\n"
    <<"manifest_seed=1\n"
    <<"manifest_batch_schedule=pattern_round_robin_4\n"
    <<"manifest_graph_configuration=B1_T8_V32_D16_OneHotTransposeMatMulDX_FP32\n"
    <<"raw_hash=SHA256_all_float_bytes\n"
    <<"canonical_float_hash=SHA256_IEEE754_binary32_little_endian_negzero_to_poszero_nan_to_7fc00000\n"
    <<"app_write_poison_patterns=finite_plus_minus_1.1415926\n"
    <<"app_read_poison_patterns=finite_plus_minus_1.1415926\n"
    <<"physical_guard=UNSUPPORTED_RUNTIME_VECTOR_API\n";
  bool all=true, variability=false; std::string error, fullFirstChanging;
  std::string optimizerFirstChanging;
  const auto fixedSnapshots=snapshots();
  for(const auto& s:fixedSnapshots) {
    const std::string base=std::string("snapshot_")+s.id;
    out<<base<<"_id="<<s.id<<'\n'
       <<base<<"_one_hot_shape=8x32\n"
       <<base<<"_dx_shape=8x16\n"
       <<base<<"_output_shape=32x16\n"
       <<base<<"_one_hot_raw_hash="<<rawHash(s.oneHot)<<'\n'
       <<base<<"_one_hot_canonical_hash="<<canonicalHash(s.oneHot)<<'\n'
       <<base<<"_dx_raw_hash="<<rawHash(s.dx)<<'\n'
       <<base<<"_dx_canonical_hash="<<canonicalHash(s.dx)<<'\n'
       <<base<<"_target_canonical_hash="<<canonicalHash(s.target)<<'\n'
       <<base<<"_current_parameter_canonical_hash="<<canonicalHash(flattenParameters(s.current))<<'\n'
       <<base<<"_cpu_canonical_hash="<<canonicalHash(s.expected)<<'\n';
    Aggregate a; {
      Runtime rt;
      if(!rt.initialize(QnnBackendKind::HTP,error)) { a.success=false; all=false; }
      else if(!runSame(rt,s,100,a,error)) all=false;
    }
    emit(out,base+"_scope_a_same_graph",a);
    variability|=a.canonical.size()>1;
    Aggregate b;
    {
      Runtime rt;
      if(!rt.initialize(QnnBackendKind::HTP,error)) { b.success=false; all=false; }
      else {
        for(int i=0;i<30;++i)
          if(!runSame(rt,s,1,b,error)){all=false;break;}
      }
    }
    emit(out,base+"_scope_b_recreate_graph_context_reuse",b);
    variability|=b.canonical.size()>1;
    Aggregate c;
    {
      Runtime rt;
      if(!rt.initialize(QnnBackendKind::HTP,error)) { c.success=false; all=false; }
      else {
        for(int i=0;i<20;++i) {
          if(!runSame(rt,s,1,c,error)){all=false;break;}
          if(i!=19&&!rt.recreateContext(error)){c.success=false;all=false;break;}
        }
      }
    }
    emit(out,base+"_scope_c_recreate_context_backend_reuse",c);
    variability|=c.canonical.size()>1;
    Aggregate d;
    for(int i=0;i<10;++i) {
      Runtime rt;
      if(!rt.initialize(QnnBackendKind::HTP,error)) { d.success=false; all=false; break; }
      if(!runSame(rt,s,1,d,error)) { all=false; break; }
    }
    emit(out,base+"_scope_d_backend_runtime_recreate",d);
    variability|=d.canonical.size()>1;
    size_t unused=0, nonzero=0, htpNonzero=0, htpSignedZero=0;
    double htpUnusedMaxAbs=0;
    for(uint32_t row=0;row<32;++row){
      bool used=false;
      for(uint32_t t=0;t<8;++t)used|=s.oneHot[size_t(t)*32+row]>0.5f;
      if(!used){
        ++unused;
        for(uint32_t col=0;col<16;++col) {
          const size_t index=size_t(row)*16+col;
          if(s.expected[index]!=0) ++nonzero;
          if(!a.firstOutput.empty()) {
            const float value=a.firstOutput[index];
            if(value!=0) ++htpNonzero;
            if(value==0&&std::signbit(value)) ++htpSignedZero;
            htpUnusedMaxAbs=std::max(htpUnusedMaxAbs,std::abs(double(value)));
          }
        }
      }
    }
    out<<base<<"_unused_embedding_rows="<<unused<<'\n'
       <<base<<"_cpu_unused_embedding_nonzero_elements="<<nonzero<<'\n'
       <<base<<"_htp_unused_embedding_nonzero_elements="<<htpNonzero<<'\n'
       <<base<<"_htp_unused_embedding_signed_zero_elements="<<htpSignedZero<<'\n'
       <<base<<"_htp_unused_embedding_max_abs="<<htpUnusedMaxAbs<<'\n';
    if(!runFullGraphFixed(s,100,out,fullFirstChanging,error)) all=false;
    if(!runAdamFixed(s,100,out,optimizerFirstChanging,error)) all=false;
  }
  if(!fixedSnapshots.empty()&&!runZeroGradientOptimizer(fixedSnapshots.front(),out,error))
    all=false;
  const auto fullChangingNode = [&]() -> const char* {
    if (fullFirstChanging == "embedding_input_gradient") return "lm_dinput";
    if (fullFirstChanging == "token_embedding_gradient") return "lm_dembedding";
    if (fullFirstChanging == "next_token_embedding") return "lm_embedding_next";
    if (fullFirstChanging == "output_projection_gradient")
      return "lm_doutput_projection";
    return fullFirstChanging.empty() ? "NO_FIXED_STATE_VARIABILITY" :
                                      "FULL_GRAPH_UPSTREAM";
  };
  out<<"scope_b_status=EXECUTED\n"
     <<"scope_b_graph_lifetime=BOUNDED_30_GRAPHS_RELEASED_WITH_CONTEXT\n"
     <<"scope_c_status=EXECUTED\n"
     <<"scope_e_status=EXTERNAL_INSTRUMENTATION_CONTROL\n"
     <<"standalone_fixed_state_variability="<<(variability?"true":"false")<<'\n'
     <<"full_graph_fixed_state_variability="<<(!fullFirstChanging.empty()?"true":"false")<<'\n'
     <<"optimizer_fixed_state_variability="<<(!optimizerFirstChanging.empty()?"true":"false")<<'\n'
     <<"optimizer_first_changing_tensor="
     <<(optimizerFirstChanging.empty()?"NO_FIXED_STATE_VARIABILITY":optimizerFirstChanging)<<'\n'
     <<"first_changing_tensor="<<(variability?"lm_dembedding_output":
         !fullFirstChanging.empty()?fullFirstChanging:"NO_FIXED_STATE_VARIABILITY")<<'\n'
     <<"first_changing_node="<<(variability?"matmul":
         fullChangingNode())<<'\n'
     <<"classification="<<(variability?"BACKEND_EXECUTION_VARIABILITY_MICROGRAPH":
         !fullFirstChanging.empty()?"BACKEND_EXECUTION_VARIABILITY_FULL_GRAPH":
         "NO_FIXED_STATE_VARIABILITY")<<'\n'
     <<"status="<<(all?"SUCCESS":"FAILED")<<"\nerror="<<(all?"none":error)<<'\n';
  return out.str();
}
}  // namespace phonelm::qnn
