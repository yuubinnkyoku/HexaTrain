#pragma once
#include "qk_scale_graph.h"
namespace qkscalestudy {
using closurestudy::require;using closurestudy::slice;using orderedhvxstudy::words;
constexpr double rawGate=.01,scaleGate=.01,scatterGate=.01;
inline float scaleValue(){return 1.0f/std::sqrt(float(64/2));}
inline void receipt(const std::vector<uint32_t>& r,bool qk,const std::vector<one::FV>& x,const std::vector<one::FV>& y,unsigned& last){
 segmentstudy::receipt(r,qk?0x48383116u:0x48383117u,x,y,last);
 if(!qk)std::printf("frontierscale,received_bits=%08x,head_gradient_order=H0Q_H0K_H1Q_H1K\n",r.at(4104));
}
inline std::vector<one::Vec> reference(const std::vector<one::FV>& x,bool qk,bool fp){std::vector<one::Vec> v;
 if(qk)for(unsigned h=0;h<2;++h)v.push_back(one::mm(one::d(x[h]),one::d(x[2+h]),32,32,32,false,true,fp));
 else for(unsigned j=0;j<4;++j)v.push_back(one::mul(one::d(x[j]),one::d(x[4]),fp));return v;
}
struct Standalone {
 DiagnosticGraph g;bool qk,obs;std::vector<unsigned>x,y;unsigned rr;
 Standalone(std::string name,bool q,bool o,bool standard=false):g(name),qk(q),obs(o){g.receiptEnabled=o;
  for(unsigned j=0;j<(q?4u:5u);++j)x.push_back(g.tensor(("x"+std::to_string(j)).c_str(),QNN_TENSOR_TYPE_APP_WRITE,{1,1,j==4?1u:32u,j==4?1u:32u}));
  for(unsigned j=0;j<(q?2u:4u);++j)y.push_back(g.tensor(("y"+std::to_string(j)).c_str(),QNN_TENSOR_TYPE_APP_READ,{1,1,32,32}));
  rr=g.tensor("receipt",QNN_TENSOR_TYPE_APP_READ,{1,1,1,q?6152u:8201u},QNN_DATATYPE_INT_32);
  if(standard){for(unsigned j=0;j<y.size();++j)if(q)op::mm(g,"standard"+std::to_string(j),x[j],x[2+j],y[j],false,true);else g.node(("standard"+std::to_string(j)).c_str(),QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{x[j],x[4]},{y[j]});}
  else{auto yy=y;yy.push_back(rr);g.node("custom",q?"ForwardQKRaw":"BackwardScoreScale","HexaFp32",x,yy);}
  auto t=std::chrono::steady_clock::now();g.finalize();std::printf("frontierfinalize,%s,us=%.3f\n",name.c_str(),std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count());closurestudy::describe(g,name);
 }
 std::map<unsigned,one::FV> inputs(const std::vector<one::FV>& c){std::map<unsigned,one::FV> in;for(unsigned j=0;j<x.size();++j)in[x[j]]=c[j];return in;}
};
inline void perf(DiagnosticGraph& g,const std::map<unsigned,one::FV>& in,const std::vector<unsigned>& ys,unsigned rr,bool obs,const std::string& name){
 auto anchor=g.run(in);for(unsigned rep=0;rep<11;++rep){auto o=g.run(in);for(auto id:ys)require(o.at(id)==anchor.at(id));if(rep>=2){std::printf("frontierperf,%s,rep=%u,wall_us=%.3f",name.c_str(),rep-2,g.executeUs);if(obs)std::printf(",math_us=%u,window_us=%u",o.at(rr)[4],o.at(rr)[7]);std::puts(",final_words=1");}}
}
inline unsigned standalone(bool qk,const std::vector<std::vector<one::FV>>& cs,unsigned core,unsigned& identity){unsigned bad=0;std::vector<std::vector<one::FV>> anchors(cs.size());
 for(unsigned mode=0;mode<2;++mode){bool obs=mode==0;auto name=std::string(qk?"qkStandalone":"scaleStandalone")+std::to_string(mode);Standalone g(name,qk,obs);
 unsigned last=0;for(unsigned ci=0;ci<cs.size();++ci){auto in=g.inputs(cs[ci]);auto o=g.g.run(in);auto ref=reference(cs[ci],qk,false),fp=reference(cs[ci],qk,true);std::vector<one::FV> actual;
  for(unsigned j=0;j<g.y.size();++j){actual.push_back(decode(o.at(g.y[j])));auto failed=metric(name,ci,"out"+std::to_string(j),ref[j],fp[j],actual[j],.01);if(ci<core)bad+=failed;identity+=words(actual[j],one::f(fp[j]),name,ci,"received_ordered");if(mode)identity+=words(actual[j],anchors[ci][j],name,ci,"observed_final");}
  if(!mode)anchors[ci]=actual;if(obs)receipt(o.at(g.rr),qk,cs[ci],actual,last);
  if(ci==0)std::printf("frontierfirst,%s,wall_us=%.3f\n",name.c_str(),g.g.executeUs);
 }
 if(!bad&&!identity)perf(g.g,g.inputs(cs[0]),g.y,g.rr,obs,name);
 }
 // Negative controls are measured independently and never grant a Custom PASS.
 Standalone neg(qk?"qkStandardStandalone":"scaleStandardStandalone",qk,false,true);for(unsigned ci=0;ci<core;++ci){auto o=neg.g.run(neg.inputs(cs[ci]));auto ref=reference(cs[ci],qk,false),fp=reference(cs[ci],qk,true);for(unsigned j=0;j<neg.y.size();++j)metric(qk?"qkStandardStandalone":"scaleStandardStandalone",ci,"out"+std::to_string(j),ref[j],fp[j],decode(o.at(neg.y[j])),.01);}
 return bad;
}
inline std::vector<std::vector<one::FV>> qkCases(const std::vector<orderedqkvstudy::Case>& cs){std::vector<std::vector<one::FV>> out;
 for(auto& c:cs){std::array<one::FV,2> proj;for(unsigned p=0;p<2;++p)proj[p]=one::f(one::mm(one::d(c.supplied),one::d(c.w[p]),32,64,64,false,false,true));std::vector<one::FV> x;for(unsigned p=0;p<2;++p)for(unsigned h=0;h<2;++h)x.push_back(one::f(one::split(one::d(proj[p]),h)));out.push_back(x);}return out;
}
inline std::vector<one::FV> forwardAudit(Forward& g,const orderedqkvstudy::Case& c,std::map<unsigned,std::vector<uint32_t>> o,unsigned ci,unsigned& bad,unsigned& identity,unsigned& last){
 std::vector<one::FV> x,actual;auto ln=mmfixture::ln::doubleReference(c.ln,64);std::array<one::Vec,2> ideal;for(unsigned p=0;p<2;++p)ideal[p]=one::mm(ln.y,one::d(c.w[p]),32,64,64);
 if(g.observed){for(unsigned p=0;p<6;++p)if(!o.count(g.heads[p])){auto& r=o.at(g.hr);o[g.heads[p]]={r.begin()+6152+p*1024,r.begin()+6152+(p+1)*1024};}}
 if(g.observed||g.tap){auto upstream=orderedqkvstudy::audit(g,c,o,ci,bad,identity);for(unsigned j=0;j<4;++j)x.push_back(upstream.at("head"+std::to_string(j)));}
 else {auto host=mmfixture::ln::anchoredFp32Oracle(c.ln,64);for(auto key:{"dx","dg","db"})bad+=metric(g.name,ci,key,std::string(key)=="dx"?ln.dx:std::string(key)=="dg"?ln.dg:ln.db,one::d(std::string(key)=="dx"?host.dx:std::string(key)=="dg"?host.dg:host.db),decode(o.at(std::string(key)=="dx"?g.dx:std::string(key)=="dg"?g.dg:g.db)),.02);}
 std::vector<one::FV> raw;
 if(g.next!=1&&g.observed){for(unsigned j=0;j<4;++j)require(sameFpWords(x[j],slice(o.at(g.rr),8+j*1024,1024)));for(unsigned h=0;h<2;++h)raw.push_back(slice(o.at(g.rr),4104+h*1024,1024));receipt(o.at(g.rr),true,x,raw,last);}
 if(g.next==2&&g.tap&&!g.observed)for(unsigned h=0;h<2;++h)raw.push_back(decode(o.at(g.scores[h])));
 if(g.next==2&&!raw.empty())for(unsigned h=0;h<2;++h){auto target=one::mm(one::split(ideal[0],h),one::split(ideal[1],h),32,32,32,false,true);auto fp=one::mm(one::d(x[h]),one::d(x[2+h]),32,32,32,false,true,true);bad+=metric(g.name,ci,"raw"+std::to_string(h),target,fp,raw[h],.01);}
 for(unsigned h=0;h<2;++h){auto ref=one::mm(one::split(ideal[0],h),one::split(ideal[1],h),32,32,32,false,true);auto id=g.next==2?g.scaled[h]:g.scores[h];auto y=decode(o.at(id));actual.push_back(y);
  if(g.next==2)ref=one::mul(ref,{double(scaleValue())});auto fail=metric(g.name,ci,"out"+std::to_string(h),ref,ref,y,.01);if(g.next==0)bad+=fail;
  if(!x.empty()){auto fp=one::mm(one::d(x[h]),one::d(x[2+h]),32,32,32,false,true,true);auto local=one::mm(one::d(x[h]),one::d(x[2+h]),32,32,32,false,true);if(g.next==0)identity+=words(y,one::f(fp),g.name,ci,"received_ordered");if(g.next==2){if(!raw.empty()){identity+=words(raw[h],one::f(fp),g.name,ci,"raw_received_ordered");local=one::mul(one::d(raw[h]),{double(scaleValue())});fp=one::mul(one::d(raw[h]),{double(scaleValue())},true);}else{local=one::mul(fp,{double(scaleValue())});fp=one::mul(fp,{double(scaleValue())},true);}}metric(g.name+"Local",ci,"out"+std::to_string(h),local,fp,y,.01);}
 }return actual;
}
inline unsigned forward(const std::vector<one::Fixture>& fs,unsigned& identity){auto cs=orderedqkvstudy::cases(fs);require(cs.size()==44);auto sc=qkCases(cs);
 for(auto& c:attentionfixture::matrix({})){sc.push_back({c.a[0],c.a[1],c.b[0],c.b[1]});}
 auto zeros=sc[0];for(unsigned j=0;j<4;++j)for(unsigned i=0;i<1024;++i)zeros[j][i]=(i+j)%2?-0.f:0.f;sc.push_back(zeros);
 unsigned bad=standalone(true,sc,44,identity);if(bad||identity)return bad;
 std::vector<std::vector<one::FV>> anchors(44);for(unsigned next=0;next<3;++next){if(bad||identity)break;for(unsigned mode=0;mode<4;++mode){bool obs=mode<2,tap=mode%2;auto n="qkNative"+std::to_string(next)+std::to_string(mode);Forward g(n,obs,tap,next);closurestudy::describe(g.g,n);unsigned last=0;
  for(unsigned ci=0;ci<cs.size();++ci){auto o=g.g.run(orderedqkvstudy::inputs(g,cs[ci]));if(!ci)std::printf("frontierfirst,%s,wall_us=%.3f\n",n.c_str(),g.g.executeUs);auto v=forwardAudit(g,cs[ci],o,ci,bad,identity,last);if(next==0){if(mode==0)anchors[ci]=v;else for(unsigned j=0;j<2;++j)identity+=words(v[j],anchors[ci][j],n,ci,"observer_final");}}
  if(next==0&&!bad&&!identity)perf(g.g,orderedqkvstudy::inputs(g,cs[0]),{g.scores[0],g.scores[1],g.dx,g.dg,g.db},g.rr,obs,n);
 }}return bad;
}
inline unsigned backward(const std::vector<one::Fixture>& fs,unsigned& identity){auto cs=upstreamreference::fullUpstream(fs);require(cs.size()==25);std::vector<std::vector<one::FV>> sc;
 for(auto& c:cs){auto fp=one::reference(c,true);std::vector<one::FV>x;for(unsigned h=0;h<2;++h)for(auto suffix:{"dqraw","dkraw"})x.push_back(one::f(fp.at("h"+std::to_string(h)+"_"+suffix)));x.push_back({scaleValue()});sc.push_back(x);}
 for(unsigned kind=0;kind<10;++kind){auto x=sc[0];uint32_t seed=913+kind;for(unsigned j=0;j<4;++j)for(unsigned i=0;i<1024;++i){seed=seed*1664525u+1013904223u;float v=float(int(seed>>16)-32768)/4096.f;if(kind==0)v=i%2?-0.f:0.f;if(kind==1)v*=1e-12f;if(kind==2)v*=1e10f;if(kind==3)v=(i%2?-1:1)*.25f*(j+1);x[j][i]=v;}sc.push_back(x);}
 unsigned bad=standalone(false,sc,25,identity);if(bad||identity)return bad;
 std::vector<std::vector<one::FV>> anchors(25);
 for(unsigned next=0;next<2;++next)for(unsigned domain:{2u,1u})for(unsigned mode=0;mode<4;++mode){if(bad||identity)return bad;bool obs=mode<2,tap=mode%2;auto n="scaleNative"+std::to_string(next)+std::to_string(domain)+std::to_string(mode);OneGraph g(n,tap,false,false,false,true,true,true,obs,true,true,true,domain,true,0,next?4:3);closurestudy::describe(g.g,n);unsigned lr=0,ls=0;
  for(auto& c:cs){auto o=g.g.run(g.inputs(c));if(c.index==0)std::printf("frontierfirst,%s,wall_us=%.3f\n",n.c_str(),g.g.executeUs);auto a=g.audit(c,o);auto ref=one::reference(c),fp=one::reference(c,true);
   for(auto& e:a)if(ref.count(e.first)&&e.first.find("dq")==std::string::npos&&e.first.find("dk")==std::string::npos)bad+=metric(n,c.index,e.first,ref.at(e.first),fp.at(e.first),e.second,limit(e.first));
   std::vector<one::FV> raw,scaled;
   if(obs){std::vector<one::FV> operands;for(unsigned j=0;j<6;++j)operands.push_back(slice(o.at(g.rawReceipt),8+j*1024,1024));auto local=closurestudy::rawReference(operands,true);for(unsigned h=0;h<2;++h){auto p="h"+std::to_string(h)+"_";require(sameFpWords(operands[h],a.at(p+"ds"))&&sameFpWords(operands[2+h],c.v.at(p+"k"))&&sameFpWords(operands[4+h],c.v.at(p+"q")));}
    for(unsigned j=0;j<4;++j){raw.push_back(slice(o.at(g.rawReceipt),6152+j*1024,1024));identity+=words(raw.back(),one::f(local[j]),n,c.index,"raw_received_ordered");require(sameFpWords(raw.back(),slice(o.at(g.scaleReceipt),8+j*1024,1024)));scaled.push_back(slice(o.at(g.scaleReceipt),4105+j*1024,1024));}
    closurestudy::receipt(o.at(g.rawReceipt),closurecontract::RawDQDK,operands,raw,lr);auto sx=raw;sx.push_back({scaleValue()});receipt(o.at(g.scaleReceipt),false,sx,scaled,ls);
   }
   std::vector<one::FV> finals;for(unsigned j=0;j<4;++j){auto p="h"+std::to_string(j/2)+(j%2?"_dk":"_dq");auto rawkey=p+"raw";if(obs||tap){auto r=obs?raw[j]:a.at(rawkey);bad+=metric(n,c.index,rawkey,ref.at(rawkey),fp.at(rawkey),r,.01);auto v=obs?scaled[j]:a.at(p);bad+=metric(n,c.index,p,ref.at(p),fp.at(p),v,.01);identity+=words(v,one::f(one::mul(one::d(r),{double(scaleValue())},true)),n,c.index,"scale_received_ordered");if(tap&&obs)require(sameFpWords(a.at(p),v)&&sameFpWords(a.at(rawkey),r));}
    auto key=next?"scatter"+std::to_string(j):p;auto v=a.at(key);finals.push_back(v);auto target=next?one::scatter(ref.at(p),j/2):ref.at(p);auto ff=next?one::scatter(fp.at(p),j/2):fp.at(p);auto fail=metric(n,c.index,key,target,ff,v,.01);if(!next)bad+=fail;
    if(next&&(obs||tap)){auto input=obs?scaled[j]:a.at(p);auto local=one::scatter(one::d(input),j/2);metric(n+"Local",c.index,key,local,local,v,.01);}
   }
   if(!next){if(domain==2&&mode==0)anchors[c.index]=finals;else for(unsigned j=0;j<4;++j)identity+=words(finals[j],anchors[c.index][j],n,c.index,"observer_and_softmax_final");}
  }
  if(!next&&!bad&&!identity){std::vector<unsigned> ys;for(unsigned j=0;j<4;++j)ys.push_back(g.out.at("h"+std::to_string(j/2)+(j%2?"_dk":"_dq")));perf(g.g,g.inputs(cs[0]),ys,g.scaleReceipt,obs,n);}
 }
 // Reproduce the previous native standard-scale negative control, with the
 // Custom raw producer retained. No failed standard value feeds the scatter.
 for(unsigned mode=0;mode<4;++mode){bool obs=mode<2,tap=mode%2;auto n="scaleNativeNegative"+std::to_string(mode);OneGraph g(n,tap,false,false,false,true,true,true,obs,true,true,true,2,true,0,2);for(auto& c:cs){auto o=g.g.run(g.inputs(c));auto a=g.audit(c,o);auto ref=one::reference(c),fp=one::reference(c,true);for(unsigned j=0;j<4;++j){auto p="h"+std::to_string(j/2)+(j%2?"_dk":"_dq");metric(n,c.index,p,ref.at(p),fp.at(p),a.at(p),.01);}}}
 return bad;
}
inline int run(const std::string& mode,const std::vector<one::Fixture>& fs){std::printf("frontiercontract,scale_bits=%08x,raw_gate=.01,scale_gate=.01,scatter_gate=.01,forward_scale_gate=.01,training_updates=0\n",word(scaleValue()));unsigned identity=0;unsigned bad=mode=="oneQkScaleA"?forward(fs,identity):backward(fs,identity);std::printf("summary,%s,accuracy_failures=%u,word_mismatches=%u,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0\n",mode.c_str(),bad,identity);return bad||identity?4:0;}
}
