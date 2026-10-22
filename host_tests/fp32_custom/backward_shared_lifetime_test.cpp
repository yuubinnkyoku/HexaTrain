// Reuse the original descriptor/graph construction audit, then inject failures
// against the exact ownership class used by the independent ARM candidate.
#define main originalLifecycleAudit
#include "backward_repair_cohort_lifecycle_test.cpp"
#undef main
#include "backward_shared_lifetime.h"
#include <set>
namespace testlife {
std::string failure;unsigned occurrence=0,target=1;bool b=false,d=false;std::set<Qnn_ContextHandle_t> live;
std::map<std::string,unsigned> calls;std::set<std::string> packages;
Qnn_ErrorHandle_t gate(const char* n){++calls[n];audit::note(n);if(failure==n&&++occurrence==target)return 1;return QNN_SUCCESS;}
void demand(bool x){if(!x)throw std::runtime_error("mock ownership contract");}
Qnn_ErrorHandle_t logCreate(QnnLog_Callback_t,QnnLog_Level_t,Qnn_LogHandle_t* p){*p=(Qnn_LogHandle_t)0x5001;return gate("logCreate");}
Qnn_ErrorHandle_t backendCreate(Qnn_LogHandle_t l,const QnnBackend_Config_t**,Qnn_BackendHandle_t* p){demand(l==(Qnn_LogHandle_t)0x5001&&!b);auto rc=gate("backendCreate");if(!rc){b=true;*p=(Qnn_BackendHandle_t)0x6001;}return rc;}
Qnn_ErrorHandle_t reg(Qnn_BackendHandle_t h,const char* p,const char*,const char*){demand(b&&h==backend);auto rc=gate("packageRegister");if(!rc)demand(packages.insert(p).second);return rc;}
Qnn_ErrorHandle_t deviceCreate(Qnn_LogHandle_t l,const QnnDevice_Config_t**,Qnn_DeviceHandle_t* p){demand(b&&!d&&l==(Qnn_LogHandle_t)0x5001);auto rc=gate("deviceCreate");if(!rc){d=true;*p=(Qnn_DeviceHandle_t)0x7001;}return rc;}
Qnn_ErrorHandle_t contextCreate(Qnn_BackendHandle_t bh,Qnn_DeviceHandle_t dh,const QnnContext_Config_t**,Qnn_ContextHandle_t* p){demand(b&&d&&live.empty()&&bh==backend&&dh==device);auto rc=gate("contextCreate");if(!rc){*p=(Qnn_ContextHandle_t)(uintptr_t(0x8000+calls["contextCreate"]));live.insert(*p);audit::st.liveContexts.push_back(*p);audit::st.contexts[*p]={};}return rc;}
Qnn_ErrorHandle_t contextFree(Qnn_ContextHandle_t c,Qnn_ProfileHandle_t){demand(live.count(c));auto rc=gate("contextFree");if(!rc){audit::mContextFree(c,nullptr);live.erase(c);}return rc;}
Qnn_ErrorHandle_t deviceFree(Qnn_DeviceHandle_t h){demand(d&&live.empty()&&h==device);auto rc=gate("deviceFree");if(!rc)d=false;return rc;}
Qnn_ErrorHandle_t backendFree(Qnn_BackendHandle_t h){demand(b&&!d&&live.empty()&&h==backend);auto rc=gate("backendFree");if(!rc)b=false;return rc;}
Qnn_ErrorHandle_t finalize(Qnn_GraphHandle_t g,Qnn_ProfileHandle_t p,Qnn_SignalHandle_t s){
 demand(audit::st.graphs.count(g)&&!audit::st.graphs[g].dead&&live.count(audit::st.graphs[g].context));
 auto rc=gate("graphFinalize");if(rc)throw std::runtime_error("injected graphFinalize");return audit::mFinalize(g,p,s);
}
void execute(Qnn_GraphHandle_t g){demand(audit::st.graphs.count(g)&&!audit::st.graphs[g].dead&&audit::st.graphs[g].finalized&&live.count(audit::st.graphs[g].context));if(gate("graphExecute"))throw std::runtime_error("injected graphExecute");}
void install(){audit::install();api.logCreate=logCreate;api.backendCreate=backendCreate;api.deviceCreate=deviceCreate;api.backendRegisterOpPackage=reg;api.contextCreate=contextCreate;api.contextFree=contextFree;api.deviceFree=deviceFree;api.backendFree=backendFree;api.graphFinalize=finalize;}
void reset(const std::string& f,unsigned at){audit::st={};failure=f;occurrence=0;target=at;b=d=false;live.clear();calls.clear();packages.clear();backend=nullptr;device=nullptr;context=nullptr;install();}
void run(const char* name,const std::string& fail="",unsigned at=1,bool early=false){
 reset(fail,at);bool failed=false;Qnn_GraphHandle_t stale{};sharedlife::Owner owner;
 try {
  owner.start();
  for(unsigned role=0;role<2;++role){owner.beginRole();for(bool o:{true,false})for(bool t:{true,false}){
   repairprobe::Graph z(role?savedp::Mode::StandaloneDValue:savedp::Mode::StandaloneSoftmax,o,t);stale=z.g.graph;execute(stale);
  }owner.endRole();if(early)break;}
 }catch(const std::exception&){failed=true;}
 bool cleaned=owner.cleanup();demand(!owner.cleanup());
 if(fail=="contextFree"){demand(!cleaned&&owner.poisoned&&calls["contextFree"]==at&&calls["deviceFree"]==0&&calls["backendFree"]==0);}
 else{demand(cleaned&&live.empty()&&!b&&!d);}
 if(fail.empty()){
  demand(!failed&&calls["backendCreate"]==1&&calls["backendFree"]==1&&calls["deviceCreate"]==1&&calls["deviceFree"]==1&&calls["contextCreate"]==(early?1u:2u)&&calls["contextFree"]==(early?1u:2u));
  demand(packages.size()==10&&audit::st.graphCount==(early?4u:8u)&&audit::st.descriptorFaults==0&&audit::st.handleFaults==0);
  bool rejected=false;try{execute(stale);}catch(...){rejected=true;}demand(rejected);
  rejected=false;try{owner.beginRole();}catch(...){rejected=true;}demand(rejected);
  rejected=false;try{owner.endRole();}catch(...){rejected=true;}demand(rejected);
 }else demand(failed);
 std::printf("shared_lifetime_case=%s PASS backend_create=%u backend_free=%u device_create=%u device_free=%u context_create=%u context_free=%u\n",name,calls["backendCreate"],calls["backendFree"],calls["deviceCreate"],calls["deviceFree"],calls["contextCreate"],calls["contextFree"]);
}
}
int main(){
 if(originalLifecycleAudit())return 1;
 using namespace testlife;
 try{
  run("normal");run("backend_create","backendCreate");run("device_create","deviceCreate");
  run("softmax_context_create","contextCreate");run("softmax_finalize","graphFinalize");run("softmax_teardown","contextFree");
  run("dvalue_context_create","contextCreate",2);run("dvalue_finalize","graphFinalize",5);run("dvalue_execute","graphExecute",5);run("dvalue_teardown","contextFree",2);
  run("early_exit","",1,true);run("package_registration","packageRegister");run("repair_registration","packageRegister",9);
  // Repeated cleanup, stale graph/context and logger validity are asserted in
  // every successful case. Free failure stops parents and never retries.
  std::puts("shared_lifetime_host=PASS fault_injection=PASS stale_handle_rejection=PASS logger_valid_through_backend_free=PASS HOST_ONLY=1");
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}
}
