// Reuse the original descriptor/graph construction audit, then inject failures
// against the exact ownership class used by the independent ARM candidate.
#define main originalLifecycleAudit
#include "backward_repair_cohort_lifecycle_test.cpp"
#undef main
#include "backward_shared_lifetime_formal.h"
#include <set>
namespace testlife {
std::string failure;unsigned occurrence=0,target=1;bool exitInjection=false;bool b=false,d=false;std::set<Qnn_ContextHandle_t> live;
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
 auto rc=gate("graphFinalize");if(rc){if(exitInjection)return rc;throw std::runtime_error("injected graphFinalize");}return audit::mFinalize(g,p,s);
}
void execute(Qnn_GraphHandle_t g){demand(audit::st.graphs.count(g)&&!audit::st.graphs[g].dead&&audit::st.graphs[g].finalized&&live.count(audit::st.graphs[g].context));if(gate("graphExecute"))throw std::runtime_error("injected graphExecute");}
Qnn_ErrorHandle_t graphCreate(Qnn_ContextHandle_t c,const char* n,const QnnGraph_Config_t** cfg,Qnn_GraphHandle_t* out){demand(live.count(c));auto rc=gate("graphCreate");if(rc){if(exitInjection)return rc;throw std::runtime_error("injected graphCreate");}return audit::mGraphCreate(c,n,cfg,out);}
Qnn_ErrorHandle_t addNode(Qnn_GraphHandle_t g,Qnn_OpConfig_t cfg){demand(audit::st.graphs.count(g)&&!audit::st.graphs[g].dead&&live.count(audit::st.graphs[g].context));auto rc=gate("graphAddNode");if(rc){if(exitInjection)return rc;throw std::runtime_error("injected graphAddNode");}return audit::mAddNode(g,cfg);}
Qnn_ErrorHandle_t executeApi(Qnn_GraphHandle_t,const Qnn_Tensor_t*,uint32_t,Qnn_Tensor_t*,uint32_t,Qnn_ProfileHandle_t,Qnn_SignalHandle_t){return 1;}
void install(){audit::install();api.logCreate=logCreate;api.backendCreate=backendCreate;api.deviceCreate=deviceCreate;api.backendRegisterOpPackage=reg;api.contextCreate=contextCreate;api.contextFree=contextFree;api.deviceFree=deviceFree;api.backendFree=backendFree;api.graphFinalize=finalize;api.graphCreate=graphCreate;api.graphAddNode=addNode;api.graphExecute=executeApi;}
void reset(const std::string& f,unsigned at){audit::st={};failure=f;exitInjection=false;occurrence=0;target=at;b=d=false;live.clear();calls.clear();packages.clear();backend=nullptr;device=nullptr;context=nullptr;install();}
void run(const char* name,const std::string& fail="",unsigned at=1,unsigned early=0){
 reset(fail,at);bool failed=false;Qnn_GraphHandle_t stale{};formalsharedlife::Owner owner;
 try {
  owner.start();
  for(unsigned role=0;role<5;++role){owner.beginRole();for(bool o:{true,false})for(bool t:{true,false}){
   repairprobe::Graph z(audit::kModes[role],o,t);stale=z.g.graph;execute(stale);
  }owner.endRole();if(early==role+1)break;}
 }catch(const std::exception&){failed=true;}
 bool cleaned=owner.cleanup();demand(!owner.cleanup());
 if(fail=="contextFree"){demand(!cleaned&&owner.poisoned&&calls["contextFree"]==at&&calls["deviceFree"]==0&&calls["backendFree"]==0);}
 else if(fail=="deviceFree"){demand(!cleaned&&owner.poisoned&&live.empty()&&d&&b&&calls["backendFree"]==0);}
 else if(fail=="backendFree"){demand(!cleaned&&owner.poisoned&&live.empty()&&!d&&b&&calls["backendFree"]==1);}
 else{demand(cleaned&&live.empty()&&!b&&!d);}
 if(fail.empty()){
  demand(!failed&&calls["backendCreate"]==1&&calls["backendFree"]==1&&calls["deviceCreate"]==1&&calls["deviceFree"]==1&&calls["contextCreate"]==(early?early:5u)&&calls["contextFree"]==(early?early:5u));
  demand(packages.size()==10&&audit::st.graphCount==(early?early*4u:20u)&&audit::st.descriptorFaults==0&&audit::st.handleFaults==0);
  bool rejected=false;try{execute(stale);}catch(...){rejected=true;}demand(rejected);
  rejected=false;try{Qnn_GraphHandle_t bad{};graphCreate(audit::st.graphs.at(stale).context,"stale",nullptr,&bad);}catch(...){rejected=true;}demand(rejected);
  rejected=false;try{owner.beginRole();}catch(...){rejected=true;}demand(rejected);
  rejected=false;try{owner.endRole();}catch(...){rejected=true;}demand(rejected);
 }else demand(failed||fail=="deviceFree"||fail=="backendFree");
 std::printf("shared_lifetime_case=%s PASS backend_create=%u backend_free=%u device_create=%u device_free=%u context_create=%u context_free=%u\n",name,calls["backendCreate"],calls["backendFree"],calls["deviceCreate"],calls["deviceFree"],calls["contextCreate"],calls["contextFree"]);
}
}
int main(int argc,char** argv){
 if(originalLifecycleAudit())return 1;
 using namespace testlife;
 try{
  if(argc==2){
   std::string name=argv[1];reset(name,1);exitInjection=true;formalsharedlife::Owner owner;formalsharedlife::owner=&owner;
   std::atexit([](){if(b||d||!live.empty()||calls["contextFree"]!=1||calls["deviceFree"]!=1||calls["backendFree"]!=1)std::_Exit(9);std::puts("formal_atexit_cleanup=PASS API_FAILURE_SEPARATE=1");});
   std::atexit(formalsharedlife::emergencyCleanup);owner.start();owner.beginRole();repairprobe::Graph z(savedp::Mode::StandaloneSoftmax,true,true);
   if(name=="graphExecute"){std::map<unsigned,std::vector<float>> in;for(auto id:z.g.inputs)in[id]=std::vector<float>(z.g.size(id),0.f);z.g.run(in);}
   throw std::runtime_error("exit fault did not trigger");
  }
  run("normal_five_role");run("logger_create","logCreate");run("backend_create","backendCreate");run("device_create","deviceCreate");run("package_registration","packageRegister");run("repair_registration","packageRegister",9);
  for(unsigned role=1;role<=5;++role){
   run(("context_create_role"+std::to_string(role)).c_str(),"contextCreate",role);
   run(("finalize_role"+std::to_string(role)).c_str(),"graphFinalize",(role-1)*4+1);
   run(("execute_role"+std::to_string(role)).c_str(),"graphExecute",(role-1)*4+1);
   run(("context_free_role"+std::to_string(role)).c_str(),"contextFree",role);
   run(("early_exit_role"+std::to_string(role)).c_str(),"",1,role);
  }
  run("graph_create","graphCreate");run("graph_add_node","graphAddNode");run("device_free","deviceFree");run("backend_free","backendFree");
  std::puts("formal_shared_lifetime_host=PASS five_roles=5 contexts=5 graphs=20 backend=1 device=1 fault_injection=PASS stale_handles=PASS logger_validity=PASS HOST_ONLY=1");
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}
}
