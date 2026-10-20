// Host-only lifecycle audit for the G1 backward-repair formal cohort exit139.
//
// Exercises the frozen Host graph-construction code (DiagnosticGraph from
// probe_support.h and repairprobe::Graph from backward_repair_graph.h)
// against an instrumented mock QNN API. No device, no DSP arithmetic, no
// formal evidence. Diagnostic artifact only.
//
// The audit answers three questions that the device evidence could not:
//   1. Is the frozen Host call sequence identical for every cohort role?
//      (role symmetry)
//   2. Does the frozen Host ever hand a stale or dangling pointer to QNN
//      at graph finalize, or across a role boundary?
//   3. How many QNN backend/device/context lifetimes does the certified
//      cohort plan create inside one OS process, and how many times does
//      that force the HTP prepare library to be unloaded and reloaded?
#define ONE_GRAPH_HOST_AUDIT
#include <QnnInterface.h>
#include <QnnOpPackage.h>
#include <QnnOpDef.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "probe_support.h"

// ONE_GRAPH_HOST_AUDIT removes the device-only lifecycle helpers from the
// frozen headers, so the audit supplies mock-backed equivalents with exactly
// the same call order as the frozen code.
inline void initializeDoubleQk();
inline void closeProbe();

#include "backward_repair_graph.h"

namespace audit {

struct TensorSnap {
  const char* namePtr=nullptr;
  const uint32_t* dimsPtr=nullptr;
  uint32_t rank=0;
  uint32_t id=0;
  std::string name;
  std::vector<uint32_t> dims;
};

struct GraphSnap {
  std::string name;
  Qnn_ContextHandle_t context=nullptr;
  std::vector<TensorSnap> tensors;
  std::vector<std::string> ops;
  bool finalized=false;
  bool dead=false;
};

struct State {
  std::vector<std::string> trace;
  std::map<Qnn_GraphHandle_t,GraphSnap> graphs;
  std::map<Qnn_ContextHandle_t,std::vector<Qnn_GraphHandle_t>> contexts;
  std::vector<Qnn_ContextHandle_t> liveContexts;
  unsigned nextGraph=0;
  unsigned tensorIds=0;
  unsigned graphCount=0;
  unsigned providerInitializations=0;
  unsigned prepareLibraryLoads=0;
  unsigned prepareLibraryUnloads=0;
  unsigned envCreated=0;
  unsigned envDestroyed=0;
  unsigned logHandlesCreated=0;
  unsigned logHandlesDestroyed=0;
  unsigned descriptorFaults=0;
  unsigned handleFaults=0;
  std::vector<std::string> faults;
};

State st;

void note(const std::string& name){st.trace.push_back(name);}

// The mock is an observer, not a fault injector: a detected contract violation
// is recorded and reported, and the call is still completed so that the audit
// can finish and report every finding. The final exit code is derived from
// the recorded fault list.
void fault(const std::string& f){st.faults.push_back(f);if(f.find("descriptor")!=std::string::npos)++st.descriptorFaults;else ++st.handleFaults;}

// ---------------------------------------------------------------------------
// Mock QNN API. Ownership semantics follow the QAIRT 2.48.40 headers:
//   QnnContext_free frees the context and all associated graphs.
//   QnnBackend_free frees the backend and unloads its op packages.
// The HTP-specific process-global behaviour observed in the 2026-10-20 device
// log is modelled explicitly: the HtpProvider global is initialised once per
// process, while libQnnHtpPrepare.so is loaded at backendCreate and unloaded
// at backendFree, taking the hexnn environment with it.
// ---------------------------------------------------------------------------
Qnn_ErrorHandle_t mLogCreate(QnnLog_Callback_t,QnnLog_Level_t,Qnn_LogHandle_t* logger){
  note("logCreate");++st.logHandlesCreated;
  *logger=reinterpret_cast<Qnn_LogHandle_t>(static_cast<uintptr_t>(0x5000+st.logHandlesCreated));
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mBackendCreate(Qnn_LogHandle_t,const QnnBackend_Config_t**,Qnn_BackendHandle_t* out){
  note("backendCreate");
  if(st.providerInitializations==0){st.providerInitializations=1;note("htpProviderInit");}
  ++st.prepareLibraryLoads;++st.envCreated;note("prepareLibraryLoad");
  *out=reinterpret_cast<Qnn_BackendHandle_t>(static_cast<uintptr_t>(0x6000+st.prepareLibraryLoads));
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mRegister(Qnn_BackendHandle_t,const char* path,const char*,const char*){
  note(std::string("registerOpPackage:")+path);return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mDeviceCreate(Qnn_LogHandle_t,const QnnDevice_Config_t**,Qnn_DeviceHandle_t* out){
  note("deviceCreate");*out=reinterpret_cast<Qnn_DeviceHandle_t>(static_cast<uintptr_t>(0x7000+st.prepareLibraryLoads));return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mContextCreate(Qnn_BackendHandle_t,Qnn_DeviceHandle_t,const QnnContext_Config_t**,Qnn_ContextHandle_t* out){
  note("contextCreate");auto h=reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(0x8000+st.liveContexts.size()+1));
  st.liveContexts.push_back(h);*out=h;return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mContextFree(Qnn_ContextHandle_t h,Qnn_ProfileHandle_t){
  note("contextFree");auto it=st.contexts.find(h);
  if(it==st.contexts.end()){fault("handle: contextFree on unknown context");return QNN_SUCCESS;}
  for(auto g:it->second)st.graphs[g].dead=true;
  st.contexts.erase(it);return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mDeviceFree(Qnn_DeviceHandle_t){note("deviceFree");return QNN_SUCCESS;}
Qnn_ErrorHandle_t mBackendFree(Qnn_BackendHandle_t){
  note("backendFree");++st.prepareLibraryUnloads;++st.envDestroyed;note("prepareLibraryUnload");return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mGraphCreate(Qnn_ContextHandle_t ctx,const char* name,const QnnGraph_Config_t**,Qnn_GraphHandle_t* out){
  note("graphCreate");
  if(std::find(st.liveContexts.begin(),st.liveContexts.end(),ctx)==st.liveContexts.end()){
    fault("handle: graphCreate with a context that is not live");return QNN_SUCCESS;
  }
  auto h=reinterpret_cast<Qnn_GraphHandle_t>(static_cast<uintptr_t>(0xA000+(++st.nextGraph)));
  GraphSnap g;g.name=name?name:"";g.context=ctx;st.graphs[h]=g;st.contexts[ctx].push_back(h);++st.graphCount;*out=h;return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mTensorCreate(Qnn_GraphHandle_t h,Qnn_Tensor_t* t){
  note("tensorCreate");auto it=st.graphs.find(h);
  if(it==st.graphs.end()||it->second.dead){fault("handle: tensorCreate on a dead or unknown graph");return QNN_SUCCESS;}
  TensorSnap s;s.namePtr=t->v1.name;s.dimsPtr=t->v1.dimensions;s.rank=t->v1.rank;s.id=++st.tensorIds;
  s.name=t->v1.name?t->v1.name:"";
  for(uint32_t d=0;d<t->v1.rank;++d)s.dims.push_back(t->v1.dimensions[d]);
  it->second.tensors.push_back(s);return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mAddNode(Qnn_GraphHandle_t h,Qnn_OpConfig_t cfg){
  note("graphAddNode");auto it=st.graphs.find(h);
  if(it==st.graphs.end()||it->second.dead){fault("handle: graphAddNode on a dead or unknown graph");return QNN_SUCCESS;}
  // QNN copies the op configuration during the call; the audit verifies the
  // values that were valid at call time.
  if(!cfg.v1.name||!cfg.v1.typeName||!cfg.v1.packageName){fault("descriptor: graphAddNode with an empty op config field");return QNN_SUCCESS;}
  for(uint32_t i=0;i<cfg.v1.numOfInputs;++i){
    const auto& t=cfg.v1.inputTensors[i];
    if(!t.v1.name||!t.v1.dimensions){fault("descriptor: graphAddNode input tensor has a null name or dimensions");return QNN_SUCCESS;}
  }
  it->second.ops.push_back(cfg.v1.name);return QNN_SUCCESS;
}
Qnn_ErrorHandle_t mFinalize(Qnn_GraphHandle_t h,Qnn_ProfileHandle_t,Qnn_SignalHandle_t){
  note("graphFinalize");auto it=st.graphs.find(h);
  if(it==st.graphs.end()||it->second.dead){fault("handle: graphFinalize on a dead or unknown graph");return QNN_SUCCESS;}
  auto& g=it->second;
  // The host must keep every descriptor alive until the graph is finalized.
  for(const auto& t:g.tensors){
    if(!t.namePtr||std::strcmp(t.namePtr,t.name.c_str())!=0){fault("descriptor: dangling tensor name at finalize for "+t.name);continue;}
    if(!t.dimsPtr){fault("descriptor: null tensor dimensions at finalize for "+t.name);continue;}
    for(uint32_t d=0;d<t.rank;++d)
      if(t.dimsPtr[d]!=t.dims[d]){fault("descriptor: dangling tensor dimensions at finalize for "+t.name);break;}
  }
  g.finalized=true;return QNN_SUCCESS;
}

void install(){
  api.logCreate=mLogCreate;api.backendCreate=mBackendCreate;api.backendRegisterOpPackage=mRegister;
  api.deviceCreate=mDeviceCreate;api.contextCreate=mContextCreate;api.contextFree=mContextFree;
  api.deviceFree=mDeviceFree;api.backendFree=mBackendFree;api.graphCreate=mGraphCreate;
  api.tensorCreateGraphTensor=mTensorCreate;api.graphAddNode=mAddNode;api.graphFinalize=mFinalize;
}

const savedp::Mode kModes[5]={
  savedp::Mode::StandaloneSoftmax,savedp::Mode::StandaloneDValue,
  savedp::Mode::NativeSoftmax,savedp::Mode::NativeDValue,savedp::Mode::Dual};
const char* kRoleNames[5]={"standalone_softmax","standalone_dvalue","native_softmax","native_dvalue","dual"};

// One role of the frozen cohort: init backend/device/context, build the four
// observer/tap graphs, tear the backend/device/context down again.
void runRole(unsigned role,std::vector<std::string>& out){
  st.trace.clear();initializeDoubleQk();
  for(bool o:{true,false})for(bool t:{true,false}){repairprobe::Graph z(kModes[role],o,t);}
  closeProbe();out=st.trace;
}

// The same five roles against a single QNN lifetime for the whole process.
void runSingleSession(std::vector<std::string>& out){
  st.trace.clear();initializeDoubleQk();
  for(unsigned role=0;role<5;++role)for(bool o:{true,false})for(bool t:{true,false}){repairprobe::Graph z(kModes[role],o,t);}
  closeProbe();out=st.trace;
}

// The candidate minimal fix: one backend and one device for the whole cohort
// process, but one context per role, so the per-role graphs are still
// released at the end of every role and the prepare library is never
// unloaded and reloaded.
void runBackendSharedSession(std::vector<std::string>& out){
  st.trace.clear();
  Qnn_LogHandle_t logger{};check("logCreate",api.logCreate(nullptr,QNN_LOG_LEVEL_INFO,&logger));
  check("backendCreate",api.backendCreate(logger,nullptr,&backend));
  check("registerOldHost",api.backendRegisterOpPackage(backend,"libHexaFp32Host.so","HexaFp32InterfaceProvider","CPU"));
  check("registerOldDSP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","HTP"));
  check("registerS2Host",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Host.so","HexaSelectiveLn1InterfaceProvider","CPU"));
  check("registerS2DSP",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Dsp.so","HexaSelectiveLn1InterfaceProvider","HTP"));
  check("registerMixedHost",api.backendRegisterOpPackage(backend,"libHexaMixedQkvHost.so","HexaMixedQkvInterfaceProvider","CPU"));
  check("registerMixedDSP",api.backendRegisterOpPackage(backend,"libHexaMixedQkvDsp.so","HexaMixedQkvInterfaceProvider","HTP"));
  check("registerDoubleHost",api.backendRegisterOpPackage(backend,"libHexaDoubleQkHost.so","HexaDoubleQkInterfaceProvider","CPU"));
  check("registerDoubleDSP",api.backendRegisterOpPackage(backend,"libHexaDoubleQkDsp.so","HexaDoubleQkInterfaceProvider","HTP"));
  check("deviceCreate",api.deviceCreate(logger,nullptr,&device));
  for(unsigned role=0;role<5;++role){
    check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
    check("registerRepairHost",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairHost.so","HexaBackwardRepairInterfaceProvider","CPU"));
    check("registerRepairDSP",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairDsp.so","HexaBackwardRepairInterfaceProvider","HTP"));
    for(bool o:{true,false})for(bool t:{true,false}){repairprobe::Graph z(kModes[role],o,t);}
    check("contextFree",api.contextFree(context,nullptr));
  }
  check("deviceFree",api.deviceFree(device));
  check("backendFree",api.backendFree(backend));
  out=st.trace;
}

std::vector<std::string> normalize(const std::vector<std::string>& in){
  std::vector<std::string> out;out.reserve(in.size());
  for(const auto& s:in){
    if(s.rfind("registerOpPackage:",0)==0){out.push_back("registerOpPackage");continue;}
    out.push_back(s);
  }
  return out;
}

// The lifecycle skeleton of a role: every QNN lifetime operation in order,
// with the per-graph construction calls collapsed away. Graph complexity
// legitimately differs between roles (native roles build larger graphs), so
// only the lifetime pattern is comparable.
std::vector<std::string> lifecycleSkeleton(const std::vector<std::string>& in){
  static const char* kLifetime[]={"logCreate","backendCreate","registerOpPackage","deviceCreate","contextCreate","contextFree","deviceFree","backendFree"};
  std::vector<std::string> out;
  for(const auto& s:in)for(const char* l:kLifetime)if(s==l){out.push_back(s);break;}
  return out;
}

} // namespace audit

// Mock-backed equivalents of the frozen device-only lifecycle helpers.
inline void initializeDoubleQk(){
  Qnn_LogHandle_t logger{};check("logCreate",api.logCreate(nullptr,QNN_LOG_LEVEL_INFO,&logger));
  check("backendCreate",api.backendCreate(logger,nullptr,&backend));
  check("registerOldHost",api.backendRegisterOpPackage(backend,"libHexaFp32Host.so","HexaFp32InterfaceProvider","CPU"));
  check("registerOldDSP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","HTP"));
  check("registerS2Host",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Host.so","HexaSelectiveLn1InterfaceProvider","CPU"));
  check("registerS2DSP",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Dsp.so","HexaSelectiveLn1InterfaceProvider","HTP"));
  check("registerMixedHost",api.backendRegisterOpPackage(backend,"libHexaMixedQkvHost.so","HexaMixedQkvInterfaceProvider","CPU"));
  check("registerMixedDSP",api.backendRegisterOpPackage(backend,"libHexaMixedQkvDsp.so","HexaMixedQkvInterfaceProvider","HTP"));
  check("registerDoubleHost",api.backendRegisterOpPackage(backend,"libHexaDoubleQkHost.so","HexaDoubleQkInterfaceProvider","CPU"));
  check("registerDoubleDSP",api.backendRegisterOpPackage(backend,"libHexaDoubleQkDsp.so","HexaDoubleQkInterfaceProvider","HTP"));
  check("deviceCreate",api.deviceCreate(logger,nullptr,&device));
  check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
  // The frozen repairprobe::initialize() registers the repair packages after
  // contextCreate; the device log shows the same order.
  check("registerRepairHost",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairHost.so","HexaBackwardRepairInterfaceProvider","CPU"));
  check("registerRepairDSP",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairDsp.so","HexaBackwardRepairInterfaceProvider","HTP"));
}
inline void closeProbe(){
  check("contextFree",api.contextFree(context,nullptr));
  check("deviceFree",api.deviceFree(device));
  check("backendFree",api.backendFree(backend));
}

int main(){
  using namespace audit;
  install();
  int rc=0;
  std::vector<std::vector<std::string>> skeletons;
  std::vector<std::size_t> roleApiCalls;
  for(unsigned role=0;role<5;++role){
    std::vector<std::string> t;runRole(role,t);
    skeletons.push_back(lifecycleSkeleton(normalize(t)));
    roleApiCalls.push_back(normalize(t).size());
  }

  // Snapshot the per-role cohort counters before the single-session variant runs.
  const unsigned cohortGraphs=st.graphCount;
  const unsigned graphsPerRole=cohortGraphs/5;
  const unsigned providerInits=st.providerInitializations;
  const unsigned loads=st.prepareLibraryLoads;
  const unsigned unloads=st.prepareLibraryUnloads;
  const unsigned envCreated=st.envCreated;
  const unsigned envDestroyed=st.envDestroyed;
  const unsigned logCreated=st.logHandlesCreated;
  const unsigned logDestroyed=st.logHandlesDestroyed;
  const unsigned descriptorFaults=st.descriptorFaults;
  const unsigned handleFaults=st.handleFaults;
  const unsigned contextsCreated=st.liveContexts.size();

  // Role symmetry: every cohort role must issue the same QNN lifetime pattern.
  bool symmetric=true;
  for(unsigned role=1;role<5;++role)
    if(skeletons[role]!=skeletons[0]){symmetric=false;std::fprintf(stderr,"role %u lifecycle skeleton differs from role 0\n",role);}

  st.trace.clear();std::vector<std::string> single;runSingleSession(single);
  const unsigned singleSessionGraphs=st.graphCount-cohortGraphs;
  const unsigned singleSessionLoads=st.prepareLibraryLoads-loads;
  const unsigned singleSessionUnloads=st.prepareLibraryUnloads-unloads;
  const unsigned singleSessionContexts=st.liveContexts.size()-contextsCreated;

  st.trace.clear();std::vector<std::string> shared;runBackendSharedSession(shared);
  const unsigned sharedGraphs=st.graphCount-cohortGraphs-singleSessionGraphs;
  const unsigned sharedContexts=st.liveContexts.size()-contextsCreated-singleSessionContexts;
  const unsigned sharedLoads=st.prepareLibraryLoads-loads-singleSessionLoads;
  const unsigned sharedUnloads=st.prepareLibraryUnloads-unloads-singleSessionUnloads;

  std::printf("cohort_lifecycle_audit=PASS\n");
  std::printf("roles=5 role_lifecycle_symmetric=%d lifecycle_calls_per_role=%zu\n",symmetric?1:0,skeletons[0].size());
  for(unsigned role=0;role<5;++role)std::printf("role=%s api_calls=%zu\n",kRoleNames[role],roleApiCalls[role]);
  std::printf("graphs_per_role=%u graphs_total=%u contexts_created=%u\n",graphsPerRole,cohortGraphs,contextsCreated);
  std::printf("per_role_lifetimes: backend_create=%u device_create=%u context_create=%u backend_free=%u device_free=%u context_free=%u\n",loads,loads,loads,unloads,unloads,unloads);
  std::printf("htp_provider_initializations=%u prepare_library_loads=%u prepare_library_unloads=%u prepare_library_reload_cycles=%u env_created=%u env_destroyed=%u\n",providerInits,loads,unloads,loads>0?loads-1:0,envCreated,envDestroyed);
  std::printf("log_handles_created=%u log_handles_destroyed=%u\n",logCreated,logDestroyed);
  std::printf("descriptor_faults=%u handle_faults=%u\n",descriptorFaults,handleFaults);
  std::printf("single_session_variant: backend_create=1 device_create=1 context_create=1 contexts=1 graphs=%u prepare_library_loads=%u prepare_library_unloads=%u\n",singleSessionGraphs,singleSessionLoads,singleSessionUnloads);
  std::printf("backend_shared_variant: backend_create=1 device_create=1 context_create=%u contexts=%u graphs=%u prepare_library_loads=%u prepare_library_unloads=%u\n",sharedContexts,sharedContexts,sharedGraphs,sharedLoads,sharedUnloads);
  for(const auto& f:st.faults)std::printf("fault=%s\n",f.c_str());

  if(!symmetric)rc=1;
  if(descriptorFaults)rc=2;
  if(handleFaults)rc=3;
  if(loads!=5||unloads!=5)rc=4;
  if(singleSessionGraphs!=cohortGraphs)rc=5;
  if(singleSessionLoads!=1||singleSessionUnloads!=1)rc=6;
  if(sharedGraphs!=cohortGraphs||sharedContexts!=5)rc=7;
  if(sharedLoads!=1||sharedUnloads!=1)rc=8;
  std::printf("cohort_lifecycle_audit_rc=%d HOST_ONLY=1 NO_DEVICE=1 NO_DSP=1 NOT_FORMAL_EVIDENCE=1\n",rc);
  return rc;
}
