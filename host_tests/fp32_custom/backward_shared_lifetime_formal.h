#pragma once
// Diagnostic ownership only. Graph construction and arithmetic remain frozen.
#include "probe_support.h"
#include <stdexcept>
#include <set>
namespace formalsharedlife {
inline void result(const char* name,Qnn_ErrorHandle_t rc){
 std::printf("api,%s,%llu\n",name,(unsigned long long)QNN_GET_ERROR_CODE(rc));std::fflush(stdout);
 if(rc!=QNN_SUCCESS)throw std::runtime_error(name);
}
struct Owner {
 Qnn_LogHandle_t logger{};bool started=false,ended=false,poisoned=false,contextAttempted=false,deviceAttempted=false,backendAttempted=false;
 unsigned roles=0;
 Owner()=default;Owner(const Owner&)=delete;Owner& operator=(const Owner&)=delete;
 void start(){
  if(started||ended)throw std::runtime_error("repeated process initialization");started=true;
  result("logCreate",api.logCreate(log,QNN_LOG_LEVEL_INFO,&logger));
  Qnn_BackendHandle_t b{};result("backendCreate",api.backendCreate(logger,nullptr,&b));backend=b;
  const char* libs[]={"Fp32","SelectiveLn1","MixedQkv","DoubleQk"};
  for(auto lib:libs)for(auto target:{"Host","Dsp"}){
   std::string path="libHexa"+std::string(lib)+target+".so",provider="Hexa"+std::string(lib)+"InterfaceProvider";
   result("registerForwardPackage",api.backendRegisterOpPackage(backend,path.c_str(),provider.c_str(),std::strcmp(target,"Host")==0?"CPU":"HTP"));
  }
  Qnn_DeviceHandle_t d{};result("deviceCreate",api.deviceCreate(logger,nullptr,&d));device=d;
 }
 void beginRole(){
  if(!started||ended||poisoned||context||roles>=5)throw std::runtime_error("illegal role begin");
  contextAttempted=false;
  Qnn_ContextHandle_t c{};result("contextCreate",api.contextCreate(backend,device,nullptr,&c));context=c;
  if(!roles){
   result("registerRepairHost",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairHost.so","HexaBackwardRepairInterfaceProvider","CPU"));
   result("registerRepairDSP",api.backendRegisterOpPackage(backend,"libHexaBackwardRepairDsp.so","HexaBackwardRepairInterfaceProvider","HTP"));
  }else{std::puts("lifetime,registered_backend_packages_reused=1");}
  ++roles;
 }
 void endRole(){
  if(!context||contextAttempted||poisoned)throw std::runtime_error("repeated or invalid context cleanup");
  contextAttempted=true;
  auto rc=api.contextFree(context,nullptr);
  if(rc!=QNN_SUCCESS)poisoned=true;
  result("contextFree",rc);context=nullptr;
 }
 // A failed free has ambiguous ownership. Do not retry it or free its parents.
 bool cleanup() noexcept {
  if(ended)return false;ended=true;
  try {
   if(poisoned)return false;
   if(context){if(contextAttempted)return false;endRole();}
   if(device){if(deviceAttempted)return false;deviceAttempted=true;auto rc=api.deviceFree(device);if(rc!=QNN_SUCCESS)poisoned=true;result("deviceFree",rc);device=nullptr;}
   if(backend){if(backendAttempted)return false;backendAttempted=true;auto rc=api.backendFree(backend);if(rc!=QNN_SUCCESS)poisoned=true;result("backendFree",rc);backend=nullptr;}
   // Intentionally retain the original Qnn_LogHandle_t leak for this comparison.
   // Backend does not own the logger; it stays valid through backendFree.
   std::puts("lifetime,logger_repair_excluded=1");return true;
  }catch(...){poisoned=true;return false;}
 }
 ~Owner(){if(!ended)cleanup();}
};
inline Owner* owner=nullptr;
inline void emergencyCleanup(){if(owner&&!owner->ended)owner->cleanup();}
}
