#pragma once
#include "probe_support.h"
#ifndef ONE_GRAPH_HOST_AUDIT
inline void initializeMixed(){
 void* lib=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_GLOBAL);if(!lib){std::puts(dlerror());std::exit(2);}const QnnInterface_t** ps=nullptr;uint32_t n=0;auto providers=(Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(lib,"QnnInterface_getProviders");if(!providers)std::exit(2);check("getProviders",providers(&ps,&n));for(unsigned i=0;i<n;++i)if(ps[i]->apiVersion.coreApiVersion.major==2&&ps[i]->apiVersion.coreApiVersion.minor==37)api=ps[i]->QNN_INTERFACE_VER_NAME;if(!api.backendCreate)std::exit(2);Qnn_LogHandle_t logger{};check("logCreate",api.logCreate(log,QNN_LOG_LEVEL_INFO,&logger));check("backendCreate",api.backendCreate(logger,nullptr,&backend));
 check("registerOldHost",api.backendRegisterOpPackage(backend,"libHexaFp32Host.so","HexaFp32InterfaceProvider","CPU"));check("registerOldDSP",api.backendRegisterOpPackage(backend,"libHexaFp32Dsp.so","HexaFp32InterfaceProvider","HTP"));
 check("registerS2Host",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Host.so","HexaSelectiveLn1InterfaceProvider","CPU"));check("registerS2DSP",api.backendRegisterOpPackage(backend,"libHexaSelectiveLn1Dsp.so","HexaSelectiveLn1InterfaceProvider","HTP"));
 check("registerMixedHost",api.backendRegisterOpPackage(backend,"libHexaMixedQkvHost.so","HexaMixedQkvInterfaceProvider","CPU"));check("registerMixedDSP",api.backendRegisterOpPackage(backend,"libHexaMixedQkvDsp.so","HexaMixedQkvInterfaceProvider","HTP"));
 check("deviceCreate",api.deviceCreate(logger,nullptr,&device));check("contextCreate",api.contextCreate(backend,device,nullptr,&context));
}
#endif
