// Reuse immutable standalone graph/math/evidence. Diagnostic graph config only.
#include "double_qk_initialize.h"
#include <HTP/QnnHtpGraph.h>
#ifndef BACKWARD_PRECISION_MODE
#define BACKWARD_PRECISION_MODE 1
#endif
inline decltype(api.graphCreate) originalGraphCreate=nullptr;
inline Qnn_ErrorHandle_t precisionCreate(Qnn_ContextHandle_t c,const char* name,
 const QnnGraph_Config_t** incoming,Qnn_GraphHandle_t* result) {
 if(incoming)return QNN_GRAPH_ERROR_INVALID_ARGUMENT;
 QnnHtpGraph_CustomConfig_t precision=QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
 precision.option=QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;precision.precision=QNN_PRECISION_FLOAT32;
 QnnHtpGraph_CustomConfig_t compensation=QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
 compensation.option=QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION;compensation.precisionCompensation=true;
 QnnGraph_Config_t a=QNN_GRAPH_CONFIG_INIT,b=QNN_GRAPH_CONFIG_INIT;
 a.option=b.option=QNN_GRAPH_CONFIG_OPTION_CUSTOM;a.customConfig=&precision;b.customConfig=&compensation;
 const QnnGraph_Config_t* configs[]={&a,BACKWARD_PRECISION_MODE==2?&b:nullptr,nullptr};
 std::printf("diagnostic_precision,mode=%d,production_changed=0\n",BACKWARD_PRECISION_MODE);
 return originalGraphCreate(c,name,BACKWARD_PRECISION_MODE?configs:nullptr,result);
}
inline void initializePrecision(){initializeDoubleQk();originalGraphCreate=api.graphCreate;api.graphCreate=precisionCreate;}
#define initializeDoubleQk initializePrecision
#include "saved_p_backward_probe.cpp"
