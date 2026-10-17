#include <QnnOpPackage.h>
#include <HTP/core/qhpi.h>
#include <cstring>
#include <cstdio>
struct QHPI_Tensor {};
extern "C" QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor*){return {};}
extern "C" QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor*){return QHPI_FLOAT32;}
extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor*){return nullptr;}
unsigned count=0;QHPI_OpInfo_v1* registered=nullptr;
extern "C" QHPI_StatusCode qhpi_register_ops_v1(uint32_t n,QHPI_OpInfo_v1* ops,const char* package){if(std::strcmp(package,"HexaSelectiveLn1"))return QHPI_ERROR_FATAL;count=n;registered=ops;return QHPI_SUCCESS;}
#include "selective_ln1_package.cpp"
int main(){QnnOpPackage_Interface_t api{};if(HexaSelectiveLn1InterfaceProvider(&api)!=QNN_SUCCESS||api.v1_4.init(nullptr)!=QNN_SUCCESS)return 1;const QnnOpPackage_Info_t* info=nullptr;if(api.v1_4.getInfo(&info)!=QNN_SUCCESS||info->numOperations!=4||!qhpi_init()||count!=4)return 2;for(unsigned i=0;i<4;++i){if(registered[i].kernels[0].function(nullptr,0,nullptr,0,nullptr)!=QHPI_ERROR_FATAL)return 3;Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.packageName="HexaSelectiveLn1";op.v1.typeName=s2package::names[i];op.v1.numOfInputs=4;op.v1.numOfOutputs=i%2?3:4;if(api.v1_4.validateOpConfig(op)!=QNN_SUCCESS)return 4;++op.v1.numOfInputs;if(api.v1_4.validateOpConfig(op)==QNN_SUCCESS)return 5;}if(api.v1_4.terminate()!=QNN_SUCCESS)return 6;std::puts("selective_host_prepare=PASS operations=4 cpu_arithmetic_rejected=4 invalid_signature_rejected=4");}
