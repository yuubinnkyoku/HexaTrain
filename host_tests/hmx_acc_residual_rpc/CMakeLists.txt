cmake_minimum_required(VERSION 3.14.3)
project(hexatrain_hmx_acc_residual C)

# Research-only target. Do not include in the Android app's production CMake.
# Requires a locally installed Qualcomm Hexagon SDK with V81 HMX support.
set(CMAKE_C_STANDARD 11)

if(HEXAGON_SDK_ROOT)
  include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
  include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${HEXAGON_SDK_ROOT}/incs
  ${HEXAGON_SDK_ROOT}/incs/stddef
  ${HEXAGON_SDK_ROOT}/rtos/qurt
  ${HEXAGON_SDK_ROOT}/utils/examples
)

if(OS_TYPE MATCHES "HLOS")
  add_library(hexatrain_hmx_acc_residual SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/hexatrain_hmx_acc_residual_stub.c
  )
  choose_dsprpc(${DSP_TYPE} dsprpc)
  link_custom_library(hexatrain_hmx_acc_residual ${dsprpc})
  build_idl(hexatrain_hmx_acc_residual.idl hexatrain_hmx_acc_residual)

  add_executable(hexatrain_hmx_acc_residual_test probe_host.c)
  set_common_compile_and_link_options(hexatrain_hmx_acc_residual_test)
  target_link_libraries(hexatrain_hmx_acc_residual_test
    hexatrain_hmx_acc_residual m)
  if(CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_options(hexatrain_hmx_acc_residual_test PUBLIC -ldl -llog)
  endif()
  add_dependencies(hexatrain_hmx_acc_residual_test
                   hexatrain_hmx_acc_residual)
  copy_binaries(hexatrain_hmx_acc_residual_test
                hexatrain_hmx_acc_residual)
else()
  if(NOT HEXAGON_ARCH STREQUAL "v81")
    message(FATAL_ERROR "This experimental HMX kernel is V81-only (DSP_ARCH=v81).")
  endif()
  add_library(hexatrain_hmx_acc_residual_skel SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/hexatrain_hmx_acc_residual_skel.c
    probe_dsp.c
  )
  build_idl(hexatrain_hmx_acc_residual.idl hexatrain_hmx_acc_residual_skel)
  target_compile_options(hexatrain_hmx_acc_residual_skel PRIVATE -mhmx)
  copy_binaries(hexatrain_hmx_acc_residual_skel)
endif()
