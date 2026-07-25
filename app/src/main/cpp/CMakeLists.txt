cmake_minimum_required(VERSION 3.22.1)

project(PhoneLMNative LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PHONELM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(MNN_ROOT "${PHONELM_ROOT}/third_party/MNN")

option(PHONELM_ENABLE_QNN "Compile the SDK-specific QAIRT/QNN adapter" OFF)
set(QAIRT_SDK_ROOT "" CACHE PATH "Root of a locally installed QAIRT SDK")
set(PHONELM_EXPECTED_QAIRT_BUILD_ID "" CACHE STRING "Expected QAIRT build id")
if(QAIRT_SDK_ROOT STREQUAL "" AND DEFINED ENV{QAIRT_SDK_ROOT})
    file(TO_CMAKE_PATH "$ENV{QAIRT_SDK_ROOT}" QAIRT_SDK_ROOT)
endif()

set(PHONELM_QAIRT_SDK_DETECTED 0)
set(PHONELM_QAIRT_SDK_ROOT_TEXT "NOT_SET")
if(NOT QAIRT_SDK_ROOT STREQUAL "" AND EXISTS "${QAIRT_SDK_ROOT}")
    # Header-name discovery is only an installation check. It is deliberately
    # not treated as proof that an Android runtime, HTP libraries, or samples
    # are usable. scripts/check_qairt.* performs the wider inventory.
    file(GLOB_RECURSE PHONELM_QNN_INTERFACE_HEADERS
        LIST_DIRECTORIES FALSE "${QAIRT_SDK_ROOT}/QnnInterface.h")
    file(GLOB_RECURSE PHONELM_QNN_TYPES_HEADERS
        LIST_DIRECTORIES FALSE "${QAIRT_SDK_ROOT}/QnnTypes.h")
    if(PHONELM_QNN_INTERFACE_HEADERS AND PHONELM_QNN_TYPES_HEADERS)
        set(PHONELM_QAIRT_SDK_DETECTED 1)
        set(PHONELM_QAIRT_SDK_ROOT_TEXT "${QAIRT_SDK_ROOT}")
    endif()
endif()

if(PHONELM_ENABLE_QNN AND NOT PHONELM_QAIRT_SDK_DETECTED)
    message(FATAL_ERROR
        "PHONELM_ENABLE_QNN=ON but required QnnInterface.h and QnnTypes.h were not found below "
        "QAIRT_SDK_ROOT='${QAIRT_SDK_ROOT}'. The MNN-only build works with QNN OFF.")
endif()

if(PHONELM_ENABLE_QNN)
    file(READ "${QAIRT_SDK_ROOT}/include/QNN/QnnSdkBuildId.h" _qairt_build_header)
    string(REGEX MATCH "QNN_SDK_BUILD_ID[ \t]+\"v([^\"]+)\"" _ "${_qairt_build_header}")
    if(NOT PHONELM_EXPECTED_QAIRT_BUILD_ID STREQUAL "" AND
       NOT CMAKE_MATCH_1 STREQUAL PHONELM_EXPECTED_QAIRT_BUILD_ID)
        message(FATAL_ERROR "Mixed QAIRT distribution: expected=${PHONELM_EXPECTED_QAIRT_BUILD_ID}, headers=${CMAKE_MATCH_1}")
    endif()
    message(STATUS "PhoneLM QAIRT SDK: ${QAIRT_SDK_ROOT} (${CMAKE_MATCH_1})")
endif()
# QNN OFF must never compile or link a QAIRT-bound source. Conversely, QNN ON
# must never fall back to the stub: it selects the audited adapter only. The
# adapter is intentionally absent until a locally installed SDK can be read.
set(PHONELM_QNN_RUNTIME_SOURCE qnn/qnn_runtime_stub.cpp)
set(PHONELM_QNN_OPTIONAL_SOURCES "")
if(PHONELM_ENABLE_QNN)
    set(PHONELM_QNN_RUNTIME_QAIRT_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/qnn/qnn_runtime_qairt.cpp")
    if(NOT EXISTS "${PHONELM_QNN_RUNTIME_QAIRT_SOURCE}")
        message(FATAL_ERROR
            "BLOCKED_BY_QAIRT_ADAPTER_NOT_IMPLEMENTED: QAIRT headers were found, but "
            "qnn_runtime_qairt.cpp does not exist. Audit this installed SDK's headers, "
            "official Android samples, exact libraries, and DSP deployment procedure "
            "before adding the adapter. The stub is never linked into a QNN-enabled build.")
    endif()
    set(PHONELM_QNN_RUNTIME_SOURCE qnn/qnn_runtime_qairt.cpp)
    list(APPEND PHONELM_QNN_OPTIONAL_SOURCES
        qnn/qnn_host_quantization.cpp
        qnn/qnn_hybrid_training.cpp
    )
endif()

if(NOT EXISTS "${MNN_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "Pinned MNN source is missing at ${MNN_ROOT}. "
        "Run scripts/fetch_mnn.ps1 from the repository root first.")
endif()

# MNN 3.5.0, commit c35f14f3ab5cb65094863b9a0e888370b027a670.
# A monolithic shared library deliberately avoids static-link dead stripping of
# gradient registrations and GPU RuntimeCreator initializers.
set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
set(MNN_BUILD_TRAIN ON CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_VULKAN ON CACHE BOOL "" FORCE)
set(MNN_VULKAN_IMAGE ON CACHE BOOL "" FORCE)
set(MNN_ARM82 ON CACHE BOOL "" FORCE)

# SME2 and KleidiAI are unrelated to this FP32 GPU-training probe. Keeping them
# off also avoids making the CPU baseline depend on optional new CPU features.
set(MNN_SME2 OFF CACHE BOOL "" FORCE)
set(MNN_KLEIDIAI OFF CACHE BOOL "" FORCE)

set(MNN_USE_SYSTEM_LIB OFF CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT ON CACHE BOOL "" FORCE)
set(MNN_BUILD_MINI OFF CACHE BOOL "" FORCE)
set(MNN_SKIPBUILD_GEOMETRY OFF CACHE BOOL "" FORCE)
set(MNN_REDUCE_SIZE OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_OPENCV OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_LLM OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DIFFUSION OFF CACHE BOOL "" FORCE)
set(MNN_JNI OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE OFF CACHE BOOL "" FORCE)
set(MNN_GPU_TIME_PROFILE OFF CACHE BOOL "" FORCE)
set(MNN_GPU_TRACE OFF CACHE BOOL "" FORCE)

# EXCLUDE_FROM_ALL prevents MNN's desktop-oriented train demo executables from
# being built for Android. The MNN target still pulls the official MNNTrain
# object sources into libMNN.so because MNN_BUILD_TRAIN is ON.
add_subdirectory("${MNN_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/mnn" EXCLUDE_FROM_ALL)

add_library(phonelm_native SHARED
    native_bridge.cpp
    benchmark_runner.cpp
    cpu_reference_training.cpp
    tiny_language_model_cpu.cpp
    mnn_training_test.cpp
    training_engine.cpp
    qnn/qnn_backend_info.cpp
    qnn/qnn_linear_training.cpp
    qnn/qnn_mlp_training.cpp
    qnn/qnn_transformer.cpp
    qnn/qnn_transformer_training.cpp
    qnn/qnn_training_benchmark.cpp
    ${PHONELM_QNN_RUNTIME_SOURCE}
    ${PHONELM_QNN_OPTIONAL_SOURCES}
    qnn/qnn_tensor.cpp
)

target_include_directories(phonelm_native PRIVATE
    "${MNN_ROOT}/include"
    "${MNN_ROOT}/source"
    "${MNN_ROOT}/express"
    "${MNN_ROOT}/schema/current"
    "${MNN_ROOT}/3rd_party/flatbuffers/include"
    "${MNN_ROOT}/tools/train/source/grad"
    "${MNN_ROOT}/tools/train/source/optimizer"
)
if(PHONELM_ENABLE_QNN)
    target_include_directories(phonelm_native PRIVATE "${QAIRT_SDK_ROOT}/include/QNN")
    target_compile_options(phonelm_native PRIVATE
        "-ffile-prefix-map=${QAIRT_SDK_ROOT}=QAIRT_SDK"
        "-fdebug-prefix-map=${QAIRT_SDK_ROOT}=QAIRT_SDK")
endif()

target_compile_options(phonelm_native PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_definitions(phonelm_native PRIVATE
    PHONELM_ENABLE_QNN=$<BOOL:${PHONELM_ENABLE_QNN}>
    PHONELM_QAIRT_SDK_DETECTED=${PHONELM_QAIRT_SDK_DETECTED}
    PHONELM_QAIRT_SDK_ROOT_TEXT="LOCAL_SDK_SELECTED_AT_BUILD_TIME"
)

target_link_libraries(phonelm_native PRIVATE
    MNN
    android
    log
)
