cmake_minimum_required(VERSION 3.18)
project(DeepLearningCPP LANGUAGES CXX CUDA)

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native) # Tối ưu cho GPU hiện tại

# Tăng tốc CPU build
if(NOT MSVC)
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
endif()

# (Public Interface)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ==============================================================================
# 2. SOURCE COLLECTION 
# ==============================================================================
# Gom toàn bộ source của CPU Backend
file(GLOB_RECURSE SRC_CPU
    "${CMAKE_SOURCE_DIR}/src/CPU/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/CPU/*.h"
)

# Gom toàn bộ source của GPU Naive Backend
file(GLOB_RECURSE SRC_GPU_NAIVE
    "${CMAKE_SOURCE_DIR}/src/GPU_naive/*.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_naive/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/GPU_naive/*.cpp"
)

# Use all GPU_naive sources as a base
set(SRC_GPU_OPT1 ${SRC_GPU_NAIVE})

file(GLOB_RECURSE SRC_GPU_OPT1
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v1/*.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v1/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v1/*.cpp"
)

set(SRC_GPU_OPT2 ${SRC_GPU_NAIVE})

list(REMOVE_ITEM SRC_GPU_OPT2

    "${CMAKE_SOURCE_DIR}/src/GPU_naive/layers/Conv2D.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_naive/core/Allocator.cpp"
)

list(APPEND SRC_GPU_OPT2

    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2/layers/Conv2D.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2/core/Allocator.cpp"
)

set(SRC_GPU_OPT_CUBLAS ${SRC_GPU_NAIVE})

list(REMOVE_ITEM SRC_GPU_OPT_CUBLAS

    "${CMAKE_SOURCE_DIR}/src/GPU_naive/layers/Conv2D.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_naive/core/Allocator.cpp"
)

# Add the optimized v1 versions instead
list(APPEND SRC_GPU_OPT_CUBLAS

    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2_cublas/layers/Conv2D.cu"
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2_cublas/core/Allocator.cpp"
)



# ==============================================================================
# 3. DEFINE LIBRARIES 
# ==============================================================================

# --- A. Library: CPU Version ---
add_library(dl_cpu STATIC ${SRC_CPU})
target_compile_definitions(dl_cpu PUBLIC USE_CPU_BACKEND)

# --- B. Library: GPU Naive Version ---
if(CMAKE_CUDA_COMPILER)
    add_library(dl_gpu_naive STATIC ${SRC_GPU_NAIVE})
    find_package(CUDAToolkit REQUIRED)

    # Link với thư viện CUDA runtime
    target_link_libraries(dl_gpu_naive PUBLIC CUDA::cudart CUDA::cublas)

    target_compile_definitions(dl_gpu_naive PUBLIC
        USE_GPU_NAIVE_BACKEND
        USE_CUDA
    )

    message(STATUS ">> ENABLED: GPU Naive Backend")
else()
    message(WARNING ">> CUDA NOT FOUND: GPU Backends will be disabled")
endif()

if(CMAKE_CUDA_COMPILER)
    add_library(dl_gpu_opt1 STATIC ${SRC_GPU_OPT1})
    find_package(CUDAToolkit REQUIRED)

    # Link với thư viện CUDA runtime
    target_link_libraries(dl_gpu_opt1 PUBLIC CUDA::cudart CUDA::cublas)

    target_compile_definitions(dl_gpu_opt1 PUBLIC
        USE_GPU_OPT1_BACKEND
        USE_CUDA
    )

    message(STATUS ">> ENABLED: GPU Optimized V1 Backend")
else()
    message(WARNING ">> CUDA NOT FOUND: GPU Backends will be disabled")
endif()

if(CMAKE_CUDA_COMPILER)
    add_library(dl_gpu_opt2_cublas STATIC ${SRC_GPU_OPT_CUBLAS})
    find_package(CUDAToolkit REQUIRED)

    # Link với thư viện CUDA runtime
    target_link_libraries(dl_gpu_opt2_cublas PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(dl_gpu_opt2_cublas PUBLIC
        USE_GPU_OPT1_CUBLAS_BACKEND
        USE_CUDA
    )

    message(STATUS ">> ENABLED: GPU Optimized V2 Backend")
else()
    message(WARNING ">> CUDA NOT FOUND: GPU Backends will be disabled")
endif()

# --- C. Library: GPU Optimized V2 ---
if(CMAKE_CUDA_COMPILER)
    add_library(dl_gpu_opt2 STATIC ${SRC_GPU_OPT2})
    find_package(CUDAToolkit REQUIRED)

    # Link với thư viện CUDA runtime
    target_link_libraries(dl_gpu_opt2 PUBLIC CUDA::cudart CUDA::cublas)

    target_compile_definitions(dl_gpu_opt2 PUBLIC
        USE_GPU_OPT2_BACKEND
        USE_CUDA
    )

    message(STATUS ">> ENABLED: GPU Optimized V2 Backend")
endif()

if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/visualize.cpp")
    add_executable(visualize apps/visualize.cpp)
    target_link_libraries(visualize PRIVATE dl_gpu_opt2) 
    message(STATUS "   + Added Target: visualize")
endif()

# ==============================================================================
# 4. EXECUTABLES & TESTS
# ==============================================================================

# Hàm tiện ích để tạo executable cho nhiều backend cùng lúc
message(STATUS "Configuring Apps...")

# --- APP 1: TRAIN CPU (train_net_cpu) ---
if(EXISTS "${CMAKE_SOURCE_DIR}/apps/train_cpu.cpp")
    add_executable(train_net_cpu apps/train_cpu.cpp)
    target_link_libraries(train_net_cpu PRIVATE dl_cpu)
    message(STATUS "   + Added Target: train_net_cpu")
endif()

# --- APP 2: TRAIN GPU (train_net_gpu) ---
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_naive.cpp")
    add_executable(train_net_gpu_naive apps/train_gpu_naive.cpp)
    target_link_libraries(train_net_gpu_naive PRIVATE dl_gpu_naive)
    message(STATUS "   + Added Target: train_net_gpu_naive")
endif()

if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_opt_v2.cpp")
    add_executable(train_net_gpu_opt2 apps/train_gpu_opt_v2.cpp)
    target_link_libraries(train_net_gpu_opt2 PRIVATE dl_gpu_opt2)
    message(STATUS "   + Added Target: train_net_gpu_opt2")
endif()

if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_opt_v2.cpp")
    add_executable(train_net_gpu_opt2_cublas apps/train_gpu_opt_v2.cpp)
    target_link_libraries(train_net_gpu_opt2_cublas PRIVATE dl_gpu_opt2_cublas)
    message(STATUS "   + Added Target: train_net_gpu_opt2_cublas")
endif()

if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_opt_v1.cpp")
    add_executable(train_net_gpu_opt1 apps/train_gpu_opt_v1.cpp)
    target_link_libraries(train_net_gpu_opt1 PRIVATE dl_gpu_opt1)
    message(STATUS "   + Added Target: train_net_gpu_opt1")
endif()

# --- APP: Feature Extraction ---
# if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/extract_features.cpp")
#     add_executable(extract_features apps/extract_features.cpp)
#     target_link_libraries(extract_features PRIVATE dl_gpu_opt1)
#     message(STATUS "   + Added Target: extract_features")
# endif()

if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/extract_features.cpp")
    add_executable(extract_features apps/extract_features.cpp)
    target_link_libraries(extract_features PRIVATE dl_gpu_opt2_cublas)
    message(STATUS "   + Added Target: extract_features")
endif()