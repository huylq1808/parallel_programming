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

# Include thư mục Header chung (Public Interface)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ==============================================================================
# 2. SOURCE COLLECTION (TỰ ĐỘNG GOM FILE)
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

# optimize v2 
file(GLOB_RECURSE SRC_GPU_OPT2 
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2/*.cu" 
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/GPU_opt_v2/*.cpp"
)

# ==============================================================================
# 3. DEFINE LIBRARIES (TẠO CÁC PHIÊN BẢN THƯ VIỆN)
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

# --- C. Library: GPU Optimized V1 (Template cho tương lai) ---
# if(EXISTS "${CMAKE_SOURCE_DIR}/src_gpu_opt1")
#     file(GLOB_RECURSE SRC_GPU_OPT1 "src_gpu_opt1/*.cu")
#     add_library(dl_gpu_opt1 STATIC ${SRC_GPU_OPT1})
#     target_link_libraries(dl_gpu_opt1 PRIVATE CUDA::cudart)
# endif()

# ==============================================================================
# 4. EXECUTABLES & TESTS
# ==============================================================================

# Hàm tiện ích để tạo executable cho nhiều backend cùng lúc
message(STATUS "Configuring Apps...")

# --- APP 1: TRAIN CPU (train_net_cpu) ---
# Chỉ dùng code CPU, link với thư viện dl_cpu
if(EXISTS "${CMAKE_SOURCE_DIR}/apps/train_cpu.cpp")
    add_executable(train_net_cpu apps/train_cpu.cpp)
    target_link_libraries(train_net_cpu PRIVATE dl_cpu)
    message(STATUS "   + Added Target: train_net_cpu")
endif()

# --- APP 2: TRAIN GPU (train_net_gpu) ---
# Dùng code GPU riêng, link với thư viện dl_gpu_naive
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_naive.cpp")
    add_executable(train_net_gpu_naive apps/train_gpu_naive.cpp)
    target_link_libraries(train_net_gpu_naive PRIVATE dl_gpu_naive)
    message(STATUS "   + Added Target: train_net_gpu_naive")
endif()

# --- APP 3: TRAIN GPU Optimized V2 (train_net_gpu_opt2) ---
# Dùng code GPU riêng, link với thư viện dl_gpu_opt2
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/apps/train_gpu_naive.cpp")
    add_executable(train_net_gpu_opt2 apps/train_gpu_naive.cpp)
    target_link_libraries(train_net_gpu_opt2 PRIVATE dl_gpu_opt2)
    message(STATUS "   + Added Target: train_net_gpu_opt2")
endif()