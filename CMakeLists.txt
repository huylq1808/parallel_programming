cmake_minimum_required(VERSION 3.18)
project(DeepLearningCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# --- THÊM ĐOẠN NÀY ĐỂ TĂNG TỐC ---
if(NOT MSVC)
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
endif()

# --- 1. CUDA DETECTION ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(USE_CUDA)
    message(STATUS ">> CUDA FOUND: GPU Acceleration Enabled")
else()
    message(STATUS ">> CUDA NOT FOUND: Compiling for CPU only")
endif()

# --- 2. INCLUDES ---
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- 3. SOURCES ---
# Core Sources (Dùng chung cho cả App và Test)
set(CORE_SRC 
    src/core/Config.cpp
    src/core/Allocator.cpp 
    src/core/Storage.cpp 
    src/core/Tensor.cpp
)

# CPU Ops (Dùng chung)
set(BACKEND_CPU_SRC src/backend/cpu/ops_cpu.cpp)

# Layers (Cho App chính)
set(LAYER_SRC 
    # src/layers/Dense.cpp 
    src/layers/Conv2D.cpp
    src/layers/ReLU.cpp   
    src/layers/Sigmoid.cpp
    src/layers/MaxPool2D.cpp   
    src/layers/Upsample.cpp
)

# loss function 
set(LOSS_SRC src/loss/MSELoss.cpp)

# optimizer
set(OPTIM_SRC src/optim/SGD.cpp)

# GPU Ops (Nếu có)
if(CMAKE_CUDA_COMPILER)
    set(BACKEND_GPU_SRC 
        src/backend/cuda/ops_bridge.cu
        src/backend/cuda/kernels/naive/ElementWise.cu
        src/backend/cuda/kernels/naive/Conv2D.cu
        src/backend/cuda/kernels/naive/Activation.cu
        src/backend/cuda/kernels/naive/Pooling.cu
        src/backend/cuda/kernels/naive/Upsample.cu
        src/backend/cuda/kernels/naive/Loss_Optimizer.cu
    )
endif()

# =========================================================
# --- 4. EXECUTABLES (APP CHÍNH) ---
# =========================================================
add_executable(benchmark 
    tests/benchmark.cpp
    ${CORE_SRC} 
    ${BACKEND_CPU_SRC} 
    ${LAYER_SRC} 
    ${BACKEND_GPU_SRC}
)

# Link thư viện cho benchmark
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(benchmark PRIVATE CUDA::cudart)
endif()

# --- APP 1: CPU TRAINER ---
add_executable(train_cpu
    apps/train_cpu.cpp  # <--- File mới
    ${CORE_SRC} ${BACKEND_CPU_SRC} ${LAYER_SRC} 
    ${LOSS_SRC} ${OPTIM_SRC}
    ${BACKEND_GPU_SRC} # Vẫn link vào để tránh lỗi linker nếu Ops.h có tham chiếu chéo, nhưng runtime không dùng
)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(train_cpu PRIVATE CUDA::cudart)
endif()

# --- APP 2: GPU TRAINER ---
if(CMAKE_CUDA_COMPILER)
    add_executable(train_gpu
        apps/train_gpu.cpp  # <--- File mới
        ${CORE_SRC} ${BACKEND_CPU_SRC} ${LAYER_SRC} 
        ${LOSS_SRC} ${OPTIM_SRC}
        ${BACKEND_GPU_SRC}
    )
    target_link_libraries(train_gpu PRIVATE CUDA::cudart)
endif()

# 2. Inference App
add_executable(infer_net 
    apps/inference.cpp
    ${CORE_SRC} ${BACKEND_CPU_SRC} ${LAYER_SRC} 
    ${BACKEND_GPU_SRC}
)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(infer_net PRIVATE CUDA::cudart)
endif()

# 3. Visualization Export App
add_executable(visualize_export
    apps/visualize_export.cpp
    ${CORE_SRC} ${BACKEND_CPU_SRC} ${LAYER_SRC} 
    ${BACKEND_GPU_SRC}
)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(visualize_export PRIVATE CUDA::cudart)
endif()

# =========================================================
# --- 5. TESTS (KHU VỰC THÊM FILE TEST) ---
# =========================================================

# [TEST 1] Tensor CPU Test
# Đây là executable mới, chỉ test Tensor và CPU Ops
add_executable(tensor_cpu_test
    tests/Tensor_cpu_test.cpp
    ${CORE_SRC}
    ${BACKEND_CPU_SRC}
    ${BACKEND_GPU_SRC}   # <--- THÊM DÒNG NÀY (Chứa ops_bridge.cu)
)

# Link thư viện CUDA
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(tensor_cpu_test PRIVATE CUDA::cudart)
endif()

# [TEST 2] Layer CPU Test
# =========================================================
add_executable(layer_cpu_test
    tests/layer_cpu_test.cpp  # File test chính
    ${CORE_SRC}
    ${BACKEND_CPU_SRC}
    ${LAYER_SRC}
    ${BACKEND_GPU_SRC}
)

# Nếu có CUDA thì link thêm thư viện (dù test CPU nhưng project là hybrid)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(layer_cpu_test PRIVATE CUDA::cudart)
endif()

# [TEST 3] Loss & Optimizer
add_executable(loss_optim_test
    tests/loss_optim_test.cpp
    ${CORE_SRC}
    ${BACKEND_CPU_SRC}
    ${BACKEND_GPU_SRC}
    ${LOSS_SRC}
    ${OPTIM_SRC}
)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(loss_optim_test PRIVATE CUDA::cudart)
endif()

# [TEST 4] DataLoader
add_executable(dataloader_test
    tests/dataloader_test.cpp
    ${CORE_SRC}
    ${BACKEND_CPU_SRC}
    ${BACKEND_GPU_SRC}
)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(dataloader_test PRIVATE CUDA::cudart)
endif()

# [HƯỚNG DẪN THÊM TEST MỚI]
# Nếu sau này bạn muốn thêm file test mới, ví dụ: tests/Layer_test.cpp
# Hãy copy đoạn block dưới đây và đổi tên:
#
# add_executable(layer_test
#     tests/Layer_test.cpp
#     ${CORE_SRC}
#     ${BACKEND_CPU_SRC}
#     ${LAYER_SRC}          <-- Nếu test Layer thì cần thêm biến này
# )
# if(CMAKE_CUDA_COMPILER)
#     target_link_libraries(layer_test PRIVATE CUDA::cudart)
# endif()