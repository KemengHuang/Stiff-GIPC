cmake_minimum_required(VERSION 3.18)    
# support relative path, cuda_std_17, archive manipulation

# project meta
# ============

set(CMAKE_BUILD_TYPE "Release")
if(UNIX)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
    endif()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

project(StiffGIPC LANGUAGES CXX CUDA)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find packages
enable_language(CUDA)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)

# add external libraries
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
#add_subdirectory("External/muda")


add_executable(gipc)
target_compile_features(gipc PRIVATE cuda_std_17)
target_include_directories(gipc PRIVATE "StiffGIPC/")
target_include_directories(gipc PRIVATE "StiffGIPC/muda/src/")
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS 
    "StiffGIPC/*.cu" "StiffGIPC/*.cpp")
file(GLOB_RECURSE GIPC_HANDER CONFIGURE_DEPENDS 
    "StiffGIPC/*.h" "StiffGIPC/*.cuh" "StiffGIPC/*.hpp" "StiffGIPC/*.inl" "StiffGIPC/*.inc")
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HANDER})

# add assets and output directory, that we can easily access them in the code
target_compile_definitions(gipc PRIVATE GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(gipc PRIVATE GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")

target_compile_definitions(gipc PRIVATE USE_SNK1 ADAPTIVE_KAPPA USE_FRICTION)


find_package(CUDAToolkit REQUIRED)

# link cuda libraries
#include cuda directories
message(STATUS "CUDA library includes: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(gipc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(gipc PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_link_libraries(gipc PRIVATE cusparse cublas cusolver cudart_static)
target_link_libraries(gipc PRIVATE mesh_process CUDA::cudart)
target_link_libraries(gipc PRIVATE GLUT::GLUT GLEW::GLEW OpenGL::GLU Eigen3::Eigen TBB::tbb)
target_link_libraries(gipc PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(gipc PRIVATE cxx_std_17)

set_target_properties(gipc PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES native)

target_compile_options(gipc
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: 
        ${CMAKE_CUDA_FLAGS} 
        --use_fast_math 
        -lineinfo 
        --ptxas-options=-allow-expensive-optimizations=true 
        -Xcudafe 
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: 
        --extended-lambda 
        --expt-relaxed-constexpr 
        --default-stream=per-thread
    >
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HANDER})

add_subdirectory("MeshProcess")
