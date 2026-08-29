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
#include(CTest)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find packages
enable_language(CUDA)

# Platform detection: MSVC vs GNU/Clang
if(MSVC)
    # Keep only the Windows define; host options (/GR /EHsc) are added below via
    # target_compile_options to avoid CMake merging them into a preprocessor define.
    set(CMAKE_CUDA_FLAGS "-D_WINDOWS")
else()
    set(CMAKE_CUDA_FLAGS "")
endif()
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)

# add external libraries
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_executable(gipc)
target_compile_features(gipc PRIVATE cuda_std_17)
target_include_directories(gipc PRIVATE "StiffGIPC/")
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS 
    "StiffGIPC/*.cu" "StiffGIPC/*.cpp")
file(GLOB_RECURSE GIPC_HANDER CONFIGURE_DEPENDS 
    "StiffGIPC/*.h" "StiffGIPC/*.cuh" "StiffGIPC/*.hpp" "StiffGIPC/*.inl" "StiffGIPC/*.inc")
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HANDER})

# add assets and output directory, that we can easily access them in the code
target_compile_definitions(gipc PRIVATE GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(gipc PRIVATE GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")

target_compile_definitions(gipc PRIVATE 
USE_SNK1 
ADAPTIVE_KAPPA 
USE_FRICTION 
USE_QUADRATIC_BENDING
CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)


find_package(CUDAToolkit REQUIRED)

# link cuda libraries
#include cuda directories
message(STATUS "CUDA library includes: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(gipc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(gipc PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_link_libraries(gipc PRIVATE cusparse cublas cusolver cudart_static)
target_link_libraries(gipc PRIVATE metis_partition CUDA::cudart)
target_link_libraries(gipc PRIVATE GLUT::GLUT GLEW::GLEW OpenGL::GLU Eigen3::Eigen)
target_link_libraries(gipc PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
    # Some external C dependencies still pull in the static CRT; gipc uses the
    # dynamic CRT, so drop the static one to avoid LNK4098 conflicts.
    target_link_options(gipc PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

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
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/GR>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/EHsc>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:preprocessor>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=-fPIC>
    >
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HANDER})

add_subdirectory("MeshProcess")

# Ensure external C libraries use the same dynamic runtime as the executable
# to avoid LIBCMT/MSVCRT conflicts (MSVC only).
if(MSVC)
    foreach(_tgt GKlib metis metis_partition)
        if(TARGET ${_tgt})
            set_property(TARGET ${_tgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        endif()
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(dynamic_memory_tests tests/dynamic_memory_tests.cu)
    target_include_directories(dynamic_memory_tests PRIVATE "StiffGIPC/")
    target_include_directories(dynamic_memory_tests PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
        target_include_directories(dynamic_memory_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
    endif()
    target_compile_features(dynamic_memory_tests PRIVATE cxx_std_17 cuda_std_17)
    target_compile_definitions(dynamic_memory_tests PRIVATE
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
    target_compile_options(dynamic_memory_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
            --default-stream=per-thread
            $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/EHsc>
            $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:preprocessor>
        >)
    set_target_properties(dynamic_memory_tests PROPERTIES
        CUDA_EXTENSIONS ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(dynamic_memory_tests PRIVATE CUDA::cudart Eigen3::Eigen)
    add_test(NAME dynamic-memory-zero-capacity COMMAND dynamic_memory_tests)
endif()
