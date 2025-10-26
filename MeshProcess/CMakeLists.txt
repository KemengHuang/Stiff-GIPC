find_package(Eigen3 REQUIRED)
find_package(metis REQUIRED)
add_library(mesh_process STATIC)
#add_library(mesh_process SHARED)

file(GLOB_RECURSE MESH_PROCESS_SOURCE CONFIGURE_DEPENDS 
    "${PROJECT_SOURCE_DIR}/MeshProcess/metis_partition/*.cpp" "${PROJECT_SOURCE_DIR}/MeshProcess/metis_partition/*.h")
target_sources(mesh_process PRIVATE ${MESH_PROCESS_SOURCE})

#add_executable(mesh_process ${MESH_PROCESS_SOURCE})
target_include_directories(mesh_process PUBLIC "${PROJECT_SOURCE_DIR}/MeshProcess/metis_partition")



#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MESH_PROCESS_SOURCE})


#source_group(TREE "${CMAKE_CURRENT_LIST_DIR}" PREFIX "mesh_process" FILES ${MESH_PROCESS_SOURCE})

target_link_libraries(mesh_process PRIVATE metis Eigen3::Eigen)
target_compile_definitions(mesh_process PRIVATE ASSETS_DIR="${PROJECT_SOURCE_DIR}/Assets/")
target_compile_definitions(mesh_process PRIVATE OUTPUT_DIR="${PROJECT_SOURCE_DIR}/Assets/sorted_mesh/")
target_compile_features(mesh_process PRIVATE cxx_std_17)

source_group(TREE "${CMAKE_CURRENT_LIST_DIR}" FILES ${MESH_PROCESS_SOURCE})
set_target_properties(mesh_process PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mesh_process PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



