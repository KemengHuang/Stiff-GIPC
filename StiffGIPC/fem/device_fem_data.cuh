//
// device_fem_data.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#ifndef __DEVICE_FEM_MESHES_CUH__
#define __DEVICE_FEM_MESHES_CUH__

//#include <cuda_runtime.h>
#include <math/gpu_eigen_libs.cuh>
#include <cstdint>
#include <core/body_boundary_type.h>
#include <cuda_tools/cuda_buffer_view.h>
#include "Eigen/Eigen"
class device_TetraData
{
  public:
    cudatool::DeviceBuffer<double3> vertexes;
    cudatool::DeviceBuffer<double3> o_vertexes;
    cudatool::DeviceBuffer<double3> rest_vertexes;
    cudatool::DeviceBuffer<double3> targetVert;
    cudatool::DeviceBuffer<double3> temp_double3Mem;
    cudatool::DeviceBuffer<double3> velocities;
    cudatool::DeviceBuffer<double3> xTilta;
    cudatool::DeviceBuffer<double3> fb;
    cudatool::DeviceBuffer<double3> totalForce;
    cudatool::DeviceBuffer<uint4>   tetrahedras;
    cudatool::DeviceBuffer<uint3>   triangles;

    cudatool::DeviceBuffer<uint2> tri_edges;
    cudatool::DeviceBuffer<uint2> tri_edge_adj_vertex;

#ifdef USE_QUADRATIC_BENDING
    // Precomputed Q matrices for quadratic bending
    cudatool::DeviceBuffer<Eigen::Matrix4d> quad_bending_Q;
#endif

    cudatool::DeviceBuffer<uint32_t> targetIndex;
    cudatool::DeviceBuffer<uint4>    tempTetrahedras;
    cudatool::DeviceBuffer<double>   volum;
    cudatool::DeviceBuffer<double>   area;

    cudatool::DeviceBuffer<double> lengthRate;
    cudatool::DeviceBuffer<double> volumeRate;
    cudatool::DeviceBuffer<double> masses;
    cudatool::DeviceBuffer<int>    apply_gravity;
    cudatool::DeviceBuffer<double> tempDouble;

    cudatool::DeviceBuffer<__GEIGEN__::Matrix3x3d> DmInverses;
    cudatool::DeviceBuffer<__GEIGEN__::Matrix2x2d> triDmInverses;
    cudatool::DeviceBuffer<int>                    BoundaryType;

    cudatool::DeviceBuffer<double3>          shape_grads;
    cudatool::DeviceBuffer<BodyBoundaryType> body_id_to_boundary_type;
    cudatool::DeviceBuffer<int>              point_id_to_body_id;
    cudatool::DeviceBuffer<int>              tet_id_to_body_id;

    int                   m_soft_num   = 0;
    int                   m_vertex_num = 0;
    std::vector<uint32_t> host_target_indices;
    std::vector<double3>  host_target_vertices;
    std::function<double3(double3 vertex, int step_id, double ipc_dt)> update_soft_constraint_functor =
        nullptr;
    void update_soft_constraint_target_position(int step_id, double ipc_dt);


  public:
    device_TetraData() {}
    ~device_TetraData();
    void Malloc_DEVICE_MEM(const int& vertex_num,
                           const int& tetradedra_num,
                           const int& triangle_num,
                           const int& softNum,
                           const int& tri_edgeNum,
                           const int& bodyNum);
    void FREE_DEVICE_MEM();
};


#endif  // ! __DEVICE_FEM_MESHES_CUH__
