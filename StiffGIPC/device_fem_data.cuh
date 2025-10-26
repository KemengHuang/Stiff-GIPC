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
#include "gpu_eigen_libs.cuh"
#include <cstdint>
#include <body_boundary_type.h>
#include "Eigen/Eigen"
class device_TetraData
{
  public:
    double3* vertexes        = nullptr;
    double3* o_vertexes      = nullptr;
    double3* rest_vertexes   = nullptr;
    double3* targetVert      = nullptr;
    double3* temp_double3Mem = nullptr;
    double3* velocities      = nullptr;
    double3* xTilta          = nullptr;
    double3* fb              = nullptr;
    double3* totalForce      = nullptr;
    uint4*   tetrahedras     = nullptr;
    uint3*   triangles       = nullptr;

    uint2* tri_edges           = nullptr;
    uint2* tri_edge_adj_vertex = nullptr;

    uint32_t* targetIndex     = nullptr;
    uint4*    tempTetrahedras = nullptr;
    double*   volum           = nullptr;
    double*   area            = nullptr;

    double* lengthRate = nullptr;
    double* volumeRate = nullptr;
    //double* tempV;
    double*   masses           = nullptr;
    double*   tempDouble       = nullptr;
    uint64_t* MChash           = nullptr;
    uint32_t* sortIndex        = nullptr;
    uint32_t* sortMapVertIndex = nullptr;
    //uint32_t* sortTetIndex;
    __GEIGEN__::Matrix3x3d* DmInverses       = nullptr;
    __GEIGEN__::Matrix2x2d* triDmInverses    = nullptr;
    __GEIGEN__::Matrix3x3d* Constraints      = nullptr;
    int*                    BoundaryType     = nullptr;
    int*                    tempBoundaryType = nullptr;


    Eigen::Matrix<double, 3, 2>* svd3x2F;
    Eigen::Matrix3d*             svd3x2U;
    Eigen::Matrix2d*             svd3x2V;
    Eigen::Vector2d*             svd3x2S;

    //__GEIGEN__::Matrix3x3d* tempDmInverses;
    __GEIGEN__::Matrix3x3d* tempMat3x3 = nullptr;

    double3*          shape_grads              = nullptr;
    BodyBoundaryType* body_id_to_boundary_type = nullptr;
    int*              point_id_to_body_id      = nullptr;
    int*              tet_id_to_body_id        = nullptr;

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
