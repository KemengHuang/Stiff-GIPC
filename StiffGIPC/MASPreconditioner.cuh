//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"

class MASPreconditioner
{

    int totalNodes              = 0;
    int totalMapNodes           = 0;
    int levelnum                = 0;
    int collision_node_Offset   = 0;
    int totalNumberClusters     = 0;
    //int bankSize;
    int2  h_clevelSize          = {};
    int4* _collisonPairs        = nullptr;

    int2*               d_levelSize          = nullptr;
    int*                d_coarseSpaceTables  = nullptr;
    int*                d_prefixOriginal     = nullptr;
    int*                d_prefixSumOriginal  = nullptr;
    int*                d_goingNext          = nullptr;
    int*                d_denseLevel         = nullptr;
    __GEIGEN__::itable* d_coarseTable        = nullptr;
    unsigned int*       d_fineConnectMask    = nullptr;
    unsigned int*       d_nextConnectMask    = nullptr;
    unsigned int*       d_nextPrefix         = nullptr;
    unsigned int*       d_nextPrefixSum      = nullptr;


    __GEIGEN__::MasMatrixT*    d_MatMas       = nullptr;
    __GEIGEN__::MasMatrixSymT* d_inverseMatMas = nullptr;
    __GEIGEN__::MasMatrixSymf* d_precondMatMas = nullptr;
    Eigen::Vector3f*           d_multiLevelR   = nullptr;
    Precision_T3*              d_multiLevelZ   = nullptr;

  public:
    int           neighborListSize    = 0;
    unsigned int* d_neighborList      = nullptr;
    unsigned int* d_neighborStart     = nullptr;
    unsigned int* d_neighborStartTemp = nullptr;
    unsigned int* d_neighborNum       = nullptr;
    unsigned int* d_neighborListInit  = nullptr;
    unsigned int* d_neighborNumInit   = nullptr;
    int*          d_partId_map_real   = nullptr;
    int*          d_real_map_partId   = nullptr;

  public:
    void initPreconditioner_Neighbor(int   vertNum,
                                     int   mCollision_node_offset,
                                     int   totalNeighborNum,
                                     int4* m_collisonPairs,
                                     int   partMapSize);
    void computeNumLevels(int vertNum);  // called in initPreconditioner_Neighbor

    void initPreconditioner_Matrix();


    int  ReorderRealtime(int cpNum);
    void BuildConnectMaskL0();           // called in ReorderRealtime
    void PreparePrefixSumL0();           // called in ReorderRealtime
    void BuildLevel1();                  // called in ReorderRealtime
    void BuildConnectMaskLx(int level);  // called in ReorderRealtime
    void NextLevelCluster(int level);    // called in ReorderRealtime
    void PrefixSumLx(int level);         // called in ReorderRealtime
    void ComputeNextLevel(int level);    // called in ReorderRealtime
    void AggregationKernel();            // called in ReorderRealtime
    void BuildCollisionConnection(unsigned int* connectionMsk,
                                  int*          coarseTableSpace,
                                  int           level,
                                  int cpNum);  // called in ReorderRealtime

    void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                int*             row_ids,
                                int*             col_ids,
                                uint32_t*        indices,
                                int              offset,
                                int              triplet_num,
                                int              cpNum);
    void PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                             int*             row_ids,
                             int*             col_ids,
                             uint32_t*        indices,
                             int              offset,
                             int              triplet_number);

    void preconditioning(const double3* R, double3* Z);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void SchwarzLocalXSym_block3();                  // called in preconditioning
    void SchwarzLocalXSym_sym();           // called in preconditioning
    void CollectFinalZ(double3* Z);           // called in preconditioning

    void FreeMAS();
};