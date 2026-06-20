//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <gipc/cuda/all.h>
#include "linear_system/linear_system/global_matrix.h"

class MASPreconditioner
{

    int totalNodes;
    int totalMapNodes;
    int levelnum;
    int collision_node_Offset;
    int totalNumberClusters;
    //int bankSize;
    int2  h_clevelSize;
    int4* _collisonPairs;

    int2*               d_levelSize;
    int*                d_coarseSpaceTables;
    int*                d_prefixOriginal;
    int*                d_prefixSumOriginal;
    int*                d_goingNext;
    int*                d_denseLevel;
    __GEIGEN__::itable* d_coarseTable;
    unsigned int*       d_fineConnectMask;
    unsigned int*       d_nextConnectMask;
    unsigned int*       d_nextPrefix;
    unsigned int*       d_nextPrefixSum;


    __GEIGEN__::MasMatrixT*    d_MatMas;
    __GEIGEN__::MasMatrixSymT* d_inverseMatMas;
    __GEIGEN__::MasMatrixSymf* d_precondMatMas;
    Eigen::Vector3f*              d_multiLevelR;
    Precision_T3*              d_multiLevelZ;

  public:
    int           neighborListSize;
    unsigned int* d_neighborList;
    unsigned int* d_neighborStart;
    unsigned int* d_neighborStartTemp;
    unsigned int* d_neighborNum;
    unsigned int* d_neighborListInit;
    unsigned int* d_neighborNumInit;
    int*          d_partId_map_real;
    int*          d_real_map_partId;

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