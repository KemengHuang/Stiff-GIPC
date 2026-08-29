//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <fem/device_fem_data.cuh>
#include <math/eigen_data.h>
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

class MASPreconditioner
{

    int totalNodes            = 0;
    int totalMapNodes         = 0;
    int levelnum              = 0;
    int collision_node_Offset = 0;
    int totalNumberClusters   = 0;
    //int bankSize;
    int2 h_clevelSize = {};

    cudatool::DeviceBuffer<int2>               d_levelSize;
    cudatool::DeviceBuffer<int>                d_coarseSpaceTables;
    cudatool::DeviceBuffer<int>                d_prefixOriginal;
    cudatool::DeviceBuffer<int>                d_prefixSumOriginal;
    cudatool::DeviceBuffer<int>                d_goingNext;
    cudatool::DeviceBuffer<int>                d_denseLevel;
    cudatool::DeviceBuffer<__GEIGEN__::itable> d_coarseTable;
    cudatool::DeviceBuffer<unsigned int>       d_fineConnectMask;
    cudatool::DeviceBuffer<unsigned int>       d_nextConnectMask;
    cudatool::DeviceBuffer<unsigned int>       d_nextPrefix;
    cudatool::DeviceBuffer<unsigned int>       d_nextPrefixSum;


    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixT>    d_MatMas;
    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymT> d_inverseMatMas;
    cudatool::DeviceBuffer<__GEIGEN__::MasMatrixSymf> d_precondMatMas;
    cudatool::DeviceBuffer<Eigen::Vector3f>           d_multiLevelR;
    cudatool::DeviceBuffer<Precision_T3>              d_multiLevelZ;

  public:
    int                                  neighborListSize = 0;
    cudatool::DeviceBuffer<unsigned int> d_neighborList;
    cudatool::DeviceBuffer<unsigned int> d_neighborStart;
    cudatool::DeviceBuffer<unsigned int> d_neighborStartTemp;
    cudatool::DeviceBuffer<unsigned int> d_neighborNum;
    cudatool::DeviceBuffer<unsigned int> d_neighborListInit;
    cudatool::DeviceBuffer<unsigned int> d_neighborNumInit;
    cudatool::DeviceBuffer<int>          d_partId_map_real;
    cudatool::DeviceBuffer<int>          d_real_map_partId;

  public:
    static std::size_t requiredGoingNextCapacity(std::size_t vertex_count,
                                                 std::size_t mapped_node_count,
                                                 std::size_t level_count)
    {
        const std::size_t base_count = std::max(vertex_count, mapped_node_count);
        if(base_count == 0 || level_count == 0)
            return 0;
        constexpr std::size_t bank_size = BANKSIZE;
        if(base_count
           > std::numeric_limits<std::size_t>::max() - (bank_size - 1))
            std::abort();
        const std::size_t padded_count =
            (base_count + bank_size - 1) / bank_size * bank_size;
        if(padded_count > std::numeric_limits<std::size_t>::max() / level_count)
            std::abort();
        return padded_count * level_count;
    }

    void initPreconditioner_Neighbor(int vertNum,
                                     int mCollision_node_offset,
                                     int totalNeighborNum,
                                     int partMapSize);
    void computeNumLevels(int vertNum);  // called in initPreconditioner_Neighbor

    void initPreconditioner_Matrix();


    int  ReorderRealtime(int cpNum, const int4* collisionPairs = nullptr);
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
                                  int           cpNum,
                                  const int4* collisionPairs);  // called in ReorderRealtime

    void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                int*             row_ids,
                                int*             col_ids,
                                uint32_t*        indices,
                                int              offset,
                                int              triplet_num,
                                int              cpNum,
                                const int4*      collisionPairs);
    void PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                             int*             row_ids,
                             int*             col_ids,
                             uint32_t*        indices,
                             int              offset,
                             int              triplet_number);

    void preconditioning(const double3* R, double3* Z);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void SchwarzLocalXSym_block3();           // called in preconditioning
    void SchwarzLocalXSym_sym();              // called in preconditioning
    void CollectFinalZ(double3* Z);           // called in preconditioning

    void FreeMAS();
};
