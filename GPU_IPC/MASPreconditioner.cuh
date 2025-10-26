//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <muda/ext/linear_system/bcoo_matrix_view.h>
class BHessian
{
  public:
    uint32_t*                 D1Index;  //pIndex, DpeIndex, DptIndex;
    uint3*                    D3Index;
    uint4*                    D4Index;
    uint2*                    D2Index;
    __GEIGEN__::Matrix12x12d* H12x12;
    __GEIGEN__::Matrix3x3d*   H3x3;
    __GEIGEN__::Matrix6x6d*   H6x6;
    __GEIGEN__::Matrix9x9d*   H9x9;

    uint32_t DNum[4];

  public:
    BHessian() {}
    ~BHessian() {};
    void updateDNum(const int&      tri_Num,
                    const int&      tet_number,
                    const uint32_t* cpNums,
                    const uint32_t* last_cpNums,
                    const int&      tri_edge_number);
    void MALLOC_DEVICE_MEM_O(const int& tet_number,
                             const int& surfvert_number,
                             const int& surface_number,
                             const int& edge_number,
                             const int& triangle_num,
                             const int& tri_Edge_number);
    void FREE_DEVICE_MEM();
    //void init(const int& edgeNum, const int& faceNum, const int& vertNum);
};

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
    Precision_T3*              d_multiLevelR;
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

    void setPreconditioner(const BHessian& BH, const double* masses, int cpNum);  // init the preconditioner for PCG
    void setPreconditioner_bcoo(muda::CBCOOMatrixView<double, 3> hessian,
                                muda::CBufferView<int>           indices,
                                int                              offset,
                                int                              cpNum);
    void PrepareHessian(const BHessian& BH, const double* masses);  // called in setPreconditioner
    void PrepareHessian_bcoo(muda::CBCOOMatrixView<double, 3> hessian,
                             int                              offset,
                             muda::CBufferView<int>           indices);

    void preconditioning(const double3* R, double3* Z);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void CollectFinalZ(double3* Z);           // called in preconditioning

    void FreeMAS();
};