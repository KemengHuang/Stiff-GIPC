//
// MASPreconditioner.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "MASPreconditioner.cuh"
#include "cuda_tools.h"
#include "device_launch_parameters.h"
#include <muda/launch/launch.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cooperative_groups.h"
using namespace cooperative_groups;
//#include "Eigen/Eigen"
using namespace std;
#define SYME
#define GROUP


__global__ void _buildCML0(const unsigned int* _neighborStart,
                           unsigned int*       _neighborNum,
                           unsigned int*       _neighborList,
                           unsigned int*       _fineConnectedMsk,
                           int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    int          numNeighbor = _neighborNum[idx];
    unsigned int connectMsk  = (1U << laneId);
    int          nk          = 0;
    int          startId     = _neighborStart[idx];
    for(int i = 0; i < numNeighbor; i++)
    {
        int vIdConnected     = _neighborList[startId + i];
        int warpIdxConnected = vIdConnected / BANKSIZE;
        if(warpId == warpIdxConnected)
        {
            unsigned int laneIdxConnected = vIdConnected % BANKSIZE;
            connectMsk |= (1U << laneIdxConnected);
        }
        else
        {
            _neighborList[startId + nk] = vIdConnected;
            nk++;
        }
    }
    _neighborNum[idx]      = nk;
    _fineConnectedMsk[idx] = connectMsk;
}

__global__ void _buildCML0_new(const unsigned int* _neighborStart,
                               unsigned int*       _neighborNum,
                               unsigned int*       _neighborList,
                               unsigned int*       _fineConnectedMsk,
                               int*                _partId_map_real,
                               int*                _real_map_partId,
                               int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int warpId = tdx / BANKSIZE;
    int laneId = tdx % BANKSIZE;
    int idx    = _partId_map_real[tdx];
    if(idx >= 0)
    {

        int          numNeighbor = _neighborNum[idx];
        unsigned int connectMsk  = (1U << laneId);
        int          nk          = 0;
        int          startId     = _neighborStart[idx];
        for(int i = 0; i < numNeighbor; i++)
        {
            int vIdConnected = _neighborList[startId + i];
            //vIdConnected         = _real_map_partId[vIdConnected];
            int warpIdxConnected = _real_map_partId[vIdConnected] / BANKSIZE;
            if(warpId == warpIdxConnected)
            {
                unsigned int laneIdxConnected = _real_map_partId[vIdConnected] % BANKSIZE;
                connectMsk |= (1U << laneIdxConnected);
            }
            else
            {
                _neighborList[startId + nk] = vIdConnected;
                nk++;
            }
        }
        _neighborNum[idx]      = nk;
        _fineConnectedMsk[idx] = connectMsk;
    }
}


__device__ unsigned int _LanemaskLt(int laneIdx)
{
    return (1U << laneIdx) - 1;
}

__global__ void _preparePrefixSumL0(int* _prefixOriginal, unsigned int* _fineConnectedMsk, int vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          localWarpId = threadIdx.x / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    unsigned int connectMsk  = _fineConnectedMsk[idx];
    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    unsigned int visited   = (1U << laneId);
    while(connectMsk != -1)
    {
        unsigned int todo = visited ^ connectMsk;

        if(!todo)
            break;

        unsigned int nextVist = __ffs(todo) - 1;
        visited |= (1U << nextVist);
        connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
    }

    _fineConnectedMsk[idx] = connectMsk;

    unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        //prefixSum[warpId]++;
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
    {
        _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}

__global__ void _preparePrefixSumL0_new(int*          _prefixOriginal,
                                        unsigned int* _fineConnectedMsk,
                                        int*          _partId_map_real,
                                        //int*          _real_map_partId,
                                        int vertNum)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= vertNum)
        return;
    int warpId      = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = tdx % BANKSIZE;

    int idx = _partId_map_real[tdx];


    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectedMsk[idx];
        if(laneId == 0)
        {
            prefixSum[localWarpId] = 0;
        }
        cacheMask[threadIdx.x] = connectMsk;
        unsigned int visited   = (1U << laneId);
        while(connectMsk != -1)
        {
            unsigned int todo = visited ^ connectMsk;

            if(!todo)
                break;

            unsigned int nextVist = __ffs(todo) - 1;
            visited |= (1U << nextVist);
            connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
        }

        _fineConnectedMsk[idx] = connectMsk;

        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            //prefixSum[warpId]++;
            atomicAdd(prefixSum + localWarpId, 1);
        }

        if(laneId == 0)
        {
            _prefixOriginal[warpId] = prefixSum[localWarpId];
        }
    }
}


__global__ void _buildLevel1(int2*               _levelSize,
                             int*                _coarseSpaceTable,
                             int*                _goingNext,
                             const unsigned int* _fineConnectedMsk,
                             const int*          _prefixSumOriginal,
                             const int*          _prefixOriginal,
                             int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(idx == vertNum - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _fineConnectedMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    //lanePrefix2 += _prefixSumOriginal[warpId];

    //unsigned int elected_lane = __ffs(connMsk) - 1;
    //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


    _coarseSpaceTable[idx + 0 * vertNum] = theLanePrefix;
    _goingNext[idx] = theLanePrefix + (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}


__global__ void _buildLevel1_new(int2*               _levelSize,
                                 int*                _coarseSpaceTable,
                                 int*                _goingNext,
                                 const unsigned int* _fineConnectedMsk,
                                 const int*          _prefixSumOriginal,
                                 const int*          _prefixOriginal,
                                 int*                _partId_map_real,
                                 int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int warpId      = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = tdx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(tdx == number - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
    int idx = _partId_map_real[tdx];
    if(idx >= 0)
    {

        unsigned int connMsk = _fineConnectedMsk[idx];

        unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            atomicOr(electedMask + localWarpId, (1U << laneId));
        }

        //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        //lanePrefix2 += _prefixSumOriginal[warpId];

        //unsigned int elected_lane = __ffs(connMsk) - 1;
        //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

        lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

        unsigned int elected_lane = __ffs(connMsk) - 1;
        unsigned int theLanePrefix =
            lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


        _coarseSpaceTable[idx] = theLanePrefix;
        _goingNext[idx] = theLanePrefix + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
}


__global__ void _buildConnectMaskLx(const unsigned int* _neighborStart,
                                    unsigned int*       _neighborNum,
                                    unsigned int*       _neighborList,
                                    int*                _coarseSpaceTable,
                                    unsigned int*       _nextConnectedMsk,
                                    const unsigned int* _fineConnectedMsk,
                                    int                 level,
                                    int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    unsigned int prefixMsk = _fineConnectedMsk[idx];
    unsigned int connMsk   = 0;
    unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
    int          kn        = _neighborNum[idx];
    int          nk        = 0;
    int          startId   = _neighborStart[idx];
    for(int i = 0; i < kn; i++)
    {
        unsigned int connect = _neighborList[startId + i];
        unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

        if(coarseIdx / BANKSIZE == coarseConnect / BANKSIZE)
        {
            unsigned int off = coarseConnect % BANKSIZE;
            connMsk |= (1U << off);
        }
        else
        {
            _neighborList[startId + nk] = connect;
            nk++;
        }
    }

    _neighborNum[idx] = nk;

    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    cacheMsk[threadIdx.x] = 0;

    if(__popc(prefixMsk) == BANKSIZE)
    {
        atomicOr(cacheMsk + localWarpId * BANKSIZE, connMsk);
        connMsk = cacheMsk[localWarpId * BANKSIZE];
        //if (laneId == 0) {
        //	cacheMsk[localWarpId] = 0;
        //}
    }
    else
    {
        unsigned int electedLane = __ffs(prefixMsk) - 1;
        if(connMsk)
        {
            atomicOr(cacheMsk + localWarpId * BANKSIZE + electedLane, connMsk);
        }
        connMsk = cacheMsk[localWarpId * BANKSIZE + electedLane];
    }

    unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

    if(connMsk && electedPrefix == 0)
    {
        atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
    }
}

__global__ void _buildConnectMaskLx_new(const unsigned int* _neighborStart,
                                        unsigned int*       _neighborNum,
                                        unsigned int*       _neighborList,
                                        int*                _coarseSpaceTable,
                                        unsigned int*       _nextConnectedMsk,
                                        const unsigned int* _fineConnectedMsk,
                                        int                 level,
                                        int*                _partId_map_real,
                                        //int*                _real_map_partId,
                                        int vertNum,
                                        int number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int            warpId      = tdx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = tdx % BANKSIZE;
    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    int            idx = _partId_map_real[tdx];
    if(idx >= 0)
    {

        unsigned int prefixMsk = _fineConnectedMsk[idx];
        unsigned int connMsk   = 0;
        unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
        int          kn        = _neighborNum[idx];
        int          nk        = 0;
        int          startId   = _neighborStart[idx];
        for(int i = 0; i < kn; i++)
        {
            unsigned int connect = _neighborList[startId + i];
            unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

            if(coarseIdx / BANKSIZE == coarseConnect / BANKSIZE)
            {
                unsigned int off = coarseConnect % BANKSIZE;
                connMsk |= (1U << off);
            }
            else
            {
                _neighborList[startId + nk] = connect;
                nk++;
            }
        }

        _neighborNum[idx] = nk;


        cacheMsk[threadIdx.x] = 0;

        if(__popc(prefixMsk) == BANKSIZE)
        {
            atomicOr(cacheMsk + localWarpId * BANKSIZE, connMsk);
            connMsk = cacheMsk[localWarpId * BANKSIZE];
            //if (laneId == 0) {
            //	cacheMsk[localWarpId] = 0;
            //}
        }
        else
        {
            unsigned int electedLane = __ffs(prefixMsk) - 1;
            if(connMsk)
            {
                atomicOr(cacheMsk + localWarpId * BANKSIZE + electedLane, connMsk);
            }
            connMsk = cacheMsk[localWarpId * BANKSIZE + electedLane];
        }

        unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

        if(connMsk && electedPrefix == 0)
        {
            atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
        }
    }
}


__global__ void _nextLevelCluster(unsigned int* _nextConnectedMsk, unsigned int* _nextPrefix, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int            warpId      = idx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = idx % BANKSIZE;
    __shared__ int prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    unsigned int connMsk = (1U << laneId);

    connMsk |= _nextConnectedMsk[idx];

    //unsigned int cachedMsk = connMsk;

    __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
    cachedMsk[threadIdx.x] = connMsk;
    unsigned int visited   = (1U << laneId);

    while(true)
    {
        unsigned int todo = visited ^ connMsk;

        if(!todo)
            break;

        unsigned int nextVisit = __ffs(todo) - 1;

        visited |= (1U << nextVisit);

        connMsk |= cachedMsk[nextVisit + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cachedMsk, nextVisit);
    }

    _nextConnectedMsk[idx] = connMsk;

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
        _nextPrefix[warpId] = prefixSum[localWarpId];
}

__global__ void _prefixSumLx(int2*         _levelSize,
                             unsigned int* _nextPrefix,
                             unsigned int* _nextPrefixSum,
                             unsigned int* _nextConnectMsk,
                             int*          _goingNext,
                             int           level,
                             int           levelBegin,
                             int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }

    if(idx == number - 1)
    {
        _levelSize[level + 1].x = _nextPrefixSum[warpId] + _nextPrefix[warpId];
        _levelSize[level + 1].y = levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _nextConnectMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _nextPrefixSum[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);

    _nextConnectMsk[idx] = theLanePrefix;
    _goingNext[idx + levelBegin] =
        theLanePrefix + levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}

__global__ void _computeNextLevel(int*          _coarseSpaceTable,
                                  unsigned int* _nextConnectMsk,
                                  int           level,
                                  int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int next = _coarseSpaceTable[(level - 1) * number + idx];
    _coarseSpaceTable[(level)*number + idx] = _nextConnectMsk[next];
}

__global__ void _aggregationKernel(int*                _denseLevel,
                                   __GEIGEN__::itable* _coarseTable,
                                   int*                _goingNext,
                                   int                 levelNum,
                                   int                 number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int currentId = idx;
    //int aggLevel  = levelNum - 1;
    //__shared__ int4 ctable[DEFAULT_BLOCKSIZE];
    __GEIGEN__::itable ctable;
    for(int l = 0; l < levelNum - 1; l++)
    {
        int next = _goingNext[currentId];

        //int next0 = __shfl_sync(0xffffffff, next, 0);
        //printf("%d   %d   %d    %d\n", next, next0, l,  idx);
        //if (next == next0) {
        //	aggLevel = std::min(l, aggLevel);
        //}

        currentId           = next;
        *(ctable.index + l) = next;
    }

    //_denseLevel[idx] = aggLevel;

    //printf("%d   %d\n", aggLevel, idx);

    _coarseTable[idx] = ctable;
}


__global__ void _prepareHessian(const __GEIGEN__::Matrix12x12d* Hessians12,
                                const __GEIGEN__::Matrix9x9d*   Hessians9,
                                const __GEIGEN__::Matrix6x6d*   Hessians6,
                                const __GEIGEN__::Matrix3x3d*   Hessians3,
                                const uint4*                    D4Index,
                                const uint3*                    D3Index,
                                const uint2*                    D2Index,
                                const uint32_t*                 D1Index,
                                __GEIGEN__::MasMatrixT*         P96,
                                int                             numbers4,
                                int                             numbers3,
                                int                             numbers2,
                                int                             numbers1,
                                int*                            _goingNext,
                                int                             levelNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers4 + numbers3 + numbers2 + numbers1)
        return;

    if(idx < numbers4)
    {
        int Hid  = idx / 144;
        int qid  = idx % 144;
        int qrid = qid / 12;
        int qcid = qid % 12;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D4Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);

        //int cha = vertCid - vertRid;

        int         roffset = qrid % 3;
        int         coffset = qcid % 3;
        Precision_T Hval    = Hessians12[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;

        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else if(numbers4 <= idx && idx < numbers3 + numbers4)
    {
        idx -= numbers4;
        int Hid = idx / 81;
        int qid = idx % 81;

        int qrid = qid / 9;
        int qcid = qid % 9;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D3Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);
        //int Pid = vertCid / 12;
        //int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians9[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else if(numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2)
    {
        idx -= numbers3 + numbers4;
        int Hid = idx / 36;
        int qid = idx % 36;

        int qrid = qid / 6;
        int qcid = qid % 6;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D2Index[Hid].x);

        int vertCid = *(nodeInex + vcid);
        int vertRid = *(nodeInex + vrid);
        //int Pid = vertCid / 12;
        int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians6[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else
    {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int qrid = qid / 3;
        int qcid = qid % 3;

        int nodeIndex = D1Index[Hid];

        Precision_T Hval = Hessians3[Hid].m[qrid][qcid];

        int cPid  = nodeIndex / BANKSIZE;
        int Pod   = nodeIndex % BANKSIZE;
        int level = 0;


        atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            nodeIndex = _goingNext[nodeIndex];
            Pod       = nodeIndex % BANKSIZE;
            cPid      = nodeIndex / BANKSIZE;
            atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);
        }
    }
}


__global__ void _prepareHessian_new(const __GEIGEN__::Matrix12x12d* Hessians12,
                                    const __GEIGEN__::Matrix9x9d*   Hessians9,
                                    const __GEIGEN__::Matrix6x6d*   Hessians6,
                                    const __GEIGEN__::Matrix3x3d*   Hessians3,
                                    const uint4*                    D4Index,
                                    const uint3*                    D3Index,
                                    const uint2*                    D2Index,
                                    const uint32_t*                 D1Index,
                                    __GEIGEN__::MasMatrixT*         P96,
                                    int                             numbers4,
                                    int                             numbers3,
                                    int                             numbers2,
                                    int                             numbers1,
                                    int*                            _goingNext,
                                    int* _real_map_partId,
                                    int  levelNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers4 + numbers3 + numbers2 + numbers1)
        return;

    if(idx < numbers4)
    {
        int Hid  = idx / 144;
        int qid  = idx % 144;
        int qrid = qid / 12;
        int qcid = qid % 12;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex     = &(D4Index[Hid].x);
        int   vertCid_real = *(nodeInex + vcid);
        int   vertRid_real = *(nodeInex + vrid);

        //int cha = vertCid - vertRid;

        int         roffset = qrid % 3;
        int         coffset = qcid % 3;
        Precision_T Hval    = Hessians12[Hid].m[qrid][qcid];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;
        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);


        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else if(numbers4 <= idx && idx < numbers3 + numbers4)
    {
        idx -= numbers4;
        int Hid = idx / 81;
        int qid = idx % 81;

        int qrid = qid / 9;
        int qcid = qid % 9;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex     = &(D3Index[Hid].x);
        int   vertCid_real = *(nodeInex + vcid);
        int   vertRid_real = *(nodeInex + vrid);
        //int Pid = vertCid / 12;
        //int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians9[Hid].m[qrid][qcid];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;
        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else if(numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2)
    {
        idx -= numbers3 + numbers4;
        int Hid = idx / 36;
        int qid = idx % 36;

        int qrid = qid / 6;
        int qcid = qid % 6;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D2Index[Hid].x);

        int vertCid_real = *(nodeInex + vcid);
        int vertRid_real = *(nodeInex + vrid);

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians6[Hid].m[qrid][qcid];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;
        atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
            atomicAdd(&(P96[cPid].m[(vertRid % BANKSIZE) * 3 + roffset][(vertCid % BANKSIZE) * 3 + coffset]),
                      Hval);
        }
    }
    else
    {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int qrid = qid / 3;
        int qcid = qid % 3;

        int nodeIndex = D1Index[Hid];

        Precision_T Hval = Hessians3[Hid].m[qrid][qcid];

        int cPid  = _real_map_partId[nodeIndex] / BANKSIZE;
        int Pod   = _real_map_partId[nodeIndex] % BANKSIZE;
        int level = 0;


        atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);


        while(level < levelNum - 1)
        {
            level++;
            nodeIndex = _goingNext[nodeIndex];
            Pod       = nodeIndex % BANKSIZE;
            cPid      = nodeIndex / BANKSIZE;
            atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);
        }
    }
}


__global__ void _prepareSymHessian(const __GEIGEN__::Matrix12x12d* Hessians12,
                                   const __GEIGEN__::Matrix9x9d*   Hessians9,
                                   const __GEIGEN__::Matrix6x6d*   Hessians6,
                                   const __GEIGEN__::Matrix3x3d*   Hessians3,
                                   const uint4*                    D4Index,
                                   const uint3*                    D3Index,
                                   const uint2*                    D2Index,
                                   const uint32_t*                 D1Index,
                                   __GEIGEN__::MasMatrixSymT*      _invMatrix,
                                   int                             numbers4,
                                   int                             numbers3,
                                   int                             numbers2,
                                   int                             numbers1,
                                   int*                            _goingNext,
                                   int                             levelNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers4 + numbers3 + numbers2 + numbers1)
        return;

    if(idx < numbers4)
    {
        int Hid  = idx / 144;
        int qid  = idx % 144;
        int qrid = qid / 12;
        int qcid = qid % 12;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D4Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);

        //int cha = vertCid - vertRid;

        int         roffset = qrid % 3;
        int         coffset = qcid % 3;
        Precision_T Hval    = Hessians12[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else if(numbers4 <= idx && idx < numbers3 + numbers4)
    {
        idx -= numbers4;
        int Hid = idx / 81;
        int qid = idx % 81;

        int qrid = qid / 9;
        int qcid = qid % 9;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D3Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);

        //int Pid = vertCid / 12;
        //int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians9[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else if(numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2)
    {
        idx -= numbers3 + numbers4;
        int Hid = idx / 36;
        int qid = idx % 36;

        int qrid = qid / 6;
        int qcid = qid % 6;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D2Index[Hid].x);

        int vertCid = *(nodeInex + vcid);
        int vertRid = *(nodeInex + vrid);

        //int Pid = vertCid / 12;
        int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians6[Hid].m[qrid][qcid];

        int cPid  = vertCid / BANKSIZE;
        int level = 0;
        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else
    {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int qrid = qid / 3;
        int qcid = qid % 3;

        int nodeIndex = D1Index[Hid];

        Precision_T Hval = Hessians3[Hid].m[qrid][qcid];

        int cPid  = nodeIndex / BANKSIZE;
        int Pod   = nodeIndex % BANKSIZE;
        int level = 0;


        int index = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;


        atomicAdd(&(_invMatrix[cPid].M[index].m[qrid][qcid]), Hval);


        while(level < levelNum - 1)
        {
            level++;
            nodeIndex = _goingNext[nodeIndex];
            Pod       = nodeIndex % BANKSIZE;
            cPid      = nodeIndex / BANKSIZE;
            index     = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;
            atomicAdd(&(_invMatrix[cPid].M[index].m[qrid][qcid]), Hval);
        }
    }
}


__global__ void _prepareSymHessian_new(const __GEIGEN__::Matrix12x12d* Hessians12,
                                       const __GEIGEN__::Matrix9x9d* Hessians9,
                                       const __GEIGEN__::Matrix6x6d* Hessians6,
                                       const __GEIGEN__::Matrix3x3d* Hessians3,
                                       const uint4*                  D4Index,
                                       const uint3*                  D3Index,
                                       const uint2*                  D2Index,
                                       const uint32_t*               D1Index,
                                       __GEIGEN__::MasMatrixSymT*    _invMatrix,
                                       int                           numbers4,
                                       int                           numbers3,
                                       int                           numbers2,
                                       int                           numbers1,
                                       int*                          _goingNext,
                                       int* _real_map_partId,
                                       int  levelNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers4 + numbers3 + numbers2 + numbers1)
        return;

    if(idx < numbers4)
    {
        int Hid  = idx / 144;
        int qid  = idx % 144;
        int qrid = qid / 12;
        int qcid = qid % 12;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex     = &(D4Index[Hid].x);
        int   vertCid_real = *(nodeInex + vcid);
        int   vertRid_real = *(nodeInex + vrid);

        //int cha = vertCid - vertRid;

        int         roffset = qrid % 3;
        int         coffset = qcid % 3;
        Precision_T Hval    = Hessians12[Hid].m[qrid][qcid];


        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else if(numbers4 <= idx && idx < numbers3 + numbers4)
    {
        idx -= numbers4;
        int Hid = idx / 81;
        int qid = idx % 81;

        int qrid = qid / 9;
        int qcid = qid % 9;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex     = &(D3Index[Hid].x);
        int   vertCid_real = *(nodeInex + vcid);
        int   vertRid_real = *(nodeInex + vrid);

        //int Pid = vertCid / 12;
        //int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians9[Hid].m[qrid][qcid];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else if(numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2)
    {
        idx -= numbers3 + numbers4;
        int Hid = idx / 36;
        int qid = idx % 36;

        int qrid = qid / 6;
        int qcid = qid % 6;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D2Index[Hid].x);

        int vertCid_real = *(nodeInex + vcid);
        int vertRid_real = *(nodeInex + vrid);


        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians6[Hid].m[qrid][qcid];

        int vertCid = _real_map_partId[vertCid_real];
        int vertRid = _real_map_partId[vertRid_real];
        int cPid    = vertCid / BANKSIZE;

        int level = 0;

        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;
        }
        if(level >= levelNum)
        {
            return;
        }
        int bvRid = vertRid % BANKSIZE;
        int bvCid = vertCid % BANKSIZE;
        int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
        if(vertCid >= vertRid)
            atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / BANKSIZE;

            bvRid = vertRid % BANKSIZE;
            bvCid = vertCid % BANKSIZE;
            index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
            if(vertCid >= vertRid)
                atomicAdd(&(_invMatrix[cPid].M[index].m[roffset][coffset]), Hval);
        }
    }
    else
    {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int qrid = qid / 3;
        int qcid = qid % 3;

        int nodeIndex = D1Index[Hid];

        Precision_T Hval = Hessians3[Hid].m[qrid][qcid];

        int cPid  = _real_map_partId[nodeIndex] / BANKSIZE;
        int Pod   = _real_map_partId[nodeIndex] % BANKSIZE;
        int level = 0;


        int index = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;


        atomicAdd(&(_invMatrix[cPid].M[index].m[qrid][qcid]), Hval);


        while(level < levelNum - 1)
        {
            level++;
            nodeIndex = _goingNext[nodeIndex];
            Pod       = nodeIndex % BANKSIZE;
            cPid      = nodeIndex / BANKSIZE;
            index     = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;
            atomicAdd(&(_invMatrix[cPid].M[index].m[qrid][qcid]), Hval);
        }
    }
}


__global__ void __setMassMat_P(const double*           _masses,
                               const int*              _goingNext,
                               __GEIGEN__::MasMatrixT* _Mat96,
                               int                     levelNum,
                               int                     number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId = idx / BANKSIZE;
    int laneId = idx % BANKSIZE;

    Precision_T mass = _masses[idx];

    int Pid = idx / BANKSIZE;
    int Pod = idx % BANKSIZE;

    _Mat96[Pid].m[Pod * 3][Pod * 3]         = mass;
    _Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1] = mass;
    _Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2] = mass;

    int level = 0;

    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        Pid = idx / BANKSIZE;
        Pod = idx % BANKSIZE;
        atomicAdd(&(_Mat96[Pid].m[Pod * 3][Pod * 3]), mass);
        atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1]), mass);
        atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2]), mass);
    }
}

__global__ void __setMassMat_P_new(const double*           _masses,
                                   const int*              _goingNext,
                                   const int*              _partId_map_real,
                                   __GEIGEN__::MasMatrixT* _Mat96,
                                   int                     levelNum,
                                   int                     number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    //int warpId = idx / BANKSIZE;
    //int laneId = idx % BANKSIZE;

    int rdx = _partId_map_real[idx];


    Precision_T mass = 0;
    //_masses[rdx];
    if(rdx >= 0)
    {
        mass = _masses[rdx];


        int Pid = idx / BANKSIZE;
        int Pod = idx % BANKSIZE;

        _Mat96[Pid].m[Pod * 3][Pod * 3]         = mass;
        _Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1] = mass;
        _Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2] = mass;

        int level = 0;

        while(level < levelNum - 1)
        {
            level++;
            rdx = _goingNext[rdx];
            Pid = rdx / BANKSIZE;
            Pod = rdx % BANKSIZE;
            atomicAdd(&(_Mat96[Pid].m[Pod * 3][Pod * 3]), mass);
            atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1]), mass);
            atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2]), mass);
        }
    }
}

__global__ void __setSymMassMat_P_new(const double* _masses,
                                      const int*    _goingNext,
                                      const int*    _partId_map_real,
                                      __GEIGEN__::MasMatrixSymT* _invMat,
                                      int                        levelNum,
                                      int                        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    //int warpId = idx / BANKSIZE;
    //int laneId = idx % BANKSIZE;
    int rdx = _partId_map_real[idx];


    Precision_T mass = 0;
    //_masses[rdx];
    if(rdx >= 0)
    {
        mass = _masses[rdx];


        int Pid = idx / BANKSIZE;
        int Pod = idx % BANKSIZE;

        int index = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;
        _invMat[Pid].M[index].m[0][0] = mass;
        _invMat[Pid].M[index].m[1][1] = mass;
        _invMat[Pid].M[index].m[2][2] = mass;

        int level = 0;

        while(level < levelNum - 1)
        {
            level++;
            rdx = _goingNext[rdx];
            Pid = rdx / BANKSIZE;
            Pod = rdx % BANKSIZE;

            index = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;

            atomicAdd(&(_invMat[Pid].M[index].m[0][0]), mass);
            atomicAdd(&(_invMat[Pid].M[index].m[1][1]), mass);
            atomicAdd(&(_invMat[Pid].M[index].m[2][2]), mass);
        }
    }
}

__global__ void __setSymMassMat_P(const double*              _masses,
                                  const int*                 _goingNext,
                                  __GEIGEN__::MasMatrixSymT* _invMat,
                                  int                        levelNum,
                                  int                        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId = idx / BANKSIZE;
    int laneId = idx % BANKSIZE;

    Precision_T mass = _masses[idx];

    int Pid = idx / BANKSIZE;
    int Pod = idx % BANKSIZE;

    int index                     = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;
    _invMat[Pid].M[index].m[0][0] = mass;
    _invMat[Pid].M[index].m[1][1] = mass;
    _invMat[Pid].M[index].m[2][2] = mass;

    int level = 0;

    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        Pid = idx / BANKSIZE;
        Pod = idx % BANKSIZE;

        index = BANKSIZE * Pod - (Pod + 1) * Pod / 2 + Pod;

        atomicAdd(&(_invMat[Pid].M[index].m[0][0]), mass);
        atomicAdd(&(_invMat[Pid].M[index].m[1][1]), mass);
        atomicAdd(&(_invMat[Pid].M[index].m[2][2]), mass);
    }
}

__global__ void __inverse1_P96x96(__GEIGEN__::MasMatrixT*    sPMas,
                                  __GEIGEN__::MasMatrixSymT* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId       = idx / (BANKSIZE * 3);
    int i           = idx % (BANKSIZE * 3);
    int block_matId = threadIdx.x / (BANKSIZE * 3);

    __shared__ Precision_T colm[32 / BANKSIZE][BANKSIZE * 3];

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            sPMas[matId].m[j][i] = _invMatrix[matId].M[index].m[j % 3][i % 3];
        }
        else
        {
            index = BANKSIZE * colId - colId * (colId + 1) / 2 + rowId;
            sPMas[matId].m[j][i] = _invMatrix[matId].M[index].m[i % 3][j % 3];
        }
        if(i == j)
        {
            if(sPMas[matId].m[j][i] == 0)
            {
                sPMas[matId].m[j][i] = 1;
            }
        }
    }

    __syncthreads();
    __threadfence();

    if(i % 3 < 2)
        sPMas[matId].m[i + 1][i] = sPMas[matId].m[i][i + 1];
    else
        sPMas[matId].m[i][i - 2] = sPMas[matId].m[i - 2][i];
    __syncthreads();
    __threadfence();

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();
        __threadfence();

        rt = sPMas[matId].m[j][j];

        colm[block_matId][i] = sPMas[matId].m[i][j];

        __syncthreads();
        __threadfence();
        if(i == j)
        {

            sPMas[matId].m[i][j] = 1;
        }
        else
        {
            sPMas[matId].m[i][j] = 0;
        }
        __syncthreads();
        __threadfence();

        sPMas[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                __threadfence();

                sPMas[matId].m[k][i] += rate * sPMas[matId].m[j][i];
            }
        }

        j++;
    }
    __syncthreads();
    __threadfence();
    if(i % 3 < 2)
        sPMas[matId].m[i + 1][i] = sPMas[matId].m[i][i + 1];
    else
        sPMas[matId].m[i][i - 2] = sPMas[matId].m[i - 2][i];
    __syncthreads();
    __threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        //PMas[matId].m[j][i] = sPMas[block_matId][j][i];
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            _invMatrix[matId].M[index].m[j % 3][i % 3] = sPMas[matId].m[j][i];
        }
    }
}


__global__ void __inverse2_P96x96(__GEIGEN__::MasMatrixT*    PMas,
                                  __GEIGEN__::MasMatrixSymf* invPMas,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / (BANKSIZE * 3);
    int i     = idx % (BANKSIZE * 3);
    //int localMatId = threadIdx.x / 96;
    int                    block_matId = threadIdx.x / (BANKSIZE * 3);
    __shared__ Precision_T colm[32 / BANKSIZE][BANKSIZE * 3];
    //invPMas[matId].m[j][i] = 1;
    if(PMas[matId].m[i][i] == 0)
    {
        PMas[matId].m[i][i] = 1;
    }

    __syncthreads();
    __threadfence();
    //if(i % 3 < 2)
    //    PMas[matId].m[i + 1][i] = PMas[matId].m[i][i + 1];
    //else
    //    PMas[matId].m[i][i - 2] = PMas[matId].m[i - 2][i];
    //__syncthreads();
    //__threadfence();

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();
        __threadfence();

        rt = PMas[matId].m[j][j];


        //if(rt <= 1e-3)
        //{
        //    int k = j + 1;
        //    for(k; k < (BANKSIZE * 3); k++)
        //    {
        //        if(PMas[matId].m[k][j] > 1e-3)
        //            break;
        //    }
        //    if(k == (BANKSIZE * 3))
        //    {
        //        j++;
        //        continue;
        //    }
        //    if(i >= j)
        //        PMas[matId].m[j][i] = PMas[matId].m[k][i];
        //}
        //__syncthreads();
        //__threadfence();

        colm[block_matId][i] = PMas[matId].m[i][j];

        __syncthreads();
        __threadfence();
        if(i == j)
        {

            PMas[matId].m[i][j] = 1;
        }
        else
        {
            PMas[matId].m[i][j] = 0;
        }
        __syncthreads();
        __threadfence();

        PMas[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                __threadfence();

                PMas[matId].m[k][i] += rate * PMas[matId].m[j][i];
            }
        }

        j++;
    }
    __syncthreads();
    __threadfence();
    if(i % 3 < 2)
        PMas[matId].m[i + 1][i] = PMas[matId].m[i][i + 1];
    else
        PMas[matId].m[i][i - 2] = PMas[matId].m[i - 2][i];
    __syncthreads();
    __threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        //PMas[matId].m[j][i] = sPMas[block_matId][j][i];
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            invPMas[matId].M[index].m[j % 3][i % 3] = PMas[matId].m[j][i];
        }
    }
}


__global__ void __inverse3_P96x96(__GEIGEN__::MasMatrixT* PMas,
                                  __GEIGEN__::MasMatrixT* invPMas,
                                  int                     numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / (BANKSIZE * 3);
    int i     = idx % (BANKSIZE * 3);
    //int localMatId = threadIdx.x / 96;

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        if(i == j)
        {
            invPMas[matId].m[j][i] = 1;
            if(PMas[matId].m[j][i] == 0)
            {
                PMas[matId].m[j][i] = 1;
            }
        }
        else
        {
            invPMas[matId].m[j][i] = 0;
        }
    }
    __syncthreads();
    __threadfence();
    int         j  = 0;
    Precision_T rt = PMas[matId].m[0][0];
    __syncthreads();
    __threadfence();
    while(/*loopId[localMatId]*/ j < (BANKSIZE * 3))
    {
        if(i <= j)
            invPMas[matId].m[j][i] /= rt;
        if(i > j)
            PMas[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -PMas[matId].m[k][j];
                __syncthreads();
                __threadfence();
                if(i <= j)
                    invPMas[matId].m[k][i] += rate * invPMas[matId].m[j][i];
                if(i > j)
                    PMas[matId].m[k][i] += rate * PMas[matId].m[j][i];
            }
        }

        __syncthreads();
        __threadfence();
        j++;
        rt = PMas[matId].m[j][j];
    }
}

__global__ void __inverse4_P96x96(__GEIGEN__::MasMatrixT*    PMas,
                                  __GEIGEN__::MasMatrixSymf* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId       = idx / (BANKSIZE * 3);
    int i           = idx % (BANKSIZE * 3);
    int block_matId = threadIdx.x / (BANKSIZE * 3);

    __shared__ float sPMas[32 / BANKSIZE][BANKSIZE * 3][BANKSIZE * 3];
    __shared__ float colm[32 / BANKSIZE][BANKSIZE * 3];

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        sPMas[block_matId][j][i] = PMas[matId].m[j][i];
        if(i == j)
        {
            if(sPMas[block_matId][j][i] == 0)
            {
                sPMas[block_matId][j][i] = 1;
            }
        }
    }

    __syncthreads();
    __threadfence();
    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();
        __threadfence();

        rt = sPMas[block_matId][j][j];

        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        __threadfence();
        if(i == j)
        {

            sPMas[block_matId][i][j] = 1;
        }
        else
        {
            sPMas[block_matId][i][j] = 0;
        }
        __syncthreads();
        __threadfence();

        sPMas[block_matId][j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                __threadfence();

                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }

        j++;
    }
    __syncthreads();
    __threadfence();
    //sPMas[block_matId][i][i] += 1e-8;
    if(i % 3 < 2)
        sPMas[block_matId][i + 1][i] = sPMas[block_matId][i][i + 1];
    else
        sPMas[block_matId][i][i - 2] = sPMas[block_matId][i - 2][i];
    __syncthreads();
    __threadfence();
    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        //PMas[matId].m[j][i] = sPMas[block_matId][j][i];

        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            _invMatrix[matId].M[index].m[j % 3][i % 3] = sPMas[block_matId][j][i];
        }
    }
}

//TODO: check
__global__ void __inverse5_TEST(__GEIGEN__::MasMatrixT* PMas,
                                __GEIGEN__::MasMatrixT* invPMas,
                                int                     numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / (BANKSIZE * 3);
    int i     = idx % (BANKSIZE * 3);
    //int localMatId = threadIdx.x / 96;

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        if(i == j)
        {
            invPMas[matId].m[j][i] = 1;
            if(PMas[matId].m[j][i] == 0)
            {
                PMas[matId].m[j][i] = 1;
            }
        }
        else
        {
            invPMas[matId].m[j][i] = 0;
        }
    }
    __syncthreads();
    __threadfence();

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        Precision_T rt = PMas[matId].m[j][j];

        //if(i <= j)
        //    invPMas[matId].m[j][i] /= rt;
        //if(i > j)
        //    PMas[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = j + 1; k < (BANKSIZE * 3); k++)
        {
            //if(k != j)

            Precision_T rate    = -PMas[matId].m[k][j] / rt;
            PMas[matId].m[k][j] = 0;
            __syncthreads();
            __threadfence();
            if(i < j)
            {
                invPMas[matId].m[k][i] += rate * invPMas[matId].m[j][i];
            }
            else if(i == j)
            {
                invPMas[matId].m[k][i] = rate;
            }

            if(i > j)
                PMas[matId].m[k][i] += rate * PMas[matId].m[j][i];
        }

        __syncthreads();
        __threadfence();
        //rt = PMas[matId].m[j][j];
    }

    for(int j = (BANKSIZE * 3) - 1; j >= 0; j--)
    {
        Precision_T rt = PMas[matId].m[j][j];

        __syncthreads();
        __threadfence();
        for(int k = j - 1; k >= 0; k--)
        {
            //if(k != j)

            Precision_T rate    = -PMas[matId].m[k][j] / rt;
            PMas[matId].m[k][j] = 0;
            __syncthreads();
            __threadfence();
            /*if(i < j)
             {*/
            invPMas[matId].m[k][i] += rate * invPMas[matId].m[j][i];
            //}


            if(i > j)
                PMas[matId].m[k][i] += rate * PMas[matId].m[j][i];
        }

        __syncthreads();
        __threadfence();
        //rt = PMas[matId].m[j][j];
    }

    for(int x = 0; x < 96; x++)
    {

        invPMas[matId].m[x][i] *= 1.f / PMas[matId].m[x][x];
    }
}


__global__ void __inverse6_P96x96(__GEIGEN__::MasMatrixSymf* _preMatrix,
                                  __GEIGEN__::MasMatrixSymT* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId       = idx / (BANKSIZE * 3);
    int i           = idx % (BANKSIZE * 3);
    int block_matId = threadIdx.x / (BANKSIZE * 3);

    __shared__ double sPMas[32 / BANKSIZE][BANKSIZE * 3][BANKSIZE * 3];
    __shared__ double colm[32 / BANKSIZE][BANKSIZE * 3];

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index].m[j % 3][i % 3];
        }
        else
        {
            index = BANKSIZE * colId - colId * (colId + 1) / 2 + rowId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index].m[i % 3][j % 3];
        }
        if(i == j)
        {
            if(sPMas[block_matId][j][i] == 0)
            {
                sPMas[block_matId][j][i] = 1;
            }
        }
    }

    //__syncthreads();
    //__threadfence();

    /*if(i % 3 < 2)
        sPMas[block_matId][i + 1][i] = sPMas[block_matId][i][i + 1];
    else
        sPMas[block_matId][i][i - 2] = sPMas[block_matId][i - 2][i];
    __syncthreads();
    __threadfence();*/

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();
        __threadfence();

        rt = sPMas[block_matId][j][j];

        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        __threadfence();
        if(i == j)
        {

            sPMas[block_matId][i][j] = 1;
        }
        else
        {
            sPMas[block_matId][i][j] = 0;
        }
        __syncthreads();
        __threadfence();

        sPMas[block_matId][j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                __threadfence();

                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }

        j++;
    }
    __syncthreads();
    __threadfence();
    if(i % 3 < 2)
        sPMas[block_matId][i + 1][i] = sPMas[block_matId][i][i + 1];
    else
        sPMas[block_matId][i][i - 2] = sPMas[block_matId][i - 2][i];
    __syncthreads();
    __threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        //PMas[matId].m[j][i] = sPMas[block_matId][j][i];
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            _preMatrix[matId].M[index].m[j % 3][i % 3] = sPMas[block_matId][j][i];
        }
    }
}


__global__ void __buildMultiLevelR_optimized(const double3* _R,
                                             Precision_T3*  _multiLR,
                                             int*           _goingNext,
                                             unsigned int*  _fineConnectMsk,
                                             int            levelNum,
                                             int            numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    Precision_T3 r;
    r.x = _R[idx].x;
    r.y = _R[idx].y;
    r.z = _R[idx].z;

    int laneId      = threadIdx.x % BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int level       = 0;
    _multiLR[idx]   = r;

    __shared__ double c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    unsigned int connectMsk = _fineConnectMsk[idx];
    if(__popc(connectMsk) == BANKSIZE)
    {
        for(int iter = 1; iter < BANKSIZE; iter <<= 1)
        {
            r.x += __shfl_down_sync(0xffffffff, r.x, iter);
            r.y += __shfl_down_sync(0xffffffff, r.y, iter);
            r.z += __shfl_down_sync(0xffffffff, r.z, iter);
        }
        //int level = 0;

        if(laneId == 0)
        {
            while(level < levelNum - 1)
            {
                level++;
                idx = _goingNext[idx];
                atomicAdd((&((_multiLR + idx)->x)), r.x);
                atomicAdd((&((_multiLR + idx)->x) + 1), r.y);
                atomicAdd((&((_multiLR + idx)->x) + 2), r.z);
            }
        }
        return;
    }
    else
    {
        int elected_lane = __ffs(connectMsk) - 1;

        c_sumResidual[threadIdx.x]                         = 0;
        c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = 0;
        c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
        atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane, r.x);
        atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE,
                  r.y);
        atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                  r.z);

        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
        if(electedPrefix == 0)
        {
            while(level < levelNum - 1)
            {
                level++;
                idx = _goingNext[idx];
                atomicAdd((&((_multiLR + idx)->x)), c_sumResidual[threadIdx.x]);
                atomicAdd((&((_multiLR + idx)->x) + 1),
                          c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                atomicAdd((&((_multiLR + idx)->x) + 2),
                          c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
            }
        }
    }
}

__global__ void __buildMultiLevelR_optimized_new(const double3* _R,
                                                 Precision_T3*  _multiLR,
                                                 int*           _goingNext,
                                                 int*           _prefixOrigin,
                                                 unsigned int*  _fineConnectMsk,
                                                 int* _partId_map_real,
                                                 int  levelNum,
                                                 int  numbers)
{
    int pdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pdx >= numbers)
        return;

    Precision_T3 r;
    int          idx = _partId_map_real[pdx];
    if(idx >= 0)
    {

        r.x = _R[idx].x;
        r.y = _R[idx].y;
        r.z = _R[idx].z;
    }
    else
    {
        r.x = 0;
        r.y = 0;
        r.z = 0;
    }

    int laneId      = threadIdx.x % BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int gwarpId     = pdx / BANKSIZE;
    int level       = 0;
    //int rdx         = _real_map_partId[idx];
    _multiLR[pdx] = r;

    __shared__ FloatP c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    __shared__ int prefixSum[DEFAULT_WARPNUM];

    if(laneId == 0)
    {
        prefixSum[localWarpId] = _prefixOrigin[gwarpId];
    }

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        if(prefixSum[localWarpId] == 1)
        {
            int  warpId    = threadIdx.x & 0x1f;
            bool bBoundary = (laneId == 0) || (warpId == 0);

            unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
            mark              = __brev(mark);
            int          clzlen   = __clz(mark << (warpId + 1));
            unsigned int interval = std::min(clzlen, 31 - warpId);


            for(int iter = 1; iter < BANKSIZE; iter <<= 1)
            {
                double tmpx = __shfl_down_sync(0xffffffff, r.x, iter);
                double tmpy = __shfl_down_sync(0xffffffff, r.y, iter);
                double tmpz = __shfl_down_sync(0xffffffff, r.z, iter);
                if(interval >= iter)
                {
                    r.x += tmpx;
                    r.y += tmpy;
                    r.z += tmpz;
                }
            }
            //int level = 0;

            if(bBoundary)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd((&((_multiLR + idx)->x)), r.x);
                    atomicAdd((&((_multiLR + idx)->y)), r.y);
                    atomicAdd((&((_multiLR + idx)->z)), r.z);
                }
            }
            return;
        }
        else
        {
            int elected_lane = __ffs(connectMsk) - 1;

            c_sumResidual[threadIdx.x]                         = 0;
            c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = 0;
            c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane, r.x);
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE,
                      r.y);
            atomicAdd(c_sumResidual + localWarpId * BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                      r.z);

            unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
            if(electedPrefix == 0)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd((&((_multiLR + idx)->x)), c_sumResidual[threadIdx.x]);
                    atomicAdd((&((_multiLR + idx)->y)),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                    atomicAdd((&((_multiLR + idx)->z)),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
                }
            }
        }
    }
}


__global__ void __buildMultiLevelR(
    const double3* _R, Precision_T3* _multiLR, int* _goingNext, int levelNum, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    Precision_T3 r;
    r.x = _R[idx].x;
    r.y = _R[idx].y;
    r.z = _R[idx].z;

    int level     = 0;
    _multiLR[idx] = r;
    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        atomicAdd((&((_multiLR + idx)->x)), r.x);
        atomicAdd((&((_multiLR + idx)->x) + 1), r.y);
        atomicAdd((&((_multiLR + idx)->x) + 2), r.z);
    }
}


__global__ void __buildMultiLevelR_new(const double3* _R,
                                       Precision_T3*  _multiLR,
                                       int*           _goingNext,
                                       int*           _real_map_partId,
                                       int            levelNum,
                                       int            numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    Precision_T3 r;
    r.x = _R[idx].x;
    r.y = _R[idx].y;
    r.z = _R[idx].z;

    int level = 0;

    int pdx = _real_map_partId[idx];

    _multiLR[pdx] = r;
    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        atomicAdd((&((_multiLR + idx)->x)), r.x);
        atomicAdd((&((_multiLR + idx)->x) + 1), r.y);
        atomicAdd((&((_multiLR + idx)->x) + 2), r.z);
    }
}


__global__ void __collectFinalZ(double3*                  _Z,
                                const Precision_T3*       d_multiLevelZ,
                                const __GEIGEN__::itable* _coarseTable,
                                int                       levelnum,
                                int                       number)
{
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx >= number) return;

    //Precision_T3 cz;// = d_multiLevelZ[idx];
    //cz.x = d_multiLevelZ[idx].x;
    //cz.y = d_multiLevelZ[idx].y;
    //cz.z = d_multiLevelZ[idx].z;
    //   __GEIGEN__::itable table    = _coarseTable[idx];
    //int* tablePtr = table.index;
    //   for(int i = 1; i < levelnum; i++)
    //   {
    //	int now = *(tablePtr + i - 1);
    //	cz.x += d_multiLevelZ[now].x;
    //	cz.y += d_multiLevelZ[now].y;
    //	cz.z += d_multiLevelZ[now].z;
    //}

    //_Z[idx].x = cz.x;
    //_Z[idx].y = cz.y;
    //_Z[idx].z = cz.z;


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Precision_T3 cz;  // = d_multiLevelZ[idx];
    cz.x                        = d_multiLevelZ[idx].x;
    cz.y                        = d_multiLevelZ[idx].y;
    cz.z                        = d_multiLevelZ[idx].z;
    __GEIGEN__::itable table    = _coarseTable[idx];
    int*               tablePtr = table.index;
    for(int i = 1; i < levelnum; i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}

__global__ void __collectFinalZ_new(double3*                  _Z,
                                    const Precision_T3*       d_multiLevelZ,
                                    const __GEIGEN__::itable* _coarseTable,
                                    int*                      _real_map_partId,
                                    int                       levelnum,
                                    int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Precision_T3 cz;  // = d_multiLevelZ[idx];
    int          rdx            = _real_map_partId[idx];
    cz.x                        = d_multiLevelZ[rdx].x;
    cz.y                        = d_multiLevelZ[rdx].y;
    cz.z                        = d_multiLevelZ[rdx].z;
    __GEIGEN__::itable table    = _coarseTable[idx];
    int*               tablePtr = table.index;
    for(int i = 1; i < levelnum; i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}


__global__ void _schwarzLocalXSym0(const __GEIGEN__::MasMatrixT* P96,
                                   const Precision_T3*           mR,
                                   Precision_T3*                 mZ,
                                   int                           number)
{
    namespace cg = ::cooperative_groups;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    auto tile = cg::tiled_partition<BANKSIZE>(cg::this_thread_block());

    int tileNo = idx / BANKSIZE;
    int Hid    = tileNo / (BANKSIZE * 3);
    int MRid   = tileNo % (BANKSIZE * 3);

    int  vrid   = Hid * BANKSIZE + MRid / 3;
    auto laneid = tile.thread_rank();

    Precision_T sum      = 0.;
    auto        get_vcid = [Hid](int cid) { return Hid * BANKSIZE + cid / 3; };
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));
    laneid += BANKSIZE;
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));
    laneid += BANKSIZE;
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));

    auto val = cg::reduce(tile, sum, cg::plus<Precision_T>());
    if(tile.thread_rank() == 0)
        *(&(mZ[vrid].x) + MRid % 3) += val;
}



__global__ void _schwarzLocalXSym1(const __GEIGEN__::MasMatrixT* P96,
                                   const Precision_T3*           mR,
                                   Precision_T3*                 mZ,
                                   int                           number)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE * 3);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE * 3);
    int MCid = (idx % hessianSize) % (BANKSIZE * 3);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid / 3;

    //int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid  = idx % (BANKSIZE * 3);


    double         rdata = P96[Hid].m[MRid][MCid] * (*(&(mR[vcid].x) + axisId));
    __shared__ int offset;

    if(threadIdx.x == 0)
    {
        offset = ((BANKSIZE * 3) - GRtid);
    }
    __syncthreads();

    int BRid    = (threadIdx.x - offset + (BANKSIZE * 3)) / (BANKSIZE * 3);
    int landidx = (threadIdx.x - offset) % (BANKSIZE * 3);
    if(BRid == 0)
    {
        landidx = threadIdx.x;
    }

    int  warpId    = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE * 3);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        double tmp = __shfl_down_sync(0xffffffff, rdata, iter);
        if(interval >= iter)
            rdata += tmp;
    }

    if(bBoundary)
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
}


__global__ void _schwarzLocalXSym2(const __GEIGEN__::MasMatrixT* P96,
                                   const Precision_T3*           mR,
                                   Precision_T3*                 mZ,
                                   int                           number)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    double rdata = P96[Hid].m[MRid][MCid * 3] * mR[vcid].x
                   + P96[Hid].m[MRid][MCid * 3 + 1] * mR[vcid].y
                   + P96[Hid].m[MRid][MCid * 3 + 2] * mR[vcid].z;


    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    //unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    //mark                  = __brev(mark);
    //int          clzlen   = __clz(mark << (warpId + 1));
    //unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        double tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        //if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}


__global__ void _schwarzLocalXSym3(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Precision_T3*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    int r3id = MRid % 3;

    int    lvrid = vrid % BANKSIZE;
    int    lvcid = vcid % BANKSIZE;
    FloatP rdata = 0;

    __shared__ Precision_T3 smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(lvcid >= lvrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index].m[r3id][0] * smR[lvcid].x
                + Pred[Hid].M[index].m[r3id][1] * smR[lvcid].y
                + Pred[Hid].M[index].m[r3id][2] * smR[lvcid].z;
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].m[0][r3id] * smR[lvcid].x
                + Pred[Hid].M[index].m[1][r3id] * smR[lvcid].y
                + Pred[Hid].M[index].m[2][r3id] * smR[lvcid].z;
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}

__global__ void _schwarzLocalXSym4(const __GEIGEN__::MasMatrixT* Pred,
                                   const Precision_T3*           mR,
                                   Precision_T3*                 mZ,
                                   int                           number)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    //int r3id = MRid % 3;

    int    lvrid = vrid % BANKSIZE;
    int    lvcid = vcid % BANKSIZE;
    double rdata = 0;

    if(lvcid >= lvrid)
    {
        //int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata = Pred[Hid].m[MRid][MCid * 3] * mR[vcid].x
                + Pred[Hid].m[MRid][MCid * 3 + 1] * mR[vcid].y
                + Pred[Hid].m[MRid][MCid * 3 + 2] * mR[vcid].z;
    }
    else
    {
        rdata = Pred[Hid].m[MCid * 3][MRid] * mR[vcid].x
                + Pred[Hid].m[MCid * 3 + 1][MRid] * mR[vcid].y
                + Pred[Hid].m[MCid * 3 + 2][MRid] * mR[vcid].z;

        //rdata = Pred[Hid].m[MRid][MCid * 3] * mR[vcid].x
        //        + Pred[Hid].m[MRid][MCid * 3 + 1] * mR[vcid].y
        //        + Pred[Hid].m[MRid][MCid * 3 + 2] * mR[vcid].z;
    }
    //__syncthreads();
    //__threadfence();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    //unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    //mark                  = __brev(mark);
    //int          clzlen   = __clz(mark << (warpId + 1));
    //unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        double tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        //if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}


__global__ void _schwarzLocalXSym5(const __GEIGEN__::MasMatrixT* P96,
                                   const Precision_T3*           mR,
                                   Precision_T3*                 mZ,
                                   int                           number)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3);

    int matId    = idx / hessianSize;
    int locMatId = threadIdx.x / hessianSize;
    int laneId   = idx % hessianSize;
    //int vcid   = matId * BANKSIZE;
    int                    vrid = laneId / 3 + matId * BANKSIZE;
    __shared__ Precision_T vecValue[3][BANKSIZE * 3];


    vecValue[locMatId][laneId] = (*(&(mR[vrid].x) + laneId % 3));

    __syncthreads();
    __threadfence();

    for(int i = 0; i < hessianSize; i++)
    {
        int colId = (laneId + i) % hessianSize;
        //int         vcid   = matId * BANKSIZE + colId / 3;
        //int         axisId = colId % 3;
        //Precision_T rdata = P96[matId].m[laneId][colId] * (*(&(mR[vcid].x) + axisId));
        Precision_T rdata = 0;
        if(laneId + i < hessianSize)
        {
            rdata = P96[matId].m[laneId][colId] * vecValue[locMatId][colId];
        }
        else
        {
            rdata = P96[matId].m[colId][laneId] * vecValue[locMatId][colId];
        }
        atomicAdd((&(mZ[vrid].x) + laneId % 3), rdata);
    }
}

__global__ void _buildCollisionConnection(unsigned int*     _pConnect,
                                          const int*        _pCoarseSpaceTable,
                                          const const int4* _collisionPair,
                                          int               level,
                                          int               node_offset,
                                          int               vertNum,
                                          int               number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collitionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }
        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                cpVid[i] = _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
        }
        else
        {
            for(int i = 0; i < 4; i++)
                cpVid[i] = collitionPairStartId[i];
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId)
                {
                    continue;
                }
                if(myId / BANKSIZE == otId / BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }

        for(int i = 0; i < 4; i++)
            atomicOr(_pConnect + cpVid[i], connMsk[i]);
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] = collitionPairStartId[i];
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 4; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        cpVid[i] = collitionPairStartId[i];
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 2; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] = collitionPairStartId[i];
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 4; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        cpVid[i] = collitionPairStartId[i];
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 3; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    cpVid[i] =
                        _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    cpVid[i] = collitionPairStartId[i];
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId)
                    {
                        continue;
                    }
                    if(myId / BANKSIZE == otId / BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % BANKSIZE));
                        connMsk[j] |= (1U << (myId % BANKSIZE));
                    }
                }
            }

            for(int i = 0; i < 4; i++)
                atomicOr(_pConnect + cpVid[i], connMsk[i]);
        }
    }
}


__global__ void _buildCollisionConnection_new(unsigned int* _pConnect,
                                              const int*    _pCoarseSpaceTable,
                                              const const int4* _collisionPair,
                                              const int* _real_map_partId,
                                              int        level,
                                              int        node_offset,
                                              int        vertNum,
                                              int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collitionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }

        for(int i = 0; i < 4; i++)
            collitionPairStartId[i] -= node_offset;

        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] =
                        _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                else
                    cpVid[i] = -1;
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                else
                    cpVid[i] = -1;
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId || myId < 0 || otId < 0)
                {
                    continue;
                }
                if(myId / BANKSIZE == otId / BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(cpVid[i] >= 0)
                {
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
                }
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                {
                    atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                }
        }
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];

                for(int i = 0; i < 2; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;
                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                for(int i = 0; i < 3; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            for(int i = 0; i < 4; i++)
                collitionPairStartId[i] -= node_offset;
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                    else
                        cpVid[i] = -1;
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                    else
                        cpVid[i] = -1;
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId || myId < 0 || otId < 0)
                    {
                        continue;
                    }
                    if(myId / BANKSIZE == otId / BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % BANKSIZE));
                        connMsk[j] |= (1U << (myId % BANKSIZE));
                    }
                }
            }

            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(cpVid[i] >= 0)
                    {
                        atomicOr(_pConnect + cpVid[i], connMsk[i]);
                    }
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                    {
                        atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                    }
            }
        }
    }
}


void MASPreconditioner::BuildConnectMaskL0()
{

    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                             d_neighborNum,
                                             d_neighborList,
                                             d_fineConnectMask,
                                             d_partId_map_real,
                                             d_real_map_partId,
                                             number);
#else
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0<<<numBlocks, blockSize>>>(
        d_neighborStart, d_neighborNum, d_neighborList, d_fineConnectMask, number);
#endif
}

void MASPreconditioner::PreparePrefixSumL0()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0_new<<<numBlocks, blockSize>>>(
        d_prefixOriginal, d_fineConnectMask, d_partId_map_real, number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0<<<numBlocks, blockSize>>>(d_prefixOriginal, d_fineConnectMask, number);
#endif
}

void MASPreconditioner::BuildLevel1()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
    _buildLevel1_new<<<numBlocks, blockSize>>>(d_levelSize,
                                               d_coarseSpaceTables,
                                               d_goingNext,
                                               d_fineConnectMask,
                                               d_prefixSumOriginal,
                                               d_prefixOriginal,
                                               d_partId_map_real,
                                               number);
#else
    int number    = totalNodes;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
    _buildLevel1<<<numBlocks, blockSize>>>(d_levelSize,
                                           d_coarseSpaceTables,
                                           d_goingNext,
                                           d_fineConnectMask,
                                           d_prefixSumOriginal,
                                           d_prefixOriginal,
                                           number);
#endif
}

void MASPreconditioner::BuildConnectMaskLx(int level)
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                                      d_neighborNum,
                                                      d_neighborList,
                                                      d_coarseSpaceTables,
                                                      d_nextConnectMask,
                                                      d_fineConnectMask,
                                                      level,
                                                      d_partId_map_real,
                                                      totalNodes,
                                                      number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx<<<numBlocks, blockSize>>>(d_neighborStart,
                                                  d_neighborNum,
                                                  d_neighborList,
                                                  d_coarseSpaceTables,
                                                  d_nextConnectMask,
                                                  d_fineConnectMask,
                                                  level,
                                                  number);
#endif
}

void MASPreconditioner::NextLevelCluster(int level)
{
    int number    = h_clevelSize.x;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _nextLevelCluster<<<numBlocks, blockSize>>>(d_nextConnectMask, d_nextPrefix, number);
}

void MASPreconditioner::ComputeNextLevel(int level)
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _computeNextLevel<<<numBlocks, blockSize>>>(
        d_coarseSpaceTables, d_nextConnectMask, level, number);
}

void MASPreconditioner::PrefixSumLx(int level)
{
    int number     = h_clevelSize.x;
    if(number < 1)
        return;
    int levelBegin = h_clevelSize.y;
    int blockSize  = BANKSIZE * BANKSIZE;
    int numBlocks  = (number + blockSize - 1) / blockSize;

    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(d_nextPrefix),
                           thrust::device_ptr<unsigned int>(d_nextPrefix) + warpNum,
                           thrust::device_ptr<unsigned int>(d_nextPrefixSum));

    _prefixSumLx<<<numBlocks, blockSize>>>(
        d_levelSize, d_nextPrefix, d_nextPrefixSum, d_nextConnectMask, d_goingNext, level, levelBegin, number);
}

void MASPreconditioner::AggregationKernel()
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _aggregationKernel<<<numBlocks, blockSize>>>(
        d_denseLevel, d_coarseTable, d_goingNext, levelnum, number);
}


void MASPreconditioner::computeNumLevels(int vertNum)
{
    int totalSz = 0;
    int nLevel  = 1;
    int levelSz = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    totalSz += levelSz;

    while(levelSz > BANKSIZE)
    {
        levelSz /= BANKSIZE;

        nLevel++;
        levelSz = (levelSz + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
        totalSz += levelSz;
    }
    nLevel   = nLevel + 1;
    levelnum = nLevel > 6 ? 6 : nLevel;
    printf("level num:  %d\n", levelnum);
    //totalSize = totalSz * SizeRatio;
}

void MASPreconditioner::BuildCollisionConnection(unsigned int* connectionMsk,
                                                 int*          coarseTableSpace,
                                                 int           level,
                                                 int           cpNum)
{
    int number    = cpNum;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    _buildCollisionConnection_new<<<numBlocks, blockSize>>>(connectionMsk,
                                                            coarseTableSpace,
                                                            _collisonPairs,
                                                            d_real_map_partId,
                                                            level,
                                                            collision_node_Offset,
                                                            totalNodes,
                                                            number);

#else
    _buildCollisionConnection<<<numBlocks, blockSize>>>(
        connectionMsk, coarseTableSpace, _collisonPairs, level, collision_node_Offset, totalNodes, number);

#endif
}
#include <fstream>
int MASPreconditioner::ReorderRealtime(int cpNum)
{
    CUDA_SAFE_CALL(cudaMemset(d_levelSize, 0, levelnum * sizeof(int2)));


    BuildConnectMaskL0();


    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(cpNum)
        BuildCollisionConnection(d_fineConnectMask, nullptr, -1, cpNum);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    PreparePrefixSumL0();

    //vector<unsigned int> h_fineCMsk(totalNodes);
    //   vector<unsigned int> h_prefix(totalNodes);
    //   CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(),
    //                             d_fineConnectMask,
    //                             totalNodes * sizeof(unsigned int),
    //                             cudaMemcpyDeviceToHost));

    //    CUDA_SAFE_CALL(cudaMemcpy(h_prefix.data(),
    //                             d_prefixOriginal,
    //                             totalNodes * sizeof(unsigned int),
    //                             cudaMemcpyDeviceToHost));

    //for (int i = 0; i < totalNodes; i++) {
    //	/*char s[40];
    //	itoa(h_fineCMsk[i], s, 2);
    //	printf("%s\n", s);*/
    //       cout << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << "-----";
    //       cout << /*std::bitset<32>*/ (h_prefix[i]) << endl;
    //}

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    BuildLevel1();

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(int level = 1; level < levelnum; level++)
    {
        CUDA_SAFE_CALL(cudaMemset(d_nextConnectMask, 0, totalNodes * sizeof(int)));

        BuildConnectMaskLx(level);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        if(cpNum)
            BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum);


        CUDA_SAFE_CALL(cudaDeviceSynchronize());


        CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + level, sizeof(int2), cudaMemcpyDeviceToHost));

        //cout << "hello:    " << h_clevelSize.x << endl;

        NextLevelCluster(level);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        if(level == 50)
        {

            //         vector<unsigned int> h_fineCMsk(totalNodes);
            //         vector<unsigned int> mapid(totalMapNodes);
            //         CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(),
            //                                   d_nextConnectMask,
            //                                   totalNodes * sizeof(unsigned int),
            //                                   cudaMemcpyDeviceToHost));

            //         CUDA_SAFE_CALL(cudaMemcpy(mapid.data(),
            //                                   d_partId_map_real,
            //                                   totalMapNodes * sizeof(unsigned int),
            //                                   cudaMemcpyDeviceToHost));

            //         ofstream out("bitcheck.txt");
            //         for(int i = 0; i < h_clevelSize.x; i++)
            //         {
            //             out << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << endl;
            //             int index = mapid[i];
            //             /*char s[40];
            //itoa(h_fineCMsk[i], s, 2);
            //printf("%s\n", s);*/
            //             //if(index >= 0)
            //             //{
            //             //out << bitset<sizeof(h_fineCMsk[index]) * 8>(h_fineCMsk[index]) << endl;
            //             //}
            //             //else
            //             //{
            //             //    out << bitset<32>(0) << endl;
            //             //}
            //             if(i % 16 == 15)
            //             {
            //                 out << "next warp\n\n";
            //             }
            //             //cout << h_fineCMsk[i] << endl;
            //         }
            //         out.close();
            //system("pause");
        }
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        PrefixSumLx(level);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        ComputeNextLevel(level);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + levelnum, sizeof(int2), cudaMemcpyDeviceToHost));

    totalNumberClusters = h_clevelSize.y;

    AggregationKernel();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return totalNumberClusters;

    //vector<unsigned int> h_fineCMsk(totalNumberClusters);
    //CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(), d_goingNext, totalNumberClusters * sizeof(unsigned int), cudaMemcpyDeviceToHost));


    //for (int i = 0; i < totalNumberClusters; i++) {
    //	/*char s[40];
    //	itoa(h_fineCMsk[i], s, 2);
    //	printf("%s\n", s);*/
    //	//cout << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << endl;
    //	cout << i << "    " << h_fineCMsk[i] << endl;
    //}
}

#include <fstream>

void MASPreconditioner::PrepareHessian(const BHessian& BH, const double* masses)
{
    cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);


    //int number = totalNodes;
#ifdef GROUP
    int number = totalMapNodes;
#else
    int number = totalNodes;
#endif
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //cout << totalSize / 32 << endl;
    cudaEventRecord(start);
#ifdef SYME
#ifdef GROUP

    __setSymMassMat_P_new<<<numBlocks, blockSize>>>(
        masses, d_goingNext, d_partId_map_real, d_inverseMatMas, levelnum, number);
#else
    __setSymMassMat_P<<<numBlocks, blockSize>>>(
        masses, d_goingNext, d_inverseMatMas, levelnum, number);
#endif

#else

#ifdef GROUP

    __setMassMat_P_new<<<numBlocks, blockSize>>>(
        masses, d_goingNext, d_partId_map_real, d_MatMas, levelnum, number);
#else
    __setMassMat_P<<<numBlocks, blockSize>>>(masses, d_goingNext, d_MatMas, levelnum, totalNodes);
#endif
#endif
    cudaEventRecord(end0);


    number = BH.DNum[3] * 144 + BH.DNum[2] * 81 + BH.DNum[1] * 36 + BH.DNum[0] * 9;
    numBlocks = (number + blockSize - 1) / blockSize;

#ifdef SYME


#ifdef GROUP
    _prepareSymHessian_new<<<numBlocks, blockSize>>>(BH.H12x12,
                                                     BH.H9x9,
                                                     BH.H6x6,
                                                     BH.H3x3,
                                                     BH.D4Index,
                                                     BH.D3Index,
                                                     BH.D2Index,
                                                     BH.D1Index,
                                                     d_inverseMatMas,
                                                     BH.DNum[3] * 144,
                                                     BH.DNum[2] * 81,
                                                     BH.DNum[1] * 36,
                                                     BH.DNum[0] * 9,
                                                     d_goingNext,
                                                     d_real_map_partId,
                                                     levelnum);
#else
    _prepareSymHessian<<<numBlocks, blockSize>>>(BH.H12x12,
                                                 BH.H9x9,
                                                 BH.H6x6,
                                                 BH.H3x3,
                                                 BH.D4Index,
                                                 BH.D3Index,
                                                 BH.D2Index,
                                                 BH.D1Index,
                                                 d_inverseMatMas,
                                                 BH.DNum[3] * 144,
                                                 BH.DNum[2] * 81,
                                                 BH.DNum[1] * 36,
                                                 BH.DNum[0] * 9,
                                                 d_goingNext,
                                                 levelnum);
#endif


#else
#ifdef GROUP
    _prepareHessian_new<<<numBlocks, blockSize>>>(BH.H12x12,
                                                  BH.H9x9,
                                                  BH.H6x6,
                                                  BH.H3x3,
                                                  BH.D4Index,
                                                  BH.D3Index,
                                                  BH.D2Index,
                                                  BH.D1Index,
                                                  d_MatMas,
                                                  BH.DNum[3] * 144,
                                                  BH.DNum[2] * 81,
                                                  BH.DNum[1] * 36,
                                                  BH.DNum[0] * 9,
                                                  d_goingNext,
                                                  d_real_map_partId,
                                                  levelnum);
#else
    _prepareHessian<<<numBlocks, blockSize>>>(BH.H12x12,
                                              BH.H9x9,
                                              BH.H6x6,
                                              BH.H3x3,
                                              BH.D4Index,
                                              BH.D3Index,
                                              BH.D2Index,
                                              BH.D1Index,
                                              d_MatMas,
                                              BH.DNum[3] * 144,
                                              BH.DNum[2] * 81,
                                              BH.DNum[1] * 36,
                                              BH.DNum[0] * 9,
                                              d_goingNext,

                                              levelnum);
#endif
#endif
    cudaEventRecord(end1);

    blockSize = 32 * 3;
    number    = totalNumberClusters / BANKSIZE;
    number *= BANKSIZE * 3;
    numBlocks = (number + blockSize - 1) / blockSize;
#ifdef SYME
    __inverse6_P96x96<<<numBlocks, blockSize>>>(d_precondMatMas, d_inverseMatMas, number);
#else
    __inverse4_P96x96<<<numBlocks, blockSize>>>(d_MatMas, d_precondMatMas, number);
#endif
    //__inverse6_P96x96<<<numBlocks, blockSize>>>(d_inverseMatMas, number);
    cudaEventRecord(end2);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    //__GEIGEN__::MasMatrixT h_mat96;
    //CUDA_SAFE_CALL(cudaMemcpy(&h_mat96, d_MatMas + 1, sizeof(__GEIGEN__::MasMatrixT), cudaMemcpyDeviceToHost));

    //ofstream out("inverseS.txt");
    //cout << "matNum: " << totalNumberClusters / BANKSIZE << endl;
    //for(int i = 0; i < 48; i += 1)
    //{
    //    for(int j = 0; j < 16; j += 1)
    //    {
    //        out << h_mat96.m[i][j] << " ";
    //    }
    //    out << endl;
    //    //cout << h_fineCMsk[i] << endl;
    //}


    //exit(0);


    float time0, time1, time2, time3, time4;
    cudaEventElapsedTime(&time0, start, end0);
    cudaEventElapsedTime(&time1, end0, end1);
    cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\ntime0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    (cudaEventDestroy(start));
    (cudaEventDestroy(end0));
    (cudaEventDestroy(end1));
    (cudaEventDestroy(end2));
}


void MASPreconditioner::PrepareHessian_bcoo(muda::CBCOOMatrixView<double, 3> hessian,
                                            int                    offset,
                                            muda::CBufferView<int> indices)
{
    cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);

    cudaEventRecord(start);

#ifdef SYME


#ifdef GROUP

    using namespace muda;
    int tripletNum = indices.size();
    if(true)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                tripletNum,
                [offset           = offset,
                 levelNum         = levelnum,
                 _goingNext       = d_goingNext,
                 _invMatrix       = d_inverseMatMas,
                 _real_map_partId = d_real_map_partId,
                 indices          = indices.viewer().name("indices"),
                 hessian = hessian.viewer().name("hessian")] __device__(int I) mutable
                {
                    int index                              = indices(I);
                    auto&& [vertRid_real, vertCid_real, H] = hessian(index);
                    vertRid_real -= offset;
                    vertCid_real -= offset;
                    int vertCid = _real_map_partId[vertCid_real];
                    int vertRid = _real_map_partId[vertRid_real];
                    int cPid    = vertCid / BANKSIZE;


                    if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                    {
                        if(vertCid >= vertRid)
                        {
                            int bvRid = vertRid % BANKSIZE;
                            int bvCid = vertCid % BANKSIZE;
                            int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
                            for(int i = 0; i < 3; i++)
                            {
                                for(int j = 0; j < 3; j++)
                                {
                                    _invMatrix[cPid].M[index].m[i][j] = H(i, j);
                                }
                            }
                        }
                    }
                    else
                    {
                        int level = 0;
                        while(level < levelNum - 1)
                        {
                            level++;
                            if(level == 1)
                            {
                                vertCid = _goingNext[vertCid_real];
                                vertRid = _goingNext[vertRid_real];
                            }
                            else
                            {
                                vertCid = _goingNext[vertCid];
                                vertRid = _goingNext[vertRid];
                            }
                            cPid = vertCid / BANKSIZE;
                            if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                            {

                                if(vertCid >= vertRid)
                                {
                                    int bvRid = vertRid % BANKSIZE;
                                    int bvCid = vertCid % BANKSIZE;
                                    int index = BANKSIZE * bvRid
                                                - bvRid * (bvRid + 1) / 2 + bvCid;
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            atomicAdd(
                                                &(_invMatrix[cPid].M[index].m[i][j]),
                                                H(i, j));
                                            if(vertCid == vertRid)
                                            {
                                                atomicAdd(
                                                    &(_invMatrix[cPid].M[index].m[i][j]),
                                                    H(j, i));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

        tripletNum    = totalMapNodes * BANKSIZE;
        int threadNum = BANKSIZE * BANKSIZE;
        int blockNum  = (tripletNum + threadNum - 1) / threadNum;

        ParallelFor(blockNum, threadNum)
            .file_line(__FILE__, __LINE__)
            .apply(
                tripletNum,
                [levelNum         = levelnum,
                 _goingNext       = d_goingNext,
                 _invMatrix       = d_inverseMatMas,
                 _partId_map_real = d_partId_map_real,
                 _fineConnectMsk  = d_fineConnectMask,
                 _prefix0 = d_prefixOriginal] __device__(int idx) mutable
                {
                    int HSIZE = (BANKSIZE * BANKSIZE);
                    int Hid   = idx / HSIZE;
                    int LMRid = (idx % HSIZE) / BANKSIZE;
                    int LMCid = (idx % HSIZE) % BANKSIZE;

                    int MRid = Hid * BANKSIZE + LMRid;
                    int MCid = Hid * BANKSIZE + LMCid;

                    int            rdx = _partId_map_real[MRid];
                    int            cdx = _partId_map_real[MCid];
                    __shared__ int prefix;

                    if(threadIdx.x == 0)
                    {
                        prefix = _prefix0[Hid];
                    }
                    __syncthreads();
                    __GEIGEN__::Matrix3x3F mat3;
                    if(LMCid >= LMRid)
                    {
                        int index = BANKSIZE * LMRid - LMRid * (LMRid + 1) / 2 + LMCid;
                        mat3 = _invMatrix[Hid].M[index];
                    }
                    else
                    {
                        int index = BANKSIZE * LMCid - LMCid * (LMCid + 1) / 2 + LMRid;
                        mat3 = __GEIGEN__::__Transpose3x3(_invMatrix[Hid].M[index]);
                    }

                    if((rdx >= 0) && (cdx >= 0))
                    {
                        if(prefix == 1)
                        {
                            int warpId = threadIdx.x & 0x1f;
                            bool bBoundary = (warpId == 0) || (rdx < 0) || (cdx < 0);
                            unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
                            mark = __brev(mark);
                            int clzlen = __clz(mark << (warpId + 1));
                            unsigned int interval = std::min(clzlen, 31 - warpId);
                            for(int iter = 1; iter < 32; iter <<= 1)
                            {
                                __GEIGEN__::Matrix3x3F matTemp;
                                for(int i = 0; i < 3; i++)
                                {
                                    for(int j = 0; j < 3; j++)
                                    {
                                        matTemp.m[i][j] =
                                            __shfl_down_sync(0xffffffff, mat3.m[i][j], iter);
                                    }
                                }
                                if(interval >= iter)
                                {
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            mat3.m[i][j] += matTemp.m[i][j];
                                        }
                                    }
                                }
                            }
                            int level = 0;
                            if(bBoundary)
                            {
                                int nextId = _goingNext[rdx];
                                while(level < levelNum - 1)
                                {
                                    level++;
                                    int cPid  = nextId / BANKSIZE;
                                    int bvRid = nextId % BANKSIZE;
                                    int bvCid = nextId % BANKSIZE;
                                    int index = BANKSIZE * bvRid
                                                - bvRid * (bvRid + 1) / 2 + bvCid;
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            atomicAdd(
                                                &(_invMatrix[cPid].M[index].m[i][j]),
                                                mat3.m[i][j]);
                                        }
                                    }
                                    nextId = _goingNext[nextId];
                                }
                            }
                        }
                        else
                        {
                            int level = 0;
                            while(level < levelNum - 1)
                            {
                                level++;
                                rdx      = _goingNext[rdx];
                                cdx      = _goingNext[cdx];
                                int cPid = cdx / BANKSIZE;
                                if(rdx / BANKSIZE == cdx / BANKSIZE)
                                {

                                    if(cdx >= rdx)
                                    {

                                        int bvRid = rdx % BANKSIZE;
                                        int bvCid = cdx % BANKSIZE;
                                        int index = BANKSIZE * bvRid
                                                    - bvRid * (bvRid + 1) / 2 + bvCid;


                                        for(int i = 0; i < 3; i++)
                                        {
                                            for(int j = 0; j < 3; j++)
                                            {
                                                atomicAdd(
                                                    &(_invMatrix[cPid].M[index].m[i][j]),
                                                    mat3.m[i][j]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
    }
    else
    {

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                tripletNum * 2,
                [tripletNum       = tripletNum,
                 levelNum         = levelnum,
                 _goingNext       = d_goingNext,
                 _invMatrix       = d_inverseMatMas,
                 _real_map_partId = d_real_map_partId,
                 indices          = indices.viewer().name("indices"),
                 hessian = hessian.viewer().name("hessian")] __device__(int I) mutable
                {
                    if(I < tripletNum)
                    {

                        int index                              = indices(I);
                        auto&& [vertRid_real, vertCid_real, H] = hessian(index);

                        int vertCid = _real_map_partId[vertCid_real];
                        int vertRid = _real_map_partId[vertRid_real];
                        int cPid    = vertCid / BANKSIZE;

                        int level = 0;
                        //printf("vertCid:   %d\n", vertCid);
                        while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
                        {
                            level++;
                            if(level == 1)
                            {
                                vertCid = _goingNext[vertCid_real];
                                vertRid = _goingNext[vertRid_real];
                            }
                            else
                            {
                                vertCid = _goingNext[vertCid];
                                vertRid = _goingNext[vertRid];
                            }
                            cPid = vertCid / BANKSIZE;
                        }
                        if(level >= levelNum)
                        {
                            return;
                        }
                        //int cPid = vertCid / 32;

                        if(vertCid >= vertRid)
                        {
                            int bvRid = vertRid % BANKSIZE;
                            int bvCid = vertCid % BANKSIZE;
                            int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
                            for(int i = 0; i < 3; i++)
                            {
                                for(int j = 0; j < 3; j++)
                                {
                                    atomicAdd(&(_invMatrix[cPid].M[index].m[i][j]),
                                              H(i, j));
                                }
                            }
                        }


                        while(level < levelNum - 1)
                        {
                            level++;
                            if(level == 1)
                            {
                                vertCid = _goingNext[vertCid_real];
                                vertRid = _goingNext[vertRid_real];
                            }
                            else
                            {
                                vertCid = _goingNext[vertCid];
                                vertRid = _goingNext[vertRid];
                            }
                            cPid = vertCid / BANKSIZE;
                            if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                            {

                                if(vertCid >= vertRid)
                                {
                                    int bvRid = vertRid % BANKSIZE;
                                    int bvCid = vertCid % BANKSIZE;
                                    int index = BANKSIZE * bvRid
                                                - bvRid * (bvRid + 1) / 2 + bvCid;
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            atomicAdd(
                                                &(_invMatrix[cPid].M[index].m[i][j]),
                                                H(i, j));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        int index = indices(I % tripletNum);
                        auto&& [vertCid_real, vertRid_real, H] = hessian(index);
                        if(vertCid_real != vertRid_real)
                        {
                            int vertCid = _real_map_partId[vertCid_real];
                            int vertRid = _real_map_partId[vertRid_real];
                            int cPid    = vertCid / BANKSIZE;

                            int level = 0;
                            //printf("vertCid:   %d\n", vertCid);
                            while(vertCid / BANKSIZE != vertRid / BANKSIZE && level < levelNum)
                            {
                                level++;
                                if(level == 1)
                                {
                                    vertCid = _goingNext[vertCid_real];
                                    vertRid = _goingNext[vertRid_real];
                                }
                                else
                                {
                                    vertCid = _goingNext[vertCid];
                                    vertRid = _goingNext[vertRid];
                                }
                                cPid = vertCid / BANKSIZE;
                            }
                            if(level >= levelNum)
                            {
                                return;
                            }
                            //int cPid = vertCid / 32;

                            if(vertCid >= vertRid)
                            {
                                int bvRid = vertRid % BANKSIZE;
                                int bvCid = vertCid % BANKSIZE;
                                int index = BANKSIZE * bvRid
                                            - bvRid * (bvRid + 1) / 2 + bvCid;
                                for(int i = 0; i < 3; i++)
                                {
                                    for(int j = 0; j < 3; j++)
                                    {
                                        atomicAdd(&(_invMatrix[cPid].M[index].m[i][j]),
                                                  H(j, i));
                                    }
                                }
                            }
                            while(level < levelNum - 1)
                            {
                                level++;
                                if(level == 1)
                                {
                                    vertCid = _goingNext[vertCid_real];
                                    vertRid = _goingNext[vertRid_real];
                                }
                                else
                                {
                                    vertCid = _goingNext[vertCid];
                                    vertRid = _goingNext[vertRid];
                                }
                                cPid = vertCid / BANKSIZE;
                                if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                                {
                                    if(vertCid >= vertRid)
                                    {
                                        int bvRid = vertRid % BANKSIZE;
                                        int bvCid = vertCid % BANKSIZE;
                                        int index = BANKSIZE * bvRid
                                                    - bvRid * (bvRid + 1) / 2 + bvCid;
                                        for(int i = 0; i < 3; i++)
                                        {
                                            for(int j = 0; j < 3; j++)
                                            {
                                                atomicAdd(
                                                    &(_invMatrix[cPid].M[index].m[i][j]),
                                                    H(j, i));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
    }

#endif


#else
#ifdef GROUP
    //_prepareHessian_new<<<numBlocks, blockSize>>>(BH.H12x12,
    //                                              BH.H9x9,
    //                                              BH.H6x6,
    //                                              BH.H3x3,
    //                                              BH.D4Index,
    //                                              BH.D3Index,
    //                                              BH.D2Index,
    //                                              BH.D1Index,
    //                                              d_MatMas,
    //                                              BH.DNum[3] * 144,
    //                                              BH.DNum[2] * 81,
    //                                              BH.DNum[1] * 36,
    //                                              BH.DNum[0] * 9,
    //                                              d_goingNext,
    //                                              d_real_map_partId,
    //                                              levelnum);
#else
    //_prepareHessian<<<numBlocks, blockSize>>>(BH.H12x12,
    //                                          BH.H9x9,
    //                                          BH.H6x6,
    //                                          BH.H3x3,
    //                                          BH.D4Index,
    //                                          BH.D3Index,
    //                                          BH.D2Index,
    //                                          BH.D1Index,
    //                                          d_MatMas,
    //                                          BH.DNum[3] * 144,
    //                                          BH.DNum[2] * 81,
    //                                          BH.DNum[1] * 36,
    //                                          BH.DNum[0] * 9,
    //                                          d_goingNext,

    //                                          levelnum);
#endif
#endif
    cudaEventRecord(end0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int blockSize2 = 32 * 3;
    //int number2    = totalNumberClusters / BANKSIZE;
    int number2    = totalNumberClusters * 3;
    if(number2 < 1)
        return;
    int numBlocks2 = (number2 + blockSize2 - 1) / blockSize2;
#ifdef SYME
    __inverse6_P96x96<<<numBlocks2, blockSize2>>>(d_precondMatMas, d_inverseMatMas, number2);
#else
    __inverse4_P96x96<<<numBlocks2, blockSize2>>>(d_MatMas, d_precondMatMas, number2);
#endif
    //__inverse6_P96x96<<<numBlocks, blockSize>>>(d_inverseMatMas, number);
    cudaEventRecord(end1);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    float time0, time1, time2, time3, time4;
    cudaEventElapsedTime(&time0, start, end0);
    cudaEventElapsedTime(&time1, end0, end1);
    //cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\ntime0 = %f,  time1 = %f\n\n", time0, time1);

    (cudaEventDestroy(start));
    (cudaEventDestroy(end0));
    (cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}


void MASPreconditioner::BuildMultiLevelR(const double3* R)
{


#ifdef GROUP
    int number = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized_new<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_prefixOriginal, d_fineConnectMask, d_partId_map_real, levelnum, number);


    //__buildMultiLevelR_new<<<numBlocks, blockSize>>>(
    //    R, d_multiLevelR, d_goingNext, d_real_map_partId, levelnum, number);

#else
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_fineConnectMask, levelnum, number);
#endif
}

void MASPreconditioner::SchwarzLocalXSym()
{
    //int matNum    = totalNumberClusters / BANKSIZE;
    int number    = totalNumberClusters * BANKSIZE * 3;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym3<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void MASPreconditioner::CollectFinalZ(double3* Z)
{
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    __collectFinalZ_new<<<numBlocks, blockSize>>>(
        Z, d_multiLevelZ, d_coarseTable, d_real_map_partId, levelnum, number);
#else
    __collectFinalZ<<<numBlocks, blockSize>>>(Z, d_multiLevelZ, d_coarseTable, levelnum, number);
#endif

    //vector<int4> h_r(totalNodes);
    //CUDA_SAFE_CALL(cudaMemcpy(h_r.data(), d_coarseTable, totalNodes * sizeof(int4), cudaMemcpyDeviceToHost));

    //for (int i = 0; i < totalNodes; i++) {

    //	cout << h_r[i].x << " " << h_r[i].y << " " << h_r[i].z<<"  "<<h_r[i].w << endl;
    //	//cout << h_fineCMsk[i] << endl;
    //}
    //exit(0);
}

void MASPreconditioner::setPreconditioner(const BHessian& BH, const double* masses, int cpNum)
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ReorderRealtime(cpNum);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef SYME

    CUDA_SAFE_CALL(cudaMemset(
        d_inverseMatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMemset(
        d_MatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixT)));
#endif
    PrepareHessian(BH, masses);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void MASPreconditioner::setPreconditioner_bcoo(muda::CBCOOMatrixView<double, 3> hessian,
                                               muda::CBufferView<int> indices,
                                               int                    offset,
                                               int                    cpNum)
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ReorderRealtime(cpNum);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef SYME

    CUDA_SAFE_CALL(cudaMemset(
        d_inverseMatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMemset(
        d_MatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixT)));
#endif
    PrepareHessian_bcoo(hessian, offset, indices);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


void MASPreconditioner::preconditioning(const double3* R, double3* Z)
{
    if(totalNodes < 1)
        return;
#ifdef GROUP
    CUDA_SAFE_CALL(cudaMemset(d_multiLevelR + totalMapNodes,
                              0,
                              (totalNumberClusters - totalMapNodes) * sizeof(Precision_T3)));
#else
    CUDA_SAFE_CALL(cudaMemset(d_multiLevelR + totalNodes,
                              0,
                              (totalNumberClusters - totalNodes) * sizeof(Precision_T3)));
#endif
    CUDA_SAFE_CALL(cudaMemset(d_multiLevelZ, 0, (totalNumberClusters) * sizeof(Precision_T3)));

    cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    cudaEventRecord(start);
    BuildMultiLevelR(R);
    cudaEventRecord(end0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    SchwarzLocalXSym();
    cudaEventRecord(end1);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CollectFinalZ(Z);
    cudaEventRecord(end2);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    float time0, time1, time2, time3, time4;
    cudaEventElapsedTime(&time0, start, end0);
    cudaEventElapsedTime(&time1, end0, end1);
    cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\npreconditioning  time0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    (cudaEventDestroy(start));
    (cudaEventDestroy(end0));
    (cudaEventDestroy(end1));
    (cudaEventDestroy(end2));
}

void MASPreconditioner::initPreconditioner_Neighbor(int vertNum,
                                                    int mCollision_node_offset,
                                                    int totalNeighborNum,
                                                    int4* m_collisonPairs,
                                                    int   partMapSize)
{
    //bankSize = 32;
    if(vertNum < 1)
        return;
    int maxNodes = partMapSize > vertNum ? partMapSize : vertNum;
    computeNumLevels(maxNodes);
    totalMapNodes         = partMapSize;
    collision_node_Offset = mCollision_node_offset;
    _collisonPairs        = m_collisonPairs;
    totalNodes            = vertNum;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_denseLevel, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_real_map_partId, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseTable, vertNum * sizeof(__GEIGEN__::itable)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseSpaceTables,
                              vertNum * levelnum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_levelSize, (levelnum + 1) * sizeof(int2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_goingNext,
                              vertNum * levelnum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefix, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefixSum, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixSumOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fineConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborList, totalNeighborNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStartTemp, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNum, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborListInit, totalNeighborNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNumInit, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_partId_map_real, partMapSize * sizeof(int)));
}

void MASPreconditioner::initPreconditioner_Matrix()
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

    int totalCluster = ReorderRealtime(0) * 1.05;
#ifdef SYME
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_inverseMatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_MatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixT)));
#endif

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_precondMatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymf)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelR, totalCluster * sizeof(Precision_T3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelZ, totalCluster * sizeof(Precision_T3)));
}

void MASPreconditioner::FreeMAS()
{
    CUDA_SAFE_CALL(cudaFree(d_denseLevel));
    CUDA_SAFE_CALL(cudaFree(d_coarseSpaceTables));
    CUDA_SAFE_CALL(cudaFree(d_levelSize));
    CUDA_SAFE_CALL(cudaFree(d_goingNext));
    CUDA_SAFE_CALL(cudaFree(d_prefixOriginal));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefix));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefixSum));
    CUDA_SAFE_CALL(cudaFree(d_prefixSumOriginal));
    CUDA_SAFE_CALL(cudaFree(d_fineConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_nextConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_neighborList));
    CUDA_SAFE_CALL(cudaFree(d_neighborListInit));
    CUDA_SAFE_CALL(cudaFree(d_neighborStart));
    CUDA_SAFE_CALL(cudaFree(d_neighborStartTemp));
    CUDA_SAFE_CALL(cudaFree(d_neighborNum));
    CUDA_SAFE_CALL(cudaFree(d_neighborNumInit));
    CUDA_SAFE_CALL(cudaFree(d_partId_map_real));
    CUDA_SAFE_CALL(cudaFree(d_real_map_partId));
#ifdef SYME
    CUDA_SAFE_CALL(cudaFree(d_inverseMatMas));
#else
    CUDA_SAFE_CALL(cudaFree(d_MatMas));
#endif

    CUDA_SAFE_CALL(cudaFree(d_precondMatMas));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelR));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelZ));
}
