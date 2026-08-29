//
// PCG_SOLVER.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "PCG_SOLVER.cuh"
#include "device_launch_parameters.h"
#include "cuda_tools/cuda_tools.h"

void PCG_Data::Malloc_DEVICE_MEM(const int& vertexNum, const int& tetrahedraNum)
{
    (void)tetrahedraNum;
    // Reduction scratch starts empty and grows from the actual launch size.
    squeue.release();
    dx.resize(vertexNum);
}

double* PCG_Data::prepare_reduction_queue(size_t item_count, size_t block_size)
{
    size_t block_count = item_count == 0 ? 0 : (item_count + block_size - 1) / block_size;
    squeue.resize_discard(block_count);
    return squeue.data();
}

void PCG_Data::FREE_DEVICE_MEM()
{
    squeue.release();
    dx.release();

    if(P_type == 1)
    {
        MP.FreeMAS();
    }
}
