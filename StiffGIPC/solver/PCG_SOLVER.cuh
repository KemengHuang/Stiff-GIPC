//
// PCG_SOLVER.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _PCG_SOLVER_CUH_
#define _PCG_SOLVER_CUH_
#include <cuda_runtime.h>
#include <cstddef>
#include "MASPreconditioner.cuh"

class PCG_Data
{
  public:
    cudatool::DeviceBuffer<double>  squeue;
    cudatool::DeviceBuffer<double>  reduction_scalar;
    cudatool::DeviceBuffer<double2> reduction_pair;
    cudatool::DeviceBuffer<double3> dx;
    MASPreconditioner               MP;

    int P_type = 1;

  public:
    void    Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
    double* prepare_reduction_queue(size_t item_count, size_t block_size = 256);
    double* prepare_reduction_scalar();
    double2* prepare_reduction_pair();
    void    FREE_DEVICE_MEM();
};
#endif  // ! _PCG_SOLVER_CUH_
