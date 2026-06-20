//
// cuda_tool.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang and Jiming Ruan. All rights reserved.
//

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include<vector>
#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
const static int default_threads = 256;
//#define CUDA_KERNEL_CHECK(err)  cuda_kernel_check_(err, __FILE__, __LINE__)

inline unsigned long long LogIte(unsigned long long value)
{
    if(value == 0)
    {
        return 0;
    }
    return 1 + LogIte(value >> 1);
}
inline unsigned long long Log2(unsigned long long value)
{
    value -= 1;
    if(value == 0)
    {
        return 1;
    }
    return LogIte(value);
}


inline void cuda_safe_call_(cudaError err, const char* file_name, const int num_line)
{
    if(cudaSuccess != err)
    {
        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;

        std::abort();
    }
}



template <typename F, typename... Arguments>
void LaunchCudaKernal(int gs, int bs, size_t mem, F f, Arguments... args)
{
    if(gs < 1)
        return;
    if(!mem)
    {
        f<<<gs, bs>>>(args...);
    }
    else
    {
        f<<<gs, bs, mem>>>(args...);
    }
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr << __FILE__ << "[" << __LINE__ << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}

inline void wait_device()
{
    cudaDeviceSynchronize();
}

template <typename F, typename... Arguments>
void LaunchCudaKernal_default(int total, int bs, size_t mem, F f, Arguments... args)
{
    int gs = (total + bs - 1) / bs;
    if(gs < 1)
        return;
    if(!mem)
    {
        f<<<gs, bs>>>(args...);
    }
    else
    {
        f<<<gs, bs, mem>>>(args...);
    }
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr << __FILE__ << "[" << __LINE__ << "]: "
                  << "CUDA Running API error[" << (int)err
                  << "]: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}