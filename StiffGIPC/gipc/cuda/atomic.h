#pragma once
#include <cuda_runtime.h>
#include <gipc/cuda/muda_def.h>

namespace gipc
{
namespace cuda
{

MUDA_INLINE MUDA_DEVICE double atomic_add(double* address, double val)
{
    return atomicAdd(address, val);
}

MUDA_INLINE MUDA_DEVICE float atomic_add(float* address, float val)
{
    return atomicAdd(address, val);
}

MUDA_INLINE MUDA_DEVICE int atomic_add(int* address, int val)
{
    return atomicAdd(address, val);
}

MUDA_INLINE MUDA_DEVICE unsigned int atomic_add(unsigned int* address, unsigned int val)
{
    return atomicAdd(address, val);
}

MUDA_INLINE MUDA_DEVICE unsigned long long int atomic_add(unsigned long long int* address,
                                                          unsigned long long int val)
{
    return atomicAdd(address, val);
}

}  // namespace cuda
}  // namespace gipc
