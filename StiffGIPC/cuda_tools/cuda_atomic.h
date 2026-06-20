#pragma once

#include "cuda_def.h"

namespace cudatool
{

CT_INLINE CT_DEVICE double atomic_add(double* address, double val)
{
    return atomicAdd(address, val);
}

CT_INLINE CT_DEVICE float atomic_add(float* address, float val)
{
    return atomicAdd(address, val);
}

CT_INLINE CT_DEVICE int atomic_add(int* address, int val)
{
    return atomicAdd(address, val);
}

CT_INLINE CT_DEVICE unsigned int atomic_add(unsigned int* address, unsigned int val)
{
    return atomicAdd(address, val);
}

CT_INLINE CT_DEVICE unsigned long long int atomic_add(unsigned long long int* address,
                                                      unsigned long long int val)
{
    return atomicAdd(address, val);
}

}  // namespace cudatool
