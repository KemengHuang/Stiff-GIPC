#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CT_HOST __host__
#define CT_DEVICE __device__
#define CT_GLOBAL __global__
#define CT_CONSTANT __constant__
#define CT_SHARED __shared__
#define CT_MANAGED __managed__

#ifdef __CUDA_ARCH__
#define CT_GENERIC __host__ __device__
#else
#define CT_GENERIC
#endif

#define CT_NODISCARD [[nodiscard]]
#define CT_DEPRECATED [[deprecated]]
#define CT_FALLTHROUGH [[fallthrough]]
#define CT_MAYBE_UNUSED [[maybe_unused]]
#define CT_NORETURN [[noreturn]]

#define CT_NOEXCEPT noexcept
#define CT_INLINE inline
#define CT_CONSTEXPR constexpr

#define CT_REQUIRES(...)

#ifndef CT_ASSERT
#define CT_ASSERT(cond, ...) ((void)0)
#endif
