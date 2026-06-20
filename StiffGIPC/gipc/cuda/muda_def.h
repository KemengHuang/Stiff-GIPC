#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define MUDA_HOST __host__
#define MUDA_DEVICE __device__
#define MUDA_GLOBAL __global__
#define MUDA_CONSTANT __constant__
#define MUDA_SHARED __shared__
#define MUDA_MANAGED __managed__

#ifdef __CUDA_ARCH__
#define MUDA_GENERIC __host__ __device__
#else
#define MUDA_GENERIC
#endif

#define MUDA_NODISCARD [[nodiscard]]
#define MUDA_DEPRECATED [[deprecated]]
#define MUDA_FALLTHROUGH [[fallthrough]]
#define MUDA_MAYBE_UNUSED [[maybe_unused]]
#define MUDA_NORETURN [[noreturn]]

#define MUDA_NOEXCEPT noexcept
#define MUDA_INLINE inline
#define MUDA_CONSTEXPR constexpr

#define MUDA_REQUIRES(...)

#ifndef MUDA_ASSERT
#define MUDA_ASSERT(cond, ...) ((void)0)
#endif
