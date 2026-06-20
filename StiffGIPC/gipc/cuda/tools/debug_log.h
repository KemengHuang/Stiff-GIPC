#pragma once

#include "../muda_def.h"
#include <cuda_runtime.h>
#include <cassert>

// Host-side CUDA error checker used by project code.
#ifndef __CUDA_ARCH__
#include <cstdio>
#include <stdexcept>
#include <string>
namespace gipc {
namespace cuda {
inline void check(cudaError_t result,
                  const char* const func,
                  const char* const file,
                  int const         line)
{
    if(result != cudaSuccess)
    {
        std::fprintf(stderr,
                     "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                     file,
                     line,
                     static_cast<unsigned int>(result),
                     cudaGetErrorString(result),
                     func);
        cudaDeviceReset();
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result));
    }
}
}  // namespace cuda
}  // namespace gipc
#define checkCudaErrors(val) gipc::cuda::check((val), #val, __FILE__, __LINE__)
#else
#define checkCudaErrors(val) (val)
#endif

// MUDA debug/logging macros - kept as no-ops for source compatibility.
#define MUDA_KERNEL_PRINT(fmt, ...) ((void)0)
#define MUDA_DEBUG_TRAP() ((void)0)
#define MUDA_KERNEL_ASSERT(res, fmt, ...) ((void)0)
#define MUDA_KERNEL_CHECK(res, fmt, ...) ((void)0)
#define MUDA_KERNEL_ERROR(fmt, ...) ((void)0)
#define MUDA_KERNEL_ERROR_WITH_LOCATION(fmt, ...) ((void)0)
#define MUDA_KERNEL_WARN(fmt, ...) ((void)0)
#define MUDA_KERNEL_WARN_WITH_LOCATION(fmt, ...) ((void)0)
#define MUDA_ERROR(fmt, ...) ((void)0)
#define MUDA_ERROR_WITH_LOCATION(fmt, ...) ((void)0)

namespace gipc {
namespace cuda {
class Debug
{
  public:
    static void debug_sync_all(bool) {}
};
}  // namespace cuda
}  // namespace gipc
