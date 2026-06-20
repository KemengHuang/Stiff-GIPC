#pragma once

#include "muda_def.h"
#include <cuda_runtime.h>
#include <cinttypes>
#include <type_traits>
#include <utility>

namespace gipc {
namespace cuda {

// Launch base
template <typename Derived>
class LaunchBase
{
  protected:
    cudaStream_t m_stream = nullptr;

  public:
    MUDA_HOST LaunchBase(cudaStream_t stream = nullptr) MUDA_NOEXCEPT : m_stream(stream) {}

    MUDA_HOST Derived& wait() MUDA_NOEXCEPT
    {
        cudaStreamSynchronize(m_stream);
        return static_cast<Derived&>(*this);
    }

    MUDA_GENERIC cudaStream_t stream() const MUDA_NOEXCEPT { return m_stream; }

    MUDA_HOST Derived& kernel_name(const char*) MUDA_NOEXCEPT { return static_cast<Derived&>(*this); }
    MUDA_HOST Derived& kernel_name(const std::string&) MUDA_NOEXCEPT { return static_cast<Derived&>(*this); }
    MUDA_HOST Derived& file_line(const char*, int) MUDA_NOEXCEPT { return static_cast<Derived&>(*this); }
};

namespace details {

template <typename T>
using raw_type_t = std::remove_reference_t<std::remove_cv_t<T>>;

template <typename F>
struct ParallelForCallable
{
    F   callable;
    int count;
    template <typename U>
    MUDA_HOST ParallelForCallable(U&& c, int n) MUDA_NOEXCEPT
        : callable(std::forward<U>(c)),
          count(n)
    {
    }
};

template <typename F>
__global__ void parallel_for_kernel(ParallelForCallable<F> f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < f.count)
        f.callable(i);
}

template <typename F>
__global__ void grid_stride_loop_kernel(ParallelForCallable<F> f)
{
    int total = gridDim.x * blockDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < f.count; i += total)
        f.callable(i);
}

template <typename F>
struct LaunchCallable
{
    F    callable;
    dim3 active_dim;
    template <typename U>
    MUDA_HOST LaunchCallable(U&& c, dim3 d) MUDA_NOEXCEPT
        : callable(std::forward<U>(c)),
          active_dim(d)
    {
    }
};

template <typename F>
__global__ void launch_kernel(LaunchCallable<F> f)
{
    f.callable();
}

template <typename F>
__global__ void launch_kernel_with_coord(LaunchCallable<F> f)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x < f.active_dim.x && y < f.active_dim.y && z < f.active_dim.z)
        f.callable(dim3(x, y, z));
}

}  // namespace details

template <typename T>
struct Tag {};
using Default = void;

class ParallelFor : public LaunchBase<ParallelFor>
{
    int  m_grid_dim   = 0;
    int  m_block_dim  = -1;
    size_t m_shared_mem_size = 0;
    bool m_grid_stride = false;

  public:
    MUDA_HOST ParallelFor(size_t shared_mem_size = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(0),
          m_block_dim(-1),
          m_shared_mem_size(shared_mem_size)
    {
    }

    MUDA_HOST ParallelFor(int blockDim, size_t shared_mem_size = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(0),
          m_block_dim(blockDim),
          m_shared_mem_size(shared_mem_size)
    {
    }

    MUDA_HOST ParallelFor(int gridDim,
                          int blockDim,
                          size_t       shared_mem_size = 0,
                          cudaStream_t stream          = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(shared_mem_size),
          m_grid_stride(true)
    {
    }

    template <typename F>
    MUDA_HOST ParallelFor& apply(int count, F&& f)
    {
        if(count <= 0)
            return *this;
        int block_dim = (m_block_dim > 0) ? m_block_dim : 256;
        int grid_dim  = (m_grid_dim > 0) ? m_grid_dim : (count + block_dim - 1) / block_dim;
        using RawF    = details::raw_type_t<F>;
        details::ParallelForCallable<RawF> callable(std::forward<F>(f), count);
        if(m_grid_stride)
            details::grid_stride_loop_kernel<<<grid_dim, block_dim, m_shared_mem_size, m_stream>>>(callable);
        else
            details::parallel_for_kernel<<<grid_dim, block_dim, m_shared_mem_size, m_stream>>>(callable);
        return *this;
    }

    template <typename F, typename UserTag>
    MUDA_HOST ParallelFor& apply(int count, F&& f, Tag<UserTag>)
    {
        return apply(count, std::forward<F>(f));
    }

    static MUDA_GENERIC int round_up_blocks(int count, int block_dim) MUDA_NOEXCEPT
    {
        return (count + block_dim - 1) / block_dim;
    }
};

class Launch : public LaunchBase<Launch>
{
    dim3 m_grid_dim;
    dim3 m_block_dim;
    size_t m_shared_mem_size;

  public:
    MUDA_HOST Launch(dim3 gridDim, dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    MUDA_HOST Launch(int gridDim       = 1,
                     int blockDim      = 1,
                     size_t       sharedMemSize = 0,
                     cudaStream_t stream        = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(gridDim, 1, 1),
          m_block_dim(blockDim, 1, 1),
          m_shared_mem_size(sharedMemSize)
    {
    }

    MUDA_HOST Launch(dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(0, 0, 0),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    template <typename F>
    MUDA_HOST Launch& apply(F&& f)
    {
        using RawF = details::raw_type_t<F>;
        dim3 total_threads(m_grid_dim.x * m_block_dim.x,
                           m_grid_dim.y * m_block_dim.y,
                           m_grid_dim.z * m_block_dim.z);
        details::LaunchCallable<RawF> callable(std::forward<F>(f), total_threads);
        details::launch_kernel<<<m_grid_dim, m_block_dim, m_shared_mem_size, m_stream>>>(callable);
        return *this;
    }

    template <typename F, typename UserTag>
    MUDA_HOST Launch& apply(F&& f, Tag<UserTag>)
    {
        return apply(std::forward<F>(f));
    }

    template <typename F>
    MUDA_HOST Launch& apply(const dim3& active_dim, F&& f)
    {
        using RawF = details::raw_type_t<F>;
        dim3 grid  = m_grid_dim;
        if(grid.x == 0)
        {
            grid.x = (active_dim.x + m_block_dim.x - 1) / m_block_dim.x;
            grid.y = (active_dim.y + m_block_dim.y - 1) / m_block_dim.y;
            grid.z = (active_dim.z + m_block_dim.z - 1) / m_block_dim.z;
        }
        details::LaunchCallable<RawF> callable(std::forward<F>(f), active_dim);
        details::launch_kernel_with_coord<<<grid, m_block_dim, m_shared_mem_size, m_stream>>>(callable);
        return *this;
    }

    template <typename F, typename UserTag>
    MUDA_HOST Launch& apply(const dim3& active_dim, F&& f, Tag<UserTag>)
    {
        return apply(active_dim, std::forward<F>(f));
    }
};

MUDA_INLINE MUDA_HOST void wait_device()
{
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace gipc
