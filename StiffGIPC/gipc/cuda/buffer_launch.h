#pragma once

#include "muda_def.h"
#include "buffer.h"
#include "parallel_for.h"

namespace gipc {
namespace cuda {

namespace details {

template <typename T>
__global__ void fill_buffer_kernel(T* data, size_t size, T value)
{
    int total = gridDim.x * blockDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += total)
        data[i] = value;
}

template <typename T>
__global__ void fill_var_kernel(T* data, T value)
{
    *data = value;
}

}  // namespace details

class BufferLaunch : public LaunchBase<BufferLaunch>
{
    int m_grid_dim  = 0;
    int m_block_dim = -1;

  public:
    MUDA_HOST BufferLaunch(cudaStream_t s = nullptr) MUDA_NOEXCEPT : LaunchBase(s) {}

    MUDA_HOST BufferLaunch(int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_block_dim(block_dim)
    {
    }

    MUDA_HOST BufferLaunch(int grid_dim, int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_grid_dim(grid_dim),
          m_block_dim(block_dim)
    {
    }

    template <typename T>
    MUDA_HOST BufferLaunch& fill(BufferView<T> buffer, const T& val)
    {
        if(buffer.size() == 0)
            return *this;
        int block_dim = (m_block_dim > 0) ? m_block_dim : 256;
        int grid_dim  = (m_grid_dim > 0) ? m_grid_dim : (buffer.size() + block_dim - 1) / block_dim;
        details::fill_buffer_kernel<<<grid_dim, block_dim, 0, m_stream>>>(
            buffer.data(), buffer.size(), val);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& fill(VarView<T> var, const T& val)
    {
        cudaMemcpy(var.data(), &val, sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& fill(CVarView<T> var, const T& val) = delete;

    template <typename T>
    MUDA_HOST BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size)
    {
        buffer.resize(size);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& reserve(DeviceBuffer<T>& buffer, size_t capacity)
    {
        buffer.reserve(capacity);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& clear(DeviceBuffer<T>& buffer)
    {
        buffer.clear();
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& alloc(DeviceBuffer<T>& buffer, size_t n)
    {
        buffer.resize(n);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& free(DeviceBuffer<T>& buffer)
    {
        buffer.clear();
        buffer.shrink_to_fit();
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& copy(BufferView<T> dst, CBufferView<T> src)
    {
        size_t n = std::min(dst.size(), src.size());
        if(n > 0)
            cudaMemcpy(dst.data(), src.data(), n * sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& copy(VarView<T> dst, CVarView<T> src)
    {
        cudaMemcpy(dst.data(), src.data(), sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& copy(BufferView<T> dst, const T* src)
    {
        if(dst.size() > 0)
            cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    template <typename T>
    MUDA_HOST BufferLaunch& copy(T* dst, CBufferView<T> src)
    {
        if(src.size() > 0)
            cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
        return *this;
    }
};

}  // namespace cuda
}  // namespace gipc
