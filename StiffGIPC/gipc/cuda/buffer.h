#pragma once

#include "muda_def.h"
#include <cuda_runtime.h>
#include <cinttypes>
#include <vector>
#include <type_traits>
#include <Eigen/Core>

namespace gipc {
namespace cuda {

// Forward declarations
template <typename T> class DeviceBuffer;
template <typename T> class DeviceVar;

template <typename T> class Dense;
template <typename T> class CDense;
template <typename T> class Dense1D;
template <typename T> class CDense1D;

template <bool IsConst, typename T> class BufferViewT;
template <typename T> using BufferView = BufferViewT<false, T>;
template <typename T> using CBufferView = BufferViewT<true, T>;

#ifdef __CUDACC__
namespace details {
template <typename T>
__global__ void buffer_fill_kernel(T* data, size_t size, T value)
{
    int total = gridDim.x * blockDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += total)
        data[i] = value;
}
}  // namespace details
#endif

template <bool IsConst, typename T> class VarViewT;
template <typename T> using VarView = VarViewT<false, T>;
template <typename T> using CVarView = VarViewT<true, T>;

// 0D viewer
template <typename T>
class Dense
{
    T* m_data;

  public:
    MUDA_GENERIC Dense(T* data = nullptr) MUDA_NOEXCEPT : m_data(data) {}
    MUDA_GENERIC T& operator*() const MUDA_NOEXCEPT { return *m_data; }
    MUDA_GENERIC T& operator()() const MUDA_NOEXCEPT { return *m_data; }
    MUDA_GENERIC T* data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC Dense& name(const char*) MUDA_NOEXCEPT { return *this; }
};

template <typename T>
class CDense
{
    const T* m_data;

  public:
    MUDA_GENERIC CDense(const T* data = nullptr) MUDA_NOEXCEPT : m_data(data) {}
    MUDA_GENERIC const T& operator*() const MUDA_NOEXCEPT { return *m_data; }
    MUDA_GENERIC const T& operator()() const MUDA_NOEXCEPT { return *m_data; }
    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC CDense& name(const char*) MUDA_NOEXCEPT { return *this; }
};

// 1D viewer
template <typename T>
class Dense1D
{
    T*     m_data;
    size_t m_size;

  public:
    MUDA_GENERIC Dense1D(T* data = nullptr, size_t size = 0) MUDA_NOEXCEPT
        : m_data(data),
          m_size(size)
    {
    }
    MUDA_GENERIC T& operator()(int i) const MUDA_NOEXCEPT { return m_data[i]; }
    MUDA_GENERIC T& operator[](int i) const MUDA_NOEXCEPT { return m_data[i]; }
    MUDA_GENERIC T*       data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC size_t   size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC Dense1D& name(const char*) MUDA_NOEXCEPT { return *this; }
};

template <typename T>
class CDense1D
{
    const T* m_data;
    size_t   m_size;

  public:
    MUDA_GENERIC CDense1D(const T* data = nullptr, size_t size = 0) MUDA_NOEXCEPT
        : m_data(data),
          m_size(size)
    {
    }
    MUDA_GENERIC const T& operator()(int i) const MUDA_NOEXCEPT { return m_data[i]; }
    MUDA_GENERIC const T& operator[](int i) const MUDA_NOEXCEPT { return m_data[i]; }
    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC size_t   size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC CDense1D& name(const char*) MUDA_NOEXCEPT { return *this; }
};

// Buffer view
template <bool IsConst, typename T>
class BufferViewT
{
    using pointer_type = std::conditional_t<IsConst, const T*, T*>;

  protected:
    pointer_type m_data   = nullptr;
    size_t       m_offset = 0;
    size_t       m_size   = 0;

  public:
    using value_type = T;

    MUDA_GENERIC BufferViewT() MUDA_NOEXCEPT = default;
    MUDA_GENERIC BufferViewT(pointer_type data, size_t size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(0),
          m_size(size)
    {
    }
    MUDA_GENERIC BufferViewT(pointer_type data, size_t offset, size_t size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    template <bool OtherIsConst,
              typename = std::enable_if_t<!OtherIsConst || IsConst>>
    MUDA_GENERIC BufferViewT(const BufferViewT<OtherIsConst, T>& other) MUDA_NOEXCEPT
        : m_data(other.origin_data()),
          m_offset(other.offset()),
          m_size(other.size())
    {
    }

    MUDA_GENERIC pointer_type data() const MUDA_NOEXCEPT { return m_data + m_offset; }
    MUDA_GENERIC pointer_type origin_data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC size_t       size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC size_t       offset() const MUDA_NOEXCEPT { return m_offset; }

    MUDA_GENERIC BufferViewT<IsConst, T> subview(size_t offset, size_t size = ~size_t(0)) const MUDA_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? ((m_size > offset) ? (m_size - offset) : 0) : size;
        return BufferViewT<IsConst, T>(m_data, m_offset + offset, s);
    }

    MUDA_GENERIC Dense1D<std::conditional_t<IsConst, const T, T>> viewer() const MUDA_NOEXCEPT
    {
        return Dense1D<std::conditional_t<IsConst, const T, T>>(data(), m_size);
    }

    MUDA_GENERIC CDense1D<T> cviewer() const MUDA_NOEXCEPT
    {
        return CDense1D<T>(data(), m_size);
    }

    MUDA_GENERIC auto& operator[](size_t i) const MUDA_NOEXCEPT { return data()[i]; }
    MUDA_GENERIC auto& operator*() const MUDA_NOEXCEPT { return *data(); }
    MUDA_GENERIC auto& operator[](int i) const MUDA_NOEXCEPT { return data()[i]; }

#ifdef __CUDACC__
    MUDA_HOST void fill(const T& value) const
    {
        static_assert(!IsConst, "Cannot fill const view");
        if(m_size == 0)
            return;
        int block_dim = 256;
        int grid_dim  = static_cast<int>((m_size + block_dim - 1) / block_dim);
        details::buffer_fill_kernel<T><<<grid_dim, block_dim, 0>>>(data(), m_size, value);
    }
#else
    MUDA_HOST void fill(const T&) const { static_assert(!IsConst, "Cannot fill const view"); }
#endif

    MUDA_HOST void copy_from(const BufferViewT<true, T>& other) const
    {
        static_assert(!IsConst, "Cannot copy into const view");
        size_t n = std::min(m_size, other.size());
        cudaMemcpy(data(), other.data(), n * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    MUDA_HOST void copy_from(const T* host) const
    {
        static_assert(!IsConst, "Cannot copy into const view");
        cudaMemcpy(data(), host, m_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    MUDA_HOST void copy_to(T* host) const
    {
        cudaMemcpy(host, data(), m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    // Iterator-like
    using reference = T&;
    using pointer   = T*;
    MUDA_GENERIC BufferViewT operator+(int i) const MUDA_NOEXCEPT
    {
        return BufferViewT(m_data, m_offset + i, 1);
    }
};

template <typename T>
using BufferView = BufferViewT<false, T>;

template <typename T>
using CBufferView = BufferViewT<true, T>;

// Var view
template <bool IsConst, typename T>
class VarViewT
{
    using pointer_type = std::conditional_t<IsConst, const T*, T*>;

  protected:
    pointer_type m_data = nullptr;

  public:
    using value_type = T;

    MUDA_GENERIC VarViewT() MUDA_NOEXCEPT = default;
    MUDA_GENERIC VarViewT(pointer_type data) MUDA_NOEXCEPT : m_data(data) {}

    MUDA_GENERIC pointer_type data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC Dense<std::conditional_t<IsConst, const T, T>> viewer() const MUDA_NOEXCEPT
    {
        return Dense<std::conditional_t<IsConst, const T, T>>(const_cast<T*>(m_data));
    }

    MUDA_GENERIC CDense<T> cviewer() const MUDA_NOEXCEPT { return CDense<T>(m_data); }

    MUDA_HOST void fill(const T& value) const
    {
        static_assert(!IsConst, "Cannot fill const var");
        cudaMemcpy(const_cast<T*>(m_data), &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    MUDA_HOST void copy_from(const T* host) const
    {
        static_assert(!IsConst, "Cannot copy into const var");
        cudaMemcpy(const_cast<T*>(m_data), host, sizeof(T), cudaMemcpyHostToDevice);
    }

    MUDA_HOST void copy_to(T* host) const
    {
        cudaMemcpy(host, m_data, sizeof(T), cudaMemcpyDeviceToHost);
    }

    MUDA_GENERIC T& operator*() const MUDA_NOEXCEPT { return *const_cast<T*>(m_data); }
    MUDA_GENERIC T& operator[](int) const MUDA_NOEXCEPT { return *const_cast<T*>(m_data); }
};

template <typename T>
using VarView = VarViewT<false, T>;

template <typename T>
using CVarView = VarViewT<true, T>;

// Device buffer
template <typename T>
class DeviceBuffer
{
    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_data     = nullptr;

    void realloc(size_t new_capacity)
    {
        if(new_capacity == 0)
        {
            if(m_data)
            {
                cudaFree(m_data);
                m_data = nullptr;
            }
            m_capacity = 0;
            return;
        }
        T* new_data = nullptr;
        cudaMalloc(&new_data, new_capacity * sizeof(T));
        if(m_data)
        {
            cudaMemcpy(new_data, m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaFree(m_data);
        }
        m_data     = new_data;
        m_capacity = new_capacity;
    }

  public:
    using value_type = T;

    DeviceBuffer() = default;
    DeviceBuffer(size_t n) { resize(n); }
    DeviceBuffer(const std::vector<T>& host) { copy_from(host); }
    DeviceBuffer(CBufferView<T> view) { copy_from(view); }

    DeviceBuffer(const DeviceBuffer<T>& other) { copy_from(other.view()); }
    DeviceBuffer(DeviceBuffer<T>&& other) MUDA_NOEXCEPT
        : m_size(other.m_size),
          m_capacity(other.m_capacity),
          m_data(other.m_data)
    {
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_data     = nullptr;
    }

    DeviceBuffer<T>& operator=(const DeviceBuffer<T>& other)
    {
        copy_from(other.view());
        return *this;
    }

    DeviceBuffer<T>& operator=(DeviceBuffer<T>&& other) MUDA_NOEXCEPT
    {
        if(m_data)
            cudaFree(m_data);
        m_size           = other.m_size;
        m_capacity       = other.m_capacity;
        m_data           = other.m_data;
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_data     = nullptr;
        return *this;
    }

    ~DeviceBuffer()
    {
        if(m_data)
            cudaFree(m_data);
    }

    void resize(size_t new_size)
    {
        if(new_size > m_capacity)
            reserve(new_size);
        m_size = new_size;
    }

    void resize(size_t new_size, const T& value)
    {
        resize(new_size);
    }

    void reserve(size_t new_capacity)
    {
        if(new_capacity > m_capacity)
            realloc(new_capacity);
    }

    void clear() { m_size = 0; }

    void shrink_to_fit()
    {
        if(m_capacity > m_size)
            realloc(m_size);
    }

    void copy_to(std::vector<T>& host) const
    {
        host.resize(m_size);
        cudaMemcpy(host.data(), m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void copy_from(const std::vector<T>& host)
    {
        resize(host.size());
        cudaMemcpy(m_data, host.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_from(CBufferView<T> view)
    {
        resize(view.size());
        cudaMemcpy(m_data, view.data(), m_size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    void fill(const T& value) { view().fill(value); }

    Dense1D<T> viewer() MUDA_NOEXCEPT { return Dense1D<T>(m_data, m_size); }
    CDense1D<T> cviewer() const MUDA_NOEXCEPT { return CDense1D<T>(m_data, m_size); }

    BufferView<T> view(size_t offset, size_t size = ~size_t(0)) MUDA_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? m_size - offset : size;
        return BufferView<T>(m_data, offset, s);
    }

    BufferView<T> view() MUDA_NOEXCEPT { return BufferView<T>(m_data, m_size); }

    CBufferView<T> view(size_t offset, size_t size = ~size_t(0)) const MUDA_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? m_size - offset : size;
        return CBufferView<T>(m_data, offset, s);
    }

    CBufferView<T> view() const MUDA_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    operator BufferView<T>() MUDA_NOEXCEPT { return view(); }
    operator CBufferView<T>() const MUDA_NOEXCEPT { return view(); }

    size_t size() const MUDA_NOEXCEPT { return m_size; }
    size_t capacity() const MUDA_NOEXCEPT { return m_capacity; }
    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
};

// Device var
template <typename T>
class DeviceVar
{
    T* m_data = nullptr;

  public:
    using value_type = T;

    DeviceVar() { cudaMalloc(&m_data, sizeof(T)); }
    DeviceVar(const T& value) : DeviceVar() { operator=(value); }

    DeviceVar(const DeviceVar& other) : DeviceVar() { copy_from(other.view()); }
    DeviceVar(DeviceVar&& other) MUDA_NOEXCEPT : m_data(other.m_data)
    {
        other.m_data = nullptr;
    }

    DeviceVar& operator=(const DeviceVar<T>& other)
    {
        copy_from(other.view());
        return *this;
    }

    DeviceVar& operator=(DeviceVar<T>&& other) MUDA_NOEXCEPT
    {
        if(m_data)
            cudaFree(m_data);
        m_data       = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    ~DeviceVar()
    {
        if(m_data)
            cudaFree(m_data);
    }

    DeviceVar& operator=(CVarView<T> other)
    {
        cudaMemcpy(m_data, other.data(), sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    void copy_from(CVarView<T> other)
    {
        cudaMemcpy(m_data, other.data(), sizeof(T), cudaMemcpyDeviceToDevice);
    }

    DeviceVar& operator=(const T& val)
    {
        cudaMemcpy(m_data, &val, sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }

    operator T() const
    {
        T val;
        cudaMemcpy(&val, m_data, sizeof(T), cudaMemcpyDeviceToHost);
        return val;
    }

    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }

    VarView<T>  view() MUDA_NOEXCEPT { return VarView<T>(m_data); }
    CVarView<T> view() const MUDA_NOEXCEPT { return CVarView<T>(m_data); }

    operator VarView<T>() MUDA_NOEXCEPT { return view(); }
    operator CVarView<T>() const MUDA_NOEXCEPT { return view(); }

    Dense<T>  viewer() MUDA_NOEXCEPT { return Dense<T>(m_data); }
    CDense<T> cviewer() const MUDA_NOEXCEPT { return CDense<T>(m_data); }
};

}  // namespace cuda
}  // namespace gipc
