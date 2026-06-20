#pragma once

#include "cuda_def.h"
#include <cuda_runtime.h>
#include <cinttypes>
#include <vector>
#include <type_traits>
#include <Eigen/Core>

namespace cudatool {

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

namespace details {
template <typename T>
__global__ void buffer_fill_kernel(T* data, size_t size, T value)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        data[i] = value;
}
}  // namespace details

template <bool IsConst, typename T> class VarViewT;
template <typename T> using VarView = VarViewT<false, T>;
template <typename T> using CVarView = VarViewT<true, T>;

// 0D viewer
template <typename T>
class Dense
{
    T* m_data;

  public:
    CT_GENERIC Dense(T* data = nullptr) CT_NOEXCEPT : m_data(data) {}
    CT_GENERIC T& operator*() const CT_NOEXCEPT { return *m_data; }
    CT_GENERIC T& operator()() const CT_NOEXCEPT { return *m_data; }
    CT_GENERIC T* data() const CT_NOEXCEPT { return m_data; }
};

template <typename T>
class CDense
{
    const T* m_data;

  public:
    CT_GENERIC CDense(const T* data = nullptr) CT_NOEXCEPT : m_data(data) {}
    CT_GENERIC const T& operator*() const CT_NOEXCEPT { return *m_data; }
    CT_GENERIC const T& operator()() const CT_NOEXCEPT { return *m_data; }
    CT_GENERIC const T* data() const CT_NOEXCEPT { return m_data; }
};

// 1D viewer
template <typename T>
class Dense1D
{
    T*     m_data;
    size_t m_size;

  public:
    CT_GENERIC Dense1D(T* data = nullptr, size_t size = 0) CT_NOEXCEPT
        : m_data(data),
          m_size(size)
    {
    }
    CT_GENERIC T& operator()(int i) const CT_NOEXCEPT { return m_data[i]; }
    CT_GENERIC T& operator[](int i) const CT_NOEXCEPT { return m_data[i]; }
    CT_GENERIC T*       data() const CT_NOEXCEPT { return m_data; }
    CT_GENERIC size_t   size() const CT_NOEXCEPT { return m_size; }
    CT_GENERIC Dense1D& name(const char*) CT_NOEXCEPT { return *this; }
    CT_GENERIC Dense1D& name(const std::string&) CT_NOEXCEPT { return *this; }
};

template <typename T>
class CDense1D
{
    const T* m_data;
    size_t   m_size;

  public:
    CT_GENERIC CDense1D(const T* data = nullptr, size_t size = 0) CT_NOEXCEPT
        : m_data(data),
          m_size(size)
    {
    }
    CT_GENERIC const T& operator()(int i) const CT_NOEXCEPT { return m_data[i]; }
    CT_GENERIC const T& operator[](int i) const CT_NOEXCEPT { return m_data[i]; }
    CT_GENERIC const T* data() const CT_NOEXCEPT { return m_data; }
    CT_GENERIC size_t   size() const CT_NOEXCEPT { return m_size; }
    CT_GENERIC CDense1D& name(const char*) CT_NOEXCEPT { return *this; }
    CT_GENERIC CDense1D& name(const std::string&) CT_NOEXCEPT { return *this; }
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

    CT_GENERIC BufferViewT() CT_NOEXCEPT = default;
    CT_GENERIC BufferViewT(pointer_type data, size_t size) CT_NOEXCEPT
        : m_data(data),
          m_offset(0),
          m_size(size)
    {
    }
    CT_GENERIC BufferViewT(pointer_type data, size_t offset, size_t size) CT_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    template <bool OtherIsConst,
              typename = std::enable_if_t<!OtherIsConst || IsConst>>
    CT_GENERIC BufferViewT(const BufferViewT<OtherIsConst, T>& other) CT_NOEXCEPT
        : m_data(other.origin_data()),
          m_offset(other.offset()),
          m_size(other.size())
    {
    }

    CT_GENERIC pointer_type data() const CT_NOEXCEPT { return m_data + m_offset; }
    CT_GENERIC pointer_type origin_data() const CT_NOEXCEPT { return m_data; }
    CT_GENERIC size_t       size() const CT_NOEXCEPT { return m_size; }
    CT_GENERIC size_t       offset() const CT_NOEXCEPT { return m_offset; }

    CT_GENERIC BufferViewT<IsConst, T> subview(size_t offset, size_t size = ~size_t(0)) const CT_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? ((m_size > offset) ? (m_size - offset) : 0) : size;
        return BufferViewT<IsConst, T>(m_data, m_offset + offset, s);
    }

    CT_GENERIC Dense1D<std::conditional_t<IsConst, const T, T>> viewer() const CT_NOEXCEPT
    {
        return Dense1D<std::conditional_t<IsConst, const T, T>>(data(), m_size);
    }

    CT_GENERIC CDense1D<T> cviewer() const CT_NOEXCEPT
    {
        return CDense1D<T>(data(), m_size);
    }

    CT_GENERIC auto& operator[](size_t i) const CT_NOEXCEPT { return data()[i]; }
    CT_GENERIC auto& operator*() const CT_NOEXCEPT { return *data(); }
    CT_GENERIC auto& operator[](int i) const CT_NOEXCEPT { return data()[i]; }

    CT_HOST void copy_from(const BufferViewT<true, T>& other) const
    {
        static_assert(!IsConst, "Cannot copy into const view");
        size_t n = std::min(m_size, other.size());
        cudaMemcpy(data(), other.data(), n * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    CT_HOST void copy_from(const T* host) const
    {
        static_assert(!IsConst, "Cannot copy into const view");
        cudaMemcpy(data(), host, m_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    CT_HOST void copy_to(T* host) const
    {
        cudaMemcpy(host, data(), m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    CT_HOST void fill(const T& value) const
    {
        static_assert(!IsConst, "Cannot fill const view");
        if(m_size == 0)
            return;
        const int block = 256;
        const int grid  = static_cast<int>((m_size + block - 1) / block);
        details::buffer_fill_kernel<<<grid, block>>>(data(), m_size, value);
    }

    // Iterator-like
    using reference = T&;
    using pointer   = T*;
    CT_GENERIC BufferViewT operator+(int i) const CT_NOEXCEPT
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

    CT_GENERIC VarViewT() CT_NOEXCEPT = default;
    CT_GENERIC VarViewT(pointer_type data) CT_NOEXCEPT : m_data(data) {}

    CT_GENERIC pointer_type data() const CT_NOEXCEPT { return m_data; }

    CT_GENERIC Dense<std::conditional_t<IsConst, const T, T>> viewer() const CT_NOEXCEPT
    {
        return Dense<std::conditional_t<IsConst, const T, T>>(const_cast<T*>(m_data));
    }

    CT_GENERIC CDense<T> cviewer() const CT_NOEXCEPT { return CDense<T>(m_data); }

    CT_HOST void fill(const T& value) const
    {
        static_assert(!IsConst, "Cannot fill const var");
        cudaMemcpy(const_cast<T*>(m_data), &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    CT_HOST void copy_from(const T* host) const
    {
        static_assert(!IsConst, "Cannot copy into const var");
        cudaMemcpy(const_cast<T*>(m_data), host, sizeof(T), cudaMemcpyHostToDevice);
    }

    CT_HOST void copy_to(T* host) const
    {
        cudaMemcpy(host, m_data, sizeof(T), cudaMemcpyDeviceToHost);
    }

    CT_GENERIC T& operator*() const CT_NOEXCEPT { return *const_cast<T*>(m_data); }
    CT_GENERIC T& operator[](int) const CT_NOEXCEPT { return *const_cast<T*>(m_data); }
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
    DeviceBuffer(DeviceBuffer<T>&& other) CT_NOEXCEPT
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

    DeviceBuffer<T>& operator=(DeviceBuffer<T>&& other) CT_NOEXCEPT
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

    Dense1D<T> viewer() CT_NOEXCEPT { return Dense1D<T>(m_data, m_size); }
    CDense1D<T> cviewer() const CT_NOEXCEPT { return CDense1D<T>(m_data, m_size); }

    BufferView<T> view(size_t offset, size_t size = ~size_t(0)) CT_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? m_size - offset : size;
        return BufferView<T>(m_data, offset, s);
    }

    BufferView<T> view() CT_NOEXCEPT { return BufferView<T>(m_data, m_size); }

    CBufferView<T> view(size_t offset, size_t size = ~size_t(0)) const CT_NOEXCEPT
    {
        size_t s = (size == ~size_t(0)) ? m_size - offset : size;
        return CBufferView<T>(m_data, offset, s);
    }

    CBufferView<T> view() const CT_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    operator BufferView<T>() CT_NOEXCEPT { return view(); }
    operator CBufferView<T>() const CT_NOEXCEPT { return view(); }

    size_t size() const CT_NOEXCEPT { return m_size; }
    size_t capacity() const CT_NOEXCEPT { return m_capacity; }
    T*       data() CT_NOEXCEPT { return m_data; }
    const T* data() const CT_NOEXCEPT { return m_data; }
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
    DeviceVar(DeviceVar&& other) CT_NOEXCEPT : m_data(other.m_data)
    {
        other.m_data = nullptr;
    }

    DeviceVar& operator=(const DeviceVar<T>& other)
    {
        copy_from(other.view());
        return *this;
    }

    DeviceVar& operator=(DeviceVar<T>&& other) CT_NOEXCEPT
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

    T*       data() CT_NOEXCEPT { return m_data; }
    const T* data() const CT_NOEXCEPT { return m_data; }

    VarView<T>  view() CT_NOEXCEPT { return VarView<T>(m_data); }
    CVarView<T> view() const CT_NOEXCEPT { return CVarView<T>(m_data); }

    operator VarView<T>() CT_NOEXCEPT { return view(); }
    operator CVarView<T>() const CT_NOEXCEPT { return view(); }

    Dense<T>  viewer() CT_NOEXCEPT { return Dense<T>(m_data); }
    CDense<T> cviewer() const CT_NOEXCEPT { return CDense<T>(m_data); }
};

}  // namespace cudatool
