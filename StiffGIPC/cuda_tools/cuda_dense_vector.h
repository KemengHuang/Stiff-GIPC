#pragma once

#include "cuda_def.h"
#include "cuda_buffer_view.h"
#include "cuda_atomic.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <Eigen/Core>

namespace cudatool {

template <typename T>
class CDenseVectorView;

// DenseVectorViewer
template <typename T>
class CDenseVectorViewer
{
  protected:
    const T* m_data;
    int      m_offset;
    int      m_size;

  public:
    CT_GENERIC CDenseVectorViewer() CT_NOEXCEPT
        : m_data(nullptr),
          m_offset(0),
          m_size(0)
    {
    }
    CT_GENERIC CDenseVectorViewer(const T* data, int offset, int size) CT_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    CT_GENERIC const T& operator()(int i) const CT_NOEXCEPT { return m_data[m_offset + i]; }
    CT_GENERIC const T& operator[](int i) const CT_NOEXCEPT { return m_data[m_offset + i]; }
    CT_GENERIC int      size() const CT_NOEXCEPT { return m_size; }
    CT_GENERIC int      offset() const CT_NOEXCEPT { return m_offset; }
    CT_GENERIC const T* origin_data() const CT_NOEXCEPT { return m_data; }

    CT_GENERIC CDenseVectorViewer& name(const char*) CT_NOEXCEPT { return *this; }
    CT_GENERIC CDenseVectorViewer& name(const std::string&) CT_NOEXCEPT { return *this; }

    CT_GENERIC CDenseVectorViewer segment(int offset, int size) const CT_NOEXCEPT
    {
        return CDenseVectorViewer(m_data, m_offset + offset, size);
    }

    template <int N>
    CT_GENERIC CDenseVectorViewer segment(int offset) const CT_NOEXCEPT
    {
        return segment(offset, N);
    }

    CT_GENERIC Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> as_eigen() const CT_NOEXCEPT
    {
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(m_data + m_offset, m_size);
    }

    CT_GENERIC operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>() const CT_NOEXCEPT
    {
        return as_eigen();
    }
};

template <typename T>
class DenseVectorViewer : public CDenseVectorViewer<T>
{
    using Base = CDenseVectorViewer<T>;

  public:
    CT_GENERIC DenseVectorViewer() CT_NOEXCEPT = default;
    CT_GENERIC DenseVectorViewer(T* data, int offset, int size) CT_NOEXCEPT
        : Base(data, offset, size)
    {
    }

    CT_GENERIC T& operator()(int i) const CT_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data())[Base::offset() + i];
    }
    CT_GENERIC T& operator[](int i) const CT_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data())[Base::offset() + i];
    }

    CT_GENERIC DenseVectorViewer& name(const char*) CT_NOEXCEPT { return *this; }
    CT_GENERIC DenseVectorViewer& name(const std::string&) CT_NOEXCEPT { return *this; }

    CT_GENERIC DenseVectorViewer segment(int offset, int size) const CT_NOEXCEPT
    {
        return DenseVectorViewer(const_cast<T*>(Base::origin_data()), Base::offset() + offset, size);
    }

    template <int N>
    CT_GENERIC DenseVectorViewer segment(int offset) const CT_NOEXCEPT
    {
        return segment(offset, N);
    }

    CT_GENERIC Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> as_eigen() const CT_NOEXCEPT
    {
        return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            const_cast<T*>(Base::origin_data()) + Base::offset(), Base::size());
    }

    CT_GENERIC operator Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>() const CT_NOEXCEPT
    {
        return as_eigen();
    }

    CT_DEVICE T atomic_add(int i, T val)
    {
        return cudatool::atomic_add(&(*this)(i), val);
    }

    template <int N>
    CT_DEVICE Eigen::Matrix<T, N, 1> atomic_add(const Eigen::Matrix<T, N, 1>& val)
    {
        Eigen::Matrix<T, N, 1> ret;
        for(int i = 0; i < N; ++i)
            ret(i) = atomic_add(i, val(i));
        return ret;
    }

    CT_DEVICE T atomic_add(const T& val)
    {
        return atomic_add(0, val);
    }

    template <int N>
    CT_GENERIC DenseVectorViewer& operator=(const Eigen::Matrix<T, N, 1>& other)
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = other(i);
        return *this;
    }
};

// DenseVectorView
template <typename T>
class DenseVectorView
{
    T* m_data = nullptr;
    int m_size = 0;

  public:
    using value_type = T;

    DenseVectorView() = default;
    DenseVectorView(T* data, int size) CT_NOEXCEPT : m_data(data), m_size(size) {}
    DenseVectorView(DeviceBuffer<T>& buf) CT_NOEXCEPT : m_data(buf.data()), m_size((int)buf.size()) {}

    T* data() CT_NOEXCEPT { return m_data; }
    const T* data() const CT_NOEXCEPT { return m_data; }
    int size() const CT_NOEXCEPT { return m_size; }

    DenseVectorViewer<T> viewer() CT_NOEXCEPT { return DenseVectorViewer<T>(m_data, 0, m_size); }
    CDenseVectorViewer<T> cviewer() const CT_NOEXCEPT { return CDenseVectorViewer<T>(m_data, 0, m_size); }

    BufferView<T> buffer_view() CT_NOEXCEPT { return BufferView<T>(m_data, m_size); }
    CBufferView<T> buffer_view() const CT_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    DenseVectorView subview(int offset, int size) CT_NOEXCEPT
    {
        return DenseVectorView(m_data + offset, size);
    }

    void fill(const T& value)
    {
        buffer_view().fill(value);
    }

    void copy_from(const CBufferView<T>& other)
    {
        size_t n = std::min((size_t)m_size, other.size());
        cudaMemcpy(m_data, other.data(), n * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    void copy_from(const T* host)
    {
        cudaMemcpy(m_data, host, m_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_to(T* host) const
    {
        cudaMemcpy(host, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    operator CDenseVectorView<T>() const CT_NOEXCEPT;
};

template <typename T>
class CDenseVectorView
{
    const T* m_data = nullptr;
    int      m_size = 0;

  public:
    using value_type = T;

    CDenseVectorView() = default;
    CDenseVectorView(const T* data, int size) CT_NOEXCEPT : m_data(data), m_size(size) {}
    CDenseVectorView(const DeviceBuffer<T>& buf) CT_NOEXCEPT : m_data(buf.data()), m_size((int)buf.size()) {}
    CDenseVectorView(const DenseVectorView<T>& v) CT_NOEXCEPT : m_data(v.data()), m_size(v.size()) {}

    const T* data() const CT_NOEXCEPT { return m_data; }
    int      size() const CT_NOEXCEPT { return m_size; }

    CDenseVectorViewer<T> cviewer() const CT_NOEXCEPT { return CDenseVectorViewer<T>(m_data, 0, m_size); }
    CDenseVectorViewer<T> viewer() const CT_NOEXCEPT { return cviewer(); }

    CBufferView<T> buffer_view() const CT_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    CDenseVectorView subview(int offset, int size) const CT_NOEXCEPT
    {
        return CDenseVectorView(m_data + offset, size);
    }

    void copy_to(T* host) const
    {
        cudaMemcpy(host, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

template <typename T>
DenseVectorView<T>::operator CDenseVectorView<T>() const CT_NOEXCEPT
{
    return CDenseVectorView<T>(m_data, m_size);
}

// DeviceDenseVector
template <typename T>
class DeviceDenseVector
{
    DeviceBuffer<T> m_data;

  public:
    using value_type = T;

    DeviceDenseVector() = default;
    DeviceDenseVector(size_t size) { resize(size); }

    void resize(size_t size) { m_data.resize(size); }
    void reserve(size_t size) { m_data.reserve(size); }
    void fill(T value)
    {
        for(size_t i = 0; i < m_data.size(); ++i)
            cudaMemcpy(m_data.data() + i, &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    size_t size() const CT_NOEXCEPT { return m_data.size(); }
    size_t capacity() const CT_NOEXCEPT { return m_data.capacity(); }
    T*       data() CT_NOEXCEPT { return m_data.data(); }
    const T* data() const CT_NOEXCEPT { return m_data.data(); }

    DenseVectorView<T> view() CT_NOEXCEPT { return DenseVectorView<T>(m_data.data(), (int)m_data.size()); }
    CDenseVectorView<T> view() const CT_NOEXCEPT { return CDenseVectorView<T>(m_data.data(), (int)m_data.size()); }
    CDenseVectorView<T> cview() const CT_NOEXCEPT { return view(); }

    DenseVectorViewer<T> viewer() CT_NOEXCEPT { return view().viewer(); }
    CDenseVectorViewer<T> cviewer() const CT_NOEXCEPT { return view().cviewer(); }

    BufferView<T> buffer_view() CT_NOEXCEPT { return m_data.view(); }
    CBufferView<T> buffer_view() const CT_NOEXCEPT { return m_data.view(); }

    void copy_from(const std::vector<T>& host)
    {
        resize(host.size());
        cudaMemcpy(m_data.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_to(std::vector<T>& host) const { m_data.copy_to(host); }

    operator DenseVectorView<T>() CT_NOEXCEPT { return view(); }
    operator CDenseVectorView<T>() const CT_NOEXCEPT { return view(); }
};

}  // namespace cudatool
