#pragma once

#include "muda_def.h"
#include "buffer.h"
#include "atomic.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <Eigen/Core>

namespace gipc {
namespace cuda {

// Forward declarations for matrix/vector views
template <typename T>
class DenseVectorViewer;
template <typename T>
class CDenseVectorViewer;

template <typename T>
class DenseVectorView;
template <typename T>
class CDenseVectorView;

template <typename T, int N>
class DeviceDoubletVector;
template <typename T>
class DeviceDenseMatrix;
template <typename T>
class DeviceCSRMatrix;

template <typename T>
class CCSRMatrixView
{
    int m_row = 0, m_col = 0, m_nnz = 0;

  public:
    CCSRMatrixView() = default;
    CCSRMatrixView(int row, int col, const int*, const int*, const T*, int nnz) MUDA_NOEXCEPT
        : m_row(row),
          m_col(col),
          m_nnz(nnz)
    {
    }
    int rows() const { return m_row; }
    int cols() const { return m_col; }
    int non_zeros() const { return m_nnz; }
};

// DenseVectorViewer
template <typename T>
class CDenseVectorViewer
{
  protected:
    const T* m_data;
    int      m_offset;
    int      m_size;

  public:
    MUDA_GENERIC CDenseVectorViewer() MUDA_NOEXCEPT
        : m_data(nullptr),
          m_offset(0),
          m_size(0)
    {
    }
    MUDA_GENERIC CDenseVectorViewer(const T* data, int offset, int size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    MUDA_GENERIC const T& operator()(int i) const MUDA_NOEXCEPT { return m_data[m_offset + i]; }
    MUDA_GENERIC const T& operator[](int i) const MUDA_NOEXCEPT { return m_data[m_offset + i]; }
    MUDA_GENERIC int      size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC int      offset() const MUDA_NOEXCEPT { return m_offset; }
    MUDA_GENERIC const T* origin_data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC CDenseVectorViewer& name(const char*) MUDA_NOEXCEPT { return *this; }
    MUDA_GENERIC CDenseVectorViewer& name(const std::string&) MUDA_NOEXCEPT { return *this; }

    MUDA_GENERIC CDenseVectorViewer segment(int offset, int size) const MUDA_NOEXCEPT
    {
        return CDenseVectorViewer(m_data, m_offset + offset, size);
    }

    template <int N>
    MUDA_GENERIC CDenseVectorViewer segment(int offset) const MUDA_NOEXCEPT
    {
        return segment(offset, N);
    }

    MUDA_GENERIC Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> as_eigen() const MUDA_NOEXCEPT
    {
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(m_data + m_offset, m_size);
    }

    MUDA_GENERIC operator Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>() const MUDA_NOEXCEPT
    {
        return as_eigen();
    }
};

template <typename T>
class DenseVectorViewer : public CDenseVectorViewer<T>
{
    using Base = CDenseVectorViewer<T>;

  public:
    MUDA_GENERIC DenseVectorViewer() MUDA_NOEXCEPT = default;
    MUDA_GENERIC DenseVectorViewer(T* data, int offset, int size) MUDA_NOEXCEPT
        : Base(data, offset, size)
    {
    }

    MUDA_GENERIC T& operator()(int i) const MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data())[Base::offset() + i];
    }
    MUDA_GENERIC T& operator[](int i) const MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data())[Base::offset() + i];
    }

    MUDA_GENERIC DenseVectorViewer& name(const char*) MUDA_NOEXCEPT { return *this; }
    MUDA_GENERIC DenseVectorViewer& name(const std::string&) MUDA_NOEXCEPT { return *this; }

    MUDA_GENERIC DenseVectorViewer segment(int offset, int size) const MUDA_NOEXCEPT
    {
        return DenseVectorViewer(const_cast<T*>(Base::origin_data()), Base::offset() + offset, size);
    }

    template <int N>
    MUDA_GENERIC DenseVectorViewer segment(int offset) const MUDA_NOEXCEPT
    {
        return segment(offset, N);
    }

    MUDA_GENERIC Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> as_eigen() const MUDA_NOEXCEPT
    {
        return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            const_cast<T*>(Base::origin_data()) + Base::offset(), Base::size());
    }

    MUDA_GENERIC operator Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>() const MUDA_NOEXCEPT
    {
        return as_eigen();
    }

    MUDA_DEVICE T atomic_add(int i, T val)
    {
        return gipc::cuda::atomic_add(&(*this)(i), val);
    }

    template <int N>
    MUDA_DEVICE Eigen::Matrix<T, N, 1> atomic_add(const Eigen::Matrix<T, N, 1>& val)
    {
        Eigen::Matrix<T, N, 1> ret;
        for(int i = 0; i < N; ++i)
            ret(i) = atomic_add(i, val(i));
        return ret;
    }

    MUDA_DEVICE T atomic_add(const T& val)
    {
        return atomic_add(0, val);
    }

    template <int N>
    MUDA_GENERIC DenseVectorViewer& operator=(const Eigen::Matrix<T, N, 1>& other)
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
    DenseVectorView(T* data, int size) MUDA_NOEXCEPT : m_data(data), m_size(size) {}
    DenseVectorView(DeviceBuffer<T>& buf) MUDA_NOEXCEPT : m_data(buf.data()), m_size((int)buf.size()) {}

    T* data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
    int size() const MUDA_NOEXCEPT { return m_size; }

    DenseVectorViewer<T> viewer() MUDA_NOEXCEPT { return DenseVectorViewer<T>(m_data, 0, m_size); }
    CDenseVectorViewer<T> cviewer() const MUDA_NOEXCEPT { return CDenseVectorViewer<T>(m_data, 0, m_size); }

    BufferView<T> buffer_view() MUDA_NOEXCEPT { return BufferView<T>(m_data, m_size); }
    CBufferView<T> buffer_view() const MUDA_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    DenseVectorView subview(int offset, int size) MUDA_NOEXCEPT
    {
        return DenseVectorView(m_data + offset, size);
    }

    void fill(const T& value)
    {
        cudaMemcpy(m_data, &value, sizeof(T), cudaMemcpyHostToDevice);  // only valid for size==1
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

    operator CDenseVectorView<T>() const MUDA_NOEXCEPT;
};

template <typename T>
class CDenseVectorView
{
    const T* m_data = nullptr;
    int      m_size = 0;

  public:
    using value_type = T;

    CDenseVectorView() = default;
    CDenseVectorView(const T* data, int size) MUDA_NOEXCEPT : m_data(data), m_size(size) {}
    CDenseVectorView(const DeviceBuffer<T>& buf) MUDA_NOEXCEPT : m_data(buf.data()), m_size((int)buf.size()) {}
    CDenseVectorView(const DenseVectorView<T>& v) MUDA_NOEXCEPT : m_data(v.data()), m_size(v.size()) {}

    const T* data() const MUDA_NOEXCEPT { return m_data; }
    int      size() const MUDA_NOEXCEPT { return m_size; }

    CDenseVectorViewer<T> cviewer() const MUDA_NOEXCEPT { return CDenseVectorViewer<T>(m_data, 0, m_size); }
    CDenseVectorViewer<T> viewer() const MUDA_NOEXCEPT { return cviewer(); }

    CBufferView<T> buffer_view() const MUDA_NOEXCEPT { return CBufferView<T>(m_data, m_size); }

    CDenseVectorView subview(int offset, int size) const MUDA_NOEXCEPT
    {
        return CDenseVectorView(m_data + offset, size);
    }

    void copy_to(T* host) const
    {
        cudaMemcpy(host, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

template <typename T>
DenseVectorView<T>::operator CDenseVectorView<T>() const MUDA_NOEXCEPT
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

    size_t size() const MUDA_NOEXCEPT { return m_data.size(); }
    size_t capacity() const MUDA_NOEXCEPT { return m_data.capacity(); }
    T*       data() MUDA_NOEXCEPT { return m_data.data(); }
    const T* data() const MUDA_NOEXCEPT { return m_data.data(); }

    DenseVectorView<T> view() MUDA_NOEXCEPT { return DenseVectorView<T>(m_data.data(), (int)m_data.size()); }
    CDenseVectorView<T> view() const MUDA_NOEXCEPT { return CDenseVectorView<T>(m_data.data(), (int)m_data.size()); }
    CDenseVectorView<T> cview() const MUDA_NOEXCEPT { return view(); }

    DenseVectorViewer<T> viewer() MUDA_NOEXCEPT { return view().viewer(); }
    CDenseVectorViewer<T> cviewer() const MUDA_NOEXCEPT { return view().cviewer(); }

    BufferView<T> buffer_view() MUDA_NOEXCEPT { return m_data.view(); }
    CBufferView<T> buffer_view() const MUDA_NOEXCEPT { return m_data.view(); }

    void copy_from(const std::vector<T>& host)
    {
        resize(host.size());
        cudaMemcpy(m_data.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copy_to(std::vector<T>& host) const { m_data.copy_to(host); }

    operator DenseVectorView<T>() MUDA_NOEXCEPT { return view(); }
    operator CDenseVectorView<T>() const MUDA_NOEXCEPT { return view(); }
};

// Stubs for declared-but-unused types
template <typename T, int N>
class DeviceDoubletVector
{
    DeviceBuffer<Eigen::Matrix<T, N, 1>> m_values;
    DeviceBuffer<int>                    m_indices;
    int                                  m_count = 0;

  public:
    using SegmentVector = Eigen::Matrix<T, N, 1>;

    void reshape(int num) { m_count = num; }
    void resize_doublets(size_t n)
    {
        m_values.resize(n);
        m_indices.resize(n);
    }
    void reserve_doublets(size_t n)
    {
        m_values.reserve(n);
        m_indices.reserve(n);
    }
    void resize(int num, size_t n)
    {
        reshape(num);
        resize_doublets(n);
    }
    void clear()
    {
        m_values.clear();
        m_indices.clear();
    }
    int  segment_count() const { return m_count; }
    auto segment_values() { return m_values.view(); }
    auto segment_values() const { return m_values.view(); }
    auto segment_indices() { return m_indices.view(); }
    auto segment_indices() const { return m_indices.view(); }
    auto doublet_count() const { return m_values.size(); }
    auto doublet_capacity() const { return m_values.capacity(); }
};

template <typename T>
class DeviceDoubletVector<T, 1>
{
    DeviceBuffer<T>   m_values;
    DeviceBuffer<int> m_indices;
    int               m_size = 0;

  public:
    void reshape(int num) { m_size = num; }
    void resize_doublet(size_t n)
    {
        m_values.resize(n);
        m_indices.resize(n);
    }
    void resize(int num, size_t n)
    {
        reshape(num);
        resize_doublet(n);
    }
    void clear()
    {
        m_values.clear();
        m_indices.clear();
    }
    int  size() const { return m_size; }
    auto values() { return m_values.view(); }
    auto values() const { return m_values.view(); }
    auto indices() { return m_indices.view(); }
    auto indices() const { return m_indices.view(); }
    auto doublet_count() const { return m_values.size(); }
};

template <typename T>
class DeviceDenseMatrix
{
    DeviceBuffer<T> m_data;
    size_t          m_row = 0;
    size_t          m_col = 0;

  public:
    using value_type = T;

    DeviceDenseMatrix() = default;
    DeviceDenseMatrix(size_t row, size_t col) { reshape(row, col); }

    void reshape(size_t row, size_t col)
    {
        m_row = row;
        m_col = col;
        m_data.resize(row * col);
    }
    void fill(T value)
    {
        for(size_t i = 0; i < m_data.size(); ++i)
            cudaMemcpy(m_data.data() + i, &value, sizeof(T), cudaMemcpyHostToDevice);
    }
    size_t row() const { return m_row; }
    size_t col() const { return m_col; }
    T*       data() MUDA_NOEXCEPT { return m_data.data(); }
    const T* data() const MUDA_NOEXCEPT { return m_data.data(); }
};

template <typename T>
class DeviceCSRMatrix
{
    DeviceBuffer<int> m_row_offsets;
    DeviceBuffer<int> m_col_indices;
    DeviceBuffer<T>   m_values;
    int               m_row = 0;
    int               m_col = 0;

  public:
    using value_type = T;

    DeviceCSRMatrix() = default;
    void reshape(int row, int col)
    {
        m_row = row;
        m_col = col;
    }
    void reserve(int non_zeros) { m_values.reserve(non_zeros); }
    void clear()
    {
        m_row_offsets.clear();
        m_col_indices.clear();
        m_values.clear();
    }
    int rows() const { return m_row; }
    int cols() const { return m_col; }
    int non_zeros() const { return (int)m_values.size(); }
    auto values() { return m_values.view(); }
    auto values() const { return m_values.view(); }
    auto row_offsets() { return m_row_offsets.view(); }
    auto row_offsets() const { return m_row_offsets.view(); }
    auto col_indices() { return m_col_indices.view(); }
    auto col_indices() const { return m_col_indices.view(); }
};

// LinearSystemContext
class LinearSystemContext
{
    cublasHandle_t   m_cublas   = nullptr;
    cusparseHandle_t m_cusparse = nullptr;
    cudaStream_t     m_stream   = nullptr;

  public:
    LinearSystemContext(cudaStream_t stream = nullptr) : m_stream(stream)
    {
        cublasCreate(&m_cublas);
        cusparseCreate(&m_cusparse);
        cublasSetStream(m_cublas, m_stream);
        cusparseSetStream(m_cusparse, m_stream);
    }

    LinearSystemContext(const LinearSystemContext&)            = delete;
    LinearSystemContext& operator=(const LinearSystemContext&) = delete;

    ~LinearSystemContext()
    {
        if(m_cublas)
            cublasDestroy(m_cublas);
        if(m_cusparse)
            cusparseDestroy(m_cusparse);
    }

    cublasHandle_t   cublas() const { return m_cublas; }
    cusparseHandle_t cusparse() const { return m_cusparse; }
    cudaStream_t     stream() const { return m_stream; }
    void             stream(cudaStream_t s)
    {
        m_stream = s;
        cublasSetStream(m_cublas, s);
        cusparseSetStream(m_cusparse, s);
    }
    void sync() { cudaStreamSynchronize(m_stream); }

    template <typename T>
    T norm(CDenseVectorView<T> x)
    {
        return T(0);
    }
    template <typename T>
    void dot(CDenseVectorView<T> x, CDenseVectorView<T> y, VarView<T> result)
    {
    }
    template <typename T>
    void axpby(const T& alpha, CDenseVectorView<T> x, const T& beta, DenseVectorView<T> y)
    {
    }
    template <typename T>
    void plus(CDenseVectorView<T> x, CDenseVectorView<T> y, DenseVectorView<T> z)
    {
    }
    template <typename T>
    void spmv(CCSRMatrixView<T> A, CDenseVectorView<T> x, DenseVectorView<T> y)
    {
    }
};

}  // namespace cuda
}  // namespace gipc
