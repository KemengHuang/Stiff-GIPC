#pragma once

#include "cuda_def.h"
#include "cuda_buffer_view.h"
#include "cuda_dense_vector.h"
#include <Eigen/Core>

namespace cudatool {

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
    void fill(T value) { m_data.view().fill(value); }
    size_t row() const { return m_row; }
    size_t col() const { return m_col; }
    T*       data() CT_NOEXCEPT { return m_data.data(); }
    const T* data() const CT_NOEXCEPT { return m_data.data(); }
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

// LinearSystemContext: owns the cuBLAS/cuSPARSE handles and the stream
// shared by the linear-system solvers.
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
};

}  // namespace cudatool
