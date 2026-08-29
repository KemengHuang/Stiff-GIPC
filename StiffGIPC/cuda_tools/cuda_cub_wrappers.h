#pragma once
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include "cuda_def.h"
#include "cuda_debug.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace cudatool
{

namespace details
{
    class TempBuffer
    {
        size_t m_capacity = 0;
        void*  m_ptr      = nullptr;

      public:
        TempBuffer()                             = default;
        TempBuffer(const TempBuffer&)            = delete;
        TempBuffer& operator=(const TempBuffer&) = delete;

        ~TempBuffer()
        {
            if(m_ptr)
                cudaFree(m_ptr);
        }

        void* get(size_t bytes)
        {
            if(bytes <= m_capacity)
                return m_ptr;

            size_t growth = bytes / 2;
            if(growth == 0)
                growth = 1;
            if(bytes > std::numeric_limits<size_t>::max() - growth)
            {
                std::cerr << "CUB redundant workspace overflow: required="
                          << bytes << std::endl;
                std::abort();
            }
            size_t new_capacity = bytes + growth;

            // CUB calls in this wrapper use the calling thread's default
            // stream. cudaFree drains prior work before the workspace moves.
            if(m_ptr)
            {
                checkCudaErrors(cudaFree(m_ptr));
                m_ptr = nullptr;
            }
            checkCudaErrors(cudaMalloc(&m_ptr, new_capacity));
            m_capacity = new_capacity;
            return m_ptr;
        }
    };

    CT_INLINE void* get_temp_buffer(size_t bytes)
    {
        // CMake selects per-thread default streams, so each host thread needs
        // an independent persistent CUB workspace.
        static thread_local TempBuffer buffer;
        return buffer.get(bytes);
    }

    template <typename T>
    struct Less
    {
        CT_GENERIC bool operator()(const T& a, const T& b) const CT_NOEXCEPT
        {
            return a < b;
        }
    };
}  // namespace details

template <typename T>
struct Plus
{
    CT_GENERIC T operator()(const T& a, const T& b) const CT_NOEXCEPT
    {
        return a + b;
    }
};

class DeviceReduce
{
  public:
    template <typename T, typename ReduceOp>
    CT_HOST void Reduce(const T* src, T* dst, int count, ReduceOp op, const T& init)
    {
        if(count <= 0)
        {
            if(dst)
                checkCudaErrors(cudaMemcpy(dst, &init, sizeof(T), cudaMemcpyHostToDevice));
            return;
        }
        size_t temp_bytes = 0;
        cub::DeviceReduce::Reduce(nullptr, temp_bytes, src, dst, count, op, init);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst, count, op, init);
    }

    template <typename T>
    CT_HOST void Sum(const T* src, T* dst, int count)
    {
        if(count <= 0)
        {
            if(dst)
                checkCudaErrors(cudaMemset(dst, 0, sizeof(T)));
            return;
        }
        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, temp_bytes, src, dst, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceReduce::Sum(temp, temp_bytes, src, dst, count);
    }

    template <typename T>
    CT_HOST void Max(const T* src, T* dst, int count)
    {
        if(count <= 0)
        {
            if(dst)
                checkCudaErrors(cudaMemset(dst, 0, sizeof(T)));
            return;
        }
        size_t temp_bytes = 0;
        cub::DeviceReduce::Max(nullptr, temp_bytes, src, dst, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceReduce::Max(temp, temp_bytes, src, dst, count);
    }
};

class DeviceRadixSort
{
  public:
    template <typename KeyT, typename ValueT>
    CT_HOST void SortPairs(const KeyT*   keys_in,
                           KeyT*         keys_out,
                           const ValueT* values_in,
                           ValueT*       values_out,
                           int           count)
    {
        if(count <= 0)
            return;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes, keys_in, keys_out, values_in, values_out, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceRadixSort::SortPairs(
            temp, temp_bytes, keys_in, keys_out, values_in, values_out, count);
    }
};

class DeviceScan
{
  public:
    template <typename T>
    CT_HOST void ExclusiveSum(const T* src, T* dst, int count)
    {
        if(count <= 0)
            return;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, src, dst, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceScan::ExclusiveSum(temp, temp_bytes, src, dst, count);
    }
};

class DeviceSelect
{
  public:
    template <typename T, typename FlagT>
    CT_HOST void Flagged(const T* in, const FlagT* flags, T* out, int* d_count, int count)
    {
        if(count <= 0)
        {
            if(d_count)
                checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));
            return;
        }
        size_t temp_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, temp_bytes, in, flags, out, d_count, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceSelect::Flagged(temp, temp_bytes, in, flags, out, d_count, count);
    }
};

class DeviceRunLengthEncode
{
  public:
    template <typename T, typename CountT>
    CT_HOST void Encode(const T* in, T* unique, CountT* counts, int* d_count, int count)
    {
        if(count <= 0)
        {
            if(d_count)
                checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));
            return;
        }
        size_t temp_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes, in, unique, counts, d_count, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceRunLengthEncode::Encode(temp, temp_bytes, in, unique, counts, d_count, count);
    }
};

class DeviceMergeSort
{
  public:
    template <typename T>
    CT_HOST void SortKeys(T* in, int count)
    {
        if(count <= 1)
            return;
        size_t temp_bytes = 0;
        cub::DeviceMergeSort::SortKeys(nullptr, temp_bytes, in, count, details::Less<T>{});
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceMergeSort::SortKeys(temp, temp_bytes, in, count, details::Less<T>{});
    }

    template <typename T, typename CompareOp>
    CT_HOST void SortKeys(T* in, int count, CompareOp compare_op)
    {
        if(count <= 1)
            return;
        size_t temp_bytes = 0;
        cub::DeviceMergeSort::SortKeys(nullptr, temp_bytes, in, count, compare_op);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceMergeSort::SortKeys(temp, temp_bytes, in, count, compare_op);
    }
};

class DeviceSegmentedReduce
{
  public:
    template <typename T, typename OffsetT, typename ReduceOp>
    CT_HOST void Reduce(const T*       in,
                        T*             out,
                        int            num_segments,
                        const OffsetT* begin_offsets,
                        const OffsetT* end_offsets,
                        ReduceOp       op,
                        T              init)
    {
        if(num_segments <= 0)
            return;
        size_t temp_bytes = 0;
        cub::DeviceSegmentedReduce::Reduce(
            nullptr, temp_bytes, in, out, num_segments, begin_offsets, end_offsets, op, init);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceSegmentedReduce::Reduce(
            temp, temp_bytes, in, out, num_segments, begin_offsets, end_offsets, op, init);
    }
};

}  // namespace cudatool
