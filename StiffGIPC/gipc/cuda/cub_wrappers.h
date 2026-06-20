#pragma once
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <gipc/cuda/muda_def.h>
#include <cuda/std/functional>

namespace gipc
{
namespace cuda
{

namespace details
{
    MUDA_INLINE void* get_temp_buffer(size_t bytes)
    {
        static size_t   capacity = 0;
        static void*    ptr      = nullptr;
        if(bytes > capacity)
        {
            if(ptr)
                cudaFree(ptr);
            cudaMalloc(&ptr, bytes);
            capacity = bytes;
        }
        return ptr;
    }
}  // namespace details

class DeviceReduce
{
  public:
    template <typename T>
    MUDA_HOST void Sum(const T* src, T* dst, int count)
    {
        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, temp_bytes, src, dst, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceReduce::Sum(temp, temp_bytes, src, dst, count);
    }

    template <typename T>
    MUDA_HOST void Max(const T* src, T* dst, int count)
    {
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
    MUDA_HOST void SortPairs(const KeyT*   keys_in,
                             KeyT*         keys_out,
                             const ValueT* values_in,
                             ValueT*       values_out,
                             int           count)
    {
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_in, keys_out,
                                        values_in, values_out, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys_in, keys_out,
                                        values_in, values_out, count);
    }
};

class DeviceScan
{
  public:
    template <typename T>
    MUDA_HOST void ExclusiveSum(const T* src, T* dst, int count)
    {
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
    MUDA_HOST void Flagged(const T* in, const FlagT* flags, T* out, int* d_count, int count)
    {
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
    MUDA_HOST void Encode(const T* in, T* unique, CountT* counts, int* d_count, int count)
    {
        size_t temp_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes, in, unique, counts, d_count, count);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceRunLengthEncode::Encode(temp, temp_bytes, in, unique, counts, d_count, count);
    }
};

namespace details
{
    template <typename T>
    struct Less
    {
        MUDA_GENERIC bool operator()(const T& a, const T& b) const MUDA_NOEXCEPT
        {
            return a < b;
        }
    };
}  // namespace details

template <typename T>
struct Plus
{
    MUDA_GENERIC T operator()(const T& a, const T& b) const MUDA_NOEXCEPT
    {
        return a + b;
    }
};

class DeviceMergeSort
{
  public:
    template <typename T>
    MUDA_HOST void SortKeys(T* in, int count)
    {
        size_t temp_bytes = 0;
        cub::DeviceMergeSort::SortKeys(nullptr, temp_bytes, in, count, details::Less<T>{});
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceMergeSort::SortKeys(temp, temp_bytes, in, count, details::Less<T>{});
    }

    template <typename T, typename CompareOp>
    MUDA_HOST void SortKeys(T* in, int count, CompareOp compare_op)
    {
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
    MUDA_HOST void Reduce(const T*      in,
                          T*            out,
                          int           num_segments,
                          const OffsetT* begin_offsets,
                          const OffsetT* end_offsets,
                          ReduceOp      op,
                          T             init)
    {
        size_t temp_bytes = 0;
        cub::DeviceSegmentedReduce::Reduce(nullptr, temp_bytes, in, out, num_segments,
                                           begin_offsets, end_offsets, op, init);
        void* temp = details::get_temp_buffer(temp_bytes);
        cub::DeviceSegmentedReduce::Reduce(temp, temp_bytes, in, out, num_segments,
                                           begin_offsets, end_offsets, op, init);
    }
};

}  // namespace cuda
}  // namespace gipc
