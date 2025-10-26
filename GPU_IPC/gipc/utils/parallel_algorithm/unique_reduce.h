#pragma once
#include <muda/cub/device/device_segmented_reduce.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_merge_sort.h>
namespace gipc
{
template <typename T>
class UniqueReduce
{
    muda::DeviceBuffer<std::byte> m_workspace;
    muda::DeviceBuffer<T>         m_unique_out;
    muda::DeviceVar<int>          m_unique_num;
    muda::DeviceBuffer<int>       m_unique_offsets;
    muda::DeviceBuffer<int>       m_unique_counts;
    muda::DeviceBuffer<T>         m_temp_sort_in;

  public:
    UniqueReduce() = default;
    template <typename ReduceOp>
    void sort_unique_reduce(muda::CBufferView<T> in, muda::DeviceBuffer<T>& out, ReduceOp op, T init);
    template <typename ReduceOp>
    void unique_reduce(muda::CBufferView<T> in, muda::DeviceBuffer<T>& out, ReduceOp op, T init);
};
}  // namespace gipc

#include "details/unique_reduce.inl"