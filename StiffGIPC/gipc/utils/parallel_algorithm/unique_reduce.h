#pragma once
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
namespace gipc
{
template <typename T>
class UniqueReduce
{
    gipc::cuda::DeviceBuffer<std::byte> m_workspace;
    gipc::cuda::DeviceBuffer<T>         m_unique_out;
    gipc::cuda::DeviceVar<int>          m_unique_num;
    gipc::cuda::DeviceBuffer<int>       m_unique_offsets;
    gipc::cuda::DeviceBuffer<int>       m_unique_counts;
    gipc::cuda::DeviceBuffer<T>         m_temp_sort_in;

  public:
    UniqueReduce() = default;
    template <typename ReduceOp>
    void sort_unique_reduce(gipc::cuda::CBufferView<T> in, gipc::cuda::DeviceBuffer<T>& out, ReduceOp op, T init);
    template <typename ReduceOp>
    void unique_reduce(gipc::cuda::CBufferView<T> in, gipc::cuda::DeviceBuffer<T>& out, ReduceOp op, T init);
};
}  // namespace gipc

#include "details/unique_reduce.inl"