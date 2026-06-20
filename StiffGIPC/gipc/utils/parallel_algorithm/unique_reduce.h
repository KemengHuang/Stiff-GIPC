#pragma once
#include <cuda_tools/cuda_all.h>

namespace gipc
{
template <typename T>
class UniqueReduce
{
    cudatool::DeviceBuffer<std::byte> m_workspace;
    cudatool::DeviceBuffer<T>         m_unique_out;
    cudatool::DeviceVar<int>          m_unique_num;
    cudatool::DeviceBuffer<int>       m_unique_offsets;
    cudatool::DeviceBuffer<int>       m_unique_counts;
    cudatool::DeviceBuffer<T>         m_temp_sort_in;

  public:
    UniqueReduce() = default;
    template <typename ReduceOp>
    void sort_unique_reduce(cudatool::CBufferView<T> in, cudatool::DeviceBuffer<T>& out, ReduceOp op, T init);
    template <typename ReduceOp>
    void unique_reduce(cudatool::CBufferView<T> in, cudatool::DeviceBuffer<T>& out, ReduceOp op, T init);
};
}  // namespace gipc

#include "details/unique_reduce.inl"
