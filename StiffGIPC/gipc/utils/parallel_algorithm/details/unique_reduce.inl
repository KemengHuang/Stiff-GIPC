#include <gipc/utils/print_buffer.h>
#include <gipc/utils/timer.h>
#include <cuda_tools/cuda_all.h>

namespace
{
__global__ void calculate_offset_end_kernel(int size,
                                            cudatool::BufferView<int> offsets,
                                            cudatool::BufferView<int> counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    counts[i] += offsets[i];
}
}  // namespace

namespace gipc
{
template <typename T>
template <typename ReduceOp>
void UniqueReduce<T>::sort_unique_reduce(cudatool::CBufferView<T>   in,
                                         cudatool::DeviceBuffer<T>& out,
                                         ReduceOp               op,
                                         T                      init)
{
    m_temp_sort_in.resize(in.size());
    m_temp_sort_in.view().copy_from(in);
    {
        Timer timer{__FUNCTION__ "-sort"};
        cudatool::DeviceMergeSort().SortKeys(
                                         m_temp_sort_in.data(),
                                         in.size(),
                                         [] __host__ __device__(const T& left, const T& right)
                                         { return left < right; });
    }

    unique_reduce(m_temp_sort_in, out, op, init);
}

template <typename T>
template <typename ReduceOp>
void UniqueReduce<T>::unique_reduce(cudatool::CBufferView<T>   in,
                                    cudatool::DeviceBuffer<T>& out,
                                    ReduceOp               op,
                                    T                      init)
{
    m_unique_out.resize(in.size());
    m_unique_counts.resize(in.size());

    {
        Timer timer{__FUNCTION__ "-unique"};
        cudatool::DeviceRunLengthEncode().Encode(
                                             in.data(),
                                             m_unique_out.data(),
                                             m_unique_counts.data(),
                                             m_unique_num.data(),
                                             in.size());
    }

    int h_unique_num = m_unique_num;
    m_unique_offsets.resize(h_unique_num);
    m_unique_counts.resize(h_unique_num);
    m_unique_out.resize(h_unique_num);

    cudatool::DeviceScan().ExclusiveSum(
         m_unique_counts.data(), m_unique_offsets.data(), h_unique_num);

    LaunchCudaKernal_default(h_unique_num,
                             256,
                             0,
                             calculate_offset_end_kernel,
                             h_unique_num,
                             m_unique_offsets.view(),
                             m_unique_counts.view());

    out.resize(h_unique_num);

    {
        Timer timer{__FUNCTION__ "-reduce"};
        cudatool::DeviceSegmentedReduce().Reduce(
                                             in.data(),
                                             out.data(),
                                             out.size(),
                                             m_unique_offsets.data(),
                                             m_unique_counts.data(),
                                             op,
                                             init);
    }
}
}  // namespace gipc
