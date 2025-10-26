#include <gipc/utils/print_buffer.h>
#include <gipc/utils/timer.h>
namespace gipc
{
template <typename T>
template <typename ReduceOp>
void UniqueReduce<T>::sort_unique_reduce(muda::CBufferView<T>   in,
                                         muda::DeviceBuffer<T>& out,
                                         ReduceOp               op,
                                         T                      init)
{
    

    m_temp_sort_in.resize(in.size());
    m_temp_sort_in.view().copy_from(in);
    {
        Timer timer{__FUNCTION__ "-sort"};
        muda::DeviceMergeSort().SortKeys(
                                         m_temp_sort_in.data(),
                                         in.size(),
                                         [] __host__ __device__(const T& left, const T& right)
                                         { return left < right; });
    }


    // std::cout << "Sorted input: \n" << m_temp_sort_in << std::endl;

    unique_reduce(m_temp_sort_in, out, op, init);
}

template <typename T>
template <typename ReduceOp>
void UniqueReduce<T>::unique_reduce(muda::CBufferView<T>   in,
                                    muda::DeviceBuffer<T>& out,
                                    ReduceOp               op,
                                    T                      init)
{
    m_unique_out.resize(in.size());
    m_unique_counts.resize(in.size());

    {
        Timer timer{__FUNCTION__ "-unique"};
        muda::DeviceRunLengthEncode().Encode(
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

    muda::DeviceScan().ExclusiveSum(
         m_unique_counts.data(), m_unique_offsets.data(), h_unique_num);

    muda::ParallelFor()
        .kernel_name(__FUNCTION__ "-calculate_offset_end")
        .apply(h_unique_num,
               [offsets = m_unique_offsets.viewer().name("offsets"),
                counts = m_unique_counts.viewer().name("counts")] __device__(int i) mutable
               { counts(i) += offsets(i); });

    out.resize(h_unique_num);

    {
        Timer timer{__FUNCTION__ "-reduce"};
        muda::DeviceSegmentedReduce().Reduce(
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