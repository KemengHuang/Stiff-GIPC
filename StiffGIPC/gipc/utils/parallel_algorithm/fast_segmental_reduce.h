#pragma once
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_cub_wrappers.h>
#include <Eigen/Core>
#include <cub/util_type.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/warp/warp_reduce.cuh>

namespace cudatool::parallel
{
template <int BlockSize = 128, int WarpSize = 32>
class FastSegmentalReduce
{
  public:
    template <typename T, int M, int N, typename ReduceOp = cudatool::Plus<T>>
    static void reduce(CBufferView<int>                    dst,
                       CBufferView<Eigen::Matrix<T, M, N>> in,
                       BufferView<Eigen::Matrix<T, M, N>>  out,
                       ReduceOp                            op = ReduceOp{});

    template <typename T, int M, int N, typename ReduceOp = cudatool::Plus<T>>
    static void reduce(int                     length,
                       uint32_t*               offset_in,
                       Eigen::Matrix<T, M, N>* input,
                       Eigen::Matrix<T, M, N>* output,
                       ReduceOp                op = ReduceOp{});

    template <typename T, typename ReduceOp = cudatool::Plus<T>>
    static void reduce(CBufferView<int> dst, CBufferView<T> in, BufferView<T> out, ReduceOp op = ReduceOp{});
};
}  // namespace cudatool::parallel

#include "details/fast_segmental_reduce.inl"
