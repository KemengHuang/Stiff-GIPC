#pragma once
#include <gipc/cuda/all.h>
#include <Eigen/Core>
#include <cub/util_type.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cuda/std/functional>

namespace gipc::cuda
{
template <int BlockSize = 128, int WarpSize = 32>
class FastSegmentalReduce : public LaunchBase<FastSegmentalReduce<BlockSize, WarpSize>>
{
    using Base = LaunchBase<FastSegmentalReduce<BlockSize, WarpSize>>;

    struct Flags
    {
        union
        {
            struct
            {
                unsigned char is_head;
                unsigned char is_cross_warp;
                unsigned char is_valid;
            };
            unsigned int flags;
        };

        __host__ __device__ void b2i()
        {
            is_head       = is_head ? 1 : 0;
            is_cross_warp = is_cross_warp ? 1 : 0;
            is_valid      = is_valid ? 1 : 0;
        }
    };

  public:
    FastSegmentalReduce(cudaStream_t s = nullptr)
        : Base(s)
    {
    }

    template <typename T, int M, int N, typename ReduceOp = Plus<T>>
    void reduce(CBufferView<int>                    dst,
                CBufferView<Eigen::Matrix<T, M, N>> in,
                BufferView<Eigen::Matrix<T, M, N>>  out,
                ReduceOp                            op = ReduceOp{});

    template <typename T, int M, int N, typename ReduceOp = Plus<T>>
    void reduce(int                     length,
                uint32_t*               offset_in,
                Eigen::Matrix<T, M, N>* input,
                Eigen::Matrix<T, M, N>* output,
                ReduceOp                op = ReduceOp{});

    template <typename T, typename ReduceOp = Plus<T>>
    void reduce(CBufferView<int> dst, CBufferView<T> in, BufferView<T> out, ReduceOp op = ReduceOp{});
};
}  // namespace gipc::cuda

#include "details/fast_segmental_reduce.inl"
