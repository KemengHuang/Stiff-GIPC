#include <cub/warp/warp_reduce.cuh>
#include <muda/ext/eigen/atomic.h>
namespace muda
{
//constexpr int BlockSize = 128;
//constexpr int WarpSize  = 32;
//using T                 = float;
//constexpr int M         = 3;
//constexpr int N         = 3;

namespace details::fast_segmental_reduce
{
    __host__ __device__ constexpr int b2i(bool b)
    {
        return b ? 1 : 0;
    }
}  // namespace details::fast_segmental_reduce

template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename ReduceOp>
void FastSegmentalReduce<BlockSize, WarpSize>::reduce(CBufferView<int> offset,
                                                      CBufferView<Eigen::Matrix<T, M, N>> in,
                                                      BufferView<Eigen::Matrix<T, M, N>> out,
                                                      ReduceOp op)
{
    using namespace details::fast_segmental_reduce;
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");

    using Matrix = Eigen::Matrix<T, M, N>;

    auto                   size       = in.size();
    constexpr int          warp_size  = WarpSize;
    constexpr unsigned int warp_mask  = ~0u;
    constexpr int          block_dim  = BlockSize;
    constexpr int          warp_count = block_dim / warp_size;

    BufferLaunch(this->stream()).fill<Matrix>(out, Matrix::Zero().eval());

    int block_count = (size + block_dim - 1) / block_dim;

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(size,
               [in     = in.cviewer().name("in"),
                out    = out.viewer().name("out"),
                offset = offset.cviewer().name("offset"),
                op] __device__(int global_thread_id) mutable
               {
                   using WarpReduceT = cub::WarpReduce<T, warp_size>;
                   __shared__ typename WarpReduceT::TempStorage t_storage[warp_count];

                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   int    prev_i  = -1;
                   int    i       = -1;
                   int    is_head = 0;
                   Matrix value;

                   if(global_thread_id > 0)
                   {
                       prev_i = offset(global_thread_id - 1);
                   }

                   i     = offset(global_thread_id);
                   value = in(global_thread_id);

                   if(lane_id == 0 || prev_i != i)
                   {
                       is_head = 1;
                   }

                   for(int j = 0; j < M; j++)
                   {
                       for(int k = 0; k < N; k++)
                       {
                           value(j, k) =
                               WarpReduceT(t_storage[warp_id])
                                   .HeadSegmentedReduce(value(j, k), is_head, op);
                       }
                   }

                   if(is_head)
                   {
                       auto& out_value = out(i);
                       eigen::atomic_add(out_value, value);
                   }
                   //}
               });
}
template <int BlockSize, int WarpSize>
template <typename T, typename ReduceOp>
void FastSegmentalReduce<BlockSize, WarpSize>::reduce(CBufferView<int> offset,
                                                      CBufferView<T>   in,
                                                      BufferView<T>    out,
                                                      ReduceOp         op)
{
    using namespace details::fast_segmental_reduce;
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");

    using ValueT = T;

    auto                   size       = in.size();
    constexpr int          warp_size  = WarpSize;
    constexpr unsigned int warp_mask  = ~0u;
    constexpr int          block_dim  = BlockSize;
    constexpr int          warp_count = block_dim / warp_size;

    BufferLaunch(this->stream()).fill<ValueT>(out, ValueT{0});

    int block_count = (size + block_dim - 1) / block_dim;
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(size,
               [in     = in.cviewer().name("in"),
                out    = out.viewer().name("out"),
                offset = offset.cviewer().name("offset"),
                op     = op] __device__(int global_thread_id) mutable
               {
                   using WarpReduceInt = cub::WarpReduce<int, warp_size>;
                   using WarpReduceT   = cub::WarpReduce<T, warp_size>;


                   __shared__ union
                   {
                       typename WarpReduceInt::TempStorage index_storage[warp_count];
                       typename WarpReduceT::TempStorage t_storage[warp_count];
                   };

                   auto thread_id_in_block = threadIdx.x;
                   auto warp_id            = thread_id_in_block / warp_size;
                   auto lane_id = thread_id_in_block & (warp_size - 1);

                   int    prev_i  = -1;
                   int    i       = -1;
                   int    is_head = 0;
                   ValueT value;

                   if(global_thread_id > 0)
                   {
                       prev_i = offset(global_thread_id - 1);
                   }

                   i     = offset(global_thread_id);
                   value = in(global_thread_id);

                   if(lane_id == 0 || prev_i != i)
                   {
                       is_head = 1;
                   }

                   value = WarpReduceT(t_storage[warp_id]).HeadSegmentedReduce(value, is_head, op);

                   if(is_head)
                   {
                       auto& out_value = out(i);
                       atomic_add(&out_value, value);
                   }
               });
}
}  // namespace muda