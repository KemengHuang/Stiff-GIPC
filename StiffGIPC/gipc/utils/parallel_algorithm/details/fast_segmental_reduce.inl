#include <cub/warp/warp_reduce.cuh>
#include <cuda_tools/cuda_all.h>

namespace cudatool::parallel
{
namespace details::fast_segmental_reduce
{
    __host__ __device__ constexpr int b2i(bool b)
    {
        return b ? 1 : 0;
    }
}  // namespace details::fast_segmental_reduce

namespace
{
template <typename T, int M, int N, int BlockSize, int WarpSize, typename ReduceOp>
__global__ void fast_segmental_reduce_ptr_kernel(int                     length,
                                                 uint32_t*               offset_in,
                                                 Eigen::Matrix<T, M, N>* input,
                                                 Eigen::Matrix<T, M, N>* output,
                                                 ReduceOp                op)
{
    using namespace details::fast_segmental_reduce;
    using Matrix = Eigen::Matrix<T, M, N>;

    constexpr int warp_size  = WarpSize;
    constexpr int block_dim  = BlockSize;
    constexpr int warp_count = block_dim / warp_size;

    using WarpReduceT = cub::WarpReduce<T, warp_size>;
    __shared__ typename WarpReduceT::TempStorage t_storage[warp_count];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_thread_id >= length)
        return;

    auto thread_id_in_block = threadIdx.x;
    auto warp_id            = thread_id_in_block / warp_size;
    auto lane_id            = thread_id_in_block & (warp_size - 1);

    int    prev_i  = -1;
    int    i       = -1;
    int    is_head = 0;
    Matrix value;

    if(global_thread_id > 0)
    {
        prev_i = offset_in[global_thread_id - 1];
    }

    i     = offset_in[global_thread_id];
    value = input[global_thread_id];

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
        cudatool::eigen::atomic_add(output[i], value);
    }
}

template <typename T, int M, int N, int BlockSize, int WarpSize, typename ReduceOp>
__global__ void fast_segmental_reduce_matrix_view_kernel(
    int                                   size,
    CBufferView<int>                      offset,
    CBufferView<Eigen::Matrix<T, M, N>>   in,
    BufferView<Eigen::Matrix<T, M, N>>    out,
    ReduceOp                              op)
{
    using namespace details::fast_segmental_reduce;
    using Matrix = Eigen::Matrix<T, M, N>;

    constexpr int warp_size  = WarpSize;
    constexpr int block_dim  = BlockSize;
    constexpr int warp_count = block_dim / warp_size;

    using WarpReduceT = cub::WarpReduce<T, warp_size>;
    __shared__ typename WarpReduceT::TempStorage t_storage[warp_count];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_thread_id >= size)
        return;

    auto thread_id_in_block = threadIdx.x;
    auto warp_id            = thread_id_in_block / warp_size;
    auto lane_id            = thread_id_in_block & (warp_size - 1);

    int    prev_i  = -1;
    int    i       = -1;
    int    is_head = 0;
    Matrix value;

    if(global_thread_id > 0)
    {
        prev_i = offset[global_thread_id - 1];
    }

    i     = offset[global_thread_id];
    value = in[global_thread_id];

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
        auto& out_value = out[i];
        cudatool::eigen::atomic_add(out_value, value);
    }
}

template <typename T, int BlockSize, int WarpSize, typename ReduceOp>
__global__ void fast_segmental_reduce_scalar_view_kernel(int       size,
                                                         CBufferView<int> offset,
                                                         CBufferView<T>   in,
                                                         BufferView<T>    out,
                                                         ReduceOp         op)
{
    using namespace details::fast_segmental_reduce;
    using ValueT = T;

    constexpr int warp_size  = WarpSize;
    constexpr int block_dim  = BlockSize;
    constexpr int warp_count = block_dim / warp_size;

    using WarpReduceInt = cub::WarpReduce<int, warp_size>;
    using WarpReduceT   = cub::WarpReduce<T, warp_size>;

    __shared__ union
    {
        typename WarpReduceInt::TempStorage index_storage[warp_count];
        typename WarpReduceT::TempStorage t_storage[warp_count];
    };

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_thread_id >= size)
        return;

    auto thread_id_in_block = threadIdx.x;
    auto warp_id            = thread_id_in_block / warp_size;
    auto lane_id            = thread_id_in_block & (warp_size - 1);

    int    prev_i  = -1;
    int    i       = -1;
    int    is_head = 0;
    ValueT value;

    if(global_thread_id > 0)
    {
        prev_i = offset[global_thread_id - 1];
    }

    i     = offset[global_thread_id];
    value = in[global_thread_id];

    if(lane_id == 0 || prev_i != i)
    {
        is_head = 1;
    }

    value = WarpReduceT(t_storage[warp_id]).HeadSegmentedReduce(value, is_head, op);

    if(is_head)
    {
        auto& out_value = out[i];
        cudatool::atomic_add(&out_value, value);
    }
}

template <typename T>
__global__ void fill_kernel(T* data, size_t size, T value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    data[i] = value;
}
}  // namespace


template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename ReduceOp>
void FastSegmentalReduce<BlockSize, WarpSize>::reduce(int       length,
                                                      uint32_t* offset_in,
                                                      Eigen::Matrix<T, M, N>* input,
                                                      Eigen::Matrix<T, M, N>* output,
                                                      ReduceOp op)
{
    using namespace details::fast_segmental_reduce;
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");

    auto                   size       = length;
    constexpr int          block_dim  = BlockSize;

    LaunchCudaKernal_default(size,
                             block_dim,
                             0,
                             fast_segmental_reduce_ptr_kernel<T, M, N, BlockSize, WarpSize, ReduceOp>,
                             length,
                             offset_in,
                             input,
                             output,
                             op);
}


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
    constexpr int          block_dim  = BlockSize;

    LaunchCudaKernal_default(out.size(),
                             256,
                             0,
                             fill_kernel<Matrix>,
                             out.data(),
                             out.size(),
                             Matrix::Zero().eval());

    LaunchCudaKernal_default(size,
                             block_dim,
                             0,
                             fast_segmental_reduce_matrix_view_kernel<T, M, N, BlockSize, WarpSize, ReduceOp>,
                             size,
                             offset,
                             in,
                             out,
                             op);
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
    constexpr int          block_dim  = BlockSize;

    LaunchCudaKernal_default(out.size(),
                             256,
                             0,
                             fill_kernel<ValueT>,
                             out.data(),
                             out.size(),
                             ValueT{0});

    LaunchCudaKernal_default(size,
                             block_dim,
                             0,
                             fast_segmental_reduce_scalar_view_kernel<T, BlockSize, WarpSize, ReduceOp>,
                             size,
                             offset,
                             in,
                             out,
                             op);
}
}  // namespace cudatool::parallel
