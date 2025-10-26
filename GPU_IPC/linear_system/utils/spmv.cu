#include <linear_system/utils/spmv.h>
#include <muda/launch/launch.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>
#include <muda/buffer/device_var.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/util_math.cuh>
namespace gipc
{
void Spmv::sym_spmv(Float                          a,
                    muda::CBSRMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>  x,
                    Float                          b,
                    muda::DenseVectorView<Float>   y)
{
}

void Spmv::sym_spmv(Float                           a,
                    muda::CBCOOMatrixView<Float, 3> A,
                    muda::CDenseVectorView<Float>   x,
                    Float                           b,
                    muda::DenseVectorView<Float>    y)
{

    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    muda::ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int index) mutable
               {
                   auto&& [i, j, block] = A(index);

                   if(i == j)  // diagonal block
                   {
                       auto seg_x = x.segment<N>(j * N);

                       Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                       auto                result = a * block * vec_x;

                       auto seg_y = y.segment<N>(i * N);
                       seg_y.atomic_add(result.eval());
                   }
                   else  // off-diagonal block
                   {
                       // ij-th block
                       {
                           auto seg_x = x.segment<N>(j * N);

                           Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                           auto                result = a * block * vec_x;

                           auto seg_y = y.segment<N>(i * N);
                           seg_y.atomic_add(result.eval());
                       }

                       // ji-th block
                       {
                           auto seg_x = x.segment<N>(i * N);

                           Eigen::Vector<T, N> vec_x = seg_x.as_eigen();
                           auto result = a * block.transpose() * vec_x;

                           auto seg_y = y.segment<N>(j * N);
                           seg_y.atomic_add(result.eval());
                       }
                   }
               });
}

__host__ __device__ constexpr int b2i(bool b)
{
    return b ? 1 : 0;
}

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

// __device__​ unsigned __fns ( unsigned mask, unsigned base, int  offset )

// find ths n-th set bit in mask, starting from base
__device__ __forceinline__ unsigned fns(unsigned mask, unsigned base, int offset)
{
    return __fns(mask, base, offset);
}

__device__ __forceinline__ int popc(unsigned long long int x)
{
    return __popc(x);
}

void Spmv::warp_reduce_spmv(Float                           a,
                            muda::CBCOOMatrixView<Float, 3> A,
                            muda::CDenseVectorView<Float>   x,
                            Float                           b,
                            muda::DenseVectorView<Float>    y)
{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 128;
    int block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a = a,
             A = A.viewer().name("A"),
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceInt     = cub::WarpReduce<int, warp_size>;
                using WarpReduceFloat   = cub::WarpReduce<Float, warp_size>;
                using WarpScanInt       = cub::WarpScan<int>;
                using BlockScan         = cub::BlockScan<int, block_dim>;
                using BlockShuffle      = cub::BlockShuffle<int, block_dim>;
                using BlockShuffleFloat = cub::BlockShuffle<Float, block_dim>;
                using BlockShuffleVector3 =
                    cub::BlockShuffle<Eigen::Vector<T, N>, block_dim>;

                auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                int rest = A.triplet_count() - blockIdx.x * block_dim;
                int valid_count_in_block = rest > block_dim ? block_dim : rest;

                __shared__ union
                {
                    typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                    typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                };

                int prev_i = -1;
                int next_i = -1;
                int i      = -1;

                Flags   flags;
                Vector3 vec;
                flags.is_cross_warp = 0;


                if(global_thread_id > 0 && global_thread_id < A.triplet_count())
                {
                    auto prev_triplet = A(global_thread_id - 1);
                    prev_i            = prev_triplet.block_row_index;
                }

                if(global_thread_id < A.triplet_count() - 1)
                {
                    auto next_triplet = A(global_thread_id + 1);
                    next_i            = next_triplet.block_row_index;
                }

                if(global_thread_id < A.triplet_count())
                {
                    auto Triplet = A(global_thread_id);
                    i            = Triplet.block_row_index;
                    auto j       = Triplet.block_col_index;

                    vec = Triplet.block_value * x.segment<N>(j * N).as_eigen();

                    flags.is_valid = 1;
                }
                else
                {
                    i = -1;
                    vec.setZero();
                    flags.is_valid      = 0;
                    flags.is_cross_warp = 0;
                }

                if(lane_id == 0)
                {
                    flags.is_head = 1;
                    // if this thread is the first thread in the warp
                    // check if the previous triplet is in the same row
                    // if so, this row crosses the warp boundary, we need use atomic add
                    flags.is_cross_warp = b2i(prev_i == i);
                }
                else
                {
                    flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined

                    if(lane_id == warp_size - 1)
                    {
                        // if this thread is the last thread in the warp
                        // check if the next triplet is in the same row
                        // if so, this row crosses the warp boundary, we need use atomic add
                        flags.is_cross_warp = b2i(next_i == i);
                    }
                }

                flags.flags = WarpReduceInt(temp_storage_int[warp_id])
                                  .HeadSegmentedReduce(flags.flags,
                                                       flags.is_head,
                                                       [](uint32_t a, uint32_t b)
                                                       { return a + b; });

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });


                // cub::WARP_SYNC(warp_mask);

                flags.is_head = b2i(flags.is_head && flags.is_valid);

                flags.b2i();
                int      is_head_mask = __ballot_sync(warp_mask, flags.is_head);
                uint32_t offset  = fns(is_head_mask, 0, lane_id + 1);

                int valid_bit    = (offset != ~0u);
                int shuffle_mask = __ballot_sync(warp_mask, valid_bit);

                i = cub::ShuffleIndex<32>(i, offset, shuffle_mask);
                flags.flags = cub::ShuffleIndex<32>(flags.flags, offset, shuffle_mask);
                vec.x() = cub::ShuffleIndex<32>(vec.x(), offset, shuffle_mask);
                vec.y() = cub::ShuffleIndex<32>(vec.y(), offset, shuffle_mask);
                vec.z() = cub::ShuffleIndex<32>(vec.z(), offset, shuffle_mask);


                // int valid_bit = true;


                if(valid_bit && flags.is_head && flags.is_valid)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    if(flags.is_cross_warp)
                    {
                        seg_y.atomic_add(result.eval());
                        // printf("atomic add to [%d]\n", i);
                    }
                    else
                    {
                        //seg_y.atomic_add(result.eval());
                        seg_y.as_eigen() += result.eval();
                        // printf("non atomic add to [%d]\n", i);
                    }
                }

                //__shared__ int warp_id_to_valid_count[block_dim / warp_size];
                //__shared__ int warp_id_to_valid_offset[block_dim / warp_size];

                //if(lane_id == 0)
                //    warp_id_to_valid_count[warp_id] = popc(shuffle_mask);

                //cub::CTA_SYNC();

                //if(thread_id_in_block == 0)  // exclusive sum
                //{
                //    int sum = 0;
                //    for(int i = 0; i < warp_size; i++)
                //    {
                //        warp_id_to_valid_offset[i] = sum;
                //        sum += warp_id_to_valid_count[i];
                //    }
                //}

                //// search where the thread is in the warp
                //auto W = 0;
                //auto l = 0;
                //auto r = block_dim / warp_size;
                //while(l < r)
                //{
                //    auto m = (l + r) / 2;
                //    if(warp_id_to_valid_offset[m] <= warp_id)
                //    {
                //        W = m;
                //        l = m + 1;
                //    }
                //    else
                //    {
                //        r = m;
                //    }
                //}

                //auto last_W = W == 0 ? 0 : W - 1;
                //auto offset_in_W = thread_id_in_block - warp_id_to_valid_offset[W];

                //int offset_in_block = 0;
                //int total_valid_count = warp_id_to_valid_count[warp_size - 1]
                //                        + warp_id_to_valid_offset[warp_size - 1];
                //if(thread_id_in_block < total_valid_count)
                //    offset_in_block = warp_size * last_W + offset_in_W;

                //BlockShuffle().Offset(i, i, offset_in_block);
                //BlockShuffle().Offset(rd.flags, rd.flags, offset_in_block);
                //BlockShuffleFloat().Offset(vec.x(), vec.x(), offset_in_block);
                //BlockShuffleFloat().Offset(vec.y(), vec.y(), offset_in_block);
                //BlockShuffleFloat().Offset(vec.z(), vec.z(), offset_in_block);


                //if(thread_id_in_block < total_valid_count && rd.is_head && rd.is_valid)
                //{
                //    auto seg_y  = y.segment<N>(i * N);
                //    auto result = a * vec;

                //    if(rd.is_cross_warp)
                //    {
                //        seg_y.atomic_add(result.eval());
                //    }
                //    else
                //    {
                //        seg_y.as_eigen() += result.eval();
                //    }
                //}
            });

    //int atomic_count_host     = atomic_count;
    //int non_atomic_count_host = non_atomic_count;

    //std::cout << "atomic_count: " << atomic_count_host
    //          << ", non_atomic_count: " << non_atomic_count_host << std::endl;
}

void Spmv::warp_reduce_sym_spmv(Float                           a,
                                muda::CBCOOMatrixView<Float, 3> A,
                                muda::CDenseVectorView<Float>   x,
                                Float                           b,
                                muda::DenseVectorView<Float>    y)

{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 64;
    int block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a = a,
             A = A.viewer().name("A"),
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceInt     = cub::WarpReduce<int, warp_size>;
                using WarpReduceFloat   = cub::WarpReduce<Float, warp_size>;
                using WarpScanInt       = cub::WarpScan<int>;
                using BlockScan         = cub::BlockScan<int, block_dim>;
                using BlockShuffle      = cub::BlockShuffle<int, block_dim>;
                using BlockShuffleFloat = cub::BlockShuffle<Float, block_dim>;
                using BlockShuffleVector3 =
                    cub::BlockShuffle<Eigen::Vector<T, N>, block_dim>;

                auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                int rest = A.triplet_count() - blockIdx.x * block_dim;
                int valid_count_in_block = rest > block_dim ? block_dim : rest;

                __shared__ union
                {
                    typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                    typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                };

                int     prev_i = -1;
                int     next_i = -1;
                int     i      = -1;
                Flags   flags;
                Vector3 vec;

                // In symmtric version, we don't need to check the cross warp
                flags.is_cross_warp = 0;

                // set the previous row index
                if(global_thread_id > 0 && global_thread_id < A.triplet_count())
                {
                    auto prev_triplet = A(global_thread_id - 1);
                    prev_i            = prev_triplet.block_row_index;
                }

                // set the next row index
                if(global_thread_id < A.triplet_count() - 1 /* && global_thread_id>=0 */)
                {
                    auto next_triplet = A(global_thread_id + 1);
                    next_i            = next_triplet.block_row_index;
                }

                if(global_thread_id < A.triplet_count())
                {
                    auto Triplet = A(global_thread_id);
                    i            = Triplet.block_row_index;
                    auto j       = Triplet.block_col_index;

                    vec = Triplet.block_value * x.segment<N>(j * N).as_eigen();

                    flags.is_valid = 1;

                    if(i != j)  // process lower triangle
                    {
                        Vector3 vec_ = a * Triplet.block_value.transpose()
                                       * x.segment<N>(i * N).as_eigen();

                        y.segment<N>(j * N).atomic_add(vec_);
                    }
                }
                else
                {
                    i = -1;
                    vec.setZero();
                    flags.is_valid = 0;
                }

                if(lane_id == 0)
                {
                    flags.is_head = 1;
                }
                else
                {
                    flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined
                }


                // ----------------------------------- warp reduce ----------------------------------------------
                flags.flags = WarpReduceInt(temp_storage_int[warp_id])
                                  .HeadSegmentedReduce(flags.flags,
                                                       flags.is_head,
                                                       [](uint32_t a, uint32_t b)
                                                       { return a + b; });

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags.is_head,
                                                   [](Float a, Float b)
                                                   { return a + b; });
                // ----------------------------------- warp reduce -----------------------------------------------


                if(flags.is_head && flags.is_valid)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    // Must use atomic add!
                    // Because the same row may be processed by different warps
                    seg_y.atomic_add(result.eval());
                }
            });
}
}  // namespace gipc
