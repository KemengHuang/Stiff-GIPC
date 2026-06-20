#include <linear_system/utils/spmv.h>
#include <gipc/cuda/all.h>
#include <cub/warp/warp_reduce.cuh>

namespace gipc
{

void Spmv::warp_reduce_sym_spmv(Float                         a,
                                Eigen::Matrix3d*              triplet_values,
                                int*                          row_ids,
                                int*                          col_ids,
                                int                           triplet_count,
                                gipc::cuda::CDenseVectorView<Float> x,
                                Float                         b,
                                gipc::cuda::DenseVectorView<Float>  y)

{
    using namespace gipc::cuda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        gipc::cuda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        gipc::cuda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 256;
    int block_count = (triplet_count + block_dim - 1) / block_dim;

    gipc::cuda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a     = a,
             Mats3 = triplet_values,
             rows  = row_ids,
             cols  = col_ids,
             triplet_count,
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
                if(global_thread_id >= triplet_count)
                    return;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];

                int     prev_i = -1;
                int     i      = -1;
                char    flags;
                Vector3 vec;

                // set the previous row index
                if(global_thread_id > 0)
                {
                    //auto prev_triplet = A(global_thread_id - 1);
                    prev_i = rows[global_thread_id - 1];
                }


                {
                    //auto Triplet = Mats3[];
                    i                = rows[global_thread_id];
                    auto j           = cols[global_thread_id];
                    auto block_value = Mats3[global_thread_id];
                    vec = block_value * x.segment<N>(j * N).as_eigen();

                    //flags.is_valid = 1;

                    if(i != j)  // process lower triangle
                    {
                        Vector3 vec_ = a * block_value.transpose()
                                       * x.segment<N>(i * N).as_eigen();

                        y.segment<N>(j * N).atomic_add(vec_);
                    }
                }


                if((lane_id == 0) || (prev_i != i))
                {
                    flags = 1;
                }
                else
                {
                    flags = 0;
                }

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });
                // ----------------------------------- warp reduce -----------------------------------------------


                if(flags)
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
