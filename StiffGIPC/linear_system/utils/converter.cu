#include <linear_system/utils/converter.h>
#include <muda/cub/device/device_merge_sort.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_segmented_reduce.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_select.h>
#include <cub/warp/warp_reduce.cuh>
#include <muda/ext/eigen/atomic.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/cub/device/device_partition.h>

namespace gipc
{
void print_matrix(const muda::DeviceTripletMatrix<Float, 3>& mat)
{
    auto row_indices = mat.block_row_indices();
    auto col_indices = mat.block_col_indices();
    auto blocks      = mat.block_values();


    std::vector<int>       row_indices_host(row_indices.size());
    std::vector<int>       col_indices_host(col_indices.size());
    std::vector<Matrix3x3> blocks_host(blocks.size());

    row_indices.copy_to(row_indices_host.data());
    col_indices.copy_to(col_indices_host.data());
    blocks.copy_to(blocks_host.data());

    for(int i = 0; i < row_indices_host.size(); i++)
    {
        std::cout << "(" << row_indices_host[i] << "," << col_indices_host[i]
                  << ")" << std::endl
                  << " block: " << std::endl
                  << blocks_host[i] << std::endl;
    }
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(const muda::DeviceTripletMatrix<T, N>& from,
                        muda::DeviceBCOOMatrix<T, N>&          to)
{
    gipc::Timer timer("convert3x3");
    to.reshape(from.block_rows(), from.block_cols());
    to.resize_triplets(from.triplet_count());


    if(to.triplet_count() == 0)
        return;


    if constexpr(UseRadixSort)
    {
        // NOTE: this branch is faster

        //Timer timer("radix_sort_indices_and_blocks");
        _radix_sort_indices_and_blocks(from, to);
    }
    else
    {
        //Timer timer("merge_sort_indices_and_blocks");
        _merge_sort_indices_and_blocks(from, to);
    }


    if constexpr(UseReduceByKey)
    {
        //Timer timer("make_unique_indices_and_blocks");
        _make_unique_indices_and_blocks(from, to);
    }
    else
    {
        // NOTE: on 4060 laptop, this branch is faster

        {
            //Timer timer("make_unique_indices");
            _make_unique_indices(from, to);
        }

        {
            // Timer timer("make_unique_blocks");
            // _make_unique_blocks_naive(from, to);
            // _make_unique_blocks_task_based(from, to);
            _make_unique_block_warp_reduction(from, to);
        }
    }
}


void Converter::_merge_sort_indices_and_blocks(const muda::DeviceTripletMatrix<T, N>& from,
                                               muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = from.block_row_indices();
    auto src_col_indices = from.block_col_indices();
    auto src_blocks      = from.block_values();

    loose_resize(sort_index, src_row_indices.size());
    loose_resize(ij_pairs, src_row_indices.size());


    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_pairs    = ij_pairs.viewer().name("ij_pairs"),
                sort_index = sort_index.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   sort_index(i) = i;
               });

    DeviceMergeSort().SortPairs(ij_pairs.data(),
                                sort_index.data(),
                                ij_pairs.size(),
                                [] __device__(const int2& a, const int2& b) {
                                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                                });

    auto dst_row_indices = to.block_row_indices();
    auto dst_col_indices = to.block_col_indices();

    // sort the block values

    loose_resize(blocks_sorted, from.block_values().size());

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(src_blocks.size(),
               [src_blocks = src_blocks.cviewer().name("blocks"),
                sort_index = sort_index.cviewer().name("sort_index"),
                dst_blocks = blocks_sorted.viewer().name("block_values")] __device__(int i) mutable
               { dst_blocks(i) = src_blocks(sort_index(i)); });
}


void Converter::_radix_sort_indices_and_blocks(const muda::DeviceTripletMatrix<T, N>& from,
                                               muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = from.block_row_indices();
    auto src_col_indices = from.block_col_indices();
    auto src_blocks      = from.block_values();

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    ij_pairs.resize(src_row_indices.size());


    // hash ij
    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_hash(i) =
                       (uint64_t{row_indices(i)} << 32) + uint64_t{col_indices(i)};
                   sort_index(i) = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size());

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.block_row_indices();
    auto dst_col_indices = to.block_col_indices();

    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(dst_row_indices.size(),
               [ij_hash = ij_hash.viewer().name("ij_hash"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = int{hash >> 32};
                   auto col_index = int{hash & 0xFFFFFFFF};
                   ij_pairs(i).x  = row_index;
                   ij_pairs(i).y  = col_index;
               });

    // sort the block values

    {
        Timer timer("set block values");
        loose_resize(blocks_sorted, from.block_values().size());
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(src_blocks.size(),
                   [src_blocks = src_blocks.cviewer().name("blocks"),
                    sort_index = sort_index.cviewer().name("sort_index"),
                    dst_blocks = blocks_sorted.viewer().name("block_values")] __device__(int i) mutable
                   { dst_blocks(i) = src_blocks(sort_index(i)); });
    }
}

void Converter::_radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = to.block_row_indices();
    auto src_col_indices = to.block_col_indices();
    auto src_blocks      = to.block_values();

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    ij_pairs.resize(src_row_indices.size());


    // hash ij
    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_hash(i) =
                       (uint64_t{row_indices(i)} << 32) + uint64_t{col_indices(i)};
                   sort_index(i) = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size());

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.block_row_indices();
    auto dst_col_indices = to.block_col_indices();

    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(dst_row_indices.size(),
               [ij_hash = ij_hash.viewer().name("ij_hash"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = int{hash >> 32};
                   auto col_index = int{hash & 0xFFFFFFFF};
                   ij_pairs(i).x  = row_index;
                   ij_pairs(i).y  = col_index;
               });

    // sort the block values

    {
        Timer timer("set indice & block values");
        loose_resize(blocks_sorted, to.block_values().size());
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(src_blocks.size(),
                   [src_blocks = src_blocks.cviewer().name("blocks"),
                    sort_index = sort_index.cviewer().name("sort_index"),
                    ij_pairs   = ij_pairs.cviewer().name("ij_pairs"),
                    dst_row = to.block_row_indices().viewer().name("row_indices"),
                    dst_col = to.block_col_indices().viewer().name("col_indices"),

                    dst_blocks = blocks_sorted.viewer().name("block_values")] __device__(int i) mutable
                   {
                       dst_blocks(i) = src_blocks(sort_index(i));
                       dst_row(i)    = ij_pairs(i).x;
                       dst_col(i)    = ij_pairs(i).y;
                   });

        to.block_values().copy_from(blocks_sorted);
    }
}

void Converter::_make_unique_indices(const muda::DeviceTripletMatrix<T, N>& from,
                                     muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto row_indices = to.block_row_indices();
    auto col_indices = to.block_col_indices();

    loose_resize(unique_ij_pairs, ij_pairs.size());
    loose_resize(unique_counts, ij_pairs.size());


    DeviceRunLengthEncode().Encode(ij_pairs.data(),
                                   unique_ij_pairs.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   ij_pairs.size());

    int h_count = count;

    unique_ij_pairs.resize(h_count);
    unique_counts.resize(h_count);

    offsets.resize(unique_counts.size() + 1);  // +1 for the last offset_end

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());


    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(unique_counts.size(),
               [unique_ij_pairs = unique_ij_pairs.viewer().name("unique_ij_pairs"),
                row_indices = row_indices.viewer().name("row_indices"),
                col_indices = col_indices.viewer().name("col_indices")] __device__(int i) mutable
               {
                   row_indices(i) = unique_ij_pairs(i).x;
                   col_indices(i) = unique_ij_pairs(i).y;
               });

    to.resize_triplets(h_count);
}

void Converter::_make_unique_indices_and_blocks(const muda::DeviceTripletMatrix<T, N>& from,
                                                muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    // alias to reuse the memory
    auto& unique_ij_hash = ij_hash_input;

    if(ij_hash.size() == 0)
    {

        auto src_row_indices = from.block_row_indices();
        auto src_col_indices = from.block_col_indices();

        // hash ij
        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(src_row_indices.size(),
                   [row_indices = src_row_indices.cviewer().name("row_indices"),
                    col_indices = src_col_indices.cviewer().name("col_indices"),
                    ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                    sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable {
                       ij_hash(i) = (uint64_t{row_indices(i)} << 32)
                                    + uint64_t{col_indices(i)};
                   });
    }

    muda::DeviceReduce().ReduceByKey(
        ij_hash.data(),
        unique_ij_hash.data(),
        blocks_sorted.data(),
        to.block_values().data(),
        count.data(),
        [] CUB_RUNTIME_FUNCTION(const Matrix3x3& l, const Matrix3x3& r) -> Matrix3x3
        { return l + r; },
        ij_hash.size());

    int h_count = count;

    to.resize_triplets(h_count);

    // set ij_hash back to row_indices and col_indices
    ParallelFor()
        .kernel_name("set col row indices")
        .apply(to.block_row_indices().size(),
               [ij_hash = unique_ij_hash.viewer().name("ij_hash"),
                row_indices = to.block_row_indices().viewer().name("row_indices"),
                col_indices = to.block_col_indices().viewer().name("col_indices")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = int{hash >> 32};
                   auto col_index = int{hash & 0xFFFFFFFF};
                   row_indices(i) = row_index;
                   col_indices(i) = col_index;
               });
}

void Converter::_make_unique_blocks_task_based(const muda::DeviceTripletMatrix<T, N>& from,
                                               muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto row_indices = to.block_row_indices();
    auto col_indices = to.block_col_indices();
    auto blocks      = to.block_values();

    int h_count = unique_counts.size();
    warp_count_per_unique_block.resize(h_count);
    warp_offset_per_unique_block.resize(h_count);

    //           UBlock1 UBlock2 UBlock3 UBlock4
    // WarpCount    3       4       5       3
    // WarpOffset   0       3       7       12
    ParallelFor(256)
        .kernel_name("classify")
        .apply(unique_counts.size(),
               [offsets    = offsets.viewer().name("offset"),
                counts     = unique_counts.viewer().name("count"),
                warp_count = warp_count_per_unique_block.viewer().name(
                    "warp_count")] __device__(int i) mutable
               {
                   auto count    = counts(i);
                   auto offset   = offsets(i);
                   warp_count(i) = (count + 31) / 32;
               });

    DeviceScan().ExclusiveSum(warp_count_per_unique_block.data(),
                              warp_offset_per_unique_block.data(),
                              warp_count_per_unique_block.size());

    int h_last_warp_count;
    warp_count_per_unique_block.view(warp_count_per_unique_block.size() - 1).copy_to(&h_last_warp_count);
    int h_last_warp_offset;
    warp_offset_per_unique_block.view(warp_offset_per_unique_block.size() - 1).copy_to(&h_last_warp_offset);
    int total_warps = h_last_warp_offset + h_last_warp_count;

    warp_id_to_unique_block_id.resize(total_warps);
    warp_id_in_unique_block.resize(total_warps);

    // scatter
    ParallelFor(256).kernel_name("scatter").apply(
        unique_counts.size(),
        [warp_counts = warp_count_per_unique_block.viewer().name("warp_count"),
         warp_offsets = warp_offset_per_unique_block.viewer().name("warp_offset"),
         unique_block_id = warp_id_to_unique_block_id.viewer().name("unique_block_id"),
         warp_id_in_unique_block = warp_id_in_unique_block.viewer().name(
             "warp_id_per_unique_block")] __device__(int i) mutable
        {
            auto warp_offset = warp_offsets(i);
            auto warp_count  = warp_counts(i);

            for(int j = 0; j < warp_count; j++)
            {
                // which unique block does this warp process for?
                unique_block_id(warp_offset + j) = i;
                // the index of this warp in the unique block
                warp_id_in_unique_block(warp_offset + j) = j;
            }
        });

    constexpr auto thread_dim = 128;
    auto           block_dim = (total_warps * 32 + thread_dim - 1) / thread_dim;
    constexpr auto warp_dim  = 32;

    blocks.fill(Matrix3x3::Zero());

    Launch(block_dim, thread_dim)
        .kernel_name("balance make unique blocks")
        .apply(
            [origin_blocks = blocks_sorted.viewer().name("origin_blocks"),
             dst_blocks    = blocks.viewer().name("dst_blocks"),
             offsets       = offsets.viewer().name("offset"),
             counts        = unique_counts.viewer().name("count"),
             unique_block_ids = warp_id_to_unique_block_id.viewer().name("unique_block_ids"),
             warp_id_in_block =
                 warp_id_in_unique_block.viewer().name("warp_id_in_block")] __device__() mutable
            {
                auto tid = blockDim.x * blockIdx.x + threadIdx.x;
                auto wid = tid / warp_dim;

                if(wid >= unique_block_ids.dim())
                    return;  // this warp do nothing)

                auto wid_in_this_block       = threadIdx.x / warp_dim;
                auto tid_in_this_warp        = tid % warp_dim;
                auto unique_block_index      = unique_block_ids(wid);
                auto warp_id_in_unique_block = warp_id_in_block(wid);
                auto rest = counts(unique_block_index) - warp_id_in_unique_block * warp_dim;
                auto process_count_this_warp = rest > warp_dim ? warp_dim : rest;

                if(tid_in_this_warp >= process_count_this_warp)
                    return;  // this thread do nothing

                auto offset_into_sorted_block = offsets(unique_block_index)
                                                + warp_id_in_unique_block * warp_dim
                                                + tid_in_this_warp;

                auto origin_block_index = offset_into_sorted_block;

                Matrix3x3 mat_block = origin_blocks(origin_block_index);

                //print("origin_block_index: %d unique_block_index: %d\n",
                //      origin_block_index,
                //      unique_block_index);


                // muda::eigen::atomic_add(dst_blocks(unique_block_index), mat_block);

                // Specialize WarpReduce for type int
                typedef cub::WarpReduce<Matrix3x3> WarpReduce;
                // Allocate WarpReduce shared memory for 4 warps
                __shared__ typename WarpReduce::TempStorage temp_storage[thread_dim / 32];
                Matrix3x3 aggregate =
                    WarpReduce(temp_storage[wid_in_this_block]).Sum(mat_block, process_count_this_warp);

                if(tid_in_this_warp == 0)
                {
                    muda::eigen::atomic_add(dst_blocks(unique_block_index), aggregate);
                }
            });
}

void Converter::_make_unique_block_warp_reduction(const muda::DeviceTripletMatrix<T, N>& from,
                                                  muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    loose_resize(sorted_partition_input, ij_pairs.size());
    loose_resize(sorted_partition_output, ij_pairs.size());


    BufferLaunch().fill<int>(sorted_partition_input, 0);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(unique_counts.size(),
               [sorted_partition = sorted_partition_input.viewer().name("sorted_partition"),
                unique_counts = unique_counts.viewer().name("unique_counts"),
                offsets = offsets.viewer().name("offsets")] __device__(int i) mutable
               {
                   auto offset = offsets(i);
                   auto count  = unique_counts(i);

                   sorted_partition(offset + count - 1) = 1;
               });

    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input.data(),
                              sorted_partition_output.data(),
                              sorted_partition_input.size());

    //std::vector<int> sorted_partition_input_host;
    //std::vector<int> sorted_partition_output_host;

    //sorted_partition_input.copy_to(sorted_partition_input_host);
    //sorted_partition_output.copy_to(sorted_partition_output_host);

    //std::cout << "sorted_partition [unique_count=" << unique_counts.size()
    //          << "]:" << std::endl;
    //for(int i = 0; i < sorted_partition_input.size(); i++)
    //{
    //    std::cout << sorted_partition_input_host[i] << " "
    //              << sorted_partition_output_host[i] << std::endl;
    //}

    auto blocks = to.block_values();


    FastSegmentalReduce()
        .kernel_name(__FUNCTION__)
        .reduce(std::as_const(sorted_partition_output).view(),
                std::as_const(blocks_sorted).view(),
                blocks);


    //temp_blocks.resize(blocks.size());
    //blocks.copy_to(temp_blocks.data());
}

void Converter::_make_unique_blocks_naive(const muda::DeviceTripletMatrix<T, N>& from,
                                          muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto row_indices = to.block_row_indices();
    auto col_indices = to.block_col_indices();
    auto blocks      = to.block_values();

    muda::ParallelFor(256)
        .kernel_name("naive make unique blocks")
        .apply(unique_counts.size(),
               [unique_ij_pairs = unique_ij_pairs.viewer().name("unique_ij_pairs"),
                offsets    = offsets.viewer().name("offset"),
                counts     = unique_counts.viewer().name("count"),
                blocks     = blocks.viewer().name("blocks"),
                blocks_tmp = blocks_sorted.viewer().name("blocks_tmp"),
                sort_index = sort_index.viewer().name("sort_index")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   // calculate the block
                   Matrix3x3 block = Matrix3x3::Zero();

                   for(int j = 0; j < count; j++)
                   {
                       auto origin_block_index = offset + j;
                       block += blocks_tmp(origin_block_index);
                   }

                   blocks(i) = block;
               });

    //temp_blocks2.resize(blocks.size());
    //blocks.copy_to(temp_blocks2.data());

    //for(int i = 0; i < blocks.size(); i++)
    //{
    //    if(!temp_blocks[i].isApprox(temp_blocks2[i], 1e-6))
    //    {
    //        std::cout << "block " << i << " is not equal" << std::endl;
    //        std::cout << "expected: " << temp_blocks[i] << std::endl;
    //        std::cout << "actual: " << temp_blocks2[i] << std::endl;
    //    }
    //}
}


void Converter::convert(const muda::DeviceBCOOMatrix<T, N>& from,
                        muda::DeviceBSRMatrix<T, N>&        to)
{
    // calculate the row offsets
    _calculate_block_offsets(from, to);

    to.resize(from.non_zero_blocks());

    auto vals        = to.block_values();
    auto col_indices = to.block_col_indices();

    vals.copy_from(from.block_values());  // BCOO and BSR have the same block values
    col_indices.copy_from(from.block_col_indices());  // BCOO and BSR have the same block col indices
}

void Converter::_calculate_block_offsets(const muda::DeviceBCOOMatrix<T, N>& from,
                                         muda::DeviceBSRMatrix<T, N>& to)
{
    //Timer timer{__FUNCTION__};

    using namespace muda;
    to.reshape(from.block_rows(), from.block_cols());


    auto dst_row_offsets = to.block_row_offsets();

    col_counts_per_row.resize(dst_row_offsets.size());
    col_counts_per_row.fill(0);

    unique_indices.resize(from.non_zero_blocks());
    unique_counts.resize(from.non_zero_blocks());


    // run length encode the row
    DeviceRunLengthEncode().Encode(from.block_row_indices().data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   from.non_zero_blocks());
    int h_count = count;

    unique_indices.resize(h_count);
    unique_counts.resize(h_count);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(unique_counts.size(),
               [unique_indices     = unique_indices.cviewer().name("offset"),
                counts             = unique_counts.viewer().name("counts"),
                col_counts_per_row = col_counts_per_row.viewer().name(
                    "col_counts_per_row")] __device__(int i) mutable
               {
                   auto row                = unique_indices(i);
                   col_counts_per_row(row) = counts(i);
               });

    // calculate the offsets
    DeviceScan().ExclusiveSum(col_counts_per_row.data(),
                              dst_row_offsets.data(),
                              col_counts_per_row.size());
}

void Converter::ge2sym(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    //print_matrix(to);

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize(counts, to.non_zero_blocks());
    loose_resize(offsets, to.non_zero_blocks());
    loose_resize(ij_pairs, to.non_zero_blocks());
    loose_resize(block_temp, to.block_values().size());

    // 0. find the upper triangular part (where i <= j)
    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(to.non_zero_blocks(),
               [row_indices = to.block_row_indices().cviewer().name("row_indices"),
                col_indices = to.block_col_indices().cviewer().name("col_indices"),
                ij_pairs   = ij_pairs.viewer().name("ij_pairs"),
                blocks     = to.block_values().cviewer().name("block_temp"),
                block_temp = block_temp.viewer().name("block_temp"),
                counts = counts.viewer().name("counts")] __device__(int i) mutable
               {
                   counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
                   block_temp(i) = blocks(i);
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.block_values();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(dst_block.size(),
               [dst_blocks = dst_block.viewer().name("blocks"),
                src_blocks = block_temp.cviewer().name("src_blocks"),
                ij_pairs   = ij_pairs.cviewer().name("ij_pairs"),
                row_indices = to.block_row_indices().viewer().name("row_indices"),
                col_indices = to.block_col_indices().viewer().name("col_indices"),
                counts  = counts.cviewer().name("counts"),
                offsets = offsets.cviewer().name("offsets"),
                total_count = count.viewer().name("total_count")] __device__(int i) mutable
               {
                   auto count  = counts(i);
                   auto offset = offsets(i);

                   if(count != 0)
                   {
                       dst_blocks(offset)  = src_blocks(i);
                       auto ij             = ij_pairs(i);
                       row_indices(offset) = ij.x;
                       col_indices(offset) = ij.y;
                   }

                   if(i == offsets.total_size() - 1)
                   {
                       total_count = offsets(i) + counts(i);
                   }
               });

    int h_total_count = count;

    to.resize_triplets(h_total_count);

    //print_matrix(to);
}


void Converter::sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                       muda::DeviceBCOOMatrix<T, N>&       to)
{
    using namespace muda;

    auto sym_size = from.non_zero_blocks();

    // alias to reuse the memory
    auto& flags                 = offsets;
    auto& partitioned           = blocks_sorted;
    auto& partition_index_input = sort_index_input;
    auto& partition_index       = sort_index;
    auto& selected_count        = count;
    auto  diag_count            = from.block_rows();


    loose_resize(flags, sym_size);
    loose_resize(partitioned, sym_size);
    loose_resize(partition_index, sym_size);

    // setup select flag
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [flags = flags.viewer().name("flags"),
                row_indices = from.block_row_indices().cviewer().name("row_indices"),
                col_indices = from.block_col_indices().cviewer().name("col_indices"),
                partition_index = partition_index_input.viewer().name(
                    "partitioned")] __device__(int i) mutable
               {
                   flags(i) = (row_indices(i) == col_indices(i)) ? 1 : 0;
                   partition_index(i) = i;
               });


    muda::DevicePartition().Flagged(partition_index_input.data(),
                                    flags.data(),
                                    partition_index.data(),
                                    selected_count.data(),
                                    sym_size);


    auto general_bcoo_size = 2 * (sym_size - diag_count) + diag_count;

    to.resize(from.block_rows(), from.block_cols(), general_bcoo_size);

    // copy blocks and ij
    // in this sequence:
    // [ Diag | Upper | Lower ]
    //
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [to   = to.viewer().name("to"),
                from = from.cviewer().name("from"),
                partition_index = partition_index.cviewer().name("partition_index"),
                diag_count = diag_count,
                sym_size   = sym_size] __device__(int i) mutable
               {
                   auto index = partition_index(i);
                   auto f     = from(index);
                   // diag + upper
                   to(i).write(f.block_row_index, f.block_col_index, f.block_value);
                   if(i >= diag_count)
                   {
                       // lower
                       to(i + sym_size - diag_count)
                           .write(f.block_col_index,
                                  f.block_row_index,
                                  f.block_value.transpose());
                   }
               });

    //std::cout << "unsorted" << std::endl;
    //print_matrix(to);

    _radix_sort_indices_and_blocks(to);

    //std::cout << "sym2ge" << std::endl;
    //std::cout << "from" << std::endl;
    //print_matrix(from);
    //std::cout << "to" << std::endl;
    //print_matrix(to);
}
}  // namespace gipc