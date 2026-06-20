#include <linear_system/utils/converter.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>

namespace gipc
{

template <typename T>
__global__ inline void moveMemory_2(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&                          start,
                        const int&                          length,
                        const int&                          out_start_id)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    _radix_sort_indices_and_blocks(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    //_make_unique_indices(global_triplets, start, length, out_start_id);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    _make_unique_block_warp_reduction(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



void Converter::_radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                               const int& start,
                                               const int& length,
                                               const int& out_start_id)
{
    using namespace gipc::cuda;

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input   = global_triplets.block_index();
    auto ij_hash_input = global_triplets.block_hash_value();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(length,
               [row_indices = src_row_indices,
                col_indices = src_col_indices,
                ij_hash_input,
                index_input] __device__(int i) mutable
               {
                   ij_hash_input[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   index_input[i] = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                length);

    auto dst_val = global_triplets.block_values() + out_start_id;
    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(length,
               [sort_index = global_triplets.block_sort_index(),
                src_blocks,
                dst_val] __device__(int i) mutable
               {
                   dst_val[i] = src_blocks[sort_index[i]];

               });
}


void Converter::_make_unique_indices(GIPCTripletMatrix& global_triplets,
                                     const int&         start,
                                     const int&         length,
                                     const int&         out_start_id)
{
    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);

    auto unique_key = global_triplets.block_hash_value();
    auto sort_key   = global_triplets.block_sort_hash_value();

    gipc::cuda::DeviceRunLengthEncode().Encode(sort_key,
                                         unique_key,
                                         global_triplets.block_temp_buffer(),
                                         global_triplets.d_unique_key_number,
                                         length);

    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    gipc::cuda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(global_triplets.h_unique_key_number,

               [row_indices, col_indices, unique_key] __device__(int i) mutable
               {
                   row_indices[i] = unique_key[i] >> 32;
                   col_indices[i] = unique_key[i] & 0xffffffff;
               });
}





void Converter::_make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                                  const int& start, const int& length, const int& out_start_id)
{
    using namespace gipc::cuda;

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(length - 1,
               [sorted_partition_input,
                ij_hash = global_triplets.block_sort_hash_value()] __device__(int i) mutable
               {
                   sorted_partition_input[i] = ij_hash[i] != ij_hash[i + 1] ? 1 : 0;
               });
    auto sorted_partition_output = global_triplets.block_index();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input, sorted_partition_output, length);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);


    gipc::cuda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(length,
               [row_indices,
                col_indices,
                ij_hash = global_triplets.block_sort_hash_value(),
                sorted_partition_output] __device__(int i) mutable
               {
                   int index = sorted_partition_output[i];
                   if(i == 0)
                   {

                       auto key           = ij_hash[i];
                       row_indices[index] = key >> 32;
                       col_indices[index] = key & 0xffffffff;
                   }
                   else
                   {
                       if(index != sorted_partition_output[i - 1])
                       {
                           auto key           = ij_hash[i];
                           row_indices[index] = key >> 32;
                           col_indices[index] = key & 0xffffffff;
                       }
                   }
               });


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              sorted_partition_output + length - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    global_triplets.h_unique_key_number += 1;

    CUDA_SAFE_CALL(cudaMemset(global_triplets.block_values(start),
                              0,
                              global_triplets.h_unique_key_number * sizeof(Eigen::Matrix3d)));

    FastSegmentalReduce()
        .kernel_name(__FUNCTION__)
        .reduce(length,
                sorted_partition_output,
                global_triplets.block_values(out_start_id),
                global_triplets.block_values(start));
}

void Converter::ge2sym(GIPCTripletMatrix& global_triplets)
{
    using namespace gipc::cuda;

    auto counts  = global_triplets.block_index();
    auto offsets = global_triplets.block_sort_index();
    auto block_temp = global_triplets.block_values(global_triplets.h_unique_key_number);
    auto blocks      = global_triplets.block_values();
    auto ij_hash     = global_triplets.block_hash_value();
    auto row_indices = global_triplets.block_row_indices();
    auto col_indices = global_triplets.block_col_indices();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [row_indices, col_indices, ij_hash, blocks, block_temp, counts] __device__(int i) mutable
               {
                   counts[i] = row_indices[i] <= col_indices[i] ? 1 : 0;
                   ij_hash[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   block_temp[i] = blocks[i];
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts, offsets, global_triplets.h_unique_key_number);

    // set the values
    auto dst_blocks = global_triplets.block_values();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [dst_blocks,
                block_temp,
                ij_hash,
                row_indices,
                col_indices,
                counts,
                offsets,
                total_count = global_triplets.d_unique_key_number,
                number = global_triplets.h_unique_key_number] __device__(int i) mutable
               {
                   auto count  = counts[i];
                   auto offset = offsets[i];

                   if(count != 0)
                   {
                       dst_blocks[offset]  = block_temp[i];
                       auto ij             = ij_hash[i];
                       row_indices[offset] = ij >> 32;
                       col_indices[offset] = ij & 0xffffffff;
                   }

                   if(i == number - 1)
                   {
                       *total_count = offsets[i] + counts[i];
                   }
               });


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
}

}  // namespace gipc