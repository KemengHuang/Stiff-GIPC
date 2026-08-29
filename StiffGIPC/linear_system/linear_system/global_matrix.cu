#include "linear_system/linear_system/global_matrix.h"
#include "cuda_tools/cuda_tools.h"


__global__ void _set_hash_value(const int* row_ids,
                                const int* col_ids,
                                uint32_t*  index,
                                uint64_t*  hashValue,
                                int        abd_vert_num,
                                int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    index[idx] = idx;
    //hashValue[idx] = (((uint64_t)rows[idx]) << 32) | ((uint64_t)cols[idx]);
    uint64_t self_hash;
    if(row_ids[idx] < abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 3;
    }
    else if(row_ids[idx] < abd_vert_num && col_ids[idx] >= abd_vert_num)
    {
        self_hash = 1;
    }
    else if(row_ids[idx] >= abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 2;
    }
    else
    {
        self_hash = 0;
    }
    hashValue[idx] = self_hash;
}


void GIPCTripletMatrix::update_hash_value(int fem_offset)
{
    //reset_zero();
    resize_conversion_scratch(global_collision_triplet_offset);

    int threadNum = 256;
    int blockNum = (global_collision_triplet_offset + threadNum - 1) / threadNum;

    LaunchCudaKernal(blockNum,
                     threadNum,
                     0,
                     _set_hash_value,
                     (const int*)m_block_row_indices.data(),
                     (const int*)m_block_col_indices.data(),
                     m_block_index.data(),
                     m_block_hash_value.data(),
                     fem_offset,
                     global_collision_triplet_offset);
}
