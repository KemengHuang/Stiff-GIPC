#pragma once
#include <gipc/type_define.h>
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc
{
class Converter
{
  public:
    // Triplet -> BCOO
    void convert(GIPCTripletMatrix& global_triplets,
                 const int&         start,
                 const int&         length,
                 const int&         out_start_id);


    void _radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                        const int&         start,
                                        const int&         length,
                                        const int&         out_start_id);


    void _make_unique_indices(GIPCTripletMatrix& global_triplets,
                              const int&         start,
                              const int&         length,
                              const int&         out_start_id);


    void _make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                           const int&         start,
                                           const int&         length,
                                           const int&         out_start_id);


    void ge2sym(GIPCTripletMatrix& global_triplets);
};
}  // namespace gipc
