#pragma once
#include <gipc/type_define.h>

#include <cuda_tools/cuda_all.h>

namespace gipc
{
class Spmv
{
  public:

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y);
};
}  // namespace gipc
