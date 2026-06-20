#pragma once
#include <gipc/type_define.h>

#include <gipc/cuda/all.h>

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
                              gipc::cuda::CDenseVectorView<Float> x,
                              Float                         b,
                              gipc::cuda::DenseVectorView<Float>  y);
};
}  // namespace gipc
