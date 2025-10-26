#pragma once
#include <gipc/type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/bcoo_matrix_view.h>
#include <muda/ext/linear_system/bsr_matrix_view.h>
#include <muda/ext/linear_system/dense_vector_view.h>
#include <muda/ext/linear_system/device_dense_vector.h>
namespace gipc
{
class Spmv
{
    // if you need any temporary buffer, do something like this
    muda::DeviceDenseVector<Float> fake_y;

  public:
    // calculate y = a * A * x + b * y
    void sym_spmv(Float                          a,
                  muda::CBSRMatrixView<Float, 3> A,
                  muda::CDenseVectorView<Float>  x,
                  Float                          b,
                  muda::DenseVectorView<Float>   y);

    void sym_spmv(Float                           a,
                  muda::CBCOOMatrixView<Float, 3> A,
                  muda::CDenseVectorView<Float>   x,
                  Float                           b,
                  muda::DenseVectorView<Float>    y);

    void warp_reduce_spmv(Float                           a,
                          muda::CBCOOMatrixView<Float, 3> A,
                          muda::CDenseVectorView<Float>   x,
                          Float                           b,
                          muda::DenseVectorView<Float>    y);

    void warp_reduce_sym_spmv(Float                           a,
                              muda::CBCOOMatrixView<Float, 3> A,
                              muda::CDenseVectorView<Float>   x,
                              Float                           b,
                              muda::DenseVectorView<Float>    y);
};
}  // namespace gipc
