#pragma once
#include <gipc/type_define.h>
#include <muda/ext/linear_system/dense_vector_view.h>
#include <muda/tools/temp_buffer.h>
#include <muda/buffer/device_var.h>
namespace gipc
{
class Blas
{
    muda::details::TempBuffer<Float> temp_buffer;
    muda::DeviceVar<Float>           result;

  public:
    Float dot(muda::CDenseVectorView<Float> x, muda::CDenseVectorView<Float> y);
    void  dot(muda::CDenseVectorView<Float> x,
              muda::CDenseVectorView<Float> y,
              muda::VarView<Float>          result);
    void  axpby(Float                         alpha,
                muda::CDenseVectorView<Float> x,
                Float                         beta,
                muda::DenseVectorView<Float>  y);
};

}  // namespace gipc
