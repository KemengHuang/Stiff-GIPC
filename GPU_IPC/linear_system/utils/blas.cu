#include <linear_system/utils/blas.h>
#include <muda/cub/device/device_reduce.h>
namespace gipc
{
Float Blas::dot(muda::CDenseVectorView<Float> x, muda::CDenseVectorView<Float> y)
{
    dot(x, y, result);
    return result;
}
void Blas::dot(muda::CDenseVectorView<Float> x,
               muda::CDenseVectorView<Float> y,
               muda::VarView<Float>          result)
{
    using namespace muda;
    temp_buffer.resize(x.size() / 3);

    ParallelFor().apply(
        x.size() / 3,
        [temp_buffer = temp_buffer.data(), x = x.data(), y = y.data()] __device__(int i)
        {
            temp_buffer[i] = x[i * 3] * y[i * 3] + x[i * 3 + 1] * y[i * 3 + 1]
                             + x[i * 3 + 2] * y[i * 3 + 2];
        });

    DeviceReduce().Sum(temp_buffer.data(), result.data(), temp_buffer.size());
}
void Blas::axpby(Float alpha, muda::CDenseVectorView<Float> x, Float beta, muda::DenseVectorView<Float> y)
{
    using namespace muda;
    ParallelFor().apply(x.size() / 3,
                        [alpha, beta, x = x.data(), y = y.data()] __device__(int i)
                        {
                            y[3 * i] = alpha * x[3 * i] + beta * y[3 * i];
                            y[3 * i + 1] = alpha * x[3 * i + 1] + beta * y[3 * i + 1];
                            y[3 * i + 2] = alpha * x[3 * i + 2] + beta * y[3 * i + 2];
                        });
}
}  // namespace gipc
