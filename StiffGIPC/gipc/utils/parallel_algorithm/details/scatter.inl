#include <cuda_tools/cuda_all.h>

namespace
{
template <typename T, typename U>
__global__ void scatter_mapped_kernel(int                size,
                                      cudatool::CBufferView<T>   from,
                                      cudatool::BufferView<U>    to,
                                      cudatool::CBufferView<int> mapper)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    to[i] = static_cast<U>(from[mapper[i]]);
}

template <typename T>
__global__ void scatter_value_kernel(int size, cudatool::BufferView<T> to, T value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    to[i] = value;
}
}  // namespace

namespace cudatool::parallel
{
template <typename T, typename U>
void Scatter::scatter(CBufferView<T> from, BufferView<U> to, CBufferView<int> mapper)
{
    static_assert(std::is_convertible_v<T, U>, "T must be is_convertible_v to U");
    CT_ASSERT(to.size() == mapper.size(),
                "to.size() != mapper.size()");
    CT_ASSERT(to.size() >= from.size(),
                "to.size() < from.size()");
    LaunchCudaKernal_default(from.size(),
                             256,
                             0,
                             scatter_mapped_kernel<T, U>,
                             from.size(),
                             from,
                             to,
                             mapper);
}

template <typename T>
void Scatter::scatter(BufferView<T> to, const T& value)
{
    LaunchCudaKernal_default(to.size(),
                             256,
                             0,
                             scatter_value_kernel<T>,
                             to.size(),
                             to,
                             value);
}
}  // namespace cudatool::parallel
