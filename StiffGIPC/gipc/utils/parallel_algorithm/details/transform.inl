#include <cuda_tools/cuda_all.h>

namespace
{
template <typename T, typename U>
__global__ void transform_copy_kernel(int size,
                                      cudatool::BufferView<T>  to,
                                      cudatool::CBufferView<U> from)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    to.data()[i] = from.data()[i];
}

template <typename T, typename U, typename F>
__global__ void transform_from_kernel(int size,
                                      cudatool::BufferView<T>  to,
                                      cudatool::CBufferView<U> from,
                                      F                        f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    to.data()[i] = f(from.data()[i]);
}

template <typename T, typename F>
__global__ void transform_index_kernel(int size, cudatool::BufferView<T> to, F f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    to.data()[i] = f(i);
}
}  // namespace

namespace cudatool::parallel
{
template <typename T, typename U>
void Transform::transform(BufferView<T> to, CBufferView<U> from)
{
    static_assert(std::is_constructible_v<T, U>, "T must be copyassignable to U");
    CT_ASSERT(from.size() == to.size(), "transform size mismatch");
    LaunchCudaKernal_default(from.size(),
                             256,
                             0,
                             transform_copy_kernel<T, U>,
                             from.size(),
                             to,
                             from);
}


template <typename T, typename U, typename F>
void Transform::transform(BufferView<T> to, CBufferView<U> from, F&& f)
{
    CT_ASSERT(from.size() == to.size(), "transform size mismatch");
    LaunchCudaKernal_default(from.size(),
                             256,
                             0,
                             transform_from_kernel<T, U, F>,
                             from.size(),
                             to,
                             from,
                             f);
}

template <typename T, typename F>
void Transform::transform(BufferView<T> to, F&& f)
{
    LaunchCudaKernal_default(to.size(),
                             256,
                             0,
                             transform_index_kernel<T, F>,
                             to.size(),
                             to,
                             f);
}
}  // namespace cudatool::parallel
