#pragma once
#include <cuda_tools/cuda_all.h>

namespace cudatool::parallel
{
class Transform
{
  public:
    // to(i) = from(i)
    template <typename T, typename U>
    static void transform(BufferView<T> to, CBufferView<U> from);

    // to(i) = f(from(i))
    // f: U(T)
    template <typename T, typename U, typename F>
    static void transform(BufferView<T> to, CBufferView<U> from, F&& f);

    // to(i) = f(i)
    // f: U(int)
    template <typename T, typename F>
    static void transform(BufferView<T> to, F&& f);
};
}  // namespace cudatool::parallel

#include "details/transform.inl"
