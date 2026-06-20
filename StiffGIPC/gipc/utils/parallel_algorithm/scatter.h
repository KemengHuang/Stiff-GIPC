#pragma once
#include <cuda_tools/cuda_all.h>

namespace cudatool::parallel
{
class Scatter
{
  public:
    // to(i) = from(mapper(i))
    template <typename T, typename U>
    static void scatter(CBufferView<T> from, BufferView<U> to, CBufferView<int> mapper);

    template <typename T>
    static void scatter(BufferView<T> to, const T& value);
};
}  // namespace cudatool::parallel

#include "details/scatter.inl"
