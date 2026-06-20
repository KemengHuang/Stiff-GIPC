#pragma once
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
namespace gipc::cuda::parallel
{
class Scatter : public LaunchBase<Scatter>
{
    using Base = LaunchBase<Scatter>;

  public:
    using Base::Base;
    Scatter()
        : Base(nullptr){};

    // to(i) = from(mapper(i))
    template <typename T, typename U>
    void scatter(CBufferView<T> from, BufferView<U> to, CBufferView<int> mapper);
    
    template <typename T>
    void scatter(BufferView<T> to, const T& value);
};
}  // namespace gipc::cuda::parallel

#include "details/scatter.inl"