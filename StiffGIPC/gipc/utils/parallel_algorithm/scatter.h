#pragma once
#include <muda/buffer/buffer_view.h>
#include <muda/launch/launch_base.h>
namespace muda::parallel
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
}  // namespace muda::parallel

#include "details/scatter.inl"