#pragma once
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
namespace gipc::cuda::parallel
{
class Transform : public LaunchBase<Transform>
{
    using Base = LaunchBase<Transform>;

  public:
    using Base::Base;
    Transform()
        : Base(nullptr){};

    // to(i) = from(i)
    template <typename T, typename U>
    void transform(BufferView<T> to, CBufferView<U> from);

    // to(i) = f(from(i))
    // f: U(T)
    template <typename T, typename U, typename F>
    void transform(BufferView<T> to, CBufferView<U> from, F&& f);

    // to(i) = f(i)
    // f: U(int)
    template <typename T, typename F>
    void transform(BufferView<T> to, F&& f);
};
}  // namespace gipc::cuda::parallel

#include "details/transform.inl"