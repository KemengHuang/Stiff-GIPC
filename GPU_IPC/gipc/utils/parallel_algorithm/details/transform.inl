#include <muda/launch/parallel_for.h>
namespace muda::parallel
{
//using F = double(float);
//using T = float;
//using U = double;


template <typename T, typename U>
void Transform::transform(BufferView<T> to, CBufferView<U> from)
{
    static_assert(std::is_constructible_v<T, U>, "T must be copyassignable to U");
    MUDA_ASSERT(from.size() == to.size(), "transform size mismatch");
    ParallelFor(0, stream())
        .apply(from.size(),
               [from, to] __device__(int i) mutable
               { *to.data(i) = *from.data(i); });
}


template <typename T, typename U, typename F>
void Transform::transform(BufferView<T> to, CBufferView<U> from, F&& f)
{
    MUDA_ASSERT(from.size() == to.size(), "transform size mismatch");
    ParallelFor(0, stream())
        .apply(from.size(),
               [from, to, f = std::move(f)] __device__(int i) mutable
               {
                   static_assert(std::is_invocable_v<F, U>, "f must be: U (T)");
                   *to.data(i) = f(*from.data(i));
               });
}
template <typename T, typename F>
void Transform::transform(BufferView<T> to, F&& f)
{
    ParallelFor(0, stream())
        .apply(to.size(),
               [to, f = std::move(f)] __device__(int i) mutable
               {
                   static_assert(std::is_invocable_v<F, int>, "f must be: T (int)");
                   *to.data(i) = f(i);
               });
}
}  // namespace muda::parallel
