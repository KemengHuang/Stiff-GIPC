#include <muda/launch/parallel_for.h>
namespace muda::parallel
{
//using T = float;
//using U = double;


template <typename T, typename U>
void Scatter::scatter(CBufferView<T> from, BufferView<U> to, CBufferView<int> mapper)
{
    static_assert(std::is_convertible_v<T, U>, "T must be is_convertible_v to U");
    MUDA_ASSERT(to.size() == mapper.size(),
                "to.size() != mapper.size()");
    MUDA_ASSERT(to.size() >= from.size(),
                "to.size() < from.size()");
    ParallelFor(0, stream())
        .file_line(__FILE__, __LINE__)
        .apply(from.size(),
               [from = from.viewer().name("from"),
                to   = to.viewer().name("to"),
                mapper = mapper.viewer().name("mapper")] __device__(int i) mutable
               { to(i) = static_cast<U>(from(mapper(i))); });
}
template <typename T>
void Scatter::scatter(BufferView<T> to, const T& value)
{
    ParallelFor(0, stream())
        .file_line(__FILE__, __LINE__)
        .apply(to.size(),
               [to = to.viewer().name("to"), value] __device__(int i) mutable
               { to(i) = value; });
}
}  // namespace muda::parallel