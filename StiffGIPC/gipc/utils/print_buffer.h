#pragma once
#include <muda/buffer.h>
#include <iostream>
namespace gipc
{
template <typename T>
std::ostream& operator<<(std::ostream& o, muda::CBufferView<T> const& b)
{
    std::vector<T> v(b.size());
    b.copy_to(v.data());
    for(int i = 0; i < v.size(); ++i)
    {
        o << "[" << i << "]" << v[i] << "\n";
    }
    return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, muda::BufferView<T> const& b)
{
    return operator<<(o, muda::CBufferView<T>(b));
}

template <typename T>
std::ostream& operator<<(std::ostream& o, muda::DeviceBuffer<T> const& b)
{
    return operator<<(o, b.view());
}
}  // namespace gipc