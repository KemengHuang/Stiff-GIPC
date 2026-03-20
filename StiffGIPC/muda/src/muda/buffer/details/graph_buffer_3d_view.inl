namespace muda
{
template <typename T>
MUDA_INLINE void ComputeGraphVar<Buffer3DView<T>>::update(const Buffer3DView<T>& view)
{
    ComputeGraphVarBase::update();
    m_value = view;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<Buffer3DView<T>>& ComputeGraphVar<Buffer3DView<T>>::operator=(const Buffer3DView<T>& view)
{
    update(view);
    return *this;
}
}  // namespace muda