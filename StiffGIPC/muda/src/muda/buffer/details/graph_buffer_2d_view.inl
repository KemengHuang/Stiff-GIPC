namespace muda
{
template <typename T>
MUDA_INLINE void ComputeGraphVar<Buffer2DView<T>>::update(const Buffer2DView<T>& view)
{
    ComputeGraphVarBase::update();
    m_value = view;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<Buffer2DView<T>>& ComputeGraphVar<Buffer2DView<T>>::operator=(const Buffer2DView<T>& view)
{
    update(view);
    return *this;
}
}  // namespace muda