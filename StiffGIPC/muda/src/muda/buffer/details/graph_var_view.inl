namespace muda
{
template <typename T>
MUDA_INLINE void ComputeGraphVar<VarView<T>>::update(const VarView<T>& view)
{
    ComputeGraphVarBase::update();
    m_value = view;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<VarView<T>>& ComputeGraphVar<VarView<T>>::operator=(const VarView<T>& view)
{
    update(view);
    return *this;
}
}  // namespace muda