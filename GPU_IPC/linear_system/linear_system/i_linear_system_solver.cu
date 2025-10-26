#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
IterativeSolver::~IterativeSolver() {}

void IterativeSolver::spmv(Float                         a,
                           muda::CDenseVectorView<Float> x,
                           Float                         b,
                           muda::DenseVectorView<Float>  y)
{
    m_system->spmv(a, x, b, y);
}

void IterativeSolver::apply_preconditioner(muda::DenseVectorView<Float> z,
                                               muda::CDenseVectorView<Float> r) const
{
    m_system->apply_preconditioner(z, r);
}

bool IterativeSolver::accuracy_statisfied(muda::DenseVectorView<Float> r) const
{
    return m_system->accuracy_statisfied(r);
}

muda::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_context;
}
}  // namespace uipc