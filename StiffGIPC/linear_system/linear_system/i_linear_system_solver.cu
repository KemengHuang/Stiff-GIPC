#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
IterativeSolver::~IterativeSolver() {}

void IterativeSolver::spmv(Float                         a,
                           cudatool::CDenseVectorView<Float> x,
                           Float                         b,
                           cudatool::DenseVectorView<Float>  y)
{
    m_system->spmv(a, x, b, y);
}

void IterativeSolver::apply_preconditioner(cudatool::DenseVectorView<Float> z,
                                           cudatool::CDenseVectorView<Float> r) const
{
    m_system->apply_preconditioner(z, r);
}


cudatool::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_context;
}
}  // namespace gipc