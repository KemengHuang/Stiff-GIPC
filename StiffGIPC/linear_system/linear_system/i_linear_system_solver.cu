#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
IterativeSolver::~IterativeSolver() {}

void IterativeSolver::spmv(Float                         a,
                           gipc::cuda::CDenseVectorView<Float> x,
                           Float                         b,
                           gipc::cuda::DenseVectorView<Float>  y)
{
    m_system->spmv(a, x, b, y);
}

void IterativeSolver::apply_preconditioner(gipc::cuda::DenseVectorView<Float> z,
                                           gipc::cuda::CDenseVectorView<Float> r) const
{
    m_system->apply_preconditioner(z, r);
}


gipc::cuda::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_context;
}
}  // namespace gipc