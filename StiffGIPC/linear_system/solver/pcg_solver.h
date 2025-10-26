#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/utils/blas.h>
#include <thrust/universal_vector.h>
namespace gipc
{
class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;
    using DeviceBCOOMatrix  = muda::DeviceBCOOMatrix<Float, 3>;
    using DeviceBSRMatrix   = muda::DeviceBSRMatrix<Float, 3>;


  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver() = default;

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:
    DeviceBSRMatrix   bsrA;
    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    PCGSolverConfig   m_config;
    Blas              blas;
    muda::DeviceVar<Float> dot_res;
    Float* host_pinned_dot_res;

  protected:
    // Inherited via ILinearSystemSolver
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
    SizeT pcg_0(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
