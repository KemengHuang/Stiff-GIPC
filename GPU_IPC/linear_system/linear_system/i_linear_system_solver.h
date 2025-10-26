#pragma once
#include <list>
#include <linear_system/linear_system/linear_subsystem.h>
#include <muda/ext/linear_system.h>

namespace gipc
{
class GlobalLinearSystem;
class IterativeSolver
{
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system;

  public:
    IterativeSolver() = default;
    virtual ~IterativeSolver();

    // delete copy
    IterativeSolver(const IterativeSolver&)            = delete;
    IterativeSolver& operator=(const IterativeSolver&) = delete;

  protected:
    /**
     * \brief Subclass of ILinearSystemSolver must implement this method to solve the linear system Ax = b, directly or iteratively.
     * 
     * \details if the solver is iterative, you can use the following methods to help you implement the solve method:
     * - apply_preconditioner(): Apply the preconditioner to the residual r = b - Ax. The preconditioner is defined by others,
     * you don't need to take care of it.
     * 
     * - accuracy_statisfied(): check if the accuracy is satisfied. If the accuracy is satisfied, you can stop the iteration.
     * The accuracy checking is defined by \ref LinearSubsystem, you don't need to take care of it.
     * 
     * - ctx(): get the context of the basic linear algorithm, to do `dot()/norm()/spmv()/mv() ...` 
     * 
     * \param[out] x: the solution of the linear system
     *  you can change the format to BSR or CSR if you want using `ctx().convert()`
     * \param[in]  b: the right-hand side of the linear system
     * 
     * \return the number of iterations used to solve the linear system, if the solver is iterative.
     * otherwise, return 0.
     * 
     */
    virtual SizeT solve(muda::DenseVectorView<Float>  x,
                        muda::CDenseVectorView<Float> b) = 0;

    void spmv(Float a, muda::CDenseVectorView<Float> x, Float b, muda::DenseVectorView<Float> y);
    void spmv(muda::CDenseVectorView<Float> x, muda::DenseVectorView<Float> y)
    {
        spmv(1.0, x, 0.0, y);
    }

    void apply_preconditioner(muda::DenseVectorView<Float>  z,
                              muda::CDenseVectorView<Float> r) const;

    bool accuracy_statisfied(muda::DenseVectorView<Float> r) const;
    muda::LinearSystemContext& ctx() const;

  private:
    void system(GlobalLinearSystem& system) { m_system = &system; }
};
}  // namespace gipc