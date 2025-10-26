#pragma once
#include <linear_system/linear_system/i_preconditioner.h>
class BHessian;
class MASPreconditioner;

namespace gipc
{
class FEMLinearSubsystem;

class MAS_Preconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    BHessian&   BH;
    MASPreconditioner& MAS_Prec;
    double*            masses;
    uint32_t*          cpNum;

  public:
    MAS_Preconditioner(FEMLinearSubsystem& subsystem, BHessian& mBH, MASPreconditioner& mMAS, double* mMasses, uint32_t* mCpNum);
    virtual void assemble() override;
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;
};
}  // namespace gipc
