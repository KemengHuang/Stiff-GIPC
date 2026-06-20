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
    MASPreconditioner& MAS_Prec;
    double*            masses;
    uint32_t*          cpNum;

  public:
    MAS_Preconditioner(FEMLinearSubsystem& subsystem, MASPreconditioner& mMAS, double* mMasses, uint32_t* mCpNum);
    virtual void assemble() override;
    virtual void apply(gipc::cuda::CDenseVectorView<Float> r, gipc::cuda::DenseVectorView<Float> z) override;
    //const int preconditioner_id = 1;
};
}  // namespace gipc
