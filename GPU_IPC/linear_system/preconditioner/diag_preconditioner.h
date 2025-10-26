#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class DiagPreconditioner : public GlobalPreconditioner
{
  public:
    DiagPreconditioner() = default;


  public:
    // Inherited via GlobalPreconditioner
    virtual void assemble(muda::CBCOOMatrixView<gipc::Float, 3> hessian) override;
    virtual void apply(muda::CDenseVectorView<gipc::Float> r,
                       muda::DenseVectorView<gipc::Float>  z) override;

  private:
    muda::DeviceBuffer<gipc::Matrix3x3> m_diag3x3;
};
}  // namespace gipc
