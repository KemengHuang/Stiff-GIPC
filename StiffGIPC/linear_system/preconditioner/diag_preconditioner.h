#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class DiagPreconditioner : public GlobalPreconditioner
{
  public:
    DiagPreconditioner() = default;


  public:

    virtual void assemble(GIPCTripletMatrix& global_triplets) override;

    virtual void apply(cudatool::CDenseVectorView<gipc::Float> r,
                       cudatool::DenseVectorView<gipc::Float>  z) override;

  private:
    cudatool::DeviceBuffer<gipc::Matrix3x3> m_diag3x3;
};
}  // namespace gipc
