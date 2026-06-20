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

    virtual void apply(gipc::cuda::CDenseVectorView<gipc::Float> r,
                       gipc::cuda::DenseVectorView<gipc::Float>  z) override;

  private:
    gipc::cuda::DeviceBuffer<gipc::Matrix3x3> m_diag3x3;
};
}  // namespace gipc
