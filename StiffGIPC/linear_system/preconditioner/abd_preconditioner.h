#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class ABDSystem;
class ABDSimData;
class ABDLinearSubsystem;
class ABDPreconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    ABDSystem&  m_abd;
    ABDSimData& m_sim_data;

  public:
    ABDPreconditioner(ABDLinearSubsystem& subsystem, ABDSystem& abd, ABDSimData& sim_data);
    virtual void assemble() override;
    virtual void apply(cudatool::CDenseVectorView<Float> r, cudatool::DenseVectorView<Float> z) override;
    //const int preconditioner_id = 0;
};
}  // namespace gipc
