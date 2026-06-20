#pragma once
#include <linear_system/linear_system/linear_subsystem.h>

class GIPC;
namespace gipc
{
class ABDSystem;
class ABDSimData;

class ABDLinearSubsystem : public gipc::DiagonalSubsystem
{
  public:
    ABDLinearSubsystem(GIPC&          gipc,
                       ABDSystem&     abd_system,
                       ABDSimData&    abd_sim_data);

  public:
    virtual void report_subsystem_info() override;
    virtual void assemble(DenseVectorView gradient) override;
    virtual void retrieve_solution(CDenseVectorView dx) override;
    void         set_local_tolerance(gipc::Float tol) { m_local_tol = tol; }

  private:
    GIPC&          m_gipc;
    ABDSystem&     m_abd_system;
    ABDSimData&    m_abd_sim_data;

    gipc::cuda::DeviceBuffer<Float> m_local_squared_norm;
    gipc::cuda::DeviceVar<Float>    m_max_squared_norm;
    Float                     m_local_tol = 1e-5;
};
}  // namespace gipc
