#pragma once
#include <linear_system/linear_system/linear_subsystem.h>

class GIPC;
namespace gipc
{
class ContactSystem;
class ABDSystem;
class ABDSimData;

class ABDLinearSubsystem : public gipc::DiagonalSubsystem
{
    friend class ABDFEMOffDiagonal;
    // Inherited via LinearSubsystem
  public:
    ABDLinearSubsystem(GIPC&          gipc,
                       ContactSystem& contact_system,
                       ABDSystem&     abd_system,
                       ABDSimData&    abd_sim_data);

  public:
    virtual void report_subsystem_info() override;
    virtual void assemble(TripletMatrixView hessian, DenseVectorView gradient) override;
    virtual void retrieve_solution(CDenseVectorView dx) override;
    virtual bool accuracy_statisfied(CDenseVectorView residual) override;
    void         set_local_tolerance(gipc::Float tol) { m_local_tol = tol; }

  private:
    GIPC&          m_gipc;
    ContactSystem& m_contact_system;
    ABDSystem&     m_abd_system;
    ABDSimData&    m_abd_sim_data;

    muda::DeviceBuffer<Float> m_local_squared_norm;
    muda::DeviceVar<Float>    m_max_squared_norm;
    Float                     m_local_tol = 1e-5;
};
}  // namespace gipc
