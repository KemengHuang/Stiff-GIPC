#pragma once
#include <linear_system/linear_system/linear_subsystem.h>
#include <GIPC.cuh>
#include <device_fem_data.cuh>

namespace gipc
{
class ContactSystem;
}

namespace gipc
{
class FEMLinearSubsystem : public DiagonalSubsystem
{
    // Inherited via LinearSubsystem
  public:
    FEMLinearSubsystem(GIPC& gipc, device_TetraData& tetra_data, ContactSystem& contact_system);

    muda::BufferView<__GEIGEN__::Matrix12x12d> H12x12() const;
    muda::BufferView<uint4>                    H12x12_index() const;
    muda::BufferView<__GEIGEN__::Matrix9x9d>   H9x9() const;
    muda::BufferView<uint3>                    H9x9_index() const;
    muda::CBufferView<int>                     boundary_type() const;

    muda::BufferView<double3> barrier_gradient() const;
    muda::BufferView<double3> shape_gradient() const;
    muda::BufferView<double3> dx() const;
    muda::BufferView<double>  mass() const;

  public:
    virtual void report_subsystem_info() override;
    virtual void assemble(TripletMatrixView hessian, DenseVectorView gradient) override;
    virtual void retrieve_solution(CDenseVectorView dx) override;
    virtual bool accuracy_statisfied(CDenseVectorView residual) override;
    void         set_local_tolerance(Float tol);

  private:
    GIPC&                m_gipc;
    device_TetraData&    m_tetra_data;
    gipc::ContactSystem& m_contact_system;

    muda::DeviceBuffer<gipc::Float> m_local_squared_norm;
    muda::DeviceVar<gipc::Float>    m_max_squared_norm;
    gipc::Float                     m_local_tol = 1e-5;
};
}  // namespace gipc
