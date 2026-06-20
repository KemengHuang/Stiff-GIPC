#pragma once
#include <linear_system/linear_system/linear_subsystem.h>
#include <GIPC.cuh>
#include <device_fem_data.cuh>

namespace gipc
{
class FEMLinearSubsystem : public DiagonalSubsystem
{
    // Inherited via LinearSubsystem
  public:
    FEMLinearSubsystem(GIPC& gipc, device_TetraData& tetra_data);

    cudatool::CBufferView<int> boundary_type() const;

    cudatool::BufferView<double3> barrier_gradient() const;
    cudatool::BufferView<double3> shape_gradient() const;
    cudatool::BufferView<double3> dx() const;
    cudatool::BufferView<double>  mass() const;

  public:
    virtual void report_subsystem_info() override;
    virtual void assemble(DenseVectorView gradient) override;
    virtual void retrieve_solution(CDenseVectorView dx) override;
    void         set_local_tolerance(Float tol);

  private:
    GIPC&             m_gipc;
    device_TetraData& m_tetra_data;

    cudatool::DeviceBuffer<gipc::Float> m_local_squared_norm;
    cudatool::DeviceVar<gipc::Float>    m_max_squared_norm;
    gipc::Float                     m_local_tol = 1e-5;
};
}  // namespace gipc
