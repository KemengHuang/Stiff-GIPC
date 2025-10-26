#pragma once
#include <contact_system/i_contact_reporter.h>
#include <gpu_eigen_libs.cuh>

class GIPC;
class device_TetraData;

namespace gipc
{
class ContactInfoReporter : public gipc::IContactReporter
{
  public:
    ContactInfoReporter(GIPC& gipc, device_TetraData& tet);

    // Inherited via IContactReporter
    void report_contact_count(ContactCountInfo& info) override;
    void assemble(muda::BufferView<gipc::ContactGradient> gradients,
                  muda::BufferView<gipc::ContactHessian>  hessians) override;

  private:
    muda::BufferView<__GEIGEN__::Matrix3x3d>   H3x3() const;
    muda::BufferView<uint32_t>                 H3x3_index() const;
    muda::BufferView<__GEIGEN__::Matrix6x6d>   H6x6() const;
    muda::BufferView<uint2>                    H6x6_index() const;
    muda::BufferView<__GEIGEN__::Matrix9x9d>   H9x9() const;
    muda::BufferView<uint3>                    H9x9_index() const;
    muda::BufferView<__GEIGEN__::Matrix12x12d> H12x12() const;
    muda::BufferView<uint4>                    H12x12_index() const;
    muda::BufferView<double3>                  barrier_gradient() const;

    GIPC&             m_gipc;
    device_TetraData& m_tetra_data;
};
}  // namespace OLD_GIPC
