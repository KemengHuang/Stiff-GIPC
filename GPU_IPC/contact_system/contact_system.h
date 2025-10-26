#pragma once
#include <memory>
#include <contact_system/i_contact_reporter.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>

class GIPC;
namespace gipc
{
class ContactSystem
{
    template <typename T>
    using U = std::unique_ptr<T>;
    GIPC& m_gipc;

  public:
    ContactSystem(GIPC& gipc);
    template <typename T, typename... Args>
    T& create_contact_reporter(Args&&... args)
    {
        static_assert(std::is_base_of_v<IContactReporter, T>,
                      "T must be derived from IContactReporter");
        static_assert(std::is_constructible_v<T, Args...>, "T must be constructible with Args");
        m_contact_reporter = std::make_unique<T>(std::forward<Args>(args)...);
        return *static_cast<T*>(m_contact_reporter.get());
    }

    muda::CBufferView<ContactGradient> fem_contact_gradients()
    {
        return m_fem_contact_gradients;
    }
    muda::CBufferView<ContactHessian> fem_contact_hessians()
    {
        return m_fem_contact_hessians;
    }
    muda::CBufferView<ContactGradient> abd_contact_gradients()
    {
        return m_abd_contact_gradients;
    }
    muda::CBufferView<ContactHessian> abd_contact_hessians()
    {
        return m_abd_contact_hessians;
    }
    muda::CBufferView<ContactHessian> abd_fem_contact_hessians()
    {
        return m_abd_fem_contact_hessians;
    }
    muda::CBufferView<ContactHessian> fem_abd_contact_hessians()
    {
        return m_fem_abd_contact_hessians;
    }

    void solve();
    void _assemble();
    void _partition();

    void report_info(bool on) { m_report_info = on; }

  private:
    muda::DeviceBuffer<ContactGradient> m_contact_gradients;
    muda::DeviceBuffer<ContactHessian>  m_contact_hessians;
    muda::DeviceBuffer<ContactHessian>  m_partitioned_contact_hessians;
    muda::DeviceVar<int>                m_partition_count;
    U<IContactReporter>                 m_contact_reporter;

    muda::CBufferView<ContactGradient> m_fem_contact_gradients;
    muda::CBufferView<ContactHessian>  m_fem_contact_hessians;
    muda::CBufferView<ContactGradient> m_abd_contact_gradients;
    muda::CBufferView<ContactHessian>  m_abd_contact_hessians;
    muda::CBufferView<ContactHessian>  m_abd_fem_contact_hessians;
    muda::CBufferView<ContactHessian>  m_fem_abd_contact_hessians;
    bool                               m_report_info = true;
};
}  // namespace gipc
