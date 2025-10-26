#pragma once
#include <contact_system/contact_info.h>
#include <muda/buffer/buffer_view.h>
namespace gipc
{
class IContactReporter
{
  public:
    class ContactCountInfo
    {
      public:
        size_t gradient_count;
        size_t hessian_count;
    };

    virtual void report_contact_count(ContactCountInfo& info)  = 0;
    virtual void assemble(muda::BufferView<ContactGradient> gradients,
                          muda::BufferView<ContactHessian>  hessians) = 0;
};
}  // namespace gipc
