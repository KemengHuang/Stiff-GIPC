#include <linear_system/linear_system/i_preconditioner.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/global_linear_system.h>
#include <muda/cub/device/device_select.h>

namespace gipc
{
IPreconditioner::~IPreconditioner() {}

Json IPreconditioner::as_json() const
{
    Json j;
    j["type"] = typeid(*this).name();
    return j;
}

muda::LinearSystemContext& IPreconditioner::ctx() const
{
    return m_system->m_context;
}

LocalPreconditioner::LocalPreconditioner(DiagonalSubsystem& subsystem)
    : m_subsystem(&subsystem)
{
}

int LocalPreconditioner::get_offset() const
{
    return m_subsystem->m_dof_offset / 3;
}

muda::CBufferView<int> LocalPreconditioner::calculate_subsystem_bcoo_indices() const
{
    auto  offset = m_subsystem->m_dof_offset / 3;
    auto  end    = offset + m_subsystem->m_right_hand_side_dof / 3;
    auto& bcoo   = m_system->m_bcoo_A;

    loose_resize(m_indices_input, bcoo.triplet_count());
    loose_resize(m_indices_output, bcoo.triplet_count());
    loose_resize(m_flags, bcoo.triplet_count());

    muda::ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(bcoo.triplet_count(),
               [bcoo          = bcoo.cviewer().name("bcoo"),
                offset        = offset,
                end           = end,
                indices_input = m_indices_input.viewer().name("indices_input"),
                flags = m_flags.viewer().name("flags")] __device__(int I) mutable
               {
                   auto&& [i, j, H] = bcoo(I);
                   auto in_range = [&](int m) { return m >= offset && m < end; };
                   bool valid       = in_range(i) && in_range(j);
                   indices_input(I) = valid ? I : -I;  // -I for invalid
                   flags(I)         = valid ? 1 : 0;
               });

    muda::DeviceSelect().Flagged(m_indices_input.data(),
                                 m_flags.data(),
                                 m_indices_output.data(),
                                 m_count.data(),
                                 m_indices_input.size());

    int h_count = m_count;

    m_indices_output.resize(h_count);
    return m_indices_output;
}

muda::CBCOOMatrixView<Float, 3> LocalPreconditioner::system_bcoo_matrix() const
{
    return m_system->m_bcoo_A;
}

void LocalPreconditioner::do_apply(muda::CDenseVectorView<Float> r,
                                   muda::DenseVectorView<Float>  z)
{
    auto dof_offset = m_subsystem->dof_offset()[0];
    auto dof_count  = m_subsystem->right_hand_side_dof();

    apply(r.subview(dof_offset, dof_count), z.subview(dof_offset, dof_count));
}

void LocalPreconditioner::do_assemble(muda::CBCOOMatrixView<Float, 3> hessian)
{
    assemble();
}

void GlobalPreconditioner::do_apply(muda::CDenseVectorView<Float> r,
                                    muda::DenseVectorView<Float>  z)
{
    apply(r, z);
}

void GlobalPreconditioner::do_assemble(muda::CBCOOMatrixView<Float, 3> hessian)
{
    assemble(hessian);
}
}  // namespace gipc