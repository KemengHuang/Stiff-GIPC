#include <linear_system/subsystem/fem_linear_subsystem.h>

namespace gipc
{
FEMLinearSubsystem::FEMLinearSubsystem(GIPC& gipc, device_TetraData& tetra_data)
    : m_gipc(gipc)
    , m_tetra_data(tetra_data)
{
    //gipc::cuda::Debug::debug_sync_all(true);
}


gipc::cuda::CBufferView<int> FEMLinearSubsystem::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return gipc::cuda::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

gipc::cuda::BufferView<double3> FEMLinearSubsystem::barrier_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return gipc::cuda::BufferView<double3>{m_tetra_data.fb, m_gipc.vertexNum}.subview(offset, fem_count);
}

gipc::cuda::BufferView<double3> FEMLinearSubsystem::shape_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return gipc::cuda::BufferView<double3>{m_tetra_data.shape_grads, m_gipc.vertexNum}.subview(
        offset, fem_count);
}

gipc::cuda::BufferView<double3> FEMLinearSubsystem::dx() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return gipc::cuda::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(offset, fem_count);
}

gipc::cuda::BufferView<double> FEMLinearSubsystem::mass() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return gipc::cuda::BufferView<double>{m_tetra_data.masses, m_gipc.vertexNum}.subview(
        fem_offset, fem_count);
}

void FEMLinearSubsystem::report_subsystem_info()
{
    this->right_hand_side_dof(dx().size() * 3);
}


void FEMLinearSubsystem::assemble(DenseVectorView gradient)
{
    using namespace gipc::cuda;

    if(m_gipc.abd_fem_count_info.fem_point_num < 1)
        return;

    auto barrier_gradient = this->barrier_gradient();
    auto shape_gradient   = this->shape_gradient();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(barrier_gradient.size(),
               [b     = barrier_gradient.viewer().name("barrier_gradient"),
                s     = shape_gradient.viewer().name("shape_gradient"),
                btype = boundary_type().cviewer().name("boundary_type"),
                gradient = gradient.viewer().name("gradient")] __device__(int i) mutable
               {
                   if(btype(i) != 0)
                   {
                       gradient.segment<3>(i * 3).as_eigen() = Vector3::Zero();
                   }
                   else
                   {
                       gradient.segment<3>(i * 3).as_eigen() =
                           eigen::as_eigen(b(i)) + eigen::as_eigen(s(i));
                   }
               });
}

void FEMLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace gipc::cuda;

    auto move_dir = this->dx();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(move_dir.size(),
               [dx        = dx.viewer().name("dx"),
                move_dir  = move_dir.viewer().name("move_dir"),
                local_tol = m_local_tol] __device__(int i) mutable
               {
                   auto md = eigen::as_eigen(move_dir(i));
                   md      = dx.segment<3>(i * 3).as_eigen();
               });
}

void FEMLinearSubsystem::set_local_tolerance(gipc::Float tol)
{
    m_local_tol = tol;
}
}  // namespace gipc