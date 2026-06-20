#include <linear_system/subsystem/abd_linear_subsystem.h>
#include <GIPC.cuh>
#include <abd_system/abd_system.h>
#include <abd_system/abd_sim_data.h>

namespace gipc
{
ABDLinearSubsystem::ABDLinearSubsystem(GIPC& gipc, ABDSystem& abd_system, ABDSimData& abd_sim_data)
    : m_gipc(gipc)
    , m_abd_system(abd_system)
    , m_abd_sim_data(abd_sim_data)
{
}

void ABDLinearSubsystem::report_subsystem_info()
{
    right_hand_side_dof(m_abd_system.system_gradient.size());
}

void ABDLinearSubsystem::assemble(DenseVectorView gradient)
{
    if(m_gipc.abd_fem_count_info.abd_body_num < 1)
        return;
    using namespace gipc::cuda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(gradient.size(),
               [dst = gradient.viewer().name("gradient"),
                src = m_abd_system.system_gradient.viewer().name(
                    "system_gradient")] __device__(int i) mutable
               { dst(i) = src(i); });
}

void ABDLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace gipc::cuda;
    auto& sim_data = m_abd_sim_data;

    auto& dq               = sim_data.device.body_id_to_dq;
    auto  abd_body_count   = dq.size();
    auto  abd_point_offset = sim_data.abd_fem_count_info().abd_point_offset;
    auto  abd_point_num    = sim_data.abd_fem_count_info().abd_point_num;

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_body_count,
               [dx        = dx.cviewer().name("dx"),
                dq        = dq.viewer().name("dq"),
                local_tol = m_local_tol] __device__(int i) mutable
               { dq(i) = dx.segment<12>(i * 12).as_eigen(); });

    m_abd_system.cal_dx_from_dq(
        sim_data,
        gipc::cuda::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(
            abd_point_offset, abd_point_num));
}

}  // namespace gipc
