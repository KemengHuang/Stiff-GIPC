#include <linear_system/subsystem/abd_linear_subsystem.h>
#include <GIPC.cuh>
#include <abd_system/abd_system.h>
#include <abd_system/abd_sim_data.h>
#include <muda/cub/device/device_reduce.h>

namespace gipc
{
ABDLinearSubsystem::ABDLinearSubsystem(GIPC&          gipc,
                                       ContactSystem& contact_system,
                                       ABDSystem&     abd_system,
                                       ABDSimData&    abd_sim_data)
    : m_gipc(gipc)
    , m_contact_system(contact_system)
    , m_abd_system(abd_system)
    , m_abd_sim_data(abd_sim_data)
{
}

void ABDLinearSubsystem::report_subsystem_info()
{
    // H12x12 -> 16 * H3x3
    auto h3x3_count = 16 * m_abd_system.bcoo_hessian.non_zero_blocks();

    hessian_block_count(h3x3_count);
    //std::cout << "ABDLinearSubsystem::report_subsystem_info: h3x3_count = " << h3x3_count
    //          << " gradient_dof_count = " << m_abd_system.system_gradient.size()
    //          << std::endl;
    right_hand_side_dof(m_abd_system.system_gradient.size());
}

void ABDLinearSubsystem::assemble(TripletMatrixView hessian, DenseVectorView gradient)
{
    if(m_gipc.abd_fem_count_info.abd_body_num < 1)
        return;
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(gradient.size(),
               [dst = gradient.viewer().name("gradient"),
                src = m_abd_system.system_gradient.viewer().name(
                    "system_gradient")] __device__(int i) mutable
               { dst(i) = src(i); });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(m_abd_system.bcoo_hessian.non_zero_blocks(),
               [dst = hessian.viewer().name("hessian"),
                src = m_abd_system.bcoo_hessian.cviewer().name("bcoo_hessian")] __device__(int I) mutable
               {
                   auto offset = I * 16;
                   for(int i = 0; i < 4; ++i)
                       for(int j = 0; j < 4; ++j)
                       {
                           auto&& [row, col, H12x12] = src(I);
                           dst(offset++).write(row * 4 + i,
                                               col * 4 + j,
                                               H12x12.block<3, 3>(3 * i, 3 * j));
                       }
               });
}

void ABDLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace muda;
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
               {
                   dq(i) = dx.segment<12>(i * 12).as_eigen();

                   //printf("solve dq: %f %f %f %f %f %f %f %f %f %f %f %f\n",
                   //       dq(i)(0),
                   //       dq(i)(1),
                   //       dq(i)(2),
                   //       dq(i)(3),
                   //       dq(i)(4),
                   //       dq(i)(5),
                   //       dq(i)(6),
                   //       dq(i)(7),
                   //       dq(i)(8),
                   //       dq(i)(9),
                   //       dq(i)(10),
                   //       dq(i)(11));
               });

    m_abd_system.cal_dx_from_dq(
        sim_data,
        muda::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(
            abd_point_offset, abd_point_num));
}

bool ABDLinearSubsystem::accuracy_statisfied(CDenseVectorView residual)
{
    using namespace muda;

    //auto abd_body_count = m_abd_sim_data.abd_fem_count_info().abd_body_num;
    //m_local_squared_norm.resize(abd_body_count);

    //ParallelFor()
    //    .kernel_name(__FUNCTION__)
    //    .apply(abd_body_count,
    //           [residual     = residual.viewer().name("residual"),
    //            squared_norm = m_local_squared_norm.viewer().name(
    //                "squared_norm")] __device__(int i) mutable {
    //               squared_norm(i) =
    //                   residual.segment<12>(i * 12).as_eigen().squaredNorm();
    //           });

    //DeviceReduce().Max(m_local_squared_norm.data(),
    //                   m_max_squared_norm.data(),
    //                   m_local_squared_norm.size());

    //gipc::Float max_norm = m_max_squared_norm;
    //return max_norm < m_local_tol * m_local_tol;
    return true;
}
}  // namespace gipc
