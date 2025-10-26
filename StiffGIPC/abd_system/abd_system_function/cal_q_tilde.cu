#include <abd_system/abd_system.h>
#include <muda/cub/device/device_reduce.h>
#include <gipc/utils/timer.h>
namespace gipc
{
void ABDSystem::cal_q_tilde(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd            = sim_data.device;
    auto  abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  kappa          = parms.kappa;
    auto  dt             = parms.dt;
    auto  boundary_type  = sim_data.body_id_to_boundary_type();

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_body_count,
               [boundary_type = boundary_type.cviewer().name("btype"),
                q_prevs  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                q_vs     = abd.body_id_to_q_v.cviewer().name("q_velocities"),
                q_tildes = abd.body_id_to_q_tilde.viewer().name("q_tilde"),
                affine_gravity = abd.body_id_to_abd_gravity.cviewer().name("affine_gravity"),
                dt = dt] __device__(int i) mutable
               {
                   auto& q_prev = q_prevs(i);
                   auto& q_v    = q_vs(i);
                   auto& g      = affine_gravity(i);
                   // TODO: this time, we only consider gravity
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       q_tildes(i) = q_prev;
                   }
                   else
                   {
                       q_tildes(i) = q_prev + q_v * dt + g * (dt * dt);
                   }
               });

    m_local_tolerance.resize(abd.body_id_to_q_tilde.size());

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(abd_body_count,
               [local_tolerance = m_local_tolerance.viewer().name("local_tolerance"),
                q_tildes = abd.body_id_to_q_tilde.cviewer().name("q_tilde"),
                qs = abd.body_id_to_q.cviewer().name("q")] __device__(int i) mutable
               {
                   auto& q_tilde      = q_tildes(i);
                   auto& q            = qs(i);
                   local_tolerance(i) = (q_tilde - q).norm();
               });

    muda::DeviceReduce().Max(m_local_tolerance.data(),
                             m_local_tolerance_max.data(),
                             m_local_tolerance.size());

    m_suggest_max_tolerance = m_local_tolerance_max;
}
}  // namespace gipc