#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
void ABDSystem::update_velocity(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd            = sim_data.device;
    auto& abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  boundary_type  = sim_data.body_id_to_boundary_type();
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(abd.body_id_to_q.size(),
               [boundary_type = boundary_type.cviewer().name("btype"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                q_vs     = abd.body_id_to_q_v.viewer().name("q_vs"),
                q_prevs  = abd.body_id_to_q_prev.viewer().name("q_prevs"),
                dt       = parms.dt] __device__(int i) mutable
               {
                   auto& q_v    = q_vs(i);
                   auto& q_prev = q_prevs(i);

                   const auto& q = qs(i);

                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       q_v = Vector12::Zero();
                   }
                   else
                   {
                       q_v = (q - q_prev) * (1.0 / dt);
                       //q_v = Vector12::Zero();
                   }

                   q_prev = q;
               });
}
}  // namespace gipc