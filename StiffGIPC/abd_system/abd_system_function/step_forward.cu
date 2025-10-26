#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
void ABDSystem::copy_q_to_q_temp(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd             = sim_data.device;
    abd.body_id_to_q_temp = abd.body_id_to_q;
}

void ABDSystem::copy_q_to_q_temp(ABDSimData& sim_data, muda::BufferView<double3> vertices_temp)
{
    using namespace muda;
    auto& abd                      = sim_data.device;
    abd.body_id_to_q_temp          = abd.body_id_to_q;
    auto abd_body_count            = sim_data.abd_fem_count_info().abd_body_num;
    auto unique_point_count        = sim_data.abd_fem_count_info().abd_point_num;
    auto unique_poit_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto body_id_to_is_fixed       = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_point_count,
               [vertices = vertices_temp.viewer().name("vertices"),
                unique_point_id_to_body_id =
                    unique_poit_id_to_body_id.cviewer().name("unique_point_id_to_body_id"),
                is_fixed = body_id_to_is_fixed.cviewer().name("is_fixed"),
                Js       = abd.unique_point_id_to_J.cviewer().name("Js"),
                qs = abd.body_id_to_q.viewer().name("qs")] __device__(int i) mutable
               {
                   auto        unique_point_id = i;
                   auto        body_id         = unique_point_id_to_body_id(i);
                   const auto& q               = qs(body_id);
                   const auto& J               = Js(unique_point_id);
                   auto        v               = J * q;
                   auto&       res             = vertices(unique_point_id);
                   res.x                       = v.x();
                   res.y                       = v.y();
                   res.z                       = v.z();
               });

    abd.body_id_to_q_temp = abd.body_id_to_q;
}

void ABDSystem::step_forward(ABDSimData&                sim_data,
                             muda::BufferView<double3>  vertexes,
                             muda::CBufferView<double3> vertexesTemp,
                             double                     alpha)
{
    using namespace muda;
    auto& abd                       = sim_data.device;
    auto  abd_body_count            = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count        = sim_data.abd_fem_count_info().abd_point_num;
    auto  unique_poit_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  boundary_type       = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(abd_body_count,
               [vertices = vertexes.viewer().name("vertices"),
                boundary_type = boundary_type.cviewer().name("is_fixed"),
                q_temps  = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                dqs      = abd.body_id_to_dq.cviewer().name("dqs"),
                alpha] __device__(int i) mutable
               {
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                       return;
                   qs(i) = q_temps(i) - alpha * dqs(i);
               });

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_point_count,
               [vertices     = vertexes.viewer().name("vertices"),
                vertexesTemp = vertexesTemp.viewer().name("vertexesTemp"),
                unique_point_id_to_body_id =
                    unique_poit_id_to_body_id.cviewer().name("unique_point_id_to_body_id"),
                boundary_type = boundary_type.cviewer().name("is_fixed"),
                Js       = abd.unique_point_id_to_J.cviewer().name("Js"),
                q_temps  = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                dqs      = abd.body_id_to_dq.cviewer().name("dqs"),
                alpha] __device__(int i) mutable
               {
                   auto        unique_point_id = i;
                   auto        body_id         = unique_point_id_to_body_id(i);
                   const auto& q               = qs(body_id);
                   const auto& J               = Js(unique_point_id);
                   auto&       vert            = vertices(unique_point_id);
                   auto&       vert_old        = vertexesTemp(unique_point_id);
                   auto        v               = J * q;
                   vert.x                      = v.x();
                   vert.y                      = v.y();
                   vert.z                      = v.z();
               });
}

void ABDSystem::step_forward(ABDSimData&                sim_data,
                             muda::BufferView<Vector3>  vertexes,
                             muda::CBufferView<Vector3> vertexesTemp,
                             double                     alpha)
{
    using namespace muda;
    auto& abd                       = sim_data.device;
    auto  abd_count                 = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count        = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_poit_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_id_to_is_fixed       = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(abd_count,
               [vertices = vertexes.viewer().name("vertices"),
                is_fixed = body_id_to_is_fixed.cviewer().name("is_fixed"),
                q_temps  = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                dqs      = abd.body_id_to_dq.cviewer().name("dqs"),
                alpha] __device__(int i) mutable
               {
                   if(is_fixed(i) == BodyBoundaryType::Fixed)
                       return;
                   qs(i) = q_temps(i) - alpha * dqs(i);
               });

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_point_count,
               [vertices     = vertexes.viewer().name("vertices"),
                vertexesTemp = vertexesTemp.viewer().name("vertexesTemp"),
                unique_point_id_to_body_id =
                    unique_poit_id_to_body_id.cviewer().name("unique_point_id_to_body_id"),
                is_fixed = body_id_to_is_fixed.cviewer().name("is_fixed"),
                Js       = abd.unique_point_id_to_J.cviewer().name("Js"),
                q_temps  = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                dqs      = abd.body_id_to_dq.cviewer().name("dqs"),
                alpha] __device__(int i) mutable
               {
                   auto        unique_point_id = i;
                   auto        body_id         = unique_point_id_to_body_id(i);
                   const auto& q               = qs(body_id);
                   const auto& J               = Js(unique_point_id);
                   auto&       vert            = vertices(unique_point_id);
                   auto&       vert_old        = vertexesTemp(unique_point_id);
                   auto        v               = J * q;
                   vert                        = v;
               });
}
}  // namespace gipc