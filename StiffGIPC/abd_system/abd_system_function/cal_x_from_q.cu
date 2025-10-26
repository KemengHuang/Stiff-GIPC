#include <abd_system/abd_system.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
void ABDSystem::cal_x_from_q(ABDSimData& sim_data, muda::BufferView<double3> vertices)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(unique_point_count,
               [verts    = vertices.viewer().name("verts"),
                body_ids = body_id.cviewer().name("body_id"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                Js = abd.unique_point_id_to_J.cviewer().name("Js")] __device__(int i) mutable
               {
                   auto        body_id = body_ids(i);
                   const auto& J       = Js(i);
                   const auto& q       = qs(body_id);
                   Vector3     x       = J * q;

                   auto& vert = verts(i);

                   vert.x = x.x();
                   vert.y = x.y();
                   vert.z = x.z();
               });
}

void ABDSystem::cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<double3> move_dir)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(move_dir.size(),
               [move_dirs = move_dir.viewer().name("move_dirs"),
                dqs       = abd.body_id_to_dq.cviewer().name("dqs"),
                qs        = abd.body_id_to_q.cviewer().name("qs"),
                body_ids  = body_id.cviewer().name("body_ids"),
                Js = abd.unique_point_id_to_J.cviewer().name("Js")] __device__(int i) mutable
               {
                   auto body_id = body_ids(i);

                   const auto& J  = Js(i);
                   const auto& dq = dqs(body_id);
                   const auto& q  = qs(body_id);
                   // print("body_id = %d\n", body_id);

                   auto q_new = q - dq;
                   auto x_new = J * q_new;

                   Vector3 dx       = J * q - x_new;
                   auto&   move_dir = move_dirs(i);

                   // we need to negate the dx, because GIPC use gradient, not the negative gradient
                   move_dir.x = dx.x();
                   move_dir.y = dx.y();
                   move_dir.z = dx.z();

                   //print("dx(%d): %f %f %f\n", i, dx.x(), dx.y(), dx.z());
               });
}
void ABDSystem::cal_x_from_q(ABDSimData& sim_data, muda::BufferView<Vector3> vertices)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(unique_point_count,
               [verts    = vertices.viewer().name("verts"),
                body_ids = body_id.cviewer().name("body_id"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                Js = abd.unique_point_id_to_J.cviewer().name("Js")] __device__(int i) mutable
               {
                   auto        body_id = body_ids(i);
                   const auto& J       = Js(i);
                   const auto& q       = qs(body_id);
                   Vector3     x       = J * q;

                   auto& vert = verts(i);

                   vert = x;
               });
}

void ABDSystem::cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<Vector3> move_dir)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(move_dir.size(),
               [move_dirs = move_dir.viewer().name("move_dirs"),
                dqs       = abd.body_id_to_dq.cviewer().name("dqs"),
                qs        = abd.body_id_to_q.cviewer().name("qs"),
                body_ids  = body_id.cviewer().name("body_ids"),
                Js = abd.unique_point_id_to_J.cviewer().name("Js")] __device__(int i) mutable
               {
                   auto body_id = body_ids(i);

                   const auto& J  = Js(i);
                   const auto& dq = dqs(body_id);
                   const auto& q  = qs(body_id);
                   // print("body_id = %d\n", body_id);

                   auto q_new = q - dq;
                   auto x_new = J * q_new;

                   Vector3 dx       = J * q - x_new;
                   auto&   move_dir = move_dirs(i);

                   // we need to negate the dx, because GIPC use gradient, not the negative gradient
                   move_dir = dx;

                   //print("dx(%d): %f %f %f\n", i, dx.x(), dx.y(), dx.z());
               });
}
}  // namespace gipc