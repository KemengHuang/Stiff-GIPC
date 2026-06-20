#include <abd_system/abd_system.h>
#include "cuda_tools/cuda_tools.h"
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
__global__ void cal_x_from_q_double3_kernel(int                         n,
                                            cudatool::BufferView<double3>  verts,
                                            cudatool::CBufferView<I32>     body_ids,
                                            cudatool::BufferView<Vector12> qs,
                                            cudatool::BufferView<ABDJacobi> Js)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto        body_id = body_ids[i];
    const auto& J       = Js[i];
    const auto& q       = qs[body_id];
    Vector3     x       = J * q;

    auto& vert = verts[i];

    vert.x = x.x();
    vert.y = x.y();
    vert.z = x.z();
}

__global__ void cal_dx_from_dq_double3_kernel(int                         n,
                                              cudatool::BufferView<double3>  move_dirs,
                                              cudatool::BufferView<Vector12> dqs,
                                              cudatool::BufferView<Vector12> qs,
                                              cudatool::CBufferView<I32>     body_ids,
                                              cudatool::BufferView<ABDJacobi> Js)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto body_id = body_ids[i];

    const auto& J  = Js[i];
    const auto& dq = dqs[body_id];
    const auto& q  = qs[body_id];
    // print("body_id = %d\n", body_id);

    auto q_new = q - dq;
    auto x_new = J * q_new;

    Vector3 dx       = J * q - x_new;
    auto&   move_dir = move_dirs[i];

    // we need to negate the dx, because GIPC use gradient, not the negative gradient
    move_dir.x = dx.x();
    move_dir.y = dx.y();
    move_dir.z = dx.z();

    //print("dx(%d): %f %f %f\n", i, dx.x(), dx.y(), dx.z());
}

__global__ void cal_x_from_q_Vector3_kernel(int                         n,
                                            cudatool::BufferView<Vector3>  verts,
                                            cudatool::CBufferView<I32>     body_ids,
                                            cudatool::BufferView<Vector12> qs,
                                            cudatool::BufferView<ABDJacobi> Js)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto        body_id = body_ids[i];
    const auto& J       = Js[i];
    const auto& q       = qs[body_id];
    Vector3     x       = J * q;

    auto& vert = verts[i];

    vert = x;
}

__global__ void cal_dx_from_dq_Vector3_kernel(int                         n,
                                              cudatool::BufferView<Vector3>  move_dirs,
                                              cudatool::BufferView<Vector12> dqs,
                                              cudatool::BufferView<Vector12> qs,
                                              cudatool::CBufferView<I32>     body_ids,
                                              cudatool::BufferView<ABDJacobi> Js)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto body_id = body_ids[i];

    const auto& J  = Js[i];
    const auto& dq = dqs[body_id];
    const auto& q  = qs[body_id];
    // print("body_id = %d\n", body_id);

    auto q_new = q - dq;
    auto x_new = J * q_new;

    Vector3 dx       = J * q - x_new;
    auto&   move_dir = move_dirs[i];

    // we need to negate the dx, because GIPC use gradient, not the negative gradient
    move_dir = dx;

    //print("dx(%d): %f %f %f\n", i, dx.x(), dx.y(), dx.z());
}

void ABDSystem::cal_x_from_q(ABDSimData& sim_data, cudatool::BufferView<double3> vertices)
{
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    LaunchCudaKernal_default((int)unique_point_count,
                             256,
                             0,
                             cal_x_from_q_double3_kernel,
                             (int)unique_point_count,
                             vertices,
                             body_id,
                             abd.body_id_to_q.view(),
                             abd.unique_point_id_to_J.view());
}

void ABDSystem::cal_dx_from_dq(ABDSimData& sim_data, cudatool::BufferView<double3> move_dir)
{
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    LaunchCudaKernal_default((int)move_dir.size(),
                             256,
                             0,
                             cal_dx_from_dq_double3_kernel,
                             (int)move_dir.size(),
                             move_dir,
                             abd.body_id_to_dq.view(),
                             abd.body_id_to_q.view(),
                             body_id,
                             abd.unique_point_id_to_J.view());
}
void ABDSystem::cal_x_from_q(ABDSimData& sim_data, cudatool::BufferView<Vector3> vertices)
{
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    LaunchCudaKernal_default((int)unique_point_count,
                             256,
                             0,
                             cal_x_from_q_Vector3_kernel,
                             (int)unique_point_count,
                             vertices,
                             body_id,
                             abd.body_id_to_q.view(),
                             abd.unique_point_id_to_J.view());
}

void ABDSystem::cal_dx_from_dq(ABDSimData& sim_data, cudatool::BufferView<Vector3> move_dir)
{
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();

    LaunchCudaKernal_default((int)move_dir.size(),
                             256,
                             0,
                             cal_dx_from_dq_Vector3_kernel,
                             (int)move_dir.size(),
                             move_dir,
                             abd.body_id_to_dq.view(),
                             abd.body_id_to_q.view(),
                             body_id,
                             abd.unique_point_id_to_J.view());
}
}  // namespace gipc
