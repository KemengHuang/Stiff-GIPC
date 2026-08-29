#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_all.h>
#include "cuda_tools/cuda_tools.h"
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
__global__ void step_forward_q_kernel(int                               n,
                                      cudatool::CBufferView<BodyBoundaryType> boundary_type,
                                      cudatool::BufferView<Vector12>          q_temps,
                                      cudatool::BufferView<Vector12>          qs,
                                      cudatool::BufferView<Vector12>          dqs,
                                      double                                  alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    if(boundary_type[i] == BodyBoundaryType::Fixed)
        return;
    qs[i] = q_temps[i] - alpha * dqs[i];
}

__global__ void step_forward_vertices_kernel(int                         n,
                                             cudatool::BufferView<double3>  vertices,
                                             cudatool::CBufferView<I32>     unique_point_id_to_body_id,
                                             cudatool::BufferView<ABDJacobi> Js,
                                             cudatool::BufferView<Vector12> q_temps,
                                             cudatool::BufferView<Vector12> qs,
                                             cudatool::BufferView<Vector12> dqs,
                                             double                         alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto        body_id = unique_point_id_to_body_id[i];
    const auto& q       = qs[body_id];
    const auto& J       = Js[i];
    auto&       vert    = vertices[i];
    auto        v       = J * q;
    vert.x              = v.x();
    vert.y              = v.y();
    vert.z              = v.z();
}

void ABDSystem::copy_q_to_q_temp(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd             = sim_data.device;
    abd.body_id_to_q_temp.copy_from(abd.body_id_to_q);
}

void ABDSystem::step_forward(ABDSimData&                sim_data,
                             cudatool::BufferView<double3>  vertexes,
                             double                     alpha)
{
    using namespace cudatool;
    auto& abd                       = sim_data.device;
    auto  abd_body_count            = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count        = sim_data.abd_fem_count_info().abd_point_num;
    auto  unique_poit_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  boundary_type       = sim_data.body_id_to_boundary_type();

    LaunchCudaKernal_default((int)abd_body_count,
                             256,
                             0,
                             step_forward_q_kernel,
                             (int)abd_body_count,
                             boundary_type,
                             abd.body_id_to_q_temp.view(),
                             abd.body_id_to_q.view(),
                             abd.body_id_to_dq.view(),
                             alpha);

    LaunchCudaKernal_default((int)unique_point_count,
                             256,
                             0,
                             step_forward_vertices_kernel,
                             (int)unique_point_count,
                             vertexes,
                             unique_poit_id_to_body_id,
                             abd.unique_point_id_to_J.view(),
                             abd.body_id_to_q_temp.view(),
                             abd.body_id_to_q.view(),
                             abd.body_id_to_dq.view(),
                             alpha);
}

}  // namespace gipc
