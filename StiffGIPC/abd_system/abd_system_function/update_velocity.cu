#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_all.h>
#include "cuda_tools/cuda_tools.h"
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
__global__ void update_velocity_kernel(int                               n,
                                       cudatool::CBufferView<BodyBoundaryType> boundary_type,
                                       cudatool::BufferView<Vector12>          qs,
                                       cudatool::BufferView<Vector12>          q_vs,
                                       cudatool::BufferView<Vector12>          q_prevs,
                                       Float                                   dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto& q_v    = q_vs[i];
    auto& q_prev = q_prevs[i];

    const auto& q = qs[i];

    if(boundary_type[i] == BodyBoundaryType::Fixed)
    {
        q_v = Vector12::Zero();
    }
    else
    {
        q_v = (q - q_prev) * (1.0 / dt);
        //q_v = Vector12::Zero();
    }

    q_prev = q;
}

void ABDSystem::update_velocity(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd            = sim_data.device;
    auto& abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  boundary_type  = sim_data.body_id_to_boundary_type();
    LaunchCudaKernal_default((int)abd.body_id_to_q.size(),
                             256,
                             0,
                             update_velocity_kernel,
                             (int)abd.body_id_to_q.size(),
                             boundary_type,
                             abd.body_id_to_q.view(),
                             abd.body_id_to_q_v.view(),
                             abd.body_id_to_q_prev.view(),
                             parms.dt);
}
}  // namespace gipc
