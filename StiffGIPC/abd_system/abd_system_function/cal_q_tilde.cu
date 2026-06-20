#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_all.h>
#include "cuda_tools/cuda_tools.h"
#include <gipc/utils/timer.h>
namespace gipc
{
__global__ void cal_q_tilde_kernel(int                               n,
                                   cudatool::CBufferView<BodyBoundaryType> boundary_type,
                                   cudatool::BufferView<Vector12>          q_prevs,
                                   cudatool::BufferView<Vector12>          q_vs,
                                   cudatool::BufferView<Vector12>          q_tildes,
                                   cudatool::BufferView<Vector12>          affine_gravity,
                                   Float                                   dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto& q_prev = q_prevs[i];
    auto& q_v    = q_vs[i];
    auto& g      = affine_gravity[i];
    // TODO: this time, we only consider gravity
    if(boundary_type[i] == BodyBoundaryType::Fixed)
    {
        q_tildes[i] = q_prev;
    }
    else
    {
        q_tildes[i] = q_prev + q_v * dt + g * (dt * dt);
    }
}

void ABDSystem::cal_q_tilde(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd            = sim_data.device;
    auto  abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  kappa          = parms.kappa;
    auto  dt             = parms.dt;
    auto  boundary_type  = sim_data.body_id_to_boundary_type();

    LaunchCudaKernal_default((int)abd_body_count,
                             256,
                             0,
                             cal_q_tilde_kernel,
                             (int)abd_body_count,
                             boundary_type,
                             abd.body_id_to_q_prev.view(),
                             abd.body_id_to_q_v.view(),
                             abd.body_id_to_q_tilde.view(),
                             abd.body_id_to_abd_gravity.view(),
                             dt);

    //m_local_tolerance.resize(abd.body_id_to_q_tilde.size());

    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(abd_body_count,
    //           [local_tolerance = m_local_tolerance.viewer().name("local_tolerance"),
    //            q_tildes = abd.body_id_to_q_tilde.cviewer().name("q_tilde"),
    //            qs = abd.body_id_to_q.cviewer().name("q")] __device__(int i) mutable
    //           {
    //               auto& q_tilde      = q_tildes(i);
    //               auto& q            = qs(i);
    //               local_tolerance(i) = (q_tilde - q).norm();
    //           });

    //cudatool::DeviceReduce().Max(m_local_tolerance.data(),
    //                         m_local_tolerance_max.data(),
    //                         m_local_tolerance.size());

    //m_suggest_max_tolerance = m_local_tolerance_max;
}
}  // namespace gipc
