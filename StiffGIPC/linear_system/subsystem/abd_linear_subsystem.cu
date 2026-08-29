#include <linear_system/subsystem/abd_linear_subsystem.h>
#include <core/GIPC.cuh>
#include <abd_system/abd_system.h>
#include <abd_system/abd_sim_data.h>
#include <cuda_tools/cuda_tools.h>

namespace
{
using gipc::Float;

__global__ void abd_assemble_gradient_kernel(int                                  n,
                                             cudatool::DenseVectorViewer<Float>   dst,
                                             cudatool::CDenseVectorViewer<Float>  src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    dst(i) = src(i);
}

__global__ void abd_retrieve_solution_kernel(int                                  n,
                                             cudatool::CDenseVectorViewer<Float>  dx,
                                             cudatool::Dense1D<gipc::Vector12>    dq,
                                             gipc::Float                          local_tol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    dq(i) = dx.segment<12>(i * 12).as_eigen();
}
}  // namespace

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
    using namespace cudatool;
    LaunchCudaKernal_default(gradient.size(),
                             256,
                             0,
                             abd_assemble_gradient_kernel,
                             gradient.size(),
                             gradient.viewer(),
                             m_abd_system.system_gradient.cviewer());
}

void ABDLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace cudatool;
    auto& sim_data = m_abd_sim_data;

    auto& dq               = sim_data.device.body_id_to_dq;
    auto  abd_body_count   = dq.size();
    auto  abd_point_offset = sim_data.abd_fem_count_info().abd_point_offset;
    auto  abd_point_num    = sim_data.abd_fem_count_info().abd_point_num;

    LaunchCudaKernal_default(static_cast<int>(abd_body_count),
                             256,
                             0,
                             abd_retrieve_solution_kernel,
                             static_cast<int>(abd_body_count),
                             dx.cviewer(),
                             dq.viewer(),
                             m_local_tol);

    m_abd_system.cal_dx_from_dq(
        sim_data,
        cudatool::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(
            abd_point_offset, abd_point_num));
}

}  // namespace gipc
