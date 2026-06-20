#include <linear_system/preconditioner/abd_preconditioner.h>
#include <abd_system/abd_system.h>
#include <linear_system/subsystem/abd_linear_subsystem.h>
#include <cuda_tools/cuda_tools.h>

namespace
{
using gipc::Float;

__global__ void abd_preconditioner_apply_kernel(int                                 n,
                                                cudatool::CDenseVectorViewer<Float> r,
                                                cudatool::DenseVectorViewer<Float>  z,
                                                cudatool::Dense1D<gipc::Matrix12x12> inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    z.segment<12>(i * 12).as_eigen() =
        inv(i) * r.segment<12>(i * 12).as_eigen();
}
}  // namespace

namespace gipc
{
ABDPreconditioner::ABDPreconditioner(ABDLinearSubsystem& subsystem, ABDSystem& abd, ABDSimData& sim_data)
    : Base(subsystem)
    , m_abd(abd)
    , m_sim_data(sim_data)
{
    preconditioner_id = 0;
}

void ABDPreconditioner::assemble()
{
    m_abd._cal_abd_system_preconditioner(m_sim_data);
}

void ABDPreconditioner::apply(cudatool::CDenseVectorView<Float> r,
                              cudatool::DenseVectorView<Float>  z)
{
    using namespace cudatool;

    auto abd_body_count = m_sim_data.abd_fem_count_info().abd_body_num;
    auto abd_inv_diag   = m_abd.abd_system_diag_preconditioner.view();

    LaunchCudaKernal_default(static_cast<int>(abd_body_count),
                             256,
                             0,
                             abd_preconditioner_apply_kernel,
                             static_cast<int>(abd_body_count),
                             r.cviewer(),
                             z.viewer(),
                             abd_inv_diag.viewer());
}
}  // namespace gipc
