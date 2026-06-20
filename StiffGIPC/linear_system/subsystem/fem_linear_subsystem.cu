#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <cuda_tools/cuda_tools.h>

namespace
{
using gipc::Float;

__global__ void fem_assemble_kernel(int                                  n,
                                    cudatool::Dense1D<double3>           b,
                                    cudatool::Dense1D<double3>           s,
                                    cudatool::CDense1D<int>              btype,
                                    cudatool::DenseVectorViewer<Float>   gradient)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    if(btype(i) != 0)
    {
        gradient.segment<3>(i * 3).as_eigen() = gipc::Vector3::Zero();
    }
    else
    {
        gradient.segment<3>(i * 3).as_eigen() =
            cudatool::eigen::as_eigen(b(i)) + cudatool::eigen::as_eigen(s(i));
    }
}

__global__ void fem_retrieve_solution_kernel(int                                  n,
                                             cudatool::CDenseVectorViewer<Float>  dx,
                                             cudatool::Dense1D<double3>           move_dir,
                                             gipc::Float                          local_tol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    auto md = cudatool::eigen::as_eigen(move_dir(i));
    md      = dx.segment<3>(i * 3).as_eigen();
}
}  // namespace

namespace gipc
{
FEMLinearSubsystem::FEMLinearSubsystem(GIPC& gipc, device_TetraData& tetra_data)
    : m_gipc(gipc)
    , m_tetra_data(tetra_data)
{
    //cudatool::Debug::debug_sync_all(true);
}


cudatool::CBufferView<int> FEMLinearSubsystem::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return cudatool::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

cudatool::BufferView<double3> FEMLinearSubsystem::barrier_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return cudatool::BufferView<double3>{m_tetra_data.fb, m_gipc.vertexNum}.subview(offset, fem_count);
}

cudatool::BufferView<double3> FEMLinearSubsystem::shape_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return cudatool::BufferView<double3>{m_tetra_data.shape_grads, m_gipc.vertexNum}.subview(
        offset, fem_count);
}

cudatool::BufferView<double3> FEMLinearSubsystem::dx() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return cudatool::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(offset, fem_count);
}

cudatool::BufferView<double> FEMLinearSubsystem::mass() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return cudatool::BufferView<double>{m_tetra_data.masses, m_gipc.vertexNum}.subview(
        fem_offset, fem_count);
}

void FEMLinearSubsystem::report_subsystem_info()
{
    this->right_hand_side_dof(dx().size() * 3);
}


void FEMLinearSubsystem::assemble(DenseVectorView gradient)
{
    using namespace cudatool;

    if(m_gipc.abd_fem_count_info.fem_point_num < 1)
        return;

    auto barrier_gradient = this->barrier_gradient();
    auto shape_gradient   = this->shape_gradient();

    LaunchCudaKernal_default(static_cast<int>(barrier_gradient.size()),
                             256,
                             0,
                             fem_assemble_kernel,
                             static_cast<int>(barrier_gradient.size()),
                             barrier_gradient.viewer(),
                             shape_gradient.viewer(),
                             boundary_type().cviewer(),
                             gradient.viewer());
}

void FEMLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace cudatool;

    auto move_dir = this->dx();

    LaunchCudaKernal_default(static_cast<int>(move_dir.size()),
                             256,
                             0,
                             fem_retrieve_solution_kernel,
                             static_cast<int>(move_dir.size()),
                             dx.cviewer(),
                             move_dir.viewer(),
                             m_local_tol);
}

void FEMLinearSubsystem::set_local_tolerance(gipc::Float tol)
{
    m_local_tol = tol;
}
}  // namespace gipc
