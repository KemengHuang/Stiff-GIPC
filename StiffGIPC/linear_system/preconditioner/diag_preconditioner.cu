#include <linear_system/preconditioner/diag_preconditioner.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/timer.h>

namespace
{
using gipc::Float;

__global__ void diag_assemble_kernel(int                                  n,
                                     cudatool::Dense1D<gipc::Matrix3x3>   diag,
                                     gipc::Matrix3x3*                     hessian,
                                     int*                                 rows,
                                     int*                                 cols)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    auto i = rows[I];
    auto j = cols[I];
    auto H = hessian[I];
    if(i != j)
        return;

    diag(i) = cudatool::eigen::inverse(H);
}

__global__ void apply_diag_kernel(int                                  n,
                                  cudatool::CDenseVectorViewer<Float>  r,
                                  cudatool::DenseVectorViewer<Float>   z,
                                  cudatool::Dense1D<gipc::Matrix3x3>   diag_inv)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    auto& D = diag_inv(I);
    z.segment<3>(I * 3).as_eigen() =
        D * r.segment<3>(I * 3).as_eigen();
}
}  // namespace

namespace gipc
{
namespace details
{

    void diag_assemble(cudatool::BufferView<gipc::Matrix<3, 3>>  diag_inv,
                       GIPCTripletMatrix&                   global_triplets)
    {
        LaunchCudaKernal_default(global_triplets.h_unique_key_number,
                                 256,
                                 0,
                                 diag_assemble_kernel,
                                 global_triplets.h_unique_key_number,
                                 diag_inv.viewer(),
                                 global_triplets.block_values(),
                                 global_triplets.block_row_indices(),
                                 global_triplets.block_col_indices());
    }

    void apply_diag(cudatool::CDenseVectorView<gipc::Float>  r,
                    cudatool::DenseVectorView<gipc::Float>   z,
                    cudatool::BufferView<gipc::Matrix<3, 3>> diag_inv)
    {
        LaunchCudaKernal_default(static_cast<int>(diag_inv.size()),
                                 256,
                                 0,
                                 apply_diag_kernel,
                                 static_cast<int>(diag_inv.size()),
                                 r.cviewer(),
                                 z.viewer(),
                                 diag_inv.viewer());
    }
}  // namespace details


void DiagPreconditioner::assemble(GIPCTripletMatrix& global_triplets)
{
    gipc::Timer timer{"precomputing Preconditioner"};
    auto        cols = global_triplets.block_cols();
    m_diag3x3.resize(cols);
    m_diag3x3.view().fill(gipc::Matrix3x3::Identity());
    details::diag_assemble(m_diag3x3.view(), global_triplets);
}

void DiagPreconditioner::apply(cudatool::CDenseVectorView<gipc::Float> r,
                                  cudatool::DenseVectorView<gipc::Float>  z)
{
    //z.buffer_view().copy_from(r.buffer_view());
    details::apply_diag(r, z, m_diag3x3);
}
}  // namespace OLD_GIPC
