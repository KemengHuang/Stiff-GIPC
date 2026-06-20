#include <linear_system/linear_system/i_preconditioner.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/global_linear_system.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_tools.h>

namespace
{
__global__ void calculate_subsystem_bcoo_indices_kernel(int           n,
                                                        int*          rows,
                                                        int*          cols,
                                                        gipc::IndexT  offset,
                                                        gipc::IndexT  end,
                                                        uint32_t*     indices_input,
                                                        uint32_t*     flags)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    auto i           = rows[I];
    auto j           = cols[I];
    bool valid       = (i >= offset && i < end) && (j >= offset && j < end);
    indices_input[I] = I;  // -I for invalid
    flags[I]         = valid ? 1 : 0;
}
}  // namespace

namespace gipc
{
IPreconditioner::~IPreconditioner() {}

Json IPreconditioner::as_json() const
{
    Json j;
    j["type"] = typeid(*this).name();
    return j;
}

cudatool::LinearSystemContext& IPreconditioner::ctx() const
{
    return m_system->m_context;
}

LocalPreconditioner::LocalPreconditioner(DiagonalSubsystem& subsystem)
    : m_subsystem(&subsystem)
{
}

int LocalPreconditioner::get_offset() const
{
    return m_subsystem->m_dof_offset / 3;
}

uint32_t* LocalPreconditioner::calculate_subsystem_bcoo_indices(int& number) const
{
    auto offset = m_subsystem->m_dof_offset / 3;
    auto end    = offset + m_subsystem->m_right_hand_side_dof / 3;

    auto index_input  = m_system->gipc_global_triplet->block_index();
    auto index_output = m_system->gipc_global_triplet->block_sort_index();
    auto flags        = m_system->gipc_global_triplet->block_temp_buffer();

    LaunchCudaKernal_default(m_system->gipc_global_triplet->h_unique_key_number,
                             256,
                             0,
                             calculate_subsystem_bcoo_indices_kernel,
                             m_system->gipc_global_triplet->h_unique_key_number,
                             m_system->gipc_global_triplet->block_row_indices(),
                             m_system->gipc_global_triplet->block_col_indices(),
                             offset,
                             end,
                             index_input,
                             flags);

    cudatool::DeviceSelect().Flagged(
        index_input,
        flags,
        index_output,
        m_system->gipc_global_triplet->d_unique_key_number,
        m_system->gipc_global_triplet->h_unique_key_number);

    int h_count;
    CUDA_SAFE_CALL(cudaMemcpy(&h_count,
                              m_system->gipc_global_triplet->d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    number = h_count;
    return index_output;
}

Eigen::Matrix3d* LocalPreconditioner::system_bcoo_matrix() const
{
    return m_system->gipc_global_triplet->block_values();
}

int* LocalPreconditioner::system_bcoo_rows() const
{
    return m_system->gipc_global_triplet->block_row_indices();
}

int* LocalPreconditioner::system_bcoo_cols() const
{
    return m_system->gipc_global_triplet->block_col_indices();
}

void LocalPreconditioner::do_apply(cudatool::CDenseVectorView<Float> r,
                                   cudatool::DenseVectorView<Float>  z)
{
    auto dof_offset = m_subsystem->dof_offset()[0];
    auto dof_count  = m_subsystem->right_hand_side_dof();

    apply(r.subview(dof_offset, dof_count), z.subview(dof_offset, dof_count));
}

void LocalPreconditioner::do_assemble(GIPCTripletMatrix& global_triplets)
{
    assemble();
}



void GlobalPreconditioner::do_apply(cudatool::CDenseVectorView<Float> r,
                                    cudatool::DenseVectorView<Float>  z)
{
    apply(r, z);
}

void GlobalPreconditioner::do_assemble(GIPCTripletMatrix& global_triplets)
{
    assemble(global_triplets);
}

}  // namespace gipc
