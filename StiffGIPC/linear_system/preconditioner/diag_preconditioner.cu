#include <linear_system/preconditioner/diag_preconditioner.h>
#include <gipc/cuda/all.h>
#include <gipc/utils/timer.h>
namespace gipc
{
namespace details
{

    void diag_assemble(gipc::cuda::BufferView<gipc::Matrix<3, 3>>  diag_inv,
                       GIPCTripletMatrix&                   global_triplets)
    {
        using namespace gipc::cuda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(global_triplets.h_unique_key_number,
                   [diag = diag_inv.viewer().name("diag"),
                    hessian = global_triplets.block_values(),
                    rows = global_triplets.block_row_indices(),
                    cols = global_triplets.block_col_indices()] __device__(int I) mutable
                   {
                       auto i           = rows[I];
                       auto j           = cols[I];
                       auto H           = hessian[I];
                       if(i != j)
                           return;

                       diag(i) = eigen::inverse(H);
                   });
    }

    void apply_diag(gipc::cuda::CDenseVectorView<gipc::Float>  r,
                    gipc::cuda::DenseVectorView<gipc::Float>   z,
                    gipc::cuda::BufferView<gipc::Matrix<3, 3>> diag_inv)
    {
        using namespace gipc::cuda;

        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = r.viewer().name("r"),
                    z = z.viewer().name("z"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int I) mutable
                   {
                       auto& D = diag_inv(I);
                       z.segment<3>(I * 3).as_eigen() =
                           D * r.segment<3>(I * 3).as_eigen();
                   });
    }
}  // namespace details


void DiagPreconditioner::assemble(GIPCTripletMatrix& global_triplets)
{
    gipc::Timer timer{"precomputing Preconditioner"};
    auto        cols = global_triplets.block_cols();
    m_diag3x3.resize(cols);
    details::diag_assemble(m_diag3x3.view(), global_triplets);
}

void DiagPreconditioner::apply(gipc::cuda::CDenseVectorView<gipc::Float> r,
                                  gipc::cuda::DenseVectorView<gipc::Float>  z)
{
    //z.buffer_view().copy_from(r.buffer_view());
    details::apply_diag(r, z, m_diag3x3);
}
}  // namespace OLD_GIPC
