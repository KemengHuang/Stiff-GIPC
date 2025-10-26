#include <linear_system/preconditioner/diag_preconditioner.h>
#include <muda/ext/eigen/inverse.h>
#include <gipc/utils/timer.h>
namespace gipc
{
namespace details
{
    // constexpr int N = 6;

    template <int N>
    void diag_assemble(muda::BufferView<gipc::Matrix<N, N>>  diag_inv,
                       muda::CBCOOMatrixView<gipc::Float, 3> hessian)
    {
        using namespace muda;
        static_assert(N % 3 == 0, "N must be a multiple of 3");
        constexpr int Multiplier = N / 3;

        diag_inv.fill(gipc::Matrix<N, N>::Identity());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(hessian.triplet_count(),
                   [diag = diag_inv.viewer().name("diag"),
                    hessian = hessian.viewer().name("hessian")] __device__(int I) mutable
                   {
                       auto&& [i, j, H] = hessian(I);
                       auto i_div_M     = i / Multiplier;
                       auto j_div_M     = j / Multiplier;

                       if(i_div_M != j_div_M)
                           return;

                       auto i_mod_M = i % Multiplier;
                       auto j_mod_M = j % Multiplier;

                       auto& D = diag(i_div_M);

                       // (i,j) in BCOO is unique, so just assign H
                       D.template block<3, 3>(i_mod_M * 3, j_mod_M * 3) = H;
                   });


        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int I) mutable
                   {
                       auto& D = diag_inv(I);
                       D       = eigen::inverse(D);
                   });
    }

    void diag_assemble(muda::BufferView<gipc::Matrix<3, 3>>  diag_inv,
                       muda::CBCOOMatrixView<gipc::Float, 3> hessian)
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(hessian.triplet_count(),
                   [diag = diag_inv.viewer().name("diag"),
                    hessian = hessian.viewer().name("hessian")] __device__(int I) mutable
                   {
                       auto&& [i, j, H] = hessian(I);
                       if(i != j)
                           return;

                       diag(i) = eigen::inverse(H);
                   });
    }

    void apply_diag(muda::CDenseVectorView<gipc::Float>  r,
                    muda::DenseVectorView<gipc::Float>   z,
                    muda::BufferView<gipc::Matrix<3, 3>> diag_inv)
    {
        using namespace muda;

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

void DiagPreconditioner::assemble(muda::CBCOOMatrixView<gipc::Float, 3> hessian)
{
    gipc::Timer timer{"precomputing Preconditioner"};
    auto cols = hessian.total_block_cols();
    m_diag3x3.resize(cols);
    details::diag_assemble(m_diag3x3.view(), hessian);
}

void DiagPreconditioner::apply(muda::CDenseVectorView<gipc::Float> r,
                                  muda::DenseVectorView<gipc::Float>  z)
{
    //z.buffer_view().copy_from(r.buffer_view());
    details::apply_diag(r, z, m_diag3x3);
}
}  // namespace OLD_GIPC
