#include <linear_system/preconditioner/fem_mas_preconditioner.h>
#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <gipc/utils/timer.h>
namespace gipc
{
MAS_Preconditioner::MAS_Preconditioner(FEMLinearSubsystem& subsystem,
                                       BHessian&           mBH,
                                       MASPreconditioner&  mMAS,
                                       double*             mMasses,
                                       uint32_t*           mCpNum)
    : Base(subsystem)
    , BH(mBH)
    , MAS_Prec(mMAS)
    , masses(mMasses)
    , cpNum(mCpNum)
{
}

void MAS_Preconditioner::assemble()
{
    double collision_num = *cpNum;
    gipc::Timer timer{"precomputing mas Preconditioner"};
    //MAS_Prec.setPreconditioner(BH, masses, collision_num);
    MAS_Prec.setPreconditioner_bcoo(system_bcoo_matrix(),
                                    calculate_subsystem_bcoo_indices(), get_offset(), collision_num);
}

void MAS_Preconditioner::apply(muda::CDenseVectorView<Float> r,
                              muda::DenseVectorView<Float>  z)
{

    MAS_Prec.preconditioning((double3*)r.data(), (double3*)z.data());

    //using namespace muda;

    //auto abd_body_count = m_sim_data.abd_fem_count_info().abd_body_num;
    //auto abd_inv_diag   = m_abd.abd_system_diag_preconditioner.view();

    //ParallelFor()
    //    .kernel_name(__FUNCTION__)
    //    .apply(abd_body_count,
    //           [r = r.viewer().name("r"),
    //            z = z.viewer().name("z"),
    //            inv = abd_inv_diag.viewer().name("inv")] __device__(int i) mutable
    //           {
    //               z.segment<12>(i * 12).as_eigen() =
    //                   inv(i) * r.segment<12>(i * 12).as_eigen();
    //           });
}
}  // namespace gipc
