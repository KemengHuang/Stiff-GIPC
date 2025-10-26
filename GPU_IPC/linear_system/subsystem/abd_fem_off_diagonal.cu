#include <linear_system/subsystem/abd_fem_off_diagonal.h>
#include <abd_system/abd_system.h>
#include <contact_system/contact_system.h>

namespace gipc
{
ABDFEMOffDiagonal::ABDFEMOffDiagonal(GIPC&               gipc,
                                     device_TetraData&   tetra_data,
                                     ContactSystem&      contact_system,
                                     ABDLinearSubsystem& abd,
                                     FEMLinearSubsystem& fem)
    : Base(abd, fem)
    , m_gipc(gipc)
    , m_contact_system(contact_system)
    , m_abd_sim_data(abd.m_abd_sim_data)
    , m_tetra_data(tetra_data)
{
}

muda::CBufferView<int> ABDFEMOffDiagonal::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

void ABDFEMOffDiagonal::report_subsystem_info()
{
    //tex: $\mathbf{J} \mathbf{H}$
    auto upper_hessian_count = m_contact_system.abd_fem_contact_hessians().size() * 4;
    auto lower_hessian_count = m_contact_system.fem_abd_contact_hessians().size() * 4;

    hessian_block_count(upper_hessian_count, lower_hessian_count);
}

void ABDFEMOffDiagonal::assemble(TripletMatrixView upper, TripletMatrixView lower)
{
    // NOTE: the (i,j) indexes into the upper and lower matrices are the local point ids w.r.t.
    // the related linear system, i.e. local in the sense of the ABD or FEM system.

    // We need to fill the upper and lower matrices, as if we are filling two submatrix, indexing from (0,0).


    auto upper_count = m_contact_system.abd_fem_contact_hessians().size();
    auto lower_count = m_contact_system.fem_abd_contact_hessians().size();

    MUDA_ASSERT(upper_count == lower_count,
                "ABDFEMOffDiagonal::assemble: upper_count != lower_count");

    auto count = upper_count;
    if(count < 1)
        return;
    using namespace muda;
    auto  point_to_body = m_abd_sim_data.unique_point_id_to_body_id();
    auto& Js            = m_abd_sim_data.device.unique_point_id_to_J;

    auto abd_point_offset = m_gipc.abd_fem_count_info.abd_point_offset;
    auto abd_body_offset  = m_gipc.abd_fem_count_info.abd_body_offset;
    auto fem_point_offset = m_gipc.abd_fem_count_info.fem_point_offset;


    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(count,
               [point_to_body = point_to_body.cviewer().name("point_to_body"),
                btype         = boundary_type().cviewer().name("boundary_type"),
                Js            = Js.cviewer().name("Js"),
                upper         = upper.viewer().name("upper"),
                abd_fem_contact =
                    m_contact_system.abd_fem_contact_hessians().cviewer().name("abd_fem_contact"),
                fem_abd_contact =
                    m_contact_system.fem_abd_contact_hessians().cviewer().name("fem_abd_contact"),
                lower = lower.viewer().name("lower"),
                abd_point_offset,
                abd_body_offset,
                fem_point_offset] __device__(int I) mutable
               {
                   // 1. process upper : ABD-FEM
                   {
                       auto&& [ij, H3x3]       = abd_fem_contact(I);
                       auto i_abd              = ij(0);  // global point id
                       auto j_fem              = ij(1);  // global point id
                       //if(btype(j_fem) == 0)
                       {

                           auto local_fem_point_id = j_fem - fem_point_offset;

                           auto body_id = point_to_body(i_abd);  // global body id

                           auto local_abd_body_id  = body_id - abd_body_offset;
                           auto local_abd_point_id = i_abd - abd_point_offset;
                           //tex: $\mathbf{J}_{3\times 12}$
                           gipc::ABDJacobi  J = Js(local_abd_point_id);
                           gipc::Matrix12x3 H = J.to_mat().transpose() * H3x3;
                           //tex:
                           //$$
                           // \mathbf{H} = \begin{bmatrix}
                           //  \mathbf{H}_{1} \\ \mathbf{H}_{2} \\ \mathbf{H}_{3} \\ \mathbf{H}_{4}
                           //\end{bmatrix}
                           //$$
                           auto offset = 4 * I;
                           if(btype(local_fem_point_id) != 0)
                           {
                               H.setZero();
                           }
                           for(int i = 0; i < 4; ++i, ++offset)
                           {
                               upper(offset).write(4 * local_abd_body_id + i,
                                                   local_fem_point_id,
                                                   H.block<3, 3>(3 * i, 0));
                           }
                       }
                   }
                   // 2. process lower : FEM-ABD
                   {
                       auto&& [ij, H3x3]       = fem_abd_contact(I);
                       auto i_fem              = ij(0);  // global point id
                       auto j_abd              = ij(1);  // global point id

                       //if(btype(i_fem) == 0)
                       {

                           auto local_fem_point_id = i_fem - fem_point_offset;

                           auto body_id = point_to_body(j_abd);  // global body id

                           auto local_abd_body_id  = body_id - abd_body_offset;
                           auto local_abd_point_id = j_abd - abd_point_offset;
                           //tex: $\mathbf{J}_{3\times 12}$
                           gipc::ABDJacobi  J = Js(local_abd_point_id);
                           gipc::Matrix3x12 H = H3x3 * J.to_mat();
                           //tex:
                           //$$
                           // \mathbf{H} = \begin{bmatrix}
                           //  \mathbf{H}_{1} & \mathbf{H}_{2} & \mathbf{H}_{3} & \mathbf{H}_{4}
                           //\end{bmatrix}
                           //$$
                           auto offset = 4 * I;
                           if(btype(local_fem_point_id) != 0)
                           {
                               H.setZero();
                           }
                           for(int i = 0; i < 4; ++i, ++offset)
                           {
                               lower(offset).write(local_fem_point_id,
                                                   4 * local_abd_body_id + i,
                                                   H.block<3, 3>(0, 3 * i));
                           }
                       }
                   }
               });
}
}  // namespace gipc