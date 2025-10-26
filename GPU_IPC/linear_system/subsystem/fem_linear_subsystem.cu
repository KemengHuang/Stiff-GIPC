#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <muda/ext/eigen.h>
#include <muda/cub/device/device_reduce.h>
#include <gipc/utils/host_log.h>
#include <contact_system/contact_system.h>
namespace gipc
{
FEMLinearSubsystem::FEMLinearSubsystem(GIPC&                gipc,
                                       device_TetraData&    tetra_data,
                                       gipc::ContactSystem& contact_system)
    : m_gipc(gipc)
    , m_tetra_data(tetra_data)
    , m_contact_system(contact_system)
{
    //muda::Debug::debug_sync_all(true);
}

muda::BufferView<__GEIGEN__::Matrix12x12d> FEMLinearSubsystem::H12x12() const
{
    auto tet_offset = 0;  //m_gipc.abd_fem_count_info.fem_tet_offset;
    //auto tet_count   = m_gipc.abd_fem_count_info.fem_tet_num;
    //auto bend_count  = m_gipc.tri_edge_num;
    auto total_count = m_gipc.BH.DNum[3];
    auto offset      = m_gipc.h_cpNum[4] + m_gipc.h_cpNum_last[4] + tet_offset;
    return muda::BufferView<__GEIGEN__::Matrix12x12d>{m_gipc.BH.H12x12, total_count}
        .subview(offset);
}

muda::BufferView<uint4> FEMLinearSubsystem::H12x12_index() const
{
    auto tet_offset = 0;
    //m_gipc.abd_fem_count_info.fem_tet_offset;
    //auto tet_count   = m_gipc.abd_fem_count_info.fem_tet_num;
    //auto bend_count  = m_gipc.tri_edge_num;
    auto total_count = m_gipc.BH.DNum[3];
    auto offset      = m_gipc.h_cpNum[4] + m_gipc.h_cpNum_last[4] + tet_offset;
    return muda::BufferView<uint4>{m_gipc.BH.D4Index, total_count}.subview(offset);
}

muda::BufferView<__GEIGEN__::Matrix9x9d> FEMLinearSubsystem::H9x9() const
{
    auto tri_offset  = m_gipc.h_cpNum[3] + m_gipc.h_cpNum_last[3];
    auto tri_count   = m_gipc.triangleNum;
    auto total_count = m_gipc.BH.DNum[2];
    MUDA_ASSERT(tri_offset + tri_count == total_count, "");
    return muda::BufferView<__GEIGEN__::Matrix9x9d>{m_gipc.BH.H9x9, total_count}.subview(
        tri_offset, tri_count);
}

muda::BufferView<uint3> FEMLinearSubsystem::H9x9_index() const
{
    auto tri_offset  = m_gipc.h_cpNum[3] + m_gipc.h_cpNum_last[3];
    auto tri_count   = m_gipc.triangleNum;
    auto total_count = m_gipc.BH.DNum[2];
    MUDA_ASSERT(tri_offset + tri_count == total_count, "");
    return muda::BufferView<uint3>{m_gipc.BH.D3Index, total_count}.subview(tri_offset, tri_count);
}

muda::CBufferView<int> FEMLinearSubsystem::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::barrier_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_tetra_data.fb, m_gipc.vertexNum}.subview(offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::shape_gradient() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_tetra_data.shape_grads, m_gipc.vertexNum}.subview(
        offset, fem_count);
}

muda::BufferView<double3> FEMLinearSubsystem::dx() const
{
    auto offset    = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double3>{m_gipc._moveDir, m_gipc.vertexNum}.subview(offset, fem_count);
}

muda::BufferView<double> FEMLinearSubsystem::mass() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::BufferView<double>{m_tetra_data.masses, m_gipc.vertexNum}.subview(
        fem_offset, fem_count);
}

void FEMLinearSubsystem::report_subsystem_info()
{
    size_t hessian_block_count = 0;

    hessian_block_count += mass().size();
    hessian_block_count += m_contact_system.fem_contact_hessians().size();
    hessian_block_count += H9x9().size() * 9;
    hessian_block_count += H12x12().size() * 16;


    //std::cout << "FEMLinearSubsystem::report_subsystem_info" << std::endl;
    //std::cout << "mass size: " << mass().size() << std::endl;
    //std::cout << "contact hessian size: "
    //          << m_contact_system.fem_contact_hessians().size() << std::endl;
    //std::cout << "shape H12x12 size: " << shape_H12x12().size() << std::endl;

    this->hessian_block_count(hessian_block_count);
    this->right_hand_side_dof(dx().size() * 3);
}

namespace details
{
    template <int N>
    __device__ void fill_hessian_block(int I,
                                       muda::TripletMatrixViewer<gipc::Float, 3>& viewer,
                                       Eigen::Map<Eigen::Vector<uint32_t, N>> index,
                                       const gipc::Matrix<3 * N, 3 * N>& matrix)
    {
        constexpr int N2     = N * N;
        auto          offset = I * N2;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                viewer(offset++).write(index(i),
                                       index(j),
                                       matrix.template block<3, 3>(i * 3, j * 3));
            }
        }
    }

      template <int N>
    __device__ void fill_hessian_block(int                        I,
                                       const muda::CDense1D<int>& boundary,
                                       muda::TripletMatrixViewer<gipc::Float, 3>& viewer,
                                       Eigen::Map<Eigen::Vector<uint32_t, N>> index,
                                       const gipc::Matrix<3 * N, 3 * N>& matrix)
    {
        constexpr int N2     = N * N;
        auto          offset = I * N2;
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                if(boundary(index(i)) != 0 || boundary(index(j)) != 0)
                {
                    viewer(offset++).write(index(i), index(j), gipc::Matrix3x3::Zero());
                }
                else
                {
                    viewer(offset++).write(index(i),
                                           index(j),
                                           matrix.template block<3, 3>(i * 3, j * 3));
                }

                //viewer(offset++).write(index(i),
                //                       index(j),
                //                       matrix.template block<3, 3>(i * 3, j * 3));
            }
        }
    }

}  // namespace details


void FEMLinearSubsystem::assemble(TripletMatrixView hessian, DenseVectorView gradient)
{
    using namespace muda;

    auto mass               = this->mass();

    if(m_gipc.abd_fem_count_info.fem_point_num < 1)
        return;

    auto contact_hessian    = m_contact_system.fem_contact_hessians();
    auto shape_H9x9         = this->H9x9();
    auto shape_H9x9_index   = this->H9x9_index();
    auto shape_H12x12       = this->H12x12();
    auto shape_H12x12_index = this->H12x12_index();
    auto barrier_gradient   = this->barrier_gradient();
    auto shape_gradient     = this->shape_gradient();
    auto fem_point_offset   = m_gipc.abd_fem_count_info.fem_point_offset;

    auto move_dir = this->dx();

    auto offset = 0;
    auto count  = 0;

    {  // lumped mass hessian
        offset += count;
        count = mass.size();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(mass.size(),
                   [mass = mass.viewer().name("mass"),
                    hessian = hessian.subview(offset, count).viewer().name("hessian")] __device__(int i) mutable
                   {
                       // mass
                       hessian(i).write(i, i, mass(i) * gipc::Matrix3x3::Identity());
                   });
    }

 {  // contact hessian
        offset += count;
        count = contact_hessian.size();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(m_contact_system.fem_contact_hessians().size(),
                   [btype = boundary_type().cviewer().name("boundary_type"),
                    hessian = hessian.subview(offset, count).viewer().name("hessian"),
                    contact_hessian =
                        m_contact_system.fem_contact_hessians().viewer().name("contact_hessian"),
                    fem_point_offset] __device__(int I) mutable
                   {
                       auto& H = contact_hessian(I);
                       // because point id is global, we need to subtract the offset of fem points
                       Vector2i ij = H.point_id - Vector2i::Ones() * fem_point_offset;

                       if(btype(ij(0)) != 0 || btype(ij(1)) != 0)
                       {
                           hessian(I).write(ij(0), ij(1), Matrix3x3::Zero());
                       }
                       else
                       {
                           hessian(I).write(ij(0), ij(1), H.hessian);
                       }
                   });
    }


    {  //shape 9x9 hessian
        offset += count;
        count = shape_H9x9.size() * 9;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(shape_H9x9.size(),
                   [H9x9       = shape_H9x9.viewer().name("H9x9"),
                    H9x9_index = shape_H9x9_index.viewer().name("index"),
                    hessian = hessian.subview(offset, count).viewer().name("hessian"),
                    btype = boundary_type().cviewer().name("boundary_type"),
                    fem_point_offset] __device__(int I) mutable
                   {
                       gipc::Matrix9x9 H;
                       auto&           srcH = H9x9(I);
                       for(int j = 0; j < 9; ++j)
                       {
                           for(int k = 0; k < 9; ++k)
                           {
                               H(j, k) = srcH.m[j][k];
                           }
                       }
                       //H          = gipc::Matrix9x9::Zero();
                       auto index = eigen::as_eigen(H9x9_index(I));
                       index.array() -= fem_point_offset;
                       details::fill_hessian_block(I, btype, hessian, index, H);
                   });
    }

    {  // shape 12x12 hessian

        offset += count;
        count = shape_H12x12.size() * 16;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(shape_H12x12.size(),
                   [H12x12       = shape_H12x12.viewer().name("H12x12"),
                    H12x12_index = shape_H12x12_index.viewer().name("index"),
                    hessian = hessian.subview(offset, count).viewer().name("hessian"),
                    btype = boundary_type().cviewer().name("boundary_type"),
                    fem_point_offset] __device__(int I) mutable
                   {
                       gipc::Matrix12x12 H;
                       auto&             srcH = H12x12(I);
                       for(int j = 0; j < 12; ++j)
                       {
                           for(int k = 0; k < 12; ++k)
                           {
                               H(j, k) = srcH.m[j][k];
                           }
                       }
                       //H          = gipc::Matrix12x12::Zero();
                       auto index = eigen::as_eigen(H12x12_index(I));
                       index.array() -= fem_point_offset;
                       details::fill_hessian_block(I, btype, hessian, index, H);
                   });
    }


    {  // gradient = contact_gradient + shape_gradient
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(barrier_gradient.size(),
                   [b     = barrier_gradient.viewer().name("barrier_gradient"),
                    s     = shape_gradient.viewer().name("shape_gradient"),
                    btype = boundary_type().cviewer().name("boundary_type"),
                    gradient = gradient.viewer().name("gradient")] __device__(int i) mutable
                   {
                       if(btype(i) != 0)
                       {
                           gradient.segment<3>(i * 3).as_eigen() = Vector3::Zero();
                       }
                       else
                       {
                           gradient.segment<3>(i * 3).as_eigen() =
                               eigen::as_eigen(b(i)) + eigen::as_eigen(s(i));
                       }

                       //gradient.segment<3>(i * 3).as_eigen() =
                       //    eigen::as_eigen(b(i)) + eigen::as_eigen(s(i));
                   });
    }


    //Eigen::VectorX<double> grad(gradient.size());
    //gradient.buffer_view().copy_to(grad.data());
    //std::cout << "gradient: " << grad.transpose() << std::endl;
}

void FEMLinearSubsystem::retrieve_solution(CDenseVectorView dx)
{
    using namespace muda;

    auto move_dir = this->dx();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(move_dir.size(),
               [dx        = dx.viewer().name("dx"),
                move_dir  = move_dir.viewer().name("move_dir"),
                local_tol = m_local_tol] __device__(int i) mutable
               {
                   auto md = eigen::as_eigen(move_dir(i));
                   md      = dx.segment<3>(i * 3).as_eigen();
                   //if(md.squaredNorm() < local_tol * local_tol)
                   //{
                   //    md.setZero();
                   //}
               });
}


bool FEMLinearSubsystem::accuracy_statisfied(CDenseVectorView residual)
{
    using namespace muda;

    //m_local_squared_norm.resize(dx().size());

    //ParallelFor()
    //    .kernel_name(__FUNCTION__)
    //    .apply(dx().size(),
    //           [residual     = residual.viewer().name("residual"),
    //            squared_norm = m_local_squared_norm.viewer().name(
    //                "squared_norm")] __device__(int i) mutable {
    //               squared_norm(i) = residual.segment<3>(i * 3).as_eigen().squaredNorm();
    //           });

    //DeviceReduce().Max(m_local_squared_norm.data(),
    //                   m_max_squared_norm.data(),
    //                   m_local_squared_norm.size());

    //gipc::Float max_norm = m_max_squared_norm;
    //// m_local_tol          = 1e-11;
    //// GIPC_INFO("FEMLinearSubsystem max_local_residual_norm: {}", std::sqrt(max_norm));
    //return max_norm < m_local_tol * m_local_tol;
    return true;
}

void FEMLinearSubsystem::set_local_tolerance(gipc::Float tol)
{
    m_local_tol = tol;
}
}  // namespace gipc