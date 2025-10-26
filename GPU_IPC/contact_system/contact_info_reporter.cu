#include <contact_system/contact_info_reporter.h>
#include <GIPC.cuh>
#include <muda/ext/eigen/as_eigen.h>

namespace gipc
{
__host__ __device__ void to_eigen(Eigen::Matrix<double, 3, 3>& h, __GEIGEN__::Matrix3x3d H)
{
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            h(i, j) = H.m[i][j];
        }
    }
}

__host__ __device__ void to_eigen(Eigen::Matrix<double, 6, 6>& h, __GEIGEN__::Matrix6x6d H)
{
    for(int i = 0; i < 6; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            h(i, j) = H.m[i][j];
        }
    }
}

__host__ __device__ void to_eigen(Eigen::Matrix<double, 9, 9>& h, __GEIGEN__::Matrix9x9d H)
{
    for(int i = 0; i < 9; ++i)
    {
        for(int j = 0; j < 9; ++j)
        {
            h(i, j) = H.m[i][j];
        }
    }
}

__host__ __device__ void to_eigen(Eigen::Matrix<double, 12, 12>& h, __GEIGEN__::Matrix12x12d H)
{
    for(int i = 0; i < 12; ++i)
    {
        for(int j = 0; j < 12; ++j)
        {
            h(i, j) = H.m[i][j];
        }
    }
}

muda::BufferView<__GEIGEN__::Matrix3x3d> ContactInfoReporter::H3x3() const
{
    return muda::BufferView<__GEIGEN__::Matrix3x3d>{m_gipc.BH.H3x3, m_gipc.BH.DNum[0]};
}

muda::BufferView<uint32_t> ContactInfoReporter::H3x3_index() const
{
    return muda::BufferView<uint32_t>{m_gipc.BH.D1Index, m_gipc.BH.DNum[0]};
}

muda::BufferView<__GEIGEN__::Matrix6x6d> ContactInfoReporter::H6x6() const
{
    return muda::BufferView<__GEIGEN__::Matrix6x6d>{m_gipc.BH.H6x6, m_gipc.BH.DNum[1]};
}

muda::BufferView<uint2> ContactInfoReporter::H6x6_index() const
{
    return muda::BufferView<uint2>{m_gipc.BH.D2Index, m_gipc.BH.DNum[1]};
}

muda::BufferView<__GEIGEN__::Matrix9x9d> ContactInfoReporter::H9x9() const
{
    auto contact_count = m_gipc.h_cpNum[3] + m_gipc.h_cpNum_last[3];
    return muda::BufferView<__GEIGEN__::Matrix9x9d>{m_gipc.BH.H9x9, contact_count};
}

muda::BufferView<uint3> ContactInfoReporter::H9x9_index() const
{
    auto contact_count = m_gipc.h_cpNum[3] + m_gipc.h_cpNum_last[3];
    return muda::BufferView<uint3>{m_gipc.BH.D3Index, contact_count};
}

muda::BufferView<__GEIGEN__::Matrix12x12d> ContactInfoReporter::H12x12() const
{
    auto total_count   = m_gipc.BH.DNum[3];
    auto barrier_count = m_gipc.h_cpNum[4] + m_gipc.h_cpNum_last[4];
    return muda::BufferView<__GEIGEN__::Matrix12x12d>{m_gipc.BH.H12x12, total_count}
        .subview(0, barrier_count);
}

muda::BufferView<uint4> ContactInfoReporter::H12x12_index() const
{
    auto total_count   = m_gipc.BH.DNum[3];
    auto barrier_count = m_gipc.h_cpNum[4] + m_gipc.h_cpNum_last[4];
    return muda::BufferView<uint4>{m_gipc.BH.D4Index, total_count}.subview(0, barrier_count);
}

muda::BufferView<double3> ContactInfoReporter::barrier_gradient() const
{
    return muda::BufferView<double3>{m_tetra_data.fb, m_gipc.vertexNum};
}


ContactInfoReporter::ContactInfoReporter(GIPC& gipc, device_TetraData& tet)
    : m_gipc(gipc)
    , m_tetra_data(tet)
{
}

void ContactInfoReporter::report_contact_count(ContactCountInfo& info)
{
    info.gradient_count = m_gipc.vertexNum;
    info.hessian_count  = H3x3().size() + 4 * H6x6().size() + 9 * H9x9().size()
                         + 16 * H12x12().size();
    //std::cout << "gradient_count: " << info.gradient_count << std::endl;
    //std::cout << "hessian_count: h3x3=" << H3x3().size()
    //          << ", h6x6=" << H6x6().size() << ", h9x9=" << H9x9().size()
    //          << ", h12x12=" << H12x12().size() << std::endl;
}

void ContactInfoReporter::assemble(muda::BufferView<gipc::ContactGradient> gradients,
                                   muda::BufferView<gipc::ContactHessian> hessians)
{
    using namespace muda;

    auto bg = barrier_gradient();
    ParallelFor().apply(bg.size(),
                        [gradient_info = gradients.viewer().name("gradient_info"),
                         bg = bg.viewer().name("bg")] __device__(int i) mutable
                        {
                            auto& ginfo    = gradient_info(i);
                            ginfo.gradient = eigen::as_eigen(bg(i));
                            ginfo.point_id = i;
                        });

    {
        int  offset     = 0;
        int  count      = 0;
        auto h3x3       = H3x3();
        auto h3x3_index = H3x3_index();
        count           = h3x3.size();

        uint32_t h_h3x3_size = 0;
        VarView<uint32_t>{m_gipc._gpNum}.copy_to(&h_h3x3_size);

        if(h_h3x3_size != h3x3.size())
        {
            std::cout << "h_h3x3_size: " << h_h3x3_size
                      << ", h3x3.size(): " << h3x3.size() << std::endl;
            std::abort();
        }

        ParallelFor().apply(
            h3x3.size(),
            [hessian_info = hessians.subview(offset, count).viewer().name("hessian_info"),
             h3x3 = h3x3.viewer().name("h3x3"),
             h3x3_index = h3x3_index.viewer().name("h3x3_index")] __device__(int i) mutable
            {
                auto& hinfo = hessian_info(i);
                hinfo.hessian;
                auto& H = h3x3(i);
                to_eigen(hinfo.hessian, H);
                hinfo.point_id = Vector2i{h3x3_index(i), h3x3_index(i)};

                MUDA_KERNEL_ASSERT(h3x3_index(i) >= 0,
                                   "ERROR! Why h3x3_index[%d]: %d, total H3x3=%d",
                                   i,
                                   h3x3_index(i),
                                   h3x3.total_size());

                //printf("h3x3_index[%d]: %d\n", i, h3x3_index(i));
                //printf("h3x3[%d]:%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n",
                //       i,
                //       H.m[0][0],
                //       H.m[0][1],
                //       H.m[0][2],
                //       H.m[1][0],
                //       H.m[1][1],
                //       H.m[1][2],
                //       H.m[2][0],
                //       H.m[2][1],
                //       H.m[2][2]);

                if(hinfo.point_id.x() < 0 || hinfo.point_id.y() < 0)
                {
                    printf("ERROR: h3x3_index[%d]: %d\n", i, h3x3_index(i));
                }
            });

        offset += count;
        auto h6x6       = H6x6();
        auto h6x6_index = H6x6_index();
        count           = 4 * h6x6.size();

        ParallelFor().apply(
            h6x6.size(),
            [hessian_info = hessians.subview(offset, count).viewer().name("hessian_info"),
             h6x6 = h6x6.viewer().name("h6x6"),
             h6x6_index = h6x6_index.viewer().name("h6x6_index")] __device__(int I) mutable
            {
                auto&           H = h6x6(I);
                gipc::Matrix6x6 h;
                to_eigen(h, H);
                int index[2] = {h6x6_index(I).x, h6x6_index(I).y};

                for(int i = 0; i < 2; ++i)
                {
                    MUDA_KERNEL_ASSERT(index[i] >= 0,
                                       "ERROR! Why H6x6[%d], index[%d,%d], total H6x6=%d",
                                       I,
                                       h6x6_index(I).x,
                                       h6x6_index(I).y,
                                       h6x6.total_size());
                }

                auto offset = 4 * I;
                for(int i = 0; i < 2; ++i)
                {
                    for(int j = 0; j < 2; ++j)
                    {
                        auto& hinfo    = hessian_info(offset++);
                        hinfo.hessian  = h.block<3, 3>(3 * i, 3 * j);
                        hinfo.point_id = Vector2i{index[i], index[j]};
                    }
                }
            });

        offset += count;
        auto h9x9       = H9x9();
        auto h9x9_index = H9x9_index();
        count           = 9 * h9x9.size();

        ParallelFor().apply(
            h9x9.size(),
            [hessian_info = hessians.subview(offset, count).viewer().name("hessian_info"),
             h9x9 = h9x9.viewer().name("h9x9"),
             h9x9_index = h9x9_index.viewer().name("h9x9_index")] __device__(int I) mutable
            {
                auto&           H = h9x9(I);
                gipc::Matrix9x9 h;
                to_eigen(h, H);
                int index[3] = {h9x9_index(I).x, h9x9_index(I).y, h9x9_index(I).z};

                for(int i = 0; i < 3; ++i)
                {
                    MUDA_KERNEL_ASSERT(index[i] >= 0,
                                       "ERROR! Why H9x9[%d], index[%d,%d,%d], total H9x9=%d",
                                       I,
                                       h9x9_index(I).x,
                                       h9x9_index(I).y,
                                       h9x9_index(I).z,
                                       h9x9.total_size());
                }

                auto offset = 9 * I;
                for(int i = 0; i < 3; ++i)
                {
                    for(int j = 0; j < 3; ++j)
                    {
                        auto& hinfo    = hessian_info(offset++);
                        hinfo.hessian  = h.block<3, 3>(3 * i, 3 * j);
                        hinfo.point_id = Vector2i{index[i], index[j]};
                    }
                }
            });

        offset += count;
        auto h12x12       = H12x12();
        auto h12x12_index = H12x12_index();
        count             = 16 * h12x12.size();

        ParallelFor().apply(
            h12x12.size(),
            [hessian_info = hessians.subview(offset, count).viewer().name("hessian_info"),
             h12x12 = h12x12.viewer().name("h12x12"),
             h12x12_index = h12x12_index.viewer().name("h12x12_index")] __device__(int I) mutable
            {
                auto&             H = h12x12(I);
                gipc::Matrix12x12 h;
                to_eigen(h, H);
                int index[4] = {h12x12_index(I).x,
                                h12x12_index(I).y,
                                h12x12_index(I).z,
                                h12x12_index(I).w};

                for(int i = 0; i < 4; ++i)
                {
                    MUDA_KERNEL_ASSERT(index[i] >= 0,
                                       "ERROR! Why H12x12[%d], index[%d,%d,%d,%d] total H12x12=%d",
                                       I,
                                       h12x12_index(I).x,
                                       h12x12_index(I).y,
                                       h12x12_index(I).z,
                                       h12x12_index(I).w,
                                       h12x12.total_size());
                }

                auto offset = 16 * I;
                for(int i = 0; i < 4; ++i)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        auto& hinfo    = hessian_info(offset++);
                        hinfo.hessian  = h.block<3, 3>(3 * i, 3 * j);
                        hinfo.point_id = Vector2i{index[i], index[j]};
                    }
                }
            });
    }
}
}  // namespace gipc
