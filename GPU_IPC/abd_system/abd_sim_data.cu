#include <abd_system/abd_sim_data.h>
#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <muda/ext/eigen/as_eigen.h>

namespace gipc
{
ABDSimData::ABDSimData(GIPC& gipc, device_TetraData& tet)
    : m_gipc(gipc)
    , m_tet(tet)
{
}

const ABDFEMCountInfo& ABDSimData::abd_fem_count_info() const
{
    return m_gipc.abd_fem_count_info;
}

muda::CBufferView<double3> ABDSimData::unique_point_id_to_position() const
{
    auto offset = abd_fem_count_info().abd_point_offset;
    auto num    = abd_fem_count_info().abd_point_num;
    return muda::CBufferView<double3>{m_tet.vertexes, m_gipc.vertexNum}.subview(offset, num);
}

muda::CBufferView<I32> ABDSimData::unique_point_id_to_body_id() const
{
    auto offset = abd_fem_count_info().abd_point_offset;
    auto num    = abd_fem_count_info().abd_point_num;
    return muda::CBufferView<I32>{m_tet.point_id_to_body_id, m_gipc.vertexNum}.subview(offset, num);
}

muda::CBufferView<Float> ABDSimData::tet_id_to_volume() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return muda::CBufferView<Float>{m_tet.volum, m_gipc.tetrahedraNum}.subview(offset, num);
}
muda::CBufferView<I32> ABDSimData::point_id_to_unique_point_id() const
{
    auto offset = abd_fem_count_info().abd_tet_offset * 4;
    auto num    = abd_fem_count_info().abd_tet_num * 4;
    return m_point_id_to_unique_point_id.view(offset, num);
}
muda::CBufferView<TetLocalInfo> ABDSimData::tet_info() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return m_tet_info.view(offset, num);
}
muda::CBufferView<I32> ABDSimData::tet_id_to_body_id() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return muda::CBufferView<I32>{m_tet.tet_id_to_body_id, m_gipc.tetrahedraNum}.subview(offset, num);
}
muda::CBufferView<BodyBoundaryType> ABDSimData::body_id_to_boundary_type() const
{
    auto offset = abd_fem_count_info().abd_body_offset;
    auto num    = abd_fem_count_info().abd_body_num;
    return muda::CBufferView<BodyBoundaryType>{m_tet.body_id_to_boundary_type,
                                  m_gipc.abd_fem_count_info.total_body_num()}
        .subview(offset, num);
}

void ABDSimData::upload()
{
    std::vector<TetLocalInfo> tet_info(m_gipc.tetrahedraNum);
    std::iota(tet_info.begin(), tet_info.end(), 0);  // just init with 0, 1, 2, 3, ...

    m_tet_info = tet_info;

    m_point_id_to_unique_point_id.resize(4 * tet_info.size());

    using namespace muda;

    auto tets = muda::CBufferView<uint4>{m_tet.tetrahedras, m_gipc.tetrahedraNum};

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(m_tet_info.size(),
               [tet_infos = m_tet_info.cviewer().name("tet_infos"),
                T2U       = tets.viewer().name("T2U"),
                P2U = m_point_id_to_unique_point_id.viewer().name("P2U")] __device__(int I) mutable
               {
                   auto ps  = tet_infos(I).tet_point_ids();
                   auto ups = eigen::as_eigen(T2U(I));

                   for(int i = 0; i < 4; ++i)
                   {
                       auto p  = ps(i);
                       auto up = ups(i);
                       P2U(p)  = up;
                   }
               });
}
}  // namespace gipc
