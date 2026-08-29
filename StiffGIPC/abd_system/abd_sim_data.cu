#include <abd_system/abd_sim_data.h>
#include <core/GIPC.cuh>
#include <fem/device_fem_data.cuh>
#include <cuda_tools/cuda_all.h>
#include "cuda_tools/cuda_tools.h"
#include <numeric>

namespace gipc
{
__global__ void abd_sim_data_upload_kernel(int                          n,
                                           cudatool::CBufferView<TetLocalInfo> tet_infos,
                                           cudatool::CBufferView<uint4>        T2U,
                                           cudatool::BufferView<I32>           P2U)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;

    auto ps  = tet_infos[I].tet_point_ids();
    auto ups = cudatool::eigen::as_eigen(T2U[I]);

    for(int i = 0; i < 4; ++i)
    {
        auto p  = ps(i);
        auto up = ups(i);
        P2U[p]  = up;
    }
}

ABDSimData::ABDSimData(GIPC& gipc, device_TetraData& tet)
    : m_gipc(gipc)
    , m_tet(tet)
{
}

const ABDFEMCountInfo& ABDSimData::abd_fem_count_info() const
{
    return m_gipc.abd_fem_count_info;
}

cudatool::CBufferView<double3> ABDSimData::unique_point_id_to_position() const
{
    auto offset = abd_fem_count_info().abd_point_offset;
    auto num    = abd_fem_count_info().abd_point_num;
    return cudatool::CBufferView<double3>{m_tet.vertexes, m_gipc.vertexNum}.subview(offset, num);
}

cudatool::CBufferView<I32> ABDSimData::unique_point_id_to_body_id() const
{
    auto offset = abd_fem_count_info().abd_point_offset;
    auto num    = abd_fem_count_info().abd_point_num;
    return cudatool::CBufferView<I32>{m_tet.point_id_to_body_id, m_gipc.vertexNum}.subview(offset, num);
}

cudatool::CBufferView<Float> ABDSimData::tet_id_to_volume() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return cudatool::CBufferView<Float>{m_tet.volum, m_gipc.tetrahedraNum}.subview(offset, num);
}
cudatool::CBufferView<I32> ABDSimData::point_id_to_unique_point_id() const
{
    auto offset = abd_fem_count_info().abd_tet_offset * 4;
    auto num    = abd_fem_count_info().abd_tet_num * 4;
    return m_point_id_to_unique_point_id.view(offset, num);
}
cudatool::CBufferView<TetLocalInfo> ABDSimData::tet_info() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return m_tet_info.view(offset, num);
}
cudatool::CBufferView<I32> ABDSimData::tet_id_to_body_id() const
{
    auto offset = abd_fem_count_info().abd_tet_offset;
    auto num    = abd_fem_count_info().abd_tet_num;
    return cudatool::CBufferView<I32>{m_tet.tet_id_to_body_id, m_gipc.tetrahedraNum}.subview(offset, num);
}
cudatool::CBufferView<BodyBoundaryType> ABDSimData::body_id_to_boundary_type() const
{
    auto offset = abd_fem_count_info().abd_body_offset;
    auto num    = abd_fem_count_info().abd_body_num;
    return cudatool::CBufferView<BodyBoundaryType>{m_tet.body_id_to_boundary_type,
                                  m_gipc.abd_fem_count_info.total_body_num()}
        .subview(offset, num);
}

void ABDSimData::upload()
{
    std::vector<TetLocalInfo> tet_info(m_gipc.tetrahedraNum);
    std::iota(tet_info.begin(), tet_info.end(), 0);  // just init with 0, 1, 2, 3, ...

    m_tet_info = tet_info;

    m_point_id_to_unique_point_id.resize(4 * tet_info.size());

    using namespace cudatool;

    auto tets = cudatool::CBufferView<uint4>{m_tet.tetrahedras, m_gipc.tetrahedraNum};

    LaunchCudaKernal_default((int)m_tet_info.size(),
                             256,
                             0,
                             abd_sim_data_upload_kernel,
                             (int)m_tet_info.size(),
                             cudatool::CBufferView<TetLocalInfo>(m_tet_info.data(), m_tet_info.size()),
                             tets,
                             m_point_id_to_unique_point_id.view());
}
}  // namespace gipc
