//
// device_fem_data.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "cuda_tools/cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num,
                                         const int& tetradedra_num,
                                         const int& triangle_num,
                                         const int& softNum,
                                         const int& tri_edgeNum,
                                         const int& bodyNum)
{
    m_vertex_num   = vertex_num;
    int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
    vertexes.resize(vertex_num);
    o_vertexes.resize(vertex_num);
    velocities.resize(vertex_num);
    rest_vertexes.resize(vertex_num);
    temp_double3Mem.resize(vertex_num);
    xTilta.resize(vertex_num);
    fb.resize(vertex_num);
    totalForce.resize(vertex_num);
    shape_grads.resize(vertex_num);

    tetrahedras.resize(tetradedra_num);
    tempTetrahedras.resize(tetradedra_num);

    tri_edges.resize(tri_edgeNum);
    tri_edge_adj_vertex.resize(tri_edgeNum);

#ifdef USE_QUADRATIC_BENDING
    quad_bending_Q.resize(tri_edgeNum);
#endif

    volum.resize(tetradedra_num);
    masses.resize(vertex_num);

    lengthRate.resize(tetradedra_num);
    volumeRate.resize(tetradedra_num);

    apply_gravity.resize(vertex_num);
    tempDouble.resize(maxNumbers);

    BoundaryType.resize(vertex_num);
    BoundaryType.reset_zero();

    DmInverses.resize(tetradedra_num);

    m_soft_num = softNum;
    targetIndex.resize(softNum);
    targetVert.resize(softNum);
    triDmInverses.resize(triangle_num);

    area.resize(triangle_num);
    triangles.resize(triangle_num);

    body_id_to_boundary_type.resize(bodyNum);
    point_id_to_body_id.resize(vertex_num);
    tet_id_to_body_id.resize(tetradedra_num);
}

device_TetraData::~device_TetraData()
{
    FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM()
{
    vertexes.release();
    o_vertexes.release();
    temp_double3Mem.release();
    velocities.release();
    rest_vertexes.release();
    xTilta.release();
    fb.release();
    apply_gravity.release();
    shape_grads.release();
    tetrahedras.release();
    tempTetrahedras.release();
    volum.release();
    masses.release();
    lengthRate.release();
    volumeRate.release();
    DmInverses.release();
    tempDouble.release();
    BoundaryType.release();

    totalForce.release();
    targetIndex.release();
    targetVert.release();
    triDmInverses.release();
    area.release();
    triangles.release();

    tri_edges.release();
    tri_edge_adj_vertex.release();

#ifdef USE_QUADRATIC_BENDING
    quad_bending_Q.release();
#endif

    body_id_to_boundary_type.release();
    point_id_to_body_id.release();
    tet_id_to_body_id.release();
}

void device_TetraData::update_soft_constraint_target_position(int step_id, double ipc_dt)
{
    if(m_soft_num < 1)
        return;

    std::vector<double3> host_vertexes(m_vertex_num);
    CUDA_SAFE_CALL(cudaMemcpy(
        host_vertexes.data(), vertexes, m_vertex_num * sizeof(double3), cudaMemcpyDeviceToHost));

    for(int i = 0; i < m_soft_num; i++)
    {
        if(update_soft_constraint_functor == nullptr)
            host_target_vertices[i] = host_vertexes[host_target_indices[i]];
        else
            host_target_vertices[i] = update_soft_constraint_functor(
                host_vertexes[host_target_indices[i]], step_id, ipc_dt);
    }

    CUDA_SAFE_CALL(cudaMemcpy(targetVert,
                              host_target_vertices.data(),
                              m_soft_num * sizeof(double3),
                              cudaMemcpyHostToDevice));
}
