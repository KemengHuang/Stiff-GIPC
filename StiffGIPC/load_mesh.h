//
// load_mesh.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef FEM_MESH_H
#define FEM_MESH_H
//#include "Eigen/Eigen"
//#include "mIPC.h"
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include "eigen_data.h"
#include <gipc/abd_fem_count_info.h>
#include <gipc/body_type.h>
#include <Eigen/Core>
#include <body_boundary_type.h>


class mesh_obj
{
  public:
    std::vector<double3> vertexes;
    std::vector<double3> normals;
    std::vector<uint3>   facenormals;
    std::vector<uint3>   faces;
    std::vector<uint2>   edges;
    int             vertexNum;
    int             faceNum;
    int             edgeNum;
    //void InitMesh(int type, double scale);
    bool load_mesh(const std::string& filename, double scale, double3 transform);
};

class tetrahedra_obj
{
  public:
    double         maxVolum = 0.0f;
    std::vector<bool>   isNBC;
    std::vector<bool>   isCollide;
    std::vector<double> volum;
    std::vector<double> area;
    std::vector<double> masses;
    std::vector<int>                    apply_gravity;
    double                         meanMass  = 0.0f;
    double                         meanVolum = 0.0f;
    std::vector<double3>                vertexes;
    std::vector<uint32_t>               partId;
    std::vector<std::vector<uint32_t>>       part_block;
    std::vector<int>                    partId_map_real;
    std::vector<int>                    real_map_partId;
    uint32_t                       part_offset = 0;
    std::vector<int>                    boundaryTypies;
    std::vector<uint4>                  tetrahedras;
    std::vector<uint3>                  triangles;
    std::vector<uint32_t>               targetIndex;
    std::vector<double3>                forces;
    std::vector<double3>                velocities;
    std::vector<double3>                d_velocities;
    std::vector<__GEIGEN__::Matrix3x3d> DM_inverse;
    std::vector<__GEIGEN__::Matrix2x2d> tri_DM_inverse;
    std::vector<__GEIGEN__::Matrix3x3d> constraints;

    std::vector<double3>    targetPos;
    std::vector<double3>    tetra_fiberDir;
    std::vector<double>     vert_youngth_modules;
    std::vector<double>     lengthRate;
    std::vector<double>     volumeRate;
    std::vector<uint2> tri_edges_adj_points;
    std::vector<uint2> tri_edges;


    std::vector<uint32_t> surfId2TetId;
    std::vector<uint3>    surface;

    std::vector<uint32_t> surfVerts;
    std::vector<uint2>    surfEdges;

    std::vector<double3> xTilta;
    std::vector<double3> dx_Elastic;
    std::vector<double3> acceleration;
    std::vector<double3> rest_V;
    std::vector<double3> V_prev;


    std::vector<std::vector<unsigned int>> vertNeighbors;
    std::vector<unsigned int>         neighborList;
    std::vector<unsigned int>         neighborStart;
    std::vector<unsigned int>         neighborNum;

    int D12x12Num        = 0;
    int D9x9Num          = 0;
    int D6x6Num          = 0;
    int D3x3Num          = 0;
    int vertexNum        = 0;
    int tetrahedraNum    = 0;
    int triangleNum      = 0;
    int softNum          = 0;
    int vertexOffset     = 0;
    int abd_vertexOffset = 0;
    int abd_tetOffset    = 0;


    double3 minTConer = make_double3(0, 0, 0);
    double3 maxTConer = make_double3(0, 0, 0);

    double3 minConer = make_double3(0, 0, 0);
    double3 maxConer = make_double3(0, 0, 0);

    gipc::ABDFEMCountInfo         abd_fem_count_info{};
    std::vector<BodyBoundaryType> body_id_to_is_fixed;
    std::vector<int>              point_id_to_body_id;
    std::vector<int>              tet_id_to_body_id;

    tetrahedra_obj();
    int getVertNeighbors();
    //void InitMesh(int type, double scale);
    bool load_tetrahedraMesh(const std::string& filename,
                             double             scale,
                             //   double             youngth_module,
                             double3        position_offset,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);

    bool load_parts(const std::string& filename);


    bool load_tetrahedraMesh(const std::string&     filename,
                             const Eigen::Matrix4d& transform,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);

    bool load_tetrahedraMesh(const std::string&     filename,
                             const Eigen::Matrix4d& transform,
                             double                 youngth_module,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);


    bool load_triMesh(const std::string& filename, double scale, double3 transform, int boundaryType);
    bool load_triMesh(const std::string& filename, const Eigen::Matrix4d& transform, int boundaryType);


    bool load_animation(const std::string& filename, double scale, double3 transform);
    bool load_tetrahedraMesh_IPC_TetMesh(const std::string& filename,
                                         double             scale,
                                         double3            position_offset,
                                         bool               isfixed);
    //void load_test(double scale, int num = 1);
    void getSurface();
    bool output_tetrahedraMesh(const std::string& filename);

    bool output_tetrahedraMesh_reorder(const std::string&      filename,
                                       const std::vector<uint32_t>& new_order,
                                       const std::vector<uint32_t>& order_map);

  private:
    bool abd_load_phase         = true;
    int  current_body_id        = 0;
    int  current_body_point_num = 0;
    void begin_load_body(const std::string& filename,
                         gipc::BodyType     body_type,
                         BodyBoundaryType   body_is_fixed);
    void set_body_tet_num(gipc::BodyType body_type, int tet_num);
    void set_body_tri_num(gipc::BodyType body_type, int tri_num);
    void set_body_point_num(gipc::BodyType body_type, int point_num);
};

#endif  // !FEM_MESH.H