//
// 
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include <iostream>
#include "load_mesh.h"
#include <unordered_map>
#include <fstream>
#include <set>
#include <queue>
#include <map>
#include <iostream>
#include <cfloat>
#include "gpu_eigen_libs.cuh"
#include "cstring"
#include "Eigen/Eigen"
using namespace std;

class Triangle
{

  public:
    uint64_t key[3];

    Triangle(const uint64_t* p_key)
    {
        key[0] = p_key[0];
        key[1] = p_key[1];
        key[2] = p_key[2];
    }
    Triangle(uint64_t key0, uint64_t key1, uint64_t key2)
    {
        key[0] = key0;
        key[1] = key1;
        key[2] = key2;
    }

    uint64_t operator[](int i) const
    {
        //assert(0 <= i && i <= 2);
        return key[i];
    }

    bool operator<(const Triangle& right) const
    {
        if(key[0] < right.key[0])
        {
            return true;
        }
        else if(key[0] == right.key[0])
        {
            if(key[1] < right.key[1])
            {
                return true;
            }
            else if(key[1] == right.key[1])
            {
                if(key[2] < right.key[2])
                {
                    return true;
                }
            }
        }
        return false;
    }

    bool operator==(const Triangle& right) const
    {
        return key[0] == right[0] && key[1] == right[1] && key[2] == right[2];
    }
};

void split(string str, vector<string>& v, string spacer)
{
    int pos1, pos2;
    int len = spacer.length();
    pos1    = 0;
    pos2    = str.find(spacer);
    while(pos2 != string::npos)
    {
        v.push_back(str.substr(pos1, pos2 - pos1));
        pos1 = pos2 + len;
        pos2 = str.find(spacer, pos1);
    }
    if(pos1 != str.length())
        v.push_back(str.substr(pos1));
}

bool tetrahedra_obj::load_triMesh(const std::string& filename, double scale, double3 translate, int mboundaryType)
{
    Eigen::Matrix4d transform   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(translate.x, translate.y, translate.z);
    return load_triMesh(filename, transform, mboundaryType);
}

bool tetrahedra_obj::load_triMesh(const std::string&     filename,
                                  const Eigen::Matrix4d& transform,
                                  int                    mboundaryType)
{
    BodyBoundaryType boundaryType = BodyBoundaryType::Free;
    if(mboundaryType == 0)
    {
        boundaryType = BodyBoundaryType::Free;
    }
    else if(mboundaryType == 1)
    {
        boundaryType = BodyBoundaryType::Fixed;
    }
    else
    {
        fprintf(stderr, "boundary type %d is not supported\n", mboundaryType);
        std::abort();
    }

    begin_load_body(filename, gipc::BodyType::FEM, boundaryType);

    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        return false;
    }
    char   buffer[1024];
    string line          = "";
    int    nodeNumber    = 0;
    int    elementNumber = 0;
    double x, y, z;

    double xmin = 1e32, ymin = 1e32, zmin = 1e32;
    double xmax = -1e32, ymax = -1e32, zmax = -1e32;

    while(getline(ifs, line))
    {
        string key = line.substr(0, 2);
        if(key.length() <= 1)
            continue;
        stringstream ss(line.substr(2));
        if(key == "v ")
        {
            ss >> x >> y >> z;

            Eigen::Vector4d V = transform * Eigen::Vector4d(x, y, z, 1);

            double3 vertex = make_double3(V(0), V(1), V(2));
            vertexes.push_back(vertex);
            vertexNum++;
            nodeNumber++;
            __GEIGEN__::Matrix3x3d constraint;
            __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);


            if(mboundaryType == 3)
            {
                masses.push_back(1);
                __GEIGEN__::__set_Mat_val(constraint, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }
            else
            {
                masses.push_back(0);
                __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            }

            constraints.push_back(constraint);
            int boundaryType = mboundaryType;

            if(mboundaryType == 2)
            {
                int tid = vertexNum - 1;
                targetIndex.push_back(tid);
            }

            boundaryTypies.push_back(boundaryType);

            double3 velocity = make_double3(0, 0, 0);
            velocities.push_back(velocity);

            if(xmin > vertex.x)
                xmin = vertex.x;
            if(ymin > vertex.y)
                ymin = vertex.y;
            if(zmin > vertex.z)
                zmin = vertex.z;
            if(xmax < vertex.x)
                xmax = vertex.x;
            if(ymax < vertex.y)
                ymax = vertex.y;
            if(zmax < vertex.z)
                zmax = vertex.z;
        }
        else if(key == "vn")
        {
            ss >> x >> y >> z;
            double3 normal = make_double3(x, y, z);
            //normals.push_back(normal);
        }
        else if(key == "f ")
        {
            if(line.length() >= 1024)
            {
                printf("[WARN]: skip line due to exceed max buffer length (1024).\n");
                continue;
            }

            std::vector<string> fs;

            {
                string         buf;
                stringstream   ss(line);
                vector<string> tokens;
                while(ss >> buf)
                    tokens.push_back(buf);

                for(size_t index = 3; index < tokens.size(); index += 1)
                {
                    fs.push_back("f " + tokens[1] + " " + tokens[index - 1]
                                 + " " + tokens[index]);
                    elementNumber++;
                }
            }

            int uv0, uv1, uv2;

            for(const auto& f : fs)
            {
                memset(buffer, 0, sizeof(char) * 1024);
                std::copy(f.begin(), f.end(), buffer);

                uint3 faceVertIndex;
                uint3 faceNormalIndex;

                if(sscanf(buffer,
                          "f %d/%d/%d %d/%d/%d %d/%d/%d",
                          &faceVertIndex.x,
                          &uv0,
                          &faceNormalIndex.x,
                          &faceVertIndex.y,
                          &uv1,
                          &faceNormalIndex.y,
                          &faceVertIndex.z,
                          &uv2,
                          &faceNormalIndex.z)
                   == 9)
                {

                    faceVertIndex.x -= (1 - vertexOffset);
                    faceVertIndex.y -= (1 - vertexOffset);
                    faceVertIndex.z -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                    //facenormals.push_back(faceNormalIndex);
                }
                else if(sscanf(buffer,
                               "f %d %d %d",
                               &faceVertIndex.x,
                               &faceVertIndex.y,
                               &faceVertIndex.z)
                        == 3)
                {
                    faceVertIndex.x -= (1 - vertexOffset);
                    faceVertIndex.y -= (1 - vertexOffset);
                    faceVertIndex.z -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                }
                else if(sscanf(buffer,
                               "f %d/%d %d/%d %d/%d",
                               &faceVertIndex.x,
                               &uv0,
                               &faceVertIndex.y,
                               &uv1,
                               &faceVertIndex.z,
                               &uv2)
                        == 6)
                {
                    faceVertIndex.x -= (1 - vertexOffset);
                    faceVertIndex.y -= (1 - vertexOffset);
                    faceVertIndex.z -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                }
                if(mboundaryType >= 2)
                {
                    surface.push_back(faceVertIndex);
                }
                else
                {
                    triangles.push_back(faceVertIndex);
                }
            }
        }
    }

    triangleNum = triangles.size();
    set_body_tri_num(gipc::BodyType::FEM, elementNumber);
    vertexOffset += vertexNum;
    set_body_point_num(gipc::BodyType::FEM, nodeNumber);

    softNum   = targetIndex.size();
    minTConer = make_double3(xmin, ymin, zmin);
    maxTConer = make_double3(xmax, ymax, zmax);

    double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y)
                       * (maxTConer.z - minTConer.z);
    double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y)
                      * (maxConer.z - minConer.z);

    if(boxTVolum > boxVolum)
    {
        maxConer = maxTConer;
        minConer = minTConer;
    }

    std::set<std::pair<int, int>>                   edge_set;
    std::map<std::pair<int, int>, std::vector<int>> edge_map;
    std::vector<Eigen::Vector2i>                    my_edges;
    for(auto tri : triangles)
    {
        auto x = tri.x;
        auto y = tri.y;
        auto z = tri.z;
        if(x < y)
        {
            edge_set.insert(make_pair(x, y));
            edge_map[make_pair(x, y)].emplace_back(z);
        }
        else
        {
            edge_set.insert(make_pair(y, x));
            edge_map[make_pair(y, x)].emplace_back(z);
        }

        if(y < z)
        {
            edge_set.insert(make_pair(y, z));
            edge_map[make_pair(y, z)].emplace_back(x);
        }
        else
        {
            edge_set.insert(make_pair(z, y));
            edge_map[make_pair(z, y)].emplace_back(x);
        }

        if(x < z)
        {
            edge_set.insert(make_pair(x, z));
            edge_map[make_pair(x, z)].emplace_back(y);
        }
        else
        {
            edge_set.insert(make_pair(z, x));
            edge_map[make_pair(z, x)].emplace_back(y);
        }
    }

    std::vector<std::vector<int>> temp_edges_adj_points;
    for(auto p : edge_set)
    {
        if(edge_map[p].size() != 2)
            continue;
        my_edges.emplace_back(p.first, p.second);
        temp_edges_adj_points.emplace_back(edge_map[make_pair(p.first, p.second)]);
    }
    for(int i = 0; i < temp_edges_adj_points.size(); i++)
    {
        tri_edges.emplace_back(make_uint2(my_edges[i].x(), my_edges[i].y()));
        if(temp_edges_adj_points[i].size() == 2)
            tri_edges_adj_points.emplace_back(
                make_uint2(temp_edges_adj_points[i][0], temp_edges_adj_points[i][1]));
        else
            tri_edges_adj_points.emplace_back(make_uint2(temp_edges_adj_points[i][0], -1));
    }
    cout << "edges_adj_points.size" << endl;
    cout << temp_edges_adj_points.size() << endl;


    return true;
}


bool tetrahedra_obj::load_animation(const std::string& filename, double scale, double3 transform)
{
    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        return false;
    }
    char   buffer[1024];
    string line = "";
    double x, y, z;
    targetPos.clear();
    while(getline(ifs, line))
    {
        string key = line.substr(0, 2);
        if(key.length() <= 1)
            continue;
        stringstream ss(line.substr(2));
        //printf(line.c_str());
        //printf("\n");
        if(key == "v ")
        {

            ss >> x >> y >> z;
            double3 vertex = make_double3(scale * x + transform.x,
                                          scale * y + transform.y,
                                          scale * z + transform.z);
            targetPos.push_back(vertex);
        }
    }
    return true;
}

tetrahedra_obj::tetrahedra_obj()
{
    softNum       = 0;
    vertexOffset  = 0;
    vertexNum     = 0;
    tetrahedraNum = 0;
    triangleNum   = 0;
    minConer      = make_double3(0, 0, 0);
    maxConer      = make_double3(0, 0, 0);
}

bool tetrahedra_obj::load_tetrahedraMesh(const std::string& filename,
                                         double             scale,
                                       /*  double             youngth_module,*/
                                         double3            position_offset,
                                         gipc::BodyType     body_type,
                                         BodyBoundaryType   body_boundary_type)
{
    Eigen::Matrix4d transform   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);
    return load_tetrahedraMesh(filename, transform, body_type, body_boundary_type);
}

bool tetrahedra_obj::load_parts(const std::string& filename)
{
    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read part info file %s\n", filename.c_str());
        ifs.close();
        //exit(-1);
        return false;
    }

    uint32_t id;
    while(ifs >> id)
    {
        partId.push_back(id + part_offset);
    }

    part_offset = partId[partId.size() - 1] + 1;

    ifs.close();

    if(partId.size() != vertexes.size()-abd_vertexOffset)
    {
        partId.clear();
        fprintf(stderr,
                "part info file %s lost the data, it is illeagle\n",
                filename.c_str());
        return false;
    }
    return true;
}


bool tetrahedra_obj::load_tetrahedraMesh(const std::string&     filename,
                                         const Eigen::Matrix4d& transform,
                                         double                 youngth_module,
                                         gipc::BodyType         body_type,
                                         BodyBoundaryType body_boundary_type)
{
    begin_load_body(filename, body_type, body_boundary_type);

    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        return false;
    }

    double x, y, z;
    int    index0, index1, index2, index3;
    string line          = "";
    int    nodeNumber    = 0;
    int    elementNumber = 0;
    while(getline(ifs, line))
    {
        if(line.length() <= 1)
            continue;
        if(line.substr(1, 5) == "Nodes")
        {
            getline(ifs, line);
            nodeNumber = atoi(line.c_str());
            vertexNum += nodeNumber;

            set_body_point_num(body_type, nodeNumber);

            if(body_type == gipc::BodyType::ABD)
            {
                abd_vertexOffset += nodeNumber;
            }

            double xmin = 1e32, ymin = 1e32, zmin = 1e32;
            double xmax = -1e32, ymax = -1e32, zmax = -1e32;
            for(int i = 0; i < nodeNumber; i++)
            {
                getline(ifs, line);
                vector<std::string> nodePos;
                std::string         spacer = " ";
                split(line, nodePos, spacer);
                x                          = atof(nodePos[1].c_str());
                y                          = atof(nodePos[2].c_str());
                z                          = atof(nodePos[3].c_str());
                double3         d_velocity = make_double3(0, 0, 0);
                Eigen::Vector4d V = transform * Eigen::Vector4d(x, y, z, 1);
                double3         vertex = make_double3(V(0), V(1), V(2));
                //Matrix3d Constraint; Constraint.setIdentity();
                //Vector3d force = Vector3d(0, 0, 0);
                double3 velocity     = make_double3(0, 0, 0);
                double3 d_pos        = make_double3(0, 0, 0);
                double  mass         = 0;
                int     boundaryType = 0;
                boundaryTypies.push_back(boundaryType);
                vertexes.push_back(vertex);
                //forces.push_back(force);
                velocities.push_back(velocity);
                
                //Constraints.push_back(Constraint);
                //isNBC.push_back(false);
                //d_velocities.push_back(d_velocity);
                masses.push_back(mass);
                //isDelete.push_back(false);
                //d_positions.push_back(d_pos);
                //externalForce.push_back(Vector3d(0, 0, 0));

                __GEIGEN__::Matrix3x3d constraint;
                __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                constraints.push_back(constraint);

                if(xmin > vertex.x)
                    xmin = vertex.x;
                if(ymin > vertex.y)
                    ymin = vertex.y;
                if(zmin > vertex.z)
                    zmin = vertex.z;
                if(xmax < vertex.x)
                    xmax = vertex.x;
                if(ymax < vertex.y)
                    ymax = vertex.y;
                if(zmax < vertex.z)
                    zmax = vertex.z;
            }
            minTConer = make_double3(xmin, ymin, zmin);
            maxTConer = make_double3(xmax, ymax, zmax);
        }

        if(line.substr(1, 8) == "Elements")
        {
            getline(ifs, line);
            elementNumber = atoi(line.c_str());
            tetrahedraNum += elementNumber;

            if(body_type == gipc::BodyType::ABD)
            {
                abd_tetOffset += elementNumber;
            }

            set_body_tet_num(body_type, elementNumber);

            for(int i = 0; i < elementNumber; i++)
            {
                getline(ifs, line);

                vector<std::string> elementIndexex;
                std::string         spacer = " ";
                split(line, elementIndexex, spacer);
                int elesize = elementIndexex.size();
                index0      = atoi(elementIndexex[elesize - 4].c_str()) - 1;
                index1      = atoi(elementIndexex[elesize - 3].c_str()) - 1;
                index2      = atoi(elementIndexex[elesize - 2].c_str()) - 1;
                index3      = atoi(elementIndexex[elesize - 1].c_str()) - 1;

                uint4 tetrahedra;
                tetrahedra.x = index0 + vertexOffset;
                tetrahedra.y = index1 + vertexOffset;
                tetrahedra.z = index2 + vertexOffset;
                tetrahedra.w = index3 + vertexOffset;
                tetrahedras.push_back(tetrahedra);
                tetra_fiberDir.push_back(make_double3(0, 0, 0));
                vert_youngth_modules.push_back(youngth_module);
            }
            break;
        }
    }
    ifs.close();

    double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y)
                       * (maxTConer.z - minTConer.z);
    double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y)
                      * (maxConer.z - minConer.z);

    if(boxTVolum > boxVolum)
    {
        maxConer = maxTConer;
        minConer = minTConer;
    }
    //V_prev = vertexes;
    vertexOffset = vertexNum;
    D12x12Num    = 0;
    D9x9Num      = 0;
    D6x6Num      = 0;
    D3x3Num      = 0;

    return true;
}

bool tetrahedra_obj::load_tetrahedraMesh(const std::string&     filename,
                                         const Eigen::Matrix4d& transform,
                                         gipc::BodyType         body_type,
                                         BodyBoundaryType body_boundary_type)
{
    begin_load_body(filename, body_type, body_boundary_type);

    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        return false;
    }

    double x, y, z;
    int    index0, index1, index2, index3;
    string line          = "";
    int    nodeNumber    = 0;
    int    elementNumber = 0;
    while(getline(ifs, line))
    {
        if(line.length() <= 1)
            continue;
        if(line.substr(1, 5) == "Nodes")
        {
            getline(ifs, line);
            nodeNumber = atoi(line.c_str());
            vertexNum += nodeNumber;

            set_body_point_num(body_type, nodeNumber);

            if(body_type == gipc::BodyType::ABD)
            {
                abd_vertexOffset += nodeNumber;
            }

            double xmin = 1e32, ymin = 1e32, zmin = 1e32;
            double xmax = -1e32, ymax = -1e32, zmax = -1e32;
            for(int i = 0; i < nodeNumber; i++)
            {
                getline(ifs, line);
                vector<std::string> nodePos;
                std::string         spacer = " ";
                split(line, nodePos, spacer);
                x                          = atof(nodePos[1].c_str());
                y                          = atof(nodePos[2].c_str());
                z                          = atof(nodePos[3].c_str());
                double3         d_velocity = make_double3(0, 0, 0);
                Eigen::Vector4d V = transform * Eigen::Vector4d(x, y, z, 1);
                double3         vertex = make_double3(V(0), V(1), V(2));
                //Matrix3d Constraint; Constraint.setIdentity();
                //Vector3d force = Vector3d(0, 0, 0);
                double3 velocity     = make_double3(0, 0, 0);
                double3 d_pos        = make_double3(0, 0, 0);
                double  mass         = 0;
                int     boundaryType = 0;
                boundaryTypies.push_back(boundaryType);
                vertexes.push_back(vertex);
                //forces.push_back(force);
                velocities.push_back(velocity);
                //Constraints.push_back(Constraint);
                //isNBC.push_back(false);
                //d_velocities.push_back(d_velocity);
                masses.push_back(mass);
                //isDelete.push_back(false);
                //d_positions.push_back(d_pos);
                //externalForce.push_back(Vector3d(0, 0, 0));

                __GEIGEN__::Matrix3x3d constraint;
                __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                constraints.push_back(constraint);

                if(xmin > vertex.x)
                    xmin = vertex.x;
                if(ymin > vertex.y)
                    ymin = vertex.y;
                if(zmin > vertex.z)
                    zmin = vertex.z;
                if(xmax < vertex.x)
                    xmax = vertex.x;
                if(ymax < vertex.y)
                    ymax = vertex.y;
                if(zmax < vertex.z)
                    zmax = vertex.z;
            }
            minTConer = make_double3(xmin, ymin, zmin);
            maxTConer = make_double3(xmax, ymax, zmax);
        }

        if(line.substr(1, 8) == "Elements")
        {
            getline(ifs, line);
            elementNumber = atoi(line.c_str());
            tetrahedraNum += elementNumber;

            if(body_type == gipc::BodyType::ABD)
            {
                abd_tetOffset += elementNumber;
            }

            set_body_tet_num(body_type, elementNumber);

            for(int i = 0; i < elementNumber; i++)
            {
                getline(ifs, line);

                vector<std::string> elementIndexex;
                std::string         spacer = " ";
                split(line, elementIndexex, spacer);
                int elesize = elementIndexex.size();
                index0      = atoi(elementIndexex[elesize - 4].c_str()) - 1;
                index1      = atoi(elementIndexex[elesize - 3].c_str()) - 1;
                index2      = atoi(elementIndexex[elesize - 2].c_str()) - 1;
                index3      = atoi(elementIndexex[elesize - 1].c_str()) - 1;

                uint4 tetrahedra;
                tetrahedra.x = index0 + vertexOffset;
                tetrahedra.y = index1 + vertexOffset;
                tetrahedra.z = index2 + vertexOffset;
                tetrahedra.w = index3 + vertexOffset;
                tetrahedras.push_back(tetrahedra);
                tetra_fiberDir.push_back(make_double3(0, 0, 0));
            }
            break;
        }
    }
    ifs.close();

    double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y)
                       * (maxTConer.z - minTConer.z);
    double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y)
                      * (maxConer.z - minConer.z);

    if(boxTVolum > boxVolum)
    {
        maxConer = maxTConer;
        minConer = minTConer;
    }
    //V_prev = vertexes;
    vertexOffset = vertexNum;
    D12x12Num    = 0;
    D9x9Num      = 0;
    D6x6Num      = 0;
    D3x3Num      = 0;

    return true;
}


bool tetrahedra_obj::load_tetrahedraMesh_IPC_TetMesh(const std::string& filename,
                                                     double  scale,
                                                     double3 position_offset,
                                                     bool    isfixed)
{

    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        return false;
    }

    double x, y, z;
    int    index0, index1, index2, index3;
    string line          = "";
    int    nodeNumber    = 0;
    int    elementNumber = 0;
    while(getline(ifs, line))
    {
        if(line.length() <= 1)
            continue;
        if(line == "$Nodes")
        {
            getline(ifs, line);
            nodeNumber = atoi(line.c_str());
            vertexNum += nodeNumber;

            double xmin = 1e32, ymin = 1e32, zmin = 1e32;
            double xmax = -1e32, ymax = -1e32, zmax = -1e32;
            for(int i = 0; i < nodeNumber; i++)
            {
                getline(ifs, line);
                vector<std::string> nodePos;
                std::string         spacer = " ";
                split(line, nodePos, spacer);
                x                  = atof(nodePos[0].c_str());
                y                  = atof(nodePos[1].c_str());
                z                  = atof(nodePos[2].c_str());
                double3 d_velocity = make_double3(0, 0, 0);
                double3 vertex     = make_double3(scale * x - position_offset.x,
                                              scale * y - position_offset.y,
                                              scale * z - position_offset.z);
                //Matrix3d Constraint; Constraint.setIdentity();
                //Vector3d force = Vector3d(0, 0, 0);
                double3 velocity     = make_double3(0, 0, 0);
                double3 d_pos        = make_double3(0, 0, 0);
                double  mass         = 0;
                int     boundaryType = 0;
                boundaryTypies.push_back(boundaryType);
                vertexes.push_back(vertex);
                //forces.push_back(force);
                velocities.push_back(velocity);
                //Constraints.push_back(Constraint);
                //isNBC.push_back(false);
                d_velocities.push_back(d_velocity);
                masses.push_back(mass);
                //isDelete.push_back(false);
                //d_positions.push_back(d_pos);
                //externalForce.push_back(Vector3d(0, 0, 0));


                __GEIGEN__::Matrix3x3d constraint;
                __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                if(isfixed)
                {
                    __GEIGEN__::__set_Mat_val(constraint, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                }
                constraints.push_back(constraint);

                if(xmin > vertex.x)
                    xmin = vertex.x;
                if(ymin > vertex.y)
                    ymin = vertex.y;
                if(zmin > vertex.z)
                    zmin = vertex.z;
                if(xmax < vertex.x)
                    xmax = vertex.x;
                if(ymax < vertex.y)
                    ymax = vertex.y;
                if(zmax < vertex.z)
                    zmax = vertex.z;
            }
            minTConer = make_double3(xmin, ymin, zmin);
            maxTConer = make_double3(xmax, ymax, zmax);
        }

        if(line == "$Elements")
        {
            getline(ifs, line);
            elementNumber = atoi(line.c_str());
            tetrahedraNum += elementNumber;
            for(int i = 0; i < elementNumber; i++)
            {
                getline(ifs, line);

                vector<std::string> elementIndexex;
                std::string         spacer = " ";
                split(line, elementIndexex, spacer);
                int elesize = elementIndexex.size();
                index0      = atoi(elementIndexex[elesize - 4].c_str()) - 1;
                index1      = atoi(elementIndexex[elesize - 3].c_str()) - 1;
                index2      = atoi(elementIndexex[elesize - 2].c_str()) - 1;
                index3      = atoi(elementIndexex[elesize - 1].c_str()) - 1;

                uint4 tetrahedra;
                tetrahedra.x = index0 + vertexOffset;
                tetrahedra.y = index1 + vertexOffset;
                tetrahedra.z = index2 + vertexOffset;
                tetrahedra.w = index3 + vertexOffset;
                tetrahedras.push_back(tetrahedra);
                tetra_fiberDir.push_back(make_double3(0, 0, 0));
            }
            break;
        }
    }
    ifs.close();

    double boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y)
                       * (maxTConer.z - minTConer.z);
    double boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y)
                      * (maxConer.z - minConer.z);

    if(boxTVolum > boxVolum)
    {
        maxConer = maxTConer;
        minConer = minTConer;
    }
    //V_prev = vertexes;
    vertexOffset = vertexNum;
    D12x12Num    = 0;
    D9x9Num      = 0;
    D6x6Num      = 0;
    D3x3Num      = 0;


    return true;
}


int tetrahedra_obj::getVertNeighbors()
{
    int fem_vertexNum = vertexNum - abd_vertexOffset;
    if(fem_vertexNum <= 0)
        return 0;
    vertNeighbors.resize(fem_vertexNum);
    set<pair<uint64_t, uint64_t>> Edges_set;
    for(int i = 0; i < tetrahedraNum-abd_tetOffset; i++)
    {
        const auto& edgI4   = tetrahedras[i + abd_tetOffset];
        uint64_t    edgI[4] = {edgI4.x, edgI4.y, edgI4.z, edgI4.w};
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[0], edgI[1]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[1], edgI[0]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[0], edgI[1]));
        }
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[0], edgI[2]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[2], edgI[0]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[0], edgI[2]));
        }
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[0], edgI[3]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[3], edgI[0]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[0], edgI[3]));
        }
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[1], edgI[2]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[2], edgI[1]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[1], edgI[2]));
        }
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[1], edgI[3]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[3], edgI[1]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[1], edgI[3]));
        }
        if(Edges_set.find(pair<uint64_t, uint64_t>(edgI[2], edgI[3]))
               == Edges_set.end()
           && Edges_set.find(pair<uint64_t, uint64_t>(edgI[3], edgI[2]))
                  == Edges_set.end())
        {
            Edges_set.insert(pair<uint64_t, uint64_t>(edgI[2], edgI[3]));
        }
    }

    for(const auto& cTri : triangles)
    {
        for(int i = 0; i < 3; i++)
        {
            if(Edges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.x))
                   == Edges_set.end()
               && Edges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.y))
                      == Edges_set.end())
            {
                Edges_set.insert(pair<uint32_t, uint32_t>(cTri.x, cTri.y));
            }
            if(Edges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.y))
                   == Edges_set.end()
               && Edges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.z))
                      == Edges_set.end())
            {
                Edges_set.insert(pair<uint32_t, uint32_t>(cTri.y, cTri.z));
            }
            if(Edges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.z))
                   == Edges_set.end()
               && Edges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.x))
                      == Edges_set.end())
            {
                Edges_set.insert(pair<uint32_t, uint32_t>(cTri.z, cTri.x));
            }
        }
    }


    for(const auto& edgI : Edges_set)
    {
        vertNeighbors[edgI.first - abd_vertexOffset].push_back(edgI.second - abd_vertexOffset);
        vertNeighbors[edgI.second - abd_vertexOffset].push_back(edgI.first - abd_vertexOffset);
    }
    //neighborStart.push_back(0);
    neighborNum.resize(fem_vertexNum);
    int offset = 0;
    for(int i = 0; i < fem_vertexNum; i++)
    {
        for(int j = 0; j < vertNeighbors[i].size(); j++)
        {
            neighborList.push_back(vertNeighbors[i][j]);
        }

        neighborStart.push_back(offset);

        offset += vertNeighbors[i].size();
        neighborNum[i] = vertNeighbors[i].size();
    }

    return neighborStart[fem_vertexNum - 1] + neighborNum[fem_vertexNum - 1];
}


void tetrahedra_obj::getSurface()
{
    uint64_t length        = vertexNum;
    auto     triangle_hash = [&](const Triangle& tri)
    { return length * (length * tri[0] + tri[1]) + tri[2]; };
    //vector<Vector4i> surface;
    std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(
        4 * tetrahedraNum, triangle_hash);
    for(int i = 0; i < tetrahedraNum; i++)
    {

        const auto& triI4   = tetrahedras[i];
        uint64_t    triI[4] = {triI4.x, triI4.y, triI4.z, triI4.w};
        for(int j = 0; j < 4; j++)
        {
            const Triangle& triVInd =
                Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
            if(tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2]))
               != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = tetrahedraNum + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] = tetrahedraNum + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] = tetrahedraNum + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] = tetrahedraNum + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] = tetrahedraNum + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] = tetrahedraNum + 1;
            }
            else
            {
                tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = i;
            }
        }
    }

    for(const auto& triI : tri2Tet)
    {
        const uint64_t& tetId   = triI.second;
        const Triangle& triVInd = triI.first;
        if(tetId < tetrahedraNum)
        {
            double3 vec1 =
                __GEIGEN__::__minus(vertexes[triVInd[1]], vertexes[triVInd[0]]);
            double3 vec2 =
                __GEIGEN__::__minus(vertexes[triVInd[2]], vertexes[triVInd[0]]);
            int id3 = 0;

            if(tetrahedras[tetId].x != triVInd[0] && tetrahedras[tetId].x != triVInd[1]
               && tetrahedras[tetId].x != triVInd[2])
            {
                id3 = tetrahedras[tetId].x;
            }
            else if(tetrahedras[tetId].y != triVInd[0] && tetrahedras[tetId].y != triVInd[1]
                    && tetrahedras[tetId].y != triVInd[2])
            {
                id3 = tetrahedras[tetId].y;
            }
            else if(tetrahedras[tetId].z != triVInd[0] && tetrahedras[tetId].z != triVInd[1]
                    && tetrahedras[tetId].z != triVInd[2])
            {
                id3 = tetrahedras[tetId].z;
            }
            else if(tetrahedras[tetId].w != triVInd[0] && tetrahedras[tetId].w != triVInd[1]
                    && tetrahedras[tetId].w != triVInd[2])
            {
                id3 = tetrahedras[tetId].w;
            }


            double3 vec3 = __GEIGEN__::__minus(vertexes[id3], vertexes[triVInd[0]]);
            double3 n = __GEIGEN__::__v_vec_cross(vec1, vec2);
            if(__GEIGEN__::__v_vec_dot(n, vec3) > 0)
            {
                surfId2TetId.push_back(tetId);
                surface.push_back(make_uint3(triVInd[0], triVInd[2], triVInd[1]));
            }
            else
            {
                surfId2TetId.push_back(tetId);
                surface.push_back(make_uint3(triVInd[0], triVInd[1], triVInd[2]));
            }
        }
    }

    for(const auto& tri : triangles)
    {
        surface.push_back(make_uint3(tri.x, tri.y, tri.z));
    }

    vector<bool> flag(vertexNum, false);
    for(const auto& cTri : surface)
    {

        if(!flag[cTri.x])
        {
            surfVerts.push_back(cTri.x);
            flag[cTri.x] = true;
        }
        if(!flag[cTri.y])
        {
            surfVerts.push_back(cTri.y);
            flag[cTri.y] = true;
        }
        if(!flag[cTri.z])
        {
            surfVerts.push_back(cTri.z);
            flag[cTri.z] = true;
        }
    }

    set<pair<uint64_t, uint64_t>> SFEdges_set;
    for(const auto& cTri : surface)
    {
        for(int i = 0; i < 3; i++)
        {
            for(int i = 0; i < 3; i++)
            {
                if(SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.x))
                       == SFEdges_set.end()
                   && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.y))
                          == SFEdges_set.end())
                {
                    SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.x, cTri.y));
                }
                if(SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.y))
                       == SFEdges_set.end()
                   && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.y, cTri.z))
                          == SFEdges_set.end())
                {
                    SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.y, cTri.z));
                }
                if(SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.x, cTri.z))
                       == SFEdges_set.end()
                   && SFEdges_set.find(pair<uint32_t, uint32_t>(cTri.z, cTri.x))
                          == SFEdges_set.end())
                {
                    SFEdges_set.insert(pair<uint32_t, uint32_t>(cTri.z, cTri.x));
                }
            }
        }
    }


    vector<pair<uint64_t, uint64_t>> tempEdge =
        vector<pair<uint64_t, uint64_t>>(SFEdges_set.begin(), SFEdges_set.end());
    for(int i = 0; i < tempEdge.size(); i++)
    {
        surfEdges.push_back(make_uint2(tempEdge[i].first, tempEdge[i].second));
    }
}

bool tetrahedra_obj::output_tetrahedraMesh(const std::string& filename)
{
    std::ofstream outmsh1(filename);
    outmsh1 << "$Nodes\n";
    outmsh1 << vertexNum << endl;
    for(int i = 0; i < vertexNum; i++)
    {
        outmsh1 << i + 1 << " " << vertexes[i].x << " " << vertexes[i].y << " "
                << vertexes[i].z << endl;
    }
    outmsh1 << "$Elements\n";
    outmsh1 << tetrahedraNum << endl;
    for(int i = 0; i < tetrahedraNum; i++)
    {
        outmsh1 << i + 1 << " 4 0 " << tetrahedras[i].x + 1 << " "
                << tetrahedras[i].y + 1 << " " << tetrahedras[i].z + 1 << " "
                << tetrahedras[i].w + 1 << endl;
    }
    outmsh1.close();
    return true;
}

bool tetrahedra_obj::output_tetrahedraMesh_reorder(const std::string& filename,
                                                   const vector<uint32_t>& new_order,
                                                   const vector<uint32_t>& order_map)
{
    std::ofstream outmsh1(filename);
    outmsh1 << "$Nodes\n";
    outmsh1 << vertexNum << endl;
    for(int ii = 0; ii < vertexNum; ii++)
    {
        int i = new_order[ii];
        outmsh1 << i + 1 << " " << vertexes[i].x << " " << vertexes[i].y << " "
                << vertexes[i].z << endl;
    }
    outmsh1 << "$Elements\n";
    outmsh1 << tetrahedraNum << endl;
    for(int i = 0; i < tetrahedraNum; i++)
    {
        outmsh1 << i + 1 << " 4 0 " << order_map[tetrahedras[i].x] + 1 << " "
                << order_map[tetrahedras[i].y] + 1 << " "
                << order_map[tetrahedras[i].z] + 1 << " "
                << order_map[tetrahedras[i].w] + 1 << endl;
    }
    outmsh1.close();
    return true;
}

void tetrahedra_obj::set_body_tri_num(gipc::BodyType body_type, int tri_num)
{
    // now the FEM triangle offset is always 0, because no ABD triangle is supported
    abd_fem_count_info.fem_tri_offset = 0;

    if(body_type == gipc::BodyType::ABD)
    {
        std::cout << "ABD triangle is not supported" << std::endl;
        std::abort();
    }
    else if(body_type == gipc::BodyType::FEM)
    {
        abd_fem_count_info.fem_tri_num += tri_num;
    }
}

void tetrahedra_obj::set_body_point_num(gipc::BodyType body_type, int point_num)
{
    if(body_type == gipc::BodyType::ABD)
    {
        abd_fem_count_info.abd_point_num += point_num;
        abd_fem_count_info.fem_point_offset = abd_fem_count_info.abd_point_num;
    }
    else if(body_type == gipc::BodyType::FEM)
    {
        abd_fem_count_info.fem_point_num += point_num;
    }
    current_body_point_num = point_num;
    if(point_id_to_body_id.size() != abd_fem_count_info.total_point_num() - point_num)
    {
        std::cerr << "error in point_id_to_body_id" << std::endl;
        std::abort();
    }
    std::fill_n(std::back_inserter(point_id_to_body_id), point_num, current_body_id);
}


void tetrahedra_obj::begin_load_body(const std::string& filename,
                                     gipc::BodyType     body_type,
                                     BodyBoundaryType   body_boundary_type)
{
    current_body_id = abd_fem_count_info.total_body_num();
    
    // check the body order: ABD then FEM
    if(body_type == gipc::BodyType::FEM)
    {
        current_body_id = -1;
        abd_load_phase = false;
        // increase the number of FEM body
        abd_fem_count_info.fem_body_num++;
    }
    else if(body_type == gipc::BodyType::ABD)
    {
        if(!abd_load_phase)
        {
            std::cerr << "ABD mesh shouldn't be loaded after FEM mesh, file: " << filename
                      << std::endl;
            std::abort();
        }
        // increase the number of ABD body
        abd_fem_count_info.abd_body_num++;
        abd_fem_count_info.fem_body_offset = abd_fem_count_info.abd_body_num;
    }

    body_id_to_is_fixed.push_back(body_boundary_type);
}

void tetrahedra_obj::set_body_tet_num(gipc::BodyType body_type, int tet_num)
{
    if(body_type == gipc::BodyType::ABD)
    {
        abd_fem_count_info.abd_tet_num += tet_num;
    }
    else if(body_type == gipc::BodyType::FEM)
    {
        abd_fem_count_info.fem_tet_num += tet_num;
    }

    abd_fem_count_info.fem_tet_offset = abd_fem_count_info.abd_tet_num;

    if(tet_id_to_body_id.size() != abd_fem_count_info.total_tet_num() - tet_num)
    {
        std::cerr << "error in tet_id_to_body_id" << std::endl;
        std::abort();
    }
    std::fill_n(std::back_inserter(tet_id_to_body_id), tet_num, current_body_id);
}