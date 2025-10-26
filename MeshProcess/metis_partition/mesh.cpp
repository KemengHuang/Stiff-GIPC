#include "mesh.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <set>

namespace gipc
{
void TetMesh::load(const std::string& filename)
{
    _load(filename);
    _build_adj();
}

void TriMesh::load(const std::string& filename)
{
    _load(filename);
    _build_adj();
}

void TetMesh::export_wireframe(const std::string& filename)
{
    // export as obj wireframe
    std::ofstream ofs(filename);

    for(auto& v : m_vertices)
    {
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
    }

    for(int i = 0; i < m_xadj.size() - 1; ++i)
    {
        auto offset = m_xadj[i];
        auto end    = m_xadj[i + 1];

        for(int j = offset; j < end; ++j)
        {
            ofs << "l " << i + 1 << " " << m_adjncy[j] + 1 << std::endl;
        }
    }
}

void TriMesh::export_wireframe(const std::string& filename)
{
    // export as obj wireframe
    std::ofstream ofs(filename);

    for(auto& v : m_vertices)
    {
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
    }

    for(int i = 0; i < m_xadj.size() - 1; ++i)
    {
        auto offset = m_xadj[i];
        auto end    = m_xadj[i + 1];

        for(int j = offset; j < end; ++j)
        {
            ofs << "l " << i + 1 << " " << m_adjncy[j] + 1 << std::endl;
        }
    }
}

std::vector<int> TetMesh::sort_index(const std::vector<int>& partition)
{
    std::vector<std::pair<idx_t, idx_t>> sorted_part(partition.size());

    std::transform(partition.begin(),
                   partition.end(),
                   sorted_part.begin(),
                   [i = 0](idx_t p) mutable -> std::pair<idx_t, idx_t> {
                       return {i++, p};
                   });

    std::sort(sorted_part.begin(),
              sorted_part.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> sort_index(partition.size());

    std::transform(sorted_part.begin(),
                   sorted_part.end(),
                   sort_index.begin(),
                   [](const auto& p) { return p.first; });

    return sort_index;
}

std::vector<int> TriMesh::sort_index(const std::vector<int>& partition)
{
    std::vector<std::pair<idx_t, idx_t>> sorted_part(partition.size());

    std::transform(partition.begin(),
                   partition.end(),
                   sorted_part.begin(),
                   [i = 0](idx_t p) mutable -> std::pair<idx_t, idx_t> {
                       return {i++, p};
                   });

    std::sort(sorted_part.begin(),
              sorted_part.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> sort_index(partition.size());

    std::transform(sorted_part.begin(),
                   sorted_part.end(),
                   sort_index.begin(),
                   [](const auto& p) { return p.first; });

    return sort_index;
}

void TetMesh::export_mesh(const std::string& filename)
{
    std::ofstream ofs(filename);
    auto          header =
        R"($MeshFormat
2.2 0 8
$EndMeshFormat)";
    ofs << header << std::endl;
    ofs << "$Nodes" << std::endl;
    ofs << vertices().size() << std::endl;
    for(int i = 0; i < vertices().size(); i++)
    {
        ofs << i + 1 << " " << m_vertices[i].x() << " " << m_vertices[i].y()
            << " " << m_vertices[i].z() << std::endl;
    }

    ofs << "$Elements" << std::endl;
    ofs << m_tetrahedra.size() << std::endl;
    for(int i = 0; i < m_tetrahedra.size(); i++)
    {
        ofs << i + 1 << " 4 0 " << m_tetrahedra[i].x() + 1 << " "
            << m_tetrahedra[i].y() + 1 << " " << m_tetrahedra[i].z() + 1 << " "
            << m_tetrahedra[i].w() + 1 << std::endl;
    }
}

void TriMesh::export_mesh(const std::string& filename)
{
    std::ofstream ofs(filename);
    for(int i = 0; i < vertices().size(); i++)
    {
        ofs << "v" << " " << m_vertices[i].x() << " " << m_vertices[i].y()
            << " " << m_vertices[i].z() << std::endl;
    }

    for(int i = 0; i < m_triangle.size(); i++)
    {
        ofs << "f " << m_triangle[i].x() + 1 << " "
            << m_triangle[i].y() + 1 << " " << m_triangle[i].z() + 1 << std::endl;
    }
}

TetMesh TetMesh::sorted(const std::vector<int>& sort_index) const
{
    TetMesh sorted_mesh;

    sorted_mesh.m_vertices.resize(m_vertices.size());
    sorted_mesh.m_tetrahedra.resize(m_tetrahedra.size());

    for(int i = 0; i < m_vertices.size(); i++)
    {
        sorted_mesh.m_vertices[i] = m_vertices[sort_index[i]];
    }

    std::vector<int> old_id_to_new_id(m_vertices.size());
    for(int i = 0; i < m_vertices.size(); i++)
    {
        old_id_to_new_id[sort_index[i]] = i;
    }

    for(int i = 0; i < m_tetrahedra.size(); i++)
    {
        auto& new_tet = sorted_mesh.m_tetrahedra[i];
        auto& tet     = m_tetrahedra[i];

        new_tet.x() = old_id_to_new_id[tet.x()];
        new_tet.y() = old_id_to_new_id[tet.y()];
        new_tet.z() = old_id_to_new_id[tet.z()];
        new_tet.w() = old_id_to_new_id[tet.w()];
    }
    sorted_mesh._build_adj();
    return sorted_mesh;
}

TriMesh TriMesh::sorted(const std::vector<int>& sort_index) const
{
    TriMesh sorted_mesh;

    sorted_mesh.m_vertices.resize(m_vertices.size());
    sorted_mesh.m_triangle.resize(m_triangle.size());

    for(int i = 0; i < m_vertices.size(); i++)
    {
        sorted_mesh.m_vertices[i] = m_vertices[sort_index[i]];
    }

    std::vector<int> old_id_to_new_id(m_vertices.size());
    for(int i = 0; i < m_vertices.size(); i++)
    {
        old_id_to_new_id[sort_index[i]] = i;
    }

    for(int i = 0; i < m_triangle.size(); i++)
    {
        auto& new_tri = sorted_mesh.m_triangle[i];
        auto& tri     = m_triangle[i];

        new_tri.x() = old_id_to_new_id[tri.x()];
        new_tri.y() = old_id_to_new_id[tri.y()];
        new_tri.z() = old_id_to_new_id[tri.z()];
       
    }
    sorted_mesh._build_adj();
    return sorted_mesh;
}

void TetMesh::export_sort_index(const std::string& filename, const std::vector<int>& partition)
{
    std::ofstream ofs(filename);
    auto          si = this->sort_index(partition);
    for(int i = 0; i < si.size(); i++)
    {
        ofs << si[i] << std::endl;
    }
}

void TriMesh::export_sort_index(const std::string& filename, const std::vector<int>& partition)
{
    std::ofstream ofs(filename);
    auto          si = this->sort_index(partition);
    for(int i = 0; i < si.size(); i++)
    {
        ofs << si[i] << std::endl;
    }
}

void TetMesh::_split(const std::string& str, std::vector<std::string>& v, const std::string& spacer)
{
    int pos1, pos2;
    int len = spacer.length();
    pos1    = 0;
    pos2    = str.find(spacer);
    while(pos2 != std::string::npos)
    {
        v.push_back(str.substr(pos1, pos2 - pos1));
        pos1 = pos2 + len;
        pos2 = str.find(spacer, pos1);
    }
    if(pos1 != str.length())
        v.push_back(str.substr(pos1));
}

void TriMesh::_split(const std::string& str, std::vector<std::string>& v, const std::string& spacer)
{
    int pos1, pos2;
    int len = spacer.length();
    pos1    = 0;
    pos2    = str.find(spacer);
    while(pos2 != std::string::npos)
    {
        v.push_back(str.substr(pos1, pos2 - pos1));
        pos1 = pos2 + len;
        pos2 = str.find(spacer, pos1);
    }
    if(pos1 != str.length())
        v.push_back(str.substr(pos1));
}


void TetMesh::_load(const std::string& filename)
{
    using namespace std;
    using namespace Eigen;
    int vertexOffset = 0;

    ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
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

            double xmin = 1e32, ymin = 1e32, zmin = 1e32;
            double xmax = -1e32, ymax = -1e32, zmax = -1e32;
            for(int i = 0; i < nodeNumber; i++)
            {
                getline(ifs, line);
                vector<std::string> nodePos;
                std::string         spacer = " ";
                _split(line, nodePos, spacer);
                x = atof(nodePos[1].c_str());
                y = atof(nodePos[2].c_str());
                z = atof(nodePos[3].c_str());
                Vector3d vertex{x, y, z};

                double mass         = 0;
                int    boundaryType = 0;

                m_vertices.push_back(vertex);
            }
        }

        if(line.substr(1, 8) == "Elements")
        {
            getline(ifs, line);
            elementNumber = atoi(line.c_str());

            for(int i = 0; i < elementNumber; i++)
            {
                getline(ifs, line);

                vector<std::string> elementIndexex;
                std::string         spacer = " ";
                _split(line, elementIndexex, spacer);
                index0 = atoi(elementIndexex[3].c_str()) - 1;
                index1 = atoi(elementIndexex[4].c_str()) - 1;
                index2 = atoi(elementIndexex[5].c_str()) - 1;
                index3 = atoi(elementIndexex[6].c_str()) - 1;

                Vector4i tet;
                tet.x() = index0 + vertexOffset;
                tet.y() = index1 + vertexOffset;
                tet.z() = index2 + vertexOffset;
                tet.w() = index3 + vertexOffset;
                m_tetrahedra.push_back(tet);
            }
            break;
        }
    }
    ifs.close();
}

void TriMesh::_load(const std::string& filename)
{
    using namespace std;
    using namespace Eigen;
    int vertexOffset = 0;

   ifstream ifs(filename);
    if(!ifs)
    {

        fprintf(stderr, "unable to read file %s\n", filename.c_str());
        ifs.close();
        exit(-1);
        //return false;
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

            Vector3d vertex{x, y, z};

            m_vertices.push_back(vertex);
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

                Vector3i faceVertIndex;
                Vector3i faceNormalIndex;

                if(sscanf(buffer,
                          "f %d/%d/%d %d/%d/%d %d/%d/%d",
                          &faceVertIndex.x(),
                          &uv0,
                          &faceNormalIndex.x(),
                          &faceVertIndex.y(),
                          &uv1,
                          &faceNormalIndex.y(),
                          &faceVertIndex.z(),
                          &uv2,
                          &faceNormalIndex.z())
                   == 9)
                {

                    faceVertIndex.x() -= (1 - vertexOffset);
                    faceVertIndex.y() -= (1 - vertexOffset);
                    faceVertIndex.z() -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                    //facenormals.push_back(faceNormalIndex);
                }
                else if(sscanf(buffer,
                               "f %d %d %d",
                               &faceVertIndex.x(),
                               &faceVertIndex.y(),
                               &faceVertIndex.z())
                        == 3)
                {
                    faceVertIndex.x() -= (1 - vertexOffset);
                    faceVertIndex.y() -= (1 - vertexOffset);
                    faceVertIndex.z() -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                }
                else if(sscanf(buffer,
                               "f %d/%d %d/%d %d/%d",
                               &faceVertIndex.x(),
                               &uv0,
                               &faceVertIndex.y(),
                               &uv1,
                               &faceVertIndex.z(),
                               &uv2)
                        == 6)
                {
                    faceVertIndex.x() -= (1 - vertexOffset);
                    faceVertIndex.y() -= (1 - vertexOffset);
                    faceVertIndex.z() -= (1 - vertexOffset);
                    //triangles.push_back(faceVertIndex);
                }

                    m_triangle.push_back(faceVertIndex);
                
            }
        }
    }
    ifs.close();
}

void TetMesh::_build_adj()
{
    std::vector<std::map<idx_t, int>> adj(vertices().size());

    auto insert_adj = [&](idx_t i, idx_t j)
    {
        auto iter = adj[i].find(j);
        if(iter == adj[i].end())
        {
            adj[i][j] = 1;
        }
        else
        {
            iter->second += 1;
        }
    };

    m_xadj.resize(vertices().size() + 1);

    for(size_t i = 0; i < m_tetrahedra.size(); i++)
    {
        auto& tet = m_tetrahedra[i];
        insert_adj(tet.x(), tet.y());
        insert_adj(tet.x(), tet.z());
        insert_adj(tet.x(), tet.w());

        insert_adj(tet.y(), tet.x());
        insert_adj(tet.y(), tet.z());
        insert_adj(tet.y(), tet.w());

        insert_adj(tet.z(), tet.x());
        insert_adj(tet.z(), tet.y());
        insert_adj(tet.z(), tet.w());

        insert_adj(tet.w(), tet.x());
        insert_adj(tet.w(), tet.y());
        insert_adj(tet.w(), tet.z());
    }

    std::transform(adj.begin(),
                   adj.end(),
                   m_xadj.begin(),
                   [](const std::map<idx_t, int>& s) mutable -> idx_t
                   { return s.size(); });

    // exclusive scan
    std::exclusive_scan(m_xadj.begin(), m_xadj.end(), m_xadj.begin(), 0);

    m_adjncy.resize(m_xadj.back());
    m_adj_wgt.resize(m_xadj.back());

    for(size_t i = 0; i < vertices().size(); i++)
    {
        auto& adj_i  = adj[i];
        auto  offset = m_xadj[i];

        for(auto& [j, weight] : adj_i)
        {
            m_adjncy[offset]  = j;
            m_adj_wgt[offset] = weight;
            offset++;
        }
    }
}

void TriMesh::_build_adj()
{
    std::vector<std::map<idx_t, int>> adj(vertices().size());

    auto insert_adj = [&](idx_t i, idx_t j)
    {
        auto iter = adj[i].find(j);
        if(iter == adj[i].end())
        {
            adj[i][j] = 1;
        }
        else
        {
            iter->second += 1;
        }
    };

    m_xadj.resize(vertices().size() + 1);

    for(size_t i = 0; i < m_triangle.size(); i++)
    {
        auto& tri = m_triangle[i];
        insert_adj(tri.x(), tri.y());
        insert_adj(tri.x(), tri.z());
        //insert_adj(tri.x(), tri.w());

        insert_adj(tri.y(), tri.x());
        insert_adj(tri.y(), tri.z());
        //insert_adj(tri.y(), tri.w());
        
        insert_adj(tri.z(), tri.x());
        insert_adj(tri.z(), tri.y());
        //insert_adj(tri.z(), tri.w());
    }

    std::transform(adj.begin(),
                   adj.end(),
                   m_xadj.begin(),
                   [](const std::map<idx_t, int>& s) mutable -> idx_t
                   { return s.size(); });

    // exclusive scan
    std::exclusive_scan(m_xadj.begin(), m_xadj.end(), m_xadj.begin(), 0);

    m_adjncy.resize(m_xadj.back());
    m_adj_wgt.resize(m_xadj.back());

    for(size_t i = 0; i < vertices().size(); i++)
    {
        auto& adj_i  = adj[i];
        auto  offset = m_xadj[i];

        for(auto& [j, weight] : adj_i)
        {
            m_adjncy[offset]  = j;
            m_adj_wgt[offset] = weight;
            offset++;
        }
    }
}

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

void TetMesh::_build_boundary_vertices()
{
    uint64_t length        = vertices().size();
    auto     triangle_hash = [&](const Triangle& tri)
    { return length * (length * tri[0] + tri[1]) + tri[2]; };

    std::vector<Eigen::Vector3i> surface;

    std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(
        4 * m_tetrahedra.size(), triangle_hash);

    for(int i = 0; i < m_tetrahedra.size(); i++)
    {

        const auto& triI4   = m_tetrahedra[i];
        uint64_t    triI[4] = {triI4.x(), triI4.y(), triI4.z(), triI4.w()};
        for(int j = 0; j < 4; j++)
        {
            const Triangle& triVInd =
                Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
            if(tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2]))
               != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] =
                    m_tetrahedra.size() + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] =
                    m_tetrahedra.size() + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] =
                    m_tetrahedra.size() + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] =
                    m_tetrahedra.size() + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] =
                    m_tetrahedra.size() + 1;
            }
            else if(tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0]))
                    != tri2Tet.end())
            {
                tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] =
                    m_tetrahedra.size() + 1;
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
        if(tetId < m_tetrahedra.size())
        {
            Eigen::Vector3d vec1 = m_vertices[triVInd[1]] - m_vertices[triVInd[0]];
            Eigen::Vector3d vec2 = m_vertices[triVInd[2]] - m_vertices[triVInd[0]];
            int id3 = 0;

            if(m_tetrahedra[tetId].x() != triVInd[0]
               && m_tetrahedra[tetId].x() != triVInd[1]
               && m_tetrahedra[tetId].x() != triVInd[2])
            {
                id3 = m_tetrahedra[tetId].x();
            }
            else if(m_tetrahedra[tetId].y() != triVInd[0]
                    && m_tetrahedra[tetId].y() != triVInd[1]
                    && m_tetrahedra[tetId].y() != triVInd[2])
            {
                id3 = m_tetrahedra[tetId].y();
            }
            else if(m_tetrahedra[tetId].z() != triVInd[0]
                    && m_tetrahedra[tetId].z() != triVInd[1]
                    && m_tetrahedra[tetId].z() != triVInd[2])
            {
                id3 = m_tetrahedra[tetId].z();
            }
            else if(m_tetrahedra[tetId].w() != triVInd[0]
                    && m_tetrahedra[tetId].w() != triVInd[1]
                    && m_tetrahedra[tetId].w() != triVInd[2])
            {
                id3 = m_tetrahedra[tetId].w();
            }

            surface.push_back(
                Eigen::Vector3i{(int)triVInd[0], (int)triVInd[1], (int)triVInd[2]});
        }
    }

    std::set<idx_t> boundary_vertices;

    for(const auto& tri : surface)
    {
        for(int i = 0; i < 3; i++)
        {
            boundary_vertices.insert(tri[i]);
        }
    }

    m_boundary_nodes.resize(boundary_vertices.size());

    std::copy(boundary_vertices.begin(),
              boundary_vertices.end(),
              m_boundary_nodes.begin());
}
}  // namespace gipc