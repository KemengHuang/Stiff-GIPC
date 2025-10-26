#pragma once
#include "node_edge_model.h"
#include <Eigen/Core>

namespace gipc
{
class TriMesh : public NodeEdgeModel
{
  public:
    void load(const std::string& filename);

    auto& vertices() { return m_vertices; }
    auto& triangle() { return m_triangle; }

    void export_wireframe(const std::string& filename);

    std::vector<int> sort_index(const std::vector<int>& partition);

    void export_mesh(const std::string& filename);

    TriMesh sorted(const std::vector<int>& sort_index) const;

    void export_sort_index(const std::string& filename, const std::vector<int>& partition);

  private:
    std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector3i> m_triangle;

    void _split(const std::string& str, std::vector<std::string>& v, const std::string& spacer);

    void _load(const std::string& filename);

    void _build_adj();

    //void _build_boundary_vertices();
};




class TetMesh : public NodeEdgeModel
{
  public:
    void load(const std::string& filename);

    auto& vertices() { return m_vertices; }
    auto& tetrahedra() { return m_tetrahedra; }

    void export_wireframe(const std::string& filename);

    std::vector<int> sort_index(const std::vector<int>& partition);

    void export_mesh(const std::string& filename);

    TetMesh sorted(const std::vector<int>& sort_index) const;

    void export_sort_index(const std::string& filename, const std::vector<int>& partition);

  private:
    std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector4i> m_tetrahedra;

    void _split(const std::string& str, std::vector<std::string>& v, const std::string& spacer);

    void _load(const std::string& filename);

    void _build_adj();

    void _build_boundary_vertices();
};
}  // namespace gipc
