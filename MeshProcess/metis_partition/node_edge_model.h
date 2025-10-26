#pragma once
#include <metis.h>
#include <fstream>
#include <vector>

namespace gipc
{
class NodeEdgeModel
{
  public:
    auto& xadj() { return m_xadj; }
    auto& adjncy() { return m_adjncy; }

    void k_way_partition(idx_t nParts, std::vector<idx_t>& part);

    void export_partition(const std::string& filename, const std::vector<idx_t>& part);

    void our_partition(idx_t block_size, std::vector<idx_t>& part);

  protected:
    std::vector<idx_t> m_boundary_nodes;
    std::vector<idx_t> m_xadj;
    std::vector<idx_t> m_adjncy;
    std::vector<idx_t> m_adj_wgt;
};
}  // namespace gipc
