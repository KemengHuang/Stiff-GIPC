#include "node_edge_model.h"

namespace gipc
{
void NodeEdgeModel::k_way_partition(idx_t nParts, std::vector<idx_t>& part)
{
    idx_t  objval;
    idx_t  nWeight   = 1;
    idx_t  nVertices = m_xadj.size() - 1;
    idx_t* vwgt      = nullptr;
    idx_t* vsize     = nullptr;

    part.resize(nVertices);

    int ret = METIS_PartGraphKway(&nVertices,
                                  &nWeight,
                                  m_xadj.data(),
                                  m_adjncy.data(),
                                  nullptr,  // vwgt
                                  nullptr,  // vsize
                                  m_adj_wgt.data(),
                                  &nParts,
                                  nullptr,  // tpwgts
                                  nullptr,  // ubvec
                                  nullptr,  // options
                                  &objval,  // edgecut
                                  part.data());
}
void NodeEdgeModel::export_partition(const std::string&        filename,
                                     const std::vector<idx_t>& part)
{
    std::ofstream ofs(filename);
    for(int i = 0; i < part.size(); i++)
    {
        ofs << part[i] << std::endl;
    }
}

void NodeEdgeModel::our_partition(idx_t block_size, std::vector<idx_t>& part) {}
}  // namespace gipc
