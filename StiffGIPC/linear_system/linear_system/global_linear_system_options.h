#pragma once

namespace gipc
{
enum class SPMVAlgorithm
{
    Triplet = 0,
    BCOO,
    WarpReduceBCOO,
    BSR,
    SymBCOO,
    SymBSR,
    SymWarpReduceBCOO
};
enum class ConvertAlgorithm
{
    MudaBuiltIn = 0,
    NewConverter
};
class GlobalLinearSystemOptions
{
  public:
    SPMVAlgorithm    spmv_algorithm    = SPMVAlgorithm::BSR;
    ConvertAlgorithm convert_algorithm = ConvertAlgorithm::MudaBuiltIn;
};
}  // namespace gipc
