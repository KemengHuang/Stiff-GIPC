#pragma once
#include <iostream>
namespace gipc
{
struct ABDFEMCountInfo
{
    size_t abd_body_offset = 0;
    size_t abd_body_num    = 0;
    size_t fem_body_offset = 0;
    size_t fem_body_num    = 0;

    size_t abd_tet_offset = 0;
    size_t abd_tet_num    = 0;
    size_t fem_tet_offset = 0;
    size_t fem_tet_num    = 0;

    size_t fem_tri_offset = 0;
    size_t fem_tri_num    = 0;

    size_t abd_point_offset = 0;
    size_t abd_point_num    = 0;
    size_t fem_point_offset = 0;
    size_t fem_point_num    = 0;

    auto total_body_num() const { return abd_body_num + fem_body_num; }
    auto total_point_num() const { return abd_point_num + fem_point_num; }
    auto total_tet_num() const { return abd_tet_num + fem_tet_num; }

    friend std::ostream& operator<<(std::ostream& os, ABDFEMCountInfo& info)
    {
        os << "abd_body_offset: " << info.abd_body_offset << std::endl
           << "abd_body_num:    " << info.abd_body_num << std::endl;
        os << "fem_body_offset: " << info.fem_body_offset << std::endl
           << "fem_body_num:    " << info.fem_body_num << std::endl;

        os << "abd_tet_offset:  " << info.abd_tet_offset << std::endl
           << "abd_tet_num:     " << info.abd_tet_num << std::endl;
        os << "fem_tet_offset:  " << info.fem_tet_offset << std::endl
           << "fem_tet_num:     " << info.fem_tet_num << std::endl;

        os << "fem_tri_offset:  " << info.fem_tri_offset << std::endl
           << "fem_tri_num:     " << info.fem_tri_num << std::endl;

        os << "abd_point_offset:" << info.abd_point_offset << std::endl
           << "abd_point_num:   " << info.abd_point_num << std::endl;
        os << "fem_point_offset:" << info.fem_point_offset << std::endl
           << "fem_point_num:   " << info.fem_point_num << std::endl;
        return os;
    }
};
}  // namespace gipc
