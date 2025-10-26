#pragma once
#include <cuda_runtime_api.h>
#include <gipc/type_define.h>
#include <muda/tools/debug_log.h>
namespace gipc
{
class TetLocalInfo
{
    int m_tet_id;

  public:
    MUDA_GENERIC TetLocalInfo()
        : m_tet_id(0)
    {
    }

    MUDA_GENERIC TetLocalInfo(int tet_id)
        : m_tet_id(tet_id)
    {
    }

    MUDA_GENERIC int tet_id() const { return m_tet_id; }
    MUDA_GENERIC int tet_point_offset() const { return m_tet_id * 4; }
    MUDA_GENERIC int tet_edge_offset() const { return m_tet_id * 6; }
    MUDA_GENERIC int tet_triangle_offset() const { return m_tet_id * 4; }

    MUDA_GENERIC Vector3i triangle_points_local_ids(int i) const
    {
        switch(i)
        {
            case 0:
                // 0 , 1, 2
                return Vector3i(0, 1, 2);
            case 1:
                // 0, 1, 3
                return Vector3i(0, 3, 1);  // for triangle normal, we need to reverse the order
            case 2:
                // 0, 2, 3
                return Vector3i(0, 2, 3);
            case 3:
                // 1, 2, 3
                return Vector3i(1, 3, 2);  // for triangle normal, we need to reverse the order
            default:
                MUDA_KERNEL_ERROR("triangle index out of range, max 4, yours=%d", i);
                break;
        }
    }

    MUDA_GENERIC Vector3i triangle_points(int i) const
    {
        return triangle_points_local_ids(i) + tet_point_offset() * Vector3i ::Ones();
    }

    MUDA_GENERIC Vector2i edge_point_local_ids(int i) const
    {
        switch(i)
        {
            case 0:
                return Vector2i(0, 1);
            case 1:
                return Vector2i(0, 2);
            case 2:
                return Vector2i(0, 3);
            case 3:
                return Vector2i(1, 2);
            case 4:
                return Vector2i(1, 3);
            case 5:
                return Vector2i(2, 3);
            default:
                MUDA_KERNEL_ERROR("edge index out of range, max 6, yours=%d", i);
                break;
        }
    }

    MUDA_GENERIC Vector2i edge_points(int i) const
    {
        return edge_point_local_ids(i) + tet_point_offset() * Vector2i ::Ones();
    }

    MUDA_GENERIC Vector3i triangle_edge_local_ids(int i) const
    {
        switch(i)
        {
            case 0:
                return Vector3i(0, 1, 3);
            case 1:
                return Vector3i(0, 2, 4);
            case 2:
                return Vector3i(1, 2, 5);
            case 3:
                return Vector3i(3, 4, 5);
            default:
                MUDA_KERNEL_ERROR("triangle index out of range, max 4, yours=%d", i);
                break;
        }
    }

    MUDA_GENERIC Vector3i triangle_edges(int i) const
    {
        return triangle_edge_local_ids(i) + tet_edge_offset() * Vector3i ::Ones();
    }


    MUDA_GENERIC Vector4i tet_points_local_ids() const
    {
        return Vector4i(0, 1, 2, 3);
    }

    MUDA_GENERIC Vector4i tet_point_ids() const
    {
        return tet_points_local_ids() + tet_point_offset() * Vector4i ::Ones();
    }

    MUDA_GENERIC Vector4i tet_triangle_local_ids() const
    {
        return Vector4i(0, 1, 2, 3);
    }

    MUDA_GENERIC Vector4i tet_triangle_ids() const
    {
        return tet_triangle_local_ids() + tet_triangle_offset() * Vector4i ::Ones();
    }

    MUDA_GENERIC Vector6i tet_edge_local_ids() const
    {
        return Vector6i(0, 1, 2, 3, 4, 5);
    }

    MUDA_GENERIC Vector6i tet_edge_ids() const
    {
        return tet_edge_local_ids() + tet_edge_offset() * Vector6i ::Ones();
    }
};
}  // namespace gipc