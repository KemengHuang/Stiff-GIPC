#include <contact_system/contact_system.h>
#include <muda/cub/device/device_partition.h>
#include <abd_system/abd_sim_data.h>
#include <GIPC.cuh>
#include <gipc/utils/host_log.h>

namespace gipc
{


ContactSystem::ContactSystem(GIPC& gipc)
    : m_gipc(gipc)
{
}

void ContactSystem::solve()
{
    _assemble();
    _partition();
}
void ContactSystem::_assemble()
{
    IContactReporter::ContactCountInfo info;
    m_contact_reporter->report_contact_count(info);
    m_contact_gradients.resize(info.gradient_count);
    m_contact_hessians.resize(info.hessian_count);
    m_partitioned_contact_hessians.resize(info.hessian_count);
    m_contact_reporter->assemble(m_contact_gradients, m_contact_hessians);
}
void ContactSystem::_partition()
{
    auto abd_fem_count_info = m_gipc.abd_fem_count_info;
    auto abd_point_offset   = abd_fem_count_info.abd_point_offset;
    auto fem_point_offset   = abd_fem_count_info.fem_point_offset;

    // partition contact gradients
    {
        int partition_count = fem_point_offset;

        m_abd_contact_gradients = m_contact_gradients.view(0, partition_count);
        m_fem_contact_gradients = m_contact_gradients.view(partition_count);
    }

    // partition contact hessian
    {
        int offset = 0;
        int count  = m_contact_hessians.size();

        {  // ABD-ABD
            muda::DevicePartition().If(
                m_contact_hessians.view(offset).data(),
                m_partitioned_contact_hessians.view(offset).data(),
                m_partition_count.data(),
                count,
                [=] CUB_RUNTIME_FUNCTION(const ContactHessian& h)
                {
                    auto is_left_abd  = h.point_id.x() < fem_point_offset;
                    auto is_right_abd = h.point_id.y() < fem_point_offset;
                    return is_left_abd && is_right_abd;
                });

            int partition_count = m_partition_count;

            m_abd_contact_hessians =
                m_partitioned_contact_hessians.view(offset, partition_count);

            offset += partition_count;
            count -= partition_count;

            // copy the rest to m_contact_hessians
            m_contact_hessians.view(offset).copy_from(
                m_partitioned_contact_hessians.view(offset));
            if(m_report_info)
                std::cout << "ABD-ABD contact hessian(3x3) count: " << partition_count
                          << std::endl;
        }

        {  // FEM-FEM
            muda::DevicePartition().If(
                m_contact_hessians.view(offset).data(),
                m_partitioned_contact_hessians.view(offset).data(),
                m_partition_count.data(),
                count,
                [=] CUB_RUNTIME_FUNCTION(const ContactHessian& h)
                {
                    auto is_left_fem  = h.point_id.x() >= fem_point_offset;
                    auto is_right_fem = h.point_id.y() >= fem_point_offset;

                    return is_left_fem && is_right_fem;
                });

            int partition_count = m_partition_count;

            m_fem_contact_hessians =
                m_partitioned_contact_hessians.view(offset, partition_count);

            offset += partition_count;
            count -= partition_count;

            // copy the rest to m_contact_hessians
            m_contact_hessians.view(offset).copy_from(
                m_partitioned_contact_hessians.view(offset));

            if(m_report_info)
                std::cout << "FEM-FEM contact hessian(3x3) count: " << partition_count
                          << std::endl;
        }

        {  // ABD-FEM
            muda::DevicePartition().If(
                m_contact_hessians.view(offset).data(),
                m_partitioned_contact_hessians.view(offset).data(),
                m_partition_count.data(),
                count,
                [=] CUB_RUNTIME_FUNCTION(const ContactHessian& h)
                {
                    auto is_left_abd  = h.point_id.x() < fem_point_offset;
                    auto is_right_fem = h.point_id.y() >= fem_point_offset;

                    return is_left_abd && is_right_fem;
                });

            int partition_count = m_partition_count;

            m_abd_fem_contact_hessians =
                m_partitioned_contact_hessians.view(offset, partition_count);

            offset += partition_count;
            count -= partition_count;

            // copy the rest to m_contact_hessians
            m_contact_hessians.view(offset).copy_from(
                m_partitioned_contact_hessians.view(offset));

            if(m_report_info)
                std::cout << "ABD-FEM contact hessian(3x3) count: "
                          << m_abd_fem_contact_hessians.size() << std::endl;
        }

        {  // FEM-ABD
            muda::DevicePartition().If(
                m_contact_hessians.view(offset).data(),
                m_partitioned_contact_hessians.view(offset).data(),
                m_partition_count.data(),
                count,
                [=] CUB_RUNTIME_FUNCTION(const ContactHessian& h)
                {
                    auto is_left_fem  = h.point_id.x() >= fem_point_offset;
                    auto is_right_abd = h.point_id.y() < fem_point_offset;

                    return is_left_fem && is_right_abd;
                });

            int partition_count = m_partition_count;

            m_fem_abd_contact_hessians =
                m_partitioned_contact_hessians.view(offset, partition_count);

            offset += partition_count;
            count -= partition_count;

            // copy the rest to m_contact_hessians
            m_contact_hessians.view(offset).copy_from(
                m_partitioned_contact_hessians.view(offset));

            if(m_report_info)
                std::cout << "FEM-ABD contact hessian(3x3) count: "
                          << m_fem_abd_contact_hessians.size() << std::endl;
        }

        MUDA_ASSERT(count == 0, "Partition failed, rest count = %d", count);
    }
}
}  // namespace gipc
