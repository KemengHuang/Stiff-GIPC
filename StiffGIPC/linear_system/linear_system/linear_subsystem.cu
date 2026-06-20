#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
ILinearSubsystem::~ILinearSubsystem() {}

//void ILinearSubsystem::hessian_block_offset(IndexT hessian_offset)
//{
//    m_hessian_offset = hessian_offset;
//}

Json ILinearSubsystem::as_json() const
{
    Json j;
    j["type"]       = typeid(*this).name();
    j["gid"]        = m_gid;
    j["hid"]        = m_hid;
    j["dof_offset"] = dof_offset();
    return j;
}

void DiagonalSubsystem::right_hand_side_dof(SizeT right_hand_side_dof)
{
    CT_ASSERT(right_hand_side_dof % 3 == 0,
                "In 3D, right_hand_side_dof must be a multiple of 3, yours %d.",
                right_hand_side_dof);
    m_right_hand_side_dof = right_hand_side_dof;
}

//void ILinearSubsystem::hessian_block_count(SizeT hessian_block_count)
//{
//    m_hessian_block_count = hessian_block_count;
//}

Json DiagonalSubsystem::as_json() const
{
    auto j                   = Base::as_json();
    //j["hessian_block_count"] = hessian_block_count();
    j["right_hand_side_dof"] = right_hand_side_dof();
    return j;
}

Vector2i DiagonalSubsystem::dof_offset() const
{
    return Vector2i::Ones() * m_dof_offset;
}

void DiagonalSubsystem::dof_offset(IndexT dof_offset)
{
    CT_ASSERT(dof_offset % 3 == 0, "In 3D, dof_offset must be a multiple of 3, yours %d.", dof_offset);
    m_dof_offset = dof_offset;
}

void DiagonalSubsystem::do_assemble(DenseVectorView gradient)
{
    auto dof_offset = this->dof_offset();

    assemble(gradient.subview(dof_offset[0], right_hand_side_dof()));
}

void DiagonalSubsystem::do_retrieve_solution(CDenseVectorView dx)
{
    retrieve_solution(dx.subview(dof_offset()[0], right_hand_side_dof()));
}

cudatool::LinearSystemContext& ILinearSubsystem::ctx() const
{
    return m_system->m_context;
}

}  // namespace gipc::linear_system
