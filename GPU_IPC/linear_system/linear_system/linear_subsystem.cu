#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/global_linear_system.h>

namespace gipc
{
ILinearSubsystem::~ILinearSubsystem() {}

void ILinearSubsystem::hessian_block_offset(IndexT hessian_offset)
{
    m_hessian_offset = hessian_offset;
}

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
    MUDA_ASSERT(right_hand_side_dof % 3 == 0,
                "In 3D, right_hand_side_dof must be a multiple of 3, yours %d.",
                right_hand_side_dof);
    m_right_hand_side_dof = right_hand_side_dof;
}

void ILinearSubsystem::hessian_block_count(SizeT hessian_block_count)
{
    m_hessian_block_count = hessian_block_count;
}

Json DiagonalSubsystem::as_json() const
{
    auto j                   = Base::as_json();
    j["hessian_block_count"] = hessian_block_count();
    j["right_hand_side_dof"] = right_hand_side_dof();
    return j;
}

Vector2i DiagonalSubsystem::dof_offset() const
{
    return Vector2i::Ones() * m_dof_offset;
}

void DiagonalSubsystem::dof_offset(IndexT dof_offset)
{
    MUDA_ASSERT(dof_offset % 3 == 0, "In 3D, dof_offset must be a multiple of 3, yours %d.", dof_offset);
    m_dof_offset = dof_offset;
}

void DiagonalSubsystem::do_assemble(TripletMatrixView hessian, DenseVectorView gradient)
{
    auto dof_offset = this->dof_offset();
    auto dof_count  = right_hand_side_dof();
    int2 ij_offset{dof_offset[0] / 3, dof_offset[1] / 3};
    int2 ij_count{dof_count / 3, dof_count / 3};
    auto view =
        hessian.subview(hessian_block_offset(), hessian_block_count()).submatrix(ij_offset, ij_count);

    assemble(view, gradient.subview(dof_offset[0], right_hand_side_dof()));
}

void DiagonalSubsystem::do_retrieve_solution(CDenseVectorView dx)
{
    retrieve_solution(dx.subview(dof_offset()[0], right_hand_side_dof()));
}

muda::LinearSystemContext& ILinearSubsystem::ctx() const
{
    return m_system->m_context;
}

Json OffDiagonalSubsystem::as_json() const
{
    auto j                         = ILinearSubsystem::as_json();
    j["coupling"]                  = {typeid(*m_a).name(), typeid(*m_b).name()};
    j["upper_hessian_block_count"] = m_upper_hessian_count;
    j["lower_hessian_block_count"] = m_lower_hessian_count;
    j["hessian_block_count"]       = hessian_block_count();
    return j;
}

Vector2i OffDiagonalSubsystem::dof_offset() const
{
    return Vector2i{m_a->dof_offset()(0), m_b->dof_offset()(0)};
}

void OffDiagonalSubsystem::do_assemble(TripletMatrixView hessian, DenseVectorView)
{
    // redirect to the coupling version
    int2 ij_offset{m_a->dof_offset()[0] / 3, m_b->dof_offset()[0] / 3};
    int2 ij_count{m_a->right_hand_side_dof() / 3, m_b->right_hand_side_dof() / 3};

    int2 ji_offset = {ij_offset.y, ij_offset.x};
    int2 ji_count  = {ij_count.y, ij_count.x};

    auto upper_view =
        hessian.subview(hessian_block_offset(), m_upper_hessian_count).submatrix(ij_offset, ij_count);
    auto lower_view = hessian
                          .subview(hessian_block_offset() + m_upper_hessian_count, m_lower_hessian_count)
                          .submatrix(ji_offset, ji_count);

    assemble(upper_view, lower_view);
}

void OffDiagonalSubsystem::hessian_block_count(SizeT upper_hessian_count, SizeT lower_hessian_count)
{
    m_upper_hessian_count = upper_hessian_count;
    m_lower_hessian_count = lower_hessian_count;
    Base::hessian_block_count(upper_hessian_count + lower_hessian_count);
}
}  // namespace gipc::linear_system
