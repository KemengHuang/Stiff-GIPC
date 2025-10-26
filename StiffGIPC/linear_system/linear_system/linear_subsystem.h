#pragma once
#include <gipc/type_define.h>
#include <muda/ext/linear_system/doublet_vector_view.h>
#include <muda/ext/linear_system/triplet_matrix_view.h>
#include <muda/ext/linear_system/dense_vector_view.h>
#include <muda/ext/linear_system/linear_system_context.h>
#include <gipc/utils/json.h>

namespace gipc
{
class GlobalLinearSystem;
class DiagonalSubsystem;
class OffDiagonalSubsystem;

class ILinearSubsystem
{
  public:
    using TripletMatrixView = muda::TripletMatrixView<Float, 3>;
    using DenseVectorView   = muda::DenseVectorView<Float>;
    using CDenseVectorView  = muda::CDenseVectorView<Float>;

  private:
    IndexT m_hid            = 0;  // ID for the hessian block
    IndexT m_gid            = 0;  // ID for the gradient block
    IndexT m_hessian_offset = 0;

    // hessian block count (triplets)
    SizeT               m_hessian_block_count = 0;
    GlobalLinearSystem* m_system;

    friend class GlobalLinearSystem;
    void hid(IndexT id) { m_hid = id; }
    auto hid() const { return m_hid; }
    void gid(IndexT id) { m_gid = id; }
    auto gid() const { return m_gid; }
    void system(GlobalLinearSystem& system) { m_system = &system; }
    auto system() const { return m_system; }

  public:
    ILinearSubsystem() = default;
    virtual ~ILinearSubsystem();
    ILinearSubsystem(const ILinearSubsystem&)            = delete;
    ILinearSubsystem& operator=(const ILinearSubsystem&) = delete;

    ///**
    //* \brief return the information of the subsystem in json
    //*/
    virtual Json as_json() const;

  protected:
    void hessian_block_count(SizeT hessian_block_count);
    auto hessian_block_count() const { return m_hessian_block_count; }
    /**
     * \brief report subsystem information 
     */
    virtual void               report_subsystem_info() = 0;
    muda::LinearSystemContext& ctx() const;

  private:
    friend class OffDiagonalSubsystem;
    friend class DiagonalSubsystem;
    friend class IPreconditioner;

    void hessian_block_offset(IndexT hessian_offset);
    auto hessian_block_offset() const { return m_hessian_offset; }

    /**
     * \brief The dof offset in the global linear system. The offset is the global start index of
     * current linear subsystem in the global linear system. 
     * - For a inner-subsystem, `dof_offset(0) == dof_offset(1)`.
     * - For a coupling-subsystem, `dof_offset(0)` is the global start index of the first coupling subsystem
     *  `dof_offset(1)` is the global start index of the second coupling subsystem.
     * 
     * \return dof_offset in the global linear system
     */
    virtual Vector2i dof_offset() const = 0;

    virtual void do_assemble(TripletMatrixView hessian, DenseVectorView gradient) = 0;
};

/**
 * \brief The base class for linear subsystem, representing a subrange of:
 * - solution vector
 * - gradient vector (right hand side)
 * - triplet hessian matrix
 * in the global linear system.
 * 
 * \sa \ref CouplingLinearSubsystem
 */
class DiagonalSubsystem : public ILinearSubsystem
{
    using Base = ILinearSubsystem;
    friend class OffDiagonalSubsystem;
    friend class LocalPreconditioner;

  public:
    DiagonalSubsystem()          = default;
    virtual ~DiagonalSubsystem() = default;

    virtual Json as_json() const override;

  private:
    friend class GlobalLinearSystem;
    // right hand side size (segment count)
    SizeT m_right_hand_side_dof = 0;
    // dof offset in the global linear system
    IndexT m_dof_offset = 0;

  protected:
    void right_hand_side_dof(SizeT right_hand_side_dof);
    auto right_hand_side_dof() const { return m_right_hand_side_dof; }

    /**
    * \brief Subclass should implement this function to report some information of the subsystem:
    * - call `right_hand_side_dof()` to setup the size of the right hand side vector
    * if the subsystem is an inner-subsystem.
    * - call `hessian_block_count()` to setup the size of the hessian matrix,
    * if the subsystem is an inner-subsystem or a coupling-subsystem.
    */
    virtual void report_subsystem_info() = 0;

    /**
     * \brief Subclass should implement this function to assemble the linear system
     * 
     * 
     * 
     * \param[out] gradient: The gradient of the linear subsystem, in the format of dense vector,
     * note that the gradient is just a sub-vector of the global gradient, when assembling the
     * gradient, you don't need to care about the global index of the gradient, just fill it from
     * index 0 to `right_hand_side_dof() - 1`. If you need to get the global index, you can
     * call gradient.offset() to get the global start index of the gradient in the global gradient.
     * 
     * \sa \ref CouplingLinearSubsystem
     */
    virtual void assemble(TripletMatrixView hessian, DenseVectorView gradient) = 0;

    /**
     * \brief Subclass should implement this function to get back the solution of the linear system
     * 
     * \param[out] dx: The solution of the linear subsystem, in the format of dense vector.
     *  Note that the solution is just a sub-vector of the global solution, the linear subsystem
     *  can just index it from 0 to `right_hand_side_dof() - 1`. 
     */
    virtual void retrieve_solution(CDenseVectorView dx) = 0;

    /**
     * \brief Subclass should implement this function to check if the accuracy of the solution is satisfied.
     * If using direct solver, this function won't be called. If using iterative solver, this function will
     * be called to check if the accuracy of the solution is satisfied. If all subsystems return true, the
     * iterative solver will stop.
     * 
     * \param[in] residual The residual of the linear subsystem, in the format of dense vector.
     * it's subsytem's responsibility to decide if the residual is satisfied. You can use a local error
     * or a global error to decide if the residual is satisfied.
     * 
     * \return true if the accuracy of the solution is satisfied, false otherwise.
     */
    virtual bool accuracy_statisfied(CDenseVectorView residual) = 0;

  private:
    virtual Vector2i dof_offset() const final override;
    void             dof_offset(IndexT dof_offset);
    virtual void do_assemble(TripletMatrixView hessian, DenseVectorView gradient) final override;
    void do_retrieve_solution(CDenseVectorView dx);
};

/**
 * \brief The base class for coupling linear subsystem, 
 * representing only a subrange of triplet hessian matrix in the global linear system.
 * 
 * \details The coupling linear subsystem is a special case of linear subsystem, 
 * which don't need to assemble the gradient, and don't need to take the solution.
 * It just provides the inter-system coupling hessian matrix (off-diagonal block).
 * 
 * \sa \ref LinearSubsystem
 */
class OffDiagonalSubsystem : public ILinearSubsystem
{
    using Base = ILinearSubsystem;
    friend class GlobalLinearSystem;

    DiagonalSubsystem* m_a;
    DiagonalSubsystem* m_b;
    SizeT              m_upper_hessian_count = 0;
    SizeT              m_lower_hessian_count = 0;

  public:
    OffDiagonalSubsystem(DiagonalSubsystem& a, DiagonalSubsystem& b)
        : m_a(&a)
        , m_b(&b)
    {
    }

    virtual Json as_json() const override;

  protected:
    /**
    * \brief Subclass should implement this function to report some information of the subsystem:
    * - call `hessian_block_count()` to setup the size of the hessian matrix,
    * if the subsystem is an inner-subsystem or a coupling-subsystem.
    */
    virtual void report_subsystem_info() = 0;

    void hessian_block_count(SizeT upper_hessian_count, SizeT lower_hessian_count);
    auto hessian_block_count() const
    {
        return m_upper_hessian_count + m_lower_hessian_count;
    }
    auto upper_hessian_block_count() const { return m_upper_hessian_count; }
    auto lower_hessian_block_count() const { return m_lower_hessian_count; }

    /**
     * \brief Subclass should implement this function to assemble the hessian matrix
     * 
     * 
     */
    virtual void assemble(TripletMatrixView upper, TripletMatrixView lower) = 0;

  private:
    /**
     * \brief The dof offset in the global linear system. The offset is the global start indices of
     * two coupling subsystems in the global linear system. 
     * - `dof_offset(0)` is the global start index of the first coupling subsystem in the global linear system.
     * - `dof_offset(1)` is the global start index of the second coupling subsystem in the global linear system.
     * \return 
     */
    virtual Vector2i dof_offset() const override;

    void do_assemble(TripletMatrixView hessian, DenseVectorView gradient) final override;
};
}  // namespace gipc