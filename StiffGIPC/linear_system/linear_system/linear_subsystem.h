#pragma once
#include <gipc/type_define.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <gipc/utils/json.h>

namespace gipc
{
class GlobalLinearSystem;
class DiagonalSubsystem;

class ILinearSubsystem
{
  public:
    using DenseVectorView   = gipc::cuda::DenseVectorView<Float>;
    using CDenseVectorView  = gipc::cuda::CDenseVectorView<Float>;

  private:
    IndexT m_hid            = 0;  // ID for the hessian block
    IndexT m_gid            = 0;  // ID for the gradient block
    IndexT m_hessian_offset = 0;

    // hessian block count (triplets)
    //SizeT               m_hessian_block_count = 0;
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
    //void hessian_block_count(SizeT hessian_block_count);
    //auto hessian_block_count() const { return m_hessian_block_count; }
    /**
     * \brief report subsystem information 
     */
    virtual void               report_subsystem_info() = 0;
    gipc::cuda::LinearSystemContext& ctx() const;

  private:
    
    friend class DiagonalSubsystem;
    friend class IPreconditioner;

    //void hessian_block_offset(IndexT hessian_offset);
    //auto hessian_block_offset() const { return m_hessian_offset; }

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

    virtual void do_assemble(DenseVectorView gradient) = 0;
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
    virtual void assemble(DenseVectorView gradient) = 0;

    /**
     * \brief Subclass should implement this function to get back the solution of the linear system
     * 
     * \param[out] dx: The solution of the linear subsystem, in the format of dense vector.
     *  Note that the solution is just a sub-vector of the global solution, the linear subsystem
     *  can just index it from 0 to `right_hand_side_dof() - 1`. 
     */
    virtual void retrieve_solution(CDenseVectorView dx) = 0;

  private:
    virtual Vector2i dof_offset() const final override;
    void             dof_offset(IndexT dof_offset);
    virtual void do_assemble(DenseVectorView gradient) final override;
    void do_retrieve_solution(CDenseVectorView dx);
};

}  // namespace gipc