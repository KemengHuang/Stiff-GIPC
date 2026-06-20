#pragma once
#include <gipc/type_define.h>
#include <cuda_tools/cuda_all.h>
#include <gipc/utils/json.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc::cuda
{
class LinearSystemContext;
}

namespace gipc
{
class GlobalLinearSystem;
class DiagonalSubsystem;
class LocalPreconditioner;
class GlobalPreconditioner;

class IPreconditioner
{
    friend class GlobalLinearSystem;
    friend class LocalPreconditioner;
    friend class GlobalPreconditioner;
    GlobalLinearSystem* m_system;
  public:
    IPreconditioner() = default;

    virtual ~IPreconditioner();

    virtual Json as_json() const;
    
  protected:
    cudatool::LinearSystemContext& ctx() const;

  private:
    void system(GlobalLinearSystem& system) { m_system = &system; }

    virtual void do_apply(cudatool::CDenseVectorView<Float> r,
                          cudatool::DenseVectorView<Float>  z) = 0;

    //virtual void do_assemble(cudatool::CBCOOMatrixView<Float, 3> hessian) = 0;
    virtual void do_assemble(GIPCTripletMatrix& global_triplets)      = 0;
};

class LocalPreconditioner : public IPreconditioner
{
    friend class GlobalLinearSystem;
    DiagonalSubsystem* m_subsystem;

  public:
    LocalPreconditioner(DiagonalSubsystem& subsystem);

    virtual ~LocalPreconditioner() = default;
    uint32_t* calculate_subsystem_bcoo_indices(int& number) const;
    int                             get_offset() const;
    Eigen::Matrix3d* system_bcoo_matrix() const;
    int* system_bcoo_rows() const;
    int* system_bcoo_cols() const;
    int                             preconditioner_id;
  protected:
    virtual void assemble(){};
    virtual void apply(cudatool::CDenseVectorView<Float> r, cudatool::DenseVectorView<Float> z) = 0;

  private:
    void do_apply(cudatool::CDenseVectorView<Float> r, cudatool::DenseVectorView<Float> z) override;
    void do_assemble(GIPCTripletMatrix& global_triplets) override;
    mutable cudatool::DeviceBuffer<int> m_indices_input;
    mutable cudatool::DeviceBuffer<int> m_flags;
    mutable cudatool::DeviceBuffer<int> m_indices_output;
    mutable cudatool::DeviceVar<int>    m_count;

    template <typename T>
    static void loose_resize(cudatool::DeviceBuffer<T>& buf, size_t new_size, Float reserve_ratio = 1.3)
    {
        if(buf.capacity() < new_size)
            buf.reserve(new_size * reserve_ratio);
        buf.resize(new_size);
    }
};

class GlobalPreconditioner : public IPreconditioner
{
    friend class GlobalLinearSystem;

  public:
    GlobalPreconditioner()          = default;
    virtual ~GlobalPreconditioner() = default;

  protected:
    //virtual void assemble(cudatool::CBCOOMatrixView<Float, 3> hessian) = 0;
    virtual void assemble(GIPCTripletMatrix& global_triplets)      = 0;
    virtual void apply(cudatool::CDenseVectorView<Float> r, cudatool::DenseVectorView<Float> z) = 0;

  private:
    void do_apply(cudatool::CDenseVectorView<Float> r, cudatool::DenseVectorView<Float> z) override;

    //virtual void do_assemble(cudatool::CBCOOMatrixView<Float, 3> hessian) override;
    virtual void do_assemble(GIPCTripletMatrix& global_triplets) override;
};
}  // namespace gipc
