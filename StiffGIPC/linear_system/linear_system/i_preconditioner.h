#pragma once
#include <gipc/type_define.h>
#include <muda/ext/linear_system.h>
#include <gipc/utils/json.h>

namespace muda
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
    muda::LinearSystemContext& ctx() const;

  private:
    void system(GlobalLinearSystem& system) { m_system = &system; }

    virtual void do_apply(muda::CDenseVectorView<Float> r,
                          muda::DenseVectorView<Float>  z) = 0;

    virtual void do_assemble(muda::CBCOOMatrixView<Float, 3> hessian) = 0;
};

class LocalPreconditioner : public IPreconditioner
{
    friend class GlobalLinearSystem;
    DiagonalSubsystem* m_subsystem;

  public:
    LocalPreconditioner(DiagonalSubsystem& subsystem);

    virtual ~LocalPreconditioner() = default;
    muda::CBufferView<int>          calculate_subsystem_bcoo_indices() const;
    int                             get_offset() const;
    muda::CBCOOMatrixView<Float, 3> system_bcoo_matrix() const;

  protected:
    virtual void assemble(){};
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) = 0;

  private:
    void do_apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;
    void do_assemble(muda::CBCOOMatrixView<Float, 3> hessian) override;

    mutable muda::DeviceBuffer<int> m_indices_input;
    mutable muda::DeviceBuffer<int> m_flags;
    mutable muda::DeviceBuffer<int> m_indices_output;
    mutable muda::DeviceVar<int>    m_count;

    template <typename T>
    static void loose_resize(muda::DeviceBuffer<T>& buf, size_t new_size, Float reserve_ratio = 1.3)
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
    virtual void assemble(muda::CBCOOMatrixView<Float, 3> hessian) = 0;
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) = 0;

  private:
    void do_apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;

    virtual void do_assemble(muda::CBCOOMatrixView<Float, 3> hessian) override;
};
}  // namespace gipc
