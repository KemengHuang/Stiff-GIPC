#pragma once
#include <list>
#include <linear_system/utils/spmv.h>
#include <linear_system/utils/converter.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/cuda/all.h>
#include <gipc/utils/json.h>


namespace gipc
{
class GlobalLinearSystem
{
    template <typename T>
    using U = std::unique_ptr<T>;
    friend class IterativeSolver;
    friend class IPreconditioner;
    friend class ILinearSubsystem;
    friend class LocalPreconditioner;

  public:
    GlobalLinearSystem() {}

    ~GlobalLinearSystem();

    static constexpr int BlockSize = 3;

    template <typename T, typename... Args>
    T& create(Args&&... args)
    {
        if constexpr(std::is_base_of_v<ILinearSubsystem, T>)
        {
            return static_cast<T&>(
                _create_subsystem(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IterativeSolver, T>)
        {
            return static_cast<T&>(
                _create_solver(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IPreconditioner, T>)
        {
            return static_cast<T&>(_create_preconditioner(
                std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else
        {
            MUDA_ASSERT(false, "Unknown type");
        }
    }

    /**
     * \brief solve the global linear system, using the specified solver
     * 
     * \details `solve_linear_system()` will:
     * - build the global linear system from the subsystems
     * - distribute the assembly assignments to the subsystems
     * - solve the linear system using the specified solver
     * - distribute the solution to the subsystems
     */
    gipc::SizeT solve_linear_system();

    Json               as_json() const;
    GIPCTripletMatrix* gipc_global_triplet = nullptr;

  private:
    std::vector<U<ILinearSubsystem>> m_subsystems;
    std::vector<DiagonalSubsystem*>  m_inner_subsystems;

    std::vector<U<LocalPreconditioner>> m_local_preconditioners;
    U<GlobalPreconditioner>             m_global_preconditioner;
    U<IterativeSolver>                  m_solver;

    gipc::cuda::LinearSystemContext      m_context;
    gipc::cuda::DeviceDenseVector<Float> m_x;
    gipc::cuda::DeviceDenseVector<Float> m_b;

    std::vector<SizeT> m_rhs_count_per_subsystem;
    std::vector<SizeT> m_rhs_offset_per_subsystem;
    std::vector<Float> m_accuracy_statisfied_per_subsystem;

    size_t                         reserved_triplet_count = 0;
    Spmv                           m_spmv;
    Converter                      m_converter;
    gipc::cuda::DeviceDenseVector<Float> fake_y;


    bool build_linear_system();
    void distribute_solution();
    void apply_preconditioner(gipc::cuda::DenseVectorView<Float>  z,
                              gipc::cuda::CDenseVectorView<Float> r);

    void convert_new();

    void spmv(Float a, gipc::cuda::CDenseVectorView<Float> x, Float b, gipc::cuda::DenseVectorView<Float> y);

    DiagonalSubsystem& _create_subsystem(U<DiagonalSubsystem>&& subsystem);

    IterativeSolver& _create_solver(U<IterativeSolver>&& solver);
    LocalPreconditioner& _create_preconditioner(U<LocalPreconditioner>&& preconditioner);
    GlobalPreconditioner& _create_preconditioner(U<GlobalPreconditioner>&& preconditioner);
};
}  // namespace gipc
