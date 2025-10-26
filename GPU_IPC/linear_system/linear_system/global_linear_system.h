//
// created by Xinyu Lu on 2024/4/3
// Copyright (c) 2024 Xinyu Lu. All rights reserved.
//
#pragma once
#include <list>
#include <linear_system/utils/spmv.h>
#include <linear_system/utils/converter.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <muda/ext/linear_system.h>
#include <linear_system/linear_system/global_linear_system_options.h>
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
    GlobalLinearSystem(const GlobalLinearSystemOptions& options)
        : m_options{options}
    {
    }

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

    Json as_json() const;

  private:
    GlobalLinearSystemOptions          m_options;
    std::vector<U<ILinearSubsystem>>   m_subsystems;
    std::vector<DiagonalSubsystem*>    m_inner_subsystems;
    std::vector<OffDiagonalSubsystem*> m_coupling_subsystems;

    std::vector<U<LocalPreconditioner>> m_local_preconditioners;
    U<GlobalPreconditioner>             m_global_preconditioner;
    U<IterativeSolver>                  m_solver;

    muda::LinearSystemContext           m_context;
    muda::DeviceDenseVector<Float>      m_x;
    muda::DeviceDenseVector<Float>      m_b;
    muda::DeviceTripletMatrix<Float, 3> m_triplet_A;
    muda::DeviceBCOOMatrix<Float, 3>    m_bcoo_A;
    muda::DeviceBCOOMatrix<Float, 3>    fake_bcoo_A;

    muda::DeviceBSRMatrix<Float, 3> m_bsr_A;

    std::vector<SizeT> m_triplet_offset_per_subsystem;
    std::vector<SizeT> m_triplet_count_per_subsystem;
    std::vector<SizeT> m_rhs_count_per_subsystem;
    std::vector<SizeT> m_rhs_offset_per_subsystem;
    std::vector<Float> m_accuracy_statisfied_per_subsystem;

    size_t                         reserved_triplet_count = 0;
    Spmv                           m_spmv;
    Converter                      m_converter;
    muda::DeviceDenseVector<Float> fake_y;


    bool build_linear_system();
    void distribute_solution();
    void apply_preconditioner(muda::DenseVectorView<Float>  z,
                              muda::CDenseVectorView<Float> r);
    bool accuracy_statisfied(muda::DenseVectorView<Float> r);
    void convert();

    void spmv(Float a, muda::CDenseVectorView<Float> x, Float b, muda::DenseVectorView<Float> y);

    DiagonalSubsystem& _create_subsystem(U<DiagonalSubsystem>&& subsystem);
    OffDiagonalSubsystem& _create_subsystem(U<OffDiagonalSubsystem>&& subsystem);

    IterativeSolver& _create_solver(U<IterativeSolver>&& solver);
    LocalPreconditioner& _create_preconditioner(U<LocalPreconditioner>&& preconditioner);
    GlobalPreconditioner& _create_preconditioner(U<GlobalPreconditioner>&& preconditioner);
};
}  // namespace gipc
