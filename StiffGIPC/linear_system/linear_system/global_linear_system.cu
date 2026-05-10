#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>

namespace gipc
{
bool GlobalLinearSystem::build_linear_system()
{
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    muda::wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    MUDA_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(!success)
        return 0;
    MUDA_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    auto iter = m_solver->solve(m_x, m_b);
    distribute_solution();
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                              muda::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



void GlobalLinearSystem::convert_new()
{
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
//#ifndef SymGH
//    m_converter.ge2sym(*gipc_global_triplet);
//#endif
}



void GlobalLinearSystem::spmv(Float                         a,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
{

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y);
}
}  // namespace gipc
