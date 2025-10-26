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

    // hessian can be provided by both LinearSubsystem and LinearCouplingSubsystem
    m_triplet_count_per_subsystem.resize(hessian_provider_count);
    m_triplet_offset_per_subsystem.resize(hessian_provider_count);

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    // collect the information of each subsystem
    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    for(auto& hp : m_subsystems)
    {
        auto i                           = hp->hid();
        m_triplet_count_per_subsystem[i] = hp->hessian_block_count();
    }

    // compute the triplet offset for each subsystem
    std::exclusive_scan(m_triplet_count_per_subsystem.begin(),
                        m_triplet_count_per_subsystem.end(),
                        m_triplet_offset_per_subsystem.begin(),
                        0);

    // compute the rhs offset for each subsystem
    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    for(auto& hp : m_subsystems)
    {
        auto i = hp->hid();
        hp->hessian_block_offset(m_triplet_offset_per_subsystem[i]);
    }

    // get the total number of triplets and right hand side dofs
    auto total_triplet_count = m_triplet_offset_per_subsystem.back()
                               + m_triplet_count_per_subsystem.back();
    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();

    if(total_triplet_count == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }

    // allocate the memory for the linear system
    m_triplet_A.resize(total_rhs_count / BlockSize, total_rhs_count / BlockSize, total_triplet_count);
    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    // assemble linear subsystems
    auto triplet_view = m_triplet_A.view();
    auto rhs_view     = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(triplet_view, rhs_view);

    convert();

    // assemble preconditioners
    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(m_bcoo_A);

    for(auto& p : m_local_preconditioners)
        p->assemble();

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

OffDiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<OffDiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(~0);
    m_coupling_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));

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
    auto triplet_view = m_triplet_A.view();

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

bool GlobalLinearSystem::accuracy_statisfied(muda::DenseVectorView<Float> r)
{
    // check if all subsystems' accuracy is satisfied
    return std::all_of(m_inner_subsystems.begin(),
                       m_inner_subsystems.end(),
                       [&](DiagonalSubsystem* s)
                       {
                           if(s->right_hand_side_dof() == 0)
                               return true;
                           else
                               return s->accuracy_statisfied(r.subview(
                                   s->dof_offset()(0), s->right_hand_side_dof()));
                       });
}
void GlobalLinearSystem::convert()
{
    auto triplet2bcoo = [&]
    {
        if(reserved_triplet_count < m_triplet_A.triplet_count())
        {
            reserved_triplet_count = m_triplet_A.triplet_count() * 1.5;
            m_bcoo_A.reserve_triplets(reserved_triplet_count);
            m_triplet_A.reserve_triplets(reserved_triplet_count);
        }

        if(m_options.convert_algorithm == ConvertAlgorithm::MudaBuiltIn)
            m_context.convert(m_triplet_A, m_bcoo_A);
        else if(m_options.convert_algorithm == ConvertAlgorithm::NewConverter)
            m_converter.convert(m_triplet_A, m_bcoo_A);
    };

    auto bcoo2bsr = [&]
    {
        if(reserved_triplet_count < m_bcoo_A.triplet_count())
        {
            reserved_triplet_count = m_bcoo_A.triplet_count() * 1.5;
            m_bsr_A.reserve(reserved_triplet_count / 4);
        }

        if(m_options.convert_algorithm == ConvertAlgorithm::MudaBuiltIn)
            m_context.convert(m_bcoo_A, m_bsr_A);
        else if(m_options.convert_algorithm == ConvertAlgorithm::NewConverter)
            m_converter.convert(m_bcoo_A, m_bsr_A);
    };

    auto bcoo2symbcoo = [&] { m_converter.ge2sym(m_bcoo_A); };

    auto symbcoo2bcoo = [&]
    {
        m_converter.sym2ge(m_bcoo_A, fake_bcoo_A);
        m_bcoo_A = std::move(fake_bcoo_A);
    };

    switch(m_options.spmv_algorithm)
    {
        case SPMVAlgorithm::BSR: {
            {
                gipc::Timer timer{"triplet->bcoo "};
                triplet2bcoo();
                // bcoo2symbcoo();
                // symbcoo2bcoo();
            }

            {
                gipc::Timer timer{"bcoo->bsr"};
                bcoo2bsr();
            }
        }
        break;
        case SPMVAlgorithm::BCOO:
        case SPMVAlgorithm::WarpReduceBCOO: {
            gipc::Timer timer{"triplet->bcoo"};
            triplet2bcoo();
        }
        break;
        case SPMVAlgorithm::Triplet: {
            // if we have global preconditioner
            // we need to convert the Triplet matrix to BCOO matrix
            // for global preconditioner assembly
            if(m_global_preconditioner)
            {
                gipc::Timer timer{"triplet->bcoo"};
                triplet2bcoo();
            }
        }
        break;
        case SPMVAlgorithm::SymBCOO:
        case SPMVAlgorithm::SymWarpReduceBCOO: {
            gipc::Timer timer{"triplet->bcoo"};
            triplet2bcoo();
            bcoo2symbcoo();
        }
        default:
            break;
    }
}
void GlobalLinearSystem::spmv(Float                         a,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
{
    // fake_y.resize(y.size());
    switch(m_options.spmv_algorithm)
    {
        case SPMVAlgorithm::BSR: {
            {
                Timer timer{"bsr_spmv"};
                m_context.spmv<Float, 3>(a, m_bsr_A, x, b, y);
            }
            //{
            //    Timer timer{"warp_reduce_spmv"};
            //    // use fake_y to store the result of bsr_spmv
            //    m_spmv.warp_reduce_spmv2(a, m_bcoo_A, x, b, fake_y);
            //}
        }
        break;
        case SPMVAlgorithm::BCOO: {
            Timer timer{"bcoo_spmv"};
            m_context.spmv<Float, 3>(a, m_bcoo_A, x, b, y);
        }
        break;
        case SPMVAlgorithm::Triplet: {
            Timer timer{"triplet_spmv"};
            m_context.spmv<Float, 3>(a, m_triplet_A, x, b, y);
        }
        break;
        case SPMVAlgorithm::SymBCOO: {
            Timer timer{"sym_bcoo_spmv"};
            m_spmv.sym_spmv(a, m_bcoo_A, x, b, y);
        }
        break;
        case SPMVAlgorithm::WarpReduceBCOO: {
            Timer timer{"warp_reduce_spmv"};
            m_spmv.warp_reduce_spmv(a, m_bcoo_A, x, b, y);
        }
        break;
        case SPMVAlgorithm::SymWarpReduceBCOO: {
            Timer timer{"warp_reduce_sym_spmv"};
            m_spmv.warp_reduce_sym_spmv(a, m_bcoo_A, x, b, y);
        }
        break;
        default:
            break;
    }
}
}  // namespace gipc
