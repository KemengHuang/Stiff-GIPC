#include <GIPC.cuh>
#include <gipc/gipc.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/json.h>
#include <fstream>

void GIPC::build_gipc_system(device_TetraData& tet)
{
    std::cout << "* Building GIPC system:" << std::endl;
    gipc::Timer::disable_all();
    
    // set up debug
    muda::Debug::debug_sync_all(false);

    std::cout << "- create ABD system..." << std::endl;
    m_abd_sim_data            = std::make_unique<gipc::ABDSimData>(*this, tet);
    m_abd_system              = std::make_unique<gipc::ABDSystem>();
    m_abd_system->parms.kappa = 1e8;
    m_abd_system->parms.dt    = IPC_dt;

    std::string config_dir = GIPC_ASSETS_DIR "scene/abd_system_config.json";

    gipc::Json json = gipc::Json::parse(std::ifstream(std::string{config_dir}));
    

    m_abd_system->parms.motor_speed = json["motor_speed"].get<double>();
    m_abd_system->parms.motor_strength = json["motor_strength"].get<double>();
    // m_abd_system->parms.init_q_v.segment<3>(0) = Eigen::Vector3d::UnitZ();
    //m_abd_system->parms.gravity = Eigen::Vector3d::Zero();

    std::cout << "- create Contact System ..." << std::endl;
    m_contact_system = std::make_unique<gipc::ContactSystem>(*this);
    std::cout << "    - create Contact Reporter ..." << std::endl;
    m_contact_system->create_contact_reporter<gipc::ContactInfoReporter>(*this, tet);
    m_contact_system->report_info(false);

    std::cout << "- create Global Linear System ..." << std::endl;
    gipc::GlobalLinearSystemOptions options;
    {
        options.spmv_algorithm = gipc::SPMVAlgorithm::SymWarpReduceBCOO;

        options.convert_algorithm = gipc::ConvertAlgorithm::NewConverter;
    }
    m_global_linear_system = std::make_unique<gipc::GlobalLinearSystem>(options);


    //std::cout << "- global linear system info:" << std::endl;
    //std::cout << std::setw(4) <<m_global_linear_system->as_json() << std::endl;

    std::cout << "* Finished building GIPC system." << std::endl;
}

void GIPC::init_abd_system()
{
    m_abd_sim_data->upload();
    m_abd_system->init_system(*m_abd_sim_data);
}

void GIPC::create_LinearSystem(device_TetraData& tet)
{
    std::cout << "    - create ABD Linear Subsystem ..." << std::endl;
    auto& abd = m_global_linear_system->create<gipc::ABDLinearSubsystem>(
        *this, *m_contact_system, *m_abd_system, *m_abd_sim_data);
    std::cout << "    - create FEM Linear Subsystem ..." << std::endl;
    auto& fem = m_global_linear_system->create<gipc::FEMLinearSubsystem>(*this, tet, *m_contact_system);
    //femSys = fem;

    std::cout << "    - create ABD FEM Off Diagonal ..." << std::endl;
    m_global_linear_system->create<gipc::ABDFEMOffDiagonal>(
        *this, tet, *m_contact_system, abd, fem);

    std::cout << "- create PCG Solver" << std::endl;
    gipc::PCGSolverConfig cfg;
    cfg.global_tol_rate = 1e-4;
    auto& pcg           = m_global_linear_system->create<gipc::PCGSolver>(cfg);

    std::cout << "- create Preconditioner" << std::endl;
    m_global_linear_system->create<gipc::DiagPreconditioner>();
    m_global_linear_system->create<gipc::ABDPreconditioner>(abd, *m_abd_system, *m_abd_sim_data);

    if(pcg_data.P_type == 1)
    {

        m_global_linear_system->create<gipc::MAS_Preconditioner>(
            fem, BH, pcg_data.MP, tet.masses, h_cpNum);
    }
}