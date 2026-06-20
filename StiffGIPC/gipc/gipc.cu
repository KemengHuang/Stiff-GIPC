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
    gipc::cuda::Debug::debug_sync_all(false);

    std::cout << "- create ABD system..." << std::endl;
    m_abd_sim_data            = std::make_unique<gipc::ABDSimData>(*this, tet);
    m_abd_system              = std::make_unique<gipc::ABDSystem>();
    m_abd_system->parms.kappa = 1e8;
    m_abd_system->parms.dt    = IPC_dt;

    std::string config_dir = GIPC_ASSETS_DIR "scene/abd_system_config.json";

    gipc::Json json = gipc::Json::parse(std::ifstream(std::string{config_dir}));
    

    m_abd_system->parms.motor_speed = json["motor_speed"].get<double>();
    m_abd_system->parms.motor_strength = json["motor_strength"].get<double>();

    std::cout << "- create Global Linear System ..." << std::endl;

    m_global_linear_system = std::make_unique<gipc::GlobalLinearSystem>();

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
        *this, *m_abd_system, *m_abd_sim_data);
    std::cout << "    - create FEM Linear Subsystem ..." << std::endl;
    auto& fem = m_global_linear_system->create<gipc::FEMLinearSubsystem>(*this, tet);


    std::cout << "- create PCG Solver" << std::endl;
    gipc::PCGSolverConfig cfg;
    cfg.global_tol_rate = pcg_threshold;
    auto& pcg           = m_global_linear_system->create<gipc::PCGSolver>(cfg);

    std::cout << "- create Preconditioner" << std::endl;
    
    m_global_linear_system->create<gipc::ABDPreconditioner>(abd, *m_abd_system, *m_abd_sim_data);

    if(pcg_data.P_type == 1)
    {

        m_global_linear_system->create<gipc::MAS_Preconditioner>(
            fem, pcg_data.MP, tet.masses, h_cpNum);
    }
    else
    {
        m_global_linear_system->create<gipc::DiagPreconditioner>();
    }
}