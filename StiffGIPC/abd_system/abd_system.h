#pragma once
#include <abd_system/abd_sim_data.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_all.h>
#include <abd_system/abd_system_parms.h>
#include <linear_system/utils/converter.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc
{
class ABDSystem
{
  private:
    cudatool::DeviceBuffer<Vector12>  m_temp_q;
    cudatool::DeviceBuffer<Vector12>  m_temp_q_v;
    cudatool::DeviceBuffer<Vector12>  m_temp_q_prev;
    cudatool::DeviceBuffer<Vector12>  m_temp_q_tilde;
    cudatool::DeviceBuffer<int>       m_temp_is_fixed;
    cudatool::DeviceBuffer<ABDJacobi> m_temp_jacobi;
    cudatool::DeviceBuffer<Vector12>  m_temp_abd_gravity_force;
    cudatool::DeviceVar<Float>        m_kinetic_energy;
    cudatool::DeviceBuffer<Float>     m_kinetic_energy_per_affine_body;

    cudatool::DeviceVar<Float>      m_shape_energy;
    cudatool::DeviceBuffer<Float>   m_shape_energy_per_affine_body;
    cudatool::DeviceBuffer<Vector3> m_body_centered_positions;

    Float                     m_suggest_max_tolerance = 0.0f;
    cudatool::DeviceBuffer<Float> m_local_tolerance;
    cudatool::DeviceVar<Float>    m_local_tolerance_max;

  public:  // public just for convenience
    int*                                  fem_boundary_type;
    cudatool::DeviceBuffer<Vector12>          abd_gradient;  // just for legacy code
    cudatool::DeviceBuffer<Matrix12x12>       abd_body_hessian;
    GIPCTripletMatrix*   global_triplet;
    cudatool::DeviceDenseMatrix<double>       dense_system_hessian;
    cudatool::DeviceCSRMatrix<double>         csr_system_hessian;
    cudatool::DeviceDenseVector<double>       system_gradient;
    cudatool::DeviceDenseVector<double>       temp_system_gradient;
    cudatool::DeviceDoubletVector<double, 12> doublet_system_gradient;
    cudatool::DeviceBuffer<Matrix12x12>       abd_system_diag_preconditioner;

    cudatool::DeviceBuffer<Vector3> body_mass_center;
    cudatool::DeviceBuffer<Float>   body_mass;
    cudatool::DeviceBuffer<int>     body_unique_point_count;
    // Vector6: [x0,x1,x2,x3,x4,x5]
    // Axis from [x0,x1,x2] to [x3,x4,x5]
    // Vector6 == Zero for non-motorized body
    cudatool::DeviceBuffer<Vector6> body_id_to_motor_rotation_axis;
    gipc::Converter             converter3x3;

    size_t triplet_vertex_hessian_reserve_size = 0;
    size_t abd_system_hessian_reserve_size     = 0;

  public:
    ABDSystemParms parms;

    /******************************************************************************
    *                             build function
    *******************************************************************************/

    /// <summary>
    /// Main API: init the abd system at frame 0
    /// </summary>
    /// <param name="sim_data"></param>
    void init_system(ABDSimData& sim_data);

    /// <summary>
    /// Main API: rebuild the abd system if needed (body broken)
    /// </summary>
    /// <param name="sim_data"></param>
    void rebuild_system(ABDSimData& sim_data);
    void rebuild_system(ABDSimData& sim_data, cudatool::CBufferView<double3> vertices);

    // init == true, means we are at frame 0, just init the abd system
    // init == false, means we are at frame > 0, we need to rebuild the abd system
    void _setup_system(bool init, ABDSimData&);

    void _setup_unique_point_mass(size_t                     unique_point_count,
                                  cudatool::DeviceBuffer<Float>& unique_point_mass,
                                  cudatool::CBufferView<TetLocalInfo> tets,
                                  cudatool::CBufferView<Float>        tet_volumes,
                                  Float                           density,
                                  cudatool::CBufferView<int> point_id_to_unique_point_id);

    void _calculate_body_mass_center(size_t body_count,
                                     cudatool::DeviceBuffer<Float>& unique_point_mass,
                                     cudatool::CBufferView<double3> unique_point_position,
                                     cudatool::CBufferView<int> unique_point_id_to_body_id);


    // setup at frame 0
    void _setup_J(cudatool::DeviceBuffer<ABDJacobi>& jacobi,
                  cudatool::CBufferView<double3>     unique_point_position,
                  cudatool::CBufferView<int>         unique_point_id_to_body_id,
                  cudatool::CBufferView<Vector12>    q);

    void _setup_abd_state(size_t                        abd_count,
                          cudatool::DeviceBuffer<Vector12>& q,
                          cudatool::DeviceBuffer<Vector12>& q_temp,
                          cudatool::DeviceBuffer<Vector12>& q_tilde,
                          cudatool::DeviceBuffer<Vector12>& q_prev,
                          cudatool::DeviceBuffer<Vector12>& q_v,
                          cudatool::DeviceBuffer<Vector12>& dq);


    // if body breakup happens, we need to spawn state
    void _spawn_abd_state(cudatool::CBufferView<int>        body_id_to_old_body_id,
                          cudatool::DeviceBuffer<int>&      body_id_to_is_fixed,
                          cudatool::DeviceBuffer<Vector12>& q,
                          cudatool::DeviceBuffer<Vector12>& q_temp,
                          cudatool::DeviceBuffer<Vector12>& q_tilde,
                          cudatool::DeviceBuffer<Vector12>& q_prev,
                          cudatool::DeviceBuffer<Vector12>& q_v,
                          cudatool::DeviceBuffer<Vector12>& dq);
    // if body breakup happens, we need to spawn J
    void _spawn_J(cudatool::DeviceBuffer<ABDJacobi>& jacobi,
                  cudatool::CBufferView<int> unique_point_to_old_unique_point);


    void _setup_tet_abd_mass(cudatool::CBufferView<TetLocalInfo> tet_local_info,
                             cudatool::CBufferView<int> point_id_to_unique_point_id,
                             cudatool::CBufferView<ABDJacobi> jacobi,
                             cudatool::CBufferView<Float>     tet_volumes,
                             Float                        density,
                             cudatool::DeviceBuffer<ABDJacobiDyadicMass>& tet_dyadic_mass);

    void _setup_abd_dyadic_mass(size_t affine_body_count,
                                cudatool::CBufferView<ABDJacobiDyadicMass> tet_dyadic_mass,
                                cudatool::CBufferView<int> tet_id_to_body_id,
                                cudatool::DeviceBuffer<ABDJacobiDyadicMass>& abd_dyadic_mass,
                                cudatool::DeviceBuffer<Matrix12x12>& abd_dyadic_mass_inv);

    void _setup_abd_volume(size_t                     affine_body_count,
                           cudatool::CBufferView<int>     tet_id_to_body_id,
                           cudatool::CBufferView<Float>   tet_volumes,
                           cudatool::DeviceBuffer<Float>& abd_volume);

    void _setup_tet_abd_gravity_force(const Vector3& gravity,
                                      cudatool::CBufferView<TetLocalInfo> tet_local_info,
                                      cudatool::CBufferView<int> point_id_to_unique_point_id,
                                      cudatool::CBufferView<ABDJacobi> jacobi,
                                      cudatool::CBufferView<Float>     tet_volumes,
                                      Float                        density,
                                      cudatool::DeviceBuffer<Vector12>& tet_abd_gravity_force);

    void _setup_abd_gravity(cudatool::CBufferView<Vector12> tet_abd_gravity_force,
                            cudatool::CBufferView<int>      tet_id_to_body_id,
                            size_t                      affine_body_count,
                            cudatool::CBufferView<Matrix12x12> abd_dyadic_mass_inv,
                            cudatool::DeviceBuffer<Vector12>&  abd_gravity);

    /*******************************************************************************
    *                                 involution
    ********************************************************************************/

    Float suggest_max_tolerance(ABDSimData& sim_data)
    {
        return m_suggest_max_tolerance;
    }

    // update veclocity from q and q_prev
    void update_velocity(ABDSimData& sim_data);
    // calculate predicted position
    void cal_q_tilde(ABDSimData& sim_data);
    // mapping q to x
    void cal_x_from_q(ABDSimData& sim_data, cudatool::BufferView<double3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, cudatool::BufferView<double3> move_dir);
    void cal_x_from_q(ABDSimData& sim_data, cudatool::BufferView<Vector3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, cudatool::BufferView<Vector3> move_dir);

    /********************************************************************************/

    /// <summary>
    /// Main API: calculate abd system gradient and hessian.
    /// before calling this, you need to fill the `triplet_vertex_hessian` (barrier + ground barrier)
    /// </summary>
    /// <param name="sim_data"></param>
    /// <param name="vertex_barrier_gradient"></param>
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           cudatool::CBufferView<double3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           cudatool::CBufferView<Vector3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           int*        fbtype,
                                           cudatool::CBufferView<double3> vertex_barrier_gradient,
                                           GIPCTripletMatrix& global_triplets);

    void _cal_abd_body_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          cudatool::CBufferView<double3> vertex_barrier_gradient);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          cudatool::CBufferView<Vector3> vertex_barrier_gradient);
    void _setup_abd_system_hessian(ABDSimData& sim_data,
                                   GIPCTripletMatrix& global_triplets);
    void _cal_abd_system_preconditioner(ABDSimData& sim_data);

    /********************************************************************************/

    // when doing line search, we need to copy q to q_temp
    void copy_q_to_q_temp(ABDSimData& sim_data);

    // move forward to test the energy
    void step_forward(ABDSimData&                sim_data,
                      cudatool::BufferView<double3>  vertices,
                      double                     alpha);

    // when doing line search, we need calculate abd energy from q
    Float cal_abd_kinetic_energy(ABDSimData& sim_data);
    Float cal_abd_shape_energy(ABDSimData& sim_data);
};
}  // namespace gipc