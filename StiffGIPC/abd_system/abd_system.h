#pragma once
#include <abd_system/abd_sim_data.h>
#include <gipc/cuda/all.h>
#include <gipc/cuda/all.h>
#include <abd_system/abd_system_parms.h>
#include <linear_system/utils/converter.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc
{
class ABDSystem
{
  private:
    gipc::cuda::DeviceBuffer<Vector12>  m_temp_q;
    gipc::cuda::DeviceBuffer<Vector12>  m_temp_q_v;
    gipc::cuda::DeviceBuffer<Vector12>  m_temp_q_prev;
    gipc::cuda::DeviceBuffer<Vector12>  m_temp_q_tilde;
    gipc::cuda::DeviceBuffer<int>       m_temp_is_fixed;
    gipc::cuda::DeviceBuffer<ABDJacobi> m_temp_jacobi;
    gipc::cuda::DeviceBuffer<Vector12>  m_temp_abd_gravity_force;
    gipc::cuda::DeviceVar<Float>        m_kinetic_energy;
    gipc::cuda::DeviceBuffer<Float>     m_kinetic_energy_per_affine_body;

    gipc::cuda::DeviceVar<Float>      m_shape_energy;
    gipc::cuda::DeviceBuffer<Float>   m_shape_energy_per_affine_body;
    gipc::cuda::DeviceBuffer<Vector3> m_body_centered_positions;

    Float                     m_suggest_max_tolerance = 0.0f;
    gipc::cuda::DeviceBuffer<Float> m_local_tolerance;
    gipc::cuda::DeviceVar<Float>    m_local_tolerance_max;

  public:  // public just for convenience
    int*                                  fem_boundary_type;
    gipc::cuda::DeviceBuffer<Vector12>          abd_gradient;  // just for legacy code
    gipc::cuda::DeviceBuffer<Matrix12x12>       abd_body_hessian;
    GIPCTripletMatrix*   global_triplet;
    gipc::cuda::DeviceDenseMatrix<double>       dense_system_hessian;
    gipc::cuda::DeviceCSRMatrix<double>         csr_system_hessian;
    gipc::cuda::DeviceDenseVector<double>       system_gradient;
    gipc::cuda::DeviceDenseVector<double>       temp_system_gradient;
    gipc::cuda::DeviceDoubletVector<double, 12> doublet_system_gradient;
    gipc::cuda::DeviceBuffer<Matrix12x12>       abd_system_diag_preconditioner;

    gipc::cuda::DeviceBuffer<Vector3> body_mass_center;
    gipc::cuda::DeviceBuffer<Float>   body_mass;
    gipc::cuda::DeviceBuffer<int>     body_unique_point_count;
    // Vector6: [x0,x1,x2,x3,x4,x5]
    // Axis from [x0,x1,x2] to [x3,x4,x5]
    // Vector6 == Zero for non-motorized body
    gipc::cuda::DeviceBuffer<Vector6> body_id_to_motor_rotation_axis;
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
    void rebuild_system(ABDSimData& sim_data, gipc::cuda::CBufferView<double3> vertices);

    // init == true, means we are at frame 0, just init the abd system
    // init == false, means we are at frame > 0, we need to rebuild the abd system
    void _setup_system(bool init, ABDSimData&);

    void _setup_unique_point_mass(size_t                     unique_point_count,
                                  gipc::cuda::DeviceBuffer<Float>& unique_point_mass,
                                  gipc::cuda::CBufferView<TetLocalInfo> tets,
                                  gipc::cuda::CBufferView<Float>        tet_volumes,
                                  Float                           density,
                                  gipc::cuda::CBufferView<int> point_id_to_unique_point_id);

    void _calculate_body_mass_center(size_t body_count,
                                     gipc::cuda::DeviceBuffer<Float>& unique_point_mass,
                                     gipc::cuda::CBufferView<double3> unique_point_position,
                                     gipc::cuda::CBufferView<int> unique_point_id_to_body_id);


    // setup at frame 0
    void _setup_J(gipc::cuda::DeviceBuffer<ABDJacobi>& jacobi,
                  gipc::cuda::CBufferView<double3>     unique_point_position,
                  gipc::cuda::CBufferView<int>         unique_point_id_to_body_id,
                  gipc::cuda::CBufferView<Vector12>    q);

    void _setup_abd_state(size_t                        abd_count,
                          gipc::cuda::DeviceBuffer<Vector12>& q,
                          gipc::cuda::DeviceBuffer<Vector12>& q_temp,
                          gipc::cuda::DeviceBuffer<Vector12>& q_tilde,
                          gipc::cuda::DeviceBuffer<Vector12>& q_prev,
                          gipc::cuda::DeviceBuffer<Vector12>& q_v,
                          gipc::cuda::DeviceBuffer<Vector12>& dq);


    // if body breakup happens, we need to spawn state
    void _spawn_abd_state(gipc::cuda::CBufferView<int>        body_id_to_old_body_id,
                          gipc::cuda::DeviceBuffer<int>&      body_id_to_is_fixed,
                          gipc::cuda::DeviceBuffer<Vector12>& q,
                          gipc::cuda::DeviceBuffer<Vector12>& q_temp,
                          gipc::cuda::DeviceBuffer<Vector12>& q_tilde,
                          gipc::cuda::DeviceBuffer<Vector12>& q_prev,
                          gipc::cuda::DeviceBuffer<Vector12>& q_v,
                          gipc::cuda::DeviceBuffer<Vector12>& dq);
    // if body breakup happens, we need to spawn J
    void _spawn_J(gipc::cuda::DeviceBuffer<ABDJacobi>& jacobi,
                  gipc::cuda::CBufferView<int> unique_point_to_old_unique_point);


    void _setup_tet_abd_mass(gipc::cuda::CBufferView<TetLocalInfo> tet_local_info,
                             gipc::cuda::CBufferView<int> point_id_to_unique_point_id,
                             gipc::cuda::CBufferView<ABDJacobi> jacobi,
                             gipc::cuda::CBufferView<Float>     tet_volumes,
                             Float                        density,
                             gipc::cuda::DeviceBuffer<ABDJacobiDyadicMass>& tet_dyadic_mass);

    void _setup_abd_dyadic_mass(size_t affine_body_count,
                                gipc::cuda::CBufferView<ABDJacobiDyadicMass> tet_dyadic_mass,
                                gipc::cuda::CBufferView<int> tet_id_to_body_id,
                                gipc::cuda::DeviceBuffer<ABDJacobiDyadicMass>& abd_dyadic_mass,
                                gipc::cuda::DeviceBuffer<Matrix12x12>& abd_dyadic_mass_inv);

    void _setup_abd_volume(size_t                     affine_body_count,
                           gipc::cuda::CBufferView<int>     tet_id_to_body_id,
                           gipc::cuda::CBufferView<Float>   tet_volumes,
                           gipc::cuda::DeviceBuffer<Float>& abd_volume);

    void _setup_tet_abd_gravity_force(const Vector3& gravity,
                                      gipc::cuda::CBufferView<TetLocalInfo> tet_local_info,
                                      gipc::cuda::CBufferView<int> point_id_to_unique_point_id,
                                      gipc::cuda::CBufferView<ABDJacobi> jacobi,
                                      gipc::cuda::CBufferView<Float>     tet_volumes,
                                      Float                        density,
                                      gipc::cuda::DeviceBuffer<Vector12>& tet_abd_gravity_force);

    void _setup_abd_gravity(gipc::cuda::CBufferView<Vector12> tet_abd_gravity_force,
                            gipc::cuda::CBufferView<int>      tet_id_to_body_id,
                            size_t                      affine_body_count,
                            gipc::cuda::CBufferView<Matrix12x12> abd_dyadic_mass_inv,
                            gipc::cuda::DeviceBuffer<Vector12>&  abd_gravity);

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
    void cal_x_from_q(ABDSimData& sim_data, gipc::cuda::BufferView<double3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, gipc::cuda::BufferView<double3> move_dir);
    void cal_x_from_q(ABDSimData& sim_data, gipc::cuda::BufferView<Vector3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, gipc::cuda::BufferView<Vector3> move_dir);

    /********************************************************************************/

    /// <summary>
    /// Main API: calculate abd system gradient and hessian.
    /// before calling this, you need to fill the `triplet_vertex_hessian` (barrier + ground barrier)
    /// </summary>
    /// <param name="sim_data"></param>
    /// <param name="vertex_barrier_gradient"></param>
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           gipc::cuda::CBufferView<double3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           gipc::cuda::CBufferView<Vector3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           int*        fbtype,
                                           gipc::cuda::CBufferView<double3> vertex_barrier_gradient,
                                           GIPCTripletMatrix& global_triplets);

    void _cal_abd_body_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          gipc::cuda::CBufferView<double3> vertex_barrier_gradient);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          gipc::cuda::CBufferView<Vector3> vertex_barrier_gradient);
    void _setup_abd_system_hessian(ABDSimData& sim_data,
                                   GIPCTripletMatrix& global_triplets);
    void _cal_abd_system_preconditioner(ABDSimData& sim_data);

    /********************************************************************************/

    // when doing line search, we need to copy q to q_temp
    void copy_q_to_q_temp(ABDSimData& sim_data);

    // move forward to test the energy
    void step_forward(ABDSimData&                sim_data,
                      gipc::cuda::BufferView<double3>  vertices,
                      double                     alpha);

    // when doing line search, we need calculate abd energy from q
    Float cal_abd_kinetic_energy(ABDSimData& sim_data);
    Float cal_abd_shape_energy(ABDSimData& sim_data);
};
}  // namespace gipc