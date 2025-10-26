#pragma once
#include <abd_system/abd_sim_data.h>
#include <muda/buffer/device_var.h>
#include <muda/ext/linear_system.h>
#include <abd_system/abd_system_parms.h>
#include <contact_system/contact_info.h>
#include <abd_system/utils/converter.h>
#include <linear_system/utils/converter.h>
namespace gipc
{
class ABDSystem
{
  private:
    muda::DeviceBuffer<Vector12>  m_temp_q;
    muda::DeviceBuffer<Vector12>  m_temp_q_v;
    muda::DeviceBuffer<Vector12>  m_temp_q_prev;
    muda::DeviceBuffer<Vector12>  m_temp_q_tilde;
    muda::DeviceBuffer<int>       m_temp_is_fixed;
    muda::DeviceBuffer<ABDJacobi> m_temp_jacobi;
    muda::DeviceBuffer<Vector12>  m_temp_abd_gravity_force;
    muda::DeviceVar<Float>        m_kinetic_energy;
    muda::DeviceBuffer<Float>     m_kinetic_energy_per_affine_body;

    muda::DeviceVar<Float>      m_shape_energy;
    muda::DeviceBuffer<Float>   m_shape_energy_per_affine_body;
    muda::DeviceBuffer<Vector3> m_body_centered_positions;

    Float                     m_suggest_max_tolerance = 0.0f;
    muda::DeviceBuffer<Float> m_local_tolerance;
    muda::DeviceVar<Float>    m_local_tolerance_max;

  public:  // public just for convenience
    muda::DeviceBuffer<Vector12>          abd_gradient;  // just for legacy code
    muda::DeviceBuffer<Matrix12x12>       abd_body_hessian;
    muda::LinearSystemContext             linear_system_context;
    muda::DeviceTripletMatrix<double, 12> triplet_hessian;
    muda::DeviceBCOOMatrix<double, 12>    bcoo_hessian;
    muda::DeviceBSRMatrix<double, 12>     bsr_hessian;
    muda::DeviceDenseMatrix<double>       dense_system_hessian;
    muda::DeviceCSRMatrix<double>         csr_system_hessian;
    muda::DeviceDenseVector<double>       system_gradient;
    muda::DeviceDenseVector<double>       temp_system_gradient;
    muda::DeviceDoubletVector<double, 12> doublet_system_gradient;
    muda::DeviceTripletMatrix<double, 3>  triplet_vertex_hessian;
    muda::DeviceBCOOMatrix<double, 3>     bcoo_vertex_hessian;
    muda::DeviceBuffer<Matrix12x12>       abd_system_diag_preconditioner;

    muda::DeviceBuffer<Vector3> body_mass_center;
    muda::DeviceBuffer<Float>   body_mass;
    muda::DeviceBuffer<int>     body_unique_point_count;
    // Vector6: [x0,x1,x2,x3,x4,x5]
    // Axis from [x0,x1,x2] to [x3,x4,x5]
    // Vector6 == Zero for non-motorized body
    muda::DeviceBuffer<Vector6> body_id_to_motor_rotation_axis;
    abd_system::Converter       converter;
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
    void rebuild_system(ABDSimData& sim_data, muda::CBufferView<double3> vertices);

    // init == true, means we are at frame 0, just init the abd system
    // init == false, means we are at frame > 0, we need to rebuild the abd system
    void _setup_system(bool init, ABDSimData&);

    void _setup_unique_point_mass(size_t                     unique_point_count,
                                  muda::DeviceBuffer<Float>& unique_point_mass,
                                  muda::CBufferView<TetLocalInfo> tets,
                                  muda::CBufferView<Float>        tet_volumes,
                                  Float                           density,
                                  muda::CBufferView<int> point_id_to_unique_point_id);

    void _calculate_body_mass_center(size_t body_count,
                                     muda::DeviceBuffer<Float>& unique_point_mass,
                                     muda::CBufferView<double3> unique_point_position,
                                     muda::CBufferView<int> unique_point_id_to_body_id);


    // setup at frame 0
    void _setup_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                  muda::CBufferView<double3>     unique_point_position,
                  muda::CBufferView<int>         unique_point_id_to_body_id,
                  muda::CBufferView<Vector12>    q);

    void _setup_abd_state(size_t                        abd_count,
                          muda::DeviceBuffer<Vector12>& q,
                          muda::DeviceBuffer<Vector12>& q_temp,
                          muda::DeviceBuffer<Vector12>& q_tilde,
                          muda::DeviceBuffer<Vector12>& q_prev,
                          muda::DeviceBuffer<Vector12>& q_v,
                          muda::DeviceBuffer<Vector12>& dq);


    // if body breakup happens, we need to spawn state
    void _spawn_abd_state(muda::CBufferView<int>        body_id_to_old_body_id,
                          muda::DeviceBuffer<int>&      body_id_to_is_fixed,
                          muda::DeviceBuffer<Vector12>& q,
                          muda::DeviceBuffer<Vector12>& q_temp,
                          muda::DeviceBuffer<Vector12>& q_tilde,
                          muda::DeviceBuffer<Vector12>& q_prev,
                          muda::DeviceBuffer<Vector12>& q_v,
                          muda::DeviceBuffer<Vector12>& dq);
    // if body breakup happens, we need to spawn J
    void _spawn_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                  muda::CBufferView<int> unique_point_to_old_unique_point);


    void _setup_tet_abd_mass(muda::CBufferView<TetLocalInfo> tet_local_info,
                             muda::CBufferView<int> point_id_to_unique_point_id,
                             muda::CBufferView<ABDJacobi> jacobi,
                             muda::CBufferView<Float>     tet_volumes,
                             Float                        density,
                             muda::DeviceBuffer<ABDJacobiDyadicMass>& tet_dyadic_mass);

    void _setup_abd_dyadic_mass(size_t affine_body_count,
                                muda::CBufferView<ABDJacobiDyadicMass> tet_dyadic_mass,
                                muda::CBufferView<int> tet_id_to_body_id,
                                muda::DeviceBuffer<ABDJacobiDyadicMass>& abd_dyadic_mass,
                                muda::DeviceBuffer<Matrix12x12>& abd_dyadic_mass_inv);

    void _setup_abd_volume(size_t                     affine_body_count,
                           muda::CBufferView<int>     tet_id_to_body_id,
                           muda::CBufferView<Float>   tet_volumes,
                           muda::DeviceBuffer<Float>& abd_volume);

    void _setup_tet_abd_gravity_force(const Vector3& gravity,
                                      muda::CBufferView<TetLocalInfo> tet_local_info,
                                      muda::CBufferView<int> point_id_to_unique_point_id,
                                      muda::CBufferView<ABDJacobi> jacobi,
                                      muda::CBufferView<Float>     tet_volumes,
                                      Float                        density,
                                      muda::DeviceBuffer<Vector12>& tet_abd_gravity_force);

    void _setup_abd_gravity(muda::CBufferView<Vector12> tet_abd_gravity_force,
                            muda::CBufferView<int>      tet_id_to_body_id,
                            size_t                      affine_body_count,
                            muda::CBufferView<Matrix12x12> abd_dyadic_mass_inv,
                            muda::DeviceBuffer<Vector12>&  abd_gravity);

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
    void cal_x_from_q(ABDSimData& sim_data, muda::BufferView<double3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<double3> move_dir);
    void cal_x_from_q(ABDSimData& sim_data, muda::BufferView<Vector3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<Vector3> move_dir);

    /********************************************************************************/

    /// <summary>
    /// Main API: calculate abd system gradient and hessian.
    /// before calling this, you need to fill the `triplet_vertex_hessian` (barrier + ground barrier)
    /// </summary>
    /// <param name="sim_data"></param>
    /// <param name="vertex_barrier_gradient"></param>
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           muda::CBufferView<double3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           muda::CBufferView<Vector3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           muda::CBufferView<double3> vertex_barrier_gradient,
                                           muda::CBufferView<ContactHessian> abd_contact_hessian);

    void _cal_triplet_vertex_hessian(ABDSimData& sim_data,
                                     muda::CBufferView<ContactHessian> abd_contact_hessian);
    void _cal_abd_body_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          muda::CBufferView<double3> vertex_barrier_gradient);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          muda::CBufferView<Vector3> vertex_barrier_gradient);
    void _setup_abd_system_hessian(ABDSimData& sim_data);
    void _cal_abd_system_preconditioner(ABDSimData& sim_data);

    /********************************************************************************/

    // when doing line search, we need to copy q to q_temp
    void copy_q_to_q_temp(ABDSimData& sim_data);
    void copy_q_to_q_temp(ABDSimData& sim_data, muda::BufferView<double3> vertices_temp);

    // move forward to test the energy
    void step_forward(ABDSimData&                sim_data,
                      muda::BufferView<double3>  vertices,
                      muda::CBufferView<double3> temp_vertices,
                      double                     alpha);
    void step_forward(ABDSimData&                sim_data,
                      muda::BufferView<Vector3>  vertices,
                      muda::CBufferView<Vector3> temp_vertices,
                      double                     alpha);
    // when doing line search, we need calculate abd energy from q
    Float cal_abd_kinetic_energy(ABDSimData& sim_data);
    Float cal_abd_shape_energy(ABDSimData& sim_data);
};
}  // namespace gipc