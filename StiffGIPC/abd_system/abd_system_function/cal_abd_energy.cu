#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_all.h>
#include "cuda_tools/cuda_tools.h"
#include <abd_system/abd_energy.h>
namespace gipc
{
__global__ void cal_abd_kinetic_energy_kernel(int                                  n,
                                              cudatool::BufferView<Float>             kinetic_energies,
                                              cudatool::BufferView<Vector12>          qs,
                                              cudatool::BufferView<Vector12>          q_prev,
                                              cudatool::BufferView<Vector12>          q_tildes,
                                              cudatool::BufferView<ABDJacobiDyadicMass> Ms,
                                              cudatool::CBufferView<BodyBoundaryType> boundary_type,
                                              Float                                   dt,
                                              Float                                   motor_speed,
                                              Float                                   motor_strength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto& K       = kinetic_energies[i];
    auto& q       = qs[i];
    auto& q_tilde = q_tildes[i];
    auto& M       = Ms[i];


    if(boundary_type[i] == BodyBoundaryType::Fixed)
    {
        K = 0.0;
    }
    else
    {
        if(boundary_type[i] == BodyBoundaryType::Free)
        {
            Vector12 dq = q - q_tilde;
            K           = 0.5 * dq.dot(M * dq);
        }

        if(boundary_type[i] == BodyBoundaryType::Motor)
        {
            {
                Vector12 dq = q - q_tilde;
                K           = 0.5 * dq.dot(M * dq);
            }

            Vector3 bar_x0 = Vector3::Zero();
            Vector3 bar_x1 = Vector3::UnitX();
            Vector3 bar_x2 = Vector3::UnitY();
            Vector3 bar_x3 = Vector3::UnitZ();

            auto mat0 = ABDJacobi{bar_x0}.to_mat();
            auto mat1 = ABDJacobi{bar_x1}.to_mat();
            auto mat2 = ABDJacobi{bar_x2}.to_mat();
            auto mat3 = ABDJacobi{bar_x3}.to_mat();

            Matrix12x12 J;
            J.block<3, 12>(0, 0) = mat0;
            J.block<3, 12>(3, 0) = mat1;
            J.block<3, 12>(6, 0) = mat2;
            J.block<3, 12>(9, 0) = mat3;

            Matrix12x12 inv_J = cudatool::eigen::inverse(J);

            auto theta_per_sec = motor_speed;
            auto theta         = theta_per_sec * dt;
            // rotate x2 and x3 around (x0, x1) by theta
            auto R = Eigen::AngleAxisd(theta, Vector3::UnitX());

            Vector3 x2_P = R * bar_x2;
            Vector3 x3_P = R * bar_x3;

            auto mat0_delta = ABDJacobi{Vector3::Zero()}.to_mat();
            auto mat1_delta = ABDJacobi{Vector3::Zero()}.to_mat();
            auto mat2_delta = ABDJacobi{x2_P - bar_x2}.to_mat();
            auto mat3_delta = ABDJacobi{x3_P - bar_x3}.to_mat();

            Matrix12x12 J_delta;
            J_delta.block<3, 12>(0, 0) = mat0_delta;
            J_delta.block<3, 12>(3, 0) = mat1_delta;
            J_delta.block<3, 12>(6, 0) = mat2_delta;
            J_delta.block<3, 12>(9, 0) = mat3_delta;

            //Vector12 q_p = inv_J * J_delta * q_prev(i) + q_prev(i);
            Vector12 q_p = inv_J * J_delta * q_tilde + q_tilde;
            q_p.segment<3>(6).normalize();
            q_p.segment<3>(9).normalize();
            Vector12 dq      = q - q_p;
            dq.segment<3>(0) = Vector3::Zero();
            dq.segment<3>(3) = Vector3::Zero();

            Matrix12x12 PowMass = Matrix12x12::Zero();
            PowMass.block<6, 6>(6, 6) = //1000 * Matrix6x6::Identity();
            motor_strength * Ms[i].to_mat().block<6, 6>(6, 6);

            K += 0.5 * dq.dot(PowMass * dq);
        }
    }
}

__global__ void cal_abd_shape_energy_kernel(int                          n,
                                            cudatool::BufferView<Float>     shape_energies,
                                            cudatool::BufferView<Vector12> qs,
                                            Float                        kappa,
                                            cudatool::BufferView<Float>    volumes,
                                            Float                        dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto& V      = shape_energies[i];
    auto& q      = qs[i];
    auto& volume = volumes[i];

    V = kappa * volume * dt * dt * shape_energy(q);
}

Float ABDSystem::cal_abd_kinetic_energy(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd       = sim_data.device;
    auto  abd_count = sim_data.abd_fem_count_info().abd_body_num;
    m_kinetic_energy_per_affine_body.resize(abd_count);
    auto& abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  boundry_type   = sim_data.body_id_to_boundary_type();
    if(!abd_count)
        return 0;
    LaunchCudaKernal_default((int)abd_count,
                             256,
                             0,
                             cal_abd_kinetic_energy_kernel,
                             (int)abd_count,
                             m_kinetic_energy_per_affine_body.view(),
                             abd.body_id_to_q.view(),
                             abd.body_id_to_q_prev.view(),
                             abd.body_id_to_q_tilde.view(),
                             abd.body_id_to_abd_mass.view(),
                             boundry_type,
                             parms.dt,
                             parms.motor_speed,
                             parms.motor_strength);
    cudatool::DeviceReduce().Sum(m_kinetic_energy_per_affine_body.data(),
                             m_kinetic_energy.data(),
                             abd_count);
    return m_kinetic_energy;
}
Float ABDSystem::cal_abd_shape_energy(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd       = sim_data.device;
    auto  abd_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  kappa     = parms.kappa;
    auto  dt        = parms.dt;
    if(!abd_count)
        return 0;
    m_shape_energy_per_affine_body.resize(abd_count);

    LaunchCudaKernal_default((int)abd_count,
                             256,
                             0,
                             cal_abd_shape_energy_kernel,
                             (int)abd_count,
                             m_shape_energy_per_affine_body.view(),
                             abd.body_id_to_q.view(),
                             kappa,
                             abd.body_id_to_volume.view(),
                             dt);

    cudatool::DeviceReduce().Sum(
        m_shape_energy_per_affine_body.data(), m_shape_energy.data(), abd_count);

    return m_shape_energy;
}
}  // namespace gipc
