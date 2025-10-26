#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <muda/ext/eigen/evd.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
#include <abd_system/abd_energy.h>
#include <gipc/utils/math.h>
#include <gipc/utils/timer.h>

namespace gipc
{
void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  muda::CBufferView<double3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  muda::CBufferView<double3> vertex_barrier_gradient,
                                                  muda::CBufferView<ContactHessian> contact_hessian)
{

    
    _cal_triplet_vertex_hessian(sim_data, contact_hessian);
    setup_abd_system_gradient_hessian(sim_data, vertex_barrier_gradient);

    //if(abd_system_hessian_reserve_size < triplet_hessian.triplet_count())
    //{
    //    abd_system_hessian_reserve_size = triplet_hessian.triplet_count() * 1.5;
    //    bcoo_hessian.reserve_triplets(abd_system_hessian_reserve_size);
    //    triplet_hessian.reserve_triplets(abd_system_hessian_reserve_size);
    //}

    // converter.convert(triplet_hessian, bcoo_hessian);
    linear_system_context.convert(triplet_hessian, bcoo_hessian);
}

// file local function, make the matrix positive definite
__device__ __host__ void make_pd(Matrix9x9& mat)
{
    Vector9   eigen_values;
    Matrix9x9 eigen_vectors;
    muda::eigen::evd(mat, eigen_values, eigen_vectors);
    for(int i = 0; i < 9; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}


void ABDSystem::_cal_triplet_vertex_hessian(ABDSimData& sim_data,
                                            muda::CBufferView<ContactHessian> abd_contact_hessian)
{
    gipc::Timer timer("_cal_triplet_vertex_hessian");
    using namespace muda;

    int abd_vertex_count = sim_data.abd_fem_count_info().abd_point_num;

    //if(triplet_vertex_hessian_reserve_size < abd_contact_hessian.size())
    //{
    //    triplet_vertex_hessian_reserve_size = abd_contact_hessian.size() * 1.5;
    //    bcoo_vertex_hessian.reserve_triplets(triplet_vertex_hessian_reserve_size);
    //    triplet_vertex_hessian.reserve_triplets(triplet_vertex_hessian_reserve_size);
    //}

    triplet_vertex_hessian.resize(
        abd_vertex_count, abd_vertex_count, abd_contact_hessian.size());

    ParallelFor().apply(abd_contact_hessian.size(),
                        [triplet = triplet_vertex_hessian.viewer().name("triplet"),
                         abd_contact_hessian = abd_contact_hessian.cviewer().name(
                             "abd_contact_hessian")] __device__(int i) mutable
                        {
                            auto&& [ij, H] = abd_contact_hessian(i);
                            triplet(i).write(ij(0), ij(1), H);
                        });
}

void ABDSystem::_cal_abd_body_gradient_and_hessian(ABDSimData& sim_data)
{
    gipc::Timer timer("_cal_abd_body_gradient_and_hessian");
    using namespace muda;
    auto& abd       = sim_data.device;
    auto  N         = sim_data.abd_fem_count_info().abd_body_num;
    auto  parameter = parms;
    abd_body_hessian.resize(N);
    abd_gradient.resize(N);
    system_gradient.resize(N * 12);

    auto boundary_type = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(N,
               [boundary_type = boundary_type.cviewer().name("btype"),
                qs            = abd.body_id_to_q.cviewer().name("q"),
                q_tildes = abd.body_id_to_q_tilde.cviewer().name("affine_q_tilde"),
                q_prev    = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                Ms        = abd.body_id_to_abd_mass.cviewer().name("M"),
                volumes   = abd.body_id_to_volume.cviewer().name("volumes"),
                gradients = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                body_hessian = abd_body_hessian.viewer().name("shape_hessian"),
                kappa        = parameter.kappa,
                dt           = parameter.dt,
                motor_speed  = parms.motor_speed,
                motor_strength = parms.motor_strength] __device__(int i) mutable
               {
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       gradients(i) = Vector12::Zero();
                       system_gradient.segment<12>(i * 12).as_eigen().setZero();
                       body_hessian(i) = Ms(i).to_mat();
                       // body_hessian(i) = Matrix12x12::Zero();
                   }
                   else
                   {
                       Matrix12x12 H = Matrix12x12::Zero();
                       Vector12    G = Vector12::Zero();

                       const auto& q       = qs(i);
                       const auto& q_tilde = q_tildes(i);
                       const auto& M       = Ms(i);

                       {  // kinetic energy
                           Vector12 dq               = (q - q_tilde);
                           Vector12 kinetic_gradient = M * dq;
                           H                         = M.to_mat();
                           G                         = kinetic_gradient;
                       }

                       {  // shape energy
                           const auto& volume = volumes(i);
                           auto        kvt2   = kappa * volume * dt * dt;
                           Vector9 shape_gradient = kvt2 * shape_energy_gradient(q);

                           Matrix9x9 shape_H = kvt2 * shape_energy_hessian(q);

                           // make H positive definite
                           make_pd(shape_H);
                           H.block<9, 9>(3, 3) += shape_H;
                           G.segment<9>(3) += shape_gradient;
                       }

                       gradients(i)                                   = G;
                       system_gradient.segment<12>(i * 12).as_eigen() = G;
                       body_hessian(i)                                = H;

                       if(boundary_type(i) == BodyBoundaryType::Motor)
                       {
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

                           Matrix12x12 inv_J = eigen::inverse(J);

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

                           // Vector12 q_p = inv_J * J_delta * q_prev(i) + q_prev(i);
                           Vector12 q_p = inv_J * J_delta * q_tilde + q_tilde;
                           q_p.segment<3>(6).normalize();
                           q_p.segment<3>(9).normalize();

                           Vector12 dq      = q - q_p;
                           dq.segment<3>(0) = Vector3::Zero();
                           dq.segment<3>(3) = Vector3::Zero();

                           //printf("motor dq: %f %f %f %f %f %f %f %f %f %f %f %f\n",
                           //       dq(0),
                           //       dq(1),
                           //       dq(2),
                           //       dq(3),
                           //       dq(4),
                           //       dq(5),
                           //       dq(6),
                           //       dq(7),
                           //       dq(8),
                           //       dq(9),
                           //       dq(10),
                           //       dq(11));

                           Matrix12x12 PowMass = Matrix12x12::Zero();
                           PowMass.block<6, 6>(6, 6) =  //1000 * Matrix6x6::Identity();
                               motor_strength * Ms(i).to_mat().block<6, 6>(6, 6);


                           system_gradient.segment<12>(i * 12).as_eigen() += PowMass * dq;
                           gradients(i) += PowMass * dq;

                           // Power Mass
                           body_hessian(i) += PowMass;
                       }
                   }
               });
}


void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 muda::CBufferView<double3> vertex_barrier_gradient)
{
    gipc::Timer timer("_cal_abd_system_barrier_gradient");
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(vertex_barrier_gradient.size(),
               [unique_point_id_to_body_id = body_id.cviewer().name("unique_point_id_to_body_id"),
                gradient = vertex_barrier_gradient.cviewer().name("gradient"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                is_fixed = body_id_is_fixed.cviewer().name("is_fixed"),
                J = abd.unique_point_id_to_J.cviewer().name("J")] __device__(int i) mutable
               {
                   auto  body_id = unique_point_id_to_body_id(i);
                   auto& dst     = affine_gradient(body_id);
                   auto& g       = gradient(i);

                   if(is_fixed(body_id) == BodyBoundaryType::Fixed)
                       return;

                   //printf("barrier gradient[%d]=%f %f %f\n", i, g.x, g.y, g.z);

                   Vector12 G = J(i).T() * Vector3{g.x, g.y, g.z};
                   eigen::atomic_add(dst, G);
                   system_gradient.segment<12>(body_id * 12).atomic_add(G);
               });
}

void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(vertex_barrier_gradient.size(),
               [unique_point_id_to_body_id = body_id.cviewer().name("unique_point_id_to_body_id"),
                gradient = vertex_barrier_gradient.cviewer().name("gradient"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                J = abd.unique_point_id_to_J.cviewer().name("J"),
                is_fixed = body_id_is_fixed.cviewer().name("is_fixed")] __device__(int i) mutable
               {
                   auto  body_id = unique_point_id_to_body_id(i);
                   auto& dst     = affine_gradient(body_id);
                   auto& g       = gradient(i);


                   if(is_fixed(body_id) == BodyBoundaryType::Fixed)
                       return;

                   Vector12 G = J(i).T() * g;

                   eigen::atomic_add(dst, G);
                   system_gradient.segment<12>(body_id * 12).atomic_add(G);
               });
}

void ABDSystem::_setup_abd_system_hessian(ABDSimData& sim_data)
{
    gipc::Timer timer("_setup_abd_system_hessian");
    using namespace muda;

    if(triplet_vertex_hessian.triplet_count())
    {
        // converter3x3.convert(triplet_vertex_hessian, bcoo_vertex_hessian);
        linear_system_context.convert(triplet_vertex_hessian, bcoo_vertex_hessian);
    }
    else
    {
        bcoo_vertex_hessian.clear();
    }

    auto abd_body_count   = sim_data.abd_fem_count_info().abd_body_num;
    auto body_id_is_fixed = sim_data.body_id_to_boundary_type();

    auto& abd                        = sim_data.device;
    auto& triplet                    = triplet_hessian;
    auto  unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_hessian_size          = abd_body_count;

    triplet.reshape(body_hessian_size, body_hessian_size);
    triplet.resize_triplets(body_hessian_size + bcoo_vertex_hessian.triplet_count());
    triplet.block_values().fill(Matrix12x12::Zero());


    {
        gipc::Timer timer("body_hessian");
        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(body_hessian_size,
                   [triplet =
                        triplet.view().subview(0, body_hessian_size).viewer().name("triplet"),
                    abd_hessians = abd_body_hessian.cviewer().name("abd_hessian")] __device__(int i) mutable
                   { triplet(i).write(i, i, abd_hessians(i)); });
    }


    if(bcoo_vertex_hessian.triplet_count())
    {
        gipc::Timer timer("barrier_hessian");
        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(bcoo_vertex_hessian.triplet_count(),  // go through all ground-vertex coollision pairs
                   [triplet = triplet.view().subview(body_hessian_size).viewer().name("triplet"),
                    is_fixed = body_id_is_fixed.cviewer().name("is_fixed"),
                    Js       = abd.unique_point_id_to_J.cviewer().name("Js"),
                    vertex_hessian = bcoo_vertex_hessian.cviewer().name("vertex_hessian"),
                    body_id = unique_point_id_to_body_id.cviewer().name(
                        "body_id")] __device__(int vI) mutable
                   {
                       auto&& [i, j, H] = vertex_hessian(vI);

                       auto body_id_i = body_id(i);
                       auto body_id_j = body_id(j);

                       if(is_fixed(body_id_i) == BodyBoundaryType::Fixed
                          || is_fixed(body_id_j) == BodyBoundaryType::Fixed)
                       {
                           triplet(vI).write(body_id_i, body_id_j, Matrix12x12::Zero());
                       }
                       else
                       {
                           auto ABD_H = ABDJacobi::JT_H_J(Js(i).T(), H, Js(j));
                           triplet(vI).write(body_id_i, body_id_j, ABD_H);
                       }
                   });
    }
}

void ABDSystem::_cal_abd_system_preconditioner(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd                        = sim_data.device;
    auto  unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_hessian_size = sim_data.abd_fem_count_info().abd_body_num;

    abd_system_diag_preconditioner.resize(body_hessian_size);
    abd_system_diag_preconditioner.fill(Matrix12x12::Zero());

    {
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(bcoo_hessian.triplet_count(),
                   [P = abd_system_diag_preconditioner.viewer().name("P"),
                    bcoo = bcoo_hessian.cviewer().name("abd_hessian")] __device__(int i) mutable
                   {
                       auto&& [row, col, H] = bcoo(i);
                       if(row == col)
                           P(row) = inverse(H);
                   });
    }
}
}  // namespace gipc