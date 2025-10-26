#include <abd_system/abd_system.h>
#include <muda/launch/parallel_for.h>
#include <gipc/utils/parallel_algorithm/transform.h>
#include <gipc/utils/parallel_algorithm/scatter.h>
#include <muda/atomic.h>
#include <Eigen/Dense>
#include <gipc/utils/print_buffer.h>
#include <gipc/utils/math.h>
#include <abd_system/abd_sim_data.h>
#include <muda/ext/eigen/atomic.h>
#include <muda/ext/eigen/svd.h>
#include <abd_system/abd_energy.h>
#include <gipc/tet_local_info.h>
#include <muda/ext/eigen.h>
#include <gipc/utils/host_log.h>

namespace gipc
{
void ABDSystem::init_system(ABDSimData& sim_data)
{
    _setup_system(true, sim_data);

    m_suggest_max_tolerance = parms.gravity.norm() * parms.dt * parms.dt;
}

void ABDSystem::rebuild_system(ABDSimData& sim_data)
{
    _setup_system(false, sim_data);
}

void ABDSystem::rebuild_system(ABDSimData& sim_data, muda::CBufferView<double3> vertices)
{
    using namespace muda::parallel;

    //Transform()
    //    .kernel_name(__FUNCTION__)
    //    .transform(sim_data.unique_point_id_to_position(),
    //               vertices,
    //               [] __device__(const double3& v) -> Vector3 {
    //                   return Vector3{v.x, v.y, v.z};
    //               });

    _setup_system(false, sim_data);
}

void ABDSystem::_setup_system(bool init, ABDSimData& data)
{
    auto& abd = data.device;

    auto abd_tet_offset      = data.abd_fem_count_info().abd_tet_offset;
    auto abd_tet_count       = data.abd_fem_count_info().abd_tet_num;
    auto abd_sep_point_count = 4 * abd_tet_count;

    auto abd_tets                        = data.tet_info();
    auto abd_tet_volumes                 = data.tet_id_to_volume();
    auto abd_point_id_to_unique_point_id = data.point_id_to_unique_point_id();


    _setup_unique_point_mass(data.abd_fem_count_info().abd_point_num,
                             abd.unique_point_id_to_mass,
                             abd_tets,
                             abd_tet_volumes,
                             parms.mass_density,
                             abd_point_id_to_unique_point_id);

    auto abd_body_count         = data.abd_fem_count_info().abd_body_num;
    auto abd_unique_point_count = data.abd_fem_count_info().abd_point_num;

    auto abd_unique_point_position      = data.unique_point_id_to_position();
    auto abd_unique_point_id_to_body_id = data.unique_point_id_to_body_id();

    _calculate_body_mass_center(abd_body_count,
                                abd.unique_point_id_to_mass,
                                abd_unique_point_position,
                                abd_unique_point_id_to_body_id);

    if(init)
    {
        // setup the abd state
        _setup_abd_state(abd_body_count,
                         abd.body_id_to_q,
                         abd.body_id_to_q_temp,
                         abd.body_id_to_q_tilde,
                         abd.body_id_to_q_prev,
                         abd.body_id_to_q_v,
                         abd.body_id_to_dq);
    }
    else  // rebuild
    {
        //_spawn_abd_state(abd.body_id_to_body_id_old,
        //                 abd.body_id_to_is_fixed,
        //                 abd.body_id_to_q,
        //                 abd.body_id_to_q_temp,
        //                 abd.body_id_to_q_tilde,
        //                 abd.body_id_to_q_prev,
        //                 abd.body_id_to_q_v,
        //                 abd.body_id_to_dq);

        //exp3_14
        MUDA_ERROR_WITH_LOCATION("In this version, we do not support the rebuild of ABDSystem!");
    }

    // always rebuild J, if body is broken
    _setup_J(abd.unique_point_id_to_J,
             abd_unique_point_position,
             abd_unique_point_id_to_body_id,
             abd.body_id_to_q);

    _setup_tet_abd_mass(abd_tets,
                        abd_point_id_to_unique_point_id,
                        abd.unique_point_id_to_J,
                        abd_tet_volumes,
                        parms.mass_density,
                        abd.tet_id_to_abd_mass);

    auto abd_tet_id_to_body_id = data.tet_id_to_body_id();

    _setup_abd_dyadic_mass(abd_body_count,
                           abd.tet_id_to_abd_mass,
                           abd_tet_id_to_body_id,
                           abd.body_id_to_abd_mass,
                           abd.body_id_to_abd_mass_inv);

    _setup_abd_volume(abd_body_count, abd_tet_id_to_body_id, abd_tet_volumes, abd.body_id_to_volume);

    _setup_tet_abd_gravity_force(parms.gravity,
                                 abd_tets,
                                 abd_point_id_to_unique_point_id,
                                 abd.unique_point_id_to_J,
                                 abd_tet_volumes,
                                 parms.mass_density,
                                 abd.tet_id_to_abd_gravity_force);

    _setup_abd_gravity(abd.tet_id_to_abd_gravity_force,
                       abd_tet_id_to_body_id,
                       abd_body_count,
                       abd.body_id_to_abd_mass_inv,
                       abd.body_id_to_abd_gravity);
}

void ABDSystem::_setup_unique_point_mass(size_t unique_point_count,
                                         muda::DeviceBuffer<Float>& unique_point_mass,
                                         muda::CBufferView<TetLocalInfo> tets,
                                         muda::CBufferView<Float> tet_volumes,
                                         Float                    density,
                                         muda::CBufferView<int> point_id_to_unique_point_id)
{
    using namespace muda;

    unique_point_mass.resize(unique_point_count, 0);
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(tets.size(),
               [unique_point_mass = unique_point_mass.viewer().name("unique_point_mass"),
                tets        = tets.viewer().name("tets"),
                tet_volumes = tet_volumes.viewer().name("tet_volumes"),
                point_id_to_unique_point_id =
                    point_id_to_unique_point_id.viewer().name("point_id_to_unique_point_id"),
                density = density] __device__(int i) mutable
               {
                   auto mass       = tet_volumes(i) * density;
                   auto tet_points = tets(i).tet_point_ids();
                   for(int j = 0; j < 4; ++j)
                   {
                       auto point_id = tet_points(j);
                       auto unique_point_id = point_id_to_unique_point_id(point_id);
                       atomic_add(&unique_point_mass(unique_point_id), mass / 4);
                   }
               });
}

void ABDSystem::_calculate_body_mass_center(size_t body_count,
                                            muda::DeviceBuffer<Float>& unique_point_mass,
                                            muda::CBufferView<double3> unique_point_position,
                                            muda::CBufferView<int> unique_point_id_to_body_id)
{
    using namespace muda;
    using namespace muda::parallel;
    body_mass.resize(body_count, 0);
    body_mass_center.resize(body_count, Vector3::Zero());

    // TODO: maybe we can use a parallel reduce here
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(unique_point_position.size(),
               [unique_point_mass = unique_point_mass.viewer().name("unique_point_mass"),
                unique_point_position = unique_point_position.viewer().name("unique_point_position"),
                unique_point_id_to_body_id =
                    unique_point_id_to_body_id.viewer().name("unique_point_id_to_body_id"),
                body_mass        = body_mass.viewer().name("body_mass"),
                body_mass_center = body_mass_center.viewer().name(
                    "body_mass_center")] __device__(int i) mutable
               {
                   auto    mass     = unique_point_mass(i);
                   auto    pos      = unique_point_position(i);
                   auto    body_id  = unique_point_id_to_body_id(i);
                   Vector3 mass_pos = mass * eigen::as_eigen(pos);
                   atomic_add(&body_mass(body_id), mass);
                   eigen::atomic_add(body_mass_center(body_id), mass_pos);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(body_count,
               [body_mass_center = body_mass_center.viewer().name("body_mass_center"),
                body_mass = body_mass.viewer().name("body_mass")] __device__(int i) mutable
               { body_mass_center(i) /= body_mass(i); });

    m_body_centered_positions.resize(unique_point_position.size());

    Transform().transform(
        m_body_centered_positions.view(),
        [body_mass_center = body_mass_center.viewer().name("body_mass_center"),
         unique_point_position = unique_point_position.viewer().name("unique_point_position"),
         unique_point_id_to_body_id = unique_point_id_to_body_id.viewer().name(
             "unique_point_id_to_body_id")] __device__(int i) mutable -> Vector3
        {
            auto body_id = unique_point_id_to_body_id(i);
            return eigen::as_eigen(unique_point_position(i)) - body_mass_center(body_id);
        });
}

void ABDSystem::_setup_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                         muda::CBufferView<double3>     unique_point_position,
                         muda::CBufferView<int>      unique_point_id_to_body_id,
                         muda::CBufferView<Vector12> q)
{
    using namespace muda;
    using namespace muda::parallel;
    jacobi.resize(unique_point_id_to_body_id.size());
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(unique_point_id_to_body_id.size(),
               [q = q.viewer().name("q"),
                J = jacobi.viewer().name("J"),
                unique_point_position = unique_point_position.viewer().name("unique_point_position"),
                unique_point_id_to_body_id = unique_point_id_to_body_id.viewer().name(
                    "unique_point_id_to_body_id")] __device__(int i) mutable
               {
                   auto    body_id = unique_point_id_to_body_id(i);
                   Vector3 pos     = eigen::as_eigen(unique_point_position(i));
                   auto    q_i     = q(body_id);

                   auto A = q_to_A(q_i);
                   auto p = q_i.segment<3>(0);

                   Vector3 pos0 = muda::eigen::inverse(A) * (pos - p);

                   J(i) = ABDJacobi{pos0};
               });
}

void ABDSystem::_setup_abd_state(size_t                        abd_count,
                                 muda::DeviceBuffer<Vector12>& q,
                                 muda::DeviceBuffer<Vector12>& q_temp,
                                 muda::DeviceBuffer<Vector12>& q_tilde,
                                 muda::DeviceBuffer<Vector12>& q_prev,
                                 muda::DeviceBuffer<Vector12>& q_v,
                                 muda::DeviceBuffer<Vector12>& dq)
{
    using namespace muda;
    using namespace muda::parallel;

    q.resize(abd_count, Vector12::Zero());
    q_v.resize(abd_count, Vector12::Zero());
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_count,
               [q   = q.viewer().name("q"),
                q_v = q_v.viewer().name("q_v"),
                body_mass_center = body_mass_center.cviewer().name("body_mass_center"),
                init_q_v = parms.init_q_v] __device__(int i) mutable
               {
                   Vector12 new_q;

                   new_q.segment<3>(0) = body_mass_center(i);

                   // new_q.segment<3>(0) = Vector3{0, 0, 0};
                   new_q.segment<3>(3) = Vector3{1, 0, 0};
                   new_q.segment<3>(6) = Vector3{0, 1, 0};
                   new_q.segment<3>(9) = Vector3{0, 0, 1};

                   Vector12 new_qv = init_q_v;
                   //new_qv.segment<3>(0) = Vector3(0, -100, 0);
                   //new_qv.segment<3>(3) = Vector3{0, 0, 0};
                   //new_qv.segment<3>(6) = Vector3{0, 0, 0};
                   //new_qv.segment<3>(9) = Vector3{0, 0, 0};

                   q(i)   = new_q;
                   q_v(i) = new_qv;
               });


    q_temp  = q;
    q_tilde = q;
    q_prev  = q;

    dq.resize(abd_count, Vector12::Zero());
    //q_v.resize(abd_count, Vector12::Zero());
}


MUDA_GENERIC Eigen::Matrix<double, 9, 9> compute_DRDF(const Matrix3x3& F)
{
    Eigen::Matrix<double, 9, 1> g1;
    g1.block<3, 1>(0, 0) = 2 * F.col(0);
    g1.block<3, 1>(3, 0) = 2 * F.col(1);
    g1.block<3, 1>(6, 0) = 2 * F.col(2);
    Eigen::Matrix<double, 9, 9> H1 = 2 * Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix3d             mat_g2 = 4 * F * F.transpose() * F;
    Eigen::Matrix<double, 9, 1> g2;
    g2.block<3, 1>(0, 0) = mat_g2.col(0);
    g2.block<3, 1>(3, 0) = mat_g2.col(1);
    g2.block<3, 1>(6, 0) = mat_g2.col(2);
    Eigen::Matrix<double, 9, 9> D, IkronFFT, FTFkronI;
    Eigen::Matrix3d             FFT       = F * F.transpose();
    Eigen::Matrix3d             FTF       = F.transpose() * F;
    Eigen::Matrix3d             Identity3 = Eigen::Matrix3d::Identity();
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            D.block<3, 3>(3 * i, 3 * j) = F.col(j) * F.col(i).transpose();
            IkronFFT.block<3, 3>(3 * i, 3 * j) = Identity3(i, j) * FFT;
            FTFkronI.block<3, 3>(3 * i, 3 * j) = FTF(i, j) * Identity3;
        }
    }
    Eigen::Matrix<double, 9, 9> H2 = 4 * (IkronFFT + FTFkronI + D);
    double                      J  = F.determinant();
    Eigen::Matrix<double, 9, 1> gJ;
    gJ.block<3, 1>(0, 0) = F.col(1).cross(F.col(2));
    gJ.block<3, 1>(3, 0) = F.col(2).cross(F.col(0));
    gJ.block<3, 1>(6, 0) = F.col(0).cross(F.col(1));
    Eigen::Matrix3d f0hat, f1hat, f2hat;
    f0hat << 0, -F(2, 0), F(1, 0), F(2, 0), 0, -F(0, 0), -F(1, 0), F(0, 0), 0;
    f1hat << 0, -F(2, 1), F(1, 1), F(2, 1), 0, -F(0, 1), -F(1, 1), F(0, 1), 0;
    f2hat << 0, -F(2, 2), F(1, 2), F(2, 2), 0, -F(0, 2), -F(1, 2), F(0, 2), 0;

    Eigen::Matrix<double, 9, 9> HJ;
    HJ.block<3, 3>(0, 0)           = Eigen::Matrix3d::Zero();
    HJ.block<3, 3>(0, 3)           = -f2hat;
    HJ.block<3, 3>(0, 6)           = f1hat;
    HJ.block<3, 3>(3, 0)           = f2hat;
    HJ.block<3, 3>(3, 3)           = Eigen::Matrix3d::Zero();
    HJ.block<3, 3>(3, 6)           = -f0hat;
    HJ.block<3, 3>(6, 0)           = -f1hat;
    HJ.block<3, 3>(6, 3)           = f0hat;
    HJ.block<3, 3>(6, 6)           = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 9, 1> g3 = 2 * J * gJ;
    Eigen::Matrix<double, 9, 9> H3 = 2 * gJ * gJ.transpose() + 2 * J * HJ;
    double                      i1 = F.squaredNorm();
    double                      i2 = (F.transpose() * F).squaredNorm();
    double                      i3 = (F.transpose() * F).determinant();

    double a = 0;
    double b = -2 * i1;
    double c = -8 * J;
    double d = i1 * i1 - 2 * (i1 * i1 - i2);


    Matrix3x3 U, V;
    Vector3   sig;
    muda::eigen::svd(F, U, sig, V);

    double f = sig.sum();

    double f1  = (2 * f * f + 2 * i1) / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f2  = -2 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f3  = (8 * f) / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f11 = (4 * f * f1 + 2 - (12 * f * f * f1 - 4 * f - 4 * i1 * f1) * f1)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f12 = (4 * f * f2 - (12 * f * f * f2 - 4 * i1 * f2) * f1)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f13 = (4 * f * f3 - (12 * f * f * f3 - 4 * i1 * f3 - 8) * f1)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f21 = -(12 * f * f * f1 - 4 * f - 4 * i1 * f1) * f2
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f22 = -(12 * f * f * f2 - 4 * i1 * f2) * f2
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f23 = -(12 * f * f * f3 - 4 * i1 * f3 - 8) * f2
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f31 = (8 * f1 - (12 * f * f * f1 - 4 * f - 4 * i1 * f1) * f3)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f32 = (8 * f2 - (12 * f * f * f2 - 4 * i1 * f2) * f3)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);
    double f33 = (8 * f3 - (12 * f * f * f3 - 4 * i1 * f3 - 8) * f3)
                 / (4 * pow(f, 3) - 4 * i1 * f - 8 * J);

    Eigen::Matrix<double, 9, 9> H =
        -0.5
        * ((-2 * f1) * H1 + (-2 * f2) * H2 + (-2 * f3) * HJ + (-2 * f11) * g1 * g1.transpose()
           + (-2 * f22) * g2 * g2.transpose() + (-2 * f33) * gJ * gJ.transpose()
           + (-2 * f12) * g2 * g1.transpose() + (-2 * f13) * gJ * g1.transpose()
           + (-2 * f21) * g1 * g2.transpose() + (-2 * f23) * gJ * g2.transpose()
           + (-2 * f31) * g1 * gJ.transpose() + (-2 * f32) * g2 * gJ.transpose());

    return H;
}

MUDA_GENERIC Vector9 flatten(const Matrix3x3& A) noexcept
{
    //tex:
    //$
    //\left[\begin{matrix}f_{00} & f_{01} & f_{02}\\f_{10} & f_{11} & f_{12}\\f_{20} & f_{21} & f_{22}\end{matrix}\right]
    //\rightarrow
    //\left[\begin{matrix}f_{00}\\f_{01}\\f_{02}\\f_{10}\\f_{11}\\f_{12}\\f_{20}\\f_{21}\\f_{22}\end{matrix}\right]
    //$
    Vector9 column;

    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            column[index] = A(i, j);

    return column;
}

MUDA_GENERIC Matrix3x3 unflatten(const Vector9& v) noexcept
{
    //tex:
    //$
    //\left[\begin{matrix}f_{00}\\f_{01}\\f_{02}\\f_{10}\\f_{11}\\f_{12}\\f_{20}\\f_{21}\\f_{22}\end{matrix}\right]
    //\rightarrow
    //\left[\begin{matrix}f_{00} & f_{01} & f_{02}\\f_{10} & f_{11} & f_{12}\\f_{20} & f_{21} & f_{22}\end{matrix}\right]
    //$
    Matrix3x3    A;
    unsigned int index = 0;
    for(unsigned int j = 0; j < A.cols(); j++)
        for(unsigned int i = 0; i < A.rows(); i++, index++)
            A(i, j) = v[index];

    return A;
}

MUDA_GENERIC Matrix3x3 ddot(const Matrix9x9& DRDF, const Matrix3x3& F_prime)
{
    auto flatten_F = flatten(F_prime);
    flatten_F      = DRDF * flatten_F;
    return unflatten(flatten_F);
}


void ABDSystem::_spawn_abd_state(muda::CBufferView<int> body_id_to_old_body_id,
                                 muda::DeviceBuffer<int>& body_id_to_is_fixed,
                                 muda::DeviceBuffer<Vector12>& q,
                                 muda::DeviceBuffer<Vector12>& q_temp,
                                 muda::DeviceBuffer<Vector12>& q_tilde,
                                 muda::DeviceBuffer<Vector12>& q_prev,
                                 muda::DeviceBuffer<Vector12>& q_v,
                                 muda::DeviceBuffer<Vector12>& dq)
{
    using namespace muda;
    using namespace muda::parallel;
    auto new_abd_count = body_id_to_old_body_id.size();
    // spawn the q, because the q will be used in integration in next frame
    // spawn the q_v, because the q_v will be used in integration in next frame
    m_temp_q.resize(new_abd_count);
    m_temp_q_v.resize(new_abd_count);
    m_temp_q_prev.resize(new_abd_count);
    m_temp_is_fixed.resize(new_abd_count);
    m_temp_q_tilde.resize(new_abd_count);


    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(new_abd_count,
               [body_id_to_old_body_id = body_id_to_old_body_id.viewer().name("body_id_to_old_body_id"),
                q             = q.cviewer().name("q"),
                temp_q        = m_temp_q.viewer().name("temp_q"),
                q_v           = q_v.cviewer().name("q_v"),
                temp_q_v      = m_temp_q_v.viewer().name("temp_q_v"),
                q_prev        = q_prev.cviewer().name("q_prev"),
                temp_q_prev   = m_temp_q_prev.viewer().name("m_temp_q_prev"),
                q_tilde       = q_tilde.cviewer().name("q_tilde"),
                temp_q_tilde  = m_temp_q_tilde.viewer().name("m_temp_q_tilde"),
                is_fixed      = body_id_to_is_fixed.cviewer().name("is_fixed"),
                temp_is_fixed = m_temp_is_fixed.viewer().name("temp_is_fixed"),
                body_mass_center = body_mass_center.cviewer().name(
                    "body_mass_center")] __device__(int i) mutable
               {
                   auto old_body_id = body_id_to_old_body_id(i);

                   temp_q(i)        = q(old_body_id);
                   temp_q_v(i)      = q_v(old_body_id);
                   temp_q_prev(i)   = q_prev(old_body_id);
                   temp_q_tilde(i)  = q_tilde(old_body_id);
                   temp_is_fixed(i) = is_fixed(old_body_id);
               });

    // swap q and temp_q
    std::swap(q, m_temp_q);
    // swap q_v and temp_q_v
    std::swap(q_v, m_temp_q_v);
    // swap q_prev and temp_q_prev
    std::swap(q_prev, m_temp_q_prev);
    // swap q_tilde and temp_q_tilde
    std::swap(q_tilde, m_temp_q_tilde);
    // swap is_fixed and temp_is_fixed
    std::swap(body_id_to_is_fixed, m_temp_is_fixed);


    // reset the dq
    dq.clear();
    dq.resize(new_abd_count, Vector12::Zero());
}

void ABDSystem::_spawn_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                         muda::CBufferView<int> unique_point_to_old_unique_point)
{
    using namespace muda;
    using namespace muda::parallel;

    //m_temp_jacobi.resize(unique_point_to_old_unique_point.size());
    //Transform()
    //    .kernel_name(__FUNCTION__)
    //    .transform(m_temp_jacobi.view(),
    //               [unique_point_to_old_unique_point =
    //                    unique_point_to_old_unique_point.viewer().name("unique_point_to_old_unique_point"),
    //                jacobi = jacobi.cviewer().name("jacobi")] __device__(int i)
    //               {
    //                   auto old_unique_point = unique_point_to_old_unique_point(i);
    //                   return jacobi(old_unique_point);
    //               });
    //std::swap(jacobi, m_temp_jacobi);
}

void ABDSystem::_setup_tet_abd_mass(muda::CBufferView<TetLocalInfo> tet_local_info,
                                    muda::CBufferView<int> point_id_to_unique_point_id,
                                    muda::CBufferView<ABDJacobi> jacobi,
                                    muda::CBufferView<Float>     tet_volumes,
                                    Float                        density,
                                    muda::DeviceBuffer<ABDJacobiDyadicMass>& tet_dyadic_mass)
{
    using namespace muda::parallel;
    tet_dyadic_mass.resize(tet_local_info.size());
    Transform()
        .kernel_name(__FUNCTION__)
        .transform(tet_dyadic_mass.view(),
                   [tet_infos = tet_local_info.viewer().name("tet_infos"),
                    point_id_to_unique_point_id =
                        point_id_to_unique_point_id.viewer().name("point_id_to_unique_point_id"),
                    jacobi      = jacobi.viewer().name("jacobi"),
                    tet_volumes = tet_volumes.viewer().name("tet_volumes"),
                    density     = density] __device__(int i)
                   {
                       auto mass      = tet_volumes(i) * density;
                       auto node_mass = mass / 4;
                       ABDJacobiDyadicMass tet_mass = ABDJacobiDyadicMass::zero();
                       Vector4i tet_points = tet_infos(i).tet_point_ids();
                       for(int j = 0; j < 4; ++j)
                       {
                           auto point_id = tet_points(j);
                           auto unique_point_id = point_id_to_unique_point_id(point_id);
                           tet_mass += ABDJacobiDyadicMass{
                               node_mass, jacobi(unique_point_id).x_bar()};
                       }
                       return tet_mass;
                   });
}

void ABDSystem::_setup_abd_dyadic_mass(size_t affine_body_count,
                                       muda::CBufferView<ABDJacobiDyadicMass> tet_dyadic_mass,
                                       muda::CBufferView<int> tet_id_to_body_id,
                                       muda::DeviceBuffer<ABDJacobiDyadicMass>& abd_dyadic_mass,
                                       muda::DeviceBuffer<Matrix12x12>& abd_dyadic_mass_inv)
{
    using namespace muda;
    using namespace muda::parallel;
    abd_dyadic_mass.resize(affine_body_count, ABDJacobiDyadicMass::zero());
    // TODO: maybe we can use a parallel reduce here
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(tet_dyadic_mass.size(),
               [tet_dyadic_mass = tet_dyadic_mass.viewer().name("tet_dyadic_mass"),
                tet_id_to_body_id = tet_id_to_body_id.viewer().name("tet_id_to_body_id"),
                abd_dyadic_mass =
                    abd_dyadic_mass.viewer().name("abd_dyadic_mass")] __device__(int i) mutable
               {
                   auto  body_id = tet_id_to_body_id(i);
                   auto& dst     = abd_dyadic_mass(body_id);
                   auto& src     = tet_dyadic_mass(i);
                   ABDJacobiDyadicMass::atomic_add(dst, src);
               });

    abd_dyadic_mass_inv.resize(affine_body_count);
    Transform()
        .file_line(__FILE__, __LINE__)
        .transform(abd_dyadic_mass_inv.view(),
                   std::as_const(abd_dyadic_mass).view(),
                   [] __device__(const ABDJacobiDyadicMass& mass) -> Matrix12x12
                   {
                       // eigen 12x12 inverse does not work in cuda kernel!!!
                       // return mass.to_mat().inverse();
                       return inverse(mass.to_mat());
                   });
}

void ABDSystem::_setup_abd_volume(size_t                     affine_body_count,
                                  muda::CBufferView<int>     tet_id_to_body_id,
                                  muda::CBufferView<Float>   tet_volumes,
                                  muda::DeviceBuffer<Float>& abd_volume)
{
    using namespace muda;
    using namespace muda::parallel;
    abd_volume.resize(affine_body_count, 0);
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(tet_id_to_body_id.size(),
               [tet_id_to_body_id = tet_id_to_body_id.viewer().name("tet_id_to_body_id"),
                tet_volumes = tet_volumes.viewer().name("tet_volumes"),
                abd_volume = abd_volume.viewer().name("abd_volume")] __device__(int i) mutable
               {
                   auto body_id = tet_id_to_body_id(i);
                   auto volume  = tet_volumes(i);
                   muda::atomic_add(&abd_volume(body_id), volume);
               });
}

void ABDSystem::_setup_tet_abd_gravity_force(const Vector3& gravity,
                                             muda::CBufferView<TetLocalInfo> tet_local_info,
                                             muda::CBufferView<int> point_id_to_unique_point_id,
                                             muda::CBufferView<ABDJacobi> jacobi,
                                             muda::CBufferView<Float> tet_volumes,
                                             Float density,
                                             muda::DeviceBuffer<Vector12>& tet_abd_gravity_force)
{
    using namespace muda::parallel;
    tet_abd_gravity_force.resize(tet_local_info.size());
    Transform()
        .kernel_name(__FUNCTION__)
        .transform(tet_abd_gravity_force.view(),
                   [gravity   = gravity,
                    tet_infos = tet_local_info.viewer().name("tet_infos"),
                    point_id_to_unique_point_id =
                        point_id_to_unique_point_id.viewer().name("point_id_to_unique_point_id"),
                    jacobi      = jacobi.viewer().name("jacobi"),
                    tet_volumes = tet_volumes.viewer().name("tet_volumes"),
                    density     = density] __device__(int i)
                   {
                       auto     mass               = tet_volumes(i) * density;
                       auto     node_mass          = mass / 4;
                       auto     node_gravity_force = node_mass * gravity;
                       Vector12 tet_gravity_force  = Vector12::Zero();
                       Vector4i tet_points = tet_infos(i).tet_point_ids();
                       for(int j = 0; j < 4; ++j)
                       {
                           auto point_id = tet_points(j);
                           auto unique_point_id = point_id_to_unique_point_id(point_id);
                           auto& J = jacobi(unique_point_id);
                           tet_gravity_force += J.T() * node_gravity_force;
                       }
                       return tet_gravity_force;
                   });
}

void ABDSystem::_setup_abd_gravity(muda::CBufferView<Vector12> tet_abd_gravity_force,
                                   muda::CBufferView<int> tet_id_to_body_id,
                                   size_t                 affine_body_count,
                                   muda::CBufferView<Matrix12x12> abd_dyadic_mass_inv,
                                   muda::DeviceBuffer<Vector12>& abd_gravity)
{
    using namespace muda;
    using namespace muda::parallel;
    m_temp_abd_gravity_force.resize(affine_body_count, Vector12::Zero());

    // TODO: maybe we can use a parallel reduce here
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(tet_abd_gravity_force.size(),
               [tet_abd_gravity_force = tet_abd_gravity_force.viewer().name("tet_abd_gravity_force"),
                tet_id_to_body_id = tet_id_to_body_id.viewer().name("tet_id_to_body_id"),

                abd_gravity_force = m_temp_abd_gravity_force.viewer().name(
                    "abd_gravity_force")] __device__(int i) mutable
               {
                   auto     body_id = tet_id_to_body_id(i);
                   auto&    dst     = abd_gravity_force(body_id);
                   Vector12 src     = tet_abd_gravity_force(i);

                   muda::eigen::atomic_add(dst, src);
               });

    abd_gravity.resize(affine_body_count);
    Transform()
        .kernel_name(__FUNCTION__)
        .transform(abd_gravity.view(),
                   [abd_gravity_force = m_temp_abd_gravity_force.viewer().name("abd_gravity_force"),
                    abd_dyadic_mass_inv = abd_dyadic_mass_inv.viewer().name(
                        "abd_dyadic_mass_inv")] __device__(int i) -> Vector12
                   {
                       auto gravity_force = abd_gravity_force(i);
                       auto mass_inv      = abd_dyadic_mass_inv(i);
                       auto gravity_acc   = mass_inv * gravity_force;
                       return gravity_acc;
                   });
}
}  // namespace gipc