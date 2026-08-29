#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_all.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
#include <abd_system/abd_energy.h>
#include <gipc/utils/math.h>
#include <gipc/utils/timer.h>
#include "cuda_tools/cuda_tools.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
namespace gipc
{
//template <int ROWS, int COLS>
__device__ inline void write_triplet_cv(Eigen::Matrix3d* triplet_value,
                                        int*             row_ids,
                                        int*             col_ids,
                                        unsigned int*    node_index,
                                        const Eigen::Matrix<double, 12, 12>& input,
                                        const int& offset)
{
    int rown = 4;
    int coln = 4;
    int kk    = 0;
    for(int ii = 0; ii < rown; ii++)
    {
        for(int jj = ii; jj < coln; jj++)
        {
            row_ids[offset + kk]       = node_index[ii];
            col_ids[offset + kk]       = node_index[jj];
            triplet_value[offset + kk] = input.block<3, 3>(ii * 3, jj * 3);
            kk++;
        }
    }
}




template <int ROWS, int COLS>
__device__ inline void write_triplet_cv2(Eigen::Matrix3d* triplet_value,
                                         int*             row_ids,
                                         int*             col_ids,
                                         unsigned int*    node_index_rows,
                                         unsigned int*    node_index_cols,
                                         const Eigen::Matrix<double, ROWS, COLS>& input,
                                         const int& offset)
{
    int rown = ROWS / 3;
    int coln = COLS / 3;
    for(int ii = 0; ii < rown; ii++)
    {
        for(int jj = 0; jj < coln; jj++)
        {
            int kk  = ii * coln + jj;
            int row = node_index_rows[ii];
            int col = node_index_cols[jj];


            if(row <= col)
            {
                row_ids[offset + kk]       = row;
                col_ids[offset + kk]       = col;
                triplet_value[offset + kk] = input.block<3, 3>(ii * 3, jj * 3);
            }
            else
            {
                row_ids[offset + kk]       = col;
                col_ids[offset + kk]       = row;
                triplet_value[offset + kk].setZero();
            }
        }
    }
}


template <typename T>
__global__ void moveMemory_0(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}



__global__ void write_barrier_hessian(//cudatool::TripletMatrixViewer<double, 12> tripletViewer,
                                      Eigen::Matrix3d*        triplet,
                                      int*              rows,
                                      int*              cols,
                                      ABDJacobi*        abd_J,
                                      const int*          body_id,
                                      const BodyBoundaryType* is_fixed,
                                      int               start_input,
                                      int               start_output,
                                      int               number)
{
    int vI = blockIdx.x * blockDim.x + threadIdx.x;
    if(vI >= number)
        return;

    auto H = triplet[vI + start_input];
    auto i = rows[vI + start_input];
    auto j = cols[vI + start_input];
    auto body_id_i = body_id[i];
    auto body_id_j = body_id[j];


    int          offset       = vI * 16 + start_output;
    const unsigned int row_base = static_cast<unsigned int>(body_id_i) * 4U;
    unsigned int index_row[4] = {row_base, row_base + 1U, row_base + 2U, row_base + 3U};

    const unsigned int col_base = static_cast<unsigned int>(body_id_j) * 4U;
    unsigned int index_col[4] = {col_base, col_base + 1U, col_base + 2U, col_base + 3U};

    if(is_fixed[body_id_i] == BodyBoundaryType::Fixed
       || is_fixed[body_id_j] == BodyBoundaryType::Fixed)
    {
        Matrix12x12 zero12 = Matrix12x12::Zero();
        write_triplet_cv2<12, 12>(triplet, rows, cols, index_row, index_col, zero12, offset);
    }
    else
    {
        auto ABD_H = ABDJacobi::JT_H_J(abd_J[i].T(), H, abd_J[j]);
        write_triplet_cv2<12, 12>(triplet, rows, cols, index_row, index_col, ABD_H, offset);
    }

}



__global__ void write_abd_body_hessian(
    Matrix12x12* matrix_input, Matrix3x3* triplet, int* row, int* col, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int          offset   = idx * 10;
    const unsigned int base = static_cast<unsigned int>(idx) * 4U;
    unsigned int index[4] = {base, base + 1U, base + 2U, base + 3U};
    write_triplet_cv(triplet, row, col, index, matrix_input[idx], offset);
}


__device__ __host__ void make_pd(Matrix9x9& mat)
{
    Vector9   eigen_values;
    Matrix9x9 eigen_vectors;
    cudatool::eigen::evd<Float, 9>(mat, eigen_values, eigen_vectors);
    for(int i = 0; i < 9; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}

namespace
{
__global__ void cal_abd_body_gradient_and_hessian_kernel(int                                  n,
                                                         cudatool::CDense1D<BodyBoundaryType> boundary_type,
                                                         cudatool::CDense1D<Vector12>         qs,
                                                         cudatool::CDense1D<Vector12>         q_tildes,
                                                         cudatool::CDense1D<Vector12>         q_prev,
                                                         cudatool::CDense1D<ABDJacobiDyadicMass> Ms,
                                                         cudatool::CDense1D<Float>            volumes,
                                                         cudatool::Dense1D<Vector12>          gradients,
                                                         cudatool::DenseVectorViewer<double>  system_gradient,
                                                         cudatool::Dense1D<Matrix12x12>       body_hessian,
                                                         Float                                kappa,
                                                         Float                                dt,
                                                         Float                                motor_speed,
                                                         Float                                motor_strength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

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
}

__global__ void cal_abd_system_barrier_gradient_double3_kernel(int                                  n,
                                                               cudatool::CDense1D<I32>              unique_point_id_to_body_id,
                                                               cudatool::CDense1D<double3>          gradient,
                                                               cudatool::Dense1D<Vector12>          affine_gradient,
                                                               cudatool::DenseVectorViewer<double>  system_gradient,
                                                               cudatool::CDense1D<BodyBoundaryType> is_fixed,
                                                               cudatool::CDense1D<ABDJacobi>        J)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto  body_id = unique_point_id_to_body_id(i);
    auto& dst     = affine_gradient(body_id);
    auto& g       = gradient(i);

    if(is_fixed(body_id) == BodyBoundaryType::Fixed)
        return;

    //printf("barrier gradient[%d]=%f %f %f\n", i, g.x, g.y, g.z);

    Vector12 G = J(i).T() * Vector3{g.x, g.y, g.z};
    cudatool::eigen::atomic_add(dst, G);
    system_gradient.segment<12>(body_id * 12).atomic_add(G);
}

__global__ void cal_abd_system_barrier_gradient_Vector3_kernel(int                                  n,
                                                               cudatool::CDense1D<I32>              unique_point_id_to_body_id,
                                                               cudatool::CDense1D<Vector3>          gradient,
                                                               cudatool::Dense1D<Vector12>          affine_gradient,
                                                               cudatool::DenseVectorViewer<double>  system_gradient,
                                                               cudatool::CDense1D<ABDJacobi>        J,
                                                               cudatool::CDense1D<BodyBoundaryType> is_fixed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto  body_id = unique_point_id_to_body_id(i);
    auto& dst     = affine_gradient(body_id);
    auto& g       = gradient(i);


    if(is_fixed(body_id) == BodyBoundaryType::Fixed)
        return;

    Vector12 G = J(i).T() * g;

    cudatool::eigen::atomic_add(dst, G);
    system_gradient.segment<12>(body_id * 12).atomic_add(G);
}

__global__ void setup_abd_system_hessian_abd_fem_kernel(int                                  n,
                                                        cudatool::CDense1D<I32>              point_to_body,
                                                        cudatool::CDense1D<int>              btype,
                                                        cudatool::CDense1D<ABDJacobi>        Js,
                                                        Eigen::Matrix3d*                     triplet_out,
                                                        int*                                 row_out,
                                                        int*                                 col_out,
                                                        const Eigen::Matrix3d*               abd_fem_contact,
                                                        const int*                           abd_fem_rows,
                                                        const int*                           abd_fem_cols,
                                                        int                                  h_abd_fem_contact_start_id,
                                                        cudatool::CDense1D<BodyBoundaryType> is_fixed,
                                                        int                                  abd_body_count,
                                                        int                                  fem_point_offset)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;

    // 1. process upper : ABD-FEM
    {
        auto abd_fem_H3x3 = abd_fem_contact[I];
        auto i_abd = abd_fem_rows[I];  // global point id
        auto j_fem = abd_fem_cols[I];  // global point id

        auto local_fem_point_id = j_fem - fem_point_offset;

        auto body_id = point_to_body(i_abd);  // global body id

        auto local_abd_body_id = body_id;  // - abd_body_offset;
        auto local_abd_point_id = i_abd;  // - abd_point_offset;
        //tex: $\mathbf{J}_{3\times 12}$
        gipc::ABDJacobi J = Js(local_abd_point_id);
        gipc::Matrix12x3 H = J.to_mat().transpose() * abd_fem_H3x3;
        //tex:
        //$$
        // \mathbf{H} = \begin{bmatrix}
        //  \mathbf{H}_{1} \\ \mathbf{H}_{2} \\ \mathbf{H}_{3} \\ \mathbf{H}_{4}
        //\end{bmatrix}
        //$$
        auto offset = 4 * I;
        if(btype(local_fem_point_id) != 0
           || is_fixed(body_id) == BodyBoundaryType::Fixed)
        {
            H.setZero();
        }
        for(int i = 0; i < 4; ++i)
        {
            triplet_out[h_abd_fem_contact_start_id + I * 4 + i] =
                H.block<3, 3>(i * 3, 0);
            row_out[h_abd_fem_contact_start_id + I * 4 + i] =
                body_id * 4 + i;

            col_out[h_abd_fem_contact_start_id + I * 4 + i] =
                abd_body_count * 4 + local_fem_point_id;
        }
    }
}

__global__ void cal_abd_system_preconditioner_kernel1(int                            n,
                                                      cudatool::Dense1D<Matrix12x12> P,
                                                      const Eigen::Matrix3d*         triplet,
                                                      const int*                     rows,
                                                      const int*                     cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto row = rows[i];
    auto H   = triplet[i];
    auto col = cols[i];
    //auto&& [row, col, H] = bcoo(i);
    if(row / 4 == col / 4)
    {
        P(row / 4).block<3, 3>((row % 4) * 3, (col % 4) * 3) = H;
        if(row != col)
        {
            P(row / 4).block<3, 3>((col % 4) * 3, (row % 4) * 3) =
                H.transpose();
        }
    }
}

__global__ void cal_abd_system_preconditioner_kernel2(int                            n,
                                                      cudatool::Dense1D<Matrix12x12> P)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    auto H = P(i);
    P(i)   = cudatool::eigen::inverse(H);
}
}  // namespace

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  cudatool::CBufferView<double3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  cudatool::CBufferView<Vector3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  int*        fbtype,
                                                  cudatool::CBufferView<double3> vertex_barrier_gradient,
                                                  GIPCTripletMatrix& global_triplets)
{
    fem_boundary_type = fbtype;
    setup_abd_system_gradient_hessian(sim_data, global_triplets, vertex_barrier_gradient);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    converter3x3.convert(global_triplets,
                         global_triplets.h_abd_abd_contact_start_id,
                         global_triplets.abd_abd_contact_num,
                         global_triplets.global_collision_triplet_offset);
    global_triplets.global_collision_triplet_offset =
        global_triplets.global_collision_triplet_offset
        - global_triplets.abd_abd_contact_num + global_triplets.h_unique_key_number;
    global_triplets.global_triplet_offset = global_triplets.global_collision_triplet_offset;
    global_triplets.abd_abd_contact_num = global_triplets.h_unique_key_number;
}


// file local function, make the matrix positive definite


void ABDSystem::_cal_abd_body_gradient_and_hessian(ABDSimData& sim_data)
{
    gipc::Timer timer("_cal_abd_body_gradient_and_hessian");
    using namespace cudatool;
    auto& abd       = sim_data.device;
    auto  N         = sim_data.abd_fem_count_info().abd_body_num;
    auto  parameter = parms;
    abd_body_hessian.resize(N);
    abd_gradient.resize(N);
    system_gradient.resize(N * 12);

    auto boundary_type = sim_data.body_id_to_boundary_type();

    LaunchCudaKernal_default((int)N,
                             256,
                             0,
                             cal_abd_body_gradient_and_hessian_kernel,
                            (int)N,
                             boundary_type.cviewer().name("btype"),
                             abd.body_id_to_q.cviewer().name("q"),
                             abd.body_id_to_q_tilde.cviewer().name("affine_q_tilde"),
                             abd.body_id_to_q_prev.cviewer().name("q_prev"),
                             abd.body_id_to_abd_mass.cviewer().name("M"),
                             abd.body_id_to_volume.cviewer().name("volumes"),
                             abd_gradient.viewer().name("abd_gradient"),
                             system_gradient.viewer().name("system_gradient"),
                             abd_body_hessian.viewer().name("shape_hessian"),
                             parameter.kappa,
                             parameter.dt,
                             parms.motor_speed,
                             parms.motor_strength);
}


void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 cudatool::CBufferView<double3> vertex_barrier_gradient)
{
    gipc::Timer timer("_cal_abd_system_barrier_gradient");
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    LaunchCudaKernal_default((int)vertex_barrier_gradient.size(),
                             256,
                             0,
                             cal_abd_system_barrier_gradient_double3_kernel,
                            (int)vertex_barrier_gradient.size(),
                             body_id.cviewer().name("unique_point_id_to_body_id"),
                             vertex_barrier_gradient.cviewer().name("gradient"),
                             abd_gradient.viewer().name("abd_gradient"),
                             system_gradient.viewer().name("system_gradient"),
                             body_id_is_fixed.cviewer().name("is_fixed"),
                             abd.unique_point_id_to_J.cviewer().name("J"));
}

void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 cudatool::CBufferView<Vector3> vertex_barrier_gradient)
{
    using namespace cudatool;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    LaunchCudaKernal_default((int)vertex_barrier_gradient.size(),
                             256,
                             0,
                             cal_abd_system_barrier_gradient_Vector3_kernel,
                            (int)vertex_barrier_gradient.size(),
                             body_id.cviewer().name("unique_point_id_to_body_id"),
                             vertex_barrier_gradient.cviewer().name("gradient"),
                             abd_gradient.viewer().name("abd_gradient"),
                             system_gradient.viewer().name("system_gradient"),
                             abd.unique_point_id_to_J.cviewer().name("J"),
                             body_id_is_fixed.cviewer().name("is_fixed"));
}

void ABDSystem::_setup_abd_system_hessian(ABDSimData& sim_data,
                                          GIPCTripletMatrix& global_triplets)
{
    gipc::Timer timer("_setup_abd_system_hessian");
    using namespace cudatool;

    if(global_triplets.abd_abd_contact_num)
    {
        converter3x3.convert(global_triplets,
                             global_triplets.h_abd_abd_contact_start_id,
                             global_triplets.abd_abd_contact_num,
                             global_triplets.global_collision_triplet_offset);
    }
    else
    {
        global_triplets.h_unique_key_number = 0;
    }

    int bcooNum = global_triplets.h_unique_key_number;

    auto abd_body_count   = sim_data.abd_fem_count_info().abd_body_num;
    auto body_id_is_fixed = sim_data.body_id_to_boundary_type();

    auto unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto body_hessian_size          = abd_body_count;

    global_triplets.abd_abd_contact_num = bcooNum;
    size_t planned_triplet_offset =
        static_cast<size_t>(global_triplets.fem_fem_contact_num)
        + static_cast<size_t>(global_triplets.abd_fem_contact_num) * 4
        + static_cast<size_t>(global_triplets.abd_abd_contact_num) * 16
        + static_cast<size_t>(abd_body_count) * 10;
    if(planned_triplet_offset > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        std::cerr << "ABD Hessian triplet count exceeds the 32-bit offset range."
                  << std::endl;
        std::abort();
    }
    int new_triplet_offset = static_cast<int>(planned_triplet_offset);

    // The ABD assembly stages the re-organized triplets in the region right
    // after the live one and compacts them back afterwards, so writes reach
    // up to 2 * new_triplet_offset. Ensure capacity before writing.
    global_triplets.ensure_triplet_capacity(2 * planned_triplet_offset);

    int h_abd_fem_contact_start_id = global_triplets.fem_fem_contact_num;
    int h_abd_abd_contact_start_id =
        h_abd_fem_contact_start_id + global_triplets.abd_fem_contact_num * 4;


    int write_offset = 0;


    int number    = body_hessian_size;
    int threadNum = 256;
    //int blockNum  = (number + threadNum - 1) / threadNum;

    LaunchCudaKernal_default(
        (int)body_hessian_size,
        threadNum,
        0,
        write_abd_body_hessian,
        abd_body_hessian.viewer().data(),
        global_triplets.block_values(h_abd_abd_contact_start_id + new_triplet_offset + write_offset),
        global_triplets.block_row_indices(h_abd_abd_contact_start_id
                                          + new_triplet_offset + write_offset),
        global_triplets.block_col_indices(h_abd_abd_contact_start_id
                                          + new_triplet_offset + write_offset),
        (int)body_hessian_size);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(bcooNum)
    {
        {

        gipc::Timer timer("barrier_hessian");
        auto abd_J = sim_data.device.unique_point_id_to_J.viewer().data();
        auto is_fixed     = body_id_is_fixed.viewer().data();
        auto my_triplet   = global_triplets.block_values();
        auto my_rows      = global_triplets.block_row_indices();
        auto my_cols      = global_triplets.block_col_indices();
        auto body_id      = unique_point_id_to_body_id.viewer().data();
        int  start_input  = global_triplets.h_abd_abd_contact_start_id;
        int  start_output = new_triplet_offset + h_abd_abd_contact_start_id
                           + write_offset + 10 * body_hessian_size;

        LaunchCudaKernal_default((int)bcooNum,
                                 threadNum,
                                 0,
                                 write_barrier_hessian,
                                my_triplet,
                                 my_rows,
                                 my_cols,
                                 abd_J,
                                 body_id,
                                 is_fixed,
                                 start_input,
                                 start_output,
                                 bcooNum);
        }


       
    }

    if(global_triplets.abd_fem_contact_num)
    {
        auto fem_point_offset = sim_data.abd_fem_count_info().abd_point_num;
        auto fem_count        = sim_data.abd_fem_count_info().fem_point_num;
        auto femb_type =
            cudatool::CBufferView<int>(fem_boundary_type, fem_point_offset, fem_count);

        LaunchCudaKernal_default((int)global_triplets.abd_fem_contact_num,
                                 256,
                                 0,
                                 setup_abd_system_hessian_abd_fem_kernel,
                                (int)global_triplets.abd_fem_contact_num,
                                 unique_point_id_to_body_id.cviewer().name("body_id"),
                                 femb_type.cviewer().name("boundary_type"),
                                 sim_data.device.unique_point_id_to_J.cviewer().name("Js"),
                                 global_triplets.block_values() + new_triplet_offset,
                                 global_triplets.block_row_indices() + new_triplet_offset,
                                 global_triplets.block_col_indices() + new_triplet_offset,
                                 global_triplets.block_values()
                                     + global_triplets.h_abd_fem_contact_start_id,
                                 global_triplets.block_row_indices()
                                     + global_triplets.h_abd_fem_contact_start_id,
                                 global_triplets.block_col_indices()
                                     + global_triplets.h_abd_fem_contact_start_id,
                                 h_abd_fem_contact_start_id,
                                 body_id_is_fixed.cviewer().name("is_fixed"),
                                 abd_body_count,
                                 fem_point_offset);
    }

    global_triplets.h_abd_abd_contact_start_id = h_abd_abd_contact_start_id;
    global_triplets.abd_abd_contact_num =
        16 * global_triplets.abd_abd_contact_num + abd_body_count * 10;


    global_triplets.global_collision_triplet_offset = new_triplet_offset;
    global_triplets.global_triplet_offset = global_triplets.global_collision_triplet_offset;


    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_values() + global_triplets.fem_fem_contact_num,
        global_triplets.block_values() + new_triplet_offset + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(Eigen::Matrix3d),
        cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_col_indices() + global_triplets.fem_fem_contact_num,
        global_triplets.block_col_indices() + new_triplet_offset
            + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(int),
        cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_row_indices() + global_triplets.fem_fem_contact_num,
        global_triplets.block_row_indices() + new_triplet_offset
            + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(int),
        cudaMemcpyDeviceToDevice));
}

void ABDSystem::_cal_abd_system_preconditioner(ABDSimData& sim_data)
{
    using namespace cudatool;
    auto& abd                        = sim_data.device;
    auto  unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_hessian_size = sim_data.abd_fem_count_info().abd_body_num;

    abd_system_diag_preconditioner.resize(body_hessian_size);
    //abd_system_diag_preconditioner.fill(Matrix12x12::Zero());
    auto triplet = global_triplet->block_values(global_triplet->h_abd_abd_contact_start_id);
    auto rows = global_triplet->block_row_indices(global_triplet->h_abd_abd_contact_start_id);
    auto cols = global_triplet->block_col_indices(global_triplet->h_abd_abd_contact_start_id);
    {
        LaunchCudaKernal_default((int)global_triplet->abd_abd_contact_num,
                                 256,
                                 0,
                                 cal_abd_system_preconditioner_kernel1,
                                (int)global_triplet->abd_abd_contact_num,
                                 abd_system_diag_preconditioner.viewer().name("P"),
                                 triplet,
                                 rows,
                                 cols);
        int count = sim_data.abd_fem_count_info().abd_body_num;
        LaunchCudaKernal_default((int)count,
                                 256,
                                 0,
                                 cal_abd_system_preconditioner_kernel2,
                                (int)count,
                                 abd_system_diag_preconditioner.viewer().name("P"));
    }
}
}  // namespace gipc
