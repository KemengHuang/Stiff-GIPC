//
// femEnergy.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "femEnergy.cuh"
#include "math.h"
#include <stdio.h>
#include <unsupported/Eigen/KroneckerProduct>
#define arap_model 3;
template <int ROWS, int COLS>
__device__ inline void write_triplet_fem(Eigen::Matrix3d* triplet_value,
                                         int*             row_ids,
                                         int*             col_ids,
                                         const unsigned int*       node_index,
                                         const double     input[ROWS][COLS],
                                         const int&       offset)
{
    int rown = ROWS / 3;
    int coln = COLS / 3;
    int kk   = 0;
    for(int ii = 0; ii < rown; ii++)
    {
        for(int jj = ii; jj < coln; jj++)
        {
            int row              = node_index[ii];
            int col              = node_index[jj];

            if(row <= col)
            {
                row_ids[offset + kk] = row;
                col_ids[offset + kk] = col;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[ii * 3 + iii][jj * 3 + jjj];
                    }
                }
            }
            else
            {
                row_ids[offset + kk] = col;
                col_ids[offset + kk] = row;
                for(int iii = 0; iii < 3; iii++)
                {
                    for(int jjj = 0; jjj < 3; jjj++)
                    {
                        triplet_value[offset + kk](iii, jjj) =
                            input[jj * 3 + iii][ii * 3 + jjj];
                    }
                }
            }
            kk++;
            
        }
    }
}

using namespace Eigen;
template <typename Scalar, int size>
__device__ void makePDSNK(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0)
    {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for(int i = 0; i < rows; i++)
    {
        if(eigenSolver.eigenvalues()[i] > 0.0)
        {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}


namespace LibShell
{
__device__ __host__ static Eigen::Matrix3d crossMatrix(Eigen::Vector3d v)
{
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;
}

__device__ __host__ static Eigen::Matrix2d adjugate(Eigen::Matrix2d M)
{
    Eigen::Matrix2d ret;
    ret << M(1, 1), -M(0, 1), -M(1, 0), M(0, 0);
    return ret;
}
__device__ __host__ static double angle(const Eigen::Vector3d& v,
                                        const Eigen::Vector3d& w,
                                        const Eigen::Vector3d& axis,
                                        Eigen::Matrix<double, 1, 9>* derivative,  // v, w
                                        Eigen::Matrix<double, 9, 9>* hessian)
{
    double theta =
        2.0 * atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());

    if(derivative)
    {
        derivative->segment<3>(0) = -axis.cross(v) / v.squaredNorm() / axis.norm();
        derivative->segment<3>(3) = axis.cross(w) / w.squaredNorm() / axis.norm();
        derivative->segment<3>(6).setZero();
    }
    if(hessian)
    {
        hessian->setZero();
        hessian->block<3, 3>(0, 0) += 2.0 * (axis.cross(v)) * v.transpose()
                                      / v.squaredNorm() / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += -2.0 * (axis.cross(w)) * w.transpose()
                                      / w.squaredNorm() / w.squaredNorm() / axis.norm();
        hessian->block<3, 3>(0, 0) += -crossMatrix(axis) / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += crossMatrix(axis) / w.squaredNorm() / axis.norm();

        Eigen::Matrix3d dahat =
            (Eigen::Matrix3d::Identity() / axis.norm()
             - axis * axis.transpose() / axis.norm() / axis.norm() / axis.norm());

        hessian->block<3, 3>(0, 6) += crossMatrix(v) * dahat / v.squaredNorm();
        hessian->block<3, 3>(3, 6) += -crossMatrix(w) * dahat / w.squaredNorm();
    }

    return theta;
}
__device__ __host__ static double edgeTheta(const Eigen::Vector3d& q0,
                                            const Eigen::Vector3d& q1,
                                            const Eigen::Vector3d& q2,
                                            const Eigen::Vector3d& q3,
                                            Eigen::Matrix<double, 1, 12>* derivative,  // edgeVertex, then edgeOppositeVertex
                                            Eigen::Matrix<double, 12, 12>* hessian)
{
    if(derivative)
        derivative->setZero();
    if(hessian)
        hessian->setZero();
    //    int v0 = mesh.edgeVertex(edge, 0);
    //    int v1 = mesh.edgeVertex(edge, 1);
    //    int v2 = mesh.edgeOppositeVertex(edge, 0);
    //    int v3 = mesh.edgeOppositeVertex(edge, 1);
    //    if (v2 == -1 || v3 == -1)
    //        return 0; // boundary edge
    //
    //    Eigen::Vector3d q0 = curPos.row(v0);
    //    Eigen::Vector3d q1 = curPos.row(v1);
    //    Eigen::Vector3d q2 = curPos.row(v2);
    //    Eigen::Vector3d q3 = curPos.row(v3);

    Eigen::Vector3d             n0   = (q0 - q2).cross(q1 - q2);
    Eigen::Vector3d             n1   = (q1 - q3).cross(q0 - q3);
    Eigen::Vector3d             axis = q1 - q0;
    Eigen::Matrix<double, 1, 9> angderiv;
    Eigen::Matrix<double, 9, 9> anghess;

    double theta =
        angle(n0, n1, axis, (derivative || hessian) ? &angderiv : NULL, hessian ? &anghess : NULL);

    if(derivative)
    {
        derivative->block<1, 3>(0, 0) +=
            angderiv.block<1, 3>(0, 0) * crossMatrix(q2 - q1);
        derivative->block<1, 3>(0, 3) +=
            angderiv.block<1, 3>(0, 0) * crossMatrix(q0 - q2);
        derivative->block<1, 3>(0, 6) +=
            angderiv.block<1, 3>(0, 0) * crossMatrix(q1 - q0);

        derivative->block<1, 3>(0, 0) +=
            angderiv.block<1, 3>(0, 3) * crossMatrix(q1 - q3);
        derivative->block<1, 3>(0, 3) +=
            angderiv.block<1, 3>(0, 3) * crossMatrix(q3 - q0);
        derivative->block<1, 3>(0, 9) +=
            angderiv.block<1, 3>(0, 3) * crossMatrix(q0 - q1);
    }

    if(hessian)
    {
        Eigen::Matrix3d vqm[3];
        vqm[0] = crossMatrix(q0 - q2);
        vqm[1] = crossMatrix(q1 - q0);
        vqm[2] = crossMatrix(q2 - q1);
        Eigen::Matrix3d wqm[3];
        wqm[0] = crossMatrix(q0 - q1);
        wqm[1] = crossMatrix(q1 - q3);
        wqm[2] = crossMatrix(q3 - q0);

        int vindices[3] = {3, 6, 0};
        int windices[3] = {9, 0, 3};

        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                hessian->block<3, 3>(vindices[i], vindices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 0) * vqm[j];
                hessian->block<3, 3>(vindices[i], windices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 3) * wqm[j];
                hessian->block<3, 3>(windices[i], vindices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 0) * vqm[j];
                hessian->block<3, 3>(windices[i], windices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 3) * wqm[j];
            }

            hessian->block<3, 3>(vindices[i], 3) +=
                vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(3, vindices[i]) += anghess.block<3, 3>(6, 0) * vqm[i];
            hessian->block<3, 3>(vindices[i], 0) +=
                -vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(0, vindices[i]) += -anghess.block<3, 3>(6, 0) * vqm[i];

            hessian->block<3, 3>(windices[i], 3) +=
                wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(3, windices[i]) += anghess.block<3, 3>(6, 3) * wqm[i];
            hessian->block<3, 3>(windices[i], 0) +=
                -wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(0, windices[i]) += -anghess.block<3, 3>(6, 3) * wqm[i];
        }

        Eigen::Vector3d dang1 = angderiv.block<1, 3>(0, 0).transpose();
        Eigen::Vector3d dang2 = angderiv.block<1, 3>(0, 3).transpose();

        Eigen::Matrix3d dang1mat = crossMatrix(dang1);
        Eigen::Matrix3d dang2mat = crossMatrix(dang2);

        hessian->block<3, 3>(6, 3) += dang1mat;
        hessian->block<3, 3>(0, 3) -= dang1mat;
        hessian->block<3, 3>(0, 6) += dang1mat;
        hessian->block<3, 3>(3, 0) += dang1mat;
        hessian->block<3, 3>(3, 6) -= dang1mat;
        hessian->block<3, 3>(6, 0) -= dang1mat;

        hessian->block<3, 3>(9, 0) += dang2mat;
        hessian->block<3, 3>(3, 0) -= dang2mat;
        hessian->block<3, 3>(3, 9) += dang2mat;
        hessian->block<3, 3>(0, 3) += dang2mat;
        hessian->block<3, 3>(0, 9) -= dang2mat;
        hessian->block<3, 3>(9, 3) -= dang2mat;
    }

    return theta;
}

};  // namespace LibShell


__device__ __host__ void __calculateDm2D_double(const double3* vertexes,
                                                const uint3&   index,
                                                __GEIGEN__::Matrix2x2d& M)
{
    double3 v01 = __GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]);
    double3 v02 = __GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]);
    double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v01, v02));
    double3 target;
    target.x                   = 0;
    target.y                   = 0;
    target.z                   = 1;
    double3                vec = __GEIGEN__::__v_vec_cross(normal, target);
    double                 cos = __GEIGEN__::__v_vec_dot(normal, target);
    __GEIGEN__::Matrix3x3d rotation;
    __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

    if(cos + 1 == 0)
    {
        rotation.m[0][0] = -1;
        rotation.m[1][1] = -1;
    }
    else
    {
        __GEIGEN__::Matrix3x3d cross_vec;
        __GEIGEN__::__set_Mat_val(
            cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);
        rotation = __GEIGEN__::__Mat_add(
            rotation,
            __GEIGEN__::__Mat_add(cross_vec,
                                  __GEIGEN__::__S_Mat_multiply(
                                      __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                      1.0 / (1 + cos))));
    }

    double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.x]);
    double3 rotate_uv1 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.y]);
    double3 rotate_uv2 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.z]);

    double2 uv0 = make_double2(rotate_uv0.x, rotate_uv0.y);
    double2 uv1 = make_double2(rotate_uv1.x, rotate_uv1.y);
    double2 uv2 = make_double2(rotate_uv2.x, rotate_uv2.y);

    double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
    double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

    __GEIGEN__::__set_Mat2x2_val_column(M, u0, u1);
}

__device__ __host__ void __calculateDs2D_double(const double3* vertexes,
                                                const uint3&   index,
                                                __GEIGEN__::Matrix3x2d& M)
{
    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;


    M.m[0][0] = o1x;
    M.m[0][1] = o2x;
    M.m[1][0] = o1y;
    M.m[1][1] = o2y;
    M.m[2][0] = o1z;
    M.m[2][1] = o2z;
}
__device__ __host__ void __calculateDms3D_double(const double3* vertexes,
                                                 const uint4&   index,
                                                 __GEIGEN__::Matrix3x3d& M)
{

    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;

    double o3x = vertexes[index.w].x - vertexes[index.x].x;
    double o3y = vertexes[index.w].y - vertexes[index.x].y;
    double o3z = vertexes[index.w].z - vertexes[index.x].z;

    M.m[0][0] = o1x;
    M.m[0][1] = o2x;
    M.m[0][2] = o3x;
    M.m[1][0] = o1y;
    M.m[1][1] = o2y;
    M.m[1][2] = o3y;
    M.m[2][0] = o1z;
    M.m[2][1] = o2z;
    M.m[2][2] = o3z;
}
__device__ __GEIGEN__::Matrix3x2d __computePEPF_BaraffWitkinStretch_double(
    const __GEIGEN__::Matrix3x2d& F, double stretchStiff, double shearStiff, double strainRate)
{

    double2 u, v;
    u.x           = 1;
    u.y           = 0;
    v.x           = 0;
    v.y           = 1;
    double I5u    = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v    = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));
    double ucoeff = 1.0 - 1 / sqrt(I5u);
    double vcoeff = 1.0 - 1 / sqrt(I5v);

    if(I5u > 1)
    {
        ucoeff += 1.5 * strainRate * (sqrt(I5u) + 1 / sqrt(I5u) - 2);
    }
    if(I5v > 1)
    {
        vcoeff += 1.5 * strainRate * (sqrt(I5v) + 1 / sqrt(I5v) - 2);
    }

    //if (I5u > 1.01) {
    //    ucoeff *= 1e1;
    //}
    //if (I5v > 1.01) {
    //    vcoeff *= 1e1;
    //}

    __GEIGEN__::Matrix2x2d uu  = __GEIGEN__::__v2_vec2_toMat2x2(u, u);
    __GEIGEN__::Matrix2x2d vv  = __GEIGEN__::__v2_vec2_toMat2x2(v, v);
    __GEIGEN__::Matrix3x2d Fuu = __GEIGEN__::__M3x2_M2x2_Multiply(F, uu);
    __GEIGEN__::Matrix3x2d Fvv = __GEIGEN__::__M3x2_M2x2_Multiply(F, vv);

    double I6 =
        __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    __GEIGEN__::Matrix2x2d uv  = __GEIGEN__::__v2_vec2_toMat2x2(u, v);
    __GEIGEN__::Matrix2x2d vu  = __GEIGEN__::__v2_vec2_toMat2x2(v, u);
    __GEIGEN__::Matrix3x2d Fuv = __GEIGEN__::__M3x2_M2x2_Multiply(F, uv);
    __GEIGEN__::Matrix3x2d Fvu = __GEIGEN__::__M3x2_M2x2_Multiply(F, vu);
    double                 utv = __GEIGEN__::__v2_vec_multiply(u, v);

    __GEIGEN__::Matrix3x2d PEPF_shear =
        __GEIGEN__::__S_Mat3x2_multiply(__GEIGEN__::__Mat3x2_add(Fuv, Fvu), 2 * (I6 - utv));
    __GEIGEN__::Matrix3x2d PEPF_stretch =
        __GEIGEN__::__Mat3x2_add(__GEIGEN__::__S_Mat3x2_multiply(Fuu, 2 * ucoeff),
                                 __GEIGEN__::__S_Mat3x2_multiply(Fvv, 2 * vcoeff));
    __GEIGEN__::Matrix3x2d PEPF = __GEIGEN__::__Mat3x2_add(
        __GEIGEN__::__S_Mat3x2_multiply(PEPF_shear, shearStiff),
        __GEIGEN__::__S_Mat3x2_multiply(PEPF_stretch, stretchStiff));
    return PEPF;
}
__device__ __GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D2_double(
    const __GEIGEN__::Matrix3x3d& F, const double& I2, const double& I3, double lengthRate, double volumRate)
{
    double                 u = lengthRate, r = volumRate;
    __GEIGEN__::Matrix3x3d pI3pF;

    pI3pF.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
    pI3pF.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
    pI3pF.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];

    pI3pF.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
    pI3pF.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
    pI3pF.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];

    pI3pF.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[1][1] * F.m[0][2];
    pI3pF.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
    pI3pF.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];


    //printf("volRate and LenRate:  %f    %f\n", volumRate, lengthRate);

    __GEIGEN__::Matrix3x3d PEPF, tempA, tempB;
    tempA = __GEIGEN__::__S_Mat_multiply(F, u * (1 - 1 / (I2 + 1)));
    tempB = __GEIGEN__::__S_Mat_multiply(pI3pF, (r * (I3 - 1 - u * 3 / (r * 4))));
    __GEIGEN__::__Mat_add(tempA, tempB, PEPF);
    return PEPF;
}

__device__ __GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D1_double(
    const __GEIGEN__::Matrix3x3d& F,
    const double& I2,
    const double& I3,
    double                        lengthRate,
    double                        volumRate)
{
    double                 u = lengthRate, r = volumRate;
    __GEIGEN__::Matrix3x3d pI3pF;

    pI3pF.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
    pI3pF.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
    pI3pF.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];

    pI3pF.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
    pI3pF.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
    pI3pF.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];

    pI3pF.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[1][1] * F.m[0][2];
    pI3pF.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
    pI3pF.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];

    __GEIGEN__::Matrix3x3d PEPF, tempA, tempB;
    tempA = __GEIGEN__::__S_Mat_multiply(F, u);
    tempB = __GEIGEN__::__S_Mat_multiply(pI3pF, (r * (I3 - 1 - u / r)));
    __GEIGEN__::__Mat_add(tempA, tempB, PEPF);
    return PEPF;
}

__device__ Eigen::Matrix<double, 3, 3> computePEPF_ARAP_double(
    const Eigen::Matrix<double, 3, 3>& F,
    const Eigen::Matrix<double, 3, 3>& U,
    const Eigen::Matrix<double, 3, 3>& V,
    const double&                      lengthRate)
{

    Eigen::Matrix<double, 3, 3> R = U * V.transpose();
    Eigen::Matrix<double, 3, 3> g = F - R;
    return lengthRate * g;
}

__device__ Eigen::Matrix<double, 9, 9> project_ARAP_H_3D(
    const Eigen::Matrix<double, 3, 1>& Sigma,
    const Eigen::Matrix<double, 3, 3>& U,
    const Eigen::Matrix<double, 3, 3>& V,
    const double&                      lengthRate)
{
    Eigen::Matrix<double, 3, 3> R = U * V.transpose();
    Eigen::Matrix<double, 3, 3> T0, T1, T2;

    T0 << 0, -1, 0, 1, 0, 0, 0, 0, 0;
    T1 << 0, 0, 0, 0, 0, 1, 0, -1, 0;
    T2 << 0, 0, 1, 0, 0, 0, -1, 0, 0;

    double ml = 1 / sqrt(2.0);

    T0 = ml * U * T0 * V.transpose();
    T1 = ml * U * T1 * V.transpose();
    T2 = ml * U * T2 * V.transpose();

    //__GEIGEN__::Vector9 t0, t1, t2;
    Eigen::Vector<double, 9> t0 = __GEIGEN__::__Mat3x3_to_vec9_Eigen_double(T0);
    Eigen::Vector<double, 9> t1 = __GEIGEN__::__Mat3x3_to_vec9_Eigen_double(T1);
    Eigen::Vector<double, 9> t2 = __GEIGEN__::__Mat3x3_to_vec9_Eigen_double(T2);

    double sx      = Sigma[0];
    double sy      = Sigma[1];
    double sz      = Sigma[2];
    double lambda0 = 2 / (sx + sy);
    double lambda1 = 2 / (sz + sy);
    double lambda2 = 2 / (sx + sz);

    if(sx + sy < 2)
        lambda0 = 1;
    if(sz + sy < 2)
        lambda1 = 1;
    if(sx + sz < 2)
        lambda2 = 1;

    //__GEIGEN__::Matrix9x9d SH, M9_temp;
    //__GEIGEN__::__identify_Mat9x9(SH, 1);
    __GEIGEN__::Vector9 V9_temp;

    Eigen::Matrix<double, 9, 9> SH;
    SH.setIdentity();
    Eigen::Matrix<double, 9, 9> M9_temp = t0 * t0.transpose();
    SH                                  = SH - lambda0 * M9_temp;
    //M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda0);
    //SH      = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = t1 * t1.transpose();
    SH      = SH - lambda1 * M9_temp;
    //M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda1);
    //SH      = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = t2 * t2.transpose();
    SH      = SH - lambda2 * M9_temp;
    //M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda2);
    //SH      = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    return lengthRate * SH;
}

__device__ __GEIGEN__::Matrix6x6d __project_BaraffWitkinStretch_H(const __GEIGEN__::Matrix3x2d& F,
                                                                  double strainRate)
{

    __GEIGEN__::Matrix6x6d H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    double2 u, v;
    u.x        = 1;
    u.y        = 0;
    v.x        = 0;
    v.y        = 1;
    double I5u = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));
    //double ucoeff = 1.0 - 1 / sqrt(I5u);
    //double vcoeff = 1.0 - 1 / sqrt(I5v);


    //if (I5u > 1.01) {
    //    ucoeff *= 1e1;
    //}
    //if (I5v > 1.01) {
    //    vcoeff *= 1e1;
    //}
    double sqrtI5u = sqrt(I5u);
    double sqrtI5v = sqrt(I5v);
    //double invSqrtI5u = 1.0 / sqrtI5u;
    //double invSqrtI5v = 1.0 / sqrtI5v;
    if(sqrtI5u > 1)
        H.m[0][0] = H.m[1][1] = H.m[2][2] =
            2 * (((sqrtI5u - 1) * (3 * sqrtI5u * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5u));
    if(sqrtI5v > 1)
        H.m[3][3] = H.m[4][4] = H.m[5][5] =
            2 * (((sqrtI5v - 1) * (3 * sqrtI5v * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5v));


    double uCoeff = (sqrtI5u > 1.0) ?
                        (3 * I5u * strainRate - 3 * strainRate + 2) / (sqrt(I5u)) :
                        2.0;
    double vCoeff = (sqrtI5v > 1.0) ?
                        (3 * I5v * strainRate - 3 * strainRate + 2) / (sqrt(I5v)) :
                        2.0;
    //uCoeff *= 2; vCoeff *= 2;


    //if(I5u < 1)
    //{
    //    uCoeff *= 1e-2;
    //}
    //if(I5v < 1)
    //{
    //    vCoeff *= 1e-2;
    //}

    double3 fu, fv;
    fu.x = F.m[0][0];
    fu.y = F.m[1][0];
    fu.z = F.m[2][0];
    fv.x = F.m[0][1];
    fv.y = F.m[1][1];
    fv.z = F.m[2][1];
    fu   = __GEIGEN__::__normalized(fu);
    fv   = __GEIGEN__::__normalized(fv);

    __GEIGEN__::Matrix3x3d cfufu =
        __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(fu, fu), uCoeff);
    __GEIGEN__::Matrix3x3d cfvfv =
        __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(fv, fv), vCoeff);
    H.m[0][0] += cfufu.m[0][0];
    H.m[0][1] += cfufu.m[0][1];
    H.m[0][2] += cfufu.m[0][2];
    H.m[1][0] += cfufu.m[1][0];
    H.m[1][1] += cfufu.m[1][1];
    H.m[1][2] += cfufu.m[1][2];
    H.m[2][0] += cfufu.m[2][0];
    H.m[2][1] += cfufu.m[2][1];
    H.m[2][2] += cfufu.m[2][2];


    H.m[3][3] += cfvfv.m[0][0];
    H.m[3][4] += cfvfv.m[0][1];
    H.m[3][5] += cfvfv.m[0][2];
    H.m[4][3] += cfvfv.m[1][0];
    H.m[4][4] += cfvfv.m[1][1];
    H.m[4][5] += cfvfv.m[1][2];
    H.m[5][3] += cfvfv.m[2][0];
    H.m[5][4] += cfvfv.m[2][1];
    H.m[5][5] += cfvfv.m[2][2];
    return H;
}


__device__ __GEIGEN__::Matrix6x6d __project_BaraffWitkinShear_H(const __GEIGEN__::Matrix3x2d& F)
{

    __GEIGEN__::Matrix6x6d H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    double2 u, v;
    u.x = 1;
    u.y = 0;
    v.x = 0;
    v.y = 1;
    __GEIGEN__::Matrix6x6d H_shear;

    H_shear.m[0][0] = H_shear.m[0][1] = H_shear.m[0][2] = H_shear.m[0][3] =
        H_shear.m[0][4] = H_shear.m[0][5] = 0;
    H_shear.m[1][0] = H_shear.m[1][1] = H_shear.m[1][2] = H_shear.m[1][3] =
        H_shear.m[1][4] = H_shear.m[1][5] = 0;
    H_shear.m[2][0] = H_shear.m[2][1] = H_shear.m[2][2] = H_shear.m[2][3] =
        H_shear.m[2][4] = H_shear.m[2][5] = 0;
    H_shear.m[3][0] = H_shear.m[3][1] = H_shear.m[3][2] = H_shear.m[3][3] =
        H_shear.m[3][4] = H_shear.m[3][5] = 0;
    H_shear.m[4][0] = H_shear.m[4][1] = H_shear.m[4][2] = H_shear.m[4][3] =
        H_shear.m[4][4] = H_shear.m[4][5] = 0;
    H_shear.m[5][0] = H_shear.m[5][1] = H_shear.m[5][2] = H_shear.m[5][3] =
        H_shear.m[5][4] = H_shear.m[5][5] = 0;
    H_shear.m[3][0] = H_shear.m[4][1] = H_shear.m[5][2] = H_shear.m[0][3] =
        H_shear.m[1][4] = H_shear.m[2][5] = 1.0;
    double I6 =
        __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    double signI6 = (I6 >= 0) ? 1.0 : -1.0;

    __GEIGEN__::Matrix2x2d uv  = __GEIGEN__::__v2_vec2_toMat2x2(u, v);
    __GEIGEN__::Matrix2x2d vu  = __GEIGEN__::__v2_vec2_toMat2x2(v, u);
    __GEIGEN__::Matrix3x2d Fuv = __GEIGEN__::__M3x2_M2x2_Multiply(F, uv);
    __GEIGEN__::Matrix3x2d Fvu = __GEIGEN__::__M3x2_M2x2_Multiply(F, vu);
    __GEIGEN__::Vector6    g;
    g.v[0]    = Fuv.m[0][0] + Fvu.m[0][0];
    g.v[1]    = Fuv.m[1][0] + Fvu.m[1][0];
    g.v[2]    = Fuv.m[2][0] + Fvu.m[2][0];
    g.v[3]    = Fuv.m[0][1] + Fvu.m[0][1];
    g.v[4]    = Fuv.m[1][1] + Fvu.m[1][1];
    g.v[5]    = Fuv.m[2][1] + Fvu.m[2][1];
    double I2 = F.m[0][0] * F.m[0][0] + F.m[0][1] * F.m[0][1]
                + F.m[1][0] * F.m[1][0] + F.m[1][1] * F.m[1][1]
                + F.m[2][0] * F.m[2][0] + F.m[2][1] * F.m[2][1];
    double              lambda0 = 0.5 * (I2 + sqrt(I2 * I2 + 12 * I6 * I6));
    __GEIGEN__::Vector6 q0      = __GEIGEN__::__M6x6_v6_multiply(H_shear, g);
    q0                          = __GEIGEN__::__s_vec6_multiply(q0, I6);
    q0 = __GEIGEN__::__add6(q0, __s_vec6_multiply(g, lambda0));
    __GEIGEN__::__normalized_vec6_double(q0);
    __GEIGEN__::Matrix6x6d T;
    T.m[0][0] = T.m[0][1] = T.m[0][2] = T.m[0][3] = T.m[0][4] = T.m[0][5] = 0;
    T.m[1][0] = T.m[1][1] = T.m[1][2] = T.m[1][3] = T.m[1][4] = T.m[1][5] = 0;
    T.m[2][0] = T.m[2][1] = T.m[2][2] = T.m[2][3] = T.m[2][4] = T.m[2][5] = 0;
    T.m[3][0] = T.m[3][1] = T.m[3][2] = T.m[3][3] = T.m[3][4] = T.m[3][5] = 0;
    T.m[4][0] = T.m[4][1] = T.m[4][2] = T.m[4][3] = T.m[4][4] = T.m[4][5] = 0;
    T.m[5][0] = T.m[5][1] = T.m[5][2] = T.m[5][3] = T.m[5][4] = T.m[5][5] = 0;
    T.m[0][0] = T.m[1][1] = T.m[2][2] = T.m[3][3] = T.m[4][4] = T.m[5][5] = 1.0;
    __GEIGEN__::__Mat_add(T, __S_Mat6x6_multiply(H_shear, signI6), T);
    T                      = __GEIGEN__::__S_Mat6x6_multiply(T, 0.5);
    __GEIGEN__::Vector6 Tq = __GEIGEN__::__M6x6_v6_multiply(T, q0);
    double normTQ = Tq.v[0] * Tq.v[0] + Tq.v[1] * Tq.v[1] + Tq.v[2] * Tq.v[2]
                    + Tq.v[3] * Tq.v[3] + Tq.v[4] * Tq.v[4] + Tq.v[5] * Tq.v[5];

    __GEIGEN__::Matrix6x6d Tmp;
    __GEIGEN__::__Mat_add(
        T, __GEIGEN__::__S_Mat6x6_multiply(__v6_vec6_toMat6x6(Tq, Tq), -1.0 / normTQ), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, fabs(I6));
    __GEIGEN__::__Mat_add(Tmp, __S_Mat6x6_multiply(__v6_vec6_toMat6x6(q0, q0), lambda0), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, 2);
    return Tmp;
}

__device__ void __project_StabbleNHK_H_3D(const double3&                sigma,
                                          const __GEIGEN__::Matrix3x3d& U,
                                          const __GEIGEN__::Matrix3x3d& V,
                                          const double&           lengthRate,
                                          const double&           volumRate,
                                          __GEIGEN__::Matrix9x9d& H)
{
    double sigxx = sigma.x * sigma.x;
    double sigyy = sigma.y * sigma.y;
    double sigzz = sigma.z * sigma.z;

    double I3 = sigma.x * sigma.y * sigma.z;
    double I2 = sigxx + sigyy + sigzz;
    double g2 = sigxx * sigyy + sigxx * sigzz + sigyy * sigzz;

    double u = lengthRate, r = volumRate;

    double n  = 2 * u / ((I2 + 1) * (I2 + 1) * (r * (I3 - 1) - 3 * u / 4));
    double p  = r / (r * (I3 - 1) - 3 * u / 4);
    double c2 = -g2 * p - I2 * n;
    double c1 = -(1 + 2 * I3 * p) * I2 - 6 * I3 * n + (g2 * I2 - 9 * I3 * I3) * p * n;
    double c0 = -(2 + 3 * I3 * p) * I3 + (I2 * I2 - 4 * g2) * n
                + 2 * I3 * p * n * (I2 * I2 - 3 * g2);

    double roots[3]     = {0};
    int    num_solution = 0;
    __GEIGEN__::__NewtonSolverForCubicEquation_satbleNeohook(
        1, c2, c1, c0, roots, num_solution, 1e-6);

    __GEIGEN__::Matrix3x3d D[3], M_temp[3];
    double                 q[3];
    __GEIGEN__::Matrix3x3d Q[9];
    double                 lamda[9];
    double                 Ut    = u * (1 - 1 / (I2 + 1));
    double                 alpha = 1 + 3 * u / r / 4;

    double I3minuAlphaDotR = (I3 - alpha) * r;

    for(int i = 0; i < num_solution; i++)
    {
        double alpha0 =
            roots[i] * (sigma.y + sigma.x * sigma.z * n + I3 * sigma.y * p)
            + sigma.x * sigma.z + sigma.y * (sigxx - sigyy + sigzz) * n
            + I3 * sigma.x * sigma.z * p
            + sigma.x * (sigxx - sigyy) * sigma.z * (sigyy - sigzz) * p * n;

        double alpha1 =
            roots[i] * (sigma.x + sigma.y * sigma.z * n + I3 * sigma.x * p)
            + sigma.y * sigma.z - sigma.x * (sigxx - sigyy - sigzz) * n
            + I3 * sigma.y * sigma.z * p
            - sigma.y * (sigxx - sigyy) * sigma.z * (sigxx - sigzz) * p * n;

        double alpha2 =
            roots[i] * roots[i] - roots[i] * (sigxx + sigyy) * (n + sigzz * p)
            - sigzz - 2 * I3 * n - 2 * I3 * sigzz * p
            + ((sigxx - sigyy) * sigma.z) * ((sigxx - sigyy) * sigma.z) * p * n;

        double normalSum = alpha0 * alpha0 + alpha1 * alpha1 + alpha2 * alpha2;

        if(normalSum == 0)
        {
            lamda[i] = 0;
            continue;
        }

        q[i] = 1 / sqrt(normalSum);
        __GEIGEN__::__set_Mat_val(D[i], alpha0, 0, 0, 0, alpha1, 0, 0, 0, alpha2);

        __GEIGEN__::__s_M_Mat_MT_multiply(U, D[i], V, q[i], Q[i]);
        lamda[i] = I3minuAlphaDotR * roots[i] + Ut;
    }


    lamda[3] = Ut + sigma.z * I3minuAlphaDotR;
    lamda[4] = Ut + sigma.x * I3minuAlphaDotR;
    lamda[5] = Ut + sigma.y * I3minuAlphaDotR;

    lamda[6] = Ut - sigma.z * I3minuAlphaDotR;
    lamda[7] = Ut - sigma.x * I3minuAlphaDotR;
    lamda[8] = Ut - sigma.y * I3minuAlphaDotR;

    __GEIGEN__::__set_Mat_val(Q[3], 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[4], 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Q[5], 0, 0, 1, 0, 0, 0, -1, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[6], 0, 1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[7], 0, 0, 0, 0, 0, 1, 0, 1, 0);
    __GEIGEN__::__set_Mat_val(Q[8], 0, 0, 1, 0, 0, 0, 1, 0, 0);

    double ml = 1 / sqrt(2.0);

    M_temp[1] = __GEIGEN__::__Transpose3x3(V);
    for(int i = 3; i < 9; i++)
    {
        __GEIGEN__::__M_Mat_multiply(U, Q[i], M_temp[0]);
        __GEIGEN__::__M_Mat_multiply(M_temp[0], M_temp[1], M_temp[2]);
        Q[i] = __GEIGEN__::__S_Mat_multiply(M_temp[2], ml);

        //Q[i] = __GEIGEN__::__s_M_Mat_MT_multiply(U, Q[i], V, ml);
    }

    __GEIGEN__::Matrix9x9d M9_temp;
    __GEIGEN__::__init_Mat9x9(H, 0);
    __GEIGEN__::Vector9 V9_temp;
    for(int i = 0; i < 9; i++)
    {
        if(lamda[i] > 0)
        {
            V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q[i]);
            M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp, lamda[i]);
            H = __GEIGEN__::__Mat9x9_add(H, M9_temp);
        }
    }
}


__device__ __GEIGEN__::Matrix9x9d __project_ANIOSI5_H_3D(const __GEIGEN__::Matrix3x3d& F,
                                                         const __GEIGEN__::Matrix3x3d& sigma,
                                                         const __GEIGEN__::Matrix3x3d& U,
                                                         const __GEIGEN__::Matrix3x3d& V,
                                                         const double3& fiber_direction,
                                                         const double& scale,
                                                         const double& contract_length)
{
    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d S, M_temp[3], Vtranspose;


    //S = V * sigma * V.transpose();
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_temp[0]);
    Vtranspose = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], Vtranspose, S);
    //__S_Mat_multiply(M_temp[2], ml, Q[i]);

    double3 v_temp = __GEIGEN__::__M_v_multiply(S, direction);
    double I4 = __GEIGEN__::__v_vec_dot(direction, v_temp);  //direction.transpose() * S * direction;
    double I5 = __GEIGEN__::__v_vec_dot(v_temp, v_temp);  //direction.transpose() * S.transpose() * S * direction;

    __GEIGEN__::Matrix9x9d H;
    __GEIGEN__::__init_Mat9x9(H, 0);
    if(abs(I5) < 1e-15)
        return H;

    double s = 0;
    if(I4 < 0)
    {
        s = -1;
    }
    else if(I4 > 0)
    {
        s = 1;
    }

    double lamda0 = scale;
    double lamda1 = scale * (1 - s * contract_length / sqrt(I5));
    double lamda2 = lamda1;
    //double lamda2 = lamda1;
    __GEIGEN__::Matrix3x3d Q0, Q1, Q2, A;
    A = __GEIGEN__::__v_vec_toMat(direction, direction);

    __GEIGEN__::__M_Mat_multiply(F, A, M_temp[0]);
    Q0 = __GEIGEN__::__S_Mat_multiply(M_temp[0], (1 / sqrt(I5)));
    //Q0 = (1 / sqrt(I5)) * F * A;

    __GEIGEN__::Matrix3x3d Tx, Ty, Tz;

    __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    //__Transpose3x3(V, M_temp[0]);
    double3 directionM = __GEIGEN__::__M_v_multiply(Vtranspose, direction);

    double ratio = 1.f / sqrt(2.f);
    Tx           = __GEIGEN__::__S_Mat_multiply(Tx, ratio);
    Ty           = __GEIGEN__::__S_Mat_multiply(Ty, ratio);
    Tz           = __GEIGEN__::__S_Mat_multiply(Tz, ratio);

    //Q1 = U * Tx * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tx, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, Q1);

    //Q2 = (sigma(1, 1) * directionM[1]) * U * Tz * sigma * V.transpose() * A - (sigma(2, 2) * directionM[2]) * U * Ty * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tz, M_temp[0]);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], sigma, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], Vtranspose, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], A, M_temp[0]);
    M_temp[0] = __S_Mat_multiply(M_temp[0], (sigma.m[1][1] * directionM.y));
    __GEIGEN__::__M_Mat_multiply(U, Ty, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, M_temp[2]);
    M_temp[2] =
        __GEIGEN__::__S_Mat_multiply(M_temp[2], -(sigma.m[2][2] * directionM.z));
    __GEIGEN__::__Mat_add(M_temp[0], M_temp[2], Q2);

    //H = lamda0 * vec_double(Q0) * vec_double(Q0).transpose();
    __GEIGEN__::Vector9 V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q0);
    __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
    H = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda0);
    //H = __Mat9x9_add(H, M9_temp[1]);
    if(lamda1 > 0)
    {
        //H += lamda1 * vec_double(Q1) * vec_double(Q1).transpose();
        //H += lamda2 * vec_double(Q2) * vec_double(Q2).transpose();
        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q1);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda1);
        H       = __GEIGEN__::__Mat9x9_add(H, M9_temp);

        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q2);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda2);
        H       = __GEIGEN__::__Mat9x9_add(H, M9_temp);
    }

    return H;
}

__device__ __GEIGEN__::Matrix3x3d __computePEPF_Aniostropic3D_double(
    const __GEIGEN__::Matrix3x3d& F, double3 fiber_direction, const double& scale, const double& contract_length)
{

    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d U, V, S, sigma, M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_Temp0, M_Temp1, S);
    double3 V_Temp0, V_Temp1;
    V_Temp0 = __GEIGEN__::__v_M_multiply(direction, S);
    double I4, I5;
    I4      = __GEIGEN__::__v_vec_dot(V_Temp0, direction);
    V_Temp1 = __GEIGEN__::__M_v_multiply(S, direction);
    I5      = __GEIGEN__::__v_vec_dot(V_Temp1, V_Temp1);

    if(I4 == 0)
    {
        // system("pause");
    }

    double s = 0;
    if(I4 < 0)
    {
        s = -1;
    }
    else if(I4 > 0)
    {
        s = 1;
    }

    __GEIGEN__::Matrix3x3d PEPF;
    double s_temp0 = scale * (1 - s * contract_length / sqrt(I5));
    M_Temp0        = __GEIGEN__::__v_vec_toMat(direction, direction);
    __GEIGEN__::__M_Mat_multiply(F, M_Temp0, M_Temp1);
    PEPF = __GEIGEN__::__S_Mat_multiply(M_Temp1, s_temp0);
    return PEPF;
}
__device__ double __cal_BaraffWitkinStretch_energy(const double3* vertexes,
                                                   const uint3&   triangle,
                                                   const __GEIGEN__::Matrix2x2d& triDmInverse,
                                                   const double& area,
                                                   const double& stretchStiff,
                                                   const double& shearhStiff,
                                                   double        strainRate)
{
    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangle, Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, triDmInverse);

    double2 u, v;
    u.x        = 1;
    u.y        = 0;
    v.x        = 0;
    v.y        = 1;
    double I5u = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));

    double ucoeff = 1;
    double vcoeff = 1;
    if(I5u <= 1)
    {
        ucoeff = 0;
    }
    if(I5v <= 1)
    {
        vcoeff = 0;
    }

    double I6 =
        __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    return area
           * (stretchStiff
                  * (pow(sqrt(I5u) - 1, 2) + ucoeff * strainRate * pow(sqrt(I5u) - 1, 3)
                     + pow(sqrt(I5v) - 1, 2) + vcoeff * strainRate * pow(sqrt(I5v) - 1, 3))
              + shearhStiff * I6 * I6);
}
__device__ double __cal_StabbleNHK_energy2_3D(const double3* vertexes,
                                             const uint4&   tetrahedra,
                                             const __GEIGEN__::Matrix3x3d& DmInverse,
                                             const double& volume,
                                             const double& lenRate,
                                             const double& volRate)
{
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);


    double I2 = __GEIGEN__::__squaredNorm(F);
    double I3 = __GEIGEN__::__Determiant(F);

    return (0.5 * lenRate * (I2 - 3)
            + 0.5 * volRate * (I3 - 1 - 3 * lenRate / 4 / volRate)
                  * (I3 - 1 - 3 * lenRate / 4 / volRate)
            - 0.5 * lenRate * log(I2 + 1)
            - (0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate)
               - 0.5 * lenRate * log(4.0)))
           * volume;
}


__device__ double __cal_StabbleNHK_energy1_3D(const double3* vertexes,
                                             const uint4&   tetrahedra,
                                             const __GEIGEN__::Matrix3x3d& DmInverse,
                                             const double& volume,
                                             const double& lenRate,
                                             const double& volRate)
{
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);


    double I2 = __GEIGEN__::__squaredNorm(F);
    double I3 = __GEIGEN__::__Determiant(F);

    double Jminus1 = I3 - 1 - lenRate / volRate;
    return 0.5 * (lenRate * (I2 - 3) + volRate * (Jminus1 * Jminus1)) * volume;
}







__device__ double __cal_ARAPX_energy_3D(const double3* vertexes,
                                        const uint4&   tetrahedra,
                                        const __GEIGEN__::Matrix3x3d& DmInverse,
                                        const double&                 volume,
                                        const double&                 mu,
                                        const double&                 lambda_)
{
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);


    Eigen::Matrix<double, 3, 3> MF, UU, VV;
    Eigen::Matrix<double, 3, 1> Sigma;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            MF(i, j) = F.m[i][j];
        }
    }

    __GEIGEN__::math::qr_svd(MF, Sigma, UU, VV);


    double          energy       = 0;
    double          volume_stiff = lambda_;
    Eigen::Matrix3d R            = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d S            = Eigen::Matrix3d::Zero();

    Matrix3d sig = Matrix3d::Zero();
    sig(0, 0)    = Sigma(0, 0);
    sig(1, 1)    = Sigma(1, 0);
    sig(2, 2)    = Sigma(2, 0);

    R = UU * VV.transpose();
    S = VV * sig * VV.transpose();

    double          J = MF.determinant();
    Eigen::Matrix3d A = S - Eigen::Matrix3d::Identity();

    double T1          = (A).trace();
    double T2          = (A * A).trace();
    double T3          = (A * A * A).trace();
    double volume_term = volume_stiff / 2.0 * (J - 1.0) * (J - 1.0);

#if arap_model == 2
    energy = mu / 2.0 * T2 + volume_term;
#elif arap_model == 3
    energy = mu / 2.0 * T2 + mu / 6.0 * T3 + volume_term;
#elif arap_model == 4
    energy = mu / 2.0 * T2 + mu / 6.0 * T3 + lambda_ / 2.0 * std::pow(T2, 2.0) + volume_term;
#endif
    return energy * volume;
}








__device__ double __cal_ARAP_energy_3D(const double3*                vertexes,
                                       const uint4&                  tetrahedra,
                                       const __GEIGEN__::Matrix3x3d& DmInverse,
                                       const double&                 volume,
                                       const double&                 lenRate)
{
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2]);
    __GEIGEN__::Matrix3x3d U, V, sigma, S, R;
    __GEIGEN__::Matrix3x3d M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, sigma),
                                     __GEIGEN__::__Transpose3x3(V));  //V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d g      = __GEIGEN__::__Mat3x3_minus(F, R);
    double                 energy = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            energy += g.m[i][j] * g.m[i][j];
        }
    }
    return energy * volume * lenRate * 0.5;
    //printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__ __GEIGEN__::Matrix9x12d __computePFDsPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm)
{
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1       = -(m + p + s);
    double t2       = -(n + q + t);
    double t3       = -(o + r + u);
    matOut.m[0][0]  = t1;
    matOut.m[0][3]  = m;
    matOut.m[0][6]  = p;
    matOut.m[0][9]  = s;
    matOut.m[1][1]  = t1;
    matOut.m[1][4]  = m;
    matOut.m[1][7]  = p;
    matOut.m[1][10] = s;
    matOut.m[2][2]  = t1;
    matOut.m[2][5]  = m;
    matOut.m[2][8]  = p;
    matOut.m[2][11] = s;
    matOut.m[3][0]  = t2;
    matOut.m[3][3]  = n;
    matOut.m[3][6]  = q;
    matOut.m[3][9]  = t;
    matOut.m[4][1]  = t2;
    matOut.m[4][4]  = n;
    matOut.m[4][7]  = q;
    matOut.m[4][10] = t;
    matOut.m[5][2]  = t2;
    matOut.m[5][5]  = n;
    matOut.m[5][8]  = q;
    matOut.m[5][11] = t;
    matOut.m[6][0]  = t3;
    matOut.m[6][3]  = o;
    matOut.m[6][6]  = r;
    matOut.m[6][9]  = u;
    matOut.m[7][1]  = t3;
    matOut.m[7][4]  = o;
    matOut.m[7][7]  = r;
    matOut.m[7][10] = u;
    matOut.m[8][2]  = t3;
    matOut.m[8][5]  = o;
    matOut.m[8][8]  = r;
    matOut.m[8][11] = u;

    return matOut;
}

__device__ __GEIGEN__::Matrix6x12d __computePFDsPX3D_6x12_double(const __GEIGEN__::Matrix2x2d& InverseDm)
{
    __GEIGEN__::Matrix6x12d matOut;
    __GEIGEN__::__init_Mat6x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1];

    matOut.m[0][0] = -m;
    matOut.m[3][0] = -n;
    matOut.m[1][1] = -m;
    matOut.m[4][1] = -n;
    matOut.m[2][2] = -m;
    matOut.m[5][2] = -n;

    matOut.m[0][3] = -p;
    matOut.m[3][3] = -q;
    matOut.m[1][4] = -p;
    matOut.m[4][4] = -q;
    matOut.m[2][5] = -p;
    matOut.m[5][5] = -q;

    matOut.m[0][6] = p;
    matOut.m[3][6] = q;
    matOut.m[1][7] = p;
    matOut.m[4][7] = q;
    matOut.m[2][8] = p;
    matOut.m[5][8] = q;

    matOut.m[0][9]  = m;
    matOut.m[3][9]  = n;
    matOut.m[1][10] = m;
    matOut.m[4][10] = n;
    matOut.m[2][11] = m;
    matOut.m[5][11] = n;

    return matOut;
}

__device__ __GEIGEN__::Matrix6x9d __computePFDsPX3D_6x9_double(const __GEIGEN__::Matrix2x2d& InverseDm)
{
    __GEIGEN__::Matrix6x9d matOut;
    __GEIGEN__::__init_Mat6x9_val(matOut, 0);
    double d0 = InverseDm.m[0][0], d2 = InverseDm.m[0][1];
    double d1 = InverseDm.m[1][0], d3 = InverseDm.m[1][1];

    double s0 = d0 + d1;
    double s1 = d2 + d3;

    matOut.m[0][0] = -s0;
    matOut.m[3][0] = -s1;

    // dF / dy0
    matOut.m[1][1] = -s0;
    matOut.m[4][1] = -s1;

    // dF / dz0
    matOut.m[2][2] = -s0;
    matOut.m[5][2] = -s1;

    // dF / dx1
    matOut.m[0][3] = d0;
    matOut.m[3][3] = d2;

    // dF / dy1
    matOut.m[1][4] = d0;
    matOut.m[4][4] = d2;

    // dF / dz1
    matOut.m[2][5] = d0;
    matOut.m[5][5] = d2;

    // dF / dx2
    matOut.m[0][6] = d1;
    matOut.m[3][6] = d3;

    // dF / dy2
    matOut.m[1][7] = d1;
    matOut.m[4][7] = d3;

    // dF / dz2
    matOut.m[2][8] = d1;
    matOut.m[5][8] = d3;

    return matOut;
}

__device__ __GEIGEN__::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm)
{
    __GEIGEN__::Matrix3x6d matOut;
    __GEIGEN__::__init_Mat3x6_val(matOut, 0);

    matOut.m[0][0] = -InverseDm;
    matOut.m[1][1] = -InverseDm;
    matOut.m[2][2] = -InverseDm;

    matOut.m[0][3] = InverseDm;
    matOut.m[1][4] = InverseDm;
    matOut.m[2][5] = InverseDm;

    return matOut;
}

__device__ __GEIGEN__::Matrix9x12d __computePFDmPX3D_double(const __GEIGEN__::Matrix12x9d& PDmPx,
                                                            const __GEIGEN__::Matrix3x3d& Ds,
                                                            const __GEIGEN__::Matrix3x3d& DmInv)
{
    __GEIGEN__::Matrix9x12d DsPDminvPx;
    __GEIGEN__::__init_Mat9x12_val(DsPDminvPx, 0);

    for(int i = 0; i < 12; i++)
    {
        __GEIGEN__::Matrix3x3d PDmPxi =
            __GEIGEN__::__vec9_to_Mat3x3_double(PDmPx.m[i]);
        __GEIGEN__::Matrix3x3d DsPDminvPxi;
        __GEIGEN__::__M_Mat_multiply(
            Ds,
            __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(DmInv, PDmPxi), DmInv),
            DsPDminvPxi);

        __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(DsPDminvPxi);

        for(int j = 0; j < 9; j++)
        {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __GEIGEN__::Matrix6x12d __computePFDmPX3D_6x12_double(
    const __GEIGEN__::Matrix12x4d& PDmPx,
    const __GEIGEN__::Matrix3x2d&  Ds,
    const __GEIGEN__::Matrix2x2d&  DmInv)
{
    __GEIGEN__::Matrix6x12d DsPDminvPx;
    __GEIGEN__::__init_Mat6x12_val(DsPDminvPx, 0);

    for(int i = 0; i < 12; i++)
    {
        __GEIGEN__::Matrix2x2d PDmPxi =
            __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(
            Ds,
            __GEIGEN__::__M2x2_Mat2x2_multiply(
                __GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for(int j = 0; j < 6; j++)
        {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __GEIGEN__::Matrix3x6d __computePFDmPX3D_3x6_double(const __GEIGEN__::Vector6& PDmPx,
                                                               const double3& Ds,
                                                               const double& DmInv)
{
    __GEIGEN__::Matrix3x6d DsPDminvPx;
    __GEIGEN__::__init_Mat3x6_val(DsPDminvPx, 0);

    for(int i = 0; i < 6; i++)
    {
        double PDmPxi = PDmPx.v[i];

        double3 DsPDminvPxi =
            __GEIGEN__::__s_vec_multiply(Ds, ((DmInv * PDmPxi) * DmInv));
        DsPDminvPx.m[0][i] = -DsPDminvPxi.x;
        DsPDminvPx.m[1][i] = -DsPDminvPxi.y;
        DsPDminvPx.m[2][i] = -DsPDminvPxi.z;
    }

    return DsPDminvPx;
}

__device__ __GEIGEN__::Matrix6x9d __computePFDmPX3D_6x9_double(
    const __GEIGEN__::Matrix9x4d& PDmPx,
    const __GEIGEN__::Matrix3x2d& Ds,
    const __GEIGEN__::Matrix2x2d& DmInv)
{
    __GEIGEN__::Matrix6x9d DsPDminvPx;
    __GEIGEN__::__init_Mat6x9_val(DsPDminvPx, 0);

    for(int i = 0; i < 9; i++)
    {
        __GEIGEN__::Matrix2x2d PDmPxi =
            __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(
            Ds,
            __GEIGEN__::__M2x2_Mat2x2_multiply(
                __GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for(int j = 0; j < 6; j++)
        {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __GEIGEN__::Matrix9x12d __computePFPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm)
{
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1       = -(m + p + s);
    double t2       = -(n + q + t);
    double t3       = -(o + r + u);
    matOut.m[0][0]  = t1;
    matOut.m[0][3]  = m;
    matOut.m[0][6]  = p;
    matOut.m[0][9]  = s;
    matOut.m[1][1]  = t1;
    matOut.m[1][4]  = m;
    matOut.m[1][7]  = p;
    matOut.m[1][10] = s;
    matOut.m[2][2]  = t1;
    matOut.m[2][5]  = m;
    matOut.m[2][8]  = p;
    matOut.m[2][11] = s;
    matOut.m[3][0]  = t2;
    matOut.m[3][3]  = n;
    matOut.m[3][6]  = q;
    matOut.m[3][9]  = t;
    matOut.m[4][1]  = t2;
    matOut.m[4][4]  = n;
    matOut.m[4][7]  = q;
    matOut.m[4][10] = t;
    matOut.m[5][2]  = t2;
    matOut.m[5][5]  = n;
    matOut.m[5][8]  = q;
    matOut.m[5][11] = t;
    matOut.m[6][0]  = t3;
    matOut.m[6][3]  = o;
    matOut.m[6][6]  = r;
    matOut.m[6][9]  = u;
    matOut.m[7][1]  = t3;
    matOut.m[7][4]  = o;
    matOut.m[7][7]  = r;
    matOut.m[7][10] = u;
    matOut.m[8][2]  = t3;
    matOut.m[8][5]  = o;
    matOut.m[8][8]  = r;
    matOut.m[8][11] = u;
    return matOut;
}

__device__ Eigen::Matrix<double, 9,12> __computePFPX3D_Eigen_double(const __GEIGEN__::Matrix3x3d& InverseDm)
{
    Eigen::Matrix<double, 9, 12> matOut;
    matOut.setZero();
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1       = -(m + p + s);
    double t2       = -(n + q + t);
    double t3       = -(o + r + u);
    matOut(0, 0)  = t1;
    matOut(0, 3)  = m;
    matOut(0, 6)  = p;
    matOut(0, 9)  = s;
    matOut(1, 1)  = t1;
    matOut(1, 4)  = m;
    matOut(1, 7)  = p;
    matOut(1, 10) = s;
    matOut(2, 2)  = t1;
    matOut(2, 5)  = m;
    matOut(2, 8)  = p;
    matOut(2, 11) = s;
    matOut(3, 0)  = t2;
    matOut(3, 3)  = n;
    matOut(3, 6)  = q;
    matOut(3, 9)  = t;
    matOut(4, 1)  = t2;
    matOut(4, 4)  = n;
    matOut(4, 7)  = q;
    matOut(4, 10) = t;
    matOut(5, 2)  = t2;
    matOut(5, 5)  = n;
    matOut(5, 8)  = q;
    matOut(5, 11) = t;
    matOut(6, 0)  = t3;
    matOut(6, 3)  = o;
    matOut(6, 6)  = r;
    matOut(6, 9)  = u;
    matOut(7, 1)  = t3;
    matOut(7, 4)  = o;
    matOut(7, 7)  = r;
    matOut(7, 10) = u;
    matOut(8, 2)  = t3;
    matOut(8, 5)  = o;
    matOut(8, 8)  = r;
    matOut(8, 11) = u;
    return matOut;
}

__device__ void __project_StabbleNHK_H_3D2_makePD(Matrix<double, 9, 9>& Hq,
                                                  const __GEIGEN__::Matrix3x3d& F,
                                                  const double& Ic,
                                                  const double& I3,
                                                  const double& lengthRate,
                                                  const double& volumRate)
{
    double u = lengthRate, r = volumRate;

    __GEIGEN__::Matrix9x9d HJ;
    __GEIGEN__::Vector9    g = __GEIGEN__::__Mat3x3_to_vec9_double(F);

    __GEIGEN__::Vector9 gJ;
    double3 gjc0 = __GEIGEN__::__v_vec_cross(make_double3(g.v[3], g.v[4], g.v[5]),
                                             make_double3(g.v[6], g.v[7], g.v[8]));
    double3 gjc1 = __GEIGEN__::__v_vec_cross(make_double3(g.v[6], g.v[7], g.v[8]),
                                             make_double3(g.v[0], g.v[1], g.v[2]));
    double3 gjc2 = __GEIGEN__::__v_vec_cross(make_double3(g.v[0], g.v[1], g.v[2]),
                                             make_double3(g.v[3], g.v[4], g.v[5]));
    g       = __GEIGEN__::__s_vec9_multiply(g, 2);
    gJ.v[0] = gjc0.x;
    gJ.v[1] = gjc0.y;
    gJ.v[2] = gjc0.z;
    gJ.v[3] = gjc1.x;
    gJ.v[4] = gjc1.y;
    gJ.v[5] = gjc1.z;
    gJ.v[6] = gjc2.x;
    gJ.v[7] = gjc2.y;
    gJ.v[8] = gjc2.z;

    __GEIGEN__::Matrix3x3d f0hat;
    __GEIGEN__::__set_Mat_val(
        f0hat, 0, -F.m[2][0], F.m[1][0], F.m[2][0], 0, -F.m[0][0], -F.m[1][0], F.m[0][0], 0);

    __GEIGEN__::Matrix3x3d f1hat;
    __GEIGEN__::__set_Mat_val(
        f1hat, 0, -F.m[2][1], F.m[1][1], F.m[2][1], 0, -F.m[0][1], -F.m[1][1], F.m[0][1], 0);

    __GEIGEN__::Matrix3x3d f2hat;
    __GEIGEN__::__set_Mat_val(
        f2hat, 0, -F.m[2][2], F.m[1][2], F.m[2][2], 0, -F.m[0][2], -F.m[1][2], F.m[0][2], 0);

    double mu = u, lambda = r;
    double J    = I3;
    double coef = lambda * (J - 1 - (3 * mu) / (4.0 * lambda));

    HJ.m[0][0] = 0;
    HJ.m[0][1] = 0;
    HJ.m[0][2] = 0;
    HJ.m[1][0] = 0;
    HJ.m[1][1] = 0;
    HJ.m[1][2] = 0;
    HJ.m[2][0] = 0;
    HJ.m[2][1] = 0;
    HJ.m[2][2] = 0;

    HJ.m[0][3] = -f2hat.m[0][0] * coef;
    HJ.m[0][4] = -f2hat.m[0][1] * coef;
    HJ.m[0][5] = -f2hat.m[0][2] * coef;
    HJ.m[1][3] = -f2hat.m[1][0] * coef;
    HJ.m[1][4] = -f2hat.m[1][1] * coef;
    HJ.m[1][5] = -f2hat.m[1][2] * coef;
    HJ.m[2][3] = -f2hat.m[2][0] * coef;
    HJ.m[2][4] = -f2hat.m[2][1] * coef;
    HJ.m[2][5] = -f2hat.m[2][2] * coef;

    HJ.m[0][6] = f1hat.m[0][0] * coef;
    HJ.m[0][7] = f1hat.m[0][1] * coef;
    HJ.m[0][8] = f1hat.m[0][2] * coef;
    HJ.m[1][6] = f1hat.m[1][0] * coef;
    HJ.m[1][7] = f1hat.m[1][1] * coef;
    HJ.m[1][8] = f1hat.m[1][2] * coef;
    HJ.m[2][6] = f1hat.m[2][0] * coef;
    HJ.m[2][7] = f1hat.m[2][1] * coef;
    HJ.m[2][8] = f1hat.m[2][2] * coef;

    HJ.m[3][0] = f2hat.m[0][0] * coef;
    HJ.m[3][1] = f2hat.m[0][1] * coef;
    HJ.m[3][2] = f2hat.m[0][2] * coef;
    HJ.m[4][0] = f2hat.m[1][0] * coef;
    HJ.m[4][1] = f2hat.m[1][1] * coef;
    HJ.m[4][2] = f2hat.m[1][2] * coef;
    HJ.m[5][0] = f2hat.m[2][0] * coef;
    HJ.m[5][1] = f2hat.m[2][1] * coef;
    HJ.m[5][2] = f2hat.m[2][2] * coef;

    HJ.m[3][3] = 0;
    HJ.m[3][4] = 0;
    HJ.m[3][5] = 0;
    HJ.m[4][3] = 0;
    HJ.m[4][4] = 0;
    HJ.m[4][5] = 0;
    HJ.m[5][3] = 0;
    HJ.m[5][4] = 0;
    HJ.m[5][5] = 0;

    HJ.m[3][6] = -f0hat.m[0][0] * coef;
    HJ.m[3][7] = -f0hat.m[0][1] * coef;
    HJ.m[3][8] = -f0hat.m[0][2] * coef;
    HJ.m[4][6] = -f0hat.m[1][0] * coef;
    HJ.m[4][7] = -f0hat.m[1][1] * coef;
    HJ.m[4][8] = -f0hat.m[1][2] * coef;
    HJ.m[5][6] = -f0hat.m[2][0] * coef;
    HJ.m[5][7] = -f0hat.m[2][1] * coef;
    HJ.m[5][8] = -f0hat.m[2][2] * coef;
    HJ.m[6][0] = -f1hat.m[0][0] * coef;
    HJ.m[6][1] = -f1hat.m[0][1] * coef;
    HJ.m[6][2] = -f1hat.m[0][2] * coef;
    HJ.m[7][0] = -f1hat.m[1][0] * coef;
    HJ.m[7][1] = -f1hat.m[1][1] * coef;
    HJ.m[7][2] = -f1hat.m[1][2] * coef;
    HJ.m[8][0] = -f1hat.m[2][0] * coef;
    HJ.m[8][1] = -f1hat.m[2][1] * coef;
    HJ.m[8][2] = -f1hat.m[2][2] * coef;

    HJ.m[6][3] = f0hat.m[0][0] * coef;
    HJ.m[6][4] = f0hat.m[0][1] * coef;
    HJ.m[6][5] = f0hat.m[0][2] * coef;
    HJ.m[7][3] = f0hat.m[1][0] * coef;
    HJ.m[7][4] = f0hat.m[1][1] * coef;
    HJ.m[7][5] = f0hat.m[1][2] * coef;
    HJ.m[8][3] = f0hat.m[2][0] * coef;
    HJ.m[8][4] = f0hat.m[2][1] * coef;
    HJ.m[8][5] = f0hat.m[2][2] * coef;

    HJ.m[6][6] = 0;
    HJ.m[6][7] = 0;
    HJ.m[6][8] = 0;
    HJ.m[7][6] = 0;
    HJ.m[7][7] = 0;
    HJ.m[7][8] = 0;
    HJ.m[8][6] = 0;
    HJ.m[8][7] = 0;
    HJ.m[8][8] = 0;


    __GEIGEN__::Matrix9x9d H1, H;  //, M9_temp[2];
    __GEIGEN__::__identify_Mat9x9(H1, 2 * (Ic * mu) / (2 * (Ic + 1)));

    H = __GEIGEN__::__Mat9x9_add(H1, HJ);

    H = __GEIGEN__::__Mat9x9_add(
        H, __GEIGEN__::__v9_vec9_toMat9x9(g, g, (mu / (2 * (Ic + 1) * (Ic + 1)))));

    H = __GEIGEN__::__Mat9x9_add(H, __GEIGEN__::__v9_vec9_toMat9x9(gJ, gJ, lambda));

    for(int i = 0; i != 9; ++i)
        for(int j = 0; j != 9; ++j)
            Hq(i, j) = H.m[i][j];

    makePDSNK<double, 9>(Hq);
}



__device__ void BuildTwistAndFlipEigenvectors(const Matrix3d& U, const Matrix3d& V, Matrix<double, 9, 9>& Q)
{
    const double scale = 1.0 / std::sqrt(2.0);
    const Matrix3d      sV    = scale * V;

    using M3 = Eigen::Matrix<double, 3, 3, Eigen::ColMajor>;

    M3 A;
    A << sV(0, 2) * U(0, 1), sV(1, 2) * U(0, 1), sV(2, 2) * U(0, 1),
        sV(0, 2) * U(1, 1), sV(1, 2) * U(1, 1), sV(2, 2) * U(1, 1),
        sV(0, 2) * U(2, 1), sV(1, 2) * U(2, 1), sV(2, 2) * U(2, 1);

    M3 B;
    B << sV(0, 1) * U(0, 2), sV(1, 1) * U(0, 2), sV(2, 1) * U(0, 2),
        sV(0, 1) * U(1, 2), sV(1, 1) * U(1, 2), sV(2, 1) * U(1, 2),
        sV(0, 1) * U(2, 2), sV(1, 1) * U(2, 2), sV(2, 1) * U(2, 2);

    M3 C;
    C << sV(0, 2) * U(0, 0), sV(1, 2) * U(0, 0), sV(2, 2) * U(0, 0),
        sV(0, 2) * U(1, 0), sV(1, 2) * U(1, 0), sV(2, 2) * U(1, 0),
        sV(0, 2) * U(2, 0), sV(1, 2) * U(2, 0), sV(2, 2) * U(2, 0);

    M3 D;
    D << sV(0, 0) * U(0, 2), sV(1, 0) * U(0, 2), sV(2, 0) * U(0, 2),
        sV(0, 0) * U(1, 2), sV(1, 0) * U(1, 2), sV(2, 0) * U(1, 2),
        sV(0, 0) * U(2, 2), sV(1, 0) * U(2, 2), sV(2, 0) * U(2, 2);

    M3 E;
    E << sV(0, 1) * U(0, 0), sV(1, 1) * U(0, 0), sV(2, 1) * U(0, 0),
        sV(0, 1) * U(1, 0), sV(1, 1) * U(1, 0), sV(2, 1) * U(1, 0),
        sV(0, 1) * U(2, 0), sV(1, 1) * U(2, 0), sV(2, 1) * U(2, 0);

    M3 F;
    F << sV(0, 0) * U(0, 1), sV(1, 0) * U(0, 1), sV(2, 0) * U(0, 1),
        sV(0, 0) * U(1, 1), sV(1, 0) * U(1, 1), sV(2, 0) * U(1, 1),
        sV(0, 0) * U(2, 1), sV(1, 0) * U(2, 1), sV(2, 0) * U(2, 1);

    // Twist eigenvectors
    Eigen::Map<M3>(Q.data())      = B - A;
    Eigen::Map<M3>(Q.data() + 9)  = D - C;
    Eigen::Map<M3>(Q.data() + 18) = F - E;

    // Flip eigenvectors
    Eigen::Map<M3>(Q.data() + 27) = A + B;
    Eigen::Map<M3>(Q.data() + 36) = C + D;
    Eigen::Map<M3>(Q.data() + 45) = E + F;
}

__device__ Matrix<double, 9, 9> __project_StabbleNHK_H_3D1_makePD(
    Eigen::Matrix<double, 9, 9>&  H,
    double                        I3,
    const double&                 mu,
    const double&                 lambda,
    const __GEIGEN__::Matrix3x3d& FF)
{
    Eigen::Matrix<double, 3, 3> F, U, V;
    Eigen::Matrix<double, 3, 1> S;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            F(i, j) = FF.m[i][j];
        }
    }

    __GEIGEN__::math::qr_svd(F, S, U, V);


    Vector<double, 9>    eigenvalues;
    Matrix<double, 9, 9> eigenvectors;

    const double J = I3;

    // Compute the twist and flip eigenvalues
    {
        // Twist eigenvalues
        eigenvalues.segment<3>(0) = S;
        // Flip eigenvalues
        eigenvalues.segment<3>(3) = -S;
        const double evScale      = lambda * (J - 1.0) - mu;
        eigenvalues.segment<6>(0) *= evScale;
        eigenvalues.segment<6>(0).array() += mu;
    }

    // Compute the twist and flip eigenvectors
    BuildTwistAndFlipEigenvectors(U, V, eigenvectors);

    // Compute the remaining three eigenvalues and eigenvectors
    {
        Matrix3d     A;
        const double s0s0    = S(0) * S(0);
        const double s1s1    = S(1) * S(1);
        const double s2s2    = S(2) * S(2);
        A(0, 0)              = mu + lambda * s1s1 * s2s2;
        A(1, 1)              = mu + lambda * s0s0 * s2s2;
        A(2, 2)              = mu + lambda * s0s0 * s1s1;
        const double evScale = lambda * (2.0 * J - 1.0) - mu;
        A(0, 1)              = evScale * S(2);
        A(1, 0)              = A(0, 1);
        A(0, 2)              = evScale * S(1);
        A(2, 0)              = A(0, 2);
        A(1, 2)              = evScale * S(0);
        A(2, 1)              = A(1, 2);

        Eigen::SelfAdjointEigenSolver<Matrix3d> Aeigs;
        Aeigs.computeDirect(A);
        eigenvalues.segment<3>(6) = Aeigs.eigenvalues();

        Eigen::Map<Matrix3d>(eigenvectors.data() + 54) =
            U * Aeigs.eigenvectors().col(0).asDiagonal() * V.transpose();
        Eigen::Map<Matrix3d>(eigenvectors.data() + 63) =
            U * Aeigs.eigenvectors().col(1).asDiagonal() * V.transpose();
        Eigen::Map<Matrix3d>(eigenvectors.data() + 72) =
            U * Aeigs.eigenvectors().col(2).asDiagonal() * V.transpose();
    }

    for(int i = 0; i < 9; i++)
    {
        if(eigenvalues(i) < 0.0)
        {
            eigenvalues(i) = 0.0;
        }
    }
    H = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}


__device__ Eigen::Matrix<double, 3, 3> computePEPF_ARAPX_double(
    const Eigen::Matrix<double, 3, 3>& F,
    const Eigen::Matrix<double, 3, 3>& U,
    const Eigen::Matrix<double, 3, 3>& V,
    const Eigen::Matrix<double, 3, 1>& Sigma,
    const double&                      mu,
    const double&                      lambda_,
    int model)
{
    double volume_stiff = lambda_;

    Matrix3d sigmaMat = Matrix3d::Zero();
    sigmaMat(0, 0)    = Sigma(0, 0);
    sigmaMat(1, 1)    = Sigma(1, 0);
    sigmaMat(2, 2)    = Sigma(2, 0);

    Eigen::Matrix3d R = U * V.transpose();
    Eigen::Matrix3d S = V * sigmaMat * V.transpose();


    Eigen::Matrix3d A = S - Eigen::Matrix3d::Identity();
    double          J = F.determinant();


    double T1          = (A).trace();
    double T2          = (A * A).trace();
    double T3          = (A * A * A).trace();
    double volume_term = volume_stiff / 2.0 * (J - 1.0) * (J - 1.0);


    Eigen::Matrix3d T2_d = 2.0 * F - 2.0 * R;
    Eigen::Matrix3d T3_d = 3.0 * (F * S - 2.0 * F + R);

    __GEIGEN__::Matrix3x3d FF, F_inv;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            FF.m[i][j] = F(i, j);
        }
    }


    __GEIGEN__::__Inverse(FF, F_inv);

    Matrix3d F_inv_eigen;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            F_inv_eigen(i, j) = F_inv.m[i][j];
        }
    }

    Eigen::Matrix3d volume_term_d =
        volume_stiff * (J - 1.0) * J * F_inv_eigen.transpose();

    Eigen::Matrix3d PK1 = Eigen::Matrix3d::Zero();
#if arap_model == 2
    PK1 = mu / 2.0 * T2_d + volume_term_d;
#elif arap_model == 3
    PK1 = mu / 2.0 * T2_d + mu / 6.0 * T3_d + volume_term_d;
#elif arap_model == 4
    PK1 = mu / 2.0 * T2_d + mu / 6.0 * T3_d + lambda_ / 2.0 * (2.0 * T2 * T2_d) + volume_term_d;
#endif
    return PK1;
}

__device__ Eigen::Matrix<double, 9, 9> compute_DRDF_Hessian(const Matrix3d& F,
                                                 //const Matrix3d& U,
                                                 const double& sigsum)
                                                 //const Matrix3d& V)
{

    Eigen::Matrix<double, 9, 1> g1;
    g1.block<3, 1>(0, 0) = 2 * F.col(0);
    g1.block<3, 1>(3, 0) = 2 * F.col(1);
    g1.block<3, 1>(6, 0) = 2 * F.col(2);
    Eigen::Matrix<double, 9, 9> H1 = 2 * Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix3d             mat_g2 = 4 * F * F.transpose() * F;
    Eigen::Matrix<double, 9, 1> g2;
    g2.block<3, 1>(0, 0)                  = mat_g2.col(0);
    g2.block<3, 1>(3, 0)                  = mat_g2.col(1);
    g2.block<3, 1>(6, 0)                  = mat_g2.col(2);
    Eigen::Matrix<double, 9, 9> D         = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> IkronFFT  = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> FTFkronI  = Eigen::Matrix<double, 9, 9>::Zero();
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


    //Matrix3d           U, V;
    //Matrix3d           sig;
    //SVDResult3D_double svdResult = QRSVD(F);
    ////Matrix3d U, sigma, V;
    //U   = svdResult.U;
    //sig = svdResult.SIGMA;
    //V   = svdResult.V;

    double f   = sigsum;
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

__device__ Vector<double, 9> flatenMatrix3d(const Eigen::Matrix3d& matrix)
{
    Vector<double, 9> result;
    result[0] = matrix(0, 0);
    result[1] = matrix(1, 0);
    result[2] = matrix(2, 0);
    result[3] = matrix(0, 1);
    result[4] = matrix(1, 1);
    result[5] = matrix(2, 1);
    result[6] = matrix(0, 2);
    result[7] = matrix(1, 2);
    result[8] = matrix(2, 2);

    return result;
}

__device__ Matrix<double, 9, 9> kron_3x3(const Matrix<double, 3, 3>& A,
                                         const Matrix<double, 3, 3>& B)
{
    Matrix<double, 9, 9> result = Matrix<double, 9, 9>::Zero();
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            double a = A(i, j);
            // 子块 (i,j) 放置 a * B
            for(int r = 0; r < 3; ++r)
            {
                for(int c = 0; c < 3; ++c)
                {
                    result(i * 3 + r, j * 3 + c) = a * B(r, c);
                }
            }
        }
    }
    return result;
}

__device__ void cal_S_and_SF_RF(const Eigen::Matrix3d& F,
                                Eigen::Matrix3d&       R,
                                Matrix<double, 9, 9>& S_F,
                                const Matrix<double, 9, 9>& R_F)
{
    // 固定矩阵 FT_F (9x9)
    Matrix<double, 9, 9> FT_F = Matrix<double, 9, 9>::Zero();
    FT_F(0, 0)                = 1.0;
    FT_F(3, 1)                = 1.0;
    FT_F(6, 2)                = 1.0;
    FT_F(1, 3)                = 1.0;
    FT_F(4, 4)                = 1.0;
    FT_F(7, 5)                = 1.0;
    FT_F(2, 6)                = 1.0;
    FT_F(5, 7)                = 1.0;
    FT_F(8, 8)                = 1.0;

    Eigen::Matrix3d      I_3        = Eigen::Matrix3d::Identity();
    Matrix<double, 9, 9> kron_I3_R = kron_3x3(I_3, R.transpose());
    Matrix<double, 9, 9> kron_FT_I3 = kron_3x3(F.transpose(), I_3);

    S_F = kron_I3_R + kron_FT_I3 * FT_F * R_F;
}




__device__ Eigen::Matrix<double, 9, 9> project_ARAPX_H_3D(
    const Eigen::Matrix<double, 3, 3>& F,
    const Eigen::Matrix<double, 3, 1>& Sigma,
    const Eigen::Matrix<double, 3, 3>& U,
    const Eigen::Matrix<double, 3, 3>& V,
    const double&                      mu,
    const double&                      lambda_)
{
    double               volume_stiff = lambda_;
    Matrix<double, 9, 9> hessian      = Matrix<double, 9, 9>::Zero();
    {
        Eigen::Matrix3d R        = U * V.transpose();
        Matrix3d        sigmaMat = Matrix3d::Zero();
        sigmaMat(0, 0)           = Sigma(0, 0);
        sigmaMat(1, 1)           = Sigma(1, 0);
        sigmaMat(2, 2)           = Sigma(2, 0);

        Eigen::Matrix3d      S   = V * sigmaMat * V.transpose();
        Matrix<double, 9, 9> S_F = Matrix<double, 9, 9>::Zero();
        Matrix<double, 9, 9> R_F = Matrix<double, 9, 9>::Zero();

        //Matrix3d sig3 = Sigma.asDiagonal();
        //double sigsum = Sigma(0) + Sigma(1) + Sigma(2);
        R_F           = compute_DRDF_Hessian(F, Sigma.sum());

        cal_S_and_SF_RF(F, R, S_F, R_F);


        double               J   = F.determinant();
        Eigen::Matrix3d      I_3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d      A   = S - Eigen::Matrix3d::Identity();
        Matrix<double, 9, 9> I_9 = Matrix<double, 9, 9>::Identity();

        double T1 = (A).trace();
        double T2 = (A * A).trace();
        double T3 = (A * A * A).trace();


        Eigen::Matrix3d      T2_d  = 2.0 * F - 2.0 * R;
        Matrix<double, 9, 9> T2_d2 = Matrix<double, 9, 9>::Zero();
        T2_d2                      = 2.0 * I_9 - 2.0 * R_F;
        //Matrix<double, 9, 9> T3_d2 = Matrix<double, 9, 9>::Zero();

        Matrix<double, 9, 9> kron_I3_S = kron_3x3(I_3, S);
        // 手动计算 kron(I_3, F)
        Matrix<double, 9, 9> kron_I3_F = kron_3x3(I_3, F);

        // 组合表达式
        Matrix<double, 9, 9> T3_d2 =
            3.0 * (kron_I3_S + kron_I3_F * S_F - 2.0 * I_9 + R_F);

        Matrix<double, 9, 9> volume_term_d2 = Matrix<double, 9, 9>::Zero();
        Eigen::Matrix<double, 9, 1> pJpF;  // = J * F.inverse().transpose();
        pJpF.block<3, 1>(0, 0) = F.col(1).cross(F.col(2));
        pJpF.block<3, 1>(3, 0) = F.col(2).cross(F.col(0));
        pJpF.block<3, 1>(6, 0) = F.col(0).cross(F.col(1));

        Eigen::Matrix3d f0hat;
        f0hat << 0, -F(2, 0), F(1, 0), F(2, 0), 0, -F(0, 0), -F(1, 0), F(0, 0), 0;
        Eigen::Matrix3d f1hat;
        f1hat << 0, -F(2, 1), F(1, 1), F(2, 1), 0, -F(0, 1), -F(1, 1), F(0, 1), 0;
        Eigen::Matrix3d f2hat;
        f2hat << 0, -F(2, 2), F(1, 2), F(2, 2), 0, -F(0, 2), -F(1, 2), F(0, 2), 0;
        Eigen::Matrix<double, 9, 9> HJ;
        HJ.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
        HJ.block<3, 3>(0, 3) = -f2hat;
        HJ.block<3, 3>(0, 6) = f1hat;
        HJ.block<3, 3>(3, 0) = f2hat;
        HJ.block<3, 3>(3, 3) = Eigen::Matrix3d::Zero();
        HJ.block<3, 3>(3, 6) = -f0hat;
        HJ.block<3, 3>(6, 0) = -f1hat;
        HJ.block<3, 3>(6, 3) = f0hat;
        HJ.block<3, 3>(6, 6) = Eigen::Matrix3d::Zero();

        volume_term_d2 += volume_stiff * pJpF * pJpF.transpose();
        volume_term_d2 += volume_stiff * (J - 1.0) * HJ;


#if arap_model == 2
        hessian = mu / 2.0 * T2_d2 + volume_term_d2;
#elif arap_model == 3
        hessian = mu / 2.0 * T2_d2 + mu / 6.0 * T3_d2 + volume_term_d2;
#elif arap_model == 4
        hessian =
            mu / 2.0 * T2_d2 + mu / 6.0 * T3_d2
            + lambda_ / 2.0
                  * (2.0 * (flatenMatrix3d(T2_d) * flatenMatrix3d(T2_d).transpose() + T2 * T2_d2))
            + volume_term_d2;
#endif
    }

    makePDSNK<double, 9>(hessian);
    return hessian;
}

__global__ void _calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                                const double3* vertexes,
                                                const uint4*   tetrahedras,
                                                const double*    volume,
                                                double3*         gradient,
                                                int              tetrahedraNum,
                                                const double*    lenRate,
                                                const double*    volRate,
                                                //uint4*           tet_ids,
                                                int              global_offset,
                                                Eigen::Matrix3d* triplet_values,
                                                int*             row_ids,
                                                int*             col_ids,
                                                double IPC_dt,int global_hessian_fem_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tetrahedraNum)
        return;

    Eigen::Matrix<double, 9, 12> PFPX = __computePFPX3D_Eigen_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    double I2 = __GEIGEN__::__squaredNorm(F);
    double I3 = __GEIGEN__::__Determiant(F);
#ifdef USE_SNK2
    __GEIGEN__::Matrix3x3d PEPF =
        __computePEPF_StableNHK3D2_double(F, I2, I3, lenRate[idx], volRate[idx]);
#elif USE_SNK1
    __GEIGEN__::Matrix3x3d PEPF =
        __computePEPF_StableNHK3D1_double(F, I2, I3, lenRate[idx], volRate[idx]);
#else
    Eigen::Matrix<double, 3, 3> MF, UU, VV;
    Eigen::Matrix<double, 3, 1> SS;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            MF(i, j) = F.m[i][j];
        }
    }
    __GEIGEN__::math::qr_svd(MF, SS, UU, VV);
    Eigen::Matrix<double, 3, 3> eso_PEPF =
        computePEPF_ARAPX_double(MF, UU, VV, SS, lenRate[idx], volRate[idx], 2);
    //Eigen::Matrix<double, 3, 3> eso_PEPF =
    //    computePEPF_ARAP_double(MF, UU, VV, lenRate[idx]);
    __GEIGEN__::Matrix3x3d PEPF;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            PEPF.m[i][j] = eso_PEPF(i, j);
        }
    }
#endif


    

    Eigen::Vector<double, 9> pepf = __GEIGEN__::__Mat3x3_to_vec9_Eigen_double(PEPF);


    //__GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);
    Eigen::Vector<double, 12> f = IPC_dt * IPC_dt * volume[idx] * PFPX.transpose() * pepf;
        //__GEIGEN__::__s_vec12_multiply(__GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf),
                                       //IPC_dt * IPC_dt * volume[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f[11]);
    }

#ifdef USE_SNK2
    Matrix<double, 9, 9> Hq;
    __project_StabbleNHK_H_3D2_makePD(Hq, F, I2, I3, lenRate[idx], volRate[idx]);
#elif USE_SNK1
    Eigen::Matrix<double, 9,9> Hq;
    __project_StabbleNHK_H_3D1_makePD(Hq, I3, lenRate[idx], volRate[idx], F);
#else
    Eigen::Matrix<double, 9, 9> Hq = project_ARAPX_H_3D(MF, SS, UU, VV, lenRate[idx], volRate[idx]);
    //Eigen::Matrix<double, 9, 9> Hq =
    //    project_ARAP_H_3D(SS, UU, VV, lenRate[idx]);
#endif

    Eigen::Matrix<double, 12, 12> H;

    __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply_Eigen(PFPX, Hq, H);

    H = volume[idx] * IPC_dt * IPC_dt * H;
    //Hessians[idx + offset] = H;
    //unsigned int* tetra = &(tet_ids[idx].x);


    unsigned int fem_index[4] = {tetrahedras[idx].x + global_hessian_fem_offset,
                        tetrahedras[idx].y + global_hessian_fem_offset,
                        tetrahedras[idx].z + global_hessian_fem_offset,
                        tetrahedras[idx].w + global_hessian_fem_offset};
    //write_triplet_fem<12, 12>(
    //    triplet_values, row_ids, col_ids, fem_index, H.m, global_offset + 16 * idx);

    int triplet_offset = global_offset + 10 * idx;
    int kk             = 0;
    for(int ii = 0; ii < 4; ii++)
    {
        for(int jj = ii; jj < 4; jj++)
        {
            int row                      = fem_index[ii];
            int col                      = fem_index[jj];
            if(row <= col)
            {
                row_ids[triplet_offset + kk] = row;
                col_ids[triplet_offset + kk] = col;
                triplet_values[triplet_offset + kk] = H.block<3, 3>(ii * 3, jj * 3);
                //kk++;
            }
            else
            {
                row_ids[triplet_offset + kk] = col;
                col_ids[triplet_offset + kk] = row;
                triplet_values[triplet_offset + kk] =
                    H.block<3, 3>(ii * 3, jj * 3).transpose();
            }
            kk++;
        }
    }
}

__device__ __host__ double g_sl(double s, double sLimit, double sHat)
{
    double a = sHat - s;
    return (2 * a * std::log((sLimit - s) / (sLimit - sHat)) + a * a / (sLimit - s))
           / ((sLimit - sHat) * (sLimit - sHat));
}

__device__ __host__ double h_sl(double s, double sLimit, double sHat)
{
    double a = sHat - s;
    double b = sLimit - s;
    double c = a / b;
    return (-2 * std::log(b / (sLimit - sHat)) + c * (c - 4))
           / ((sLimit - sHat) * (sLimit - sHat));
}


__device__ Matrix<double, 3, 2> computePEPF_strain_limit(Eigen::Matrix3d& U3x2,
                                                         Eigen::Matrix2d& V3x2,
                                                         Eigen::Vector2d& S3x2,
                                                         double sLimit,
                                                         double sHat)
{
    Matrix<double, 3, 2> db_div_dF;
    db_div_dF.setZero();
    bool triggered = false;
    for(int i = 0; i < 2; i++)
    {
        double s = S3x2[i];
        if(s > sHat)
        {
            triggered        = true;
            double db_div_dS = g_sl(s, sLimit, sHat);
            db_div_dF += U3x2.col(i) * (db_div_dS * V3x2.col(i).transpose());
        }
    }

    return db_div_dF;
}

__device__ Matrix<double, 6, 6> computeP2EPF2_strain_limit(Eigen::Matrix3d& U3x2,
                                                           Eigen::Matrix2d& V3x2,
                                                           Eigen::Vector2d& S3x2,
                                                           const Matrix2d& InverseDm,
                                                           double sLimit,
                                                           double sHat)
{
    Matrix<double, 6, 6> d2b_div_dF2;
    d2b_div_dF2.setZero();

    //Eigen::JacobiSVD<Matrix<double, 3, 2>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    bool triggered = false;
    for(int i = 0; i < 2; i++)
    {
        double s = S3x2[i];
        //assert(s < sLimit);
        if(s > sHat)
        {
            triggered = true;
        }
    }

    if(triggered)
    {
        Matrix3d V;
        V.setZero();
        V.block<2, 2>(0, 0) = V3x2;
        Vector3d sigma;
        sigma.setZero();
        sigma.segment<2>(0) = S3x2;

        Matrix<double, 9, 6> dU_div_dF;
        Matrix<double, 4, 6> dV_div_dF;
        Matrix<double, 9, 9> dU_div_dF_ext, dV_div_dF_ext;
        dU_div_dF_ext.setZero();
        dV_div_dF_ext.setZero();
        for(int cI = 0; cI < 3; ++cI)
        {
            int cI_post = (cI + 1) % 3;

            Matrix2d coefMtr;
            if(std::abs(sigma[cI] - sigma[cI_post]) < 2e-6)
            {
                coefMtr << sigma[cI] + 1e-6, sigma[cI_post] - 1e-6,
                    sigma[cI_post] - 1e-6, sigma[cI] + 1e-6;
            }
            else
            {
                coefMtr << sigma[cI], sigma[cI_post], sigma[cI_post], sigma[cI];
            }
            //Eigen::FullPivLU<Eigen::Matrix2d> solver(coefMtr);

            Vector2d b;
            for(int rowI = 0; rowI < 3; ++rowI)
            {
                for(int colI = 0; colI < 3; ++colI)
                {
                    b << U3x2(rowI, cI_post) * V(colI, cI),
                        -U3x2(rowI, cI) * V(colI, cI_post);
                    Vector2d wij21;

                    Matrix2d coefMtrTemp          = coefMtr;
                    coefMtrTemp.block<2, 1>(0, 0) = b;
                    wij21[0] = coefMtrTemp.determinant() / coefMtr.determinant();
                    coefMtrTemp                   = coefMtr;
                    coefMtrTemp.block<2, 1>(0, 1) = b;
                    wij21[1] = coefMtrTemp.determinant() / coefMtr.determinant();


                    dU_div_dF_ext.block<3, 1>(cI * 3, rowI + colI * 3) +=
                        wij21[0] * U3x2.col(cI_post);
                    dU_div_dF_ext.block<3, 1>(cI_post * 3, rowI + colI * 3) +=
                        -wij21[0] * U3x2.col(cI);

                    dV_div_dF_ext.block<3, 1>(cI * 3, rowI + colI * 3) +=
                        -wij21[1] * V.col(cI_post);
                    dV_div_dF_ext.block<3, 1>(cI_post * 3, rowI + colI * 3) +=
                        wij21[1] * V.col(cI);
                }
            }
        }
        dU_div_dF                   = dU_div_dF_ext.leftCols(6);
        dV_div_dF.block<2, 6>(0, 0) = dV_div_dF_ext.block<2, 6>(0, 0);
        dV_div_dF.block<2, 6>(2, 0) = dV_div_dF_ext.block<2, 6>(3, 0);

        for(int i = 0; i < 2; i++)
        {
            double s = S3x2[i];
            if(s > sHat)
            {
                double d2b_div_dS2 = h_sl(s, sLimit, sHat);
                Matrix<double, 3, 2> dS_div_dF = U3x2.col(i) * V3x2.col(i).transpose();
                Vector<double, 6> dS_div_dF_reshaped;

                for(int i = 0; i < 2; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        dS_div_dF_reshaped[i * 3 + j] = dS_div_dF(j, i);
                    }
                }

                d2b_div_dF2 += (dS_div_dF_reshaped * d2b_div_dS2)
                               * dS_div_dF_reshaped.transpose();

                Matrix<double, 6, 6> d2S_div_dF2;
                for(int Fij = 0; Fij < 6; ++Fij)
                {
                    Matrix<double, 3, 2> d2S_div_dF2ij =
                        dU_div_dF.block<3, 1>(i * 3, Fij) * V3x2.col(i).transpose()
                        + U3x2.col(i) * dV_div_dF.block<2, 1>(i * 2, Fij).transpose();
                    d2S_div_dF2.block<3, 1>(0, Fij) = d2S_div_dF2ij.col(0);
                    d2S_div_dF2.block<3, 1>(3, Fij) = d2S_div_dF2ij.col(1);
                }
                double db_div_dS = g_sl(s, sLimit, sHat);
                d2b_div_dF2 += db_div_dS * d2S_div_dF2;
            }
        }
    }

    //assert(d2b_div_dF2.allFinite());
    return d2b_div_dF2;
}


__global__ void _calculate_triangle_fem_strain_limiting_gradient_hessian(
    __GEIGEN__::Matrix2x2d* trimInverses,
    const double3*          vertexes,
    const uint3*            triangles,
    __GEIGEN__::Matrix9x9d* Hessians,
    uint32_t                offset,
    const double*           area,
    double3*                gradient,
    int                     triangleNum,
    Eigen::Matrix3d*        U3x2,
    Eigen::Matrix2d*        V3x2,
    Eigen::Vector2d*        S3x2,
    double                  IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= triangleNum)
        return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d mF = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);
    Eigen::Matrix<double, 3, 2> F;
    F.block<3, 1>(0, 0) = Eigen::Vector3d(mF.m[0][0], mF.m[1][0], mF.m[2][0]);
    F.block<3, 1>(0, 1) = Eigen::Vector3d(mF.m[0][1], mF.m[1][1], mF.m[2][1]);


    double slimit = 1.1;
    double sHat   = 1;
    double sKappa = 1e3;

    Matrix<double, 3, 2> PEPF =
        computePEPF_strain_limit(U3x2[idx], V3x2[idx], S3x2[idx], slimit, sHat);


    __GEIGEN__::Vector6 pepf;  //
    //= __GEIGEN__::__Mat3x2_to_vec6_double(PEPF);
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            pepf.v[i * 3 + j] = PEPF(j, i);
        }
    }

    __GEIGEN__::Matrix9x6d PFPXTranspose = __GEIGEN__::__Transpose6x9(PFPX);
    __GEIGEN__::Vector9    f =
        __GEIGEN__::__s_vec9_multiply(__GEIGEN__::__M9x6_v6_multiply(PFPXTranspose, pepf),
                                      sKappa * IPC_dt * IPC_dt * area[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);
    if(true)
    {
        atomicAdd(&(gradient[triangles[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[triangles[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[triangles[idx].x].z), f.v[2]);
        atomicAdd(&(gradient[triangles[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[triangles[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[triangles[idx].y].z), f.v[5]);
        atomicAdd(&(gradient[triangles[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[triangles[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[triangles[idx].z].z), f.v[8]);
    }

    Matrix2d triInvsm;

    triInvsm(0, 0) = trimInverses[idx].m[0][0];
    triInvsm(0, 1) = trimInverses[idx].m[0][1];
    triInvsm(1, 0) = trimInverses[idx].m[1][0];
    triInvsm(1, 1) = trimInverses[idx].m[1][1];

    Matrix<double, 6, 6> He =
        computeP2EPF2_strain_limit(U3x2[idx], V3x2[idx], S3x2[idx], triInvsm, slimit, sHat);

    makePDSNK<double, 6>(He);

    __GEIGEN__::Matrix6x6d Hq;

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            Hq.m[i][j] = He(i, j);
        }
    }

    __GEIGEN__::Matrix9x6d M9x6_temp =
        __GEIGEN__::__M9x6_M6x6_Multiply(PFPXTranspose, Hq);
    __GEIGEN__::Matrix9x9d H = __GEIGEN__::__M9x6_M6x9_Multiply(M9x6_temp, PFPX);
    H = __GEIGEN__::__s_M9x9_Multiply(H, sKappa * area[idx] * IPC_dt * IPC_dt);
    //Hessians[idx + offset] = H;
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            Hessians[idx + offset].m[i][j] += H.m[i][j];
        }
    }
}

__global__ void _calculate_triangle_fem_deformationF(__GEIGEN__::Matrix2x2d* trimInverses,
                                                     const double3* vertexes,
                                                     const uint3*   triangles,
                                                     int            triangleNum,
                                                     Eigen::Matrix<double, 3, 2>* F3x2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= triangleNum)
        return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d mF = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);
    Eigen::Matrix<double, 3, 2> F;
    F.block<3, 1>(0, 0) = Eigen::Vector3d(mF.m[0][0], mF.m[1][0], mF.m[2][0]);
    F.block<3, 1>(0, 1) = Eigen::Vector3d(mF.m[0][1], mF.m[1][1], mF.m[2][1]);

    F3x2[idx] = F;
}

__global__ void _calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* trimInverses,
                                                         const double3* vertexes,
                                                         const uint3* triangles,
                                                         const double* area,
                                                         double3*      gradient,
                                                         int    triangleNum,
                                                         double stretchStiff,
                                                         double shearhStiff,
                                                         double IPC_dt,
                                                         int    global_offset,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double strainRate,
                                                         int global_hessian_fem_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= triangleNum)
        return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __GEIGEN__::Matrix3x2d PEPF =
        __computePEPF_BaraffWitkinStretch_double(F, stretchStiff, shearhStiff, strainRate);


    __GEIGEN__::Vector6 pepf = __GEIGEN__::__Mat3x2_to_vec6_double(PEPF);


    __GEIGEN__::Matrix9x6d PFPXTranspose = __GEIGEN__::__Transpose6x9(PFPX);
    __GEIGEN__::Vector9    f =
        __GEIGEN__::__s_vec9_multiply(__GEIGEN__::__M9x6_v6_multiply(PFPXTranspose, pepf),
                                      IPC_dt * IPC_dt * area[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[triangles[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[triangles[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[triangles[idx].x].z), f.v[2]);
        atomicAdd(&(gradient[triangles[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[triangles[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[triangles[idx].y].z), f.v[5]);
        atomicAdd(&(gradient[triangles[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[triangles[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[triangles[idx].z].z), f.v[8]);
    }

    __GEIGEN__::Matrix6x6d Hq = __GEIGEN__::__s_M6x6_Multiply(
        __project_BaraffWitkinStretch_H(F, strainRate), stretchStiff);

    __GEIGEN__::__Mat_add(
        Hq, __GEIGEN__::__s_M6x6_Multiply(__project_BaraffWitkinShear_H(F), shearhStiff), Hq);

    __GEIGEN__::Matrix9x6d M9x6_temp =
        __GEIGEN__::__M9x6_M6x6_Multiply(PFPXTranspose, Hq);
    __GEIGEN__::Matrix9x9d H = __GEIGEN__::__M9x6_M6x9_Multiply(M9x6_temp, PFPX);
    H = __GEIGEN__::__s_M9x9_Multiply(H, area[idx] * IPC_dt * IPC_dt);
    //Hessians[idx + offset] = H;

    unsigned int global_fem_offset[3] = {triangles[idx].x + global_hessian_fem_offset,
                                         triangles[idx].y + global_hessian_fem_offset,
                                         triangles[idx].z + global_hessian_fem_offset};

    write_triplet_fem<9, 9>(
        triplet_values, row_ids, col_ids, global_fem_offset, H.m, global_offset + idx * 6);
}

__global__ void _calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* trimInverses,
                                                 const double3* vertexes,
                                                 const uint3*   triangles,
                                                 const double*  area,
                                                 double3*       gradient,
                                                 int            triangleNum,
                                                 double         stretchStiff,
                                                 double         shearhStiff,
                                                 double         IPC_dt,
                                                 double         strainRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= triangleNum)
        return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __GEIGEN__::Matrix3x2d PEPF =
        __computePEPF_BaraffWitkinStretch_double(F, stretchStiff, shearhStiff, strainRate);


    __GEIGEN__::Vector6 pepf = __GEIGEN__::__Mat3x2_to_vec6_double(PEPF);


    __GEIGEN__::Matrix9x6d PFPXTranspose = __GEIGEN__::__Transpose6x9(PFPX);
    __GEIGEN__::Vector9    f =
        __GEIGEN__::__s_vec9_multiply(__GEIGEN__::__M9x6_v6_multiply(PFPXTranspose, pepf),
                                      IPC_dt * IPC_dt * area[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[triangles[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[triangles[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[triangles[idx].x].z), f.v[2]);
        atomicAdd(&(gradient[triangles[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[triangles[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[triangles[idx].y].z), f.v[5]);
        atomicAdd(&(gradient[triangles[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[triangles[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[triangles[idx].z].z), f.v[8]);
    }
}

__global__ void _calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                                        const double3*          vertexes,
                                        const uint4*            tetrahedras,
                                        const double*           volume,
                                        double3*                gradient,
                                        int                     tetrahedraNum,
                                        double*                  lenRate,
                                        double*                  volRate,
                                        double                  dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tetrahedraNum)
        return;

    __GEIGEN__::Matrix9x12d PFPX = __computePFPX3D_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    double                 I2 = __GEIGEN__::__squaredNorm(F);
    double                 I3 = __GEIGEN__::__Determiant(F);
#ifdef USE_SNK2
    __GEIGEN__::Matrix3x3d PEPF =
        __computePEPF_StableNHK3D2_double(F, I2, I3, lenRate[idx], volRate[idx]);
#elif USE_SNK1
    __GEIGEN__::Matrix3x3d PEPF =
        __computePEPF_StableNHK3D1_double(F, I2, I3, lenRate[idx], volRate[idx]);
#else
    Eigen::Matrix<double, 3, 3> MF, UU, VV;
    Eigen::Matrix<double, 3, 1> SS;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            MF(i, j) = F.m[i][j];
        }
    }
    __GEIGEN__::math::qr_svd(MF, SS, UU, VV);
    Eigen::Matrix<double, 3, 3> eso_PEPF =
        computePEPF_ARAP_double(MF, UU, VV, lenRate[idx]);
    __GEIGEN__::Matrix3x3d PEPF;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            PEPF.m[i][j] = eso_PEPF(i, j);
        }
    }
#endif


    //__GEIGEN__::Matrix3x3d PEPF = Iso_PEPF;

    __GEIGEN__::Vector9 pepf = __GEIGEN__::__Mat3x3_to_vec9_double(PEPF);

    __GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);

    __GEIGEN__::Vector12 f = __GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf);

    for(int i = 0; i < 12; i++)
    {
        f.v[i] = volume[idx] * f.v[i] * dt * dt;
    }

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }
}



__global__ void _calculate_bending_gradient_hessian(const double3* vertexes,
                                                    const double3* rest_vertexes,
                                                    const uint2* edges,
                                                    const uint2* edges_adj_vertex,
                                                    double3* gradient,
                                                    int      edgeNum,
                                                    double   bendStiff,
                                                    int      global_offset,
                                                    Eigen::Matrix3d* triplet_values,
                                                    int*   row_ids,
                                                    int*   col_ids,
                                                    double IPC_dt,
                                                    int global_hessian_fem_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= edgeNum)
        return;
    uint2 edge = edges[idx];
    uint2 adj  = edges_adj_vertex[idx];
    if(adj.y == -1)
    {
        int default_index[4] = {0, 1, 2, 3};
        int gtrioff          = global_offset + idx * 10;
        int kk               = 0;
        for(int ii = 0; ii < 4; ii++)
        {
            for(int jj = ii; jj < 4; jj++)
            {
                row_ids[gtrioff + kk]        = 0;
                col_ids[gtrioff + kk]        = 0;
                triplet_values[gtrioff + kk].setZero();
                kk++;
            }
        }

        return;
    }
    auto x0 = vertexes[edge.x];
    auto x1 = vertexes[edge.y];
    auto x2 = vertexes[adj.x];
    auto x3 = vertexes[adj.y];

    Matrix<double, 1, 12>  grad_transpose;
    Matrix<double, 12, 12> H;

    Eigen::Matrix<double, 3, 1> x0_eigen =
        Eigen::Matrix<double, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<double, 3, 1> x1_eigen =
        Eigen::Matrix<double, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<double, 3, 1> x2_eigen =
        Eigen::Matrix<double, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<double, 3, 1> x3_eigen =
        Eigen::Matrix<double, 3, 1>(x3.x, x3.y, x3.z);
    double t =
        LibShell::edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, &grad_transpose, &H);
    //            cout << "t: " << t << endl;

    auto   rest_x0 = rest_vertexes[edge.x];
    auto   rest_x1 = rest_vertexes[edge.y];
    auto   rest_x2 = rest_vertexes[adj.x];
    auto   rest_x3 = rest_vertexes[adj.y];
    double length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    Eigen::Vector3d rest_x0_eigen =
        Eigen::Vector3d(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Vector3d rest_x1_eigen =
        Eigen::Vector3d(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Vector3d rest_x2_eigen =
        Eigen::Vector3d(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Vector3d rest_x3_eigen =
        Eigen::Vector3d(rest_x3.x, rest_x3.y, rest_x3.z);
    double rest_t = LibShell::edgeTheta(
        rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);

    H = 2 * ((t - rest_t) * H + grad_transpose.transpose() * grad_transpose);
    grad_transpose = 2 * (t - rest_t) * grad_transpose * IPC_dt * IPC_dt * length * bendStiff;
    makePDSNK<double, 12>(H);
    //__GEIGEN__::Vector12 f;
    //for(int i = 0; i < 12; i++)
    //{
    //    f.v[i] = IPC_dt * IPC_dt * length * grad_transpose(0, i) * bendStiff;
    //}


    {
        atomicAdd(&(gradient[edge.x].x), grad_transpose(0, 0));
        atomicAdd(&(gradient[edge.x].y), grad_transpose(0, 1));
        atomicAdd(&(gradient[edge.x].z), grad_transpose(0, 2));

        atomicAdd(&(gradient[edge.y].x), grad_transpose(0, 3));
        atomicAdd(&(gradient[edge.y].y), grad_transpose(0, 4));
        atomicAdd(&(gradient[edge.y].z), grad_transpose(0, 5));

        atomicAdd(&(gradient[adj.x].x), grad_transpose(0, 6));
        atomicAdd(&(gradient[adj.x].y), grad_transpose(0, 7));
        atomicAdd(&(gradient[adj.x].z), grad_transpose(0, 8));

        atomicAdd(&(gradient[adj.y].x), grad_transpose(0, 9));
        atomicAdd(&(gradient[adj.y].y), grad_transpose(0, 10));
        atomicAdd(&(gradient[adj.y].z), grad_transpose(0, 11));
    }

    H = IPC_dt * IPC_dt * length * bendStiff * H;
    //Hessians[idx + offset] = d_H;
    uint4 global_index = make_uint4(edge.x, edge.y, adj.x, adj.y);
    //Indices[idx + offset]  = global_index;
    unsigned int fem_global_index[4] = {edge.x + global_hessian_fem_offset,
                                        edge.y + global_hessian_fem_offset,
                                        adj.x + global_hessian_fem_offset,
                                        adj.y + global_hessian_fem_offset};

    int gtrioff = global_offset + idx * 10;
    int kk      = 0;
    for(int ii = 0; ii < 4; ii++)
    {
        for(int jj = ii; jj < 4; jj++)
        {
            int row = fem_global_index[ii];
            int col = fem_global_index[jj];
            if(row <= col)
            {
                row_ids[gtrioff + kk]        = row;
                col_ids[gtrioff + kk]        = col;
                triplet_values[gtrioff + kk] = H.block<3, 3>(ii * 3, jj * 3);
            }
            else
            {
                row_ids[gtrioff + kk] = col;
                col_ids[gtrioff + kk] = row;
                triplet_values[gtrioff + kk] = H.block<3, 3>(ii * 3, jj * 3).transpose();
            }
            kk++;
        }
    }
}



#ifdef USE_QUADRATIC_BENDING

// Host function: Precompute Q matrices for quadratic bending
// This function is called once during initialization
void PrepareQuadBendingQ(const double3*   rest_vertexes,
                         const uint2*     tri_edges,
                         const uint2*     tri_edge_adj_vertex,
                         int              edge_num,
                         Eigen::Matrix4d* Q_output  // Output: host memory
)
{
    // Completely copied from CPU-IPC PrepQuadBending logic
    int error_count    = 0;
    int boundary_count = 0;

    for(int i = 0; i < edge_num; i++)
    {
        const uint2& edge = tri_edges[i];
        const uint2& adj  = tri_edge_adj_vertex[i];

        // Skip boundary edges (adj.y == -1)
        if(adj.y == (unsigned int)(-1))
        {
            Q_output[i].setZero();
            boundary_count++;
            continue;
        }

        // Get 4 vertices in rest configuration
        // Vertex ordering MUST match CPU-IPC: [edge[0], edge[1], trig1_outer, trig2_outer]
        Eigen::Vector3d v0(rest_vertexes[edge.x].x,
                           rest_vertexes[edge.x].y,
                           rest_vertexes[edge.x].z);  // edge vertex 0
        Eigen::Vector3d v1(rest_vertexes[edge.y].x,
                           rest_vertexes[edge.y].y,
                           rest_vertexes[edge.y].z);  // edge vertex 1
        Eigen::Vector3d v2(rest_vertexes[adj.x].x,
                           rest_vertexes[adj.x].y,
                           rest_vertexes[adj.x].z);  // trig1 outer
        Eigen::Vector3d v3(rest_vertexes[adj.y].x,
                           rest_vertexes[adj.y].y,
                           rest_vertexes[adj.y].z);  // trig2 outer

        // Compute edges (EXACTLY copied from CPU-IPC)
        // v0=edge[0], v1=edge[1], v2=trig1_outer, v3=trig2_outer
        Eigen::Vector3d e0 = v1 - v0;  // edge vector (v1 - v0)
        Eigen::Vector3d e1 = v2 - v0;  // from edge[0] to trig1_outer
        Eigen::Vector3d e2 = v3 - v0;  // from edge[0] to trig2_outer
        Eigen::Vector3d e3 = v2 - v1;  // from edge[1] to trig1_outer
        Eigen::Vector3d e4 = v3 - v1;  // from edge[1] to trig2_outer

        // Compute triangle areas with safety check
        Eigen::Vector3d cross01 = e0.cross(e1);
        Eigen::Vector3d cross02 = e0.cross(e2);
        double          norm01  = cross01.norm();
        double          norm02  = cross02.norm();

        if(norm01 < 1e-12 || norm02 < 1e-12)
        {
            // Degenerate triangle
            Q_output[i].setZero();
            error_count++;
            continue;
        }

        double a0 = 0.5 * norm01;  // triangle 1 area
        double a1 = 0.5 * norm02;  // triangle 2 area

        // Cotangent function with safety check
        auto cot = [](const Eigen::Vector3d& e0, const Eigen::Vector3d& e1) -> double
        {
            double cross_norm = e0.cross(e1).norm();
            if(cross_norm < 1e-12)
                return 0.0;
            return e0.dot(e1) / cross_norm;
        };

        // Compute cotangent values (copied from CPU-IPC)
        double c01 = cot(e0, e1);
        double c02 = cot(e0, e2);
        double c03 = cot(-e0, e3);
        double c04 = cot(-e0, e4);

        // Build K vector (copied from CPU-IPC formula)
        Eigen::Vector4d K;
        K << c03 + c04, c01 + c02, -c01 - c03, -c02 - c04;

        // Compute Q matrix: Q = 3 / (a0 + a1) * K * K^T
        double area_sum = a0 + a1;
        if(area_sum < 1e-12)
        {
            Q_output[i].setZero();
            error_count++;
            continue;
        }

        Q_output[i] = 3.0 / area_sum * K * K.transpose();

        // Verify no NaN or Inf
        bool has_invalid = false;
        for(int r = 0; r < 4; r++)
        {
            for(int c = 0; c < 4; c++)
            {
                if(std::isnan(Q_output[i](r, c)) || std::isinf(Q_output[i](r, c)))
                {
                    has_invalid = true;
                    break;
                }
            }
            if(has_invalid)
                break;
        }

        if(has_invalid)
        {
            Q_output[i].setZero();
            error_count++;
        }
    }

    if(error_count > 0)
    {
        printf("PrepareQuadBendingQ: %d edges with errors (set to zero)\n", error_count);
    }
    printf("PrepareQuadBendingQ: %d boundary, %d errors, %d valid (total %d)\n",
           boundary_count,
           error_count,
           edge_num - boundary_count - error_count,
           edge_num);
}

// Calculate quadratic bending gradient and Hessian
// Gradient: ∇E_i = k * dt² * Σ_j Q_ij * x_j  (simple matrix-vector product)
// Hessian: H_ij = k * dt² * Q_ij * I_3x3  (constant matrix, much simpler than LibShell!)
__global__ void _calculate_quad_bending_gradient_hessian(const double3* vertexes,
                                                         const double3* rest_vertexes,
                                                         const uint2* edges,
                                                         const uint2* edges_adj_vertex,
                                                         const Eigen::Matrix4d* quad_bending_Q,
                                                         double3* gradient,
                                                         int      edgeNum,
                                                         double   bendStiff,
                                                         int      global_offset,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double IPC_dt,
                                                         int global_hessian_fem_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= edgeNum)
        return;

    uint2 edge = edges[idx];
    uint2 adj  = edges_adj_vertex[idx];

    // Boundary edge: set zeros to triplet
    if(adj.y == (unsigned int)(-1))
    {
        int gtrioff = global_offset + idx * 10;
        for(int kk = 0; kk < 10; kk++)
        {
            row_ids[gtrioff + kk] = 0;
            col_ids[gtrioff + kk] = 0;
            triplet_values[gtrioff + kk].setZero();
        }
        return;
    }

    const Eigen::Matrix4d& Q = quad_bending_Q[idx];

    // Collect current positions (MUST match CPU-IPC: [edge[0], edge[1], trig1_outer, trig2_outer])
    auto x0 = vertexes[edge.x];  // edge[0]
    auto x1 = vertexes[edge.y];  // edge[1]
    auto x2 = vertexes[adj.x];   // trig1_outer
    auto x3 = vertexes[adj.y];   // trig2_outer

    Eigen::Vector3d x_eigen[4];
    x_eigen[0] = Eigen::Vector3d(x0.x, x0.y, x0.z);
    x_eigen[1] = Eigen::Vector3d(x1.x, x1.y, x1.z);
    x_eigen[2] = Eigen::Vector3d(x2.x, x2.y, x2.z);
    x_eigen[3] = Eigen::Vector3d(x3.x, x3.y, x3.z);

    // Compute gradient: g_i = Q_ij * x_j
    Eigen::Matrix<double, 12, 1> grad;
    for(int i = 0; i < 4; i++)
    {
        Eigen::Vector3d g_i = Eigen::Vector3d::Zero();
        for(int j = 0; j < 4; j++)
        {
            g_i += Q(i, j) * x_eigen[j];
        }
        grad.block<3, 1>(i * 3, 0) = g_i;
    }

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];

    double length = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));

    // Scale by stiffness and time step
    double coef = length * bendStiff * IPC_dt * IPC_dt;
    grad *= coef;

    // Accumulate gradient using atomic operations
    // Order matches Q matrix: [edge[0], edge[1], trig1_outer, trig2_outer]
    unsigned int vertex_indices[4] = {edge.x, edge.y, adj.x, adj.y};
    for(int i = 0; i < 4; i++)
    {
        atomicAdd(&(gradient[vertex_indices[i]].x), grad(i * 3 + 0));
        atomicAdd(&(gradient[vertex_indices[i]].y), grad(i * 3 + 1));
        atomicAdd(&(gradient[vertex_indices[i]].z), grad(i * 3 + 2));
    }

    // Compute Hessian: H_ij = Q_ij * I_3x3
    // The Hessian is constant (doesn't depend on current position)!
    // This is the key advantage of quadratic bending over angle-based bending.
    Eigen::Matrix<double, 12, 12> H;
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            // H_ij = Q_ij * I_3x3
            H.block<3, 3>(i * 3, j * 3) = Q(i, j) * Eigen::Matrix3d::Identity();
        }
    }

    // Scale by stiffness and time step
    H *= coef;

    // Make positive definite (optional, Q should already be PSD)
    // Temporarily disable this to debug
    // makePDSNK<double, 12>(H);

    // Write upper triangular blocks to triplet format
    unsigned int fem_global_index[4] = {vertex_indices[0] + global_hessian_fem_offset,
                                        vertex_indices[1] + global_hessian_fem_offset,
                                        vertex_indices[2] + global_hessian_fem_offset,
                                        vertex_indices[3] + global_hessian_fem_offset};

    int gtrioff = global_offset + idx * 10;
    int kk      = 0;
    for(int ii = 0; ii < 4; ii++)
    {
        for(int jj = ii; jj < 4; jj++)
        {
            int row = fem_global_index[ii];
            int col = fem_global_index[jj];

            if(row <= col)
            {
                row_ids[gtrioff + kk]        = row;
                col_ids[gtrioff + kk]        = col;
                triplet_values[gtrioff + kk] = H.block<3, 3>(ii * 3, jj * 3);
            }
            else
            {
                row_ids[gtrioff + kk] = col;
                col_ids[gtrioff + kk] = row;
                triplet_values[gtrioff + kk] = H.block<3, 3>(jj * 3, ii * 3).transpose();
            }
            kk++;
        }
    }
}


__device__ double __cal_quad_bending_energy(const double3* vertexes,
                                            const double3* rest_vertexes,
                                            const uint2&   edge,
                                            const uint2&   adj,
                                            const Eigen::Matrix4d& Q,
                                            double                 bendStiff)
{
    if(adj.y == (unsigned int)(-1))
        return 0;

    // Get 4 vertices (MUST match CPU-IPC order)
    // [edge[0], edge[1], trig1_outer, trig2_outer]
    double3 x[4];
    x[0] = vertexes[edge.x];  // edge vertex 0
    x[1] = vertexes[edge.y];  // edge vertex 1
    x[2] = vertexes[adj.x];   // trig1 outer
    x[3] = vertexes[adj.y];   // trig2 outer

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];

    double length = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));

    // E = 0.5 * k * Σ_ij Q_ij * x_i · x_j
    double energy = 0.0;
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            double Qij = Q(i, j);
            energy += Qij * (x[i].x * x[j].x + x[i].y * x[j].y + x[i].z * x[j].z);
        }
    }

    return 0.5 * bendStiff * length * energy;
}

#endif


__device__ double __cal_strainL_energy(Eigen::Vector2d& sigma, double area)
{
    double slimit   = 1.1;
    double sHat     = 1;
    double sKappa   = 1e3;
    double temp_sum = 0;
    for(int i = 0; i < 2; i++)
    {
        double s = sigma[i];
        //assert(s < sLimit);
        if(s > sHat)
        {
            temp_sum += -(sHat - s) * (sHat - s)
                        * std::log((slimit - s) / (slimit - sHat))
                        / ((slimit - sHat) * (slimit - sHat)) * area;
        }
    }
    return temp_sum * sKappa;
}


__device__ double __cal_bending_energy(const double3* vertexes,
                                       const double3* rest_vertexes,
                                       const uint2&   edge,
                                       const uint2&   adj,
                                       double         length,
                                       double         bendStiff)
{
    if(adj.y == -1)
        return 0;
    auto                        x0 = vertexes[edge.x];
    auto                        x1 = vertexes[edge.y];
    auto                        x2 = vertexes[adj.x];
    auto                        x3 = vertexes[adj.y];
    Eigen::Matrix<double, 3, 1> x0_eigen =
        Eigen::Matrix<double, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<double, 3, 1> x1_eigen =
        Eigen::Matrix<double, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<double, 3, 1> x2_eigen =
        Eigen::Matrix<double, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<double, 3, 1> x3_eigen =
        Eigen::Matrix<double, 3, 1>(x3.x, x3.y, x3.z);
    double t = LibShell::edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, nullptr, nullptr);
    //            cout << "t: " << t << endl;

    auto                        rest_x0 = rest_vertexes[edge.x];
    auto                        rest_x1 = rest_vertexes[edge.y];
    auto                        rest_x2 = rest_vertexes[adj.x];
    auto                        rest_x3 = rest_vertexes[adj.y];
    Eigen::Matrix<double, 3, 1> rest_x0_eigen =
        Eigen::Matrix<double, 3, 1>(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Matrix<double, 3, 1> rest_x1_eigen =
        Eigen::Matrix<double, 3, 1>(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Matrix<double, 3, 1> rest_x2_eigen =
        Eigen::Matrix<double, 3, 1>(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Matrix<double, 3, 1> rest_x3_eigen =
        Eigen::Matrix<double, 3, 1>(rest_x3.x, rest_x3.y, rest_x3.z);
    double rest_t = LibShell::edgeTheta(
        rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);
    double bend_energy = bendStiff * (t - rest_t) * (t - rest_t) * length;
    return bend_energy;
}


double calculateVolum(const double3* vertexes, const uint4& index)
{
    int     id0 = 0;
    int     id1 = 1;
    int     id2 = 2;
    int     id3 = 3;
    double  o1x = vertexes[index.y].x - vertexes[index.x].x;
    double  o1y = vertexes[index.y].y - vertexes[index.x].y;
    double  o1z = vertexes[index.y].z - vertexes[index.x].z;
    double3 OA  = make_double3(o1x, o1y, o1z);

    double  o2x = vertexes[index.z].x - vertexes[index.x].x;
    double  o2y = vertexes[index.z].y - vertexes[index.x].y;
    double  o2z = vertexes[index.z].z - vertexes[index.x].z;
    double3 OB  = make_double3(o2x, o2y, o2z);

    double  o3x = vertexes[index.w].x - vertexes[index.x].x;
    double  o3y = vertexes[index.w].y - vertexes[index.x].y;
    double  o3z = vertexes[index.w].z - vertexes[index.x].z;
    double3 OC  = make_double3(o3x, o3y, o3z);

    double3 heightDir  = __GEIGEN__::__v_vec_cross(OA, OB);  //OA.cross(OB);
    double  bottomArea = __GEIGEN__::__norm(heightDir);      //heightDir.norm();
    heightDir          = __GEIGEN__::__normalized(heightDir);

    double volum = bottomArea * __GEIGEN__::__v_vec_dot(heightDir, OC) / 6;
    return volum > 0 ? volum : -volum;
}

double calculateArea(const double3* vertexes, const uint3& index)
{
    //double a = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]));
    //double b = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]));
    //double c = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.y], vertexes[index.z]));
    //double s = (a + b + c) / 2;
    //return sqrt(s * (s - a) *
    //    (s - b) * (s - c));
    double3 v10  = __GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]);
    double3 v20  = __GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]);
    double  area = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v10, v20));
    return 0.5 * area;
}
