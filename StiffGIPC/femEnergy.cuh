//
// femEnergy.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef __FEM_ENERGY_CUH__
#define __FEM_ENERGY_CUH__
#include <cstdint>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "Eigen/Eigen"
using namespace Eigen;
__device__ __host__ void __calculateDm2D_double(const double3* vertexes,
                                                const uint3&   index,
                                                __GEIGEN__::Matrix2x2d& M);
__device__ __host__ void __calculateDs2D_double(const double3* vertexes,
                                                const uint3&   index,
                                                __GEIGEN__::Matrix3x2d& M);
__device__ __host__ void __calculateDms3D_double(const double3* vertexes,
                                                 const uint4&   index,
                                                 __GEIGEN__::Matrix3x3d& M);
__device__ __GEIGEN__::Matrix6x6d __project_BaraffWitkinStretch_H(const __GEIGEN__::Matrix3x2d& F,
                                                                  double strainRate);
__device__ __GEIGEN__::Matrix9x9d __project_StabbleNHK_H_3D(const __GEIGEN__::Matrix3x3d& sigma,
                                                            const __GEIGEN__::Matrix3x3d& U,
                                                            const __GEIGEN__::Matrix3x3d& V,
                                                            const double& lengthRate,
                                                            const double& volumRate);

__device__ __GEIGEN__::Matrix9x9d project_ARAP_H_3D(const __GEIGEN__::Matrix3x3d& Sigma,
                                                    const __GEIGEN__::Matrix3x3d& U,
                                                    const __GEIGEN__::Matrix3x3d& V,
                                                    const double& lengthRate);

__device__ __GEIGEN__::Matrix3x3d computePEPF_ARAP_double(const __GEIGEN__::Matrix3x3d& F,
                                                          const __GEIGEN__::Matrix3x3d& Sigma,
                                                          const __GEIGEN__::Matrix3x3d& U,
                                                          const __GEIGEN__::Matrix3x3d& V,
                                                          const double& lengthRate);

__device__ __GEIGEN__::Matrix9x9d __project_ANIOSI5_H_3D(
    const __GEIGEN__::Matrix3x3d& F,
    const __GEIGEN__::Matrix3x3d& sigma,
    const __GEIGEN__::Matrix3x3d& U,
    const __GEIGEN__::Matrix3x3d& V,
    const double3&                fiber_direction,
    const double&                 scale,
    const double&                 contract_length);

__device__ __GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D2_double(
    const __GEIGEN__::Matrix3x3d& F, const double& I2, const double& I3, double lengthRate, double volumRate);



__device__ __GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D1_double(
    const __GEIGEN__::Matrix3x3d& F, const double& I2, const double& I3, double lengthRate, double volumRate);


__device__ __GEIGEN__::Matrix3x2d __computePEPF_BaraffWitkinStretch_double(
    const __GEIGEN__::Matrix3x2d& F, double stretchStiff, double shearStiff, double strainRate);
__device__ __GEIGEN__::Matrix3x3d __computePEPF_Aniostropic3D_double(
    const __GEIGEN__::Matrix3x3d& F,
    double3                       fiber_direction,
    const double&                 scale,
    const double&                 contract_length);
__device__ double __cal_BaraffWitkinStretch_energy(const double3* vertexes,
                                                   const uint3&   triangle,
                                                   const __GEIGEN__::Matrix2x2d& triDmInverse,
                                                   const double& area,
                                                   const double& stretchStiff,
                                                   const double& shearhStiff,
                                                   double        strainRate);
__device__ double __cal_StabbleNHK_energy1_3D(const double3* vertexes,
                                             const uint4&   tetrahedra,
                                             const __GEIGEN__::Matrix3x3d& DmInverse,
                                             const double& volume,
                                             const double& lenRate,
                                             const double& volRate);

__device__ double __cal_StabbleNHK_energy2_3D(const double3* vertexes,
                                             const uint4&   tetrahedra,
                                             const __GEIGEN__::Matrix3x3d& DmInverse,
                                             const double& volume,
                                             const double& lenRate,
                                             const double& volRate);

__device__ double __cal_ARAP_energy_3D(const double3*                vertexes,
                                       const uint4&                  tetrahedra,
                                       const __GEIGEN__::Matrix3x3d& DmInverse,
                                       const double&                 volume,
                                       const double&                 lenRate);


__device__ double __cal_bending_energy(const double3* vertexes,
                                       const double3* rest_vertexes,
                                       const uint2&   edge,
                                       const uint2&   adj,
                                       double         length,
                                       double         bendStiff);

__device__ double __cal_strainL_energy(Eigen::Vector2d& sigma, double area);


__global__ void _calculate_bending_gradient_hessian(const double3* vertexes,
                                                    const double3* rest_vertexes,
                                                    const uint2* edges,
                                                    const uint2* edges_adj_vertex,
                                                    __GEIGEN__::Matrix12x12d* Hessians,
                                                    uint4*   Indices,
                                                    uint32_t offset,
                                                    double3* gradient,
                                                    int      edgeNum,
                                                    double   bendStiff,
                                                    double   IPC_dt);


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
    double                  IPC_dt);

__device__ __GEIGEN__::Matrix9x12d __computePFDsPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm);

__device__ __GEIGEN__::Matrix6x12d __computePFDsPX3D_6x12_double(const __GEIGEN__::Matrix2x2d& InverseDm);

__device__ __GEIGEN__::Matrix6x9d __computePFDsPX3D_6x9_double(const __GEIGEN__::Matrix2x2d& InverseDm);

__device__ __GEIGEN__::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm);

__device__ __GEIGEN__::Matrix9x12d __computePFDmPX3D_double(
    const __GEIGEN__::Matrix12x9d& PDmPx,
    const __GEIGEN__::Matrix3x3d&  Ds,
    const __GEIGEN__::Matrix3x3d&  DmInv);

__device__ __GEIGEN__::Matrix6x12d __computePFDmPX3D_6x12_double(
    const __GEIGEN__::Matrix12x4d& PDmPx,
    const __GEIGEN__::Matrix3x2d&  Ds,
    const __GEIGEN__::Matrix2x2d&  DmInv);

__device__ __GEIGEN__::Matrix3x6d __computePFDmPX3D_3x6_double(
    const __GEIGEN__::Vector6& PDmPx, const double3& Ds, const double& DmInv);

__device__ __GEIGEN__::Matrix6x9d __computePFDmPX3D_6x9_double(
    const __GEIGEN__::Matrix9x4d& PDmPx,
    const __GEIGEN__::Matrix3x2d& Ds,
    const __GEIGEN__::Matrix2x2d& DmInv);

__device__ __GEIGEN__::Matrix9x12d __computePFPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm);

__global__ void _calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                                const double3* vertexes,
                                                const uint4*   tetrahedras,
                                                __GEIGEN__::Matrix12x12d* Hessians,
                                                uint32_t      offset,
                                                const double* volume,
                                                double3*      gradient,
                                                int           tetrahedraNum,
                                                const double* lenRate,
                                                const double* volRate,
                                                double        IPC_dt);
__global__ void _calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* trimInverses,
                                                         const double3* vertexes,
                                                         const uint3* triangles,
                                                         __GEIGEN__::Matrix9x9d* Hessians,
                                                         uint32_t      offset,
                                                         const double* area,
                                                         double3*      gradient,
                                                         int    triangleNum,
                                                         double stretchStiff,
                                                         double shearhStiff,
                                                         double IPC_dt,
                                                         double strainRate);

__global__ void _calculate_triangle_fem_deformationF(__GEIGEN__::Matrix2x2d* trimInverses,
                                                     const double3* vertexes,
                                                     const uint3*   triangles,
                                                     int            triangleNum,
                                                     Eigen::Matrix<double, 3, 2>* F3x2);


__global__ void _calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                                        const double3*          vertexes,
                                        const uint4*            tetrahedras,
                                        const double*           volume,
                                        double3*                gradient,
                                        int                     tetrahedraNum,
                                        double*                  lenRate,
                                        double*                  volRate,
                                        double                  dt);
__global__ void _calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* trimInverses,
                                                 const double3* vertexes,
                                                 const uint3*   triangles,
                                                 const double*  area,
                                                 double3*       gradient,
                                                 int            triangleNum,
                                                 double         stretchStiff,
                                                 double         shearhStiff,
                                                 double         IPC_dt,
                                                 double         strainRate);
double calculateVolum(const double3* vertexes, const uint4& index);

double calculateArea(const double3* vertexes, const uint3& index);
#endif  // ! __FEM_ENERGY_CUH__
