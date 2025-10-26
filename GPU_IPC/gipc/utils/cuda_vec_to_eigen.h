#pragma once
#include <muda/muda_def.h>
#include <cuda_runtime_api.h>
#include <eigen3/Eigen/Core>

namespace gipc
{
// map-convert to eigen vector
#define CUDA_VEC_AS_EIGEN(T, N)                                                            \
    MUDA_INLINE MUDA_GENERIC Eigen::Map<Eigen::Matrix<T, N, 1>> as_eigen(T##N& val)        \
    {                                                                                      \
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(reinterpret_cast<T*>(&val));             \
    }                                                                                      \
    MUDA_INLINE MUDA_GENERIC Eigen::Map<const Eigen::Matrix<T, N, 1>> as_eigen(            \
        const T##N& val)                                                                   \
    {                                                                                      \
        return Eigen::Map<const Eigen::Matrix<T, N, 1>>(reinterpret_cast<const T*>(&val)); \
    }

CUDA_VEC_AS_EIGEN(double, 2)
CUDA_VEC_AS_EIGEN(double, 3)
CUDA_VEC_AS_EIGEN(double, 4)
CUDA_VEC_AS_EIGEN(float, 2)
CUDA_VEC_AS_EIGEN(float, 3)
CUDA_VEC_AS_EIGEN(float, 4)

#undef CUDA_VEC_AS_EIGEN

// copy-convert to eigen vector
MUDA_GENERIC Eigen::Vector2d to_eigen(const double2&);
MUDA_GENERIC Eigen::Vector3d to_eigen(const double3&);
MUDA_GENERIC Eigen::Vector4d to_eigen(const double4&);
MUDA_GENERIC Eigen::Vector2f to_eigen(const float2&);
MUDA_GENERIC Eigen::Vector3f to_eigen(const float3&);
MUDA_GENERIC Eigen::Vector4f to_eigen(const float4&);
//
template <typename T, int M, int N>
MUDA_INLINE Eigen::Matrix<T, M, N>& as_eigen(Eigen::Matrix<T, M, N>& m)
{
    return m;
}
template <typename T, int M, int N>
MUDA_INLINE const Eigen::Matrix<T, M, N>& as_eigen(const Eigen::Matrix<T, M, N>& m)
{
    return m;
}
}  // namespace gipc

#include "details/cuda_vec_to_eigen.inl"