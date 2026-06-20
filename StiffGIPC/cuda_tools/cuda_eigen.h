#pragma once

#include "cuda_def.h"
#include "cuda_atomic.h"
#include <vector_types.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace cudatool {
namespace eigen {

// as_eigen overloads for CUDA vector types
#define CT_AS_EIGEN_VEC(TYPE, N, CT)                                                   \
    CT_INLINE CT_GENERIC Eigen::Map<Eigen::Matrix<CT, N, 1>> as_eigen(TYPE& val)         \
    {                                                                                    \
        return Eigen::Map<Eigen::Matrix<CT, N, 1>>(reinterpret_cast<CT*>(&val));         \
    }                                                                                    \
    CT_INLINE CT_GENERIC Eigen::Map<const Eigen::Matrix<CT, N, 1>> as_eigen(const TYPE& val) \
    {                                                                                    \
        return Eigen::Map<const Eigen::Matrix<CT, N, 1>>(reinterpret_cast<const CT*>(&val)); \
    }

CT_AS_EIGEN_VEC(float2, 2, float)
CT_AS_EIGEN_VEC(float3, 3, float)
CT_AS_EIGEN_VEC(float4, 4, float)
CT_AS_EIGEN_VEC(double2, 2, double)
CT_AS_EIGEN_VEC(double3, 3, double)
CT_AS_EIGEN_VEC(double4, 4, double)
CT_AS_EIGEN_VEC(int2, 2, int)
CT_AS_EIGEN_VEC(int3, 3, int)
CT_AS_EIGEN_VEC(int4, 4, int)
CT_AS_EIGEN_VEC(uint2, 2, unsigned int)
CT_AS_EIGEN_VEC(uint3, 3, unsigned int)
CT_AS_EIGEN_VEC(uint4, 4, unsigned int)

#undef CT_AS_EIGEN_VEC

// atomic_add for Eigen matrices
template <typename T, int M, int N>
CT_GENERIC Eigen::Matrix<T, M, N> atomic_add(Eigen::Matrix<T, M, N>& dst,
                                             const Eigen::Matrix<T, M, N>& src)
{
    Eigen::Matrix<T, M, N> ret;
    for(int j = 0; j < N; ++j)
        for(int i = 0; i < M; ++i)
            ret(i, j) = cudatool::atomic_add(&dst(i, j), src(i, j));
    return ret;
}

template <typename T, int M, int N>
CT_GENERIC Eigen::Matrix<T, M, N> atomic_add(Eigen::Map<Eigen::Matrix<T, M, N>>& dst,
                                             const Eigen::Matrix<T, M, N>& src)
{
    Eigen::Matrix<T, M, N> ret;
    for(int j = 0; j < N; ++j)
        for(int i = 0; i < M; ++i)
            ret(i, j) = cudatool::atomic_add(&dst(i, j), src(i, j));
    return ret;
}

// Analytic inverses for small matrices
template <typename T>
CT_INLINE CT_GENERIC Eigen::Matrix<T, 2, 2> inverse(const Eigen::Matrix<T, 2, 2>& m)
{
    Eigen::Matrix<T, 2, 2> inv;
    T det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    T id  = T(1) / det;
    inv(0, 0) = m(1, 1) * id;
    inv(0, 1) = -m(0, 1) * id;
    inv(1, 0) = -m(1, 0) * id;
    inv(1, 1) = m(0, 0) * id;
    return inv;
}

template <typename T>
CT_INLINE CT_GENERIC Eigen::Matrix<T, 3, 3> inverse(const Eigen::Matrix<T, 3, 3>& m)
{
    Eigen::Matrix<T, 3, 3> inv;
    inv(0, 0) = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    inv(0, 1) = m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2);
    inv(0, 2) = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    inv(1, 0) = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    inv(1, 1) = m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    inv(1, 2) = m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2);
    inv(2, 0) = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    inv(2, 1) = m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1);
    inv(2, 2) = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    T det = m(0, 0) * inv(0, 0) + m(0, 1) * inv(1, 0) + m(0, 2) * inv(2, 0);
    return inv / det;
}

template <typename T>
CT_INLINE CT_GENERIC Eigen::Matrix<T, 4, 4> inverse(const Eigen::Matrix<T, 4, 4>& m)
{
    Eigen::Matrix<T, 4, 4> inv;
    T a = m(0, 0), b = m(0, 1), c = m(0, 2), d = m(0, 3);
    T e = m(1, 0), f = m(1, 1), g = m(1, 2), h = m(1, 3);
    T i = m(2, 0), j = m(2, 1), k = m(2, 2), l = m(2, 3);
    T x = m(3, 0), y = m(3, 1), z = m(3, 2), w = m(3, 3);

    T a0 = k * w - l * z, a1 = j * w - l * y, a2 = j * z - k * y, a3 = i * w - l * x;
    T a4 = i * z - k * x, a5 = i * y - j * x;
    T b0 = g * w - h * z, b1 = f * w - h * y, b2 = f * z - g * y, b3 = e * w - h * x;
    T b4 = e * z - g * x, b5 = e * y - f * x;
    T c0 = g * l - h * k, c1 = f * l - h * j, c2 = f * k - g * j, c3 = e * l - h * i;
    T c4 = e * k - g * i, c5 = e * j - f * i;

    inv(0, 0) = f * a0 - g * a1 + h * a2;
    inv(1, 0) = -(e * a0 - g * a3 + h * a4);
    inv(2, 0) = e * a1 - f * a3 + h * a5;
    inv(3, 0) = -(e * a2 - f * a4 + g * a5);
    inv(0, 1) = -(b * a0 - c * a1 + d * a2);
    inv(1, 1) = a * a0 - c * a3 + d * a4;
    inv(2, 1) = -(a * a1 - b * a3 + d * a5);
    inv(3, 1) = a * a2 - b * a4 + c * a5;
    inv(0, 2) = b * b0 - c * b1 + d * b2;
    inv(1, 2) = -(a * b0 - c * b3 + d * b4);
    inv(2, 2) = a * b1 - b * b3 + d * b5;
    inv(3, 2) = -(a * b2 - b * b4 + c * b5);
    inv(0, 3) = -(b * c0 - c * c1 + d * c2);
    inv(1, 3) = a * c0 - c * c3 + d * c4;
    inv(2, 3) = -(a * c1 - b * c3 + d * c5);
    inv(3, 3) = a * c2 - b * c4 + c * c5;

    T det = a * inv(0, 0) + b * inv(1, 0) + c * inv(2, 0) + d * inv(3, 0);
    return inv / det;
}

// Gauss-Jordan for general NxN
template <typename T, int N>
CT_INLINE CT_GENERIC Eigen::Matrix<T, N, N> inverse(const Eigen::Matrix<T, N, N>& m)
{
    Eigen::Matrix<T, N, N> a = m;
    Eigen::Matrix<T, N, N> inv = Eigen::Matrix<T, N, N>::Identity();
    for(int col = 0; col < N; ++col)
    {
        // partial pivot
        int pivot = col;
        T   maxv  = fabs(a(col, col));
        for(int row = col + 1; row < N; ++row)
        {
            T v = fabs(a(row, col));
            if(v > maxv)
            {
                maxv  = v;
                pivot = row;
            }
        }
        if(pivot != col)
        {
            a.row(col).swap(a.row(pivot));
            inv.row(col).swap(inv.row(pivot));
        }
        T diag = a(col, col);
        a.row(col) /= diag;
        inv.row(col) /= diag;
        for(int row = 0; row < N; ++row)
        {
            if(row != col)
            {
                T factor = a(row, col);
                a.row(row) -= factor * a.row(col);
                inv.row(row) -= factor * inv.row(col);
            }
        }
    }
    return inv;
}

// EVD for symmetric matrices
template <typename T, int N>
CT_GENERIC void evd(const Eigen::Matrix<T, N, N>& M,
                    Eigen::Vector<T, N>&          eigen_values,
                    Eigen::Matrix<T, N, N>&       eigen_vectors)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N>> eigen_solver;
    if constexpr(N <= 3)
        eigen_solver.computeDirect(M);
    else
        eigen_solver.compute(M);
    eigen_values  = eigen_solver.eigenvalues();
    eigen_vectors = eigen_solver.eigenvectors();
}

}  // namespace eigen
}  // namespace cudatool

#include "cuda_svd3x3.h"
