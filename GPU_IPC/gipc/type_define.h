#pragma once
#include <cinttypes>
#include <eigen3/Eigen/Core>

namespace gipc
{
using I32 = int32_t;
using U32 = uint32_t;
using I64 = int64_t;
using U64 = uint64_t;
using F32 = float;
using F64 = double;

using Float  = F64;
using IndexT = U64;
using SizeT  = U64;

template <size_t M, size_t N>
using Matrix = Eigen::Matrix<Float, M, N>;
template <size_t N>
using Vector = Eigen::Matrix<Float, N, 1>;

using MatrixX = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
using VectorX = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

using Vector2  = Vector<2>;
using Vector3  = Vector<3>;
using Vector4  = Vector<4>;
using Vector6  = Vector<6>;
using Vector9  = Vector<9>;
using Vector12 = Vector<12>;

using Quaternion = Eigen::Quaternion<Float>;

using Matrix2x2   = Matrix<2, 2>;
using Matrix3x3   = Matrix<3, 3>;
using Matrix4x4   = Matrix<4, 4>;
using Matrix6x6   = Matrix<6, 6>;
using Matrix9x9   = Matrix<9, 9>;
using Matrix12x12 = Matrix<12, 12>;

using Matrix3x12 = Matrix<3, 12>;
using Matrix12x3 = Matrix<12, 3>;

using Matrix9x12 = Matrix<9, 12>;
using Matrix12x9 = Matrix<12, 9>;

using Vector2i = Eigen::Vector<int, 2>;
using Vector3i = Eigen::Vector<int, 3>;
using Vector4i = Eigen::Vector<int, 4>;
using Vector6i = Eigen::Vector<int, 6>;
}  // namespace gipc