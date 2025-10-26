namespace gipc
{
MUDA_INLINE MUDA_GENERIC Eigen::Vector2d to_eigen(const double2& val)
{
    return Eigen::Vector2d{val.x, val.y};
}

MUDA_INLINE MUDA_GENERIC Eigen::Vector3d to_eigen(const double3& val)
{
    return Eigen::Vector3d{val.x, val.y, val.z};
}

MUDA_INLINE MUDA_GENERIC Eigen::Vector4d to_eigen(const double4& val)
{
    return Eigen::Vector4d{val.x, val.y, val.z, val.w};
}

MUDA_INLINE MUDA_GENERIC Eigen::Vector2f to_eigen(const float2& val)
{
    return Eigen::Vector2f{val.x, val.y};
}

MUDA_INLINE MUDA_GENERIC Eigen::Vector3f to_eigen(const float3& val)
{
    return Eigen::Vector3f{val.x, val.y, val.z};
}

MUDA_INLINE MUDA_GENERIC Eigen::Vector4f to_eigen(const float4& val)
{
    return Eigen::Vector4f{val.x, val.y, val.z, val.w};
}
}  // namespace gipc