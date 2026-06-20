#pragma once

#include "cuda_def.h"
#include <vector_types.h>

namespace cudatool
{
CT_INLINE CT_GENERIC int2 operator+(const int2& a, const int2& b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
}  // namespace cudatool
