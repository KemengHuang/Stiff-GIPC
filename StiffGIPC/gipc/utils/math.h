#pragma once
#include <cuda_runtime_api.h>
#include <gipc/type_define.h>

namespace gipc
{
inline __device__ __host__ Matrix12x12 inverse(const Matrix12x12& input)
{
    Matrix12x12                   result;
    double                        eps = 1e-15;
    const int                     dim = 12;
    Eigen::Matrix<double, 12, 24> mat;
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < 2 * dim; j++)
        {
            if(j < dim)
            {
                mat(i, j) = input(i, j);  //(i, j);
            }
            else
            {
                mat(i, j) = j - dim == i ? 1 : 0;
            }
        }
    }

    for(int i = 0; i < dim; i++)
    {
        if(abs(mat(i, i)) < eps)
        {
            int j;
            for(j = i + 1; j < dim; j++)
            {
                if(abs(mat(j, i)) > eps)
                    break;
            }
            if(j == dim)
                return result;
            for(int r = i; r < 2 * dim; r++)
            {
                mat(i, r) += mat(j, r);
            }
        }
        double ep = mat(i, i);
        for(int r = i; r < 2 * dim; r++)
        {
            mat(i, r) /= ep;
        }

        for(int j = i + 1; j < dim; j++)
        {
            double e = -1 * (mat(j, i) / mat(i, i));
            for(int r = i; r < 2 * dim; r++)
            {
                mat(j, r) += e * mat(i, r);
            }
        }
    }

    for(int i = dim - 1; i >= 0; i--)
    {
        for(int j = i - 1; j >= 0; j--)
        {
            double e = -1 * (mat(j, i) / mat(i, i));
            for(int r = i; r < 2 * dim; r++)
            {
                mat(j, r) += e * mat(i, r);
            }
        }
    }

    for(int i = 0; i < dim; i++)
    {
        for(int r = dim; r < 2 * dim; r++)
        {
            result(i, r - dim) = mat(i, r);
        }
    }
    return result;
}
}  // namespace gipc