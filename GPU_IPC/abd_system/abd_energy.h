#pragma once
#include <muda/muda_def.h>
#include <gipc/type_define.h>

namespace gipc
{
//tex:
//$$ S = \frac{V_{\perp}}{\kappa v} $$
//we don't include the $\kappa$ and $v$ calculate it by yourself and multiply it

MUDA_GENERIC Float shape_energy(const Vector12& q);

MUDA_GENERIC Vector9 shape_energy_gradient(const Vector12& q);

MUDA_GENERIC void shape_energy_hessian(const Vector12& q,
                                       Matrix3x3&      ddVdda1,
                                       Matrix3x3&      ddVdda2,
                                       Matrix3x3&      ddVdda3,
                                       Matrix3x3&      ddVda1da2,
                                       Matrix3x3&      ddVda1da3,
                                       Matrix3x3&      ddVda2da3);

MUDA_GENERIC Matrix9x9 shape_energy_hessian(const Vector12& q);

MUDA_GENERIC Matrix3x3 q_to_A(const Vector12& q);
MUDA_GENERIC Vector9   A_to_q(const Matrix3x3& A);

MUDA_GENERIC Vector9   F_to_A(const Vector9& F);
MUDA_GENERIC Matrix9x9 HF_to_HA(const Matrix9x9& HF);
MUDA_GENERIC Float computeEnergy_CDMPM(Matrix3x3 F, double u, double r, double g);
MUDA_GENERIC void computeGradientAndHessian_CDMPM(
    Matrix3x3 F, double mu, double r, double g, Vector9& gradient, Matrix9x9& hessian);
}  // namespace gipc

#include "details/abd_energy.inl"