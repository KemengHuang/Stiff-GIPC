#include <Eigen/Dense>
namespace gipc
{
//tex:
//$$
// \frac{V_{\perp}}{\kappa v} =\sum\left(a_{i} \cdot a_{i}-1\right)^{2}
// +\sum_{i \neq j}\left(a_{i} \cdot a_{j}\right)^{2}
//$$
MUDA_INLINE MUDA_GENERIC Float shape_energy(const Vector12& q)
{
    Float       ret = 0.0;
    auto a1  = q.segment<3>(3);
    auto a2  = q.segment<3>(6);
    auto a3  = q.segment<3>(9);

    constexpr auto square = [](auto x) { return x * x; };

    ret += square(a1.squaredNorm() - 1.0);
    ret += square(a2.squaredNorm() - 1.0);
    ret += square(a3.squaredNorm() - 1.0);

    ret += square(a1.dot(a2)) * 2;
    ret += square(a2.dot(a3)) * 2;
    ret += square(a3.dot(a1)) * 2;

    return ret;
}

//tex:
// $$\frac{1}{\kappa v}\frac{\partial V_{\perp}}{\partial a_{i}}=
//2 \left(2\left(a_{i} \cdot a_{i}-1\right) a_{i}
//+ \sum a_{j}  (a_{j} \cdot a_{i})\right)$$
MUDA_INLINE MUDA_GENERIC Vector9 shape_energy_gradient(const Vector12& q)
{
    Vector9 ret;

    auto a1 = q.segment<3>(3);
    auto a2 = q.segment<3>(6);
    auto a3 = q.segment<3>(9);

    auto dEda1 = ret.segment<3>(0);
    auto dEda2 = ret.segment<3>(3);
    auto dEda3 = ret.segment<3>(6);

    dEda1 = 4.0 * (a1.squaredNorm() - 1.0) * a1 + 4.0 * a2.dot(a1) * a2
            + 4.0 * (a3.dot(a1)) * a3;

    dEda2 = 4.0 * (a2.squaredNorm() - 1.0) * a2 + 4.0 * a3.dot(a2) * a3
            + 4.0 * a1.dot(a2) * a1;

    dEda3 = 4.0 * (a3.squaredNorm() - 1.0) * a3 + 4.0 * a1.dot(a3) * a1
            + 4.0 * a2.dot(a3) * a2;

    return ret;
}


MUDA_INLINE MUDA_GENERIC void ddV_ddai(Matrix3x3& ddV_ddai,
                                       const Eigen::VectorBlock<const Vector12, 3>& ai,
                                       const Eigen::VectorBlock<const Vector12, 3>& aj,
                                       const Eigen::VectorBlock<const Vector12, 3>& ak)
{
    ddV_ddai = 8.0 * ai * ai.transpose()
               + 4.0 * (ai.squaredNorm() - 1) * Matrix3x3::Identity()
               + 4.0 * aj * aj.transpose() + 4.0 * ak * ak.transpose();
}

MUDA_INLINE MUDA_GENERIC void ddV_daidaj(Matrix3x3& ddV_daidaj,
                                         const Eigen::VectorBlock<const Vector12, 3>& ai,
                                         const Eigen::VectorBlock<const Vector12, 3>& aj,
                                         const Eigen::VectorBlock<const Vector12, 3>& ak)
{
    ddV_daidaj = 4.0 * aj * ai.transpose() + 4.0 * ai.dot(aj) * Matrix3x3::Identity();
}

MUDA_INLINE MUDA_GENERIC void shape_energy_hessian(const Vector12& q,
                                                   Matrix3x3&      ddVdda1,
                                                   Matrix3x3&      ddVdda2,
                                                   Matrix3x3&      ddVdda3,
                                                   Matrix3x3&      ddVda1da2,
                                                   Matrix3x3&      ddVda1da3,
                                                   Matrix3x3&      ddVda2da3)
{
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);

    ddV_ddai(ddVdda1, a1, a2, a3);
    ddV_ddai(ddVdda2, a2, a3, a1);
    ddV_ddai(ddVdda3, a3, a1, a2);

    ddV_daidaj(ddVda1da2, a1, a2, a3);
    ddV_daidaj(ddVda1da3, a1, a3, a2);
    ddV_daidaj(ddVda2da3, a2, a3, a1);
}

MUDA_GENERIC MUDA_INLINE gipc::Matrix9x9 shape_energy_hessian(const Vector12& q)
{
    Matrix9x9 H = Matrix9x9::Zero();

    Matrix3x3 ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3;
    shape_energy_hessian(q, ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3);


    //tex:
    //$$
    //\begin{bmatrix}
    //   \frac{\partial^2 V}{\partial a_1^2} & \frac{\partial^2 V}{\partial a_1 \partial a_2} & \frac{\partial^2 V}{\partial a_1 \partial a_3} \\
    //   \frac{\partial^2 V}{\partial a_2 \partial a_1} & \frac{\partial^2 V}{\partial a_2^2} & \frac{\partial^2 V}{\partial a_2 \partial a_3} \\
    //   \frac{\partial^2 V}{\partial a_3 \partial a_1} & \frac{\partial^2 V}{\partial a_3 \partial a_2} & \frac{\partial^2 V}{\partial a_3^2} \\
    //\end{bmatrix}
    //$$

    H.block<3, 3>(0, 0) = ddVdda1;
    H.block<3, 3>(0, 3) = ddVda1da2;
    H.block<3, 3>(0, 6) = ddVda1da3;

    H.block<3, 3>(3, 0) = ddVda1da2.transpose();
    H.block<3, 3>(3, 3) = ddVdda2;
    H.block<3, 3>(3, 6) = ddVda2da3;

    H.block<3, 3>(6, 0) = ddVda1da3.transpose();
    H.block<3, 3>(6, 3) = ddVda2da3.transpose();
    H.block<3, 3>(6, 6) = ddVdda3;

    return H;
}

MUDA_INLINE MUDA_GENERIC Matrix3x3 q_to_A(const Vector12& q)
{
    Matrix3x3 A = Matrix3x3::Zero();
    A.row(0)    = q.segment<3>(3);
    A.row(1)    = q.segment<3>(6);
    A.row(2)    = q.segment<3>(9);
    return A;
}

MUDA_INLINE MUDA_GENERIC Vector9 A_to_q(const Matrix3x3& A)
{
    Vector9 q       = Vector9::Zero();
    q.segment<3>(0) = A.row(0);
    q.segment<3>(3) = A.row(1);
    q.segment<3>(6) = A.row(2);
    return q;
}

MUDA_INLINE MUDA_GENERIC Vector9 F_to_A(const Vector9& F)
{
    Vector9 A;
    A(0) = F(0);
    A(1) = F(3);
    A(2) = F(6);
    A(3) = F(1);
    A(4) = F(4);
    A(5) = F(7);
    A(6) = F(2);
    A(7) = F(5);
    A(8) = F(8);
    return A;
}

MUDA_INLINE MUDA_GENERIC Matrix9x9 HF_to_HA(const Matrix9x9& HF)
{
    Matrix9x9 HA;
    HA(0, 0) = HF(0, 0);
    HA(0, 1) = HF(0, 3);
    HA(0, 2) = HF(0, 6);
    HA(0, 3) = HF(0, 1);
    HA(0, 4) = HF(0, 4);
    HA(0, 5) = HF(0, 7);
    HA(0, 6) = HF(0, 2);
    HA(0, 7) = HF(0, 5);
    HA(0, 8) = HF(0, 8);
    HA(1, 0) = HF(3, 0);
    HA(1, 1) = HF(3, 3);
    HA(1, 2) = HF(3, 6);
    HA(1, 3) = HF(3, 1);
    HA(1, 4) = HF(3, 4);
    HA(1, 5) = HF(3, 7);
    HA(1, 6) = HF(3, 2);
    HA(1, 7) = HF(3, 5);
    HA(1, 8) = HF(3, 8);
    HA(2, 0) = HF(6, 0);
    HA(2, 1) = HF(6, 3);
    HA(2, 2) = HF(6, 6);
    HA(2, 3) = HF(6, 1);
    HA(2, 4) = HF(6, 4);
    HA(2, 5) = HF(6, 7);
    HA(2, 6) = HF(6, 2);
    HA(2, 7) = HF(6, 5);
    HA(2, 8) = HF(6, 8);
    HA(3, 0) = HF(1, 0);
    HA(3, 1) = HF(1, 3);
    HA(3, 2) = HF(1, 6);
    HA(3, 3) = HF(1, 1);
    HA(3, 4) = HF(1, 4);
    HA(3, 5) = HF(1, 7);
    HA(3, 6) = HF(1, 2);
    HA(3, 7) = HF(1, 5);
    HA(3, 8) = HF(1, 8);
    HA(4, 0) = HF(4, 0);
    HA(4, 1) = HF(4, 3);
    HA(4, 2) = HF(4, 6);
    HA(4, 3) = HF(4, 1);
    HA(4, 4) = HF(4, 4);
    HA(4, 5) = HF(4, 7);
    HA(4, 6) = HF(4, 2);
    HA(4, 7) = HF(4, 5);
    HA(4, 8) = HF(4, 8);
    HA(5, 0) = HF(7, 0);
    HA(5, 1) = HF(7, 3);
    HA(5, 2) = HF(7, 6);
    HA(5, 3) = HF(7, 1);
    HA(5, 4) = HF(7, 4);
    HA(5, 5) = HF(7, 7);
    HA(5, 6) = HF(7, 2);
    HA(5, 7) = HF(7, 5);
    HA(5, 8) = HF(7, 8);
    HA(6, 0) = HF(2, 0);
    HA(6, 1) = HF(2, 3);
    HA(6, 2) = HF(2, 6);
    HA(6, 3) = HF(2, 1);
    HA(6, 4) = HF(2, 4);
    HA(6, 5) = HF(2, 7);
    HA(6, 6) = HF(2, 2);
    HA(6, 7) = HF(2, 5);
    HA(6, 8) = HF(2, 8);
    HA(7, 0) = HF(5, 0);
    HA(7, 1) = HF(5, 3);
    HA(7, 2) = HF(5, 6);
    HA(7, 3) = HF(5, 1);
    HA(7, 4) = HF(5, 4);
    HA(7, 5) = HF(5, 7);
    HA(7, 6) = HF(5, 2);
    HA(7, 7) = HF(5, 5);
    HA(7, 8) = HF(5, 8);
    HA(8, 0) = HF(8, 0);
    HA(8, 1) = HF(8, 3);
    HA(8, 2) = HF(8, 6);
    HA(8, 3) = HF(8, 1);
    HA(8, 4) = HF(8, 4);
    HA(8, 5) = HF(8, 7);
    HA(8, 6) = HF(8, 2);
    HA(8, 7) = HF(8, 5);
    HA(8, 8) = HF(8, 8);
    return HA;
}

MUDA_GENERIC MUDA_INLINE Float computeEnergy_CDMPM(Matrix3x3 F, double u, double r, double g)
{
    double J = F.determinant();

    Matrix3x3 FJ = cbrt(1.0 / J) * F;

    double I2 = (FJ.transpose() * FJ).trace();

    double energy_mu = 0.5 * u * (I2 - 3);
    double energy_k  = 0.5 * r * (0.5 * (J * J - 1) - log(J));

    double energy = 0;
    if(J >= 1)
    {
        energy = g * (energy_mu + energy_k);
    }
    else
    {
        energy = g * (energy_mu) + energy_k;
    }
    return energy;
}

MUDA_GENERIC void energy_grad_mu(double F1_1,
                                 double F1_2,
                                 double F1_3,
                                 double F2_1,
                                 double F2_2,
                                 double F2_3,
                                 double F3_1,
                                 double F3_2,
                                 double F3_3,
                                 double mu,
                                 double grad_mu[9]);

MUDA_GENERIC void energy_Hess_mu(double F1_1,
                                 double F1_2,
                                 double F1_3,
                                 double F2_1,
                                 double F2_2,
                                 double F2_3,
                                 double F3_1,
                                 double F3_2,
                                 double F3_3,
                                 double mu,
                                 double Hess_mu[81]);

MUDA_GENERIC void energy_grad_k(double F1_1,
                                double F1_2,
                                double F1_3,
                                double F2_1,
                                double F2_2,
                                double F2_3,
                                double F3_1,
                                double F3_2,
                                double F3_3,
                                double k,
                                double grad_k[9]);

MUDA_GENERIC void energy_Hess_k(double F1_1,
                                double F1_2,
                                double F1_3,
                                double F2_1,
                                double F2_2,
                                double F2_3,
                                double F3_1,
                                double F3_2,
                                double F3_3,
                                double k,
                                double Hess_k[81]);

MUDA_GENERIC MUDA_INLINE Vector9 to_vec(Matrix3x3 F)
{
    const int cols = F.cols();
    const int rows = F.rows();
    const int nums = cols * rows;
    Vector9   result;
    for(int i = 0; i < cols; i++)
    {
        for(int j = 0; j < rows; j++)
        {
            result(i * rows + j) = F(j, i);
        }
    }
    return result;
}

MUDA_GENERIC MUDA_INLINE void computeGradientAndHessian_CDMPM(
    Matrix3x3 F, double mu, double r, double g, Vector9& gradient, Matrix9x9& hessian)
{
    // using namespace Eigen;

    Matrix3x3 PEPF, PEPF_mu, PEPF_k;

    energy_grad_mu(F(0, 0),
                   F(0, 1),
                   F(0, 2),
                   F(1, 0),
                   F(1, 1),
                   F(1, 2),
                   F(2, 0),
                   F(2, 1),

                   F(2, 2),
                   mu,
                   PEPF_mu.data());

    energy_grad_k(F(0, 0),
                  F(0, 1),
                  F(0, 2),
                  F(1, 0),
                  F(1, 1),
                  F(1, 2),
                  F(2, 0),
                  F(2, 1),

                  F(2, 2),
                  r,
                  PEPF_k.data());

    double J = F.determinant();

    if(J >= 1)
    {
        PEPF = g * (PEPF_k + PEPF_mu);
    }
    else
    {
        PEPF = g * PEPF_mu + PEPF_k;
    }

    gradient = F_to_A(to_vec(PEPF));

    Matrix9x9 Hq, Hq_mu, Hq_k;  //

    energy_Hess_mu(F(0, 0),
                   F(0, 1),
                   F(0, 2),
                   F(1, 0),
                   F(1, 1),
                   F(1, 2),
                   F(2, 0),
                   F(2, 1),
                   F(2, 2),

                   mu,
                   Hq_mu.data());

    energy_Hess_k(F(0, 0),
                  F(0, 1),
                  F(0, 2),
                  F(1, 0),
                  F(1, 1),
                  F(1, 2),
                  F(2, 0),
                  F(2, 1),
                  F(2, 2),

                  r,
                  Hq_k.data());


    if(J >= 1)
    {
        Hq = g * (Hq_k + Hq_mu);
    }
    else
    {
        Hq = g * Hq_mu + Hq_k;
    }

    hessian = HF_to_HA(Hq);
}

}  // namespace gipc

// clang-format off
namespace gipc
{
MUDA_GENERIC MUDA_INLINE void energy_grad_mu(double F1_1,
                                 double F1_2,
                                 double F1_3,
                                 double F2_1,
                                 double F2_2,
                                 double F2_3,
                                 double F3_1,
                                 double F3_2,
                                 double F3_3,
                                 double mu,
                                 double grad_mu[9])
{
    double t10;
    double t11;
    double t12;
    double t13;
    double t14;
    double t15;
    double t16;
    double t2;
    double t3;
    double t4;
    double t47;
    double t48;
    double t49;
    double t5;
    double t50;
    double t51;
    double t52;
    double t53;
    double t54;
    double t55;
    double t6;
    double t7;
    double t8;
    double t9;
    /* ENERGY_GRAD_MU */
    /*     GRAD_MU =
   * ENERGY_GRAD_MU(F1_1,F1_2,F1_3,F2_1,F2_2,F2_3,F3_1,F3_2,F3_3,MU) */
    /*     This function was generated by the Symbolic Math Toolbox version 23.2.
   */
    /*     04-Jan-2024 14:55:07 */
    t2  = F1_1 * F1_1;
    t3  = F1_2 * F1_2;
    t4  = F1_3 * F1_3;
    t5  = F2_1 * F2_1;
    t6  = F2_2 * F2_2;
    t7  = F2_3 * F2_3;
    t8  = F3_1 * F3_1;
    t9  = F3_2 * F3_2;
    t10 = F3_3 * F3_3;
    t11 = F1_1 * F2_2;
    t12 = F1_2 * F2_1;
    t13 = F1_1 * F2_3;
    t14 = F1_3 * F2_1;
    t15 = F1_2 * F2_3;
    t16 = F1_3 * F2_2;
    t47 = t11 - t12;
    t48 = t13 - t14;
    t49 = t15 - t16;
    t50 = F1_1 * F3_2 - F1_2 * F3_1;
    t51 = F1_1 * F3_3 - F1_3 * F3_1;
    t52 = F1_2 * F3_3 - F1_3 * F3_2;
    t53 = F2_1 * F3_2 - F2_2 * F3_1;
    t54 = F2_1 * F3_3 - F2_3 * F3_1;
    t55 = F2_2 * F3_3 - F2_3 * F3_2;
    t11 = ((((F3_3 * t11 + F3_1 * t15) + F3_2 * t14) - F3_2 * t13) - F3_3 * t12) - F3_1 * t16;
    t12 = 1.0 / pow(t11, 0.66666666666666663);
    t11 = 1.0 / pow(t11, 1.6666666666666667);
    grad_mu[0] =
        mu
        * (((((((((F1_1 * t12 * -2.0 + t2 * t55 * t11 * 0.66666666666666663) + t3 * t55 * t11 * 0.66666666666666663)
                 + t4 * t55 * t11 * 0.66666666666666663)
                + t5 * t55 * t11 * 0.66666666666666663)
               + t6 * t55 * t11 * 0.66666666666666663)
              + t7 * t55 * t11 * 0.66666666666666663)
             + t8 * t55 * t11 * 0.66666666666666663)
            + t9 * t55 * t11 * 0.66666666666666663)
           + t10 * t55 * t11 * 0.66666666666666663)
        * -0.5;
    grad_mu[1] =
        mu
        * (((((((((F2_1 * t12 * 2.0 + t2 * t52 * t11 * 0.66666666666666663) + t3 * t52 * t11 * 0.66666666666666663)
                 + t4 * t52 * t11 * 0.66666666666666663)
                + t5 * t52 * t11 * 0.66666666666666663)
               + t6 * t52 * t11 * 0.66666666666666663)
              + t7 * t52 * t11 * 0.66666666666666663)
             + t8 * t52 * t11 * 0.66666666666666663)
            + t9 * t52 * t11 * 0.66666666666666663)
           + t10 * t52 * t11 * 0.66666666666666663)
        / 2.0;
    grad_mu[2] =
        mu
        * (((((((((F3_1 * t12 * -2.0 + t2 * t49 * t11 * 0.66666666666666663) + t3 * t49 * t11 * 0.66666666666666663)
                 + t4 * t49 * t11 * 0.66666666666666663)
                + t5 * t49 * t11 * 0.66666666666666663)
               + t6 * t49 * t11 * 0.66666666666666663)
              + t7 * t49 * t11 * 0.66666666666666663)
             + t8 * t49 * t11 * 0.66666666666666663)
            + t9 * t49 * t11 * 0.66666666666666663)
           + t10 * t49 * t11 * 0.66666666666666663)
        * -0.5;
    grad_mu[3] =
        mu
        * (((((((((F1_2 * t12 * 2.0 + t2 * t54 * t11 * 0.66666666666666663) + t3 * t54 * t11 * 0.66666666666666663)
                 + t4 * t54 * t11 * 0.66666666666666663)
                + t5 * t54 * t11 * 0.66666666666666663)
               + t6 * t54 * t11 * 0.66666666666666663)
              + t7 * t54 * t11 * 0.66666666666666663)
             + t8 * t54 * t11 * 0.66666666666666663)
            + t9 * t54 * t11 * 0.66666666666666663)
           + t10 * t54 * t11 * 0.66666666666666663)
        / 2.0;
    grad_mu[4] =
        mu
        * (((((((((F2_2 * t12 * -2.0 + t2 * t51 * t11 * 0.66666666666666663) + t3 * t51 * t11 * 0.66666666666666663)
                 + t4 * t51 * t11 * 0.66666666666666663)
                + t5 * t51 * t11 * 0.66666666666666663)
               + t6 * t51 * t11 * 0.66666666666666663)
              + t7 * t51 * t11 * 0.66666666666666663)
             + t8 * t51 * t11 * 0.66666666666666663)
            + t9 * t51 * t11 * 0.66666666666666663)
           + t10 * t51 * t11 * 0.66666666666666663)
        * -0.5;
    grad_mu[5] =
        mu
        * (((((((((F3_2 * t12 * 2.0 + t2 * t48 * t11 * 0.66666666666666663) + t3 * t48 * t11 * 0.66666666666666663)
                 + t4 * t48 * t11 * 0.66666666666666663)
                + t5 * t48 * t11 * 0.66666666666666663)
               + t6 * t48 * t11 * 0.66666666666666663)
              + t7 * t48 * t11 * 0.66666666666666663)
             + t8 * t48 * t11 * 0.66666666666666663)
            + t9 * t48 * t11 * 0.66666666666666663)
           + t10 * t48 * t11 * 0.66666666666666663)
        / 2.0;
    grad_mu[6] =
        mu
        * (((((((((F1_3 * t12 * -2.0 + t2 * t53 * t11 * 0.66666666666666663) + t3 * t53 * t11 * 0.66666666666666663)
                 + t4 * t53 * t11 * 0.66666666666666663)
                + t5 * t53 * t11 * 0.66666666666666663)
               + t6 * t53 * t11 * 0.66666666666666663)
              + t7 * t53 * t11 * 0.66666666666666663)
             + t8 * t53 * t11 * 0.66666666666666663)
            + t9 * t53 * t11 * 0.66666666666666663)
           + t10 * t53 * t11 * 0.66666666666666663)
        * -0.5;
    grad_mu[7] =
        mu
        * (((((((((F2_3 * t12 * 2.0 + t2 * t50 * t11 * 0.66666666666666663) + t3 * t50 * t11 * 0.66666666666666663)
                 + t4 * t50 * t11 * 0.66666666666666663)
                + t5 * t50 * t11 * 0.66666666666666663)
               + t6 * t50 * t11 * 0.66666666666666663)
              + t7 * t50 * t11 * 0.66666666666666663)
             + t8 * t50 * t11 * 0.66666666666666663)
            + t9 * t50 * t11 * 0.66666666666666663)
           + t10 * t50 * t11 * 0.66666666666666663)
        / 2.0;
    grad_mu[8] =
        mu
        * (((((((((F3_3 * t12 * -2.0 + t2 * t47 * t11 * 0.66666666666666663) + t3 * t47 * t11 * 0.66666666666666663)
                 + t4 * t47 * t11 * 0.66666666666666663)
                + t5 * t47 * t11 * 0.66666666666666663)
               + t6 * t47 * t11 * 0.66666666666666663)
              + t7 * t47 * t11 * 0.66666666666666663)
             + t8 * t47 * t11 * 0.66666666666666663)
            + t9 * t47 * t11 * 0.66666666666666663)
           + t10 * t47 * t11 * 0.66666666666666663)
        * -0.5;
}


MUDA_GENERIC MUDA_INLINE void energy_Hess_mu(double F1_1,
                                 double F1_2,
                                 double F1_3,
                                 double F2_1,
                                 double F2_2,
                                 double F2_3,
                                 double F3_1,
                                 double F3_2,
                                 double F3_3,
                                 double mu,
                                 double Hess_mu[81])
{
    double b_t689_tmp;
    double b_t690_tmp;
    double b_t691_tmp;
    double b_t695_tmp;
    double b_t697_tmp;
    double b_t700_tmp;
    double c_t689_tmp;
    double c_t690_tmp;
    double c_t691_tmp;
    double c_t695_tmp;
    double c_t697_tmp;
    double c_t700_tmp;
    double ct_idx_395;
    double ct_idx_396;
    double ct_idx_397;
    double ct_idx_398;
    double ct_idx_399;
    double ct_idx_400;
    double ct_idx_401;
    double ct_idx_402;
    double ct_idx_403;
    double ct_idx_406;
    double d_t689_tmp;
    double d_t690_tmp;
    double d_t691_tmp;
    double d_t695_tmp;
    double d_t697_tmp;
    double d_t700_tmp;
    double e_t689_tmp;
    double e_t690_tmp;
    double e_t691_tmp;
    double e_t695_tmp;
    double e_t697_tmp;
    double e_t700_tmp;
    double f_t689_tmp;
    double f_t690_tmp;
    double f_t691_tmp;
    double f_t695_tmp;
    double f_t697_tmp;
    double g_t689_tmp;
    double g_t690_tmp;
    double g_t691_tmp;
    double g_t695_tmp;
    double g_t697_tmp;
    double h_t689_tmp;
    double h_t690_tmp;
    double h_t691_tmp;
    double h_t695_tmp;
    double h_t697_tmp;
    double i_t689_tmp;
    double i_t690_tmp;
    double i_t691_tmp;
    double i_t695_tmp;
    double i_t697_tmp;
    double t10;
    double t102;
    double t107;
    double t110;
    double t113;
    double t118;
    double t12;
    double t122;
    double t126;
    double t129;
    double t134;
    double t137;
    double t14;
    double t141;
    double t147;
    double t154;
    double t156;
    double t16;
    double t18;
    double t2;
    double t20;
    double t21;
    double t22;
    double t23;
    double t24;
    double t25;
    double t4;
    double t56;
    double t57;
    double t58;
    double t59;
    double t6;
    double t60;
    double t61;
    double t62;
    double t63;
    double t64;
    double t689;
    double t689_tmp;
    double t690;
    double t690_tmp;
    double t691;
    double t691_tmp;
    double t692;
    double t693;
    double t694;
    double t695;
    double t695_tmp;
    double t696;
    double t697;
    double t697_tmp;
    double t698;
    double t699;
    double t700;
    double t700_tmp;
    double t701;
    double t702;
    double t704;
    double t709;
    double t712;
    double t717;
    double t728;
    double t729;
    double t730;
    double t731;
    double t731_tmp;
    double t732;
    double t732_tmp;
    double t733;
    double t733_tmp;
    double t735;
    double t737;
    double t739;
    double t750;
    double t751;
    double t753;
    double t76;
    double t77;
    double t8;
    double t81;
    double t85;
    double t91;
    double t93;
    /* energy_Hess_mu */
    /*     Hess_mu =
   * energy_Hess_mu(F1_1,F1_2,F1_3,F2_1,F2_2,F2_3,F3_1,F3_2,F3_3,MU) */
    /*     This function was generated by the Symbolic Math Toolbox version 23.2.
   */
    /*     04-Jan-2024 14:54:55 */
    t2  = F1_1 * F1_1;
    t4  = F1_2 * F1_2;
    t6  = F1_3 * F1_3;
    t8  = F2_1 * F2_1;
    t10 = F2_2 * F2_2;
    t12 = F2_3 * F2_3;
    t14 = F3_1 * F3_1;
    t16 = F3_2 * F3_2;
    t18 = F3_3 * F3_3;
    t20 = F1_1 * F2_2;
    t21 = F1_2 * F2_1;
    t22 = F1_1 * F2_3;
    t23 = F1_3 * F2_1;
    t24 = F1_2 * F2_3;
    t25 = F1_3 * F2_2;
    t56 = t20 - t21;
    t57 = t22 - t23;
    t58 = t24 - t25;
    t59 = F1_1 * F3_2 - F1_2 * F3_1;
    t60 = F1_1 * F3_3 - F1_3 * F3_1;
    t61 = F1_2 * F3_3 - F1_3 * F3_2;
    t62 = F2_1 * F3_2 - F2_2 * F3_1;
    t63 = F2_1 * F3_3 - F2_3 * F3_1;
    t64 = F2_2 * F3_3 - F2_3 * F3_2;
    t20 = ((((F3_3 * t20 + F3_1 * t24) + F3_2 * t23) - F3_2 * t22) - F3_3 * t21) - F3_1 * t25;
    t21        = 1.0 / pow(t20, 0.66666666666666663);
    t76        = 1.0 / pow(t20, 1.6666666666666667);
    t77        = pow(t21, 4.0);
    t81        = pow(F1_3, 3.0) * t76 * 0.66666666666666663;
    t85        = pow(F3_1, 3.0) * t76 * 0.66666666666666663;
    t91        = F1_3 * t2 * t76 * 0.66666666666666663;
    t93        = F1_3 * t4 * t76 * 0.66666666666666663;
    t102       = F1_3 * t8 * t76 * 0.66666666666666663;
    t107       = F1_3 * t10 * t76 * 0.66666666666666663;
    t110       = F1_3 * t12 * t76 * 0.66666666666666663;
    t113       = F3_1 * t2 * t76 * 0.66666666666666663;
    t118       = F3_1 * t4 * t76 * 0.66666666666666663;
    t122       = F1_3 * t14 * t76 * 0.66666666666666663;
    t126       = F3_1 * t6 * t76 * 0.66666666666666663;
    t129       = F1_3 * t16 * t76 * 0.66666666666666663;
    t134       = F1_3 * t18 * t76 * 0.66666666666666663;
    t137       = F3_1 * t8 * t76 * 0.66666666666666663;
    t141       = F3_1 * t10 * t76 * 0.66666666666666663;
    t147       = F3_1 * t12 * t76 * 0.66666666666666663;
    t154       = F3_1 * t16 * t76 * 0.66666666666666663;
    t156       = F3_1 * t18 * t76 * 0.66666666666666663;
    ct_idx_395 = t56 * t56;
    ct_idx_396 = t57 * t57;
    ct_idx_397 = t58 * t58;
    ct_idx_398 = t59 * t59;
    ct_idx_399 = t60 * t60;
    ct_idx_400 = t61 * t61;
    ct_idx_401 = t62 * t62;
    ct_idx_402 = t63 * t63;
    ct_idx_403 = t64 * t64;
    ct_idx_406 = t21 * 2.0;
    t689_tmp   = t2 * t57;
    b_t689_tmp = t4 * t57;
    c_t689_tmp = t6 * t57;
    d_t689_tmp = t8 * t57;
    e_t689_tmp = t10 * t57;
    f_t689_tmp = t12 * t57;
    g_t689_tmp = t14 * t57;
    h_t689_tmp = t16 * t57;
    i_t689_tmp = t18 * t57;
    t689       = mu
           * ((((((((((F1_2 * t57 * t76 * 1.3333333333333333 + F3_2 * t63 * t76 * 1.3333333333333333)
                      + t689_tmp * t63 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t63 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t63 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t63 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t63 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t63 * t77 * 1.1111111111111112)
                + g_t689_tmp * t63 * t77 * 1.1111111111111112)
               + h_t689_tmp * t63 * t77 * 1.1111111111111112)
              + i_t689_tmp * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t690_tmp   = t2 * t59;
    b_t690_tmp = t4 * t59;
    c_t690_tmp = t6 * t59;
    d_t690_tmp = t8 * t59;
    e_t690_tmp = t10 * t59;
    f_t690_tmp = t12 * t59;
    g_t690_tmp = t14 * t59;
    h_t690_tmp = t16 * t59;
    i_t690_tmp = t18 * t59;
    t690       = mu
           * ((((((((((F2_1 * t59 * t76 * 1.3333333333333333 + F2_3 * t61 * t76 * 1.3333333333333333)
                      + t690_tmp * t61 * t77 * 1.1111111111111112)
                     + b_t690_tmp * t61 * t77 * 1.1111111111111112)
                    + c_t690_tmp * t61 * t77 * 1.1111111111111112)
                   + d_t690_tmp * t61 * t77 * 1.1111111111111112)
                  + e_t690_tmp * t61 * t77 * 1.1111111111111112)
                 + f_t690_tmp * t61 * t77 * 1.1111111111111112)
                + g_t690_tmp * t61 * t77 * 1.1111111111111112)
               + h_t690_tmp * t61 * t77 * 1.1111111111111112)
              + i_t690_tmp * t61 * t77 * 1.1111111111111112)
           / 2.0;
    t691_tmp   = t2 * t56;
    b_t691_tmp = t4 * t56;
    c_t691_tmp = t6 * t56;
    d_t691_tmp = t8 * t56;
    e_t691_tmp = t10 * t56;
    f_t691_tmp = t12 * t56;
    g_t691_tmp = t14 * t56;
    h_t691_tmp = t16 * t56;
    i_t691_tmp = t18 * t56;
    t691       = mu
           * ((((((((((F3_2 * t56 * t76 * 1.3333333333333333 - F3_3 * t57 * t76 * 1.3333333333333333)
                      + t691_tmp * t57 * t77 * 1.1111111111111112)
                     + b_t691_tmp * t57 * t77 * 1.1111111111111112)
                    + c_t691_tmp * t57 * t77 * 1.1111111111111112)
                   + d_t691_tmp * t57 * t77 * 1.1111111111111112)
                  + e_t691_tmp * t57 * t77 * 1.1111111111111112)
                 + f_t691_tmp * t57 * t77 * 1.1111111111111112)
                + g_t691_tmp * t57 * t77 * 1.1111111111111112)
               + h_t691_tmp * t57 * t77 * 1.1111111111111112)
              + i_t691_tmp * t57 * t77 * 1.1111111111111112)
           / 2.0;
    t692 = mu
           * ((((((((((F3_2 * t58 * t76 * 1.3333333333333333 - F3_1 * t57 * t76 * 1.3333333333333333)
                      + t689_tmp * t58 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t58 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t58 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t58 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t58 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t58 * t77 * 1.1111111111111112)
                + g_t689_tmp * t58 * t77 * 1.1111111111111112)
               + h_t689_tmp * t58 * t77 * 1.1111111111111112)
              + i_t689_tmp * t58 * t77 * 1.1111111111111112)
           / 2.0;
    t693 = mu
           * ((((((((((F2_3 * t56 * t76 * 1.3333333333333333 - F3_3 * t59 * t76 * 1.3333333333333333)
                      + t691_tmp * t59 * t77 * 1.1111111111111112)
                     + b_t691_tmp * t59 * t77 * 1.1111111111111112)
                    + c_t691_tmp * t59 * t77 * 1.1111111111111112)
                   + d_t691_tmp * t59 * t77 * 1.1111111111111112)
                  + e_t691_tmp * t59 * t77 * 1.1111111111111112)
                 + f_t691_tmp * t59 * t77 * 1.1111111111111112)
                + g_t691_tmp * t59 * t77 * 1.1111111111111112)
               + h_t691_tmp * t59 * t77 * 1.1111111111111112)
              + i_t691_tmp * t59 * t77 * 1.1111111111111112)
           / 2.0;
    t694 = mu
           * ((((((((((F3_2 * t60 * t76 * 1.3333333333333333 - F2_2 * t57 * t76 * 1.3333333333333333)
                      + t689_tmp * t60 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t60 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t60 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t60 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t60 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t60 * t77 * 1.1111111111111112)
                + g_t689_tmp * t60 * t77 * 1.1111111111111112)
               + h_t689_tmp * t60 * t77 * 1.1111111111111112)
              + i_t689_tmp * t60 * t77 * 1.1111111111111112)
           / 2.0;
    t695_tmp   = t2 * t58;
    b_t695_tmp = t4 * t58;
    c_t695_tmp = t6 * t58;
    d_t695_tmp = t8 * t58;
    e_t695_tmp = t10 * t58;
    f_t695_tmp = t12 * t58;
    g_t695_tmp = t14 * t58;
    h_t695_tmp = t16 * t58;
    i_t695_tmp = t18 * t58;
    t695       = mu
           * ((((((((((F2_1 * t58 * t76 * 1.3333333333333333 - F3_1 * t61 * t76 * 1.3333333333333333)
                      + t695_tmp * t61 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t61 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t61 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t61 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t61 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t61 * t77 * 1.1111111111111112)
                + g_t695_tmp * t61 * t77 * 1.1111111111111112)
               + h_t695_tmp * t61 * t77 * 1.1111111111111112)
              + i_t695_tmp * t61 * t77 * 1.1111111111111112)
           / 2.0;
    t696 = mu
           * ((((((((((F2_3 * t60 * t76 * 1.3333333333333333 - F2_2 * t59 * t76 * 1.3333333333333333)
                      + t690_tmp * t60 * t77 * 1.1111111111111112)
                     + b_t690_tmp * t60 * t77 * 1.1111111111111112)
                    + c_t690_tmp * t60 * t77 * 1.1111111111111112)
                   + d_t690_tmp * t60 * t77 * 1.1111111111111112)
                  + e_t690_tmp * t60 * t77 * 1.1111111111111112)
                 + f_t690_tmp * t60 * t77 * 1.1111111111111112)
                + g_t690_tmp * t60 * t77 * 1.1111111111111112)
               + h_t690_tmp * t60 * t77 * 1.1111111111111112)
              + i_t690_tmp * t60 * t77 * 1.1111111111111112)
           / 2.0;
    t697_tmp   = t2 * t60;
    b_t697_tmp = t4 * t60;
    c_t697_tmp = t6 * t60;
    d_t697_tmp = t8 * t60;
    e_t697_tmp = t10 * t60;
    f_t697_tmp = t12 * t60;
    g_t697_tmp = t14 * t60;
    h_t697_tmp = t16 * t60;
    i_t697_tmp = t18 * t60;
    t697       = mu
           * ((((((((((F2_1 * t60 * t76 * 1.3333333333333333 - F2_2 * t61 * t76 * 1.3333333333333333)
                      + t697_tmp * t61 * t77 * 1.1111111111111112)
                     + b_t697_tmp * t61 * t77 * 1.1111111111111112)
                    + c_t697_tmp * t61 * t77 * 1.1111111111111112)
                   + d_t697_tmp * t61 * t77 * 1.1111111111111112)
                  + e_t697_tmp * t61 * t77 * 1.1111111111111112)
                 + f_t697_tmp * t61 * t77 * 1.1111111111111112)
                + g_t697_tmp * t61 * t77 * 1.1111111111111112)
               + h_t697_tmp * t61 * t77 * 1.1111111111111112)
              + i_t697_tmp * t61 * t77 * 1.1111111111111112)
           / 2.0;
    t698 = mu
           * ((((((((((F2_3 * t62 * t76 * 1.3333333333333333 - F1_3 * t59 * t76 * 1.3333333333333333)
                      + t690_tmp * t62 * t77 * 1.1111111111111112)
                     + b_t690_tmp * t62 * t77 * 1.1111111111111112)
                    + c_t690_tmp * t62 * t77 * 1.1111111111111112)
                   + d_t690_tmp * t62 * t77 * 1.1111111111111112)
                  + e_t690_tmp * t62 * t77 * 1.1111111111111112)
                 + f_t690_tmp * t62 * t77 * 1.1111111111111112)
                + g_t690_tmp * t62 * t77 * 1.1111111111111112)
               + h_t690_tmp * t62 * t77 * 1.1111111111111112)
              + i_t690_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t699 = mu
           * ((((((((((F1_2 * t60 * t76 * 1.3333333333333333 - F2_2 * t63 * t76 * 1.3333333333333333)
                      + t697_tmp * t63 * t77 * 1.1111111111111112)
                     + b_t697_tmp * t63 * t77 * 1.1111111111111112)
                    + c_t697_tmp * t63 * t77 * 1.1111111111111112)
                   + d_t697_tmp * t63 * t77 * 1.1111111111111112)
                  + e_t697_tmp * t63 * t77 * 1.1111111111111112)
                 + f_t697_tmp * t63 * t77 * 1.1111111111111112)
                + g_t697_tmp * t63 * t77 * 1.1111111111111112)
               + h_t697_tmp * t63 * t77 * 1.1111111111111112)
              + i_t697_tmp * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t737       = t2 * t61;
    t739       = t4 * t61;
    t733       = t6 * t61;
    t735       = t61 * t8;
    t700_tmp   = t10 * t61;
    b_t700_tmp = t12 * t61;
    c_t700_tmp = t14 * t61;
    d_t700_tmp = t16 * t61;
    e_t700_tmp = t18 * t61;
    t700       = mu
           * ((((((((((F2_1 * t64 * t76 * 1.3333333333333333 - F1_1 * t61 * t76 * 1.3333333333333333)
                      + t737 * t64 * t77 * 1.1111111111111112)
                     + t739 * t64 * t77 * 1.1111111111111112)
                    + t733 * t64 * t77 * 1.1111111111111112)
                   + t735 * t64 * t77 * 1.1111111111111112)
                  + t700_tmp * t64 * t77 * 1.1111111111111112)
                 + b_t700_tmp * t64 * t77 * 1.1111111111111112)
                + c_t700_tmp * t64 * t77 * 1.1111111111111112)
               + d_t700_tmp * t64 * t77 * 1.1111111111111112)
              + e_t700_tmp * t64 * t77 * 1.1111111111111112)
           / 2.0;
    t20  = t2 * t62;
    t21  = t4 * t62;
    t22  = t6 * t62;
    t23  = t62 * t8;
    t24  = t10 * t62;
    t25  = t12 * t62;
    t750 = t14 * t62;
    t753 = t16 * t62;
    t751 = t18 * t62;
    t701 = mu
           * ((((((((((F1_2 * t62 * t76 * 1.3333333333333333 - F1_3 * t63 * t76 * 1.3333333333333333)
                      + t20 * t63 * t77 * 1.1111111111111112)
                     + t21 * t63 * t77 * 1.1111111111111112)
                    + t22 * t63 * t77 * 1.1111111111111112)
                   + t23 * t63 * t77 * 1.1111111111111112)
                  + t24 * t63 * t77 * 1.1111111111111112)
                 + t25 * t63 * t77 * 1.1111111111111112)
                + t750 * t63 * t77 * 1.1111111111111112)
               + t753 * t63 * t77 * 1.1111111111111112)
              + t751 * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t702 = mu
           * ((((((((((F1_2 * t64 * t76 * 1.3333333333333333 - F1_1 * t63 * t76 * 1.3333333333333333)
                      + t2 * t63 * t64 * t77 * 1.1111111111111112)
                     + t4 * t63 * t64 * t77 * 1.1111111111111112)
                    + t6 * t63 * t64 * t77 * 1.1111111111111112)
                   + t63 * t8 * t64 * t77 * 1.1111111111111112)
                  + t10 * t63 * t64 * t77 * 1.1111111111111112)
                 + t12 * t63 * t64 * t77 * 1.1111111111111112)
                + t14 * t63 * t64 * t77 * 1.1111111111111112)
               + t16 * t63 * t64 * t77 * 1.1111111111111112)
              + t18 * t63 * t64 * t77 * 1.1111111111111112)
           / 2.0;
    t704 = mu
           * ((((((((((-(F3_1 * t56 * t76 * 1.3333333333333333) - F3_3 * t58 * t76 * 1.3333333333333333)
                      + t691_tmp * t58 * t77 * 1.1111111111111112)
                     + b_t691_tmp * t58 * t77 * 1.1111111111111112)
                    + c_t691_tmp * t58 * t77 * 1.1111111111111112)
                   + d_t691_tmp * t58 * t77 * 1.1111111111111112)
                  + e_t691_tmp * t58 * t77 * 1.1111111111111112)
                 + f_t691_tmp * t58 * t77 * 1.1111111111111112)
                + g_t691_tmp * t58 * t77 * 1.1111111111111112)
               + h_t691_tmp * t58 * t77 * 1.1111111111111112)
              + i_t691_tmp * t58 * t77 * 1.1111111111111112)
           / 2.0;
    t709 = mu
           * ((((((((((-(F1_3 * t56 * t76 * 1.3333333333333333) - F3_3 * t62 * t76 * 1.3333333333333333)
                      + t691_tmp * t62 * t77 * 1.1111111111111112)
                     + b_t691_tmp * t62 * t77 * 1.1111111111111112)
                    + c_t691_tmp * t62 * t77 * 1.1111111111111112)
                   + d_t691_tmp * t62 * t77 * 1.1111111111111112)
                  + e_t691_tmp * t62 * t77 * 1.1111111111111112)
                 + f_t691_tmp * t62 * t77 * 1.1111111111111112)
                + g_t691_tmp * t62 * t77 * 1.1111111111111112)
               + h_t691_tmp * t62 * t77 * 1.1111111111111112)
              + i_t691_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t712 = mu
           * ((((((((((-(F1_1 * t58 * t76 * 1.3333333333333333) - F3_1 * t64 * t76 * 1.3333333333333333)
                      + t695_tmp * t64 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t64 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t64 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t64 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t64 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t64 * t77 * 1.1111111111111112)
                + g_t695_tmp * t64 * t77 * 1.1111111111111112)
               + h_t695_tmp * t64 * t77 * 1.1111111111111112)
              + i_t695_tmp * t64 * t77 * 1.1111111111111112)
           / 2.0;
    t717 = mu
           * ((((((((((-(F1_1 * t62 * t76 * 1.3333333333333333) - F1_3 * t64 * t76 * 1.3333333333333333)
                      + t20 * t64 * t77 * 1.1111111111111112)
                     + t21 * t64 * t77 * 1.1111111111111112)
                    + t22 * t64 * t77 * 1.1111111111111112)
                   + t23 * t64 * t77 * 1.1111111111111112)
                  + t24 * t64 * t77 * 1.1111111111111112)
                 + t25 * t64 * t77 * 1.1111111111111112)
                + t750 * t64 * t77 * 1.1111111111111112)
               + t753 * t64 * t77 * 1.1111111111111112)
              + t751 * t64 * t77 * 1.1111111111111112)
           / 2.0;
    t21 = (((((((pow(F1_1, 3.0) * t76 * 0.66666666666666663 + F1_1 * t4 * t76 * 0.66666666666666663)
                + F1_1 * t6 * t76 * 0.66666666666666663)
               + F1_1 * t8 * t76 * 0.66666666666666663)
              + F1_1 * t10 * t76 * 0.66666666666666663)
             + F1_1 * t12 * t76 * 0.66666666666666663)
            + F1_1 * t14 * t76 * 0.66666666666666663)
           + F1_1 * t16 * t76 * 0.66666666666666663)
          + F1_1 * t18 * t76 * 0.66666666666666663;
    t728 = mu
           * (((((((((((t21 + F2_3 * t57 * t76 * 1.3333333333333333) + F3_2 * t59 * t76 * 1.3333333333333333)
                      + t689_tmp * t59 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t59 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t59 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t59 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t59 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t59 * t77 * 1.1111111111111112)
                + g_t689_tmp * t59 * t77 * 1.1111111111111112)
               + h_t689_tmp * t59 * t77 * 1.1111111111111112)
              + i_t689_tmp * t59 * t77 * 1.1111111111111112)
           / 2.0;
    t22 = (((((((F3_3 * t2 * t76 * 0.66666666666666663 + pow(F3_3, 3.0) * t76 * 0.66666666666666663)
                + F3_3 * t4 * t76 * 0.66666666666666663)
               + F3_3 * t6 * t76 * 0.66666666666666663)
              + F3_3 * t8 * t76 * 0.66666666666666663)
             + F3_3 * t10 * t76 * 0.66666666666666663)
            + F3_3 * t12 * t76 * 0.66666666666666663)
           + F3_3 * t14 * t76 * 0.66666666666666663)
          + F3_3 * t16 * t76 * 0.66666666666666663;
    t729 = mu
           * (((((((((((t22 + F1_2 * t61 * t76 * 1.3333333333333333) + F2_1 * t63 * t76 * 1.3333333333333333)
                      + t737 * t63 * t77 * 1.1111111111111112)
                     + t739 * t63 * t77 * 1.1111111111111112)
                    + t733 * t63 * t77 * 1.1111111111111112)
                   + t735 * t63 * t77 * 1.1111111111111112)
                  + t700_tmp * t63 * t77 * 1.1111111111111112)
                 + b_t700_tmp * t63 * t77 * 1.1111111111111112)
                + c_t700_tmp * t63 * t77 * 1.1111111111111112)
               + d_t700_tmp * t63 * t77 * 1.1111111111111112)
              + e_t700_tmp * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t23 = (((((((pow(F1_2, 3.0) * t76 * 0.66666666666666663 + F1_2 * t2 * t76 * 0.66666666666666663)
                + F1_2 * t6 * t76 * 0.66666666666666663)
               + F1_2 * t8 * t76 * 0.66666666666666663)
              + F1_2 * t10 * t76 * 0.66666666666666663)
             + F1_2 * t12 * t76 * 0.66666666666666663)
            + F1_2 * t14 * t76 * 0.66666666666666663)
           + F1_2 * t16 * t76 * 0.66666666666666663)
          + F1_2 * t18 * t76 * 0.66666666666666663;
    t730 = mu
           * (((((((((((t23 + F2_3 * t58 * t76 * 1.3333333333333333) - F3_1 * t59 * t76 * 1.3333333333333333)
                      + t695_tmp * t59 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t59 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t59 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t59 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t59 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t59 * t77 * 1.1111111111111112)
                + g_t695_tmp * t59 * t77 * 1.1111111111111112)
               + h_t695_tmp * t59 * t77 * 1.1111111111111112)
              + i_t695_tmp * t59 * t77 * 1.1111111111111112)
           / 2.0;
    t731_tmp = (((((((pow(F2_1, 3.0) * t76 * 0.66666666666666663 + F2_1 * t2 * t76 * 0.66666666666666663)
                     + F2_1 * t4 * t76 * 0.66666666666666663)
                    + F2_1 * t6 * t76 * 0.66666666666666663)
                   + F2_1 * t10 * t76 * 0.66666666666666663)
                  + F2_1 * t12 * t76 * 0.66666666666666663)
                 + F2_1 * t14 * t76 * 0.66666666666666663)
                + F2_1 * t16 * t76 * 0.66666666666666663)
               + F2_1 * t18 * t76 * 0.66666666666666663;
    t731 = mu
           * (((((((((((t731_tmp + F3_2 * t62 * t76 * 1.3333333333333333) - F1_3 * t57 * t76 * 1.3333333333333333)
                      + t689_tmp * t62 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t62 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t62 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t62 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t62 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t62 * t77 * 1.1111111111111112)
                + g_t689_tmp * t62 * t77 * 1.1111111111111112)
               + h_t689_tmp * t62 * t77 * 1.1111111111111112)
              + i_t689_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t732_tmp = (((((((F2_3 * t2 * t76 * 0.66666666666666663 + pow(F2_3, 3.0) * t76 * 0.66666666666666663)
                     + F2_3 * t4 * t76 * 0.66666666666666663)
                    + F2_3 * t6 * t76 * 0.66666666666666663)
                   + F2_3 * t8 * t76 * 0.66666666666666663)
                  + F2_3 * t10 * t76 * 0.66666666666666663)
                 + F2_3 * t14 * t76 * 0.66666666666666663)
                + F2_3 * t16 * t76 * 0.66666666666666663)
               + F2_3 * t18 * t76 * 0.66666666666666663;
    t732 = mu
           * (((((((((((t732_tmp + F1_2 * t58 * t76 * 1.3333333333333333) - F3_1 * t63 * t76 * 1.3333333333333333)
                      + t695_tmp * t63 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t63 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t63 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t63 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t63 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t63 * t77 * 1.1111111111111112)
                + g_t695_tmp * t63 * t77 * 1.1111111111111112)
               + h_t695_tmp * t63 * t77 * 1.1111111111111112)
              + i_t695_tmp * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t733_tmp = (((((((F3_2 * t2 * t76 * 0.66666666666666663 + pow(F3_2, 3.0) * t76 * 0.66666666666666663)
                     + F3_2 * t4 * t76 * 0.66666666666666663)
                    + F3_2 * t6 * t76 * 0.66666666666666663)
                   + F3_2 * t8 * t76 * 0.66666666666666663)
                  + F3_2 * t10 * t76 * 0.66666666666666663)
                 + F3_2 * t12 * t76 * 0.66666666666666663)
                + F3_2 * t14 * t76 * 0.66666666666666663)
               + F3_2 * t18 * t76 * 0.66666666666666663;
    t733 = mu
           * (((((((((((t733_tmp + F2_1 * t62 * t76 * 1.3333333333333333) - F1_3 * t61 * t76 * 1.3333333333333333)
                      + t737 * t62 * t77 * 1.1111111111111112)
                     + t739 * t62 * t77 * 1.1111111111111112)
                    + t733 * t62 * t77 * 1.1111111111111112)
                   + t735 * t62 * t77 * 1.1111111111111112)
                  + t700_tmp * t62 * t77 * 1.1111111111111112)
                 + b_t700_tmp * t62 * t77 * 1.1111111111111112)
                + c_t700_tmp * t62 * t77 * 1.1111111111111112)
               + d_t700_tmp * t62 * t77 * 1.1111111111111112)
              + e_t700_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t735 = mu
           * (((((((((((((((((((t81 + t91) + t93) + t102) + t107) + t110) + t122) + t129) + t134)
                        - F2_2 * t58 * t76 * 1.3333333333333333)
                       - F3_1 * t60 * t76 * 1.3333333333333333)
                      + t695_tmp * t60 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t60 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t60 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t60 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t60 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t60 * t77 * 1.1111111111111112)
                + g_t695_tmp * t60 * t77 * 1.1111111111111112)
               + h_t695_tmp * t60 * t77 * 1.1111111111111112)
              + i_t695_tmp * t60 * t77 * 1.1111111111111112)
           / 2.0;
    t20 = (((((((pow(F2_2, 3.0) * t76 * 0.66666666666666663 + F2_2 * t2 * t76 * 0.66666666666666663)
                + F2_2 * t4 * t76 * 0.66666666666666663)
               + F2_2 * t6 * t76 * 0.66666666666666663)
              + F2_2 * t8 * t76 * 0.66666666666666663)
             + F2_2 * t12 * t76 * 0.66666666666666663)
            + F2_2 * t14 * t76 * 0.66666666666666663)
           + F2_2 * t16 * t76 * 0.66666666666666663)
          + F2_2 * t18 * t76 * 0.66666666666666663;
    t737 = mu
           * (((((((((((t20 - F1_3 * t58 * t76 * 1.3333333333333333) - F3_1 * t62 * t76 * 1.3333333333333333)
                      + t695_tmp * t62 * t77 * 1.1111111111111112)
                     + b_t695_tmp * t62 * t77 * 1.1111111111111112)
                    + c_t695_tmp * t62 * t77 * 1.1111111111111112)
                   + d_t695_tmp * t62 * t77 * 1.1111111111111112)
                  + e_t695_tmp * t62 * t77 * 1.1111111111111112)
                 + f_t695_tmp * t62 * t77 * 1.1111111111111112)
                + g_t695_tmp * t62 * t77 * 1.1111111111111112)
               + h_t695_tmp * t62 * t77 * 1.1111111111111112)
              + i_t695_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t739 = mu
           * (((((((((((((((((((t113 + t85) + t118) + t126) + t137) + t141) + t147) + t154) + t156)
                        - F1_3 * t60 * t76 * 1.3333333333333333)
                       - F2_2 * t62 * t76 * 1.3333333333333333)
                      + t697_tmp * t62 * t77 * 1.1111111111111112)
                     + b_t697_tmp * t62 * t77 * 1.1111111111111112)
                    + c_t697_tmp * t62 * t77 * 1.1111111111111112)
                   + d_t697_tmp * t62 * t77 * 1.1111111111111112)
                  + e_t697_tmp * t62 * t77 * 1.1111111111111112)
                 + f_t697_tmp * t62 * t77 * 1.1111111111111112)
                + g_t697_tmp * t62 * t77 * 1.1111111111111112)
               + h_t697_tmp * t62 * t77 * 1.1111111111111112)
              + i_t697_tmp * t62 * t77 * 1.1111111111111112)
           / 2.0;
    t751 = mu
           * (((((((((((((((((((-t81 - t91) - t93) - t102) - t107) - t110) - t122) - t129) - t134)
                        + F2_1 * t57 * t76 * 1.3333333333333333)
                       + F3_2 * t61 * t76 * 1.3333333333333333)
                      + t689_tmp * t61 * t77 * 1.1111111111111112)
                     + b_t689_tmp * t61 * t77 * 1.1111111111111112)
                    + c_t689_tmp * t61 * t77 * 1.1111111111111112)
                   + d_t689_tmp * t61 * t77 * 1.1111111111111112)
                  + e_t689_tmp * t61 * t77 * 1.1111111111111112)
                 + f_t689_tmp * t61 * t77 * 1.1111111111111112)
                + g_t689_tmp * t61 * t77 * 1.1111111111111112)
               + h_t689_tmp * t61 * t77 * 1.1111111111111112)
              + i_t689_tmp * t61 * t77 * 1.1111111111111112)
           / 2.0;
    t753 = mu
           * (((((((((((((((((((-t85 - t113) - t118) - t126) - t137) - t141) - t147) - t154) - t156)
                        + F1_2 * t59 * t76 * 1.3333333333333333)
                       + F2_3 * t63 * t76 * 1.3333333333333333)
                      + t690_tmp * t63 * t77 * 1.1111111111111112)
                     + b_t690_tmp * t63 * t77 * 1.1111111111111112)
                    + c_t690_tmp * t63 * t77 * 1.1111111111111112)
                   + d_t690_tmp * t63 * t77 * 1.1111111111111112)
                  + e_t690_tmp * t63 * t77 * 1.1111111111111112)
                 + f_t690_tmp * t63 * t77 * 1.1111111111111112)
                + g_t690_tmp * t63 * t77 * 1.1111111111111112)
               + h_t690_tmp * t63 * t77 * 1.1111111111111112)
              + i_t690_tmp * t63 * t77 * 1.1111111111111112)
           / 2.0;
    t750 = mu
           * (((((((((((t21 + F2_2 * t56 * t76 * 1.3333333333333333) + F3_3 * t60 * t76 * 1.3333333333333333)
                      - t691_tmp * t60 * t77 * 1.1111111111111112)
                     - b_t691_tmp * t60 * t77 * 1.1111111111111112)
                    - c_t691_tmp * t60 * t77 * 1.1111111111111112)
                   - d_t691_tmp * t60 * t77 * 1.1111111111111112)
                  - e_t691_tmp * t60 * t77 * 1.1111111111111112)
                 - f_t691_tmp * t60 * t77 * 1.1111111111111112)
                - g_t691_tmp * t60 * t77 * 1.1111111111111112)
               - h_t691_tmp * t60 * t77 * 1.1111111111111112)
              - i_t691_tmp * t60 * t77 * 1.1111111111111112)
           / 2.0;
    t25 = mu
          * (((((((((((t20 + F1_1 * t56 * t76 * 1.3333333333333333) + F3_3 * t64 * t76 * 1.3333333333333333)
                     - t691_tmp * t64 * t77 * 1.1111111111111112)
                    - b_t691_tmp * t64 * t77 * 1.1111111111111112)
                   - c_t691_tmp * t64 * t77 * 1.1111111111111112)
                  - d_t691_tmp * t64 * t77 * 1.1111111111111112)
                 - e_t691_tmp * t64 * t77 * 1.1111111111111112)
                - f_t691_tmp * t64 * t77 * 1.1111111111111112)
               - g_t691_tmp * t64 * t77 * 1.1111111111111112)
              - h_t691_tmp * t64 * t77 * 1.1111111111111112)
             - i_t691_tmp * t64 * t77 * 1.1111111111111112)
          / 2.0;
    t24 = mu
          * (((((((((((t22 + F1_1 * t60 * t76 * 1.3333333333333333) + F2_2 * t64 * t76 * 1.3333333333333333)
                     - t697_tmp * t64 * t77 * 1.1111111111111112)
                    - b_t697_tmp * t64 * t77 * 1.1111111111111112)
                   - c_t697_tmp * t64 * t77 * 1.1111111111111112)
                  - d_t697_tmp * t64 * t77 * 1.1111111111111112)
                 - e_t697_tmp * t64 * t77 * 1.1111111111111112)
                - f_t697_tmp * t64 * t77 * 1.1111111111111112)
               - g_t697_tmp * t64 * t77 * 1.1111111111111112)
              - h_t697_tmp * t64 * t77 * 1.1111111111111112)
             - i_t697_tmp * t64 * t77 * 1.1111111111111112)
          / 2.0;
    t23 = mu
          * (((((((((((t23 + F3_3 * t61 * t76 * 1.3333333333333333) - F2_1 * t56 * t76 * 1.3333333333333333)
                     - t691_tmp * t61 * t77 * 1.1111111111111112)
                    - b_t691_tmp * t61 * t77 * 1.1111111111111112)
                   - c_t691_tmp * t61 * t77 * 1.1111111111111112)
                  - d_t691_tmp * t61 * t77 * 1.1111111111111112)
                 - e_t691_tmp * t61 * t77 * 1.1111111111111112)
                - f_t691_tmp * t61 * t77 * 1.1111111111111112)
               - g_t691_tmp * t61 * t77 * 1.1111111111111112)
              - h_t691_tmp * t61 * t77 * 1.1111111111111112)
             - i_t691_tmp * t61 * t77 * 1.1111111111111112)
          / 2.0;
    t22 = mu
          * (((((((((((t731_tmp + F3_3 * t63 * t76 * 1.3333333333333333) - F1_2 * t56 * t76 * 1.3333333333333333)
                     - t691_tmp * t63 * t77 * 1.1111111111111112)
                    - b_t691_tmp * t63 * t77 * 1.1111111111111112)
                   - c_t691_tmp * t63 * t77 * 1.1111111111111112)
                  - d_t691_tmp * t63 * t77 * 1.1111111111111112)
                 - e_t691_tmp * t63 * t77 * 1.1111111111111112)
                - f_t691_tmp * t63 * t77 * 1.1111111111111112)
               - g_t691_tmp * t63 * t77 * 1.1111111111111112)
              - h_t691_tmp * t63 * t77 * 1.1111111111111112)
             - i_t691_tmp * t63 * t77 * 1.1111111111111112)
          / 2.0;
    t21 = mu
          * (((((((((((t732_tmp + F1_1 * t57 * t76 * 1.3333333333333333) - F3_2 * t64 * t76 * 1.3333333333333333)
                     - t689_tmp * t64 * t77 * 1.1111111111111112)
                    - b_t689_tmp * t64 * t77 * 1.1111111111111112)
                   - c_t689_tmp * t64 * t77 * 1.1111111111111112)
                  - d_t689_tmp * t64 * t77 * 1.1111111111111112)
                 - e_t689_tmp * t64 * t77 * 1.1111111111111112)
                - f_t689_tmp * t64 * t77 * 1.1111111111111112)
               - g_t689_tmp * t64 * t77 * 1.1111111111111112)
              - h_t689_tmp * t64 * t77 * 1.1111111111111112)
             - i_t689_tmp * t64 * t77 * 1.1111111111111112)
          / 2.0;
    t20 = mu
          * (((((((((((t733_tmp + F1_1 * t59 * t76 * 1.3333333333333333) - F2_3 * t64 * t76 * 1.3333333333333333)
                     - t690_tmp * t64 * t77 * 1.1111111111111112)
                    - b_t690_tmp * t64 * t77 * 1.1111111111111112)
                   - c_t690_tmp * t64 * t77 * 1.1111111111111112)
                  - d_t690_tmp * t64 * t77 * 1.1111111111111112)
                 - e_t690_tmp * t64 * t77 * 1.1111111111111112)
                - f_t690_tmp * t64 * t77 * 1.1111111111111112)
               - g_t690_tmp * t64 * t77 * 1.1111111111111112)
              - h_t690_tmp * t64 * t77 * 1.1111111111111112)
             - i_t690_tmp * t64 * t77 * 1.1111111111111112)
          / 2.0;
    Hess_mu[0] =
        mu
        * ((((((((((ct_idx_406 - F1_1 * t64 * t76 * 2.6666666666666665) + t2 * ct_idx_403 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_403 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_403 * t77 * 1.1111111111111112)
                + ct_idx_403 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_403 * t77 * 1.1111111111111112)
              + t12 * ct_idx_403 * t77 * 1.1111111111111112)
             + t14 * ct_idx_403 * t77 * 1.1111111111111112)
            + t16 * ct_idx_403 * t77 * 1.1111111111111112)
           + t18 * ct_idx_403 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[1] = -t700;
    Hess_mu[2] = t712;
    Hess_mu[3] = -t702;
    Hess_mu[4] = -t24;
    Hess_mu[5] = t21;
    Hess_mu[6] = t717;
    Hess_mu[7] = t20;
    Hess_mu[8] = -t25;
    Hess_mu[9] = -t700;
    Hess_mu[10] =
        mu
        * ((((((((((ct_idx_406 + F2_1 * t61 * t76 * 2.6666666666666665) + t2 * ct_idx_400 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_400 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_400 * t77 * 1.1111111111111112)
                + ct_idx_400 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_400 * t77 * 1.1111111111111112)
              + t12 * ct_idx_400 * t77 * 1.1111111111111112)
             + t14 * ct_idx_400 * t77 * 1.1111111111111112)
            + t16 * ct_idx_400 * t77 * 1.1111111111111112)
           + t18 * ct_idx_400 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[11] = -t695;
    Hess_mu[12] = t729;
    Hess_mu[13] = -t697;
    Hess_mu[14] = t751;
    Hess_mu[15] = -t733;
    Hess_mu[16] = t690;
    Hess_mu[17] = t23;
    Hess_mu[18] = t712;
    Hess_mu[19] = -t695;
    Hess_mu[20] =
        mu
        * ((((((((((ct_idx_406 - F3_1 * t58 * t76 * 2.6666666666666665) + t2 * ct_idx_397 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_397 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_397 * t77 * 1.1111111111111112)
                + ct_idx_397 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_397 * t77 * 1.1111111111111112)
              + t12 * ct_idx_397 * t77 * 1.1111111111111112)
             + t14 * ct_idx_397 * t77 * 1.1111111111111112)
            + t16 * ct_idx_397 * t77 * 1.1111111111111112)
           + t18 * ct_idx_397 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[21] = -t732;
    Hess_mu[22] = t735;
    Hess_mu[23] = -t692;
    Hess_mu[24] = t737;
    Hess_mu[25] = -t730;
    Hess_mu[26] = t704;
    Hess_mu[27] = -t702;
    Hess_mu[28] = t729;
    Hess_mu[29] = -t732;
    Hess_mu[30] =
        mu
        * ((((((((((ct_idx_406 + F1_2 * t63 * t76 * 2.6666666666666665) + t2 * ct_idx_402 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_402 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_402 * t77 * 1.1111111111111112)
                + ct_idx_402 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_402 * t77 * 1.1111111111111112)
              + t12 * ct_idx_402 * t77 * 1.1111111111111112)
             + t14 * ct_idx_402 * t77 * 1.1111111111111112)
            + t16 * ct_idx_402 * t77 * 1.1111111111111112)
           + t18 * ct_idx_402 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[31] = -t699;
    Hess_mu[32] = t689;
    Hess_mu[33] = -t701;
    Hess_mu[34] = t753;
    Hess_mu[35] = t22;
    Hess_mu[36] = -t24;
    Hess_mu[37] = -t697;
    Hess_mu[38] = t735;
    Hess_mu[39] = -t699;
    Hess_mu[40] =
        mu
        * ((((((((((ct_idx_406 - F2_2 * t60 * t76 * 2.6666666666666665) + t2 * ct_idx_399 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_399 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_399 * t77 * 1.1111111111111112)
                + ct_idx_399 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_399 * t77 * 1.1111111111111112)
              + t12 * ct_idx_399 * t77 * 1.1111111111111112)
             + t14 * ct_idx_399 * t77 * 1.1111111111111112)
            + t16 * ct_idx_399 * t77 * 1.1111111111111112)
           + t18 * ct_idx_399 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[41] = -t694;
    Hess_mu[42] = t739;
    Hess_mu[43] = -t696;
    Hess_mu[44] = -t750;
    Hess_mu[45] = t21;
    Hess_mu[46] = t751;
    Hess_mu[47] = -t692;
    Hess_mu[48] = t689;
    Hess_mu[49] = -t694;
    Hess_mu[50] =
        mu
        * ((((((((((ct_idx_406 + F3_2 * t57 * t76 * 2.6666666666666665) + t2 * ct_idx_396 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_396 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_396 * t77 * 1.1111111111111112)
                + ct_idx_396 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_396 * t77 * 1.1111111111111112)
              + t12 * ct_idx_396 * t77 * 1.1111111111111112)
             + t14 * ct_idx_396 * t77 * 1.1111111111111112)
            + t16 * ct_idx_396 * t77 * 1.1111111111111112)
           + t18 * ct_idx_396 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[51] = -t731;
    Hess_mu[52] = t728;
    Hess_mu[53] = -t691;
    Hess_mu[54] = t717;
    Hess_mu[55] = -t733;
    Hess_mu[56] = t737;
    Hess_mu[57] = -t701;
    Hess_mu[58] = t739;
    Hess_mu[59] = -t731;
    Hess_mu[60] =
        mu
        * ((((((((((ct_idx_406 - F1_3 * t62 * t76 * 2.6666666666666665) + t2 * ct_idx_401 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_401 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_401 * t77 * 1.1111111111111112)
                + ct_idx_401 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_401 * t77 * 1.1111111111111112)
              + t12 * ct_idx_401 * t77 * 1.1111111111111112)
             + t14 * ct_idx_401 * t77 * 1.1111111111111112)
            + t16 * ct_idx_401 * t77 * 1.1111111111111112)
           + t18 * ct_idx_401 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[61] = -t698;
    Hess_mu[62] = t709;
    Hess_mu[63] = t20;
    Hess_mu[64] = t690;
    Hess_mu[65] = -t730;
    Hess_mu[66] = t753;
    Hess_mu[67] = -t696;
    Hess_mu[68] = t728;
    Hess_mu[69] = -t698;
    Hess_mu[70] =
        mu
        * ((((((((((ct_idx_406 + F2_3 * t59 * t76 * 2.6666666666666665) + t2 * ct_idx_398 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_398 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_398 * t77 * 1.1111111111111112)
                + ct_idx_398 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_398 * t77 * 1.1111111111111112)
              + t12 * ct_idx_398 * t77 * 1.1111111111111112)
             + t14 * ct_idx_398 * t77 * 1.1111111111111112)
            + t16 * ct_idx_398 * t77 * 1.1111111111111112)
           + t18 * ct_idx_398 * t77 * 1.1111111111111112)
        / 2.0;
    Hess_mu[71] = -t693;
    Hess_mu[72] = -t25;
    Hess_mu[73] = t23;
    Hess_mu[74] = t704;
    Hess_mu[75] = t22;
    Hess_mu[76] = -t750;
    Hess_mu[77] = -t691;
    Hess_mu[78] = t709;
    Hess_mu[79] = -t693;
    Hess_mu[80] =
        mu
        * ((((((((((ct_idx_406 - F3_3 * t56 * t76 * 2.6666666666666665) + t2 * ct_idx_395 * t77 * 1.1111111111111112)
                  + t4 * ct_idx_395 * t77 * 1.1111111111111112)
                 + t6 * ct_idx_395 * t77 * 1.1111111111111112)
                + ct_idx_395 * t8 * t77 * 1.1111111111111112)
               + t10 * ct_idx_395 * t77 * 1.1111111111111112)
              + t12 * ct_idx_395 * t77 * 1.1111111111111112)
             + t14 * ct_idx_395 * t77 * 1.1111111111111112)
            + t16 * ct_idx_395 * t77 * 1.1111111111111112)
           + t18 * ct_idx_395 * t77 * 1.1111111111111112)
        / 2.0;
}


MUDA_GENERIC MUDA_INLINE void energy_grad_k(double F1_1,
                   double F1_2,
                   double F1_3,
                   double F2_1,
                   double F2_2,
                   double F2_3,
                   double F3_1,
                   double F3_2,
                   double F3_3,
                   double k,
                   double grad_k[9])
{
    double t2;
    double t3;
    double t38;
    double t39;
    double t4;
    double t40;
    double t41;
    double t42;
    double t43;
    double t44;
    double t45;
    double t46;
    double t5;
    double t6;
    double t7;
    /* ENERGY_GRAD_K */
    /*     GRAD_K = ENERGY_GRAD_K(F1_1,F1_2,F1_3,F2_1,F2_2,F2_3,F3_1,F3_2,F3_3,K)
   */
    /*     This function was generated by the Symbolic Math Toolbox version 23.2.
   */
    /*     04-Jan-2024 14:55:48 */
    t2  = F1_1 * F2_2;
    t3  = F1_2 * F2_1;
    t4  = F1_1 * F2_3;
    t5  = F1_3 * F2_1;
    t6  = F1_2 * F2_3;
    t7  = F1_3 * F2_2;
    t38 = t2 - t3;
    t39 = t4 - t5;
    t40 = t6 - t7;
    t41 = F1_1 * F3_2 - F1_2 * F3_1;
    t42 = F1_1 * F3_3 - F1_3 * F3_1;
    t43 = F1_2 * F3_3 - F1_3 * F3_2;
    t44 = F2_1 * F3_2 - F2_2 * F3_1;
    t45 = F2_1 * F3_3 - F2_3 * F3_1;
    t46 = F2_2 * F3_3 - F2_3 * F3_2;
    t2 = ((((F3_3 * t2 + F3_1 * t6) + F3_2 * t5) - F3_2 * t4) - F3_3 * t3) - F3_1 * t7;
    t3        = 1.0 / t2;
    grad_k[0] = k * (t46 * t2 - t46 * t3) / 2.0;
    grad_k[1] = k * (t43 * t2 - t43 * t3) * -0.5;
    grad_k[2] = k * (t40 * t2 - t40 * t3) / 2.0;
    grad_k[3] = k * (t45 * t2 - t45 * t3) * -0.5;
    grad_k[4] = k * (t42 * t2 - t42 * t3) / 2.0;
    grad_k[5] = k * (t39 * t2 - t39 * t3) * -0.5;
    grad_k[6] = k * (t44 * t2 - t44 * t3) / 2.0;
    grad_k[7] = k * (t41 * t2 - t41 * t3) * -0.5;
    grad_k[8] = k * (t38 * t2 - t38 * t3) / 2.0;
}


MUDA_GENERIC MUDA_INLINE void energy_Hess_k(double F1_1,
                   double F1_2,
                   double F1_3,
                   double F2_1,
                   double F2_2,
                   double F2_3,
                   double F3_1,
                   double F3_2,
                   double F3_3,
                   double k,
                   double Hess_k[81])
{
    double t100;
    double t103;
    double t114;
    double t116;
    double t118;
    double t120;
    double t186;
    double t191;
    double t193;
    double t194;
    double t196;
    double t2;
    double t201;
    double t233;
    double t234;
    double t237;
    double t238;
    double t3;
    double t38;
    double t39;
    double t4;
    double t40;
    double t41;
    double t42;
    double t43;
    double t44;
    double t45;
    double t46;
    double t47;
    double t48;
    double t49;
    double t5;
    double t50;
    double t51;
    double t52;
    double t53;
    double t54;
    double t55;
    double t56;
    double t57;
    double t58;
    double t59;
    double t6;
    double t60;
    double t61;
    double t62;
    double t63;
    double t64;
    double t65;
    double t66;
    double t67;
    double t68;
    double t69;
    double t7;
    double t70;
    double t71;
    double t72;
    double t73;
    double t74;
    double t75;
    double t76;
    double t77;
    double t78;
    double t79;
    double t80;
    double t81;
    double t82;
    double t83;
    double t84;
    double t85;
    double t86;
    double t87;
    double t88;
    double t89;
    double t90;
    double t91;
    double t94;
    double t96;
    double t98;
    /* energy_Hess_k */
    /*     Hess_k = energy_Hess_k(F1_1,F1_2,F1_3,F2_1,F2_2,F2_3,F3_1,F3_2,F3_3,K)
   */
    /*     This function was generated by the Symbolic Math Toolbox version 23.2.
   */
    /*     04-Jan-2024 14:55:57 */
    t2  = F1_1 * F2_2;
    t3  = F1_2 * F2_1;
    t4  = F1_1 * F2_3;
    t5  = F1_3 * F2_1;
    t6  = F1_2 * F2_3;
    t7  = F1_3 * F2_2;
    t38 = t2 - t3;
    t39 = t4 - t5;
    t40 = t6 - t7;
    t41 = F1_1 * F3_2 - F1_2 * F3_1;
    t42 = F1_1 * F3_3 - F1_3 * F3_1;
    t43 = F1_2 * F3_3 - F1_3 * F3_2;
    t44 = F2_1 * F3_2 - F2_2 * F3_1;
    t45 = F2_1 * F3_3 - F2_3 * F3_1;
    t46 = F2_2 * F3_3 - F2_3 * F3_2;
    t47 = t38 * t38;
    t48 = t39 * t39;
    t49 = t40 * t40;
    t50 = t41 * t41;
    t51 = t42 * t42;
    t52 = t43 * t43;
    t53 = t44 * t44;
    t54 = t45 * t45;
    t55 = t46 * t46;
    t56 = t38 * t39;
    t57 = t38 * t40;
    t58 = t39 * t40;
    t59 = t38 * t41;
    t60 = t38 * t42;
    t61 = t39 * t41;
    t62 = t38 * t43;
    t63 = t39 * t42;
    t64 = t40 * t41;
    t65 = t39 * t43;
    t66 = t40 * t42;
    t67 = t40 * t43;
    t68 = t38 * t44;
    t69 = t38 * t45;
    t70 = t39 * t44;
    t71 = t41 * t42;
    t72 = t38 * t46;
    t73 = t39 * t45;
    t74 = t40 * t44;
    t75 = t41 * t43;
    t76 = t39 * t46;
    t77 = t40 * t45;
    t78 = t42 * t43;
    t79 = t40 * t46;
    t80 = t41 * t44;
    t81 = t41 * t45;
    t82 = t42 * t44;
    t83 = t41 * t46;
    t84 = t42 * t45;
    t85 = t43 * t44;
    t86 = t42 * t46;
    t87 = t43 * t45;
    t88 = t43 * t46;
    t89 = t44 * t45;
    t90 = t44 * t46;
    t91 = t45 * t46;
    t38 = ((((F3_3 * t2 + F3_1 * t6) + F3_2 * t5) - F3_2 * t4) - F3_3 * t3) - F3_1 * t7;
    t40  = F1_1 * t38;
    t94  = F1_2 * t38;
    t42  = F1_3 * t38;
    t96  = F2_1 * t38;
    t44  = F2_2 * t38;
    t98  = F2_3 * t38;
    t46  = F3_1 * t38;
    t100 = F3_2 * t38;
    t2   = F3_3 * t38;
    t38  = 1.0 / t38;
    t103 = t38 * t38;
    t39  = F1_1 * t38;
    t114 = F1_2 * t38;
    t41  = F1_3 * t38;
    t116 = F2_1 * t38;
    t43  = F2_2 * t38;
    t118 = F2_3 * t38;
    t45  = F3_1 * t38;
    t120 = F3_2 * t38;
    t38 *= F3_3;
    t186       = k * (t57 + t57 * t103) / 2.0;
    t191       = k * (t68 + t68 * t103) / 2.0;
    t193       = k * (t73 + t73 * t103) / 2.0;
    t194       = k * (t75 + t75 * t103) / 2.0;
    t196       = k * (t79 + t79 * t103) / 2.0;
    t201       = k * (t90 + t90 * t103) / 2.0;
    t233       = k * (((t60 + t40) - t39) + t60 * t103) / 2.0;
    t234       = k * (((t61 - t40) + t39) + t61 * t103) / 2.0;
    t237       = k * (((t65 + t42) - t41) + t65 * t103) / 2.0;
    t238       = k * (((t66 - t42) + t41) + t66 * t103) / 2.0;
    t72        = k * (((t72 + t44) - t43) + t72 * t103) / 2.0;
    t66        = k * (((t74 - t44) + t43) + t74 * t103) / 2.0;
    t65        = k * (((t81 + t46) - t45) + t81 * t103) / 2.0;
    t61        = k * (((t82 - t46) + t45) + t82 * t103) / 2.0;
    t60        = k * (((t86 + t2) - t38) + t86 * t103) / 2.0;
    t75        = k * (((t87 - t2) + t38) + t87 * t103) / 2.0;
    t79        = -(k * (t56 + t56 * t103) / 2.0);
    t73        = -(k * (t58 + t58 * t103) / 2.0);
    t68        = -(k * (t59 + t59 * t103) / 2.0);
    t57        = -(k * (t63 + t63 * t103) / 2.0);
    t90        = -(k * (t67 + t67 * t103) / 2.0);
    t7         = -(k * (t71 + t71 * t103) / 2.0);
    t6         = -(k * (t78 + t78 * t103) / 2.0);
    t5         = -(k * (t80 + t80 * t103) / 2.0);
    t4         = -(k * (t84 + t84 * t103) / 2.0);
    t3         = -(k * (t88 + t88 * t103) / 2.0);
    t2         = -(k * (t89 + t89 * t103) / 2.0);
    t46        = -(k * (t91 + t91 * t103) / 2.0);
    t45        = -(k * (((t62 + t94) - t114) + t62 * t103) / 2.0);
    t44        = -(k * (((t64 - t94) + t114) + t64 * t103) / 2.0);
    t43        = -(k * (((t69 + t96) - t116) + t69 * t103) / 2.0);
    t42        = -(k * (((t70 - t96) + t116) + t70 * t103) / 2.0);
    t41        = -(k * (((t76 + t98) - t118) + t76 * t103) / 2.0);
    t40        = -(k * (((t77 - t98) + t118) + t77 * t103) / 2.0);
    t39        = -(k * (((t83 + t100) - t120) + t83 * t103) / 2.0);
    t38        = -(k * (((t85 - t100) + t120) + t85 * t103) / 2.0);
    Hess_k[0]  = k * (t55 + t55 * t103) / 2.0;
    Hess_k[1]  = t3;
    Hess_k[2]  = t196;
    Hess_k[3]  = t46;
    Hess_k[4]  = t60;
    Hess_k[5]  = t41;
    Hess_k[6]  = t201;
    Hess_k[7]  = t39;
    Hess_k[8]  = t72;
    Hess_k[9]  = t3;
    Hess_k[10] = k * (t52 + t52 * t103) / 2.0;
    Hess_k[11] = t90;
    Hess_k[12] = t75;
    Hess_k[13] = t6;
    Hess_k[14] = t237;
    Hess_k[15] = t38;
    Hess_k[16] = t194;
    Hess_k[17] = t45;
    Hess_k[18] = t196;
    Hess_k[19] = t90;
    Hess_k[20] = k * (t49 + t49 * t103) / 2.0;
    Hess_k[21] = t40;
    Hess_k[22] = t238;
    Hess_k[23] = t73;
    Hess_k[24] = t66;
    Hess_k[25] = t44;
    Hess_k[26] = t186;
    Hess_k[27] = t46;
    Hess_k[28] = t75;
    Hess_k[29] = t40;
    Hess_k[30] = k * (t54 + t54 * t103) / 2.0;
    Hess_k[31] = t4;
    Hess_k[32] = t193;
    Hess_k[33] = t2;
    Hess_k[34] = t65;
    Hess_k[35] = t43;
    Hess_k[36] = t60;
    Hess_k[37] = t6;
    Hess_k[38] = t238;
    Hess_k[39] = t4;
    Hess_k[40] = k * (t51 + t51 * t103) / 2.0;
    Hess_k[41] = t57;
    Hess_k[42] = t61;
    Hess_k[43] = t7;
    Hess_k[44] = t233;
    Hess_k[45] = t41;
    Hess_k[46] = t237;
    Hess_k[47] = t73;
    Hess_k[48] = t193;
    Hess_k[49] = t57;
    Hess_k[50] = k * (t48 + t48 * t103) / 2.0;
    Hess_k[51] = t42;
    Hess_k[52] = t234;
    Hess_k[53] = t79;
    Hess_k[54] = t201;
    Hess_k[55] = t38;
    Hess_k[56] = t66;
    Hess_k[57] = t2;
    Hess_k[58] = t61;
    Hess_k[59] = t42;
    Hess_k[60] = k * (t53 + t53 * t103) / 2.0;
    Hess_k[61] = t5;
    Hess_k[62] = t191;
    Hess_k[63] = t39;
    Hess_k[64] = t194;
    Hess_k[65] = t44;
    Hess_k[66] = t65;
    Hess_k[67] = t7;
    Hess_k[68] = t234;
    Hess_k[69] = t5;
    Hess_k[70] = k * (t50 + t50 * t103) / 2.0;
    Hess_k[71] = t68;
    Hess_k[72] = t72;
    Hess_k[73] = t45;
    Hess_k[74] = t186;
    Hess_k[75] = t43;
    Hess_k[76] = t233;
    Hess_k[77] = t79;
    Hess_k[78] = t191;
    Hess_k[79] = t68;
    Hess_k[80] = k * (t47 + t47 * t103) / 2.0;
}
}  // namespace gipc