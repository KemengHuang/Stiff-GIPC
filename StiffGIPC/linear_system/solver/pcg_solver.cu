#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
    checkCudaErrors(cudaMallocHost(&host_pinned_dot_res, sizeof(Float)));
}
SizeT PCGSolver::solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    Ap.resize(b.size());
    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    return iter;
}


SizeT PCGSolver::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    auto dot = [this](muda::CDenseVectorView<Float> a,
                      muda::CDenseVectorView<Float> b,
                      muda::VarView<Float>          res)
    {
        ctx().dot(a, b, res);
        //blas.dot(a, b, res);
    };

    auto axpby = [this](Float                         alpha,
                        muda::CDenseVectorView<Float> x,
                        Float                         beta,
                        muda::DenseVectorView<Float>  y)
    {
        ctx().axpby(alpha, x, beta, y);
        //blas.axpby(alpha, x, beta, y);
    };

    // r = b - A * x
    {
        // r = b;
        r.buffer_view().copy_from(b.buffer_view());
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    Float alpha, beta, rz, rz0;

    {
        Timer timer{"preconditioner"};
        // z = P * r (apply preconditioner)
        apply_preconditioner(z, r);
    }


    // p = z
    p = z;

    // init rz
    // rz = r^T * z
    {
        Timer timer{"dot"};
        dot(r.cview(), z.cview(), dot_res);

        // rz = dot(r.cview(), z.cview());
    }

    {
        Timer timer{"dot_res_copy_time"};
        dot_res.view().copy_to(host_pinned_dot_res);
        rz = *host_pinned_dot_res;
    }


    rz0 = std::abs(rz);

    // check convergence
    if(/*accuracy_statisfied(r) &&*/ std::abs(rz) <= m_config.global_tol_rate * rz0)
        return k;

    for(k = 1; k < max_iter; ++k)
    {
        {
            Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            Timer timer{"dot"};
            // alpha = rz / (p^T * Ap)

            // alpha = rz / dot(p.cview(), Ap.cview());
            dot(p.cview(), Ap.cview(), dot_res);
        }

        {
            Timer timer{"dot_res_copy_time"};
            // alpha = rz / dot_res;

            dot_res.view().copy_to(host_pinned_dot_res);
            alpha = rz / *host_pinned_dot_res;
        }

        {
            Timer timer{"axpby"};

            // x = x + alpha * p
            axpby(alpha, p.cview(), 1.0, x);

            // r = r - alpha * Ap
            axpby(-alpha, Ap.cview(), 1.0, r.view());
        }

        // check convergence
        if(/*accuracy_statisfied(r) &&*/ std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            Timer timer{"preconditioner"};
            // z = P * r (apply preconditioner)
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            Timer timer{"dot"};
            // rz_new = r^T * z
            // rz_new = dot(r.cview(), z.cview());

            dot(r.cview(), z.cview(), dot_res);
        }

        {
            Timer timer{"dot_res_copy_time"};
            dot_res.view().copy_to(host_pinned_dot_res);
            rz_new = *host_pinned_dot_res;
        }

        // beta = rz_new / rz
        beta = rz_new / rz;

        {
            Timer timer{"axpby"};
            // p = z + beta * p
            axpby(1.0, z.cview(), beta, p.view());
        }

        // update rz
        rz = rz_new;
    }

    return k;
}

SizeT PCGSolver::pcg_0(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    auto dot = [this](muda::CDenseVectorView<Float> a,
                      muda::CDenseVectorView<Float> b,
                      muda::VarView<Float>          res)
    {
        //ctx().dot(a, b, res);
        blas.dot(a, b, res);
    };

    auto axpby = [this](Float                         alpha,
                        muda::CDenseVectorView<Float> x,
                        Float                         beta,
                        muda::DenseVectorView<Float>  y)
    {
        ctx().axpby(alpha, x, beta, y);
        //blas.axpby(alpha, x, beta, y);
    };

    // r = b - A * x
    {
        // r = b;
        r.buffer_view().copy_from(b.buffer_view());
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    Float alpha, beta, rz, rz0;

    {
        Timer timer{"preconditioner"};
        // z = P * r (apply preconditioner)
        apply_preconditioner(z, r);
    }


    // p = z
    p = z;

    // init rz
    // rz = r^T * z
    {
        Timer timer{"dot"};
        dot(r.cview(), z.cview(), dot_res);

        // rz = dot(r.cview(), z.cview());
    }

    {
        Timer timer{"dot_res_copy_time"};
        rz = dot_res;
    }


    rz0 = std::abs(rz);

    // check convergence
    if(/*accuracy_statisfied(r) &&*/ std::abs(rz) <= m_config.global_tol_rate * rz0)
        return k;

    for(k = 1; k < max_iter; ++k)
    {
        {
            Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            Timer timer{"dot"};
            // alpha = rz / (p^T * Ap)

            // alpha = rz / dot(p.cview(), Ap.cview());
            dot(p.cview(), Ap.cview(), dot_res);
        }

        {
            Timer timer{"dot_res_copy_time"};
            // alpha = rz / dot_res;
            alpha = rz / dot_res;
        }

        {
            Timer timer{"axpby"};

            // x = x + alpha * p
            axpby(alpha, p.cview(), 1.0, x);

            // r = r - alpha * Ap
            axpby(-alpha, Ap.cview(), 1.0, r.view());
        }

        // check convergence
        if(/*accuracy_statisfied(r) &&*/ std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            Timer timer{"preconditioner"};
            // z = P * r (apply preconditioner)
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            Timer timer{"dot"};
            // rz_new = r^T * z
            // rz_new = dot(r.cview(), z.cview());

            dot(r.cview(), z.cview(), dot_res);
        }

        {
            Timer timer{"dot_res_copy_time"};
            rz_new = dot_res;
        }

        // beta = rz_new / rz
        beta = rz_new / rz;

        {
            Timer timer{"axpby"};
            // p = z + beta * p
            axpby(1.0, z.cview(), beta, p.view());
        }

        // update rz
        rz = rz_new;
    }

    return k;
}
}  // namespace gipc