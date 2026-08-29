#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/block/block_reduce.cuh>



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    int valid_items = min(numbers - idof, static_cast<int>(blockDim.x));
    double temp = idx < numbers ? a[idx] * b[idx] : 0.0;

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    temp = BlockReduce(storage).Sum(temp, valid_items);
    if(threadIdx.x == 0)
        squeue[blockIdx.x] = temp;
}



__global__ void add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    int valid_items = min(numbers - idof, static_cast<int>(blockDim.x));
    double temp = idx < numbers ? mem[idx] : 0.0;

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    temp = BlockReduce(storage).Sum(temp, valid_items);
    if(threadIdx.x == 0)
        mem[blockIdx.x] = temp;
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    CUDA_SAFE_CALL(cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    return iter;
}


SizeT PCGSolver::pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
    }

    p.copy_from(z);
    rz0 = rz;

    for(k = 1; k < max_iter; ++k)
    {
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       z.size());

            alpha = rz / dot_res;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        if(std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
        }

        beta = rz_new / rz;

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;
    }

    return k;
}

}  // namespace gipc
