#pragma once

#include <gipc/cuda/muda_def.h>
#include <gipc/cuda/buffer.h>
#include <gipc/cuda/atomic.h>
#include <gipc/cuda/eigen_device.h>
#include <gipc/cuda/linear_system.h>
#include <gipc/cuda/tools/debug_log.h>

// The following headers use CUDA-specific syntax (kernels, <<<>>>, CUB, etc.)
// and are only meaningful when compiled by nvcc.
#ifdef __CUDACC__
#include <gipc/cuda/parallel_for.h>
#include <gipc/cuda/buffer_launch.h>
#include <gipc/cuda/cub_wrappers.h>
#include <gipc/cuda/svd3x3_impl.h>
#endif
