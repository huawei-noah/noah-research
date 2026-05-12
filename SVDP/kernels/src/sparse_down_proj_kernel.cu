#include <cuda.h>
#include <cuda_fp16.h>

#include "model_hyperparams.hpp"


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


__device__ __forceinline__ float2 warpReduceSum(float2 val) {
#pragma unroll
    for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

// weight is stored in row-major format
__global__ void sparse_down_proj(__half* __restrict__ weight, __half* __restrict__ x, __half* __restrict__ res) {

    float2 partial_sum = make_float2(0.0f, 0.0f);
    __shared__ float2 warp_sum[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    uint32_t col_id = 2*(blockIdx.y * BLOCK_SIZE_X + threadIdx.x);
    uint32_t warp_id = threadIdx.y;
    uint32_t row_id = warp_id;
    __half *vec_p = x + row_id;
    __half2 *mat_p = reinterpret_cast<__half2*>(weight + row_id * N_COLS + col_id);
    float2 mat_val = make_float2(0.0f, 0.0f);

    for (uint32_t iter = 0; iter < N_ROWS; iter += BLOCK_SIZE_Y)
    {
        float vec_val = __half2float(vec_p[iter]);
        if (vec_val == 0.0f){
            continue;
        }
        mat_val = __half22float2(mat_p[iter * N_COLS/2]);
        partial_sum.x += mat_val.x * vec_val;
        partial_sum.y += mat_val.y * vec_val;
    }

    // BLOCK REDUCE SUM
    warp_sum[threadIdx.x][threadIdx.y] = partial_sum;
    __syncthreads();
    float2 val = warp_sum[threadIdx.y][threadIdx.x];
    val = warpReduceSum(val);

    if (threadIdx.x == 0) {
        *reinterpret_cast<__half2*>(res + 2*(blockIdx.y * BLOCK_SIZE_X + threadIdx.y))  = __float22half2_rn(val);
    }
}

void launch_sparse_down_proj(__half *weight, __half *x, __half *res) {

    dim3 grid_dim(1, N_COLS / BLOCK_SIZE_X / 2);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    sparse_down_proj<<<grid_dim, block_dim>>>(weight, x, res);
}
