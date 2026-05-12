#include <cuda.h>
#include <cuda_fp16.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#include "model_hyperparams.hpp"

#define NUM_THREADS 128
#define WARP_SIZE 32
#define SHARED_MEMORY_SIZE NUM_THREADS/WARP_SIZE 

__device__ __forceinline__ float warpReduceSum(float val) {
    for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void blockReduceSum(float val, float *smem, uint32_t tid) {
    val = warpReduceSum(val);

    if (blockDim.x > warpSize) {
        uint32_t lane = tid % warpSize;
        uint32_t wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDim.x, warpSize) ? smem[tid] : 0.0f;
            val = warpReduceSum(val);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }

}

__global__ void sparse_up_proj_drelu(__half* __restrict__ weight, __half* __restrict__ x, __half* __restrict__ gate_out, float threshold) {
    __shared__ float smem[SHARED_MEMORY_SIZE];

    if (__half2float(gate_out[blockIdx.x]) <= threshold) {
        if (threadIdx.x == 0) {
            gate_out[blockIdx.x] = __float2half(0.0f);
        }
        return;
    }

    uint32_t tid = threadIdx.x;

    __half2* mat_row = reinterpret_cast<__half2*>(weight + blockIdx.x * N_COLS);
    __half2* vec = reinterpret_cast<__half2*>(x);

    float partial_sum = 0.f;

    for (uint32_t col = tid; col < N_COLS / 2; col += blockDim.x) {
        __half2 matval = mat_row[col];
        __half2 vecval = vec[col];

        partial_sum += __half2float(matval.x) * __half2float(vecval.x) + __half2float(matval.y) * __half2float(vecval.y);
    }

    blockReduceSum(partial_sum, smem, tid);
    if (tid == 0) {
        if (smem[0] > 0.f){
            gate_out[blockIdx.x] = __float2half(smem[0] * __half2float(gate_out[blockIdx.x]));
        }
        else {
            gate_out[blockIdx.x] = __float2half(0.0f);
        }
    }
}


void launch_sparse_up_proj_drelu(__half* weight, __half* x, __half* gate_out, float threshold) {

    dim3 grid_size(N_ROWS);
    dim3 block_size(NUM_THREADS);

    sparse_up_proj_drelu<<<grid_size, block_size>>>(weight, x, gate_out, threshold);
}