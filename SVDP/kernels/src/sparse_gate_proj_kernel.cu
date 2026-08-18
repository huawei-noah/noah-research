/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cuda.h>
#include <cuda_fp16.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#include "model_hyperparams.hpp"

#define NUM_THREADS 256
#define WARP_SIZE 32
#define SHARED_MEMORY_SIZE NUM_THREADS/WARP_SIZE

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void blockReduceSum(float val, float *smem, uint32_t tid) {
    val = warpReduceSum(val);

    uint32_t lane = tid % warpSize;
    uint32_t wid = tid / warpSize;
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    if (tid < warpSize) {
        val = tid < NUM_THREADS/WARP_SIZE ? smem[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) smem[0] = val;
    }
}

__global__ void sparse_gate_proj(__half* __restrict__ weight, __half* __restrict__ x, __half* __restrict__ prediction) {
    __shared__ float smem[SHARED_MEMORY_SIZE];

    if (__half2float(prediction[blockIdx.x]) <= 0) {
        if (threadIdx.x == 0) {
            prediction[blockIdx.x] = __float2half(0.0f);
        }
        return;
    }

    uint32_t tid = threadIdx.x;

    __half2* mat_row = reinterpret_cast<__half2*>(weight + blockIdx.x * N_COLS);
    __half2* vec = reinterpret_cast<__half2*>(x);

    float partial_sum = 0.f;

 #pragma unroll 16
    for (uint32_t col = tid; col < N_COLS / 2; col += blockDim.x) {
        __half2 matval = mat_row[col];
        __half2 vecval = vec[col];

        partial_sum += __half2float(matval.x) * __half2float(vecval.x) + __half2float(matval.y) *  __half2float(vecval.y);
    }

    blockReduceSum(partial_sum, smem, tid);
    if (tid == 0) {
        prediction[blockIdx.x] = __float2half(smem[0]);
    }
}


void launch_sparse_gate_proj(__half* weight, __half* x, __half* prediction) {

    dim3 grid_size(N_ROWS);
    dim3 block_size(NUM_THREADS);

    sparse_gate_proj<<<grid_size, block_size>>>(weight, x, prediction);
}