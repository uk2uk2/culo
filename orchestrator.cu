// orchestrator.cu
// A minimal proof-of-concept demonstrating Lua orchestration of two CUDA-based dot product routines.
// Compile with: nvcc orchestrator.cu -llua -o orchestrator

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#define N 1024  // array size

// -----------------------------
// CUDA Kernels
// -----------------------------

// Naive dot product kernel: each thread computes one product and adds via atomicAdd.
__global__ void dotProductKernelNaive(const float *a, const float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float prod = a[idx] * b[idx];
        atomicAdd(result, prod);
    }
}

// Optimized dot product kernel: uses shared memory reduction per block.
__global__ void dotProductKernelOptimized(const float *a, const float *b, float *result, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float prod = (idx < n) ? a[idx] * b[idx] : 0.0f;
    sdata[tid] = prod;
    __syncthreads();

    // Perform reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use atomic add to accumulate block results.
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// -----------------------------
// Helper Functions (Host)
// -----------------------------

// Simple routine to fill an array with a constant value.
void fillArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

// -----------------------------
// Exposed Functions to Lua
// -----------------------------

// This function calls the naive CUDA dot product.
// It allocates two arrays, fills them with fixed values (1.0 and 2.0),
// launches the kernel, and returns the dot product to Lua.
int lua_dotProductNaive(lua_State *L) {
    int n = N;
    size_t size = n * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float h_result = 0.0f;

    fillArray(h_a, n, 1.0f);  // For example, all ones.
    fillArray(h_b, n, 2.0f);  // For example, all twos.

    float *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dotProductKernelNaive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);

    lua_pushnumber(L, h_result);
    return 1; // one return value
}

// This function calls the optimized CUDA dot product kernel.
int lua_dotProductOptimized(lua_State *L) {
    int n = N;
    size_t size = n * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float h_result = 0.0f;

    fillArray(h_a, n, 1.0f);
    fillArray(h_b, n, 2.0f);

    float *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    dotProductKernelOptimized<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);

    lua_pushnumber(L, h_result);
    return 1;
}

// -----------------------------
// Lua Module Registration
// -----------------------------

// Register the functions under a Lua module named "myalgos".
int luaopen_myalgos(lua_State *L) {
    static const struct luaL_Reg myalgos [] = {
        {"dotProductNaive", lua_dotProductNaive},
        {"dotProductOptimized", lua_dotProductOptimized},
        {NULL, NULL}
    };
    luaL_newlib(L, myalgos);
    return 1;
}

// -----------------------------
// Main: Setup Lua and Run the Script
// -----------------------------

int main(int argc, char *argv[]) {
    // Initialize a new Lua state.
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);

    // Register our module with Lua.
    luaL_requiref(L, "myalgos", luaopen_myalgos, 1);
    lua_pop(L, 1);  // remove module table from the stack

    // Ensure a Lua script file is provided.
    if (argc < 2) {
        printf("Usage: %s <script.lua>\n", argv[0]);
        return 1;
    }

    // Execute the Lua orchestrator script.
    if (luaL_dofile(L, argv[1]) != LUA_OK) {
        const char *error = lua_tostring(L, -1);
        printf("Lua Error: %s\n", error);
        lua_close(L);
        return 1;
    }

    lua_close(L);
    return 0;
}
