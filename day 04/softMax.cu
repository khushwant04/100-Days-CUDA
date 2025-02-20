#include<iostream>
#include<cuda_runtime.h>

__global__ void softmax_kernel(float *input, float *output, int N)
{
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if(idx >=N)
    {
        return;
    }

    // load data into shared memory
    shared_mem[tid] = input[idx];
    __syncthreads();

    // Compute max for numerical stability
    float max_val = -1e20;
    for (int i = 0; i < blockDim.x;i++)
    {
        max_val = fmaxf(max_val, shared_mem[i]);
    }
    __syncthreads();

    // Compute exponentials and sum
    shared_mem[tid] = expf(shared_mem[tid] - max_val);
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++)
    {
        sum += shared_mem[i];
    }
    __syncthreads();

    // Compute final softmax output
    output[idx] = shared_mem[tid] / sum;
}

void softmax(float *d_input, float *d_output, int N)
{
    int thredsPerBlock = 256;
    int blocksPerGrid = (N + thredsPerBlock - 1) / thredsPerBlock;

    softmax_kernel<<<blocksPerGrid, thredsPerBlock, thredsPerBlock * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
}


int main()
{
    int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *d_input;
    float *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Initialize input
    for (int i = 0; i < N;i++)
    {
        h_input[i] = (float)(rand() % 10);
    }

    // Copy to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Compute softmax
    softmax(d_input, d_output, N);

    // Copy back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print a few results
    for (int i = 0; i < 10; i++)
    {
        std::cout << h_output[i] << std::endl;
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}