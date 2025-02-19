#include<iostream>
#include<cuda_runtime.h>

#define BLOCK_SIZE 16 

__global__ void matrixAdd(const float *A, const float *B, float *C, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x<width && y<width)
    {
        C[y * width + x] = A[y * width + x] + B[y * width + x];
    }
}

int main()
{
    const int WIDTH = 1024;
    int size = WIDTH * WIDTH * sizeof(float);

    //host data
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c = new float[size];

    // Initialize host data
    for (int i = 0; i < WIDTH * WIDTH; ++i)
    {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device data
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (WIDTH + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrixAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, WIDTH);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // output first few results for verification
    for (int i = 0; i < 5; ++i)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}