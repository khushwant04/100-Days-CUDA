#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to perform matrix multiplication.
// Each thread computes one element (row, col) of the output matrix C.
// A is of size MxK, B is of size KxN, and C is the resulting MxN matrix.
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    // Calculate the row index for the element to compute.
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in A and C

    // Calculate the column index for the element to compute.
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in B and C

    // Only compute the dot product if within matrix bounds.
    if (row < M && col < N)
    {
        float sum = 0.0f; // Initialize the accumulator for the dot product.

        // Loop over the K dimension to compute the dot product of row 'row' of A and column 'col' of B.
        for (int i = 0; i < K; i++)
        {
            // Multiply the corresponding elements and add to the accumulator.
            // Access A at row 'row' and column 'i': A[row * K + i]
            // Access B at row 'i' and column 'col': B[i * N + col]
            sum += A[row * K + i] * B[i * N + col];
        }

        // Write the computed value to matrix C at position (row, col).
        // C is stored in row-major order.
        C[row * N + col] = sum;
    }
}

int main(void)
{
    // Define matrix dimensions.
    // A is an MxK matrix, B is a KxN matrix, and C will be an MxN matrix.
    int M = 4; // Number of rows in A and C.
    int K = 3; // Number of columns in A and rows in B.
    int N = 5; // Number of columns in B and C.

    // Calculate the size in bytes for each matrix.
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host (CPU) memory for matrices A, B, and C.
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialize matrix A with increasing values: 1, 2, 3, ...
    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = static_cast<float>(i + 1);
    }

    // Initialize matrix B with increasing values: 1, 2, 3, ...
    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = static_cast<float>(i + 1);
    }

    // Allocate device (GPU) memory for matrices A, B, and C.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy matrices A and B from host to device memory.
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define the dimensions of each thread block (16x16 threads).
    dim3 blockDim(16, 16);

    // Calculate the grid dimensions so that every element of matrix C is covered.
    // The division rounds up to handle cases where the matrix size isn't a multiple of the block size.
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel for matrix multiplication.
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Synchronize the device to ensure the kernel has finished executing.
    cudaDeviceSynchronize();

    // Copy the result matrix C from device memory back to host memory.
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print matrix A.
    printf("Matrix A (MxK):\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            printf("%6.2f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    // Print matrix B.
    printf("\nMatrix B (KxN):\n");
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%6.2f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Print the result matrix C.
    printf("\nMatrix C (A x B) (MxN):\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%6.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free the device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free the host memory.
    free(h_A);
    free(h_B);
    free(h_C);

    // Return success.
    return 0;
}
