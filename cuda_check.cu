#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device;
    cudaGetDevice(&device); // Get the current device ID

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device); // Get device properties

    std::cout << "GPU Device: " << prop.name << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Blocks per Dimension: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "Max Threads per Dimension: " << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;

    return 0;
}
