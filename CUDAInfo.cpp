#include "CUDAInfo.h"
#include <iostream>

using namespace std;

CUDAInfo::CUDAInfo()
{
    int count = 0;
    cudaGetDeviceCount(&count);
    cout << "Number of CUDA devices: " << count << endl << endl;
    cudaDeviceProp prop;
    for( int i = 0; i < count; i++ )
    {
        cudaGetDeviceProperties(&prop, i);
        cout << "----- DEVICE " << i << " -----" << endl;
        Print(&prop);
    }
}

CUDAInfo::~CUDAInfo()
{
}

void CUDAInfo::Print(cudaDeviceProp* device)
{
    cout << "Name: " << device->name << endl;
    cout << "Integrated GPU: " << device->integrated << endl;
    cout << "Compute capability: " << device->major << "." << device->minor << endl;
    cout << "Clock rate: " << device->clockRate << endl;
    cout << "Device copy overlap (simultaneous memcpy and kernel execution): " << device->deviceOverlap << endl;
    cout << "Kernel execution timeout enabled: " << device->kernelExecTimeoutEnabled << endl;
    cout << "Total global memory: " << device->totalGlobalMem << endl;
    cout << "Total constant memory: " << device->totalConstMem << endl;
    cout << "Max memory pitch: " << device->memPitch << endl;
    cout << "Texture alignment: " << device->textureAlignment << endl;
    cout << "Multiprocessor count: " << device->multiProcessorCount << endl;
    cout << "Shared memory per multiprocessor (per block): " << device->sharedMemPerBlock << endl;
    cout << "Registers per multiprocessor (per block): " << device->regsPerBlock << endl;
    cout << "Threads per warp: " << device->warpSize << endl;
    cout << "Max threads per block: " << device->maxThreadsPerBlock << endl;
    cout << "Max thread dimensions: " << device->maxThreadsDim[0] << ", " << device->maxThreadsDim[1] << ", " << device->maxThreadsDim[2] << endl;
    cout << "Max grid dimensions: " << device->maxGridSize[0] << ", " << device->maxGridSize[1] << ", " << device->maxGridSize[2] << endl;
    cout << "Texture alignment: " << device->textureAlignment << endl;
    cout << "Can map host memory: " << device->canMapHostMemory << endl;
    cout << "Compute mode: " << device->computeMode << endl;
    cout << "Max texture 1D: " << device->maxTexture1D << endl;
    cout << "Max texture 2D: " << device->maxTexture2D[0] << ", " << device->maxTexture2D[1] << endl;
    cout << "Max texture 3D: " << device->maxTexture3D[0] << ", " << device->maxTexture3D[1] << ", " << device->maxTexture3D[2] << endl;
    //cout << "Max texture 2D array: " << device->maxTexture2DArray[0] << ", " << device->maxTexture2DArray[1] << ", " << device->maxTexture2DArray[2] << endl;
    cout << "Supports concurrent kernels: " << device->concurrentKernels << endl;
}
