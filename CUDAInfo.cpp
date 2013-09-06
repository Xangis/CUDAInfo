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
        cout << "==== DEVICE " << i << " ====" << endl;
        Print(&prop);
    }
}

CUDAInfo::~CUDAInfo()
{
}

void CUDAInfo::Print(cudaDeviceProp* device)
{
    cout << "---- General ----" << endl;
    cout << "Name: " << device->name << endl;
    cout << "Compute capability: " << device->major << "." << device->minor << endl;
    cout << "---- Processor ----" << endl;
    cout << "Multiprocessor count: " << device->multiProcessorCount << endl;
    cout << "Clock rate (kHz): " << device->clockRate << endl;
    cout << "Kernel execution timeout enabled: " << device->kernelExecTimeoutEnabled << endl;
    cout << "Registers per multiprocessor (per block): " << device->regsPerBlock << endl;
    cout << "Threads per warp: " << device->warpSize << endl;
    cout << "Max threads per block: " << device->maxThreadsPerBlock << endl;
    cout << "Max threads per multiprocessor: " << device->maxThreadsPerMultiProcessor << endl;
    cout << "Max thread dimensions: " << device->maxThreadsDim[0] << ", " << device->maxThreadsDim[1] << ", " << device->maxThreadsDim[2] << endl;
    cout << "Supports concurrent kernels: " << device->concurrentKernels << endl;
    cout << "Async engine count: " << device->asyncEngineCount << endl;
    cout << "Compute mode: " << device->computeMode << endl;
    cout << "---- Memory ----" << endl;
    cout << "Total global memory: " << device->totalGlobalMem << endl;
    cout << "Total constant memory: " << device->totalConstMem << endl;
    cout << "Memory clock rate (kHz): " << device->memoryClockRate << endl;
    cout << "Memory bus width (bits): " << device->memoryBusWidth << endl;
    cout << "L2 cache size: " << device->l2CacheSize << endl;
    cout << "Shared memory per multiprocessor (per block): " << device->sharedMemPerBlock << endl;
    cout << "Max memory pitch: " << device->memPitch << endl;
    cout << "Device copy overlap (simultaneous memcpy and kernel execution): " << device->deviceOverlap << endl;
    cout << "Can map host memory: " << device->canMapHostMemory << endl;
    cout << "Max grid dimensions: " << device->maxGridSize[0] << ", " << device->maxGridSize[1] << ", " << device->maxGridSize[2] << endl;
    cout << "---- Hardware ----" << endl;
    cout << "Integrated GPU: " << device->integrated << endl;
    cout << "PCI bus ID: " << device->pciBusID << endl;
    cout << "PCI device ID: " << device->pciDeviceID << endl;
    cout << "PCI domain ID: " << device->pciDomainID << endl;
    cout << "ECC enabled: " << device->ECCEnabled << endl;
    cout << "Unified addressing: " << device->unifiedAddressing << endl;
    cout << "---- Driver ----" << endl;
    cout << "Using Tesla TCC driver: " << device->tccDriver << endl;
    cout << "---- Texture ----" << endl;
    cout << "Texture alignment: " << device->textureAlignment << endl;
    cout << "Texture pitch alignment: " << device->texturePitchAlignment << endl;
    cout << "Max texture 1D: " << device->maxTexture1D << endl;
    cout << "Max texture 1D mipmap: " << device->maxTexture1DMipmap << endl;
    cout << "Max texture 1D linear: " << device->maxTexture1DLinear << endl;
    cout << "Max texture 1D layered: " << device->maxTexture1DLayered[0] << ", " << device->maxTexture1DLayered[1] << endl;
    cout << "Max texture 2D: " << device->maxTexture2D[0] << ", " << device->maxTexture2D[1] << endl;
    cout << "Max texture 2D mipmap: " << device->maxTexture2DMipmap[0] << ", " << device->maxTexture2DMipmap[1] << endl;
    cout << "Max texture 2D linear: " << device->maxTexture2DLinear[0] << ", " << device->maxTexture2DLinear[1] << endl;
    cout << "Max texture 2D gather: " << device->maxTexture2DGather[0] << ", " << device->maxTexture2DGather[1] << endl;
    cout << "Max texture 2D layered: " << device->maxTexture2DLayered[0] << ", " << device->maxTexture2DLayered[1] << ", " << device->maxTexture2DLayered[2] << endl;
    cout << "Max texture 3D: " << device->maxTexture3D[0] << ", " << device->maxTexture3D[1] << ", " << device->maxTexture3D[2] << endl;
    cout << "Max texture cubemap: " << device->maxTextureCubemap << endl;
    cout << "Max texture cubemap layered: " << device->maxTextureCubemapLayered[0] << ", " << device->maxTextureCubemapLayered[1] << endl;
    cout << "---- Surface ----" << endl;
    cout << "Surface alignment: " << device->surfaceAlignment << endl;
    cout << "Max surface 1D: " << device->maxSurface1D << endl;
    cout << "Max surface 1D layered: " << device->maxSurface1DLayered[0] << ", " << device->maxSurface1DLayered[1] << endl;
    cout << "Max surface 2D: " << device->maxSurface2D[0] << ", " << device->maxSurface2D[1] << endl;
    cout << "Max surface 2D layered: " << device->maxSurface2DLayered[0] << ", " << device->maxSurface2DLayered[1] << ", " << device->maxSurface2DLayered[2] << endl;
    cout << "Max surface 3D: " << device->maxSurface3D[0] << ", " << device->maxSurface3D[1] << ", " << device->maxSurface3D[2] << endl;
    cout << "Max sutface cubemap: " << device->maxSurfaceCubemap << endl;
    cout << "Max surface cubemap layered: " << device->maxSurfaceCubemapLayered[0] << ", " << device->maxSurfaceCubemapLayered[1] << endl;
    //cout << "Max texture 2D array: " << device->maxTexture2DArray[0] << ", " << device->maxTexture2DArray[1] << ", " << device->maxTexture2DArray[2] << endl;
}
