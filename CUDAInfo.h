#ifndef _CUDAINFO_H_
#define _CUDAINFO_H_
// If you have the CUDA toolkit installed, this is typically located somewhere like
// C:\NVIDIA\CUDA\CUDAToolkit\include
#include "cuda_runtime.h"

class CUDAInfo
{
public:
    CUDAInfo( );
    ~CUDAInfo( );
    void Print(cudaDeviceProp* device);
};

#endif
