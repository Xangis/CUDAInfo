#ifndef _CUDAINFO_H_
#define _CUDAINFO_H_

#include "cuda_runtime.h"

class CUDAInfo
{
public:
    CUDAInfo( );
    ~CUDAInfo( );
    void Print(cudaDeviceProp* device);
};

#endif
