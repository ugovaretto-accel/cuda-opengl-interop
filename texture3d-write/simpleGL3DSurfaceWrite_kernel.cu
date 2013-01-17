/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//Global scope surface to bind to
surface<void, cudaSurfaceType3D> surfaceWrite;

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to just write something to the texture
///////////////////////////////////////////////////////////////////////////////
__global__
void kernel(dim3 texture_dim)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if(x >= texture_dim.x || y >= texture_dim.y || z >= texture_dim.z)
	{
		return;
	}

	float4 element = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	surf3Dwrite(element, surfaceWrite, x*sizeof(float4), y, z);
}

extern "C"
void launch_kernel(cudaArray *cuda_image_array, dim3 texture_dim)
{
	dim3 block_dim(8, 8, 8);
	dim3 grid_dim(texture_dim.x/block_dim.x, texture_dim.y/block_dim.y, texture_dim.z/block_dim.z);
 
    cudaError_t err;
	//Bind voxel array to a writable CUDA surface
    err = cudaBindSurfaceToArray(surfaceWrite, cuda_image_array);
	if( err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        return;
    }

	kernel<<< grid_dim, block_dim >>>(texture_dim);

	err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}
