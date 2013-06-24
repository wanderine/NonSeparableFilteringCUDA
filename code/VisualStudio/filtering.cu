/*
	Non-separable 2D, 3D and 4D Filtering with CUDA
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "filtering.h"
#include "cuda.h"
//#include <cutil_inline.h>
#include "filtering_kernel.cu"

#include <helper_functions.h>
#include <helper_cuda.h>

Filtering::Filtering(int ndim, int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* output_data, float* filters, int nf)
{
	NDIM = ndim;
    DATA_W = dw;
    DATA_H = dh;
	DATA_D = dd;
	DATA_T = dt;
    
    FILTER_W = fw;
    FILTER_H = fh;
	FILTER_D = fd;
	FILTER_T = ft;

	h_Data = input_data;
	h_Filter_Responses = output_data;

	h_Filters = filters;
	NUMBER_OF_FILTERS = nf;
}

Filtering::~Filtering()
{
}



double Filtering::DoConvolution2DShared()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Allocate memory for the filter response and the image
    float *d_Data, *d_Filter_Response;
	
	cudaMalloc((void **)&d_Data,  DATA_W * DATA_H * sizeof(float));
    cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * sizeof(float));

	// Copy image to GPU
	cudaMemcpy(d_Data, h_Data, DATA_W * DATA_H * sizeof(float), cudaMemcpyHostToDevice);


	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter, h_Filters, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

    // 32 threads along x, 16 along y
	threadsInX = 32;
	threadsInY = 32;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)VALID_RESPONSES_X);
    blocksInY = (int)ceil((float)DATA_H / (float)VALID_RESPONSES_Y);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, 1);
    dim3 dimBlock = dim3(threadsInX, threadsInY, 1);

    int xBlockDifference = 16;
    int yBlockDifference = 16;

    // Do 2D convolution	
	Convolution_2D_Shared<<<dimGrid, dimBlock>>>(d_Filter_Responses, d_Data, DATA_W, DATA_H, FILTER_W, FILTER_H, xBlockDifference, yBlockDifference);

	// Copy result to host
	cudaMemcpy(h_Filter_Responses, d_Filter_Responses, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Data );
    cudaFree( d_Filter_Response );

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    
	sdkDeleteTimer(&hTimer);

    cudaDeviceReset();
	return gpuTime;
}
