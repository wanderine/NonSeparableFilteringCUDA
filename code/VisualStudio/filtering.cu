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
#include "filtering_kernel.cu"

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>

Filtering::Filtering(int ndim, int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* filter, float* output_data, int nf)
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
	h_Filter = filter;	
	h_Filter_Response = output_data;

	NUMBER_OF_FILTERS = nf;

	UNROLLED = false;
}

Filtering::~Filtering()
{
}

double Filtering::GetConvolutionTime()
{
    return convolution_time;
}

void Filtering::SetUnrolled(bool unr)
{
    UNROLLED = unr;
}

void Filtering::DoConvolution2DTexture()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);



    floatTex = cudaCreateChannelDesc<float>();

	tex_Image.normalized = false;                       // do not access with normalized texture coordinates
	tex_Image.filterMode = cudaFilterModeLinear;        // linear interpolation

    // Allocate memory for the image and the filter response
	cudaMallocArray(&d_Image_Array, &floatTex, DATA_W, DATA_H);
	cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * sizeof(float));

	// Copy image to texture
	cudaMemcpyToArray(d_Image_Array, 0, 0, h_Data, DATA_W * DATA_H * sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex_Image, d_Image_Array);

	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter_2D, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

    // 32 threads along x, 16 along y
	threadsInX = 32;
	threadsInY = 16;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, 1);
    dim3 dimBlock = dim3(threadsInX, threadsInY, 1);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Do 2D convolution with texture memory	
	if (!UNROLLED)
	{
		Convolution_2D_Texture<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, FILTER_W, FILTER_H);
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 7)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_7x7, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Texture_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 9)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_9x9, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Texture_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 11)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_11x11, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Texture_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = 0.001 * sdkGetTimerValue(&hTimer);    
	sdkDeleteTimer(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaUnbindTexture( tex_Image );
	cudaFreeArray( d_Image_Array );
    cudaFree( d_Filter_Response );

	
    cudaDeviceReset();
}




void Filtering::DoConvolution2DShared()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);


	
    // Allocate memory for the filter response and the image    
	cudaMalloc((void **)&d_Image,  DATA_W * DATA_H * sizeof(float));
    cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * sizeof(float));

	// Copy image to GPU
	cudaMemcpy(d_Image, h_Data, DATA_W * DATA_H * sizeof(float), cudaMemcpyHostToDevice);

	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter_2D, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);


    // 32 threads along x, 32 along y
	threadsInX = 32;
	threadsInY = 32;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)VALID_RESPONSES_X);
    blocksInY = (int)ceil((float)DATA_H / (float)VALID_RESPONSES_Y);

    dimGrid  = dim3(blocksInX, blocksInY, 1);
    dimBlock = dim3(threadsInX, threadsInY, 1);

    xBlockDifference = 16;
    yBlockDifference = 16;

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Do 2D convolution	
	if (!UNROLLED)
	{
		Convolution_2D_Shared<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H, FILTER_W, FILTER_H, xBlockDifference, yBlockDifference);
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 7)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_7x7, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Shared_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H, xBlockDifference, yBlockDifference);
		}
		else if (FILTER_W == 9)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_9x9, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Shared_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H, xBlockDifference, yBlockDifference);
		}
		else if (FILTER_W == 11)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_11x11, h_Filter, FILTER_W * FILTER_H * sizeof(float), 0, cudaMemcpyHostToDevice);

			Convolution_2D_Shared_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H, xBlockDifference, yBlockDifference);
		}


	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = 0.001 * sdkGetTimerValue(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Image );
    cudaFree( d_Filter_Response );


    
	sdkDeleteTimer(&hTimer);
    cudaDeviceReset();
}


void Filtering::DoConvolution3DTexture()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	floatTex = cudaCreateChannelDesc<float>();


	tex_Volume.normalized = false;                       // do not access with normalized texture coordinates
	tex_Volume.filterMode = cudaFilterModeLinear;        // linear interpolation

	
	// Allocate 3D array
	VOLUME_SIZE = make_cudaExtent(DATA_W, DATA_H, DATA_D);
	cudaMalloc3DArray(&d_Volume_Array, &floatTex, VOLUME_SIZE);

	// Bind the array to the 3D texture
	cudaBindTextureToArray(tex_Volume, d_Volume_Array, floatTex);

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_Data, sizeof(float)*DATA_W , DATA_W, DATA_H);
	copyParams.dstArray = d_Volume_Array;
	copyParams.extent   = VOLUME_SIZE;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

    // Allocate memory for the filter response
	cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * DATA_D * sizeof(float));

	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter_3D, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float), 0, cudaMemcpyHostToDevice);

    // 32 threads along x, 16 along y
	threadsInX = 32;
	threadsInY = 16;
	threadsInZ = 1;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);
	blocksInZ = (int)ceil((float)DATA_D / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    // Do 3D convolution with texture memory	
	if (!UNROLLED)
	{
		Convolution_3D_Texture<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D, FILTER_W, FILTER_H, FILTER_D);
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 7)
		{
			Convolution_3D_Texture_Unrolled_7x7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
	}

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaUnbindTexture( tex_Volume );
	cudaFreeArray( d_Volume_Array );
    cudaFree( d_Filter_Response );

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = 0.001 * sdkGetTimerValue(&hTimer);
    
	sdkDeleteTimer(&hTimer);
    cudaDeviceReset();
}



void Filtering::Copy3DFilterToConstantMemory(float* h_Filter, int z, int FILTER_W, int FILTER_H)
{
	if (FILTER_W == 3)
	{
		cudaMemcpyToSymbol(c_Filter_3x3,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 5)
	{
		cudaMemcpyToSymbol(c_Filter_5x5,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 7)
	{
		cudaMemcpyToSymbol(c_Filter_7x7,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 9)
	{
		cudaMemcpyToSymbol(c_Filter_9x9,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 11)
	{
		cudaMemcpyToSymbol(c_Filter_11x11,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 13)
	{
		cudaMemcpyToSymbol(c_Filter_13x13,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 15)
	{
		cudaMemcpyToSymbol(c_Filter_15x15,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 17)
	{
		cudaMemcpyToSymbol(c_Filter_17x17,  &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
	}
}

void Filtering::Copy4DFilterToConstantMemory(float* h_Filter, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D)
{
	if (FILTER_W == 3)
	{
		cudaMemcpyToSymbol(c_Filter_3x3,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 5)
	{
		cudaMemcpyToSymbol(c_Filter_5x5,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 7)
	{
		cudaMemcpyToSymbol(c_Filter_7x7,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 9)
	{
		cudaMemcpyToSymbol(c_Filter_9x9,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 11)
	{
		cudaMemcpyToSymbol(c_Filter_11x11,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 13)
	{
		cudaMemcpyToSymbol(c_Filter_13x13,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 15)
	{
		cudaMemcpyToSymbol(c_Filter_15x15,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
	else if (FILTER_W == 17)
	{
		cudaMemcpyToSymbol(c_Filter_17x17,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
	}
}


void Filtering::DoConvolution3DShared()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Allocate memory for the volume and the filter response
	cudaMalloc((void **)&d_Volume,  DATA_W * DATA_H * DATA_D * sizeof(float));
	cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * DATA_D * sizeof(float));

	// Copy volume to device
	cudaMemcpy(d_Volume, h_Data, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyHostToDevice);

	// Reset filter response
	cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

    // 32 threads along x, 32 along y
	threadsInX = 32;
	threadsInY = 32;
	threadsInZ = 1;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)VALID_RESPONSES_X);
    blocksInY = (int)ceil((float)DATA_H / (float)VALID_RESPONSES_Y);
	blocksInZ = DATA_D;

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	// Perform the non-separable 3D convolution by summing 2D convolutions

	xBlockDifference = 16;
	yBlockDifference = 16;

	// Loop over slices in filters
	int z_offset = -(FILTER_D - 1)/2;
	for (int zz = FILTER_D -1; zz >= 0; zz--)
	{
		// Copy current 2D filter coefficients to constant memory
		Copy3DFilterToConstantMemory(h_Filter, zz, FILTER_W, FILTER_H); 

		// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		Convolution_2D_Shared_For_3D<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D, FILTER_W, FILTER_H, xBlockDifference, yBlockDifference);
		z_offset++;
	}

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Volume );
    cudaFree( d_Filter_Response );

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = 0.001 * sdkGetTimerValue(&hTimer);
    
	sdkDeleteTimer(&hTimer);
    cudaDeviceReset();
}
