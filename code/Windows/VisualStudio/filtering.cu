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
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

Filtering::Filtering(int dw, int dh, int fw, int fh, float* input_data, float* filter, float* output_data)
{
    DATA_W = dw;
    DATA_H = dh;
	
    FILTER_W = fw;
    FILTER_H = fh;
	
	h_Data = input_data;
	h_Filter = filter;	
	h_Filter_Response = output_data;

	UNROLLED = false;
}


Filtering::Filtering(int dw, int dh, int dd, int fw, int fh, int fd, float* input_data, float* filter, float* output_data)
{
    DATA_W = dw;
    DATA_H = dh;
	DATA_D = dd;
    
    FILTER_W = fw;
    FILTER_H = fh;
	FILTER_D = fd;

	h_Data = input_data;
	h_Filter = filter;	
	h_Filter_Response = output_data;

	UNROLLED = false;
}


Filtering::Filtering(int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* filter, float* output_data)
{
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
	tex_Image.addressMode[0] = cudaAddressModeBorder;	// values outside image are 0
	tex_Image.addressMode[1] = cudaAddressModeBorder;	// values outside image are 0

    // Allocate memory for the image and the filter response
	cudaMallocArray(&d_Image_Array, &floatTex, DATA_W, DATA_H);
	cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * sizeof(float));

	// Copy image to texture
	cudaMemcpyToArray(d_Image_Array, 0, 0, h_Data, DATA_W * DATA_H * sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex_Image, d_Image_Array);

	// Copy filter coefficients to constant memory
	cudaMemcpyToSymbol(c_Filter_2D, h_Filter, FILTER_W * FILTER_H * sizeof(float));

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
		if (FILTER_W == 3)
		{	
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_3x3, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_3x3<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 5)
		{	
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_5x5, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_5x5<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 7)
		{	
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_7x7, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 9)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_9x9, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 11)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_11x11, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 13)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_13x13, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_13x13<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 15)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_15x15, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_15x15<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
		else if (FILTER_W == 17)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_17x17, h_Filter, FILTER_W * FILTER_H * sizeof(float));

			Convolution_2D_Texture_Unrolled_17x17<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);    
	sdkDeleteTimer(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaUnbindTexture( tex_Image );
	cudaFreeArray( d_Image_Array );
    cudaFree( d_Filter_Response );	   
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

    // 32 threads along x, 32 along y
	threadsInX = 32;
	threadsInY = 32;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)VALID_RESPONSES_X);
    blocksInY = (int)ceil((float)DATA_H / (float)VALID_RESPONSES_Y);

    dimGrid  = dim3(blocksInX, blocksInY, 1);
    dimBlock = dim3(threadsInX, threadsInY, 1);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Do 2D convolution	
	if (!UNROLLED)
	{
		// Copy filter coefficients to constant memory
		cudaMemcpyToSymbol(c_Filter_2D, h_Filter, FILTER_W * FILTER_H * sizeof(float));
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		Convolution_2D_Shared<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H, FILTER_W, FILTER_H);
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 3)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_3x3, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_3x3<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 5)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_5x5, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_5x5<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 7)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_7x7, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 9)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_9x9, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 11)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_11x11, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 13)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_13x13, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_13x13<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 15)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_15x15, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_15x15<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
		else if (FILTER_W == 17)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_17x17, h_Filter, FILTER_W * FILTER_H * sizeof(float));
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_Unrolled_17x17<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Image, DATA_W, DATA_H);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);
	sdkDeleteTimer(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Image );
    cudaFree( d_Filter_Response );
}



void Filtering::DoConvolution3DTexture()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	floatTex = cudaCreateChannelDesc<float>();

	tex_Volume.normalized = false;                       // do not access with normalized texture coordinates
	tex_Volume.filterMode = cudaFilterModeLinear;        // linear interpolation
	tex_Volume.addressMode[0] = cudaAddressModeBorder;	 // values outside volume are 0
	tex_Volume.addressMode[1] = cudaAddressModeBorder;	 // values outside volume are 0
	tex_Volume.addressMode[2] = cudaAddressModeBorder;	 // values outside volume are 0
	
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
	cudaMemcpyToSymbol(c_Filter_3D, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

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

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Do 3D convolution with texture memory	
	if (!UNROLLED)
	{
		// Copy filter coefficients to constant memory
		cudaMemcpyToSymbol(c_Filter_3D, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

		Convolution_3D_Texture<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D, FILTER_W, FILTER_H, FILTER_D);
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 3)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_3x3x3, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_3x3x3<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		else if (FILTER_W == 5)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_5x5x5, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_5x5x5<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		else if (FILTER_W == 7)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_7x7x7, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_7x7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		else if (FILTER_W == 9)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_9x9x9, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_9x9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}	
		else if (FILTER_W == 11)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_11x11x11, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_11x11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		else if (FILTER_W == 13)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_13x13x13, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_13x13x13<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		else if (FILTER_W == 15)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_15x15x15, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_15x15x15<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		/* 
		Commented because c_Filter_3D[17][17][17] and c_Filter_17x17x17[17][17][17] at the same time use to much constant memory
		else if (FILTER_W == 17)
		{
			// Copy filter coefficients to constant memory
			cudaMemcpyToSymbol(c_Filter_17x17x17, h_Filter, FILTER_W * FILTER_H * FILTER_D * sizeof(float));

			Convolution_3D_Texture_Unrolled_17x17x17<<<dimGrid, dimBlock>>>(d_Filter_Response, DATA_W, DATA_H, DATA_D);
		}
		*/
		
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);
	sdkDeleteTimer(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaUnbindTexture( tex_Volume );
	cudaFreeArray( d_Volume_Array );
    cudaFree( d_Filter_Response );    	   
}



void Filtering::Copy3DFilterToConstantMemory(float* h_Filter, int z, int FILTER_W, int FILTER_H)
{
	cudaMemcpyToSymbol(c_Filter_2D, &h_Filter[z * FILTER_W * FILTER_H], FILTER_W * FILTER_H * sizeof(float));
}

void Filtering::Copy3DFilterToConstantMemoryUnrolled(float* h_Filter, int z, int FILTER_W, int FILTER_H)
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
	cudaMemcpyToSymbol(c_Filter_2D,  &h_Filter[z * FILTER_W * FILTER_H + t * FILTER_W * FILTER_H * FILTER_D], FILTER_W * FILTER_H * sizeof(float));
}


void Filtering::Copy4DFilterToConstantMemoryUnrolled(float* h_Filter, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D)
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

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	if (!UNROLLED)
	{
		// Loop over slices in filters
		int z_offset = -(FILTER_D - 1)/2;
		for (int zz = FILTER_D -1; zz >= 0; zz--)
		{
			// Copy current 2D filter coefficients to constant memory
			Copy3DFilterToConstantMemory(h_Filter, zz, FILTER_W, FILTER_H); 
	
			// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			Convolution_2D_Shared_For_3D<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D, FILTER_W, FILTER_H);
			z_offset++;
		}
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 3)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_3x3<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;			
			}
		}
		else if (FILTER_W == 5)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_5x5<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 7)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 9)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 11)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 13)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_13x13<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 15)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_15x15<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
		else if (FILTER_W == 17)
		{
			// Loop over slices in filters
			int z_offset = -(FILTER_D - 1)/2;
			for (int zz = FILTER_D -1; zz >= 0; zz--)
			{
				// Copy current 2D filter coefficients to constant memory
				Copy3DFilterToConstantMemoryUnrolled(h_Filter, zz, FILTER_W, FILTER_H); 
		
				// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
				cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
				Convolution_2D_Shared_For_3D_Unrolled_17x17<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volume, z_offset, DATA_W, DATA_H, DATA_D);
				z_offset++;
			}
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);
    sdkDeleteTimer(&hTimer);

	// Copy result to host
	cudaMemcpy(h_Filter_Response, d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
	cudaFree( d_Volume );
    cudaFree( d_Filter_Response );	
}

void Filtering::DoConvolution4DShared()
{        
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	
    // Allocate memory for the volumes
	cudaMalloc((void **)&d_Volumes,  DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float));
	// Calculate the filter response for one volume at a time
	cudaMalloc((void **)&d_Filter_Response,  DATA_W * DATA_H * DATA_D * sizeof(float));

	// Copy volume to device
	cudaMemcpy(d_Volumes, h_Data, DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float), cudaMemcpyHostToDevice);

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
	
	// Perform the non-separable 4D convolution by summing 2D convolutions

	convolution_time = 0.0;

	if (!UNROLLED)
	{
		// Loop over volumes
		for (int t = 0; t < DATA_T; t++)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);

			// Reset filter response
			cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

			// Loop over timepoints in filter
			int t_offset = -(FILTER_T - 1)/2;
			for (int tt = FILTER_T-1; tt >= 0; tt--)
			{
				// Loop over slices in filter
				int z_offset = -(FILTER_D - 1)/2;
				for (int zz = FILTER_D-1; zz >= 0; zz--)
				{
					// Copy current 2D filter coefficients to constant memory
					Copy4DFilterToConstantMemory(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
					// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
					cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
					Convolution_2D_Shared_For_4D<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T, FILTER_W, FILTER_H);
					z_offset++;
				}
				t_offset++;
			}

			checkCudaErrors(cudaDeviceSynchronize());
			sdkStopTimer(&hTimer);
			convolution_time += sdkGetTimerValue(&hTimer);

			// Copy volume filter response to host
			cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
		}
	}
	else if (UNROLLED)
	{
		if (FILTER_W == 3)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_3x3<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 5)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_5x5<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 7)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_7x7<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 9)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_9x9<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 11)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_11x11<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 13)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_13x13<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 15)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_15x15<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}	
		else if (FILTER_W == 17)
		{
			// Loop over volumes
			for (int t = 0; t < DATA_T; t++)
			{
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&hTimer);
				sdkStartTimer(&hTimer);

				// Reset filter response
				cudaMemset(d_Filter_Response, 0, DATA_W * DATA_H * DATA_D * sizeof(float));

				// Loop over timepoints in filter
				int t_offset = -(FILTER_T - 1)/2;
				for (int tt = FILTER_T-1; tt >= 0; tt--)
				{
					// Loop over slices in filter
					int z_offset = -(FILTER_D - 1)/2;
					for (int zz = FILTER_D-1; zz >= 0; zz--)
					{
						// Copy current 2D filter coefficients to constant memory
						Copy4DFilterToConstantMemoryUnrolled(h_Filter, zz, tt, FILTER_W, FILTER_H, FILTER_D); 
	
						// Do the 2D convolution with the current filter coefficients and increment the result inside the kernel
						cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
						Convolution_2D_Shared_For_4D_Unrolled_17x17<<<dimGrid, dimBlock>>>(d_Filter_Response, d_Volumes, z_offset, t, t_offset, DATA_W, DATA_H, DATA_D, DATA_T);
						z_offset++;
					}
					t_offset++;
				}

				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&hTimer);
				convolution_time += sdkGetTimerValue(&hTimer);

				// Copy volume filter response to host
				cudaMemcpy(&h_Filter_Response[t * DATA_W * DATA_H * DATA_D], d_Filter_Response, DATA_W * DATA_H * DATA_D * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}																		
	}

    sdkDeleteTimer(&hTimer);	
	
    // Free allocated memory
	cudaFree( d_Volumes );
    cudaFree( d_Filter_Response );		    
}

// This code is only for measuring the time
void Filtering::DoFiltering2DFFT()
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	cufftHandle FFTplan;

	float2 *d_Image_float2, *d_Filter_float2;

	// Allocate memory for the volume and the filter response
	cudaMalloc((void **)&d_Image_float2,  DATA_W * DATA_H * sizeof(float2));
	cudaMalloc((void **)&d_Filter_float2,  DATA_W * DATA_H * sizeof(float2));
	
	
	// Create a plan for FFT
	cufftPlan2d(&FFTplan, DATA_W, DATA_H, CUFFT_C2C);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	// Transform data and filter
	cufftExecC2C(FFTplan, (cufftComplex *)d_Image_float2,    (cufftComplex *)d_Image_float2,   CUFFT_FORWARD);
	cufftExecC2C(FFTplan, (cufftComplex *)d_Filter_float2,   (cufftComplex *)d_Filter_float2,   CUFFT_FORWARD);
	
	// Perform multiplication
	threadsInX = 32;
	threadsInY = 16;
	threadsInZ = 1;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);
	blocksInZ = 1;

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	MultiplyAndNormalize2D<<<dimGrid, dimBlock>>>(d_Image_float2, d_Filter_float2, DATA_W, DATA_H);

	// Inverse transform of filter response
	cufftExecC2C(FFTplan, (cufftComplex *)d_Image_float2,     (cufftComplex *)d_Image_float2,   CUFFT_INVERSE);
	

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);
	sdkDeleteTimer(&hTimer);

	cudaFree( d_Image_float2 );
    cudaFree( d_Filter_float2 );	
	cufftDestroy(FFTplan);
}


// This code is only for measuring the time
void Filtering::DoFiltering3DFFT()
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	cufftHandle FFTplan;

	float2 *d_Volume_float2, *d_Filter_float2;

	// Allocate memory for the volume and the filter response
	cudaMalloc((void **)&d_Volume_float2,  DATA_W * DATA_H * DATA_D * sizeof(float2));
	cudaMalloc((void **)&d_Filter_float2,  DATA_W * DATA_H * DATA_D * sizeof(float2));	
	
	// Create a plan for FFT
	cufftPlan3d(&FFTplan, DATA_W, DATA_H, DATA_D, CUFFT_C2C);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	// Transform data and filter
	cufftExecC2C(FFTplan, (cufftComplex *)d_Volume_float2,   (cufftComplex *)d_Volume_float2,   CUFFT_FORWARD);
	cufftExecC2C(FFTplan, (cufftComplex *)d_Filter_float2,   (cufftComplex *)d_Filter_float2,   CUFFT_FORWARD);
	
	// Perform multiplication
	threadsInX = 32;
	threadsInY = 16;
	threadsInZ = 1;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);
	blocksInZ = (int)ceil((float)DATA_D / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	MultiplyAndNormalize3D<<<dimGrid, dimBlock>>>(d_Volume_float2, d_Filter_float2, DATA_W, DATA_H, DATA_D);

	// Inverse transform of filter response
	cufftExecC2C(FFTplan, (cufftComplex *)d_Volume_float2,     (cufftComplex *)d_Volume_float2,   CUFFT_INVERSE);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time = sdkGetTimerValue(&hTimer);
	sdkDeleteTimer(&hTimer);

	cudaFree( d_Volume_float2 );
    cudaFree( d_Filter_float2 );	
	cufftDestroy(FFTplan);
}

// This code is only for measuring the time
void Filtering::DoFiltering4DFFT()
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	convolution_time = 0.0;

	float2 *d_Volumes_float2, *d_Filter_float2;
	int NUMBER_OF_FFTS, FFT_sizes[2];

	// Allocate memory for the volume and the filter response
	cudaMalloc((void **)&d_Volumes_float2,  DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float2));
	cudaMalloc((void **)&d_Filter_float2,  DATA_W * DATA_H * DATA_D * DATA_T * sizeof(float2));	

	// Create plan for a batch of 2D FFTs along x and y
	cufftHandle FFTplan_XY;	
	NUMBER_OF_FFTS = DATA_D * DATA_T;
	FFT_sizes[0] = DATA_W;
	FFT_sizes[1] = DATA_H;
	cufftPlanMany(&FFTplan_XY, 2, FFT_sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, NUMBER_OF_FFTS);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	// Apply forward 2D FFT's along x and y
	cufftExecC2C(FFTplan_XY, (cufftComplex*)d_Volumes_float2, (cufftComplex *)d_Volumes_float2, CUFFT_FORWARD);
	cufftExecC2C(FFTplan_XY, (cufftComplex*)d_Filter_float2, (cufftComplex *)d_Filter_float2, CUFFT_FORWARD);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

	cufftDestroy(FFTplan_XY);

	// Flip data from (x,y,z,t) to (z,t,x,y)

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	threadsInX = 16;
	threadsInY = 1;
	threadsInZ = 16;

    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = DATA_H;
    blocksInZ = (int)ceil((float)DATA_D / (float)threadsInZ);

    dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	// Flip data from (x,y,z,t) to (z,t,x,y)
	for (int t = 0; t < DATA_T; t++)
	{
		FlipXYZTtoZTXY<<<dimGrid, dimBlock>>>(d_Volumes_float2, d_Filter_float2, t, DATA_W, DATA_H, DATA_D, DATA_T);
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

	// Create plan for a batch of 2D FFTs along z and t
	cufftHandle FFTplan_ZT;
	NUMBER_OF_FFTS = DATA_W * DATA_H;
	FFT_sizes[0] = DATA_D;
	FFT_sizes[1] = DATA_T;
	cufftPlanMany(&FFTplan_ZT, 2, FFT_sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, NUMBER_OF_FFTS);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	// Apply forward 2D FFT's along z and t
	cufftExecC2C(FFTplan_ZT, (cufftComplex*)d_Volumes_float2, (cufftComplex *)d_Volumes_float2, CUFFT_FORWARD);
	cufftExecC2C(FFTplan_ZT, (cufftComplex*)d_Filter_float2, (cufftComplex *)d_Filter_float2, CUFFT_FORWARD);	
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

	cufftDestroy(FFTplan_ZT);


	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // 32 threads along x, 16 along y
	threadsInX = 32;
	threadsInY = 16;
	threadsInZ = 1;

    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = (int)ceil((float)DATA_H / (float)threadsInY);
    blocksInZ = DATA_D;

    dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dimBlock = dim3(threadsInX, threadsInY, threadsInZ);	
	
	// Apply the filter by doing a complex multiplication in the frequency domain
	for (int t = 0; t < DATA_T; t++)
	{
		MultiplyAndNormalize4D<<<dimGrid, dimBlock>>>(d_Volumes_float2, d_Filter_float2, t, DATA_W, DATA_H, DATA_D, DATA_T);		
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

	// Create plan for a batch of 2D FFTs along z and t
	NUMBER_OF_FFTS = DATA_W * DATA_H;
	FFT_sizes[0] = DATA_D;
	FFT_sizes[1] = DATA_T;
	cufftPlanMany(&FFTplan_ZT, 2, FFT_sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, NUMBER_OF_FFTS);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	// Apply inverse 2D FFT's along z and t
	cufftExecC2C(FFTplan_ZT, (cufftComplex*)d_Volumes_float2, (cufftComplex *)d_Volumes_float2, CUFFT_INVERSE);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

	cufftDestroy(FFTplan_ZT);

	// Flip data from (z,t,x,y) to (x,y,z,t)

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	threadsInX = 16;
	threadsInY = 1;
	threadsInZ = 16;

    blocksInX = (int)ceil((float)DATA_W / (float)threadsInX);
    blocksInY = DATA_H;
    blocksInZ = (int)ceil((float)DATA_D / (float)threadsInZ);

    dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	// Flip data from (z,t,x,y) to (x,y,z,t)
	for (int t = 0; t < DATA_T; t++)
	{
		FlipZTXYtoXYZT<<<dimGrid, dimBlock>>>(d_Volumes_float2, d_Filter_float2, t, DATA_W, DATA_H, DATA_D, DATA_T);
	}

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);

    
	// Create plan for a batch of 2D FFTs along x and y
	NUMBER_OF_FFTS = DATA_D * DATA_T;
	FFT_sizes[0] = DATA_W;
	FFT_sizes[1] = DATA_H;
	cufftPlanMany(&FFTplan_XY, 2, FFT_sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, NUMBER_OF_FFTS);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
		
	// Apply inverse 2D FFT's along x and y
	cufftExecC2C(FFTplan_XY, (cufftComplex*)d_Volumes_float2, (cufftComplex *)d_Volumes_float2, CUFFT_INVERSE);
	
	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    convolution_time += sdkGetTimerValue(&hTimer);
	
	cufftDestroy(FFTplan_XY);
		
	sdkDeleteTimer(&hTimer);

	cudaFree( d_Volumes_float2 );
    cudaFree( d_Filter_float2 );	
}
