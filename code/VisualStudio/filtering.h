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

#include <cuda.h>
#include <cuda_runtime.h>

#define HALO 8

// For a halo of 8
#define VALID_RESPONSES_X 80
#define VALID_RESPONSES_Y 48

// For a halo of 4
//#define VALID_RESPONSES_X 88
//#define VALID_RESPONSES_Y 56



#ifndef FILTERING_H_
#define FILTERING_H_

class Filtering
{

public:

	Filtering(int dw, int dh, int fw, int fh, float* input_data, float* filter, float* output_data);
	Filtering(int dw, int dh, int dd, int fw, int fh, int fd, float* input_data, float* filter, float* output_data);
    Filtering(int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* filter, float* output_data);
	
    ~Filtering();

    void DoConvolution2DShared();
	void DoConvolution2DTexture();
	void DoFiltering2DFFT();

	void DoConvolution3DShared();
	void DoConvolution3DTexture();
	void DoFiltering3DFFT();              

	void DoConvolution4DShared();
	void DoFiltering4DFFT();

	double GetConvolutionTime();
	
	void SetUnrolled(bool);

private:

    void Copy3DFilterToConstantMemory(float* h_Filter, int z, int FILTER_W, int FILTER_H);
	void Copy3DFilterToConstantMemoryUnrolled(float* h_Filter, int z, int FILTER_W, int FILTER_H);
	void Copy4DFilterToConstantMemory(float* h_Filter, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D);
	void Copy4DFilterToConstantMemoryUnrolled(float* h_Filter, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D);
    
	// Number of dimensions
	int NDIM;

    // Data sizes	
	int DATA_W;
    int DATA_H;
	int DATA_D;
	int DATA_T;

	// Filter sizes
	int FILTER_W;
	int FILTER_H;
	int FILTER_D;
	int FILTER_T;

	int NUMBER_OF_FILTERS;

	bool UNROLLED;

	// Host pointers
	float	*h_Data;
	float	*h_Filter_Response;
	float	*h_Filter;
	
	// Device pointers
	float	*d_Image, *d_Volume, *d_Volumes;
	float	*d_Filter_Response;

	cudaArray *d_Image_Array, *d_Volume_Array;
	cudaExtent	VOLUME_SIZE;
	cudaChannelFormatDesc floatTex;

	int threadsInX, threadsInY, threadsInZ;
    int blocksInX, blocksInY, blocksInZ;
	dim3 dimGrid, dimBlock;

	double	convolution_time;
};

#endif 