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

#define HALO 8

#define VALID_RESPONSES_X 48
#define VALID_RESPONSES_Y 48

#ifndef FILTERING_H_
#define FILTERING_H_

class Filtering
{

public:

    Filtering(int ndim, int dw, int dh, int dd, int dt, int fw, int fh, int fd, int ft, float* input_data, float* output_data, float* filters, int nf);
    ~Filtering();

    double DoConvolution2DShared();
	double DoConvolution2DTexture();
	
	double DoConvolution3DShared();
	double DoConvolution3DTexture();
	              
	double DoConvolution4DShared();
	
private:


    void Copy3DFilterToConstantMemory(float* h_Filter, int z, int FILTER_W, int FILTER_H);
	void Copy4DFilterToConstantMemory(float* h_Filter, int z, int t, int FILTER_W, int FILTER_H, int FILTER_D);
    
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
	float	*d_Image, *d_Volume;
	float	*d_Filter_Response;

	cudaArray *d_Image_Array, *d_Volume_Array;
	cudaExtent	VOLUME_SIZE;
	cudaChannelFormatDesc floatTex;

	int threadsInX, threadsInY, threadsInZ;
    int blocksInX, blocksInY, blocksInZ;
	dim3 dimGrid, dimBlock;

	int xBlockDifference, yBlockDifference;
};

#endif 