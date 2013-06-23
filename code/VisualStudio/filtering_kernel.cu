
#ifndef CONV2_GPU_KERNEL_CU_
#define CONV2_GPU_KERNEL_CU_

#include "filtering.h"
#include "help_functions.cu"

__device__ __constant__ float c_Filter[7][7];

__device__ float Convolve_2D(float image[64][64], int y, int x)
{
	float pixel; float sum = 0.0f;
	
    pixel = image[y - 3][x - 3]; 
    sum += pixel * c_Filter[6][6];
 
    pixel = image[y - 2][x - 3]; 
    sum += pixel * c_Filter[5][6];

    pixel = image[y - 1][x - 3]; 
    sum += pixel * c_Filter[4][6];

    pixel = image[y + 0][x - 3]; 
    sum += pixel * c_Filter[3][6];

    pixel = image[y + 1][x - 3]; 
    sum += pixel * c_Filter[2][6];

    pixel = image[y + 2][x - 3]; 
    sum += pixel * c_Filter[1][6];

    pixel = image[y + 3][x - 3]; 
    sum += pixel * c_Filter[0][6];



    pixel = image[y - 3][x - 2]; 
    sum += pixel * c_Filter[6][5];

    pixel = image[y - 2][x - 2]; 
    sum += pixel * c_Filter[5][5];

    pixel = image[y - 1][x - 2]; 
    sum += pixel * c_Filter[4][5];

    pixel = image[y + 0][x - 2]; 
    sum += pixel * c_Filter[3][5];

    pixel = image[y + 1][x - 2]; 
    sum += pixel * c_Filter[2][5];

    pixel = image[y + 2][x - 2]; 
    sum += pixel * c_Filter[1][5];

    pixel = image[y + 3][x - 2]; 
    sum += pixel * c_Filter[0][5];


    pixel = image[y - 3][x - 1]; 
    sum += pixel * c_Filter[6][4];

    pixel = image[y - 2][x - 1]; 
    sum += pixel * c_Filter[5][4];

    pixel = image[y - 1][x - 1]; 
    sum += pixel * c_Filter[4][4];

    pixel = image[y + 0][x - 1]; 
    sum += pixel * c_Filter[3][4];

    pixel = image[y + 1][x - 1]; 
    sum += pixel * c_Filter[2][4];

    pixel = image[y + 2][x - 1]; 
    sum += pixel * c_Filter[1][4];

    pixel = image[y + 3][x - 1]; 
    sum += pixel * c_Filter[0][4];


    pixel = image[y - 3][x + 0]; 
    sum += pixel * c_Filter[6][3];

    pixel = image[y - 2][x + 0]; 
    sum += pixel * c_Filter[5][3];

    pixel = image[y - 1][x + 0]; 
    sum += pixel * c_Filter[4][3];

    pixel = image[y + 0][x + 0]; 
    sum += pixel * c_Filter[3][3];

    pixel = image[y + 1][x + 0]; 
    sum += pixel * c_Filter[2][3];

    pixel = image[y + 2][x + 0]; 
    sum += pixel * c_Filter[1][3];

    pixel = image[y + 3][x + 0]; 
    sum += pixel * c_Filter[0][3];

    pixel = image[y - 3][x + 1]; 
    sum += pixel * c_Filter[6][2];
  
    pixel = image[y - 2][x + 1]; 
    sum += pixel * c_Filter[5][2];
  
    pixel = image[y - 1][x + 1]; 
    sum += pixel * c_Filter[4][2];

    pixel = image[y + 0][x + 1]; 
    sum += pixel * c_Filter[3][2];

    pixel = image[y + 1][x + 1]; 
    sum += pixel * c_Filter[2][2];
 
    pixel = image[y + 2][x + 1]; 
    sum += pixel * c_Filter[1][2];

    pixel = image[y + 3][x + 1]; 
    sum += pixel * c_Filter[0][2];
 


    pixel = image[y - 3][x + 2]; 
    sum += pixel * c_Filter[6][1];

    pixel = image[y - 2][x + 2]; 
    sum += pixel * c_Filter[5][1];

    pixel = image[y - 1][x + 2]; 
    sum += pixel * c_Filter[4][1];
 
    pixel = image[y + 0][x + 2]; 
    sum += pixel * c_Filter[3][1];

    pixel = image[y + 1][x + 2]; 
    sum += pixel * c_Filter[2][1];
    
    pixel = image[y + 2][x + 2]; 
    sum += pixel * c_Filter[1][1];

    pixel = image[y + 3][x + 2]; 
    sum += pixel * c_Filter[0][1];


    pixel = image[y - 3][x + 3]; 
    sum += pixel * c_Filter[6][0];

    pixel = image[y - 2][x + 3]; 
    sum += pixel * c_Filter[5][0];

    pixel = image[y - 1][x + 3]; 
    sum += pixel * c_Filter[4][0];

    pixel = image[y + 0][x + 3]; 
    sum += pixel * c_Filter[3][0];

    pixel = image[y + 1][x + 3]; 
    sum += pixel * c_Filter[2][0];

    pixel = image[y + 2][x + 3]; 
    sum += pixel * c_Filter[1][0];
 
    pixel = image[y + 3][x + 3]; 
    sum += pixel * c_Filter[0][0];

	return sum;
}


__global__ void Convolution_2D(float *d_Filter_Response, float* d_Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{   
	volatile int x = blockIdx.x * VALID_FILTER_RESPONSES_X + threadIdx.x;
	volatile int y = blockIdx.y * VALID_FILTER_RESPONSES_Y + threadIdx.y;
	
    if (x >= (DATA_W + xBlockDifference) || y >= (DATA_H + yBlockDifference) )
	    return;
	
	__shared__ float s_Image[64][64];    

	// Blocks (16 x 16 pixels)
		
	// 1   2  3  4
	// 5   6  7  8
	// 9  10 11 12
	// 13 14 15 16

	s_Image[threadIdx.y][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 16][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 32][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y + 48][threadIdx.x] = 0.0f;
	s_Image[threadIdx.y][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 16][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
	s_Image[threadIdx.y + 48][threadIdx.x + 32] = 0.0f;


	// Read data into shared memory

	
	// First row, blocks 1 + 2
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y - 8) >= 0) && ((y - 8) < DATA_H) )
	{
		s_Image[threadIdx.y][threadIdx.x] = d_Image[Calculate_2D_Index(x - 8,y - 8,DATA_W)];	
	}

	// First row, blocks 3 + 4
	if ( ((x + 24) < DATA_W) && ((y - 8) >= 0) && ((y - 8) < DATA_H) )
	{
		s_Image[threadIdx.y][threadIdx.x + 32] = d_Image[Calculate_2D_Index(x + 24,y - 8,DATA_W)];	
	}

	// Second row, blocks 5 + 6
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 8) < DATA_H) )
	{
		s_Image[threadIdx.y + 16][threadIdx.x] = d_Image[Calculate_2D_Index(x - 8,y + 8,DATA_W)];	
	}

	// Second row, blocks 7 + 8
	if ( ((x + 24) < DATA_W) && ((y + 8) < DATA_H) )
	{
		s_Image[threadIdx.y + 16][threadIdx.x + 32] = d_Image[Calculate_2D_Index(x + 24,y + 8,DATA_W)];	
	}

	// Third row, blocks 9 + 10
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 24) < DATA_H) )
	{
		s_Image[threadIdx.y + 32][threadIdx.x] = d_Image[Calculate_2D_Index(x - 8,y + 24,DATA_W)];	
	}

	// Third row, blocks 11 + 12
	if ( ((x + 24) < DATA_W) && ((y + 24) < DATA_H) )
	{
		s_Image[threadIdx.y + 32][threadIdx.x + 32] = d_Image[Calculate_2D_Index(x + 24,y + 24,DATA_W)];	
	}

	// Fourth row, blocks 13 + 14
	if ( ((x - 8) >= 0) && ((x - 8) < DATA_W) && ((y + 40) < DATA_H) )
	{
		s_Image[threadIdx.y + 48][threadIdx.x] = d_Image[Calculate_2D_Index(x - 8,y + 40,DATA_W)];	
	}

	// Fourth row, blocks 15 + 16		
	if ( ((x + 24) < DATA_W) && ((y + 40) < DATA_H) )
	{
		s_Image[threadIdx.y + 48][threadIdx.x + 32] = d_Image[Calculate_2D_Index(x + 24,y + 40,DATA_W)];	
	}
	

	__syncthreads();

	// Only threads inside the image do the convolution, calculate valid filter responses for 48 x 48 pixels

	if ( (x < DATA_W) && (y < DATA_H) )
	{
		int idx = Calculate_2D_Index(x,y,DATA_W);
		float filter_response = Convolve_2D(s_Image, threadIdx.y + 8, threadIdx.x + 8);
		d_Filter_Response[idx] = filter_response;
	}

	if ( (x < DATA_W) && ((y + 16) < DATA_H) )
	{
		int idx = Calculate_2D_Index(x,y + 16,DATA_W);
		float filter_response = Convolve_2D(s_Image, threadIdx.y + 24, threadIdx.x + 8);
		d_Filter_Response[idx] = filter_response;
	}

	if ( (x < DATA_W) && ((y + 32) < DATA_H) )
	{
		int idx = Calculate_2D_Index(x,y + 32,DATA_W);
		float filter_response = Convolve_2D(s_Image, threadIdx.y + 40, threadIdx.x + 8);
		d_Filter_Response[idx] = filter_response;
	}

	if (threadIdx.x < 16)
	{
		if ( ((x + 32) < DATA_W) && (y < DATA_H) )
		{
			int idx = Calculate_2D_Index(x + 32,y,DATA_W);
			float filter_response = Convolve_2D(s_Image, threadIdx.y + 8, threadIdx.x + 40);
			d_Filter_Response[idx] = filter_response;
		}

		if ( ((x + 32) < DATA_W) && ((y + 16) < DATA_H) )
		{
			int idx = Calculate_2D_Index(x + 32,y + 16,DATA_W);
			float filter_response = Convolve_2D(s_Image, threadIdx.y + 24, threadIdx.x + 40);
			d_Filter_Response[idx] = filter_response;
		}

		if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
		{
    		int idx = Calculate_2D_Index(x + 32,y + 32,DATA_W);
			float filter_response = Convolve_2D(s_Image, threadIdx.y + 40, threadIdx.x + 40);
			d_Filter_Response[idx] = filter_response;
		}
	}
}


#endif

