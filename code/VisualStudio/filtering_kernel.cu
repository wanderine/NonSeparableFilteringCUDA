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

#ifndef FILTERING_KERNEL_CU_
#define FILTERING_KERNEL_CU_

#include "filtering.h"
#include "help_functions.cu"

__device__ __constant__ float c_Filter_2D[17][17];
__device__ __constant__ float c_Filter_3x3[3][3];
__device__ __constant__ float c_Filter_5x5[5][5];
__device__ __constant__ float c_Filter_7x7[7][7];
__device__ __constant__ float c_Filter_9x9[9][9];
__device__ __constant__ float c_Filter_11x11[11][11];
__device__ __constant__ float c_Filter_13x13[13][13];
__device__ __constant__ float c_Filter_15x15[15][15];
__device__ __constant__ float c_Filter_17x17[17][17];


__device__ __constant__ float c_Filter_3D[17][17][17];

__device__ __constant__ float c_Filter_3x3x3[3][3][3];
__device__ __constant__ float c_Filter_5x5x5[5][5][5];
//__device__ __constant__ float c_Filter_7x7x7[7][7][7];
//__device__ __constant__ float c_Filter_9x9x9[9][9][9];
//__device__ __constant__ float c_Filter_11x11x11[11][11][11];
//__device__ __constant__ float c_Filter_13x13x13[13][13][13];
//__device__ __constant__ float c_Filter_15x15x15[15][15][15];
//__device__ __constant__ float c_Filter_17x17x17[17][17][17];

__device__ float Conv_2D(float Image[64][96], int y, int x, int FILTER_W, int FILTER_H)
{
   float sum = 0.0f;

   int y_off = -(FILTER_H - 1)/2;
   for (int f_y = FILTER_H - 1; f_y >= 0; f_y--)
   {
      int x_off = -(FILTER_W - 1)/2;			
      for (int f_x = FILTER_W - 1; f_x >= 0; f_x--)
      {
         sum += Image[y + y_off][x + x_off] * c_Filter_2D[f_y][f_x];
         x_off++;
      }
      y_off++;
   }
	
   return sum;
}


__device__ float Conv_2D_Unrolled_3x3(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

	pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_3x3[2][2];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_3x3[1][2];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_3x3[0][2];

    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_3x3[2][1];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_3x3[1][1];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_3x3[0][1];
    
	pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_3x3[2][0];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_3x3[1][0];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_3x3[0][0];

	return sum;
}

__device__ float Conv_2D_Unrolled_5x5(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_5x5[4][4];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_5x5[3][4];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_5x5[2][4];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_5x5[1][4];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_5x5[0][4];

    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_5x5[4][3];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_5x5[3][3];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_5x5[2][3];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_5x5[1][3];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_5x5[0][3];

    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_5x5[4][2];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_5x5[3][2];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_5x5[2][2];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_5x5[1][2];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_5x5[0][2];

    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_5x5[4][1];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_5x5[3][1];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_5x5[2][1];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_5x5[1][1];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_5x5[0][1];

    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_5x5[4][0];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_5x5[3][0];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_5x5[2][0];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_5x5[1][0];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_5x5[0][0];

	return sum;
}


__device__ float Conv_2D_Unrolled_7x7(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;
	
    pixel = image[y - 3][x - 3]; 
    sum += pixel * c_Filter_7x7[6][6];
    pixel = image[y - 2][x - 3]; 
    sum += pixel * c_Filter_7x7[5][6];
	pixel = image[y - 1][x - 3]; 
    sum += pixel * c_Filter_7x7[4][6];
	pixel = image[y + 0][x - 3]; 
    sum += pixel * c_Filter_7x7[3][6];
	pixel = image[y + 1][x - 3]; 
    sum += pixel * c_Filter_7x7[2][6];
	pixel = image[y + 2][x - 3]; 
    sum += pixel * c_Filter_7x7[1][6];
	pixel = image[y + 3][x - 3]; 
	sum += pixel * c_Filter_7x7[0][6];

    pixel = image[y - 3][x - 2]; 
    sum += pixel * c_Filter_7x7[6][5];
    pixel = image[y - 2][x - 2]; 
    sum += pixel * c_Filter_7x7[5][5];
    pixel = image[y - 1][x - 2]; 
    sum += pixel * c_Filter_7x7[4][5];
    pixel = image[y + 0][x - 2]; 
    sum += pixel * c_Filter_7x7[3][5];
    pixel = image[y + 1][x - 2]; 
    sum += pixel * c_Filter_7x7[2][5];
    pixel = image[y + 2][x - 2]; 
    sum += pixel * c_Filter_7x7[1][5];
    pixel = image[y + 3][x - 2]; 
    sum += pixel * c_Filter_7x7[0][5];


    pixel = image[y - 3][x - 1]; 
    sum += pixel * c_Filter_7x7[6][4];
    pixel = image[y - 2][x - 1]; 
    sum += pixel * c_Filter_7x7[5][4];
    pixel = image[y - 1][x - 1]; 
    sum += pixel * c_Filter_7x7[4][4];
    pixel = image[y + 0][x - 1]; 
    sum += pixel * c_Filter_7x7[3][4];
    pixel = image[y + 1][x - 1]; 
    sum += pixel * c_Filter_7x7[2][4];
    pixel = image[y + 2][x - 1]; 
    sum += pixel * c_Filter_7x7[1][4];
    pixel = image[y + 3][x - 1]; 
    sum += pixel * c_Filter_7x7[0][4];


    pixel = image[y - 3][x + 0]; 
    sum += pixel * c_Filter_7x7[6][3];
    pixel = image[y - 2][x + 0]; 
    sum += pixel * c_Filter_7x7[5][3];
    pixel = image[y - 1][x + 0]; 
    sum += pixel * c_Filter_7x7[4][3];
    pixel = image[y + 0][x + 0]; 
    sum += pixel * c_Filter_7x7[3][3];
    pixel = image[y + 1][x + 0]; 
    sum += pixel * c_Filter_7x7[2][3];
    pixel = image[y + 2][x + 0]; 
    sum += pixel * c_Filter_7x7[1][3];
    pixel = image[y + 3][x + 0]; 
    sum += pixel * c_Filter_7x7[0][3];
    pixel = image[y - 3][x + 1]; 
    sum += pixel * c_Filter_7x7[6][2];
    pixel = image[y - 2][x + 1]; 
    sum += pixel * c_Filter_7x7[5][2];
    pixel = image[y - 1][x + 1]; 
    sum += pixel * c_Filter_7x7[4][2];
    pixel = image[y + 0][x + 1]; 
    sum += pixel * c_Filter_7x7[3][2];
    pixel = image[y + 1][x + 1]; 
    sum += pixel * c_Filter_7x7[2][2]; 
    pixel = image[y + 2][x + 1]; 
    sum += pixel * c_Filter_7x7[1][2];
    pixel = image[y + 3][x + 1]; 
    sum += pixel * c_Filter_7x7[0][2];
 
    pixel = image[y - 3][x + 2]; 
    sum += pixel * c_Filter_7x7[6][1];
    pixel = image[y - 2][x + 2]; 
    sum += pixel * c_Filter_7x7[5][1];
    pixel = image[y - 1][x + 2]; 
    sum += pixel * c_Filter_7x7[4][1]; 
    pixel = image[y + 0][x + 2]; 
    sum += pixel * c_Filter_7x7[3][1];
	pixel = image[y + 1][x + 2]; 
    sum += pixel * c_Filter_7x7[2][1];    
    pixel = image[y + 2][x + 2]; 
    sum += pixel * c_Filter_7x7[1][1];
    pixel = image[y + 3][x + 2]; 
    sum += pixel * c_Filter_7x7[0][1];

    pixel = image[y - 3][x + 3]; 
    sum += pixel * c_Filter_7x7[6][0];
    pixel = image[y - 2][x + 3]; 
    sum += pixel * c_Filter_7x7[5][0];
    pixel = image[y - 1][x + 3]; 
    sum += pixel * c_Filter_7x7[4][0];
    pixel = image[y + 0][x + 3]; 
    sum += pixel * c_Filter_7x7[3][0];
    pixel = image[y + 1][x + 3]; 
    sum += pixel * c_Filter_7x7[2][0];
    pixel = image[y + 2][x + 3]; 
    sum += pixel * c_Filter_7x7[1][0]; 
    pixel = image[y + 3][x + 3]; 
    sum += pixel * c_Filter_7x7[0][0];

	return sum;
}

__device__ float Conv_2D_Unrolled_9x9(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 4][x - 4];
    sum += pixel * c_Filter_9x9[8][8];
    pixel = image[y - 3][x - 4];
    sum += pixel * c_Filter_9x9[7][8];
    pixel = image[y - 2][x - 4];
    sum += pixel * c_Filter_9x9[6][8];
    pixel = image[y - 1][x - 4];
    sum += pixel * c_Filter_9x9[5][8];
    pixel = image[y + 0][x - 4];
    sum += pixel * c_Filter_9x9[4][8];
    pixel = image[y + 1][x - 4];
    sum += pixel * c_Filter_9x9[3][8];
    pixel = image[y + 2][x - 4];
    sum += pixel * c_Filter_9x9[2][8];
    pixel = image[y + 3][x - 4];
    sum += pixel * c_Filter_9x9[1][8];
    pixel = image[y + 4][x - 4];
    sum += pixel * c_Filter_9x9[0][8];

    pixel = image[y - 4][x - 3];
    sum += pixel * c_Filter_9x9[8][7];
    pixel = image[y - 3][x - 3];
    sum += pixel * c_Filter_9x9[7][7];
    pixel = image[y - 2][x - 3];
    sum += pixel * c_Filter_9x9[6][7];
    pixel = image[y - 1][x - 3];
    sum += pixel * c_Filter_9x9[5][7];
    pixel = image[y + 0][x - 3];
    sum += pixel * c_Filter_9x9[4][7];
    pixel = image[y + 1][x - 3];
    sum += pixel * c_Filter_9x9[3][7];
    pixel = image[y + 2][x - 3];
    sum += pixel * c_Filter_9x9[2][7];
    pixel = image[y + 3][x - 3];
    sum += pixel * c_Filter_9x9[1][7];
    pixel = image[y + 4][x - 3];
    sum += pixel * c_Filter_9x9[0][7];

    pixel = image[y - 4][x - 2];
    sum += pixel * c_Filter_9x9[8][6];
    pixel = image[y - 3][x - 2];
    sum += pixel * c_Filter_9x9[7][6];
    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_9x9[6][6];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_9x9[5][6];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_9x9[4][6];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_9x9[3][6];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_9x9[2][6];
    pixel = image[y + 3][x - 2];
    sum += pixel * c_Filter_9x9[1][6];
    pixel = image[y + 4][x - 2];
    sum += pixel * c_Filter_9x9[0][6];

    pixel = image[y - 4][x - 1];
    sum += pixel * c_Filter_9x9[8][5];
    pixel = image[y - 3][x - 1];
    sum += pixel * c_Filter_9x9[7][5];
    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_9x9[6][5];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_9x9[5][5];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_9x9[4][5];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_9x9[3][5];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_9x9[2][5];
    pixel = image[y + 3][x - 1];
    sum += pixel * c_Filter_9x9[1][5];
    pixel = image[y + 4][x - 1];
    sum += pixel * c_Filter_9x9[0][5];

    pixel = image[y - 4][x + 0];
    sum += pixel * c_Filter_9x9[8][4];
    pixel = image[y - 3][x + 0];
    sum += pixel * c_Filter_9x9[7][4];
    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_9x9[6][4];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_9x9[5][4];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_9x9[4][4];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_9x9[3][4];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_9x9[2][4];
    pixel = image[y + 3][x + 0];
    sum += pixel * c_Filter_9x9[1][4];
    pixel = image[y + 4][x + 0];
    sum += pixel * c_Filter_9x9[0][4];

    pixel = image[y - 4][x + 1];
    sum += pixel * c_Filter_9x9[8][3];
    pixel = image[y - 3][x + 1];
    sum += pixel * c_Filter_9x9[7][3];
    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_9x9[6][3];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_9x9[5][3];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_9x9[4][3];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_9x9[3][3];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_9x9[2][3];
    pixel = image[y + 3][x + 1];
    sum += pixel * c_Filter_9x9[1][3];
    pixel = image[y + 4][x + 1];
    sum += pixel * c_Filter_9x9[0][3];

	pixel = image[y - 4][x + 2];
    sum += pixel * c_Filter_9x9[8][2];
    pixel = image[y - 3][x + 2];
    sum += pixel * c_Filter_9x9[7][2];
    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_9x9[6][2];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_9x9[5][2];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_9x9[4][2];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_9x9[3][2];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_9x9[2][2];
    pixel = image[y + 3][x + 2];
    sum += pixel * c_Filter_9x9[1][2];
    pixel = image[y + 4][x + 2];
    sum += pixel * c_Filter_9x9[0][2];

	pixel = image[y - 4][x + 3];
    sum += pixel * c_Filter_9x9[8][1];
    pixel = image[y - 3][x + 3];
    sum += pixel * c_Filter_9x9[7][1];
    pixel = image[y - 2][x + 3];
    sum += pixel * c_Filter_9x9[6][1];
    pixel = image[y - 1][x + 3];
    sum += pixel * c_Filter_9x9[5][1];
    pixel = image[y + 0][x + 3];
    sum += pixel * c_Filter_9x9[4][1];
    pixel = image[y + 1][x + 3];
    sum += pixel * c_Filter_9x9[3][1];
    pixel = image[y + 2][x + 3];
    sum += pixel * c_Filter_9x9[2][1];
    pixel = image[y + 3][x + 3];
    sum += pixel * c_Filter_9x9[1][1];
    pixel = image[y + 4][x + 3];
    sum += pixel * c_Filter_9x9[0][1];

    pixel = image[y - 4][x + 4];
    sum += pixel * c_Filter_9x9[8][0];
    pixel = image[y - 3][x + 4];
    sum += pixel * c_Filter_9x9[7][0];
    pixel = image[y - 2][x + 4];
    sum += pixel * c_Filter_9x9[6][0];
    pixel = image[y - 1][x + 4];
    sum += pixel * c_Filter_9x9[5][0];
    pixel = image[y + 0][x + 4];
    sum += pixel * c_Filter_9x9[4][0];
    pixel = image[y + 1][x + 4];
    sum += pixel * c_Filter_9x9[3][0];
    pixel = image[y + 2][x + 4];
    sum += pixel * c_Filter_9x9[2][0];
    pixel = image[y + 3][x + 4];
    sum += pixel * c_Filter_9x9[1][0];
    pixel = image[y + 4][x + 4];
    sum += pixel * c_Filter_9x9[0][0];

	return sum;
}

__device__ float Conv_2D_Unrolled_11x11(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 5][x - 5];
    sum += pixel * c_Filter_11x11[10][10];
    pixel = image[y - 4][x - 5];
    sum += pixel * c_Filter_11x11[9][10];
    pixel = image[y - 3][x - 5];
    sum += pixel * c_Filter_11x11[8][10];
    pixel = image[y - 2][x - 5];
    sum += pixel * c_Filter_11x11[7][10];
    pixel = image[y - 1][x - 5];
    sum += pixel * c_Filter_11x11[6][10];
    pixel = image[y + 0][x - 5];
    sum += pixel * c_Filter_11x11[5][10];
    pixel = image[y + 1][x - 5];
    sum += pixel * c_Filter_11x11[4][10];
    pixel = image[y + 2][x - 5];
    sum += pixel * c_Filter_11x11[3][10];
    pixel = image[y + 3][x - 5];
    sum += pixel * c_Filter_11x11[2][10];
    pixel = image[y + 4][x - 5];
    sum += pixel * c_Filter_11x11[1][10];
    pixel = image[y + 5][x - 5];
    sum += pixel * c_Filter_11x11[0][10];

    pixel = image[y - 5][x - 4];
    sum += pixel * c_Filter_11x11[10][9];
    pixel = image[y - 4][x - 4];
    sum += pixel * c_Filter_11x11[9][9];
    pixel = image[y - 3][x - 4];
    sum += pixel * c_Filter_11x11[8][9];
    pixel = image[y - 2][x - 4];
    sum += pixel * c_Filter_11x11[7][9];
    pixel = image[y - 1][x - 4];
    sum += pixel * c_Filter_11x11[6][9];
    pixel = image[y + 0][x - 4];
    sum += pixel * c_Filter_11x11[5][9];
    pixel = image[y + 1][x - 4];
    sum += pixel * c_Filter_11x11[4][9];
    pixel = image[y + 2][x - 4];
    sum += pixel * c_Filter_11x11[3][9];
    pixel = image[y + 3][x - 4];
    sum += pixel * c_Filter_11x11[2][9];
    pixel = image[y + 4][x - 4];
    sum += pixel * c_Filter_11x11[1][9];
    pixel = image[y + 5][x - 4];
    sum += pixel * c_Filter_11x11[0][9];

    pixel = image[y - 5][x - 3];
    sum += pixel * c_Filter_11x11[10][8];
    pixel = image[y - 4][x - 3];
    sum += pixel * c_Filter_11x11[9][8];
    pixel = image[y - 3][x - 3];
    sum += pixel * c_Filter_11x11[8][8];
    pixel = image[y - 2][x - 3];
    sum += pixel * c_Filter_11x11[7][8];
    pixel = image[y - 1][x - 3];
    sum += pixel * c_Filter_11x11[6][8];
    pixel = image[y + 0][x - 3];
    sum += pixel * c_Filter_11x11[5][8];
    pixel = image[y + 1][x - 3];
    sum += pixel * c_Filter_11x11[4][8];
    pixel = image[y + 2][x - 3];
    sum += pixel * c_Filter_11x11[3][8];
    pixel = image[y + 3][x - 3];
    sum += pixel * c_Filter_11x11[2][8];
    pixel = image[y + 4][x - 3];
    sum += pixel * c_Filter_11x11[1][8];
    pixel = image[y + 5][x - 3];
    sum += pixel * c_Filter_11x11[0][8];

    pixel = image[y - 5][x - 2];
    sum += pixel * c_Filter_11x11[10][7];
    pixel = image[y - 4][x - 2];
    sum += pixel * c_Filter_11x11[9][7];
    pixel = image[y - 3][x - 2];
    sum += pixel * c_Filter_11x11[8][7];
    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_11x11[7][7];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_11x11[6][7];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_11x11[5][7];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_11x11[4][7];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_11x11[3][7];
    pixel = image[y + 3][x - 2];
    sum += pixel * c_Filter_11x11[2][7];
    pixel = image[y + 4][x - 2];
    sum += pixel * c_Filter_11x11[1][7];
    pixel = image[y + 5][x - 2];
    sum += pixel * c_Filter_11x11[0][7];

    pixel = image[y - 5][x - 1];
    sum += pixel * c_Filter_11x11[10][6];
    pixel = image[y - 4][x - 1];
    sum += pixel * c_Filter_11x11[9][6];
    pixel = image[y - 3][x - 1];
    sum += pixel * c_Filter_11x11[8][6];
    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_11x11[7][6];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_11x11[6][6];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_11x11[5][6];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_11x11[4][6];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_11x11[3][6];
    pixel = image[y + 3][x - 1];
    sum += pixel * c_Filter_11x11[2][6];
    pixel = image[y + 4][x - 1];
    sum += pixel * c_Filter_11x11[1][6];
    pixel = image[y + 5][x - 1];
    sum += pixel * c_Filter_11x11[0][6];

    pixel = image[y - 5][x + 0];
    sum += pixel * c_Filter_11x11[10][5];
    pixel = image[y - 4][x + 0];
    sum += pixel * c_Filter_11x11[9][5];
    pixel = image[y - 3][x + 0];
    sum += pixel * c_Filter_11x11[8][5];
    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_11x11[7][5];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_11x11[6][5];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_11x11[5][5];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_11x11[4][5];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_11x11[3][5];
    pixel = image[y + 3][x + 0];
    sum += pixel * c_Filter_11x11[2][5];
    pixel = image[y + 4][x + 0];
    sum += pixel * c_Filter_11x11[1][5];
    pixel = image[y + 5][x + 0];
    sum += pixel * c_Filter_11x11[0][5];

    pixel = image[y - 5][x + 1];
    sum += pixel * c_Filter_11x11[10][4];
    pixel = image[y - 4][x + 1];
    sum += pixel * c_Filter_11x11[9][4];
    pixel = image[y - 3][x + 1];
    sum += pixel * c_Filter_11x11[8][4];
    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_11x11[7][4];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_11x11[6][4];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_11x11[5][4];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_11x11[4][4];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_11x11[3][4];
    pixel = image[y + 3][x + 1];
    sum += pixel * c_Filter_11x11[2][4];
    pixel = image[y + 4][x + 1];
    sum += pixel * c_Filter_11x11[1][4];
    pixel = image[y + 5][x + 1];
    sum += pixel * c_Filter_11x11[0][4];

    pixel = image[y - 5][x + 2];
    sum += pixel * c_Filter_11x11[10][3];
    pixel = image[y - 4][x + 2];
    sum += pixel * c_Filter_11x11[9][3];
    pixel = image[y - 3][x + 2];
    sum += pixel * c_Filter_11x11[8][3];
    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_11x11[7][3];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_11x11[6][3];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_11x11[5][3];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_11x11[4][3];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_11x11[3][3];
    pixel = image[y + 3][x + 2];
    sum += pixel * c_Filter_11x11[2][3];
    pixel = image[y + 4][x + 2];
    sum += pixel * c_Filter_11x11[1][3];
    pixel = image[y + 5][x + 2];
    sum += pixel * c_Filter_11x11[0][3];

    pixel = image[y - 5][x + 3];
    sum += pixel * c_Filter_11x11[10][2];
    pixel = image[y - 4][x + 3];
    sum += pixel * c_Filter_11x11[9][2];
    pixel = image[y - 3][x + 3];
    sum += pixel * c_Filter_11x11[8][2];
    pixel = image[y - 2][x + 3];
    sum += pixel * c_Filter_11x11[7][2];
    pixel = image[y - 1][x + 3];
    sum += pixel * c_Filter_11x11[6][2];
    pixel = image[y + 0][x + 3];
    sum += pixel * c_Filter_11x11[5][2];
    pixel = image[y + 1][x + 3];
    sum += pixel * c_Filter_11x11[4][2];
    pixel = image[y + 2][x + 3];
    sum += pixel * c_Filter_11x11[3][2];
    pixel = image[y + 3][x + 3];
    sum += pixel * c_Filter_11x11[2][2];
    pixel = image[y + 4][x + 3];
    sum += pixel * c_Filter_11x11[1][2];
    pixel = image[y + 5][x + 3];
    sum += pixel * c_Filter_11x11[0][2];

	pixel = image[y - 5][x + 4];
    sum += pixel * c_Filter_11x11[10][1];
    pixel = image[y - 4][x + 4];
    sum += pixel * c_Filter_11x11[9][1];
    pixel = image[y - 3][x + 4];
    sum += pixel * c_Filter_11x11[8][1];
    pixel = image[y - 2][x + 4];
    sum += pixel * c_Filter_11x11[7][1];
    pixel = image[y - 1][x + 4];
    sum += pixel * c_Filter_11x11[6][1];
    pixel = image[y + 0][x + 4];
    sum += pixel * c_Filter_11x11[5][1];
    pixel = image[y + 1][x + 4];
    sum += pixel * c_Filter_11x11[4][1];
    pixel = image[y + 2][x + 4];
    sum += pixel * c_Filter_11x11[3][1];
    pixel = image[y + 3][x + 4];
    sum += pixel * c_Filter_11x11[2][1];
    pixel = image[y + 4][x + 4];
    sum += pixel * c_Filter_11x11[1][1];
    pixel = image[y + 5][x + 4];
    sum += pixel * c_Filter_11x11[0][1];

    pixel = image[y - 5][x + 5];
    sum += pixel * c_Filter_11x11[10][0];
    pixel = image[y - 4][x + 5];
    sum += pixel * c_Filter_11x11[9][0];
    pixel = image[y - 3][x + 5];
    sum += pixel * c_Filter_11x11[8][0];
    pixel = image[y - 2][x + 5];
    sum += pixel * c_Filter_11x11[7][0];
    pixel = image[y - 1][x + 5];
    sum += pixel * c_Filter_11x11[6][0];
    pixel = image[y + 0][x + 5];
    sum += pixel * c_Filter_11x11[5][0];
    pixel = image[y + 1][x + 5];
    sum += pixel * c_Filter_11x11[4][0];
    pixel = image[y + 2][x + 5];
    sum += pixel * c_Filter_11x11[3][0];
    pixel = image[y + 3][x + 5];
    sum += pixel * c_Filter_11x11[2][0];
    pixel = image[y + 4][x + 5];
    sum += pixel * c_Filter_11x11[1][0];
    pixel = image[y + 5][x + 5];
    sum += pixel * c_Filter_11x11[0][0];

	return sum;
}

__device__ float Conv_2D_Unrolled_13x13(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 6][x - 6];
    sum += pixel * c_Filter_13x13[12][12];
    pixel = image[y - 5][x - 6];
    sum += pixel * c_Filter_13x13[11][12];
    pixel = image[y - 4][x - 6];
    sum += pixel * c_Filter_13x13[10][12];
    pixel = image[y - 3][x - 6];
    sum += pixel * c_Filter_13x13[9][12];
    pixel = image[y - 2][x - 6];
    sum += pixel * c_Filter_13x13[8][12];
    pixel = image[y - 1][x - 6];
    sum += pixel * c_Filter_13x13[7][12];
    pixel = image[y + 0][x - 6];
    sum += pixel * c_Filter_13x13[6][12];
    pixel = image[y + 1][x - 6];
    sum += pixel * c_Filter_13x13[5][12];
    pixel = image[y + 2][x - 6];
    sum += pixel * c_Filter_13x13[4][12];
    pixel = image[y + 3][x - 6];
    sum += pixel * c_Filter_13x13[3][12];
    pixel = image[y + 4][x - 6];
    sum += pixel * c_Filter_13x13[2][12];
    pixel = image[y + 5][x - 6];
    sum += pixel * c_Filter_13x13[1][12];
    pixel = image[y + 6][x - 6];
    sum += pixel * c_Filter_13x13[0][12];

    pixel = image[y - 6][x - 5];
    sum += pixel * c_Filter_13x13[12][11];
    pixel = image[y - 5][x - 5];
    sum += pixel * c_Filter_13x13[11][11];
    pixel = image[y - 4][x - 5];
    sum += pixel * c_Filter_13x13[10][11];
    pixel = image[y - 3][x - 5];
    sum += pixel * c_Filter_13x13[9][11];
    pixel = image[y - 2][x - 5];
    sum += pixel * c_Filter_13x13[8][11];
    pixel = image[y - 1][x - 5];
    sum += pixel * c_Filter_13x13[7][11];
    pixel = image[y + 0][x - 5];
    sum += pixel * c_Filter_13x13[6][11];
    pixel = image[y + 1][x - 5];
    sum += pixel * c_Filter_13x13[5][11];
    pixel = image[y + 2][x - 5];
    sum += pixel * c_Filter_13x13[4][11];
    pixel = image[y + 3][x - 5];
    sum += pixel * c_Filter_13x13[3][11];
    pixel = image[y + 4][x - 5];
    sum += pixel * c_Filter_13x13[2][11];
    pixel = image[y + 5][x - 5];
    sum += pixel * c_Filter_13x13[1][11];
    pixel = image[y + 6][x - 5];
    sum += pixel * c_Filter_13x13[0][11];

    pixel = image[y - 6][x - 4];
    sum += pixel * c_Filter_13x13[12][10];
    pixel = image[y - 5][x - 4];
    sum += pixel * c_Filter_13x13[11][10];
    pixel = image[y - 4][x - 4];
    sum += pixel * c_Filter_13x13[10][10];
    pixel = image[y - 3][x - 4];
    sum += pixel * c_Filter_13x13[9][10];
    pixel = image[y - 2][x - 4];
    sum += pixel * c_Filter_13x13[8][10];
    pixel = image[y - 1][x - 4];
    sum += pixel * c_Filter_13x13[7][10];
    pixel = image[y + 0][x - 4];
    sum += pixel * c_Filter_13x13[6][10];
    pixel = image[y + 1][x - 4];
    sum += pixel * c_Filter_13x13[5][10];
    pixel = image[y + 2][x - 4];
    sum += pixel * c_Filter_13x13[4][10];
    pixel = image[y + 3][x - 4];
    sum += pixel * c_Filter_13x13[3][10];
    pixel = image[y + 4][x - 4];
    sum += pixel * c_Filter_13x13[2][10];
    pixel = image[y + 5][x - 4];
    sum += pixel * c_Filter_13x13[1][10];
    pixel = image[y + 6][x - 4];
    sum += pixel * c_Filter_13x13[0][10];

    pixel = image[y - 6][x - 3];
    sum += pixel * c_Filter_13x13[12][9];
    pixel = image[y - 5][x - 3];
    sum += pixel * c_Filter_13x13[11][9];
    pixel = image[y - 4][x - 3];
    sum += pixel * c_Filter_13x13[10][9];
    pixel = image[y - 3][x - 3];
    sum += pixel * c_Filter_13x13[9][9];
    pixel = image[y - 2][x - 3];
    sum += pixel * c_Filter_13x13[8][9];
    pixel = image[y - 1][x - 3];
    sum += pixel * c_Filter_13x13[7][9];
    pixel = image[y + 0][x - 3];
    sum += pixel * c_Filter_13x13[6][9];
    pixel = image[y + 1][x - 3];
    sum += pixel * c_Filter_13x13[5][9];
    pixel = image[y + 2][x - 3];
    sum += pixel * c_Filter_13x13[4][9];
    pixel = image[y + 3][x - 3];
    sum += pixel * c_Filter_13x13[3][9];
    pixel = image[y + 4][x - 3];
    sum += pixel * c_Filter_13x13[2][9];
    pixel = image[y + 5][x - 3];
    sum += pixel * c_Filter_13x13[1][9];
    pixel = image[y + 6][x - 3];
    sum += pixel * c_Filter_13x13[0][9];

    pixel = image[y - 6][x - 2];
    sum += pixel * c_Filter_13x13[12][8];
    pixel = image[y - 5][x - 2];
    sum += pixel * c_Filter_13x13[11][8];
    pixel = image[y - 4][x - 2];
    sum += pixel * c_Filter_13x13[10][8];
    pixel = image[y - 3][x - 2];
    sum += pixel * c_Filter_13x13[9][8];
    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_13x13[8][8];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_13x13[7][8];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_13x13[6][8];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_13x13[5][8];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_13x13[4][8];
    pixel = image[y + 3][x - 2];
    sum += pixel * c_Filter_13x13[3][8];
    pixel = image[y + 4][x - 2];
    sum += pixel * c_Filter_13x13[2][8];
    pixel = image[y + 5][x - 2];
    sum += pixel * c_Filter_13x13[1][8];
    pixel = image[y + 6][x - 2];
    sum += pixel * c_Filter_13x13[0][8];

    pixel = image[y - 6][x - 1];
    sum += pixel * c_Filter_13x13[12][7];
    pixel = image[y - 5][x - 1];
    sum += pixel * c_Filter_13x13[11][7];
    pixel = image[y - 4][x - 1];
    sum += pixel * c_Filter_13x13[10][7];
    pixel = image[y - 3][x - 1];
    sum += pixel * c_Filter_13x13[9][7];
    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_13x13[8][7];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_13x13[7][7];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_13x13[6][7];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_13x13[5][7];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_13x13[4][7];
    pixel = image[y + 3][x - 1];
    sum += pixel * c_Filter_13x13[3][7];
    pixel = image[y + 4][x - 1];
    sum += pixel * c_Filter_13x13[2][7];
    pixel = image[y + 5][x - 1];
    sum += pixel * c_Filter_13x13[1][7];
    pixel = image[y + 6][x - 1];
    sum += pixel * c_Filter_13x13[0][7];

    pixel = image[y - 6][x + 0];
    sum += pixel * c_Filter_13x13[12][6];
    pixel = image[y - 5][x + 0];
    sum += pixel * c_Filter_13x13[11][6];
    pixel = image[y - 4][x + 0];
    sum += pixel * c_Filter_13x13[10][6];
    pixel = image[y - 3][x + 0];
    sum += pixel * c_Filter_13x13[9][6];
    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_13x13[8][6];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_13x13[7][6];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_13x13[6][6];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_13x13[5][6];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_13x13[4][6];
    pixel = image[y + 3][x + 0];
    sum += pixel * c_Filter_13x13[3][6];
    pixel = image[y + 4][x + 0];
    sum += pixel * c_Filter_13x13[2][6];
    pixel = image[y + 5][x + 0];
    sum += pixel * c_Filter_13x13[1][6];
    pixel = image[y + 6][x + 0];
    sum += pixel * c_Filter_13x13[0][6];

    pixel = image[y - 6][x + 1];
    sum += pixel * c_Filter_13x13[12][5];
    pixel = image[y - 5][x + 1];
    sum += pixel * c_Filter_13x13[11][5];
    pixel = image[y - 4][x + 1];
    sum += pixel * c_Filter_13x13[10][5];
    pixel = image[y - 3][x + 1];
    sum += pixel * c_Filter_13x13[9][5];
    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_13x13[8][5];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_13x13[7][5];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_13x13[6][5];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_13x13[5][5];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_13x13[4][5];
    pixel = image[y + 3][x + 1];
    sum += pixel * c_Filter_13x13[3][5];
    pixel = image[y + 4][x + 1];
    sum += pixel * c_Filter_13x13[2][5];
    pixel = image[y + 5][x + 1];
    sum += pixel * c_Filter_13x13[1][5];
    pixel = image[y + 6][x + 1];
    sum += pixel * c_Filter_13x13[0][5];

    pixel = image[y - 6][x + 2];
    sum += pixel * c_Filter_13x13[12][4];
    pixel = image[y - 5][x + 2];
    sum += pixel * c_Filter_13x13[11][4];
    pixel = image[y - 4][x + 2];
    sum += pixel * c_Filter_13x13[10][4];
    pixel = image[y - 3][x + 2];
    sum += pixel * c_Filter_13x13[9][4];
    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_13x13[8][4];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_13x13[7][4];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_13x13[6][4];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_13x13[5][4];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_13x13[4][4];
    pixel = image[y + 3][x + 2];
    sum += pixel * c_Filter_13x13[3][4];
    pixel = image[y + 4][x + 2];
    sum += pixel * c_Filter_13x13[2][4];
    pixel = image[y + 5][x + 2];
    sum += pixel * c_Filter_13x13[1][4];
    pixel = image[y + 6][x + 2];
    sum += pixel * c_Filter_13x13[0][4];

    pixel = image[y - 6][x + 3];
    sum += pixel * c_Filter_13x13[12][3];
    pixel = image[y - 5][x + 3];
    sum += pixel * c_Filter_13x13[11][3];
    pixel = image[y - 4][x + 3];
    sum += pixel * c_Filter_13x13[10][3];
    pixel = image[y - 3][x + 3];
    sum += pixel * c_Filter_13x13[9][3];
    pixel = image[y - 2][x + 3];
    sum += pixel * c_Filter_13x13[8][3];
    pixel = image[y - 1][x + 3];
    sum += pixel * c_Filter_13x13[7][3];
    pixel = image[y + 0][x + 3];
    sum += pixel * c_Filter_13x13[6][3];
    pixel = image[y + 1][x + 3];
    sum += pixel * c_Filter_13x13[5][3];
    pixel = image[y + 2][x + 3];
    sum += pixel * c_Filter_13x13[4][3];
    pixel = image[y + 3][x + 3];
    sum += pixel * c_Filter_13x13[3][3];
    pixel = image[y + 4][x + 3];
    sum += pixel * c_Filter_13x13[2][3];
    pixel = image[y + 5][x + 3];
    sum += pixel * c_Filter_13x13[1][3];
    pixel = image[y + 6][x + 3];
    sum += pixel * c_Filter_13x13[0][3];

    pixel = image[y - 6][x + 4];
    sum += pixel * c_Filter_13x13[12][2];
    pixel = image[y - 5][x + 4];
    sum += pixel * c_Filter_13x13[11][2];
    pixel = image[y - 4][x + 4];
    sum += pixel * c_Filter_13x13[10][2];
    pixel = image[y - 3][x + 4];
    sum += pixel * c_Filter_13x13[9][2];
    pixel = image[y - 2][x + 4];
    sum += pixel * c_Filter_13x13[8][2];
    pixel = image[y - 1][x + 4];
    sum += pixel * c_Filter_13x13[7][2];
    pixel = image[y + 0][x + 4];
    sum += pixel * c_Filter_13x13[6][2];
    pixel = image[y + 1][x + 4];
    sum += pixel * c_Filter_13x13[5][2];
    pixel = image[y + 2][x + 4];
    sum += pixel * c_Filter_13x13[4][2];
    pixel = image[y + 3][x + 4];
    sum += pixel * c_Filter_13x13[3][2];
    pixel = image[y + 4][x + 4];
    sum += pixel * c_Filter_13x13[2][2];
    pixel = image[y + 5][x + 4];
    sum += pixel * c_Filter_13x13[1][2];
    pixel = image[y + 6][x + 4];
    sum += pixel * c_Filter_13x13[0][2];

    pixel = image[y - 6][x + 5];
    sum += pixel * c_Filter_13x13[12][1];
    pixel = image[y - 5][x + 5];
    sum += pixel * c_Filter_13x13[11][1];
    pixel = image[y - 4][x + 5];
    sum += pixel * c_Filter_13x13[10][1];
    pixel = image[y - 3][x + 5];
    sum += pixel * c_Filter_13x13[9][1];
    pixel = image[y - 2][x + 5];
    sum += pixel * c_Filter_13x13[8][1];
    pixel = image[y - 1][x + 5];
    sum += pixel * c_Filter_13x13[7][1];
    pixel = image[y + 0][x + 5];
    sum += pixel * c_Filter_13x13[6][1];
    pixel = image[y + 1][x + 5];
    sum += pixel * c_Filter_13x13[5][1];
    pixel = image[y + 2][x + 5];
    sum += pixel * c_Filter_13x13[4][1];
    pixel = image[y + 3][x + 5];
    sum += pixel * c_Filter_13x13[3][1];
    pixel = image[y + 4][x + 5];
    sum += pixel * c_Filter_13x13[2][1];
    pixel = image[y + 5][x + 5];
    sum += pixel * c_Filter_13x13[1][1];
    pixel = image[y + 6][x + 5];
    sum += pixel * c_Filter_13x13[0][1];

    pixel = image[y - 6][x + 6];
    sum += pixel * c_Filter_13x13[12][0];
    pixel = image[y - 5][x + 6];
    sum += pixel * c_Filter_13x13[11][0];
    pixel = image[y - 4][x + 6];
    sum += pixel * c_Filter_13x13[10][0];
    pixel = image[y - 3][x + 6];
    sum += pixel * c_Filter_13x13[9][0];
    pixel = image[y - 2][x + 6];
    sum += pixel * c_Filter_13x13[8][0];
    pixel = image[y - 1][x + 6];
    sum += pixel * c_Filter_13x13[7][0];
    pixel = image[y + 0][x + 6];
    sum += pixel * c_Filter_13x13[6][0];
    pixel = image[y + 1][x + 6];
    sum += pixel * c_Filter_13x13[5][0];
    pixel = image[y + 2][x + 6];
    sum += pixel * c_Filter_13x13[4][0];
    pixel = image[y + 3][x + 6];
    sum += pixel * c_Filter_13x13[3][0];
    pixel = image[y + 4][x + 6];
    sum += pixel * c_Filter_13x13[2][0];
    pixel = image[y + 5][x + 6];
    sum += pixel * c_Filter_13x13[1][0];
    pixel = image[y + 6][x + 6];
    sum += pixel * c_Filter_13x13[0][0];

	return sum;
}

__device__ float Conv_2D_Unrolled_15x15(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 7][x - 7];
    sum += pixel * c_Filter_15x15[14][14];
    pixel = image[y - 6][x - 7];
    sum += pixel * c_Filter_15x15[13][14];
    pixel = image[y - 5][x - 7];
    sum += pixel * c_Filter_15x15[12][14];
    pixel = image[y - 4][x - 7];
    sum += pixel * c_Filter_15x15[11][14];
    pixel = image[y - 3][x - 7];
    sum += pixel * c_Filter_15x15[10][14];
    pixel = image[y - 2][x - 7];
    sum += pixel * c_Filter_15x15[9][14];
    pixel = image[y - 1][x - 7];
    sum += pixel * c_Filter_15x15[8][14];
    pixel = image[y + 0][x - 7];
    sum += pixel * c_Filter_15x15[7][14];
    pixel = image[y + 1][x - 7];
    sum += pixel * c_Filter_15x15[6][14];
    pixel = image[y + 2][x - 7];
    sum += pixel * c_Filter_15x15[5][14];
    pixel = image[y + 3][x - 7];
    sum += pixel * c_Filter_15x15[4][14];
    pixel = image[y + 4][x - 7];
    sum += pixel * c_Filter_15x15[3][14];
    pixel = image[y + 5][x - 7];
    sum += pixel * c_Filter_15x15[2][14];
    pixel = image[y + 6][x - 7];
    sum += pixel * c_Filter_15x15[1][14];
    pixel = image[y + 7][x - 7];
    sum += pixel * c_Filter_15x15[0][14];

    pixel = image[y - 7][x - 6];
    sum += pixel * c_Filter_15x15[14][13];
    pixel = image[y - 6][x - 6];
    sum += pixel * c_Filter_15x15[13][13];
    pixel = image[y - 5][x - 6];
    sum += pixel * c_Filter_15x15[12][13];
    pixel = image[y - 4][x - 6];
    sum += pixel * c_Filter_15x15[11][13];
    pixel = image[y - 3][x - 6];
    sum += pixel * c_Filter_15x15[10][13];
    pixel = image[y - 2][x - 6];
    sum += pixel * c_Filter_15x15[9][13];
    pixel = image[y - 1][x - 6];
    sum += pixel * c_Filter_15x15[8][13];
    pixel = image[y + 0][x - 6];
    sum += pixel * c_Filter_15x15[7][13];
    pixel = image[y + 1][x - 6];
    sum += pixel * c_Filter_15x15[6][13];
    pixel = image[y + 2][x - 6];
    sum += pixel * c_Filter_15x15[5][13];
    pixel = image[y + 3][x - 6];
    sum += pixel * c_Filter_15x15[4][13];
    pixel = image[y + 4][x - 6];
    sum += pixel * c_Filter_15x15[3][13];
    pixel = image[y + 5][x - 6];
    sum += pixel * c_Filter_15x15[2][13];
    pixel = image[y + 6][x - 6];
    sum += pixel * c_Filter_15x15[1][13];
    pixel = image[y + 7][x - 6];
    sum += pixel * c_Filter_15x15[0][13];

    pixel = image[y - 7][x - 5];
    sum += pixel * c_Filter_15x15[14][12];
    pixel = image[y - 6][x - 5];
    sum += pixel * c_Filter_15x15[13][12];
    pixel = image[y - 5][x - 5];
    sum += pixel * c_Filter_15x15[12][12];
    pixel = image[y - 4][x - 5];
    sum += pixel * c_Filter_15x15[11][12];
    pixel = image[y - 3][x - 5];
    sum += pixel * c_Filter_15x15[10][12];
    pixel = image[y - 2][x - 5];
    sum += pixel * c_Filter_15x15[9][12];
    pixel = image[y - 1][x - 5];
    sum += pixel * c_Filter_15x15[8][12];
    pixel = image[y + 0][x - 5];
    sum += pixel * c_Filter_15x15[7][12];
    pixel = image[y + 1][x - 5];
    sum += pixel * c_Filter_15x15[6][12];
    pixel = image[y + 2][x - 5];
    sum += pixel * c_Filter_15x15[5][12];
    pixel = image[y + 3][x - 5];
    sum += pixel * c_Filter_15x15[4][12];
    pixel = image[y + 4][x - 5];
    sum += pixel * c_Filter_15x15[3][12];
    pixel = image[y + 5][x - 5];
    sum += pixel * c_Filter_15x15[2][12];
    pixel = image[y + 6][x - 5];
    sum += pixel * c_Filter_15x15[1][12];
    pixel = image[y + 7][x - 5];
    sum += pixel * c_Filter_15x15[0][12];

    pixel = image[y - 7][x - 4];
    sum += pixel * c_Filter_15x15[14][11];
    pixel = image[y - 6][x - 4];
    sum += pixel * c_Filter_15x15[13][11];
    pixel = image[y - 5][x - 4];
    sum += pixel * c_Filter_15x15[12][11];
    pixel = image[y - 4][x - 4];
    sum += pixel * c_Filter_15x15[11][11];
    pixel = image[y - 3][x - 4];
    sum += pixel * c_Filter_15x15[10][11];
    pixel = image[y - 2][x - 4];
    sum += pixel * c_Filter_15x15[9][11];
    pixel = image[y - 1][x - 4];
    sum += pixel * c_Filter_15x15[8][11];
    pixel = image[y + 0][x - 4];
    sum += pixel * c_Filter_15x15[7][11];
    pixel = image[y + 1][x - 4];
    sum += pixel * c_Filter_15x15[6][11];
    pixel = image[y + 2][x - 4];
    sum += pixel * c_Filter_15x15[5][11];
    pixel = image[y + 3][x - 4];
    sum += pixel * c_Filter_15x15[4][11];
    pixel = image[y + 4][x - 4];
    sum += pixel * c_Filter_15x15[3][11];
    pixel = image[y + 5][x - 4];
    sum += pixel * c_Filter_15x15[2][11];
    pixel = image[y + 6][x - 4];
    sum += pixel * c_Filter_15x15[1][11];
    pixel = image[y + 7][x - 4];
    sum += pixel * c_Filter_15x15[0][11];

    pixel = image[y - 7][x - 3];
    sum += pixel * c_Filter_15x15[14][10];
    pixel = image[y - 6][x - 3];
    sum += pixel * c_Filter_15x15[13][10];
    pixel = image[y - 5][x - 3];
    sum += pixel * c_Filter_15x15[12][10];
    pixel = image[y - 4][x - 3];
    sum += pixel * c_Filter_15x15[11][10];
    pixel = image[y - 3][x - 3];
    sum += pixel * c_Filter_15x15[10][10];
    pixel = image[y - 2][x - 3];
    sum += pixel * c_Filter_15x15[9][10];
    pixel = image[y - 1][x - 3];
    sum += pixel * c_Filter_15x15[8][10];
    pixel = image[y + 0][x - 3];
    sum += pixel * c_Filter_15x15[7][10];
    pixel = image[y + 1][x - 3];
    sum += pixel * c_Filter_15x15[6][10];
    pixel = image[y + 2][x - 3];
    sum += pixel * c_Filter_15x15[5][10];
    pixel = image[y + 3][x - 3];
    sum += pixel * c_Filter_15x15[4][10];
    pixel = image[y + 4][x - 3];
    sum += pixel * c_Filter_15x15[3][10];
    pixel = image[y + 5][x - 3];
    sum += pixel * c_Filter_15x15[2][10];
    pixel = image[y + 6][x - 3];
    sum += pixel * c_Filter_15x15[1][10];
    pixel = image[y + 7][x - 3];
    sum += pixel * c_Filter_15x15[0][10];

    pixel = image[y - 7][x - 2];
    sum += pixel * c_Filter_15x15[14][9];
    pixel = image[y - 6][x - 2];
    sum += pixel * c_Filter_15x15[13][9];
    pixel = image[y - 5][x - 2];
    sum += pixel * c_Filter_15x15[12][9];
    pixel = image[y - 4][x - 2];
    sum += pixel * c_Filter_15x15[11][9];
    pixel = image[y - 3][x - 2];
    sum += pixel * c_Filter_15x15[10][9];
    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_15x15[9][9];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_15x15[8][9];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_15x15[7][9];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_15x15[6][9];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_15x15[5][9];
    pixel = image[y + 3][x - 2];
    sum += pixel * c_Filter_15x15[4][9];
    pixel = image[y + 4][x - 2];
    sum += pixel * c_Filter_15x15[3][9];
    pixel = image[y + 5][x - 2];
    sum += pixel * c_Filter_15x15[2][9];
    pixel = image[y + 6][x - 2];
    sum += pixel * c_Filter_15x15[1][9];
    pixel = image[y + 7][x - 2];
    sum += pixel * c_Filter_15x15[0][9];

    pixel = image[y - 7][x - 1];
    sum += pixel * c_Filter_15x15[14][8];
    pixel = image[y - 6][x - 1];
    sum += pixel * c_Filter_15x15[13][8];
    pixel = image[y - 5][x - 1];
    sum += pixel * c_Filter_15x15[12][8];
    pixel = image[y - 4][x - 1];
    sum += pixel * c_Filter_15x15[11][8];
    pixel = image[y - 3][x - 1];
    sum += pixel * c_Filter_15x15[10][8];
    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_15x15[9][8];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_15x15[8][8];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_15x15[7][8];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_15x15[6][8];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_15x15[5][8];
    pixel = image[y + 3][x - 1];
    sum += pixel * c_Filter_15x15[4][8];
    pixel = image[y + 4][x - 1];
    sum += pixel * c_Filter_15x15[3][8];
    pixel = image[y + 5][x - 1];
    sum += pixel * c_Filter_15x15[2][8];
    pixel = image[y + 6][x - 1];
    sum += pixel * c_Filter_15x15[1][8];
    pixel = image[y + 7][x - 1];
    sum += pixel * c_Filter_15x15[0][8];

    pixel = image[y - 7][x + 0];
    sum += pixel * c_Filter_15x15[14][7];
    pixel = image[y - 6][x + 0];
    sum += pixel * c_Filter_15x15[13][7];
    pixel = image[y - 5][x + 0];
    sum += pixel * c_Filter_15x15[12][7];
    pixel = image[y - 4][x + 0];
    sum += pixel * c_Filter_15x15[11][7];
    pixel = image[y - 3][x + 0];
    sum += pixel * c_Filter_15x15[10][7];
    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_15x15[9][7];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_15x15[8][7];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_15x15[7][7];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_15x15[6][7];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_15x15[5][7];
    pixel = image[y + 3][x + 0];
    sum += pixel * c_Filter_15x15[4][7];
    pixel = image[y + 4][x + 0];
    sum += pixel * c_Filter_15x15[3][7];
    pixel = image[y + 5][x + 0];
    sum += pixel * c_Filter_15x15[2][7];
    pixel = image[y + 6][x + 0];
    sum += pixel * c_Filter_15x15[1][7];
    pixel = image[y + 7][x + 0];
    sum += pixel * c_Filter_15x15[0][7];

    pixel = image[y - 7][x + 1];
    sum += pixel * c_Filter_15x15[14][6];
    pixel = image[y - 6][x + 1];
    sum += pixel * c_Filter_15x15[13][6];
    pixel = image[y - 5][x + 1];
    sum += pixel * c_Filter_15x15[12][6];
    pixel = image[y - 4][x + 1];
    sum += pixel * c_Filter_15x15[11][6];
    pixel = image[y - 3][x + 1];
    sum += pixel * c_Filter_15x15[10][6];
    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_15x15[9][6];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_15x15[8][6];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_15x15[7][6];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_15x15[6][6];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_15x15[5][6];
    pixel = image[y + 3][x + 1];
    sum += pixel * c_Filter_15x15[4][6];
    pixel = image[y + 4][x + 1];
    sum += pixel * c_Filter_15x15[3][6];
    pixel = image[y + 5][x + 1];
    sum += pixel * c_Filter_15x15[2][6];
    pixel = image[y + 6][x + 1];
    sum += pixel * c_Filter_15x15[1][6];
    pixel = image[y + 7][x + 1];
    sum += pixel * c_Filter_15x15[0][6];

    pixel = image[y - 7][x + 2];
    sum += pixel * c_Filter_15x15[14][5];
    pixel = image[y - 6][x + 2];
    sum += pixel * c_Filter_15x15[13][5];
    pixel = image[y - 5][x + 2];
    sum += pixel * c_Filter_15x15[12][5];
    pixel = image[y - 4][x + 2];
    sum += pixel * c_Filter_15x15[11][5];
    pixel = image[y - 3][x + 2];
    sum += pixel * c_Filter_15x15[10][5];
    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_15x15[9][5];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_15x15[8][5];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_15x15[7][5];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_15x15[6][5];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_15x15[5][5];
    pixel = image[y + 3][x + 2];
    sum += pixel * c_Filter_15x15[4][5];
    pixel = image[y + 4][x + 2];
    sum += pixel * c_Filter_15x15[3][5];
    pixel = image[y + 5][x + 2];
    sum += pixel * c_Filter_15x15[2][5];
    pixel = image[y + 6][x + 2];
    sum += pixel * c_Filter_15x15[1][5];
    pixel = image[y + 7][x + 2];
    sum += pixel * c_Filter_15x15[0][5];

    pixel = image[y - 7][x + 3];
    sum += pixel * c_Filter_15x15[14][4];
    pixel = image[y - 6][x + 3];
    sum += pixel * c_Filter_15x15[13][4];
    pixel = image[y - 5][x + 3];
    sum += pixel * c_Filter_15x15[12][4];
    pixel = image[y - 4][x + 3];
    sum += pixel * c_Filter_15x15[11][4];
    pixel = image[y - 3][x + 3];
    sum += pixel * c_Filter_15x15[10][4];
    pixel = image[y - 2][x + 3];
    sum += pixel * c_Filter_15x15[9][4];
    pixel = image[y - 1][x + 3];
    sum += pixel * c_Filter_15x15[8][4];
    pixel = image[y + 0][x + 3];
    sum += pixel * c_Filter_15x15[7][4];
    pixel = image[y + 1][x + 3];
    sum += pixel * c_Filter_15x15[6][4];
    pixel = image[y + 2][x + 3];
    sum += pixel * c_Filter_15x15[5][4];
    pixel = image[y + 3][x + 3];
    sum += pixel * c_Filter_15x15[4][4];
    pixel = image[y + 4][x + 3];
    sum += pixel * c_Filter_15x15[3][4];
    pixel = image[y + 5][x + 3];
    sum += pixel * c_Filter_15x15[2][4];
    pixel = image[y + 6][x + 3];
    sum += pixel * c_Filter_15x15[1][4];
    pixel = image[y + 7][x + 3];
    sum += pixel * c_Filter_15x15[0][4];

    pixel = image[y - 7][x + 4];
    sum += pixel * c_Filter_15x15[14][3];
    pixel = image[y - 6][x + 4];
    sum += pixel * c_Filter_15x15[13][3];
    pixel = image[y - 5][x + 4];
    sum += pixel * c_Filter_15x15[12][3];
    pixel = image[y - 4][x + 4];
    sum += pixel * c_Filter_15x15[11][3];
    pixel = image[y - 3][x + 4];
    sum += pixel * c_Filter_15x15[10][3];
    pixel = image[y - 2][x + 4];
    sum += pixel * c_Filter_15x15[9][3];
    pixel = image[y - 1][x + 4];
    sum += pixel * c_Filter_15x15[8][3];
    pixel = image[y + 0][x + 4];
    sum += pixel * c_Filter_15x15[7][3];
    pixel = image[y + 1][x + 4];
    sum += pixel * c_Filter_15x15[6][3];
    pixel = image[y + 2][x + 4];
    sum += pixel * c_Filter_15x15[5][3];
    pixel = image[y + 3][x + 4];
    sum += pixel * c_Filter_15x15[4][3];
    pixel = image[y + 4][x + 4];
    sum += pixel * c_Filter_15x15[3][3];
    pixel = image[y + 5][x + 4];
    sum += pixel * c_Filter_15x15[2][3];
    pixel = image[y + 6][x + 4];
    sum += pixel * c_Filter_15x15[1][3];
    pixel = image[y + 7][x + 4];
    sum += pixel * c_Filter_15x15[0][3];

    pixel = image[y - 7][x + 5];
    sum += pixel * c_Filter_15x15[14][2];
    pixel = image[y - 6][x + 5];
    sum += pixel * c_Filter_15x15[13][2];
    pixel = image[y - 5][x + 5];
    sum += pixel * c_Filter_15x15[12][2];
    pixel = image[y - 4][x + 5];
    sum += pixel * c_Filter_15x15[11][2];
    pixel = image[y - 3][x + 5];
    sum += pixel * c_Filter_15x15[10][2];
    pixel = image[y - 2][x + 5];
    sum += pixel * c_Filter_15x15[9][2];
    pixel = image[y - 1][x + 5];
    sum += pixel * c_Filter_15x15[8][2];
    pixel = image[y + 0][x + 5];
    sum += pixel * c_Filter_15x15[7][2];
    pixel = image[y + 1][x + 5];
    sum += pixel * c_Filter_15x15[6][2];
    pixel = image[y + 2][x + 5];
    sum += pixel * c_Filter_15x15[5][2];
    pixel = image[y + 3][x + 5];
    sum += pixel * c_Filter_15x15[4][2];
    pixel = image[y + 4][x + 5];
    sum += pixel * c_Filter_15x15[3][2];
    pixel = image[y + 5][x + 5];
    sum += pixel * c_Filter_15x15[2][2];
    pixel = image[y + 6][x + 5];
    sum += pixel * c_Filter_15x15[1][2];
    pixel = image[y + 7][x + 5];
    sum += pixel * c_Filter_15x15[0][2];

    pixel = image[y - 7][x + 6];
    sum += pixel * c_Filter_15x15[14][1];
    pixel = image[y - 6][x + 6];
    sum += pixel * c_Filter_15x15[13][1];
    pixel = image[y - 5][x + 6];
    sum += pixel * c_Filter_15x15[12][1];
    pixel = image[y - 4][x + 6];
    sum += pixel * c_Filter_15x15[11][1];
    pixel = image[y - 3][x + 6];
    sum += pixel * c_Filter_15x15[10][1];
    pixel = image[y - 2][x + 6];
    sum += pixel * c_Filter_15x15[9][1];
    pixel = image[y - 1][x + 6];
    sum += pixel * c_Filter_15x15[8][1];
    pixel = image[y + 0][x + 6];
    sum += pixel * c_Filter_15x15[7][1];
    pixel = image[y + 1][x + 6];
    sum += pixel * c_Filter_15x15[6][1];
    pixel = image[y + 2][x + 6];
    sum += pixel * c_Filter_15x15[5][1];
    pixel = image[y + 3][x + 6];
    sum += pixel * c_Filter_15x15[4][1];
    pixel = image[y + 4][x + 6];
    sum += pixel * c_Filter_15x15[3][1];
    pixel = image[y + 5][x + 6];
    sum += pixel * c_Filter_15x15[2][1];
    pixel = image[y + 6][x + 6];
    sum += pixel * c_Filter_15x15[1][1];
    pixel = image[y + 7][x + 6];
    sum += pixel * c_Filter_15x15[0][1];

    pixel = image[y - 7][x + 7];
    sum += pixel * c_Filter_15x15[14][0];
    pixel = image[y - 6][x + 7];
    sum += pixel * c_Filter_15x15[13][0];
    pixel = image[y - 5][x + 7];
    sum += pixel * c_Filter_15x15[12][0];
    pixel = image[y - 4][x + 7];
    sum += pixel * c_Filter_15x15[11][0];
    pixel = image[y - 3][x + 7];
    sum += pixel * c_Filter_15x15[10][0];
    pixel = image[y - 2][x + 7];
    sum += pixel * c_Filter_15x15[9][0];
    pixel = image[y - 1][x + 7];
    sum += pixel * c_Filter_15x15[8][0];
    pixel = image[y + 0][x + 7];
    sum += pixel * c_Filter_15x15[7][0];
    pixel = image[y + 1][x + 7];
    sum += pixel * c_Filter_15x15[6][0];
    pixel = image[y + 2][x + 7];
    sum += pixel * c_Filter_15x15[5][0];
    pixel = image[y + 3][x + 7];
    sum += pixel * c_Filter_15x15[4][0];
    pixel = image[y + 4][x + 7];
    sum += pixel * c_Filter_15x15[3][0];
    pixel = image[y + 5][x + 7];
    sum += pixel * c_Filter_15x15[2][0];
    pixel = image[y + 6][x + 7];
    sum += pixel * c_Filter_15x15[1][0];
    pixel = image[y + 7][x + 7];
    sum += pixel * c_Filter_15x15[0][0];

	return sum;
}


__device__ float Conv_2D_Unrolled_17x17(float image[64][96], int y, int x)
{
	float pixel;
	float sum = 0.0f;

    pixel = image[y - 8][x - 8];
    sum += pixel * c_Filter_17x17[16][16];
    pixel = image[y - 7][x - 8];
    sum += pixel * c_Filter_17x17[15][16];
    pixel = image[y - 6][x - 8];
    sum += pixel * c_Filter_17x17[14][16];
    pixel = image[y - 5][x - 8];
    sum += pixel * c_Filter_17x17[13][16];
    pixel = image[y - 4][x - 8];
    sum += pixel * c_Filter_17x17[12][16];
    pixel = image[y - 3][x - 8];
    sum += pixel * c_Filter_17x17[11][16];
    pixel = image[y - 2][x - 8];
    sum += pixel * c_Filter_17x17[10][16];
    pixel = image[y - 1][x - 8];
    sum += pixel * c_Filter_17x17[9][16];
    pixel = image[y + 0][x - 8];
    sum += pixel * c_Filter_17x17[8][16];
    pixel = image[y + 1][x - 8];
    sum += pixel * c_Filter_17x17[7][16];
    pixel = image[y + 2][x - 8];
    sum += pixel * c_Filter_17x17[6][16];
    pixel = image[y + 3][x - 8];
    sum += pixel * c_Filter_17x17[5][16];
    pixel = image[y + 4][x - 8];
    sum += pixel * c_Filter_17x17[4][16];
    pixel = image[y + 5][x - 8];
    sum += pixel * c_Filter_17x17[3][16];
    pixel = image[y + 6][x - 8];
    sum += pixel * c_Filter_17x17[2][16];
    pixel = image[y + 7][x - 8];
    sum += pixel * c_Filter_17x17[1][16];
    pixel = image[y + 8][x - 8];
    sum += pixel * c_Filter_17x17[0][16];

    pixel = image[y - 8][x - 7];
    sum += pixel * c_Filter_17x17[16][15];
    pixel = image[y - 7][x - 7];
    sum += pixel * c_Filter_17x17[15][15];
    pixel = image[y - 6][x - 7];
    sum += pixel * c_Filter_17x17[14][15];
    pixel = image[y - 5][x - 7];
    sum += pixel * c_Filter_17x17[13][15];
    pixel = image[y - 4][x - 7];
    sum += pixel * c_Filter_17x17[12][15];
    pixel = image[y - 3][x - 7];
    sum += pixel * c_Filter_17x17[11][15];
    pixel = image[y - 2][x - 7];
    sum += pixel * c_Filter_17x17[10][15];
    pixel = image[y - 1][x - 7];
    sum += pixel * c_Filter_17x17[9][15];
    pixel = image[y + 0][x - 7];
    sum += pixel * c_Filter_17x17[8][15];
    pixel = image[y + 1][x - 7];
    sum += pixel * c_Filter_17x17[7][15];
    pixel = image[y + 2][x - 7];
    sum += pixel * c_Filter_17x17[6][15];
    pixel = image[y + 3][x - 7];
    sum += pixel * c_Filter_17x17[5][15];
    pixel = image[y + 4][x - 7];
    sum += pixel * c_Filter_17x17[4][15];
    pixel = image[y + 5][x - 7];
    sum += pixel * c_Filter_17x17[3][15];
    pixel = image[y + 6][x - 7];
    sum += pixel * c_Filter_17x17[2][15];
    pixel = image[y + 7][x - 7];
    sum += pixel * c_Filter_17x17[1][15];
    pixel = image[y + 8][x - 7];
    sum += pixel * c_Filter_17x17[0][15];

    pixel = image[y - 8][x - 6];
    sum += pixel * c_Filter_17x17[16][14];
    pixel = image[y - 7][x - 6];
    sum += pixel * c_Filter_17x17[15][14];
    pixel = image[y - 6][x - 6];
    sum += pixel * c_Filter_17x17[14][14];
    pixel = image[y - 5][x - 6];
    sum += pixel * c_Filter_17x17[13][14];
    pixel = image[y - 4][x - 6];
    sum += pixel * c_Filter_17x17[12][14];
    pixel = image[y - 3][x - 6];
    sum += pixel * c_Filter_17x17[11][14];
    pixel = image[y - 2][x - 6];
    sum += pixel * c_Filter_17x17[10][14];
    pixel = image[y - 1][x - 6];
    sum += pixel * c_Filter_17x17[9][14];
    pixel = image[y + 0][x - 6];
    sum += pixel * c_Filter_17x17[8][14];
    pixel = image[y + 1][x - 6];
    sum += pixel * c_Filter_17x17[7][14];
    pixel = image[y + 2][x - 6];
    sum += pixel * c_Filter_17x17[6][14];
    pixel = image[y + 3][x - 6];
    sum += pixel * c_Filter_17x17[5][14];
    pixel = image[y + 4][x - 6];
    sum += pixel * c_Filter_17x17[4][14];
    pixel = image[y + 5][x - 6];
    sum += pixel * c_Filter_17x17[3][14];
    pixel = image[y + 6][x - 6];
    sum += pixel * c_Filter_17x17[2][14];
    pixel = image[y + 7][x - 6];
    sum += pixel * c_Filter_17x17[1][14];
    pixel = image[y + 8][x - 6];
    sum += pixel * c_Filter_17x17[0][14];

    pixel = image[y - 8][x - 5];
    sum += pixel * c_Filter_17x17[16][13];
    pixel = image[y - 7][x - 5];
    sum += pixel * c_Filter_17x17[15][13];
    pixel = image[y - 6][x - 5];
    sum += pixel * c_Filter_17x17[14][13];
    pixel = image[y - 5][x - 5];
    sum += pixel * c_Filter_17x17[13][13];
    pixel = image[y - 4][x - 5];
    sum += pixel * c_Filter_17x17[12][13];
    pixel = image[y - 3][x - 5];
    sum += pixel * c_Filter_17x17[11][13];
    pixel = image[y - 2][x - 5];
    sum += pixel * c_Filter_17x17[10][13];
    pixel = image[y - 1][x - 5];
    sum += pixel * c_Filter_17x17[9][13];
    pixel = image[y + 0][x - 5];
    sum += pixel * c_Filter_17x17[8][13];
    pixel = image[y + 1][x - 5];
    sum += pixel * c_Filter_17x17[7][13];
    pixel = image[y + 2][x - 5];
    sum += pixel * c_Filter_17x17[6][13];
    pixel = image[y + 3][x - 5];
    sum += pixel * c_Filter_17x17[5][13];
    pixel = image[y + 4][x - 5];
    sum += pixel * c_Filter_17x17[4][13];
    pixel = image[y + 5][x - 5];
    sum += pixel * c_Filter_17x17[3][13];
    pixel = image[y + 6][x - 5];
    sum += pixel * c_Filter_17x17[2][13];
    pixel = image[y + 7][x - 5];
    sum += pixel * c_Filter_17x17[1][13];
    pixel = image[y + 8][x - 5];
    sum += pixel * c_Filter_17x17[0][13];

    pixel = image[y - 8][x - 4];
    sum += pixel * c_Filter_17x17[16][12];
    pixel = image[y - 7][x - 4];
    sum += pixel * c_Filter_17x17[15][12];
    pixel = image[y - 6][x - 4];
    sum += pixel * c_Filter_17x17[14][12];
    pixel = image[y - 5][x - 4];
    sum += pixel * c_Filter_17x17[13][12];
    pixel = image[y - 4][x - 4];
    sum += pixel * c_Filter_17x17[12][12];
    pixel = image[y - 3][x - 4];
    sum += pixel * c_Filter_17x17[11][12];
    pixel = image[y - 2][x - 4];
    sum += pixel * c_Filter_17x17[10][12];
    pixel = image[y - 1][x - 4];
    sum += pixel * c_Filter_17x17[9][12];
    pixel = image[y + 0][x - 4];
    sum += pixel * c_Filter_17x17[8][12];
    pixel = image[y + 1][x - 4];
    sum += pixel * c_Filter_17x17[7][12];
    pixel = image[y + 2][x - 4];
    sum += pixel * c_Filter_17x17[6][12];
    pixel = image[y + 3][x - 4];
    sum += pixel * c_Filter_17x17[5][12];
    pixel = image[y + 4][x - 4];
    sum += pixel * c_Filter_17x17[4][12];
    pixel = image[y + 5][x - 4];
    sum += pixel * c_Filter_17x17[3][12];
    pixel = image[y + 6][x - 4];
    sum += pixel * c_Filter_17x17[2][12];
    pixel = image[y + 7][x - 4];
    sum += pixel * c_Filter_17x17[1][12];
    pixel = image[y + 8][x - 4];
    sum += pixel * c_Filter_17x17[0][12];

    pixel = image[y - 8][x - 3];
    sum += pixel * c_Filter_17x17[16][11];
    pixel = image[y - 7][x - 3];
    sum += pixel * c_Filter_17x17[15][11];
    pixel = image[y - 6][x - 3];
    sum += pixel * c_Filter_17x17[14][11];
    pixel = image[y - 5][x - 3];
    sum += pixel * c_Filter_17x17[13][11];
    pixel = image[y - 4][x - 3];
    sum += pixel * c_Filter_17x17[12][11];
    pixel = image[y - 3][x - 3];
    sum += pixel * c_Filter_17x17[11][11];
    pixel = image[y - 2][x - 3];
    sum += pixel * c_Filter_17x17[10][11];
    pixel = image[y - 1][x - 3];
    sum += pixel * c_Filter_17x17[9][11];
    pixel = image[y + 0][x - 3];
    sum += pixel * c_Filter_17x17[8][11];
    pixel = image[y + 1][x - 3];
    sum += pixel * c_Filter_17x17[7][11];
    pixel = image[y + 2][x - 3];
    sum += pixel * c_Filter_17x17[6][11];
    pixel = image[y + 3][x - 3];
    sum += pixel * c_Filter_17x17[5][11];
    pixel = image[y + 4][x - 3];
    sum += pixel * c_Filter_17x17[4][11];
    pixel = image[y + 5][x - 3];
    sum += pixel * c_Filter_17x17[3][11];
    pixel = image[y + 6][x - 3];
    sum += pixel * c_Filter_17x17[2][11];
    pixel = image[y + 7][x - 3];
    sum += pixel * c_Filter_17x17[1][11];
    pixel = image[y + 8][x - 3];
    sum += pixel * c_Filter_17x17[0][11];

    pixel = image[y - 8][x - 2];
    sum += pixel * c_Filter_17x17[16][10];
    pixel = image[y - 7][x - 2];
    sum += pixel * c_Filter_17x17[15][10];
    pixel = image[y - 6][x - 2];
    sum += pixel * c_Filter_17x17[14][10];
    pixel = image[y - 5][x - 2];
    sum += pixel * c_Filter_17x17[13][10];
    pixel = image[y - 4][x - 2];
    sum += pixel * c_Filter_17x17[12][10];
    pixel = image[y - 3][x - 2];
    sum += pixel * c_Filter_17x17[11][10];
    pixel = image[y - 2][x - 2];
    sum += pixel * c_Filter_17x17[10][10];
    pixel = image[y - 1][x - 2];
    sum += pixel * c_Filter_17x17[9][10];
    pixel = image[y + 0][x - 2];
    sum += pixel * c_Filter_17x17[8][10];
    pixel = image[y + 1][x - 2];
    sum += pixel * c_Filter_17x17[7][10];
    pixel = image[y + 2][x - 2];
    sum += pixel * c_Filter_17x17[6][10];
    pixel = image[y + 3][x - 2];
    sum += pixel * c_Filter_17x17[5][10];
    pixel = image[y + 4][x - 2];
    sum += pixel * c_Filter_17x17[4][10];
    pixel = image[y + 5][x - 2];
    sum += pixel * c_Filter_17x17[3][10];
    pixel = image[y + 6][x - 2];
    sum += pixel * c_Filter_17x17[2][10];
    pixel = image[y + 7][x - 2];
    sum += pixel * c_Filter_17x17[1][10];
    pixel = image[y + 8][x - 2];
    sum += pixel * c_Filter_17x17[0][10];

    pixel = image[y - 8][x - 1];
    sum += pixel * c_Filter_17x17[16][9];
    pixel = image[y - 7][x - 1];
    sum += pixel * c_Filter_17x17[15][9];
    pixel = image[y - 6][x - 1];
    sum += pixel * c_Filter_17x17[14][9];
    pixel = image[y - 5][x - 1];
    sum += pixel * c_Filter_17x17[13][9];
    pixel = image[y - 4][x - 1];
    sum += pixel * c_Filter_17x17[12][9];
    pixel = image[y - 3][x - 1];
    sum += pixel * c_Filter_17x17[11][9];
    pixel = image[y - 2][x - 1];
    sum += pixel * c_Filter_17x17[10][9];
    pixel = image[y - 1][x - 1];
    sum += pixel * c_Filter_17x17[9][9];
    pixel = image[y + 0][x - 1];
    sum += pixel * c_Filter_17x17[8][9];
    pixel = image[y + 1][x - 1];
    sum += pixel * c_Filter_17x17[7][9];
    pixel = image[y + 2][x - 1];
    sum += pixel * c_Filter_17x17[6][9];
    pixel = image[y + 3][x - 1];
    sum += pixel * c_Filter_17x17[5][9];
    pixel = image[y + 4][x - 1];
    sum += pixel * c_Filter_17x17[4][9];
    pixel = image[y + 5][x - 1];
    sum += pixel * c_Filter_17x17[3][9];
    pixel = image[y + 6][x - 1];
    sum += pixel * c_Filter_17x17[2][9];
    pixel = image[y + 7][x - 1];
    sum += pixel * c_Filter_17x17[1][9];
    pixel = image[y + 8][x - 1];
    sum += pixel * c_Filter_17x17[0][9];

    pixel = image[y - 8][x + 0];
    sum += pixel * c_Filter_17x17[16][8];
    pixel = image[y - 7][x + 0];
    sum += pixel * c_Filter_17x17[15][8];
    pixel = image[y - 6][x + 0];
    sum += pixel * c_Filter_17x17[14][8];
    pixel = image[y - 5][x + 0];
    sum += pixel * c_Filter_17x17[13][8];
    pixel = image[y - 4][x + 0];
    sum += pixel * c_Filter_17x17[12][8];
    pixel = image[y - 3][x + 0];
    sum += pixel * c_Filter_17x17[11][8];
    pixel = image[y - 2][x + 0];
    sum += pixel * c_Filter_17x17[10][8];
    pixel = image[y - 1][x + 0];
    sum += pixel * c_Filter_17x17[9][8];
    pixel = image[y + 0][x + 0];
    sum += pixel * c_Filter_17x17[8][8];
    pixel = image[y + 1][x + 0];
    sum += pixel * c_Filter_17x17[7][8];
    pixel = image[y + 2][x + 0];
    sum += pixel * c_Filter_17x17[6][8];
    pixel = image[y + 3][x + 0];
    sum += pixel * c_Filter_17x17[5][8];
    pixel = image[y + 4][x + 0];
    sum += pixel * c_Filter_17x17[4][8];
    pixel = image[y + 5][x + 0];
    sum += pixel * c_Filter_17x17[3][8];
    pixel = image[y + 6][x + 0];
    sum += pixel * c_Filter_17x17[2][8];
    pixel = image[y + 7][x + 0];
    sum += pixel * c_Filter_17x17[1][8];
    pixel = image[y + 8][x + 0];
    sum += pixel * c_Filter_17x17[0][8];

    pixel = image[y - 8][x + 1];
    sum += pixel * c_Filter_17x17[16][7];
    pixel = image[y - 7][x + 1];
    sum += pixel * c_Filter_17x17[15][7];
    pixel = image[y - 6][x + 1];
    sum += pixel * c_Filter_17x17[14][7];
    pixel = image[y - 5][x + 1];
    sum += pixel * c_Filter_17x17[13][7];
    pixel = image[y - 4][x + 1];
    sum += pixel * c_Filter_17x17[12][7];
    pixel = image[y - 3][x + 1];
    sum += pixel * c_Filter_17x17[11][7];
    pixel = image[y - 2][x + 1];
    sum += pixel * c_Filter_17x17[10][7];
    pixel = image[y - 1][x + 1];
    sum += pixel * c_Filter_17x17[9][7];
    pixel = image[y + 0][x + 1];
    sum += pixel * c_Filter_17x17[8][7];
    pixel = image[y + 1][x + 1];
    sum += pixel * c_Filter_17x17[7][7];
    pixel = image[y + 2][x + 1];
    sum += pixel * c_Filter_17x17[6][7];
    pixel = image[y + 3][x + 1];
    sum += pixel * c_Filter_17x17[5][7];
    pixel = image[y + 4][x + 1];
    sum += pixel * c_Filter_17x17[4][7];
    pixel = image[y + 5][x + 1];
    sum += pixel * c_Filter_17x17[3][7];
    pixel = image[y + 6][x + 1];
    sum += pixel * c_Filter_17x17[2][7];
    pixel = image[y + 7][x + 1];
    sum += pixel * c_Filter_17x17[1][7];
    pixel = image[y + 8][x + 1];
    sum += pixel * c_Filter_17x17[0][7];

    pixel = image[y - 8][x + 2];
    sum += pixel * c_Filter_17x17[16][6];
    pixel = image[y - 7][x + 2];
    sum += pixel * c_Filter_17x17[15][6];
    pixel = image[y - 6][x + 2];
    sum += pixel * c_Filter_17x17[14][6];
    pixel = image[y - 5][x + 2];
    sum += pixel * c_Filter_17x17[13][6];
    pixel = image[y - 4][x + 2];
    sum += pixel * c_Filter_17x17[12][6];
    pixel = image[y - 3][x + 2];
    sum += pixel * c_Filter_17x17[11][6];
    pixel = image[y - 2][x + 2];
    sum += pixel * c_Filter_17x17[10][6];
    pixel = image[y - 1][x + 2];
    sum += pixel * c_Filter_17x17[9][6];
    pixel = image[y + 0][x + 2];
    sum += pixel * c_Filter_17x17[8][6];
    pixel = image[y + 1][x + 2];
    sum += pixel * c_Filter_17x17[7][6];
    pixel = image[y + 2][x + 2];
    sum += pixel * c_Filter_17x17[6][6];
    pixel = image[y + 3][x + 2];
    sum += pixel * c_Filter_17x17[5][6];
    pixel = image[y + 4][x + 2];
    sum += pixel * c_Filter_17x17[4][6];
    pixel = image[y + 5][x + 2];
    sum += pixel * c_Filter_17x17[3][6];
    pixel = image[y + 6][x + 2];
    sum += pixel * c_Filter_17x17[2][6];
    pixel = image[y + 7][x + 2];
    sum += pixel * c_Filter_17x17[1][6];
    pixel = image[y + 8][x + 2];
    sum += pixel * c_Filter_17x17[0][6];

    pixel = image[y - 8][x + 3];
    sum += pixel * c_Filter_17x17[16][5];
    pixel = image[y - 7][x + 3];
    sum += pixel * c_Filter_17x17[15][5];
    pixel = image[y - 6][x + 3];
    sum += pixel * c_Filter_17x17[14][5];
    pixel = image[y - 5][x + 3];
    sum += pixel * c_Filter_17x17[13][5];
    pixel = image[y - 4][x + 3];
    sum += pixel * c_Filter_17x17[12][5];
    pixel = image[y - 3][x + 3];
    sum += pixel * c_Filter_17x17[11][5];
    pixel = image[y - 2][x + 3];
    sum += pixel * c_Filter_17x17[10][5];
    pixel = image[y - 1][x + 3];
    sum += pixel * c_Filter_17x17[9][5];
    pixel = image[y + 0][x + 3];
    sum += pixel * c_Filter_17x17[8][5];
    pixel = image[y + 1][x + 3];
    sum += pixel * c_Filter_17x17[7][5];
    pixel = image[y + 2][x + 3];
    sum += pixel * c_Filter_17x17[6][5];
    pixel = image[y + 3][x + 3];
    sum += pixel * c_Filter_17x17[5][5];
    pixel = image[y + 4][x + 3];
    sum += pixel * c_Filter_17x17[4][5];
    pixel = image[y + 5][x + 3];
    sum += pixel * c_Filter_17x17[3][5];
    pixel = image[y + 6][x + 3];
    sum += pixel * c_Filter_17x17[2][5];
    pixel = image[y + 7][x + 3];
    sum += pixel * c_Filter_17x17[1][5];
    pixel = image[y + 8][x + 3];
    sum += pixel * c_Filter_17x17[0][5];

    pixel = image[y - 8][x + 4];
    sum += pixel * c_Filter_17x17[16][4];
    pixel = image[y - 7][x + 4];
    sum += pixel * c_Filter_17x17[15][4];
    pixel = image[y - 6][x + 4];
    sum += pixel * c_Filter_17x17[14][4];
    pixel = image[y - 5][x + 4];
    sum += pixel * c_Filter_17x17[13][4];
    pixel = image[y - 4][x + 4];
    sum += pixel * c_Filter_17x17[12][4];
    pixel = image[y - 3][x + 4];
    sum += pixel * c_Filter_17x17[11][4];
    pixel = image[y - 2][x + 4];
    sum += pixel * c_Filter_17x17[10][4];
    pixel = image[y - 1][x + 4];
    sum += pixel * c_Filter_17x17[9][4];
    pixel = image[y + 0][x + 4];
    sum += pixel * c_Filter_17x17[8][4];
    pixel = image[y + 1][x + 4];
    sum += pixel * c_Filter_17x17[7][4];
    pixel = image[y + 2][x + 4];
    sum += pixel * c_Filter_17x17[6][4];
    pixel = image[y + 3][x + 4];
    sum += pixel * c_Filter_17x17[5][4];
    pixel = image[y + 4][x + 4];
    sum += pixel * c_Filter_17x17[4][4];
    pixel = image[y + 5][x + 4];
    sum += pixel * c_Filter_17x17[3][4];
    pixel = image[y + 6][x + 4];
    sum += pixel * c_Filter_17x17[2][4];
    pixel = image[y + 7][x + 4];
    sum += pixel * c_Filter_17x17[1][4];
    pixel = image[y + 8][x + 4];
    sum += pixel * c_Filter_17x17[0][4];

    pixel = image[y - 8][x + 5];
    sum += pixel * c_Filter_17x17[16][3];
    pixel = image[y - 7][x + 5];
    sum += pixel * c_Filter_17x17[15][3];
    pixel = image[y - 6][x + 5];
    sum += pixel * c_Filter_17x17[14][3];
    pixel = image[y - 5][x + 5];
    sum += pixel * c_Filter_17x17[13][3];
    pixel = image[y - 4][x + 5];
    sum += pixel * c_Filter_17x17[12][3];
    pixel = image[y - 3][x + 5];
    sum += pixel * c_Filter_17x17[11][3];
    pixel = image[y - 2][x + 5];
    sum += pixel * c_Filter_17x17[10][3];
    pixel = image[y - 1][x + 5];
    sum += pixel * c_Filter_17x17[9][3];
    pixel = image[y + 0][x + 5];
    sum += pixel * c_Filter_17x17[8][3];
    pixel = image[y + 1][x + 5];
    sum += pixel * c_Filter_17x17[7][3];
    pixel = image[y + 2][x + 5];
    sum += pixel * c_Filter_17x17[6][3];
    pixel = image[y + 3][x + 5];
    sum += pixel * c_Filter_17x17[5][3];
    pixel = image[y + 4][x + 5];
    sum += pixel * c_Filter_17x17[4][3];
    pixel = image[y + 5][x + 5];
    sum += pixel * c_Filter_17x17[3][3];
    pixel = image[y + 6][x + 5];
    sum += pixel * c_Filter_17x17[2][3];
    pixel = image[y + 7][x + 5];
    sum += pixel * c_Filter_17x17[1][3];
    pixel = image[y + 8][x + 5];
    sum += pixel * c_Filter_17x17[0][3];

    pixel = image[y - 8][x + 6];
    sum += pixel * c_Filter_17x17[16][2];
    pixel = image[y - 7][x + 6];
    sum += pixel * c_Filter_17x17[15][2];
    pixel = image[y - 6][x + 6];
    sum += pixel * c_Filter_17x17[14][2];
    pixel = image[y - 5][x + 6];
    sum += pixel * c_Filter_17x17[13][2];
    pixel = image[y - 4][x + 6];
    sum += pixel * c_Filter_17x17[12][2];
    pixel = image[y - 3][x + 6];
    sum += pixel * c_Filter_17x17[11][2];
    pixel = image[y - 2][x + 6];
    sum += pixel * c_Filter_17x17[10][2];
    pixel = image[y - 1][x + 6];
    sum += pixel * c_Filter_17x17[9][2];
    pixel = image[y + 0][x + 6];
    sum += pixel * c_Filter_17x17[8][2];
    pixel = image[y + 1][x + 6];
    sum += pixel * c_Filter_17x17[7][2];
    pixel = image[y + 2][x + 6];
    sum += pixel * c_Filter_17x17[6][2];
    pixel = image[y + 3][x + 6];
    sum += pixel * c_Filter_17x17[5][2];
    pixel = image[y + 4][x + 6];
    sum += pixel * c_Filter_17x17[4][2];
    pixel = image[y + 5][x + 6];
    sum += pixel * c_Filter_17x17[3][2];
    pixel = image[y + 6][x + 6];
    sum += pixel * c_Filter_17x17[2][2];
    pixel = image[y + 7][x + 6];
    sum += pixel * c_Filter_17x17[1][2];
    pixel = image[y + 8][x + 6];
    sum += pixel * c_Filter_17x17[0][2];

    pixel = image[y - 8][x + 7];
    sum += pixel * c_Filter_17x17[16][1];
    pixel = image[y - 7][x + 7];
    sum += pixel * c_Filter_17x17[15][1];
    pixel = image[y - 6][x + 7];
    sum += pixel * c_Filter_17x17[14][1];
    pixel = image[y - 5][x + 7];
    sum += pixel * c_Filter_17x17[13][1];
    pixel = image[y - 4][x + 7];
    sum += pixel * c_Filter_17x17[12][1];
    pixel = image[y - 3][x + 7];
    sum += pixel * c_Filter_17x17[11][1];
    pixel = image[y - 2][x + 7];
    sum += pixel * c_Filter_17x17[10][1];
    pixel = image[y - 1][x + 7];
    sum += pixel * c_Filter_17x17[9][1];
    pixel = image[y + 0][x + 7];
    sum += pixel * c_Filter_17x17[8][1];
    pixel = image[y + 1][x + 7];
    sum += pixel * c_Filter_17x17[7][1];
    pixel = image[y + 2][x + 7];
    sum += pixel * c_Filter_17x17[6][1];
    pixel = image[y + 3][x + 7];
    sum += pixel * c_Filter_17x17[5][1];
    pixel = image[y + 4][x + 7];
    sum += pixel * c_Filter_17x17[4][1];
    pixel = image[y + 5][x + 7];
    sum += pixel * c_Filter_17x17[3][1];
    pixel = image[y + 6][x + 7];
    sum += pixel * c_Filter_17x17[2][1];
    pixel = image[y + 7][x + 7];
    sum += pixel * c_Filter_17x17[1][1];
    pixel = image[y + 8][x + 7];
    sum += pixel * c_Filter_17x17[0][1];

    pixel = image[y - 8][x + 8];
    sum += pixel * c_Filter_17x17[16][0];
    pixel = image[y - 7][x + 8];
    sum += pixel * c_Filter_17x17[15][0];
    pixel = image[y - 6][x + 8];
    sum += pixel * c_Filter_17x17[14][0];
    pixel = image[y - 5][x + 8];
    sum += pixel * c_Filter_17x17[13][0];
    pixel = image[y - 4][x + 8];
    sum += pixel * c_Filter_17x17[12][0];
    pixel = image[y - 3][x + 8];
    sum += pixel * c_Filter_17x17[11][0];
    pixel = image[y - 2][x + 8];
    sum += pixel * c_Filter_17x17[10][0];
    pixel = image[y - 1][x + 8];
    sum += pixel * c_Filter_17x17[9][0];
    pixel = image[y + 0][x + 8];
    sum += pixel * c_Filter_17x17[8][0];
    pixel = image[y + 1][x + 8];
    sum += pixel * c_Filter_17x17[7][0];
    pixel = image[y + 2][x + 8];
    sum += pixel * c_Filter_17x17[6][0];
    pixel = image[y + 3][x + 8];
    sum += pixel * c_Filter_17x17[5][0];
    pixel = image[y + 4][x + 8];
    sum += pixel * c_Filter_17x17[4][0];
    pixel = image[y + 5][x + 8];
    sum += pixel * c_Filter_17x17[3][0];
    pixel = image[y + 6][x + 8];
    sum += pixel * c_Filter_17x17[2][0];
    pixel = image[y + 7][x + 8];
    sum += pixel * c_Filter_17x17[1][0];
    pixel = image[y + 8][x + 8];
    sum += pixel * c_Filter_17x17[0][0];

	return sum;
}

/*
 This function performs non-separable 2D convolution by using texture memory.
*/

texture<float, 2, cudaReadModeElementType> tex_Image;

__global__ void Convolution_2D_Texture(float* Filter_Response, int DATA_W, int DATA_H, int FILTER_W, int FILTER_H)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= DATA_W || y >= DATA_H)
        return;

   float sum = 0.0f;

   float y_off = -(FILTER_H - 1)/2 + 0.5f;
   for (int f_y = FILTER_H - 1; f_y >= 0; f_y--)
   {
        float x_off = -(FILTER_W - 1)/2 + 0.5f;			
        for (int f_x = FILTER_W - 1; f_x >= 0; f_x--)
        {
             sum += tex2D(tex_Image,x + x_off,y + y_off) * c_Filter_2D[f_y][f_x];
             x_off += 1.0f;
        }
        y_off += 1.0f;
    }

    Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}

__global__ void Convolution_2D_Texture_Unrolled_3x3(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_3x3[2][2];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_3x3[1][2];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_3x3[0][2];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_3x3[2][1];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_3x3[1][1];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_3x3[0][1];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_3x3[2][0];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_3x3[1][0];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_3x3[0][0];

    Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}


__global__ void Convolution_2D_Texture_Unrolled_5x5(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_5x5[4][4];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_5x5[3][4];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_5x5[2][4];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_5x5[1][4];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_5x5[0][4];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_5x5[4][3];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_5x5[3][3];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_5x5[2][3];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_5x5[1][3];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_5x5[0][3];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_5x5[4][2];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_5x5[3][2];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_5x5[2][2];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_5x5[1][2];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_5x5[0][2];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_5x5[4][1];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_5x5[3][1];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_5x5[2][1];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_5x5[1][1];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_5x5[0][1];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_5x5[4][0];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_5x5[3][0];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_5x5[2][0];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_5x5[1][0];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_5x5[0][0];

    Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}


__global__ void Convolution_2D_Texture_Unrolled_7x7(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][6];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][6];
   
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][5];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][5];
   
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][4];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][4];
   
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][3];
    sum += tex2D(tex_Image, x - 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][3];
   
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][2];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][2];
   
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][1];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][1];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_7x7[6][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_7x7[5][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_7x7[4][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 0.0f + 0.5f) * c_Filter_7x7[3][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_7x7[2][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_7x7[1][0];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_7x7[0][0];

    Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}


__global__ void Convolution_2D_Texture_Unrolled_9x9(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
        return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][8];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][8];

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][7];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][7];


    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][6];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][6];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][5];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][5];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][4];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][4];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][3];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][3];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][2];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][2];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][1];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][1];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][0];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_9x9[8][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_9x9[7][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_9x9[6][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_9x9[5][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_9x9[4][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_9x9[3][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_9x9[2][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_9x9[1][0];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_9x9[0][0];

    Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}

__global__ void Convolution_2D_Texture_Unrolled_11x11(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][10];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][10];

    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][9];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][9];

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][8];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][8];

    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][7];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][7];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][6];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][6];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][5];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][5];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][4];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][4];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][3];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][3];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][2];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][2];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][1];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][1];

    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_11x11[10][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_11x11[9][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_11x11[8][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_11x11[7][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_11x11[6][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_11x11[5][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_11x11[4][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_11x11[3][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_11x11[2][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_11x11[1][0];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_11x11[0][0];

	Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}

__global__ void Convolution_2D_Texture_Unrolled_13x13(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][12];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][12];

    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][11];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][11];

    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][10];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][10];

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][9];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][9];

    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][8];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][8];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][7];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][7];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][6];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][6];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][5];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][5];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][4];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][4];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][3];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][3];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][2];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][2];

    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][1];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][1];

    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_13x13[12][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_13x13[11][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_13x13[10][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_13x13[9][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_13x13[8][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_13x13[7][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_13x13[6][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_13x13[5][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_13x13[4][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_13x13[3][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_13x13[2][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_13x13[1][0];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_13x13[0][0];

	Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}

__global__ void Convolution_2D_Texture_Unrolled_15x15(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
         return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][14];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][14];

    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][13];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][13];

    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][12];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][12];

    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][11];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][11];

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][10];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][10];

    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][9];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][9];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][8];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][8];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][7];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][7];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][6];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][6];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][5];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][5];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][4];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][4];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][3];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][3];

    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][2];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][2];

    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][1];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][1];

    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_15x15[14][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_15x15[13][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_15x15[12][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_15x15[11][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_15x15[10][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_15x15[9][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_15x15[8][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_15x15[7][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_15x15[6][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_15x15[5][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_15x15[4][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_15x15[3][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_15x15[2][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_15x15[1][0];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_15x15[0][0];

	Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}


__global__ void Convolution_2D_Texture_Unrolled_17x17(float* Filter_Response, int DATA_W, int DATA_H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DATA_W || y >= DATA_H)
        return;

    float sum = 0.0f;

    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][16];
    sum += tex2D(tex_Image, x - 8.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][16];

	sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][15];
    sum += tex2D(tex_Image, x - 7.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][15];

    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][14];
    sum += tex2D(tex_Image, x - 6.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][14];

    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][13];
    sum += tex2D(tex_Image, x - 5.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][13];

    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][12];
    sum += tex2D(tex_Image, x - 4.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][12];

    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][11];
    sum += tex2D(tex_Image, x - 3.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][11];

    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][10];
    sum += tex2D(tex_Image, x - 2.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][10];

    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][9];
    sum += tex2D(tex_Image, x - 1.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][9];

    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][8];
    sum += tex2D(tex_Image, x + 0.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][8];

    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][7];
    sum += tex2D(tex_Image, x + 1.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][7];

    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][6];
    sum += tex2D(tex_Image, x + 2.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][6];

    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][5];
    sum += tex2D(tex_Image, x + 3.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][5];

    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][4];
    sum += tex2D(tex_Image, x + 4.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][4];

    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][3];
    sum += tex2D(tex_Image, x + 5.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][3];

    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][2];
    sum += tex2D(tex_Image, x + 6.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][2];

    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][1];
    sum += tex2D(tex_Image, x + 7.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][1];

    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 8.0f + 0.5f) * c_Filter_17x17[16][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 7.0f + 0.5f) * c_Filter_17x17[15][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 6.0f + 0.5f) * c_Filter_17x17[14][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 5.0f + 0.5f) * c_Filter_17x17[13][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 4.0f + 0.5f) * c_Filter_17x17[12][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 3.0f + 0.5f) * c_Filter_17x17[11][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 2.0f + 0.5f) * c_Filter_17x17[10][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y - 1.0f + 0.5f) * c_Filter_17x17[9][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 0.0f + 0.5f) * c_Filter_17x17[8][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 1.0f + 0.5f) * c_Filter_17x17[7][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 2.0f + 0.5f) * c_Filter_17x17[6][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 3.0f + 0.5f) * c_Filter_17x17[5][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 4.0f + 0.5f) * c_Filter_17x17[4][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 5.0f + 0.5f) * c_Filter_17x17[3][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 6.0f + 0.5f) * c_Filter_17x17[2][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 7.0f + 0.5f) * c_Filter_17x17[1][0];
    sum += tex2D(tex_Image, x + 8.0f + 0.5f, y + 8.0f + 0.5f) * c_Filter_17x17[0][0];

	Filter_Response[Get_2D_Index(x,y,DATA_W)] = sum;
}



__global__ void Convolution_2D_Shared(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int FILTER_W, int FILTER_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   //if ( (x >= DATA_W) || (y >= DATA_H) )
   //     return;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+HALO,FILTER_H,FILTER_W);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO,FILTER_H,FILTER_W);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO,FILTER_H,FILTER_W);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO,FILTER_H,FILTER_W);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO,FILTER_H,FILTER_W);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO,FILTER_H,FILTER_W);
   }

}

__global__ void Convolution_2D_Shared_Unrolled_3x3(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_3x3(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}

__global__ void Convolution_2D_Shared_Unrolled_5x5(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_5x5(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}


__global__ void Convolution_2D_Shared_Unrolled_7x7(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}

__global__ void Convolution_2D_Shared_Unrolled_9x9(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_9x9(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}


__global__ void Convolution_2D_Shared_Unrolled_11x11(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_11x11(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}

__global__ void Convolution_2D_Shared_Unrolled_13x13(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_13x13(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}

__global__ void Convolution_2D_Shared_Unrolled_15x15(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_15x15(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}	


__global__ void Convolution_2D_Shared_Unrolled_17x17(float* Filter_Response, float* Image, int DATA_W, int DATA_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference)) )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )   
      s_Image[threadIdx.y][threadIdx.x] = Image[Get_2D_Index(x-HALO,y-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) )
      s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y-HALO,DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H) ) 
      s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y-HALO,DATA_W)];

   if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_2D_Index(x-HALO,y+32-HALO,DATA_W)];

   if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_2D_Index(x+32-HALO,y+32-HALO, DATA_W)];

   if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H) )
      s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_2D_Index(x+64-HALO,y+32-HALO,DATA_W)];
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x,y,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_2D_Index(x+32,y,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x,y+32,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+32,y+32,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_2D_Index(x+64,y+32,DATA_W)] = Conv_2D_Unrolled_17x17(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}


__global__ void Convolution_2D_Shared_For_3D(float* Filter_Response, float* Image, int z_offset, int DATA_W, int DATA_H, int DATA_D, int FILTER_W, int FILTER_H, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference))  )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((z + z_offset) >= 0) && ((z + z_offset) < DATA_D) )
   {
      if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
         s_Image[threadIdx.y][threadIdx.x] = Image[Get_3D_Index(x-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
         s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_3D_Index(x+32-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
         s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_3D_Index(x+64-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_3D_Index(x-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_3D_Index(x+32-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_3D_Index(x+64-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];
   }
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_3D_Index(x,y,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+HALO,FILTER_H,FILTER_W);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_3D_Index(x+32,y,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO,FILTER_H,FILTER_W);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_3D_Index(x+64,y,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO,FILTER_H,FILTER_W);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x,y+32,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO,FILTER_H,FILTER_W);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x+32,y+32,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO,FILTER_H,FILTER_W);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x+64,y+32,z,DATA_W,DATA_H)] += Conv_2D(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO,FILTER_H,FILTER_W);
   }

}

__global__ void Convolution_2D_Shared_For_3D_Unrolled_7x7(float* Filter_Response, float* Image, int z_offset, int DATA_W, int DATA_H, int DATA_D, int xBlockDifference, int yBlockDifference)
{
   int x = blockIdx.x * VALID_RESPONSES_X + threadIdx.x;
   int y = blockIdx.y * VALID_RESPONSES_Y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ( (x >= (DATA_W + xBlockDifference)) || (y >= (DATA_H + yBlockDifference))  )
        return;

   __shared__ float s_Image[64][96]; // y, x

   // Reset shared memory
   s_Image[threadIdx.y][threadIdx.x]           = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 32]      = 0.0f;
   s_Image[threadIdx.y][threadIdx.x + 64]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x]      = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 32] = 0.0f;
   s_Image[threadIdx.y + 32][threadIdx.x + 64] = 0.0f;

   // Read data into shared memory

   if ( ((z + z_offset) >= 0) && ((z + z_offset) < DATA_D) )
   {
      if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )   
         s_Image[threadIdx.y][threadIdx.x] = Image[Get_3D_Index(x-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+32-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  )
         s_Image[threadIdx.y][threadIdx.x + 32] = Image[Get_3D_Index(x+32-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+64-HALO) < DATA_W) && ((y-HALO) >= 0) && ((y-HALO) < DATA_H)  ) 
         s_Image[threadIdx.y][threadIdx.x + 64] = Image[Get_3D_Index(x+64-HALO,y-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x-HALO) >= 0) && ((x-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x] = Image[Get_3D_Index(x-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+32-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x + 32] = Image[Get_3D_Index(x+32-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];

      if ( ((x+64-HALO) < DATA_W) && ((y+32-HALO) < DATA_H)  )
         s_Image[threadIdx.y + 32][threadIdx.x + 64] = Image[Get_3D_Index(x+64-HALO,y+32-HALO,z+z_offset,DATA_W,DATA_H)];
   }
	
   __syncthreads();   

   // Only threads inside the image do the convolution

   if ( (x < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_3D_Index(x,y,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+HALO);

   if ( ((x + 32) < DATA_W) && (y < DATA_H) )
      Filter_Response[Get_3D_Index(x+32,y,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+32+HALO);

   if (threadIdx.x < (32 - HALO*2))
   {
      if ( ((x + 64) < DATA_W) && (y < DATA_H) )
         Filter_Response[Get_3D_Index(x+64,y,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+HALO,threadIdx.x+64+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( (x < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x,y+32,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+HALO);
   }

   if (threadIdx.y < (32 - HALO*2))
   {
      if ( ((x + 32) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x+32,y+32,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+32+HALO);		
   } 

   if ( (threadIdx.x < (32 - HALO*2)) && (threadIdx.y < (32 - HALO*2)) )
   {
      if ( ((x + 64) < DATA_W) && ((y + 32) < DATA_H) )
         Filter_Response[Get_3D_Index(x+64,y+32,z,DATA_W,DATA_H)] += Conv_2D_Unrolled_7x7(s_Image,threadIdx.y+32+HALO,threadIdx.x+64+HALO);
   }
}

/*
 This function performs non-separable 3D convolution by using texture memory.
*/

texture<float, 3, cudaReadModeElementType> tex_Volume;


__global__ void Convolution_3D_Texture(float* Filter_Response, int DATA_W, int DATA_H, int DATA_D, int FILTER_W, int FILTER_H, int FILTER_D)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

   float sum = 0.0f;

   float z_off = -(FILTER_D - 1)/2 + 0.5f;
   for (int f_z = FILTER_D - 1; f_z >= 0; f_z--)
   {
      float y_off = -(FILTER_H - 1)/2 + 0.5f;
      for (int f_y = FILTER_H - 1; f_y >= 0; f_y--)
      {
         float x_off = -(FILTER_W - 1)/2 + 0.5f;			
         for (int f_x = FILTER_W - 1; f_x >= 0; f_x--)
         {
            sum += tex3D(tex_Volume,x + x_off,y + y_off,z + z_off) * c_Filter_3D[f_y][f_x][f_z];
            x_off += 1.0f;
         }
         y_off += 1.0f;
     }
	 z_off += 1.0f;
   }

   Filter_Response[Get_3D_Index(x,y,z,DATA_W,DATA_H)] = sum;
}

__global__ void Convolution_3D_Texture_Unrolled_7x7x7(float* Filter_Response, int DATA_W, int DATA_H, int DATA_D)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

   float sum = 0.0f;


   Filter_Response[Get_3D_Index(x,y,z,DATA_W,DATA_H)] = sum;
}



#endif

